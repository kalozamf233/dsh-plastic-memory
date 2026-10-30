// ============================================================================
// dsh-plastic-memory —— 缺陷人格神经网络（DefectNet，对话修饰层）
// ----------------------------------------------------------------------------
// 架在操控网络与用户聊天之间，把回复内容与生理状态加工成 5 个对话修饰参数：
//   warmth 温暖度 / intimacy 亲密感 / embellish 修饰度 / deflect 回避度 / quirk 怪癖度
// 其余 3 维输出保留为内部状态。
//
// 权重在对话中固定（frozen）：
//   · apply 只前向，任何情况下都不调用 backward → 权重绝不变化；
//     frozen=true 时 forward 显式携带 opts.frozen 标记（防御性约定）。
//   · train() 是唯一更新权重的入口（用户 API 专用）；frozen=true 时跳过。
// ============================================================================

import { PlasticNetwork } from './network.mjs'
import { arrToPlain, arrFromPlain } from './core.mjs'

// 维度常量
export const DEFECT_INPUT_SIZE = 143 // 回复 latent 128 + 内分泌 15
export const DEFECT_OUTPUT_SIZE = 8
const LATENT_SIZE = 128
const ENDO_SIZE = 15

// 无情绪关键词时的默认修饰目标：[warmth, intimacy, embellish, deflect, quirk, 3×内部]
const DEFAULT_TARGET = [0.6, 0.5, 0.4, 0.3, 0.3, 0, 0, 0]

/** FNV-1a 32 位哈希（本地实现，零依赖）。 */
function fnv1a(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/**
 * 文本 → 128 维哈希词袋（字符 1/2/3-gram），L2 归一。
 * 中文对单字敏感：1-gram 兼顾单字，2/3-gram 捕获词与短句特征。
 */
function textToInput(text) {
  const bag = new Float64Array(LATENT_SIZE)
  const chars = Array.from(String(text ?? '').toLowerCase().replace(/\s+/g, ''))
  if (chars.length === 0) {
    bag[fnv1a('empty') % LATENT_SIZE] = 1
    return bag
  }
  for (let i = 0; i < chars.length; i++) {
    bag[fnv1a('c1:' + chars[i]) % LATENT_SIZE] += 0.3
    if (i + 1 < chars.length) bag[fnv1a('c2:' + chars[i] + chars[i + 1]) % LATENT_SIZE] += 0.8
    if (i + 2 < chars.length) bag[fnv1a('c3:' + chars[i] + chars[i + 1] + chars[i + 2]) % LATENT_SIZE] += 0.4
  }
  let n = 0
  for (let i = 0; i < LATENT_SIZE; i++) n += bag[i] * bag[i]
  if (n > 0) {
    n = Math.sqrt(n)
    for (let i = 0; i < LATENT_SIZE; i++) bag[i] /= n
  }
  return bag
}

/**
 * 文本情绪 → 修饰目标。情绪关键词把对应目标分量抬高到 0.8：
 *   喜悦 → warmth；悲伤 → intimacy；愤怒 → deflect；
 *   可爱/温柔/亲密 → warmth（+intimacy）。
 * 无关键词时使用默认目标。
 */
function targetFor(text) {
  const t = new Float64Array(DEFAULT_TARGET)
  const s = String(text ?? '').toLowerCase()
  if (/喜悦|开心|高兴|快乐|兴奋|愉快|joy/.test(s)) t[0] = 0.8
  if (/悲伤|难过|伤心|失落|忧郁|sad/.test(s)) t[1] = 0.8
  if (/愤怒|生气|发火|恼火|暴躁|angry/.test(s)) t[3] = 0.8
  if (/可爱|温柔|亲密|贴心/.test(s)) {
    t[0] = 0.8
    t[1] = 0.7
  }
  return t
}

export class DefectNet {
  constructor({ seed = 20261031, baseLr = 0.01 } = {}) {
    this.baseLr = baseLr
    this.frozen = false
    this.net = new PlasticNetwork({
      inputSize: DEFECT_INPUT_SIZE,
      layers: [
        { size: 64, activation: 'tanh', lr: 1, mutable: false },
        { size: DEFECT_OUTPUT_SIZE, activation: 'tanh', lr: 1, mutable: false },
      ],
      seed,
    })
    this._internal = new Float64Array(3) // 输出 3 个保留维的内部状态
  }

  /** 组装 143 维输入：回复 latent(128) + 内分泌(15，缺省 0.5)。 */
  _buildInput(replyLatent, endoLevels) {
    const input = new Float64Array(DEFECT_INPUT_SIZE)
    if (replyLatent) {
      for (let i = 0; i < LATENT_SIZE && i < replyLatent.length; i++) input[i] = replyLatent[i]
    }
    for (let i = 0; i < ENDO_SIZE; i++) {
      const v = endoLevels && i < endoLevels.length ? endoLevels[i] : 0.5
      input[LATENT_SIZE + i] = Number.isFinite(v) ? v : 0.5
    }
    return input
  }

  /**
   * 对话修饰：只前向，永不更新权重。
   * @param {Float64Array} replyLatent 回复内容 latent(128)
   * @param {Float64Array} endoLevels  内分泌浓度 endocrine.levels(15)，缺省 0.5
   * @returns {{warmth: number, intimacy: number, embellish: number,
   *            deflect: number, quirk: number}} 均在 [0,1]
   */
  apply(replyLatent, endoLevels) {
    const input = this._buildInput(replyLatent, endoLevels)
    // frozen=true 时显式携带 opts.frozen；apply 从不调用 backward，权重不变
    const { post } = this.net.forward(input, this.frozen ? { frozen: true } : {})
    const o = post[post.length - 1]
    const map = (v) => Math.min(1, Math.max(0, (v + 1) / 2))
    // 其余 3 维保留为内部状态
    this._internal[0] = o[5]
    this._internal[1] = o[6]
    this._internal[2] = o[7]
    return {
      warmth: map(o[0]),
      intimacy: map(o[1]),
      embellish: map(o[2]),
      deflect: map(o[3]),
      quirk: map(o[4]),
    }
  }

  /**
   * 唯一更新权重的入口（用户 API 专用）。
   * @param {string} text 训练文本（情绪关键词决定目标修饰偏好）
   * @param {object} opts {endoLevels?: Float64Array(15), epochs?: number}
   * @returns {{loss: number, epochs: number}} frozen 时
   *   {loss: 0, epochs: 0, skipped: 'frozen'}
   */
  train(text, opts = {}) {
    if (this.frozen) return { loss: 0, epochs: 0, skipped: 'frozen' }

    // 输入：哈希词袋(128) + 内分泌(15)
    const input = new Float64Array(DEFECT_INPUT_SIZE)
    input.set(textToInput(text), 0)
    const endo = opts.endoLevels ?? null
    for (let i = 0; i < ENDO_SIZE; i++) {
      const v = endo && i < endo.length ? endo[i] : 0.5
      input[LATENT_SIZE + i] = Number.isFinite(v) ? v : 0.5
    }

    const target = targetFor(text)
    const epochs = Math.max(1, opts.epochs ?? 8)
    let total = 0
    for (let e = 0; e < epochs; e++) {
      const res = this.net.train(input, target, { baseLr: this.baseLr })
      total += res.loss ?? 0
    }
    return { loss: total / epochs, epochs }
  }

  toJSON() {
    return {
      baseLr: this.baseLr,
      frozen: this.frozen,
      net: this.net.toJSON(),
      internal: arrToPlain(this._internal),
    }
  }

  static fromJSON(json) {
    const inst = Object.create(DefectNet.prototype)
    inst.baseLr = json.baseLr ?? 0.01
    inst.frozen = !!json.frozen
    inst.net = PlasticNetwork.fromJSON(json.net)
    inst._internal = json.internal ? arrFromPlain(json.internal) : new Float64Array(3)
    return inst
  }
}
