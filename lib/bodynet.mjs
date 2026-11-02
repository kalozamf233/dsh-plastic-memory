// ============================================================================
// dsh-plastic-memory —— 身体数据网络（BodyNet）+ 世界感知编码器（WorldEncoder）
// ----------------------------------------------------------------------------
// 依据 docs/superpowers/specs/2026-11-02-world-v3-design.md 第 4 节：
//   · 所有对外感知输入由 WorldEncoder 编码为固定向量；
//   · 控制身体的行为全部由 BodyNet 输出完成。
//
// 输入（152 维）：
//   [0..128)   六感哈希词袋（sights/sounds/smells/touches 每项 text+type
//              经 fnv1a 分桶到 128 桶，每桶计数/4 归一）
//   [128..144) 欲望向量 16（内分泌映射，desireFromEndocrine；由 brain 构造）
//   [144..152) 身体状态 8：[x/40, y/40, z/8, dir(0-3)/3, 能量, 痛感, 饱腹, 速度]
// 结构：152 → 96(tanh) → 17(tanh)。
//
// 输出（17 维）语义：
//   0-5   移动：北/南/西/东/上跳/静止
//   6-8   交互：grab/eat/rest（rest 兜底）
//   9-16  目标方位：8 个相对方向（n/ne/e/se/s/sw/w/nw），指向最近目标物
//
// 决策（decide）：6 维移动 softmax + 温度采样；静止时交互 3 维 argmax；
//   目标方位 argmax。欲望偏置：食欲高（desire[2]>0.6）且嗅到香甜类目标
//   → 移动概率以 0.5 权重偏向目标方向；痛感高 → 静止概率 +0.4 再归一。
//
// 训练（REINFORCE）：train(prevInput, output17, chosenIndex, reward) ——
//   策略梯度退化实现：advantage = reward - baseline（内部 EMA，init 0），
//   目标输出 = 原 output，chosenIndex 处按 advantage 符号置 ±|adv|(0.2..0.9)，
//   用 PlasticNetwork.train 反向传播 epochs 轮。
// ============================================================================

import { PlasticNetwork } from './nn/network.mjs'
import { mulberry32 } from './nn/core.mjs'
import { fnv1a } from './nn/encoder.mjs'
import { MOLECULE_INDEX } from './nn/endocrine.mjs'

// 维度常量
export const BODYN_INPUT_SIZE = 152 // 六感词袋 128 + 欲望 16 + 身体 8
export const BODYN_OUTPUT_SIZE = 17 // 移动 6 + 交互 3 + 目标方位 8
export const SENSE_KEYS = ['sights', 'sounds', 'smells', 'touches']

// 输出语义常量
export const MOVE_DIRS = ['n', 's', 'w', 'e'] // 输出 0-3（北/南/西/东）；4=上跳 5=静止
export const INTERACT_NAMES = ['grab', 'eat', 'rest'] // 输出 6-8（rest 兜底）
export const TARGET_DIRS = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'] // 输出 9-16
export const DESIRE_NAMES = [
  '探索欲', '回避欲', '食欲', '休息欲', '社交欲', '竞争欲', '安抚欲', '好奇欲',
  '警觉欲', '愉悦维持', '攻击欲', '亲和欲', '依赖欲', '清洁欲', '逃生欲', '成长欲',
]

// 香甜类目标关键词（食欲偏置用）
const SWEET_RE = /蛋糕|香甜|甜|水果|果子|果|cake|sweet|fruit/

function clamp01(v) {
  const n = Number(v)
  return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0
}

/** 8 向 → 主轴方向（'ne'→'n'、'se'→'s'…，取首字母）。 */
function toCardinal(dir8) {
  if (typeof dir8 !== 'string' || dir8.length === 0) return null
  return dir8[0]
}

// ---------------------------------------------------------------------------
// WorldEncoder：世界感知 → 固定向量
// ---------------------------------------------------------------------------
export class WorldEncoder {
  /**
   * @param {object} senseResult 六感反馈 {sights,sounds,smells,touches}，
   *   每项 {text, type, dir?, id?, ...} 或纯字符串
   * @param {object} bodyState {x, y, z, dir(0-3), energy|能量, pain|痛感,
   *   satiety|饱腹|fullness, speed|速度}
   * @returns {Float64Array} 152 维：词袋 128 + 欲望占位 16（0）+ 身体 8
   */
  encode(senseResult, bodyState) {
    const vec = new Float64Array(BODYN_INPUT_SIZE)
    const src = senseResult || {}
    // 六感哈希词袋 → 128 桶
    for (const key of SENSE_KEYS) {
      const list = src[key]
      if (!Array.isArray(list)) continue
      for (const item of list) {
        const obj = item && typeof item === 'object' ? item : null
        const text = String((obj ? obj.text : item) ?? '')
        const type = String((obj ? obj.type : '') ?? '')
        const seedText = (type ? type + ':' : '') + text
        if (!seedText) continue
        vec[fnv1a(seedText) % 128] += 1
      }
    }
    // 每桶计数 / 4 归一（4 次命中 = 1.0，截断上限 1）
    for (let i = 0; i < 128; i++) vec[i] = Math.min(1, vec[i] / 4)
    // 身体状态 8 维
    if (bodyState) {
      const bs = bodyState
      vec[144] = clamp01(Number(bs.x ?? 0) / 40)
      vec[145] = clamp01(Number(bs.y ?? 0) / 40)
      vec[146] = clamp01(Number(bs.z ?? 0) / 8)
      vec[147] = clamp01(Number(bs.dir ?? 0) / 3)
      vec[148] = clamp01(Number(bs.energy ?? bs.能量 ?? 0))
      vec[149] = clamp01(Number(bs.pain ?? bs.痛感 ?? 0))
      vec[150] = clamp01(Number(bs.satiety ?? bs.饱腹 ?? bs.fullness ?? 0))
      vec[151] = clamp01(Number(bs.speed ?? bs.速度 ?? 0))
    }
    return vec
  }
}

// ---------------------------------------------------------------------------
// 欲望映射：内分泌 15 分子 → 欲望向量 16（规则写死）
// ---------------------------------------------------------------------------
/**
 * @param {Float64Array} endoLevels 内分泌浓度（endocrine.levels，15 维）
 * @returns {Float64Array} 16 维欲望向量，值域 [0,1]
 *   索引见 DESIRE_NAMES；2=食欲（desire[2] 供食欲偏置使用）
 */
export function desireFromEndocrine(endoLevels) {
  const d = new Float64Array(16)
  const g = (id) => {
    const i = MOLECULE_INDEX[id]
    return endoLevels && endoLevels.length > i ? clamp01(endoLevels[i]) : 0
  }
  d[0] = clamp01(g('dopamine') * 0.8 + g('glutamate') * 0.3) // 探索欲
  d[1] = clamp01(g('cortisol') * 0.7 + g('adrenaline') * 0.4 - g('gaba') * 0.3) // 回避欲
  d[2] = clamp01(g('dopamine') * 0.45 + g('endorphin') * 0.2 + (1 - g('serotonin')) * 0.4) // 食欲（无 ghrelin，低血清素作饥饿代理）
  d[3] = clamp01(g('melatonin') * 0.8 + g('gaba') * 0.3) // 休息欲
  d[4] = clamp01(g('oxytocin') * 0.7 + g('estrogen') * 0.2) // 社交欲
  d[5] = clamp01(g('testosterone') * 0.7 + g('adrenaline') * 0.3) // 竞争欲
  d[6] = clamp01(g('serotonin') * 0.5 + g('oxytocin') * 0.3) // 安抚欲
  d[7] = clamp01(g('dopamine') * 0.4 + g('noradrenaline') * 0.4 + g('acetylcholine') * 0.3) // 好奇欲
  d[8] = clamp01(g('noradrenaline') * 0.6 + g('adrenaline') * 0.5) // 警觉欲
  d[9] = clamp01(g('endorphin') * 0.7 + g('dopamine') * 0.3) // 愉悦维持
  d[10] = clamp01(g('testosterone') * 0.5 + g('adrenaline') * 0.4 - g('oxytocin') * 0.4) // 攻击欲
  d[11] = clamp01(g('oxytocin') * 0.6 + g('prolactin') * 0.3) // 亲和欲
  d[12] = clamp01(g('prolactin') * 0.5 + g('oxytocin') * 0.4) // 依赖欲
  d[13] = clamp01((1 - g('cortisol')) * 0.5 + g('gaba') * 0.2) // 清洁欲
  d[14] = clamp01(g('adrenaline') * 0.6 + g('cortisol') * 0.4) // 逃生欲
  d[15] = clamp01(g('glutamate') * 0.5 + g('acetylcholine') * 0.3) // 成长欲
  return d
}

// ---------------------------------------------------------------------------
// BodyNet：身体数据网络
// ---------------------------------------------------------------------------
export class BodyNet {
  constructor({ seed = 20261102, baseLr = 0.01 } = {}) {
    this.seed = seed >>> 0
    this.baseLr = baseLr
    this.baseline = 0 // REINFORCE baseline（内部 EMA）
    this.rngCalls = 0 // 采样随机流调用计数（可序列化，保证往返一致）
    this.lastAction = null
    this.encoder = new WorldEncoder()
    this.net = new PlasticNetwork({
      inputSize: BODYN_INPUT_SIZE,
      layers: [
        { size: 96, activation: 'tanh', lr: 1, mutable: false },
        { size: BODYN_OUTPUT_SIZE, activation: 'tanh', lr: 1, mutable: false },
      ],
      seed: this.seed,
    })
  }

  /**
   * 最终输入组装（由 brain 调用）：
   *   senseVec(≥128) 前 128 维 + desire16 → [128..144) + body8 → [144..152)
   * body8 缺省时沿用 senseVec 尾部（WorldEncoder.encode 已填身体 8 维）。
   * @returns {Float64Array} 152 维
   */
  static assembleInput(encoded, desire16, body8) {
    const input = new Float64Array(BODYN_INPUT_SIZE)
    const enc = encoded || new Float64Array(0)
    for (let i = 0; i < 128 && i < enc.length; i++) input[i] = enc[i]
    if (desire16) {
      for (let i = 0; i < 16 && i < desire16.length; i++) input[128 + i] = clamp01(desire16[i])
    }
    if (body8 && body8.length) {
      for (let i = 0; i < 8 && i < body8.length; i++) input[144 + i] = clamp01(body8[i])
    } else if (enc.length >= BODYN_INPUT_SIZE) {
      for (let i = 0; i < 8; i++) input[144 + i] = clamp01(enc[144 + i])
    }
    return input
  }

  /**
   * 行为决策。
   * @param {object} senseResult 六感反馈
   * @param {object} bodyState 身体状态（含 pain/痛感 用于痛感偏置）
   * @param {Float64Array} desire16 欲望向量（食欲 desire[2] 用于食欲偏置）
   * @param {object} opts {temperature=0.8}
   * @returns {{action:'move'|'jump'|'grab'|'eat'|'rest', dir:'n'|'s'|'e'|'w'|null,
   *            target?:string, probs:Float64Array(6), output:Float64Array(17)}}
   */
  decide(senseResult, bodyState, desire16, opts = {}) {
    const temperature = Number.isFinite(opts.temperature) && opts.temperature > 0 ? opts.temperature : 0.8
    const encoded = this.encoder.encode(senseResult, bodyState)
    const input = BodyNet.assembleInput(encoded, desire16)
    const { post } = this.net.forward(input)
    const output = new Float64Array(post[post.length - 1])

    // ---- 移动 6 维：softmax + 温度 ----
    const logits = new Float64Array(6)
    let maxLogit = -Infinity
    for (let i = 0; i < 6; i++) {
      logits[i] = output[i] / temperature
      if (logits[i] > maxLogit) maxLogit = logits[i]
    }
    const probs = new Float64Array(6)
    let probSum = 0
    for (let i = 0; i < 6; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit)
      probSum += probs[i]
    }
    for (let i = 0; i < 6; i++) probs[i] /= probSum

    // ---- 欲望偏置：食欲高 + 香甜 smell → 移动概率偏向目标方向 0.5 权重 ----
    const sweet = this._findSweetTarget(senseResult)
    if (sweet && desire16 && desire16[2] > 0.6) {
      const dirIdx = MOVE_DIRS.indexOf(sweet.dir)
      if (dirIdx >= 0) {
        for (let i = 0; i < 6; i++) probs[i] *= 0.5
        probs[dirIdx] += 0.5
      }
    }

    // ---- 痛感偏置：痛感高 → 静止概率 +0.4（再归一）----
    const pain = Number((bodyState && (bodyState.pain ?? bodyState.痛感)) ?? 0)
    if (pain >= 0.5) {
      probs[5] += 0.4
      let s2 = 0
      for (let i = 0; i < 6; i++) s2 += probs[i]
      for (let i = 0; i < 6; i++) probs[i] /= s2
    }

    // ---- 采样（确定性：seed + 调用计数派生随机流）----
    const rng = mulberry32(((this.seed ^ 0x9e3779b9) >>> 0) + this.rngCalls * 0x2545f491)
    this.rngCalls++
    let r = rng()
    let moveIdx = 5
    for (let i = 0; i < 6; i++) {
      if (r < probs[i]) { moveIdx = i; break }
      r -= probs[i]
    }

    // ---- 交互 3 维 argmax（rest 兜底）----
    let interIdx = 8
    for (let i = 6; i <= 8; i++) if (output[i] > output[interIdx]) interIdx = i

    // ---- 目标方位 8 维 argmax ----
    let tIdx = 9
    for (let i = 10; i <= 16; i++) if (output[i] > output[tIdx]) tIdx = i
    const targetDir8 = TARGET_DIRS[tIdx - 9]

    // ---- 目标物（香甜目标优先，其次带 id 的感知项）----
    const targetItem = sweet ? sweet.item : this._findNearestTarget(senseResult, targetDir8)

    // ---- 动作合成 ----
    let action
    let dir = null
    let target
    if (moveIdx === 4) {
      action = 'jump'
    } else if (moveIdx === 5) {
      // 静止 → 执行交互 argmax；无目标物时 rest 兜底
      if (interIdx === 6 && targetItem) {
        action = 'grab'
        dir = toCardinal(targetDir8)
        target = itemIdOf(targetItem)
      } else if (interIdx === 7 && targetItem) {
        action = 'eat'
        dir = toCardinal(targetDir8)
        target = itemIdOf(targetItem)
      } else {
        action = 'rest'
      }
    } else {
      action = 'move'
      dir = MOVE_DIRS[moveIdx]
    }

    const result = { action, dir, probs, output }
    if (target !== undefined && target !== null) result.target = target
    this.lastAction = { action, dir, target: target ?? null }
    return result
  }

  /** 嗅到的香甜类目标（带方向）。 */
  _findSweetTarget(senseResult) {
    if (!senseResult) return null
    const list = Array.isArray(senseResult.smells) ? senseResult.smells : []
    for (const it of list) {
      const obj = it && typeof it === 'object' ? it : null
      const text = String((obj ? obj.text : it) ?? '')
      const type = String((obj ? obj.type : '') ?? '')
      if (!SWEET_RE.test(text + type)) continue
      let dir = null
      if (obj && typeof obj.dir === 'string') dir = obj.dir
      else if (obj && typeof obj.dx === 'number' && typeof obj.dy === 'number') {
        const adx = Math.abs(obj.dx)
        const ady = Math.abs(obj.dy)
        if (adx + ady > 0) dir = adx >= ady ? (obj.dx >= 0 ? 'e' : 'w') : (obj.dy >= 0 ? 'n' : 's')
      }
      if (dir === 'n' || dir === 's' || dir === 'e' || dir === 'w') return { dir, item: obj }
    }
    return null
  }

  /** 最近目标物：带 id 的感知项，优先方向匹配。 */
  _findNearestTarget(senseResult, preferredDir8) {
    if (!senseResult) return null
    let fallback = null
    for (const key of SENSE_KEYS) {
      const list = senseResult[key]
      if (!Array.isArray(list)) continue
      for (const it of list) {
        if (!it || typeof it !== 'object' || itemIdOf(it) === undefined) continue
        if (typeof it.dir === 'string' && it.dir === preferredDir8) return it
        if (!fallback) fallback = it
      }
    }
    return fallback
  }

  /**
   * REINFORCE 策略梯度（退化实现）。
   * @param {Float64Array} prevInput 行为前的组装输入（152 维）
   * @param {Float64Array} output17 行为前 forward 的 17 维输出
   * @param {number} chosenIndex 实际执行动作在 17 维输出中的下标
   * @param {number} reward 行为效果奖励（内分泌变化量）
   * @param {object} opts {epochs=1, lrScale=1}
   */
  train(prevInput, output17, chosenIndex, reward, opts = {}) {
    const epochs = Math.max(1, opts.epochs ?? 1)
    const lrScale = opts.lrScale ?? 1
    const idx = Math.max(0, Math.min(BODYN_OUTPUT_SIZE - 1, Number(chosenIndex) | 0))
    const adv = reward - this.baseline
    this.baseline = this.baseline * 0.9 + 0.1 * reward // baseline EMA（init 0）
    if (Math.abs(adv) < 1e-9) {
      return { loss: 0, advantage: 0, baseline: this.baseline, epochs, skipped: 'flat' }
    }
    // 目标输出 = 原 output；chosenIndex 处按 advantage 符号置 ±|adv|（0.2..0.9）
    const mag = Math.min(0.9, Math.max(0.2, Math.abs(adv)))
    const target = new Float64Array(BODYN_OUTPUT_SIZE)
    for (let i = 0; i < BODYN_OUTPUT_SIZE; i++) target[i] = output17 && i < output17.length ? output17[i] : 0
    target[idx] = adv >= 0 ? mag : -mag
    let total = 0
    for (let e = 0; e < epochs; e++) {
      const res = this.net.train(prevInput, target, { baseLr: this.baseLr * lrScale })
      total += res.loss ?? 0
    }
    return { loss: total / epochs, advantage: adv, baseline: this.baseline, epochs }
  }

  toJSON() {
    return {
      seed: this.seed,
      baseLr: this.baseLr,
      baseline: this.baseline,
      rngCalls: this.rngCalls,
      lastAction: this.lastAction,
      net: this.net.toJSON(),
    }
  }

  static fromJSON(json) {
    const inst = Object.create(BodyNet.prototype)
    inst.seed = (json.seed ?? 20261102) >>> 0
    inst.baseLr = json.baseLr ?? 0.01
    inst.baseline = json.baseline ?? 0
    inst.rngCalls = json.rngCalls ?? 0
    inst.lastAction = json.lastAction ?? null
    inst.encoder = new WorldEncoder()
    inst.net = PlasticNetwork.fromJSON(json.net)
    return inst
  }

  status() {
    return { params: this.net.paramCount, baseline: this.baseline, lastAction: this.lastAction }
  }
}

/** 取感知项的 id（target 字段用）。 */
function itemIdOf(item) {
  if (!item || typeof item !== 'object') return undefined
  const v = item.id ?? item.itemId ?? item.targetId
  return v === undefined || v === null ? undefined : String(v)
}
