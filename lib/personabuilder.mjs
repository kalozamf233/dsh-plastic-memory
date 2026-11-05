// ============================================================================
// dsh-plastic-memory —— 人格构造神经网络 + 人格构造器（PersonaBuilder）
// ----------------------------------------------------------------------------
// 构造网络：复用 lib/nn/flawnet.mjs 的 DefectNet（PlasticNetwork 143 → 64 → 8，
// 即缺陷人格的输出空间），独立文件存储 persona-builder.brain.json。
//
// 四源采样（collectSources）：
//   1. 记忆网络   brain.memory.expressPersona(0)         —— 人格向量前 64 维
//   2. 镜像网络   brain.trainer.loadMirror()?.expressPersona(0) —— 前 64 维
//   3. 回忆对话网络 brain.recallNet.net.forward(编码后).post —— 输出 2 维
//                 复制填充扩为 8 维
//   4. 缺陷人格网络 brain.flaw.net 输出层权重展平 —— 前 64 维
//   四源拼接（64+64+8+64 = 200 维）→ 合成目标 = 前 128 维（存内部）。
//   任一源缺失/抛错 → 该源用零向量（空源容错）。
//
// build({epochs=40})：
//   输入 143 = hash 词袋('persona-builder-source')(128) + 四源拼接后 15 维
//   目标 8   = 0.7 × 四源平均（各源按 8 块均值降为 8 维后四源平均）
//            + 0.3 × 当前缺陷人格输出（同输入前向的 8 维输出）
//   反复 train epochs 轮（构造网络 frozen 时跳过）。
//
// commitReplace()：生成新的缺陷人格网络 JSON（{replaced:true, before, after}），
//   真正写入 flaw.brain.json / 替换 brain.flaw 由 lead 在 brain 层执行。
// ============================================================================

import fs from 'node:fs'
import path from 'node:path'
import { DefectNet } from './nn/flawnet.mjs'
import { fnv1a } from './nn/encoder.mjs'

// 常量
export const SOURCE_TEXT = 'persona-builder-source' // 合成文本特征
const LATENT_SIZE = 128 // 词袋维度
const INPUT_SIZE = 143 // DefectNet 输入：latent 128 + 内分泌 15
const OUT_SIZE = 8 // 缺陷人格输出空间
const CONCAT_SIZE = 200 // 四源拼接：64 + 64 + 8 + 64
const TARGET_SIZE = 128 // 合成目标（四源拼接截断）
const FLAW_W = 64 // 缺陷人格输出层权重取前 64 维
const BUILD_INPUT_TAIL = 15 // build 输入取四源拼接后 15 维

function clamp1(v) {
  const n = Number(v)
  return Number.isFinite(n) ? Math.max(-1, Math.min(1, n)) : 0
}

/** 任意向量形态 → Float64Array(len)（前 len 维，不足补零；缺失/损坏 → 全零）。 */
function toFloat64(v, len) {
  const out = new Float64Array(len)
  if (!v) return out
  const arr = ArrayBuffer.isView(v) ? v : Array.isArray(v) ? v : v && ArrayBuffer.isView(v.data) ? v.data : null
  if (!arr) return out
  for (let i = 0; i < len && i < arr.length; i++) {
    const n = Number(arr[i])
    out[i] = Number.isFinite(n) ? n : 0
  }
  return out
}

/** 字符 1/2/3-gram 哈希词袋（128 维，L2 归一）。 */
function hashBag128(text) {
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

/** 短向量 → 8 维（复制填充：out[i] = src[i % len]）。 */
function expandTo8(v) {
  const out = new Float64Array(OUT_SIZE)
  const src = toFloat64(v, v && v.length ? v.length : 0)
  if (src.length === 0) return out
  for (let i = 0; i < OUT_SIZE; i++) out[i] = src[i % src.length]
  return out
}

/** 任意长向量 → 8 维（分 8 块取均值；长度 ≤8 时复制填充）。 */
function chunkMean8(v) {
  const out = new Float64Array(OUT_SIZE)
  const src = toFloat64(v, v && v.length ? v.length : 0)
  if (src.length === 0) return out
  if (src.length <= OUT_SIZE) {
    for (let i = 0; i < OUT_SIZE; i++) out[i] = src[i % src.length]
    return out
  }
  for (let b = 0; b < OUT_SIZE; b++) {
    const start = Math.floor((b * src.length) / OUT_SIZE)
    const end = Math.floor(((b + 1) * src.length) / OUT_SIZE)
    let s = 0
    for (let i = start; i < end; i++) s += src[i]
    out[b] = s / (end - start)
  }
  return out
}

export class PersonaBuilder {
  /**
   * @param {object} brain brain 实例（含 memory/trainer/recallNet/flaw）
   * @param {object} opts {builderFile?: string 独立文件 persona-builder.brain.json,
   *                       seed?: number}
   */
  constructor(brain, opts = {}) {
    this.brain = brain ?? null
    this.builderFile = opts.builderFile ?? null
    this.seed = opts.seed ?? 20261106
    this.net = new DefectNet({ seed: this.seed }) // 构造网络 = DefectNet（143→64→8）
    this.collected = null // {sources[4], concat:200, target:128, fourAvg:8, at}
    this.builds = 0
    this.commits = 0
    this.lastLoss = null
    this.lastError = null
    this.lastCommit = null
    // 独立文件存储：存在则加载（损坏则忽略，用全新网络）
    if (this.builderFile && fs.existsSync(this.builderFile)) {
      try {
        this._loadFile(this.builderFile)
      } catch {
        /* 损坏文件忽略 */
      }
    }
  }

  /** 四源采样 → 合成训练样本。 */
  collectSources() {
    const sources = []
    // 1. 记忆网络人格向量（前 64 维）
    sources.push(
      this._safeSource(
        () => (this.brain && this.brain.memory && typeof this.brain.memory.expressPersona === 'function'
          ? this.brain.memory.expressPersona(0)
          : null),
        64,
        'memory',
      ),
    )
    // 2. 镜像网络人格向量（前 64 维）
    sources.push(
      this._safeSource(
        () => {
          const mirror = this.brain && this.brain.trainer && typeof this.brain.trainer.loadMirror === 'function'
            ? this.brain.trainer.loadMirror()
            : null
          return mirror && typeof mirror.expressPersona === 'function' ? mirror.expressPersona(0) : null
        },
        64,
        'mirror',
      ),
    )
    // 3. 回忆对话网络输出 2 维 → 8 维（复制填充）
    sources.push(
      this._safeSource(
        () => {
          const rn = this.brain && this.brain.recallNet
          if (!rn || !rn.net || typeof rn.net.forward !== 'function') return null
          const fwd = rn.net.forward(hashBag128(SOURCE_TEXT))
          const out = fwd.post[fwd.post.length - 1]
          return expandTo8(out)
        },
        OUT_SIZE,
        'recall',
      ),
    )
    // 4. 缺陷人格网络输出层权重展平前 64 维
    sources.push(
      this._safeSource(
        () => {
          const net = this.brain && this.brain.flaw && this.brain.flaw.net
          const out = net && net.layers && net.layers[net.layers.length - 1]
          return out && out.W && out.W.data ? toFloat64(out.W.data, FLAW_W) : null
        },
        FLAW_W,
        'flaw',
      ),
    )

    // 四源拼接（200 维）→ 合成目标 = 前 128 维（存内部）
    const concat = new Float64Array(CONCAT_SIZE)
    let off = 0
    for (const s of sources) {
      concat.set(s, off)
      off += s.length
    }
    const target = concat.slice(0, TARGET_SIZE)
    // 四源平均 → 8 维（各源 8 块均值后再平均）
    const fourAvg = new Float64Array(OUT_SIZE)
    for (const s of sources) {
      const c = chunkMean8(s)
      for (let i = 0; i < OUT_SIZE; i++) fourAvg[i] += c[i]
    }
    for (let i = 0; i < OUT_SIZE; i++) fourAvg[i] /= sources.length

    this.collected = { sources, concat, target, fourAvg, at: Date.now() }
    return this.collected
  }

  /** 源采样安全包装：抛错/缺失 → 零向量（空源容错）。 */
  _safeSource(fn, len, name) {
    try {
      const v = fn()
      if (v === null || v === undefined) return new Float64Array(len)
      return toFloat64(v, len)
    } catch (e) {
      this.lastError = name + ': ' + (e && e.message ? e.message : String(e))
      return new Float64Array(len)
    }
  }

  /**
   * 以合成目标反复训练构造网络。
   * @param {object} opts {epochs=40}
   */
  build(opts = {}) {
    const epochs = Math.max(1, opts.epochs ?? 40)
    if (!this.collected) this.collectSources()
    if (this.net.frozen) return { loss: 0, epochs: 0, skipped: 'frozen' }

    // 输入 143 = hash 词袋('persona-builder-source') + 四源拼接后 15 维
    const input = new Float64Array(INPUT_SIZE)
    input.set(hashBag128(SOURCE_TEXT), 0)
    const concat = this.collected.concat
    for (let i = 0; i < BUILD_INPUT_TAIL; i++) {
      input[LATENT_SIZE + i] = concat[CONCAT_SIZE - BUILD_INPUT_TAIL + i]
    }

    // 当前缺陷人格输出（同输入前向的 8 维原始输出）
    let flawOut = new Float64Array(OUT_SIZE)
    try {
      const flawNet = this.brain && this.brain.flaw && this.brain.flaw.net
      if (flawNet && typeof flawNet.forward === 'function') {
        const fwd = flawNet.forward(input)
        flawOut = new Float64Array(toFloat64(fwd.post[fwd.post.length - 1], OUT_SIZE))
      }
    } catch {
      /* 缺源 → 零 */
    }

    // 目标 8 维 = 0.7×四源平均 + 0.3×当前缺陷人格输出
    const target = new Float64Array(OUT_SIZE)
    for (let i = 0; i < OUT_SIZE; i++) {
      target[i] = clamp1(0.7 * this.collected.fourAvg[i] + 0.3 * flawOut[i])
    }

    let total = 0
    for (let e = 0; e < epochs; e++) {
      const res = this.net.net.train(input, target, { baseLr: this.net.baseLr })
      total += res.loss ?? 0
    }
    this.builds++
    this.lastLoss = total / epochs
    return { loss: this.lastLoss, epochs }
  }

  /**
   * 提交替换：生成新的缺陷人格网络 JSON。
   * 真正写入 flaw.brain.json / 替换 brain.flaw 由 lead 在 brain 层执行，
   * 本方法只返回 {replaced:true, before, after}。
   */
  commitReplace() {
    let before = null
    try {
      before = this.brain && this.brain.flaw && typeof this.brain.flaw.toJSON === 'function'
        ? this.brain.flaw.toJSON()
        : null
    } catch {
      before = null
    }
    const after = this.net.toJSON() // DefectNet 序列化（143→64→8 + frozen + internal）
    this.commits++
    this.lastCommit = { at: Date.now(), before, after }
    return { replaced: true, before, after }
  }

  toJSON() {
    return {
      seed: this.seed,
      net: this.net.toJSON(),
      collected: this.collected
        ? {
            sources: this.collected.sources.map((s) => Array.from(s)),
            concat: Array.from(this.collected.concat),
            target: Array.from(this.collected.target),
            fourAvg: Array.from(this.collected.fourAvg),
            at: this.collected.at,
          }
        : null,
      builds: this.builds,
      commits: this.commits,
      lastLoss: this.lastLoss,
      lastError: this.lastError,
    }
  }

  static fromJSON(json, brain = null) {
    const inst = Object.create(PersonaBuilder.prototype)
    inst.brain = brain ?? null
    inst.builderFile = null
    inst.seed = json.seed ?? 20261106
    inst.net = DefectNet.fromJSON(json.net)
    inst.builds = json.builds ?? 0
    inst.commits = json.commits ?? 0
    inst.lastLoss = json.lastLoss ?? null
    inst.lastError = json.lastError ?? null
    inst.lastCommit = null
    if (json.collected && Array.isArray(json.collected.target)) {
      inst.collected = {
        sources: Array.isArray(json.collected.sources)
          ? json.collected.sources.map((s) => new Float64Array(s))
          : [],
        concat: new Float64Array(json.collected.concat ?? []),
        target: new Float64Array(json.collected.target),
        fourAvg: new Float64Array(json.collected.fourAvg ?? []),
        at: json.collected.at ?? null,
      }
    } else {
      inst.collected = null
    }
    return inst
  }

  /** 保存构造网络到独立文件（persona-builder.brain.json，tmp+rename 原子写）。 */
  save() {
    if (!this.builderFile) return false
    try {
      fs.mkdirSync(path.dirname(this.builderFile), { recursive: true })
      const tmp = this.builderFile + '.tmp'
      fs.writeFileSync(tmp, JSON.stringify(this.toJSON()), 'utf8')
      fs.renameSync(tmp, this.builderFile)
      return true
    } catch (e) {
      this.lastError = 'save: ' + (e && e.message ? e.message : String(e))
      return false
    }
  }

  _loadFile(file) {
    const json = JSON.parse(fs.readFileSync(file, 'utf8'))
    const loaded = PersonaBuilder.fromJSON(json, this.brain)
    this.net = loaded.net
    this.collected = loaded.collected
    this.builds = loaded.builds
    this.commits = loaded.commits
    this.lastLoss = loaded.lastLoss
    this.lastError = loaded.lastError
  }

  status() {
    return {
      params: this.net.net.paramCount,
      frozen: this.net.frozen,
      builds: this.builds,
      commits: this.commits,
      lastLoss: this.lastLoss,
      hasCollected: !!this.collected,
    }
  }
}
