// ============================================================================
// dsh-plastic-memory —— 分析型神经网络 + 分析存储器（AnalysisNet / AnalysisStore）
// ----------------------------------------------------------------------------
// AnalysisNet：对感知流做元分析。
//   输入 136 维 = latent(128) + 内分泌效应(8)：
//     [learningRateMod, exploration, arousal, valence, socialAxis,
//      memorySalience, predError, 0]
//   结构 136 → 64(tanh) → 16(tanh)。
//   每次 analyze 同时执行 1 步在线自编码式学习：目标 = latent 前 16 维，
//   使分析网络随感知流持续学习（在线学习率固定 0.002，见契约）。
//   输出：
//     vector     16 维分析向量
//     topicScore 前 8 维均值绝对值（tanh 输出 ∈ [-1,1]，均值绝对值天然 ∈ [0,1]）
//     novelty    与上一帧 vector 的欧氏距离归一化（/最大距离 sqrt(16×4)=8；首帧 0.5）
//
// AnalysisStore：分析记录存储（内存缓冲 + 批量落盘，批量语料感知时零逐条磁盘 IO）。
//   append(record) 只写内存环形缓冲（容量 200 条，超出丢最旧）；
//   flush() 把缓冲一次性 JSONL 原子追加到 dir/records.jsonl 并清空缓冲，
//     随后若文件超 1000 行则截断保留后 800 行；
//   recent(n)/count() 读文件统计并与缓冲合并（文件尾部 + 缓冲，新在后）；坏行跳过。
//   记录字段：t / text(≤100) / module / vector(16) / topicScore / novelty。
// ============================================================================

import fs from 'node:fs'
import path from 'node:path'
import { PlasticNetwork } from './network.mjs'
import { arrToPlain, arrFromPlain } from './core.mjs'

// 维度常量
export const ANALYSIS_INPUT_SIZE = 136 // latent 128 + 内分泌效应 8
export const ANALYSIS_OUTPUT_SIZE = 16
export const ANALYSIS_TOPIC_DIMS = 8 // topicScore 使用前 8 维

// 在线学习步的固定学习率（契约约定）
const ONLINE_LR = 0.002

// 分析存储：内存缓冲容量；文件超 MAX_LINES 行截断保留后 KEEP_LINES 行
const BUFFER_CAPACITY = 200
const MAX_LINES = 1000
const KEEP_LINES = 800

export class AnalysisNet {
  constructor({ seed = 20261030, baseLr = 0.01 } = {}) {
    this.baseLr = baseLr
    this.net = new PlasticNetwork({
      inputSize: ANALYSIS_INPUT_SIZE,
      layers: [
        { size: 64, activation: 'tanh', lr: 1, mutable: false },
        { size: ANALYSIS_OUTPUT_SIZE, activation: 'tanh', lr: 1, mutable: false },
      ],
      seed,
    })
    this.lastVector = null
    this.lastNovelty = 0
  }

  /**
   * @param {Float64Array} latent 128 维感知潜向量
   * @param {object} effects 内分泌效应：learningRateMod / arousal / valence /
   *   socialAxis / memorySalience（缺省 0.5 / 0 / 0 / 0 / 0.5）；
   *   exploration / predError 可选（缺省 0.2 / 0.5）
   * @returns {{vector: Float64Array, topicScore: number, novelty: number}}
   */
  analyze(latent, effects = {}) {
    effects = effects ?? {}

    // ---- 组装 136 维输入：latent(128) + 内分泌效应(8) ----
    const input = new Float64Array(ANALYSIS_INPUT_SIZE)
    if (latent) {
      for (let i = 0; i < 128 && i < latent.length; i++) input[i] = latent[i]
    }
    input[128] = effects.learningRateMod ?? 0.5
    input[129] = effects.exploration ?? 0.2
    input[130] = effects.arousal ?? 0
    input[131] = effects.valence ?? 0
    input[132] = effects.socialAxis ?? 0
    input[133] = effects.memorySalience ?? 0.5
    input[134] = effects.predError ?? 0.5
    input[135] = 0

    // ---- 前向：取输出层 16 维 ----
    const { post } = this.net.forward(input)
    const vector = new Float64Array(post[post.length - 1])

    // ---- 1 步在线学习：自编码式，目标 = latent 前 16 维 ----
    const target = new Float64Array(ANALYSIS_OUTPUT_SIZE)
    for (let i = 0; i < ANALYSIS_OUTPUT_SIZE; i++) target[i] = latent ? (latent[i] ?? 0) : 0
    this.net.train(input, target, { baseLr: ONLINE_LR })

    // ---- topicScore：前 8 维均值绝对值（0..1 归一）----
    let acc = 0
    for (let i = 0; i < ANALYSIS_TOPIC_DIMS; i++) acc += Math.abs(vector[i])
    const topicScore = Math.min(1, Math.max(0, acc / ANALYSIS_TOPIC_DIMS))

    // ---- novelty：与上一帧的欧氏距离 / 最大可能距离 sqrt(16×4) = 8 ----
    let novelty
    if (this.lastVector) {
      let d2 = 0
      for (let i = 0; i < ANALYSIS_OUTPUT_SIZE; i++) {
        const e = vector[i] - this.lastVector[i]
        d2 += e * e
      }
      novelty = Math.min(1, Math.max(0, Math.sqrt(d2) / 8))
    } else {
      novelty = 0.5 // 首帧
    }
    this.lastVector = vector
    this.lastNovelty = novelty

    return { vector, topicScore, novelty }
  }

  toJSON() {
    return {
      baseLr: this.baseLr,
      net: this.net.toJSON(),
      lastVector: this.lastVector ? arrToPlain(this.lastVector) : null,
      lastNovelty: this.lastNovelty,
    }
  }

  static fromJSON(json) {
    const inst = Object.create(AnalysisNet.prototype)
    inst.baseLr = json.baseLr ?? 0.01
    inst.net = PlasticNetwork.fromJSON(json.net)
    inst.lastVector = json.lastVector ? arrFromPlain(json.lastVector) : null
    inst.lastNovelty = json.lastNovelty ?? 0
    return inst
  }

  status() {
    return { params: this.net.paramCount, ticks: this.net.tick, lastNovelty: this.lastNovelty }
  }
}

export class AnalysisStore {
  /**
   * @param {string} dir 工作区 analysis 目录绝对路径（records.jsonl 的目标目录）
   */
  constructor(dir) {
    this.dir = dir
    this.file = path.join(dir, 'records.jsonl')
    this._buffer = [] // 内存环形缓冲（容量 BUFFER_CAPACITY，超出丢最旧）
  }

  /**
   * 写入内存环形缓冲（不落盘；容量 200 条，超出丢最旧）。
   * @param {object} record {t, text(≤100), module, vector:[16 数字], topicScore, novelty}
   */
  append(record) {
    const rec = {
      t: record.t,
      text: String(record.text ?? '').slice(0, 100),
      module: record.module,
      vector: ArrayBuffer.isView(record.vector) ? Array.from(record.vector) : record.vector,
      topicScore: record.topicScore,
      novelty: record.novelty,
    }
    this._buffer.push(rec)
    if (this._buffer.length > BUFFER_CAPACITY) this._buffer.shift() // 丢最旧
    return rec
  }

  /**
   * 把缓冲全部追加写入 records.jsonl（单次 appendFileSync 原子追加）并清空缓冲；
   * 随后若文件超 1000 行则截断保留后 800 行。返回本次落盘条数。
   */
  flush() {
    if (this._buffer.length === 0) return 0
    fs.mkdirSync(this.dir, { recursive: true })
    const lines = this._buffer.map((r) => JSON.stringify(r))
    fs.appendFileSync(this.file, lines.join('\n') + '\n', 'utf8')
    const n = this._buffer.length
    this._buffer.length = 0
    this._truncateIfNeeded()
    return n
  }

  /** 读原始非空行。 */
  _readLines() {
    if (!fs.existsSync(this.file)) return []
    return fs
      .readFileSync(this.file, 'utf8')
      .split('\n')
      .filter((l) => l.trim().length > 0)
  }

  _isValidRecord(rec) {
    return rec !== null && typeof rec === 'object' && !Array.isArray(rec) && Array.isArray(rec.vector)
  }

  /** 文件中的有效记录（坏行跳过），时间序（旧 → 新）。 */
  _fileRecords() {
    const out = []
    for (const line of this._readLines()) {
      try {
        const rec = JSON.parse(line)
        if (this._isValidRecord(rec)) out.push(rec)
      } catch {
        /* 坏行跳过 */
      }
    }
    return out
  }

  /** 最近 n 条 = 文件尾部 + 缓冲（新在后；坏行跳过）。 */
  recent(n = 100) {
    const buffered = this._buffer.filter((r) => this._isValidRecord(r))
    return this._fileRecords().concat(buffered).slice(-n)
  }

  /** 总条数 = 文件有效记录 + 缓冲有效记录（文件不存在 = 0；坏行跳过）。 */
  count() {
    let buffered = 0
    for (const r of this._buffer) if (this._isValidRecord(r)) buffered++
    return this._fileRecords().length + buffered
  }

  /** 文件超 1000 行时截断保留后 800 行（临时文件 + rename 原子替换；flush 时调用）。 */
  _truncateIfNeeded() {
    const lines = this._readLines()
    if (lines.length <= MAX_LINES) return
    const kept = lines.slice(-KEEP_LINES)
    const tmp = this.file + '.tmp'
    fs.writeFileSync(tmp, kept.join('\n') + '\n', 'utf8')
    fs.renameSync(tmp, this.file)
  }
}
