// ============================================================================
// dsh-plastic-memory —— 训练控制器（TrainingController）
// ----------------------------------------------------------------------------
// 睡眠期训练调度器：由心跳 tick 驱动，以限流方式（每 tick 最多 max 条）消化
// 预存队列（pending.jsonl），对记忆网络执行"睡眠巩固"训练，并维护训练态
// 镜像（mirror.brain.json）用于上下文命中判定与加倍训练。
//
// 文件约定（与 Workspace 一致：所有写入 tmp + rename 原子替换，崩溃安全）：
//   mirror.brain.json   记忆网络训练态镜像（MemoryNet.toJSON() 原子写入）
//   pending.jsonl       预存训练队列（JSONL：{t,text,latent32,module}，坏行跳过）
//
// 零外部依赖：node:fs / node:path + lib/nn（MemoryNet / core.cosine）。
// ============================================================================

import {
  mkdirSync, readFileSync, writeFileSync, renameSync, unlinkSync, appendFileSync, statSync,
} from 'node:fs'
import { dirname } from 'node:path'
import { MemoryNet, MODULES } from './nn/memorynet.mjs'
import { cosine } from './nn/core.mjs'

// ---------------------------------------------------------------------------
// 常量（lead 集成时可引用）
// ---------------------------------------------------------------------------

export const MIRROR_FILE_NAME = 'mirror.brain.json'
export const PENDING_FILE_NAME = 'pending.jsonl'
export const MAX_TEXT_LEN = 200
export const LATENT32_DIM = 32
export const MIRROR_HIT_THRESHOLD = 0.6

/** 与 Workspace.atomicWrite 同约定的原子写（tmp + rename，失败清理 tmp）。 */
function atomicWriteFile(file, content) {
  const dir = dirname(file)
  if (dir) mkdirSync(dir, { recursive: true })
  const tmp = `${file}.tmp`
  writeFileSync(tmp, content, 'utf8')
  try {
    renameSync(tmp, file)
  } catch (error) {
    try {
      unlinkSync(tmp)
    } catch {
      /* ignore */
    }
    throw error
  }
}

// ============================================================================
// TrainingController
// ============================================================================

export class TrainingController {
  /**
   * @param {object} brain PlasticBrain（lead 集成前可用最小 mock）：
   *   brain.memory      MemoryNet（observe / net.layers[0].W / toJSON / fromJSON）
   *   brain.controller  ControllerNet（hear(text, trainEncoder) → {latent}）
   *   brain.endocrine   EndocrineSystem（get / set / effects().emotionTarget）
   *   brain.workspace   Workspace（paths.mirror / paths.pending，可选）
   * @param {object} [opts] {mirrorFile, pendingFile}：显式路径优先；
   *   缺省取 brain.workspace.paths.mirror / paths.pending；两者均缺失则抛错提示。
   */
  constructor(brain, { mirrorFile, pendingFile } = {}) {
    if (!brain) throw new Error('TrainingController: brain 不能为空')
    this.brain = brain
    const wsPaths = brain.workspace?.paths ?? {}
    this.mirrorFile = mirrorFile ?? wsPaths.mirror ?? null
    this.pendingFile = pendingFile ?? wsPaths.pending ?? null
    if (typeof this.mirrorFile !== 'string' || this.mirrorFile === '') {
      throw new Error(
        'TrainingController: 缺少镜像文件路径（brain.workspace.paths.mirror 未提供），' +
          '请在 constructor(brain, {mirrorFile}) 中显式传入',
      )
    }
    if (typeof this.pendingFile !== 'string' || this.pendingFile === '') {
      throw new Error(
        'TrainingController: 缺少预存队列文件路径（brain.workspace.paths.pending 未提供），' +
          '请在 constructor(brain, {pendingFile}) 中显式传入',
      )
    }
    this._stopped = false
    this.lastError = null
    this.recentErrors = [] // 最近错误（≤20 条，{t, msg}）
    this.trainedTotal = 0 // 累计已训练条数
    this.doubledTotal = 0 // 累计加倍条数
    this.skippedTotal = 0 // 累计跳过条数
    this.lastEntryDetail = null // 最近一条训练明细（调试用）
    this._mirror = null // 镜像 MemoryNet 缓存（mtime 失效）
    this._mirrorMtimeMs = -1
  }

  get stopped() {
    return this._stopped
  }

  // -------------------------------------------------------------------------
  // 镜像管理
  // -------------------------------------------------------------------------

  /** 读取镜像：mirror.brain.json 不存在或损坏 → null。 */
  loadMirror() {
    try {
      const raw = readFileSync(this.mirrorFile, 'utf8')
      const json = JSON.parse(raw)
      if (!json || typeof json !== 'object' || !json.net || !Array.isArray(json.net.layers)) return null
      const mirror = MemoryNet.fromJSON(json)
      return mirror && mirror.net && mirror.net.layers.length > 0 ? mirror : null
    } catch {
      return null
    }
  }

  /** 同步镜像：把 brain.memory.toJSON() 原子写入 mirror.brain.json。成功 true，异常 false。 */
  syncMirror() {
    try {
      const json = JSON.stringify(this.brain.memory.toJSON())
      if (typeof json !== 'string' || json.length === 0) {
        this.lastError = 'syncMirror: JSON 序列化结果为空'
        return false
      }
      atomicWriteFile(this.mirrorFile, json)
      this._mirror = null
      this._mirrorMtimeMs = -1
      return true
    } catch (error) {
      this._recordError(error)
      return false
    }
  }

  /** 取镜像（带 mtime 缓存，保证心跳实时；loadMirror 始终直读磁盘）。 */
  _getMirror() {
    try {
      const st = statSync(this.mirrorFile)
      if (this._mirror && st.mtimeMs === this._mirrorMtimeMs) return this._mirror
      const mirror = this.loadMirror()
      this._mirror = mirror
      this._mirrorMtimeMs = mirror ? st.mtimeMs : -1
      return mirror
    } catch {
      this._mirror = null
      this._mirrorMtimeMs = -1
      return null
    }
  }

  /**
   * 镜像回忆（上下文查熟悉度）：用镜像 MemoryNet.recall(latent, {module:'memory', salience:0.6})，
   * similarity 直接取 predSim = recall 输出 pred 前 128 维（对应 latent）与查询 latent 的
   * cosine（预测吻合度 = 熟悉度；未训练镜像预测差 → 不命中 → 单倍训练，熟悉 → 命中 → 加倍）。
   * actSim（记忆区激活绝对均值 ×2 clamp 0..1）仅作为返回字段附带，不参与判定。
   * found = predSim > 0.6。
   * 镜像不存在/损坏 → {found:false, similarity:-1, actSim:-1}。
   * @returns {{found:boolean, similarity:number, actSim:number}}
   */
  mirrorRecall(latent) {
    const mirror = this._getMirror()
    if (!mirror) return { found: false, similarity: -1, actSim: -1 }
    try {
      const rec = mirror.recall(latent, { module: 'memory', salience: 0.6 })
      // 记忆区激活度量（仅附带）
      const act = rec.memAct
      let sum = 0
      for (let i = 0; i < act.length; i++) sum += Math.abs(act[i])
      const meanAbs = act.length > 0 ? sum / act.length : 0
      const actSim = Math.max(0, Math.min(1, meanAbs * 2))
      // 熟悉度 = 预测吻合度：pred 前 128 维 vs latent
      let predSim = 0
      if (rec.pred && rec.pred.length >= 128 && latent && latent.length === 128) {
        const p128 = new Float64Array(rec.pred.subarray(0, 128))
        const l128 = new Float64Array(latent)
        predSim = cosine(p128, l128)
      }
      return {
        found: predSim > MIRROR_HIT_THRESHOLD,
        similarity: Number(predSim.toFixed(4)),
        actSim: Number(actSim.toFixed(4)),
      }
    } catch (error) {
      this._recordError(error)
      return { found: false, similarity: -1, actSim: -1 }
    }
  }

  // -------------------------------------------------------------------------
  // 预存队列（pending.jsonl）
  // -------------------------------------------------------------------------

  /**
   * 追加一条训练条目（JSONL）。entry: {text, module?, latent(Float64Array|Array, ≥32)|latent32(数组), t?}。
   * 落盘格式：{t, text(≤200), latent32:[latent 前 32 维保留 2 位小数], module}。
   * 失败只记录错误（void），与 Workspace.appendEpisode 的容错约定一致。
   */
  appendPending(entry) {
    try {
      if (!entry || typeof entry !== 'object') throw new Error('pending entry 必须是对象')
      const text = String(entry.text ?? '').slice(0, MAX_TEXT_LEN)
      if (!text) throw new Error('pending entry 缺少 text')
      let latent32
      if (Array.isArray(entry.latent32) && entry.latent32.length >= LATENT32_DIM) {
        latent32 = entry.latent32.slice(0, LATENT32_DIM).map((v) => Number(Number(v).toFixed(2)))
      } else {
        let src = entry.latent
        if (!src && typeof this.brain?.controller?.hear === 'function') {
          src = this.brain.controller.hear(text, false)?.latent
        }
        if (!src || src.length < LATENT32_DIM) throw new Error('pending entry 缺少 latent/latent32（≥32 维）')
        latent32 = Array.from(src.slice(0, LATENT32_DIM)).map((v) => Number(Number(v).toFixed(2)))
      }
      if (!latent32.every(Number.isFinite)) throw new Error('pending entry 的 latent 含非有限值')
      const line = JSON.stringify({
        t: Number.isFinite(entry.t) ? entry.t : Date.now(),
        text,
        latent32,
        module: MODULES.includes(entry.module) ? entry.module : 'memory',
      })
      const dir = dirname(this.pendingFile)
      if (dir) mkdirSync(dir, { recursive: true })
      appendFileSync(this.pendingFile, line + '\n', 'utf8')
    } catch (error) {
      this._recordError(error)
    }
  }

  /** 读取全部有效条目（坏行跳过，旧在前）。文件不存在/不可读 → []。 */
  readPending() {
    try {
      const raw = readFileSync(this.pendingFile, 'utf8')
      const out = []
      for (const line of raw.split('\n')) {
        const t = line.trim()
        if (!t) continue
        try {
          const obj = JSON.parse(t)
          if (obj !== null && typeof obj === 'object') out.push(obj)
        } catch {
          /* 坏行跳过 */
        }
      }
      return out
    } catch {
      return []
    }
  }

  /**
   * 从队头移除 n 条有效条目（JSON 解析成功的行才算条目；坏行视为垃圾，
   * 重写时一并清理，与 readPending 的"坏行跳过"语义一致）。
   * 重写文件保留其余条目（原子写）。返回实际移除条数。
   */
  removePending(n = 1) {
    const count = Math.max(0, Math.floor(Number(n) || 0))
    if (count === 0) return 0
    let raw
    try {
      raw = readFileSync(this.pendingFile, 'utf8')
    } catch {
      return 0
    }
    const kept = []
    let removed = 0
    for (const line of raw.split('\n')) {
      const t = line.trim()
      if (!t) continue
      let valid = false
      try {
        const obj = JSON.parse(t)
        valid = obj !== null && typeof obj === 'object'
      } catch {
        valid = false
      }
      if (!valid) continue // 坏行：清理
      if (removed < count) {
        removed++
        continue
      }
      kept.push(t)
    }
    try {
      atomicWriteFile(this.pendingFile, kept.length > 0 ? kept.join('\n') + '\n' : '')
    } catch (error) {
      this._recordError(error)
      return 0
    }
    return removed
  }

  // -------------------------------------------------------------------------
  // 训练循环
  // -------------------------------------------------------------------------

  /**
   * 限流训练：每 tick 最多处理 max 条，保证心跳实时。
   * 对每条：
   *   a) brain.controller.hear(text, false) 重编码 → fullLatent(128)
   *   b) 与条目 latent32 前 32 维比对 cosine（条目 latent32 即该文本当时的编码）
   *   c) 上下文命中判定用 mirrorRecall(fullLatent)：
   *      found===true → observe ×2（加倍）；未命中 → observe ×1
   *   d) 每训一条 removePending(1)
   * 褪黑素进度由 brain 层每小时心跳统一覆写，本方法不负责更新。
   * 每条训练包 try/catch + NaN 检测（训练后 layers[0].W 抽样有限性）；
   * 异常/NaN：记录错误并跳过该条（条目保留在队头，本轮停止继续，
   * 避免 FIFO 错位与在损坏权重上继续训练）。
   * @returns {{trained:number, doubled:number, remaining:number, stopped:boolean}}
   *   remaining = 处理后 pending 文件剩余条数
   */
  processPending(max = 3) {
    const cap = Math.max(0, Math.floor(Number(max) || 0))
    const pending = this.readPending()
    if (this._stopped) {
      return { trained: 0, doubled: 0, remaining: pending.length, stopped: true }
    }
    const stats = { trained: 0, doubled: 0, remaining: pending.length, stopped: false }
    if (cap === 0 || pending.length === 0) return stats
    let remaining = pending.length
    for (let i = 0; i < cap && i < pending.length; i++) {
      try {
        const detail = this._trainOne(pending[i])
        this.removePending(1)
        remaining = this.readPending().length
        stats.trained++
        if (detail.doubled) stats.doubled++
        this.trainedTotal++
        if (detail.doubled) this.doubledTotal++
      } catch (error) {
        this.skippedTotal++
        this._recordError(error)
        break // 队头条目保留：下一轮重试
      }
    }
    stats.remaining = remaining
    return stats
  }

  /** 单条训练包：编码 → 镜像命中判定 → observe（命中 ×2 / 未命中 ×1）→ NaN 检测。 */
  _trainOne(entry) {
    const text = String(entry?.text ?? '')
    if (!text) throw new Error('pending entry 缺少 text')
    const heard = this.brain.controller.hear(text, false)
    const latent = heard?.latent
    if (!latent) throw new Error('controller.hear 未返回 latent')
    let fullLatent = new Float64Array(latent)
    if (fullLatent.length > 128) fullLatent = fullLatent.slice(0, 128)
    if (fullLatent.length !== 128) throw new Error(`controller.hear 返回的 latent 维度异常：${fullLatent.length}`)

    // b) latent32 余弦比对（上下文核对指标；命中判定以 mirrorRecall 为准）
    let cos32 = null
    if (Array.isArray(entry.latent32) && entry.latent32.length >= LATENT32_DIM) {
      const a = fullLatent.subarray(0, LATENT32_DIM)
      const b = new Float64Array(LATENT32_DIM)
      for (let i = 0; i < LATENT32_DIM; i++) b[i] = Number(entry.latent32[i])
      cos32 = cosine(a, b)
    }

    // c) 镜像命中判定 → 加倍/单次
    const mr = this.mirrorRecall(fullLatent)
    const doubled = mr.found === true
    const module = MODULES.includes(entry.module) ? entry.module : 'memory'
    const emotionTarget = this.brain.endocrine.effects().emotionTarget
    const train = (this.brain.mode ?? 'training') === 'training'
    const times = doubled ? 2 : 1
    for (let k = 0; k < times; k++) {
      this.brain.memory.observe(fullLatent, {
        module,
        salience: 0.6, // 与 mirrorRecall 查询条件一致，保证熟悉度（predSim）稳定
        lrScale: 1.2,
        emotionTarget,
        train,
      })
    }

    // NaN 检测：训练后 layers[0].W 抽样有限性（异常 → 跳过该条）
    if (!this._weightsFinite()) {
      throw new Error('NaN 检测失败：brain.memory.net.layers[0].W 出现非有限权重')
    }

    this.lastEntryDetail = {
      text: text.slice(0, 40),
      module,
      cos32: cos32 === null ? null : Number(cos32.toFixed(4)),
      similarity: mr.similarity,
      doubled,
    }
    return { doubled, cos32, similarity: mr.similarity }
  }

  /** 训练后权重有限性抽样检测（layers[0].W，步进抽样 ≤128 点；结构不可用视为通过）。 */
  _weightsFinite() {
    try {
      const W = this.brain.memory?.net?.layers?.[0]?.W?.data
      if (!W || W.length === 0) return true
      const stride = Math.max(1, Math.floor(W.length / 128))
      for (let i = 0; i < W.length; i += stride) {
        if (!Number.isFinite(W[i])) return false
      }
      return true
    } catch {
      return true
    }
  }

  // -------------------------------------------------------------------------
  // 生命周期
  // -------------------------------------------------------------------------

  /** 停止训练（苏醒前调用）。 */
  stop() {
    this._stopped = true
  }

  /** 恢复训练。 */
  resume() {
    this._stopped = false
  }

  // -------------------------------------------------------------------------
  // 内部
  // -------------------------------------------------------------------------

  _recordError(error) {
    const msg = error instanceof Error ? error.message : String(error)
    this.lastError = msg
    this.recentErrors.push({ t: Date.now(), msg })
    if (this.recentErrors.length > 20) this.recentErrors.shift()
  }
}
