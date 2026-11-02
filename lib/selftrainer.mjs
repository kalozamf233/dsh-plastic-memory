// ============================================================================
// 自我训练器（SelfTrainer）
// 类似训练控制器，但专注于训练「对话能力」：
//   1. 休眠时读「自我提示词预存文件」（self-pending.jsonl，清醒期与提示词
//      预存文件同步写入，另含他说出的话——带小标记）
//   2. 回忆对话神经网络过滤出对话内容，写入「自我训练预存文件」
//      （self-training.jsonl）；遇到带小标记的语句 → 存两份
//   3. 以自我训练预存文件内容训练「自我记忆神经网络」；
//      带小标记的语句训练两遍
// 全部原子写、坏行跳过、每 tick 限流。
// ============================================================================

import { readFileSync, writeFileSync, renameSync, existsSync } from 'node:fs'
import { RecallNet, hasMark, stripMark } from './recallnet.mjs'
import { fnv1a } from './nn/encoder.mjs'

export const MAX_SELF_PENDING = 100 // 自我提示词预存上限
export const SELF_TRAIN_PER_TICK = 3 // 每 tick 处理/训练条数

export class SelfTrainer {
  constructor(brain, opts = {}) {
    this.brain = brain
    this.pendingFile = opts.pendingFile ?? null
    this.trainingFile = opts.trainingFile ?? null
    this.stopped = false
    this.filteredTotal = 0 // 累计过滤出的对话条数（含加倍）
    this.markedTotal = 0 // 累计遇到标记句次数
    this.trainedTotal = 0
    this.doubledTrainTotal = 0
    this.lastError = null
  }

  get paths() {
    const p = this.brain?.workspace?.paths ?? {}
    return {
      pending: this.pendingFile ?? p.selfPending ?? null,
      training: this.trainingFile ?? p.selfTraining ?? null,
    }
  }

  /** 原子写 JSONL（保留行数 ≤ max）。 */
  _atomicWriteLines(file, lines, max = 1000) {
    if (!file) throw new Error('self trainer 文件路径未配置')
    const kept = lines.slice(-max)
    const tmp = file + '.tmp'
    writeFileSync(tmp, kept.join('\n') + (kept.length ? '\n' : ''), 'utf8')
    renameSync(tmp, file)
  }

  _readLines(file) {
    if (!file || !existsSync(file)) return []
    try {
      return readFileSync(file, 'utf8').trim().split('\n').filter(Boolean)
    } catch {
      return []
    }
  }

  /** 读自我提示词预存（解析 JSON，坏行跳过）。 */
  readSelfPending() {
    const out = []
    for (const line of this._readLines(this.paths.pending)) {
      try {
        const item = JSON.parse(line)
        if (item && typeof item.text === 'string') out.push(item)
      } catch {
        /* 坏行跳过 */
      }
    }
    return out
  }

  /** 读自我训练预存。 */
  readSelfTraining() {
    const out = []
    for (const line of this._readLines(this.paths.training)) {
      try {
        const item = JSON.parse(line)
        if (item && typeof item.text === 'string') out.push(item)
      } catch {
        /* 坏行跳过 */
      }
    }
    return out
  }

  /** 清醒期追加：提示词预存同步存入；他说的话带小标记也存这里。 */
  appendSelfPending(item) {
    if (!this.paths.pending) return
    try {
      const lines = this._readLines(this.paths.pending)
      lines.push(JSON.stringify(item))
      this._atomicWriteLines(this.paths.pending, lines, MAX_SELF_PENDING)
    } catch (error) {
      this.lastError = String(error?.message ?? error)
    }
  }

  /**
   * 休眠第一步：回忆对话神经网络过滤自我提示词预存。
   * 对话内容 → 自我训练预存；带小标记 → 存两份；非对话 → 丢弃。
   */
  processSelf(max = SELF_TRAIN_PER_TICK) {
    if (this.stopped) return { filtered: 0, marked: 0, remaining: 0, stopped: true }
    const recall = this.brain?.recallNet
    const items = this.readSelfPending()
    const take = items.slice(0, max)
    const rest = items.slice(max)
    const outLines = this._readLines(this.paths.training)
    let filtered = 0
    let marked = 0
    for (const item of take) {
      const markedFlag = hasMark(item.text)
      const judge = recall ? recall.judge(item.text) : null
      // 规则兜底：source=self 或问句/对话特征 = 对话内容
      const isDialogue = item.source === 'self' || (judge?.isDialogue ?? false)
      if (!isDialogue) continue
      const entry = JSON.stringify({ t: item.t, text: item.text, source: item.source, marked: markedFlag, module: item.module ?? 'audio' })
      outLines.push(entry)
      filtered++
      if (markedFlag) {
        outLines.push(entry) // 标记句存两份
        marked++
        if (recall) recall.doubled++
      }
      // 回忆网络在线学习（以本次判定为标签）
      if (recall) recall.train(item.text, { isDialogue })
      if (recall) recall.filters++
    }
    // 剩余写回自我提示词预存
    this._atomicWriteLines(this.paths.pending, rest.map((x) => JSON.stringify(x)), MAX_SELF_PENDING)
    this._atomicWriteLines(this.paths.training, outLines)
    this.filteredTotal += filtered
    this.markedTotal += marked
    return { filtered, marked, remaining: rest.length, stopped: false }
  }

  /**
   * 休眠第二步：以自我训练预存训练自我记忆神经网络。
   * 带小标记的语句训练两遍。
   */
  trainSelf(max = SELF_TRAIN_PER_TICK) {
    if (this.stopped) return { trained: 0, doubled: 0, remaining: 0, stopped: true }
    const selfNet = this.brain?.selfNet
    if (!selfNet || this.brain.mode !== 'training') {
      return { trained: 0, doubled: 0, remaining: this.readSelfTraining().length, stopped: false }
    }
    const items = this.readSelfTraining()
    const take = items.slice(0, max)
    const rest = items.slice(max)
    let trained = 0
    let doubled = 0
    for (const item of take) {
      try {
        const markedFlag = !!(item.marked ?? hasMark(item.text))
        const text = stripMark(item.text)
        const { latent } = this.brain.controller.hear(text, false)
        const passes = markedFlag ? 2 : 1 // 标记句训练两遍
        for (let i = 0; i < passes; i++) {
          selfNet.observe(latent, {
            module: item.module === 'visual' ? 'visual' : 'audio',
            salience: 0.6,
            lrScale: 1.2,
            emotionTarget: null,
            train: true,
          })
        }
        trained++
        if (markedFlag) doubled++
      } catch (error) {
        this.lastError = String(error?.message ?? error)
        break // 保留队头，下轮重试
      }
    }
    const done = items.length - rest.length
    this._atomicWriteLines(this.paths.training, items.slice(done).map((x) => JSON.stringify(x)))
    this.trainedTotal += trained
    this.doubledTrainTotal += doubled
    return { trained, doubled, remaining: items.length - done, stopped: false }
  }

  stop() {
    this.stopped = true
  }

  resume() {
    this.stopped = false
  }

  status() {
    return {
      pendingCount: this.readSelfPending().length,
      trainingCount: this.readSelfTraining().length,
      filteredTotal: this.filteredTotal,
      markedTotal: this.markedTotal,
      trainedTotal: this.trainedTotal,
      doubledTrainTotal: this.doubledTrainTotal,
      stopped: this.stopped,
    }
  }
}
