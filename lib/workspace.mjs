// ============================================================================
// dsh-plastic-memory —— 工作区管理（Workspace）
// ----------------------------------------------------------------------------
// 用户指定的文件夹 = 数字生命的工作区。布局：
//
//   plastic-memory.config.json   工作区配置（名称/种子/创建时间/版本）
//   controller.brain.json        操控神经网络本体（含编码器/策略网络/内分泌状态）
//   memory.brain.json            记忆神经网络权重（训练态发布：会话可影响）
//   memory.frozen.json           固定权重快照（freeze 时生成）
//   memory.genome.json           出厂"基因"：初始随机权重（视频01 自私基因的
//                                「先天」：永不被训练修改，reset 可返祖）
//   episodes.jsonl               情节日志（辅助索引，非记忆本体）
//   senses/                      感知文件夹（外部世界信号入口）
//     .processed/                已处理感知归档
//   logs/brain.log               运行日志
//
// 所有写入均原子（tmp + rename），崩溃安全。
// ============================================================================

import { mkdirSync, readFileSync, writeFileSync, renameSync, existsSync, appendFileSync, readdirSync, copyFileSync, unlinkSync } from 'node:fs'
import { join, resolve, basename, dirname } from 'node:path'

export const CONFIG_FILE = 'plastic-memory.config.json'
export const CONTROLLER_FILE = 'controller.brain.json'
export const MEMORY_FILE = 'memory.brain.json'
export const FROZEN_FILE = 'memory.frozen.json'
export const GENOME_FILE = 'memory.genome.json'
export const MIRROR_FILE = 'mirror.brain.json'
export const FLAW_FILE = 'flaw.brain.json'
export const ANALYSIS_NET_FILE = 'analysis.brain.json'
export const PENDING_FILE = 'pending-training.jsonl'
export const ANALYSIS_DIR = 'analysis'
export const EPISODES_FILE = 'episodes.jsonl'
export const SENSES_DIR = 'senses'
export const PROCESSED_DIR = '.processed'
export const LOGS_DIR = 'logs'

export class Workspace {
  constructor(folder) {
    this.folder = folder
  }

  get paths() {
    const f = this.folder
    return {
      config: join(f, CONFIG_FILE),
      controller: join(f, CONTROLLER_FILE),
      memory: join(f, MEMORY_FILE),
      frozen: join(f, FROZEN_FILE),
      genome: join(f, GENOME_FILE),
      mirror: join(f, MIRROR_FILE),
      flaw: join(f, FLAW_FILE),
      analysisNet: join(f, ANALYSIS_NET_FILE),
      pending: join(f, PENDING_FILE),
      analysisDir: join(f, ANALYSIS_DIR),
      episodes: join(f, EPISODES_FILE),
      senses: join(f, SENSES_DIR),
      processed: join(f, SENSES_DIR, PROCESSED_DIR),
      logs: join(f, LOGS_DIR),
    }
  }

  /** 初始化：创建目录结构（不覆盖已有数据）。 */
  ensureDirs() {
    const p = this.paths
    mkdirSync(this.folder, { recursive: true })
    mkdirSync(p.senses, { recursive: true })
    mkdirSync(p.processed, { recursive: true })
    mkdirSync(p.logs, { recursive: true })
    mkdirSync(p.analysisDir, { recursive: true })
  }

  readConfig() {
    try {
      return JSON.parse(readFileSync(this.paths.config, 'utf8'))
    } catch {
      return null
    }
  }

  writeConfig(cfg) {
    this.ensureDirs()
    atomicWrite(this.paths.config, JSON.stringify(cfg, null, 2))
  }

  exists() {
    return existsSync(this.paths.config)
  }

  /** 判断冻结态：frozen 文件存在且不是 thawed 标记。 */
  isFrozen() {
    const p = this.paths.frozen
    if (!existsSync(p)) return false
    try {
      const data = JSON.parse(readFileSync(p, 'utf8'))
      // 旧版 thaw 写入 {thawedAt} 标记（兼容：视为已解冻）
      if (data && typeof data === 'object' && data.thawedAt !== undefined && data.formatVersion === undefined) {
        return false
      }
      return true
    } catch {
      return true
    }
  }

  // -------------------------------------------------------------------------
  // 权重读写
  // -------------------------------------------------------------------------

  saveMemory(memoryNet) {
    this.ensureDirs()
    atomicWrite(this.paths.memory, JSON.stringify(memoryNet.toJSON()))
  }

  loadMemory() {
    return JSON.parse(readFileSync(this.paths.memory, 'utf8'))
  }

  /** 保存外部扩容网络权重到用户指定路径（原子写）。 */
  saveExtensionFile(path, memoryNetJson) {
    const dir = dirname(path)
    if (dir) mkdirSync(dir, { recursive: true })
    atomicWrite(path, JSON.stringify(memoryNetJson))
  }

  saveController(controllerNet, endocrine, extra = {}) {
    this.ensureDirs()
    atomicWrite(
      this.paths.controller,
      JSON.stringify({ controller: controllerNet.toJSON(), endocrine: endocrine.toJSON(), ...extra }),
    )
  }

  loadController() {
    return JSON.parse(readFileSync(this.paths.controller, 'utf8'))
  }

  /** 保存出厂基因（首次初始化时调用一次）。 */
  saveGenome(memoryJson) {
    this.ensureDirs()
    atomicWrite(this.paths.genome, JSON.stringify(memoryJson))
  }

  loadGenome() {
    try {
      return JSON.parse(readFileSync(this.paths.genome, 'utf8'))
    } catch {
      return null
    }
  }

  /**
   * 固定权重（训练完毕 → 发布冻结态）：
   * 把当前记忆网络权重快照写入 memory.frozen.json。
   */
  freeze(memoryNet) {
    this.ensureDirs()
    const json = { ...memoryNet.toJSON(), frozenAt: new Date().toISOString() }
    atomicWrite(this.paths.frozen, JSON.stringify(json))
    return json.frozenAt
  }

  /** 解除冻结：删除冻结快照，恢复训练态。 */
  thaw() {
    const p = this.paths.frozen
    try {
      unlinkSync(p)
    } catch {
      // 文件不存在或删除失败：写入 thawed 标记兜底
      atomicWrite(p, JSON.stringify({ thawedAt: new Date().toISOString() }))
    }
  }

  // -------------------------------------------------------------------------
  // 情节日志（辅助索引：记录感知摘要/情绪强度/时间，供 read/persona 检索）
  // -------------------------------------------------------------------------

  appendEpisode(episode) {
    this.ensureDirs()
    try {
      appendFileSync(this.paths.episodes, JSON.stringify(episode) + '\n', 'utf8')
    } catch {
      /* 索引失败不影响大脑 */
    }
    // 收缩：保留最近 500 条
    try {
      const data = readFileSync(this.paths.episodes, 'utf8')
      const lines = data.trim().split('\n').filter(Boolean)
      if (lines.length > 500) {
        atomicWrite(this.paths.episodes, lines.slice(-500).join('\n') + '\n')
      }
    } catch {
      /* ignore */
    }
  }

  readEpisodes(limit = 100) {
    try {
      const data = readFileSync(this.paths.episodes, 'utf8')
      return data
        .trim()
        .split('\n')
        .filter(Boolean)
        .slice(-limit)
        .map((line) => {
          try {
            return JSON.parse(line)
          } catch {
            return null
          }
        })
        .filter(Boolean)
    } catch {
      return []
    }
  }

  // -------------------------------------------------------------------------
  // 感知文件夹扫描
  // -------------------------------------------------------------------------

  /** 列出待处理感知文件（排除 .processed 与临时文件）。 */
  listPendingSenses() {
    const p = this.paths.senses
    let entries = []
    try {
      entries = readdirSync(p)
    } catch {
      return []
    }
    return entries
      .filter((name) => name !== PROCESSED_DIR && !name.startsWith('.') && !name.endsWith('.tmp'))
      .sort()
      .map((name) => join(p, name))
  }

  /** 读取感知文件内容（限制 256KB）。 */
  readSenseFile(file) {
    const data = readFileSync(file, 'utf8')
    if (data.length > 256 * 1024) return data.slice(0, 256 * 1024)
    return data
  }

  /** 归档已处理感知文件。 */
  archiveSense(file, meta = {}) {
    const p = this.paths.processed
    const name = `${Date.now()}-${basename(file)}`
    const dst = join(p, name)
    try {
      renameSync(file, dst)
    } catch {
      // 归档失败时复制+删除
      try {
        copyFileSync(file, dst)
        unlinkSync(file)
      } catch {
        /* 留待下轮 */
      }
    }
    return dst
  }

  // -------------------------------------------------------------------------
  // 日志
  // -------------------------------------------------------------------------

  log(message) {
    try {
      const line = `[${new Date().toISOString()}] ${message}\n`
      appendFileSync(join(this.paths.logs, 'brain.log'), line, 'utf8')
    } catch {
      /* ignore */
    }
  }
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

export function atomicWrite(file, content) {
  const tmp = `${file}.tmp`
  writeFileSync(tmp, content, 'utf8')
  renameSync(tmp, file)
}

/** 规范化用户输入的文件夹路径（绝对化，拒绝空）。 */
export function normalizeFolder(input) {
  if (typeof input !== 'string' || input.trim() === '') return null
  const p = resolve(input.trim())
  return p
}
