// ============================================================================
// dsh-plastic-memory —— 双网络协调器（PlasticBrain）
// ----------------------------------------------------------------------------
// 组合：操控神经网络（身体）→ 记忆神经网络（灵魂）→ 内分泌（体液）。
// 身体隐喻的分工（用户设定）：
//   现有 AI 模型/agent  ＝ 大脑与脊椎（整合信息、决策）
//   操控神经网络        ＝ 身体（感知、行动、为灵魂获取外界反馈）
//   记忆神经网络        ＝ 灵魂（人格化数据与人格化反馈）
//
// 每秒心跳（tick）：
//   1. 扫描感知文件夹 senses/：新文件 → 操控网络"听"（编码+生理反应）
//      → 格式化数据注入记忆网络（预测加工学习）→ 归档
//   2. 内分泌衰减
//   3. 周期（10s）操控网络决策：增删记忆网络神经元/参数 + 元学习
//   4. 周期（30s）权重自动保存
//
// API 能力：听 hear / 说 speak / 读 read / 写 write / 冻结 freeze / 解冻 thaw /
//          返祖 reset / 保存 save / 人格导出 persona / 状态 status
// ============================================================================

import { MemoryNet, interpretPersona, interpretEmotion } from './nn/memorynet.mjs'
import { ControllerNet } from './nn/controllernet.mjs'
import { EndocrineSystem } from './nn/endocrine.mjs'
import { Workspace, normalizeFolder } from './workspace.mjs'
import { AnalysisNet, AnalysisStore } from './nn/analysisnet.mjs'
import { DefectNet } from './nn/flawnet.mjs'
import { TrainingController } from './trainer.mjs'
import { BodyNet, WorldEncoder, desireFromEndocrine, MOVE_DIRS, INTERACT_NAMES } from './bodynet.mjs'
import { Body, BIRTH_POS } from './body.mjs'
import { generateGene, geneSummary, geneHash } from './gene.mjs'
import { RecallNet, SELF_MARK, stripMark, hasMark } from './recallnet.mjs'
import { SelfTrainer } from './selftrainer.mjs'
import { cosine } from './nn/core.mjs'
import { fnv1a } from './nn/encoder.mjs'
import { readFileSync, existsSync, writeFileSync } from 'node:fs'

export class PlasticBrain {
  constructor() {
    this.workspace = null
    this.memory = null
    this.controller = null
    this.endocrine = null
    this.analysisNet = null // 分析型神经网络
    this.analysisStore = null // 分析存储器
    this.flaw = null // 缺陷人格神经网络（对话修饰层）
    this.trainer = null // 训练控制器（镜像/同步/休眠训练）
    this.bodyNet = null // 身体数据网络（对外感知输入与身体行为输出）
    this.gene = null // 100 位永久基因
    this.body = null // 身体状态（世界坐标/能量/痛感/饱腹）
    this.selfNet = null // 自我记忆神经网络（对话能力专用，与记忆/缺陷人格/操控网络相连）
    this.recallNet = null // 回忆对话神经网络（过滤自我提示词预存中的对话内容）
    this.selfTrainer = null // 自我训练器（专注对话能力训练）
    this.mode = 'training' // training | frozen
    this.scanInterval = null
    this.saveInterval = null
    this.worldInterval = null // 世界交互心跳
    this.running = false
    this.scannedFiles = 0
    this.perceptions = 0
    this.lastError = null
    this.bootedAt = null
    this.extensions = new Map() // 外部扩容网络：path → MemoryNet
    this.worldOnline = false // 世界服务是否可达
    this.worldLastEventTs = 0 // 世界反馈事件游标
    // 休眠状态：预存文件驱动（褪黑素=预存容量进度条）
    this.asleep = false
    this.wakefulness = 1 // 0..1 清醒度
    this.sleepCount = 0
    this.sleepStartedAt = null
    this.lastPerceptionAt = Date.now()
    this.lastWakeAt = null // 最近苏醒时间（持久化，供 UI 检测弹提示框）
  }

  get initialized() {
    return this.workspace !== null && this.memory !== null
  }

  // -------------------------------------------------------------------------
  // 初始化 / 恢复
  // -------------------------------------------------------------------------

  /** 用户指定文件夹初始化（必须由用户指定：API 传 workspace 路径）。 */
  init(folderInput, opts = {}) {
    const folder = normalizeFolder(folderInput)
    if (!folder) throw new Error('工作区路径无效')
    const ws = new Workspace(folder)
    if (ws.exists()) {
      // 已有工作区 → 恢复
      return this.load(folderInput)
    }
    const seed = opts.seed ?? Math.floor(Math.random() * 1e9) >>> 0
    const name = opts.name ?? '数字生命'
    ws.ensureDirs()
    ws.writeConfig({
      version: 1,
      name,
      seed,
      createdAt: new Date().toISOString(),
      format: 'dsh-plastic-memory',
    })

    this.workspace = ws
    this.memory = new MemoryNet({ seed, baseLr: opts.baseLr ?? 0.008 })
    this.controller = new ControllerNet({ seed: seed ^ 0x9e3779b9, baseLr: opts.baseLr ?? 0.008 })
    this.endocrine = new EndocrineSystem()
    this.extensions = new Map()
    // 大脑架构 V2：分析网络 / 缺陷人格网络 / 训练控制器（镜像+同步）
    this.analysisNet = new AnalysisNet({ seed: seed ^ 0x5f3759df })
    this.analysisStore = new AnalysisStore(ws.paths.analysisDir)
    this.flaw = new DefectNet({ seed: seed ^ 0x51ab1e2f })
    this.trainer = new TrainingController(this)
    this.lastWakeAt = null
    // 大脑架构 V3：基因（100 位永久）+ 身体（出生自带世界坐标）+ 身体数据网络
    this.gene = generateGene()
    this.body = new Body(seed)
    this.bodyNet = new BodyNet({ seed: seed ^ 0x6d2b79f5 })
    this.worldOnline = false
    this.worldLastEventTs = 0
    // 大脑架构 V4：自我记忆网络（MemoryNet 实例，对话能力专用）+ 回忆网络 + 自我训练器
    this.selfNet = new MemoryNet({ seed: seed ^ 0x7c3a1b9d, baseLr: opts.baseLr ?? 0.008 })
    this.recallNet = new RecallNet({ seed: seed ^ 0x3e9a5c71 })
    this.selfTrainer = new SelfTrainer(this)

    // 出厂基因（视频01：自私的基因 —— 先天权重，永不被训练修改）
    ws.saveGenome(this.memory.toJSON())
    this.save()

    ws.log(`brain initialized: folder=${folder} seed=${seed} name=${name}`)
    this.bootedAt = new Date().toISOString()
    this.start()
    // V3：出生注册世界（出生自带世界中心坐标）
    this.registerInWorld().catch(() => {})
    return this.status()
  }

  /** 从已有工作区恢复。 */
  load(folderInput) {
    const folder = normalizeFolder(folderInput)
    if (!folder) throw new Error('工作区路径无效')
    const ws = new Workspace(folder)
    if (!ws.exists()) throw new Error('工作区未初始化，请先调用 init 指定文件夹')

    this.stop()
    this.workspace = ws
    this.memory = MemoryNet.fromJSON(ws.loadMemory())
    const ctrl = ws.loadController()
    this.controller = ControllerNet.fromJSON(ctrl.controller)
    this.endocrine = new EndocrineSystem(ctrl.endocrine)
    this.mode = ws.isFrozen() ? 'frozen' : 'training'
    this.extensions = new Map()
    // 大脑架构 V2 网络：恢复或新建（缺失容错）
    this.analysisNet = this._loadOrCreate(ws.paths.analysisNet, AnalysisNet, { seed: (ctrl.controller?.seed ?? 7) ^ 0x5f3759df })
    this.analysisStore = new AnalysisStore(ws.paths.analysisDir)
    this.flaw = this._loadOrCreate(ws.paths.flaw, DefectNet, { seed: (ctrl.controller?.seed ?? 7) ^ 0x51ab1e2f })
    this.flaw.frozen = this.mode === 'frozen' // 冻结态联动
    this.trainer = new TrainingController(this)
    this.lastWakeAt = ctrl.lastWakeAt ?? null
    // 大脑架构 V3：基因（永久不可变，文件缺失才生成）/ 身体 / 身体数据网络
    this.gene = this._loadOrCreateGene(ws)
    this.body = this._loadOrCreateBody(ws, ctrl.controller?.seed ?? 7)
    this.bodyNet = this._loadOrCreate(ws.paths.bodyNet, BodyNet, { seed: (ctrl.controller?.seed ?? 7) ^ 0x6d2b79f5 })
    this.worldOnline = false
    this.worldLastEventTs = 0
    // 大脑架构 V4：自我记忆网络 / 回忆对话网络 / 自我训练器
    this.selfNet = this._loadOrCreate(ws.paths.selfNet, MemoryNet, { seed: (ctrl.controller?.seed ?? 7) ^ 0x7c3a1b9d, baseLr: 0.008 })
    this.recallNet = this._loadOrCreate(ws.paths.recallNet, RecallNet, { seed: (ctrl.controller?.seed ?? 7) ^ 0x3e9a5c71 })
    this.selfTrainer = new SelfTrainer(this)
    // 恢复扩展网络挂载（失败容错：单个扩展损坏不影响主脑）
    const cfg = ws.readConfig() ?? {}
    for (const path of Array.isArray(cfg.extensions) ? cfg.extensions : []) {
      try {
        this._mountExtensionInternal(path)
      } catch (error) {
        ws.log(`extension mount FAILED: ${path} :: ${error?.message ?? error}`)
      }
    }
    this.bootedAt = new Date().toISOString()
    ws.log(`brain loaded: folder=${folder} mode=${this.mode} extensions=${this.extensions.size}`)
    this.start()
    // V3：向世界注册实体（出生/恢复坐标）
    this.registerInWorld().catch(() => {})
    return this.status()
  }

  /** 从文件恢复新网络实例，文件缺失/损坏则新建。 */
  _loadOrCreate(file, Cls, opts) {
    try {
      if (existsSync(file)) {
        return Cls.fromJSON(JSON.parse(readFileSync(file, 'utf8')))
      }
    } catch {
      /* 损坏 → 重建 */
    }
    return new Cls(opts)
  }

  /** 基因：文件存在则永不改变；缺失才生成（出生）。 */
  _loadOrCreateGene(ws) {
    try {
      if (existsSync(ws.paths.gene)) {
        const gene = String(JSON.parse(readFileSync(ws.paths.gene, 'utf8')).gene ?? '')
        if (/^[a-z]{100}$/.test(gene)) return gene
      }
    } catch {
      /* 损坏 → 重新生成 */
    }
    const gene = generateGene()
    try {
      writeFileSync(ws.paths.gene, JSON.stringify({ gene, createdAt: new Date().toISOString() }), 'utf8')
    } catch {
      /* 保存失败不阻塞出生 */
    }
    return gene
  }

  /** 身体：恢复或出生（出生自带世界中心坐标）。 */
  _loadOrCreateBody(ws, seed) {
    try {
      if (existsSync(ws.paths.body)) {
        return Body.fromJSON(JSON.parse(readFileSync(ws.paths.body, 'utf8')))
      }
    } catch {
      /* 损坏 → 出生 */
    }
    return new Body(seed)
  }

  /** 切换工作区（保存当前，加载目标）。 */
  switch(folderInput) {
    if (this.workspace) this.save()
    return this.load(folderInput)
  }

  save() {
    if (!this.initialized) return false
    this.workspace.saveMemory(this.memory)
    this.workspace.saveController(this.controller, this.endocrine, { lastWakeAt: this.lastWakeAt })
    // 大脑架构 V2 网络权重落盘
    if (this.analysisNet) this.workspace.saveExtensionFile(this.workspace.paths.analysisNet, this.analysisNet.toJSON())
    if (this.flaw) this.workspace.saveExtensionFile(this.workspace.paths.flaw, this.flaw.toJSON())
    if (this.analysisStore) {
      try {
        this.analysisStore.flush()
      } catch {
        /* 分析存储落盘失败不影响主脑 */
      }
    }
    // 大脑架构 V3：身体状态 + 身体数据网络（基因不可变，无需保存）
    if (this.body) this.workspace.saveExtensionFile(this.workspace.paths.body, this.body.toJSON())
    if (this.bodyNet) this.workspace.saveExtensionFile(this.workspace.paths.bodyNet, this.bodyNet.toJSON())
    // 大脑架构 V4：自我记忆网络 + 回忆对话网络
    if (this.selfNet) this.workspace.saveExtensionFile(this.workspace.paths.selfNet, this.selfNet.toJSON())
    if (this.recallNet) this.workspace.saveExtensionFile(this.workspace.paths.recallNet, this.recallNet.toJSON())
    // 保存扩展网络（各自独立文件）
    for (const [path, ext] of this.extensions) {
      try {
        this.workspace.saveExtensionFile(path, ext.toJSON())
      } catch {
        /* 单个扩展保存失败不影响主脑 */
      }
    }
    return true
  }

  // -------------------------------------------------------------------------
  // 心跳
  // -------------------------------------------------------------------------

  start() {
    if (!this.initialized || this.running) return
    this.running = true
    // 每秒感知扫描（用户设定：操控神经网络每秒定时扫过感知文件夹）
    this.scanInterval = setInterval(() => {
      try {
        this.scanSenses()
      } catch (error) {
        this.lastError = String(error?.message ?? error)
      }
    }, 1000)
    // 每 30 秒自动保存权重
    this.saveInterval = setInterval(() => {
      try {
        this.save()
      } catch (error) {
        this.lastError = String(error?.message ?? error)
      }
    }, 30000)
    // 世界交互心跳：每秒一次（常驻感知 → 身体数据网络决策 → 行为请求 → 世界反馈）
    this.worldInterval = setInterval(() => {
      try {
        this.worldTick()
      } catch (error) {
        this.lastError = String(error?.message ?? error)
      }
    }, 1000)
    // 让 timer 不阻止进程退出
    this.scanInterval.unref?.()
    this.saveInterval.unref?.()
  }

  stop() {
    this.running = false
    if (this.scanInterval) clearInterval(this.scanInterval)
    if (this.saveInterval) clearInterval(this.saveInterval)
    if (this.worldInterval) clearInterval(this.worldInterval)
    this.worldInterval = null
    this.scanInterval = null
    this.saveInterval = null
  }

  dispose() {
    this.stop()
    if (this.initialized) {
      try {
        this.save()
      } catch {
        /* ignore */
      }
    }
  }

  // -------------------------------------------------------------------------
  // 核心：感知注入（唯一进入记忆网络的口径）
  // -------------------------------------------------------------------------

  /**
   * 感知 → 格式化数据 → 记忆网络。
   * 记忆神经网络永远只接受操控神经网络编码的格式化数据（本方法即唯一入口）。
   * 同时注入所有已挂载的外部扩容网络（扩容网络与主网络共同实时训练）。
   */
  perceive(text, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化：请先指定工作区文件夹')
    const module = ['visual', 'audio', 'touch', 'memory', 'persona'].includes(opts.module)
      ? opts.module
      : 'memory'
    const source = opts.source ?? 'api'

    // 0. 任何感知都算"打扰"：若在休眠则走统一苏醒流程（用户对他说话即唤醒）
    this.lastPerceptionAt = Date.now()
    const wokeFrom = this.asleep ? (source === 'speak' ? 'speak' : 'perception') : null
    if (this.asleep) this._wakeFlow(source === 'speak' ? 'speak' : 'perception')

    // 1. 操控网络"听"：编码 + 编码器在线学习
    const { latent } = this.controller.hear(String(text))
    // 2. 身体生理反应：情绪线索 → 分子释放
    const cues = this.controller.react(String(text), this.endocrine)
    // 3. 当前体液状态 → 调制参数
    const eff = this.endocrine.effects()
    // 3.5 提示词预存：用户对他说的话自动进预存文件（休眠自训练的数据源；上限 50 条）
    if ((source === 'speak' || source === 'api' || source === 'tool') && String(text).trim().length >= 2) {
      try {
        this.trainer?.appendPending({
          t: Date.now(),
          text: String(text).trim().slice(0, 200),
          latent32: Array.from(latent.slice(0, 32)).map((v) => Number(v.toFixed(2))),
          module,
        })
      } catch (error) {
        this.lastError = String(error?.message ?? error)
      }
      // V4：清醒时提示词预存的内容同步存入自我提示词预存文件
      try {
        this.selfTrainer?.appendSelfPending({
          t: Date.now(),
          text: String(text).trim().slice(0, 200),
          source: 'user',
          module,
        })
      } catch (error) {
        this.lastError = String(error?.message ?? error)
      }
    }
    // 3.6 分析型神经网络：对感知做实时分析 → 分析存储器
    let analysis = null
    if (this.analysisNet && this.analysisStore) {
      try {
        analysis = this.analysisNet.analyze(latent, eff)
        this.analysisStore.append({
          t: Date.now(),
          text: String(text).slice(0, 100),
          module,
          vector: Array.from(analysis.vector).map((v) => Number(v.toFixed(3))),
          topicScore: Number(analysis.topicScore.toFixed(3)),
          novelty: Number(analysis.novelty.toFixed(3)),
        })
      } catch (error) {
        this.lastError = String(error?.message ?? error)
      }
    }
    // 4. 注入记忆网络（冻结态只感知不训练）
    const train = this.mode === 'training'
    const result = this.memory.observe(latent, {
      module,
      salience: eff.memorySalience,
      lrScale: eff.learningRateMod * this.controller.lrMod,
      emotionTarget: eff.emotionTarget,
      train,
    })
    // 4.5 注入外部扩容网络（独立训练，不影响主网络）
    let extensionsTrained = 0
    for (const [path, ext] of this.extensions) {
      try {
        ext.observe(latent, {
          module,
          salience: eff.memorySalience,
          lrScale: eff.learningRateMod,
          emotionTarget: eff.emotionTarget,
          train,
        })
        extensionsTrained++
      } catch (error) {
        this.lastError = `extension ${path}: ${error?.message ?? error}`
      }
    }
    // 5. 情节日志（辅助索引）
    const textDigest = String(text).slice(0, 100)
    this.workspace.appendEpisode({
      t: Date.now(),
      source,
      module,
      text: textDigest,
      kind: opts.kind ?? null,
      cues,
      salience: Number(eff.memorySalience.toFixed(3)),
      emotion: {
        valence: Number(result.emo[0].toFixed(3)),
        arousal: Number(result.emo[1].toFixed(3)),
        joy: Number(result.emo[2].toFixed(3)),
        sadness: Number(result.emo[3].toFixed(3)),
        fear: Number(result.emo[5].toFixed(3)),
        anxiety: Number(result.emo[9].toFixed(3)),
      },
      latent32: Array.from(latent.slice(0, 32)).map((v) => Number(v.toFixed(2))),
      // 对话语料字段：问句对应的参考回答（训练对话能力用）
      ...(opts.reply !== undefined ? { reply: String(opts.reply).slice(0, 120), moodTag: opts.moodTag ?? null } : {}),
    })
    this.perceptions++
    return {
      module,
      train,
      cues,
      predError: Number(result.predError.toFixed(5)),
      salience: Number(eff.memorySalience.toFixed(3)),
      emotion: interpretEmotion(result.emo),
      extensionsTrained,
      wokeFrom,
      analysis,
    }
  }

  // -------------------------------------------------------------------------
  // 休眠（预存文件驱动：褪黑素=预存容量进度条）+ 统一苏醒状态机
  // -------------------------------------------------------------------------

  /** 预存文件当前条目数（从文件统计）。 */
  get pendingCount() {
    try {
      return this.trainer ? this.trainer.readPending().length : 0
    } catch {
      return 0
    }
  }

  /** 褪黑素进度条公式：预存容量 → 褪黑素。 */
  _melatoninFromPending() {
    return 0.12 + Math.min(0.85, this.pendingCount * 0.02)
  }

  /** 入睡（自主或手动）。入睡即开始休眠自训练（由心跳驱动）。 */
  _sleep() {
    if (this.asleep) return false
    this.asleep = true
    this.sleepStartedAt = Date.now()
    this.sleepCount++
    this.endocrine.release('gaba', 0.4)
    this.trainer?.resume()
    this.workspace?.log(`brain fell asleep (#${this.sleepCount}) pending=${this.pendingCount}`)
    return true
  }

  /** 低层唤醒（仅翻转状态；外部必须走 _wakeFlow）。 */
  _wake(source = 'perception') {
    if (!this.asleep) return false
    this.asleep = false
    this.wakefulness = 1
    this.endocrine.set('melatonin', 0.12) // 褪黑素骤降
    this.endocrine.release('adrenaline', 0.25) // 惊醒
    this.lastWakeAt = new Date().toISOString()
    this.workspace?.log(`brain woke up (source=${source})`)
    return true
  }

  /**
   * 统一苏醒流程（所有唤醒路径走这里，顺序不可变）：
   * ①训练器停训（处理中条目原子保留在预存文件=剩余上下文已回传）
   * ②同步器把记忆网络权重同步到镜像网络
   * ③真正苏醒（自然醒额外释放正面情绪激素）
   */
  _wakeFlow(source = 'perception') {
    if (!this.asleep) return { woke: false }
    // ①停训（剩余上下文自然保留在预存文件）
    try {
      this.trainer?.stop()
    } catch (error) {
      this.lastError = String(error?.message ?? error)
    }
    // ②同步镜像（失败不阻塞苏醒，记录错误）
    let synced = false
    try {
      synced = this.trainer ? this.trainer.syncMirror() : false
    } catch (error) {
      this.lastError = `syncMirror: ${error?.message ?? error}`
    }
    // ③真正苏醒
    const woke = this._wake(source)
    // ④正面情绪激素增值：自然醒=大剂量；交互唤醒=小剂量
    if (source === 'rested' || source === 'pending-drained') {
      this.endocrine.releaseMany({ dopamine: 0.6, endorphin: 0.6, serotonin: 0.5 })
    } else {
      this.endocrine.releaseMany({ dopamine: 0.2, endorphin: 0.2 })
    }
    return { woke, synced, lastWakeAt: this.lastWakeAt }
  }

  /** 每秒心跳的睡眠动力学（pending 驱动版）。
   *  入睡条件：预存 ≥10 条且清醒度 ≤0.5，或预存 ≥30 条（无视清醒度）。
   *  睡眠中：心跳驱动训练控制器逐条消耗预存；预存归零 → 自然苏醒。 */
  _updateSleep(dtSeconds = 1) {
    if (!this.initialized) return
    const pending = this.pendingCount
    if (this.asleep) {
      // 睡眠中：训练控制器消耗预存（每 tick 最多 3 条）
      if (this.mode === 'training') {
        try {
          this.trainer?.processPending(3)
        } catch (error) {
          this.lastError = `processPending: ${error?.message ?? error}`
        }
        // V4：自我训练器——回忆网络过滤自我预存 → 对话内容进自我训练预存（标记句两份）
        //      → 以自我训练预存训练自我记忆网络（标记句两遍）
        try {
          this.selfTrainer?.processSelf(3)
          this.selfTrainer?.trainSelf(3)
        } catch (error) {
          this.lastError = `selfTrain: ${error?.message ?? error}`
        }
      }
      // 预存归零 → 自然苏醒
      if (this.pendingCount === 0) {
        this._wakeFlow('pending-drained')
      }
      return
    }
    // 清醒期：褪黑素进度条 = 预存容量
    this.endocrine.set('melatonin', this._melatoninFromPending())
    const idleSeconds = (Date.now() - this.lastPerceptionAt) / 1000
    if (idleSeconds > 60) {
      this.wakefulness = Math.max(0, this.wakefulness - dtSeconds / 240)
    } else {
      this.wakefulness = Math.min(1, this.wakefulness + dtSeconds / 60)
    }
    if (pending >= 30 || (pending >= 10 && this.wakefulness <= 0.5)) {
      this._sleep()
    }
  }

  /** 手动入睡（API/工具）。 */
  sleepNow() {
    if (!this.initialized) throw new Error('大脑未初始化')
    const did = this._sleep()
    return { asleep: true, slept: did }
  }

  /** 手动唤醒（API/工具，走统一苏醒流程）。 */
  wakeNow() {
    if (!this.initialized) throw new Error('大脑未初始化')
    const result = this._wakeFlow('manual')
    return { asleep: false, woke: result.woke, synced: result.synced }
  }

  // -------------------------------------------------------------------------
  // 世界交互服务（V3）：空间坐标 + 身体数据网络 + 欲望驱动行为闭环
  // -------------------------------------------------------------------------

  /** 世界服务地址。 */
  get worldUrl() {
    return process.env.PLASTIC_WORLD ?? 'http://127.0.0.1:9631'
  }

  /** 世界 HTTP 请求（超时 2 秒）。 */
  async _worldFetch(path, init = {}) {
    const res = await fetch(`${this.worldUrl}${path}`, {
      ...init,
      headers: { 'content-type': 'application/json', ...(init.headers ?? {}) },
      signal: AbortSignal.timeout(2000),
    })
    const text = await res.text()
    let data
    try {
      data = JSON.parse(text)
    } catch {
      data = { ok: false, error: text.slice(0, 120) }
    }
    return { status: res.status, data }
  }

  /** 向世界注册/恢复实体（出生自带坐标）。 */
  async registerInWorld() {
    if (!this.body) return false
    try {
      const r = await this._worldFetch('/register', {
        method: 'POST',
        body: JSON.stringify({ entity: 'digital-life', pos: this.body.pos }),
      })
      this.worldOnline = r.data?.ok === true || r.status === 200
      if (r.data?.pos) this.body.applyPos(r.data.pos)
    } catch {
      this.worldOnline = false
    }
    return this.worldOnline
  }

  /** 常驻交互：把坐标/方向/范围连同感官请求发给世界，拿回六感反馈。 */
  async worldSense(senses = ['sight', 'hearing', 'smell', 'touch'], range = 5) {
    if (!this.body) return { sights: [], sounds: [], smells: [], touches: [] }
    try {
      const r = await this._worldFetch('/sense', {
        method: 'POST',
        body: JSON.stringify({
          entity: 'digital-life',
          pos: this.body.pos,
          dir: this.body.dir,
          range,
          senses,
        }),
      })
      this.worldOnline = true
      return r.data ?? { sights: [], sounds: [], smells: [], touches: [] }
    } catch {
      this.worldOnline = false
      return { sights: [], sounds: [], smells: [], touches: [] }
    }
  }

  /** 行为交互：向世界发送行为请求（移动/跳跃/拿取/吃），世界校验并返回结果。 */
  async worldAct(action, opts = {}) {
    if (!this.body) return { ok: false, error: '身体未就绪' }
    try {
      const r = await this._worldFetch('/act', {
        method: 'POST',
        body: JSON.stringify({
          entity: 'digital-life',
          pos: this.body.pos,
          action,
          dir: opts.dir ?? this.body.dir,
          target: opts.target ?? null,
        }),
      })
      this.worldOnline = true
      return r.data ?? { ok: false }
    } catch {
      this.worldOnline = false
      return { ok: false, error: 'world unreachable' }
    }
  }

  /** 世界反馈服务：轮询世界主动发来的反馈事件（石头坠落砸中/花香/鸟鸣）。 */
  async worldEvents() {
    try {
      const r = await this._worldFetch(`/events?entity=digital-life&since=${this.worldLastEventTs}`)
      const events = Array.isArray(r.data?.events) ? r.data.events : []
      for (const ev of events) {
        if (Number.isFinite(ev.ts) && ev.ts > this.worldLastEventTs) this.worldLastEventTs = ev.ts
        this._onWorldEvent(ev)
      }
    } catch {
      /* 世界离线：跳过 */
    }
  }

  /** 世界反馈事件 → 感知刺激 + 内分泌反应。 */
  _onWorldEvent(ev) {
    if (!ev) return
    const text = ev.text ?? `世界反馈：${ev.type ?? 'unknown'}`
    // 反馈转为感知注入（视觉/听觉/触觉通道）
    const module = ev.type === 'stone_hit' ? 'touch' : ev.type === 'bird_sing' ? 'audio' : 'visual'
    try {
      this.perceive(text, { module, source: 'world', kind: 'world-event' })
    } catch {
      /* 感知失败不影响世界循环 */
    }
    if (ev.pain && this.body) {
      this.body.applyEffect({ pain: Number(ev.pain) })
      this.endocrine.releaseMany({ cortisol: 0.3, adrenaline: 0.2 })
    }
    if (ev.type === 'smell_drift' && ev.pleasant) {
      this.endocrine.releaseMany({ dopamine: 0.05 })
    }
  }

  /** 行为效果 → 内分泌/身体状态（欲望满足闭环）。 */
  _applyWorldEffects(effects = {}) {
    if (this.body) this.body.applyEffect(effects)
    const release = {}
    if (effects.dopamine) release.dopamine = clamp01(Number(effects.dopamine))
    if (effects.endorphin) release.endorphin = clamp01(Number(effects.endorphin))
    if (effects.serotonin) release.serotonin = clamp01(Number(effects.serotonin))
    if (effects.cortisol) release.cortisol = clamp01(Number(effects.cortisol))
    if (effects.adrenaline) release.adrenaline = clamp01(Number(effects.adrenaline))
    if (Object.keys(release).length > 0) this.endocrine.releaseMany(release)
  }

  /**
   * 世界交互心跳（每秒，清醒期）：
   * 1. 常驻感知（看/听/闻）→ 2. 身体数据网络决策 → 3. 行为请求 → 4. 效果处理
   * 5. 轮询世界反馈事件（痛感/花香等主动刺激）
   */
  async worldTick() {
    if (!this.initialized || !this.body || this.asleep) return
    // 身体代谢（能量消耗/痛感消退）
    this.body.tick(1)
    // 1. 常驻感知
    const sense = await this.worldSense()
    const desire = desireFromEndocrine(this.endocrine.levels)
    const bodyVec = this.body.encode()
    // 2. 身体数据网络决策（所有身体行为由它的输出完成）
    let decision = null
    let prevInput = null
    let chosenIndex = null
    if (this.bodyNet) {
      const encoded = new WorldEncoder().encode(sense, this.body)
      prevInput = BodyNet.assembleInput(encoded, desire)
      decision = this.bodyNet.decide(sense, this.body, desire, {})
      if (decision) {
        if (decision.action === 'move' && decision.dir) chosenIndex = MOVE_DIRS.indexOf(decision.dir)
        else if (decision.action === 'jump') chosenIndex = 4
        else if (INTERACT_NAMES.includes(decision.action)) chosenIndex = 6 + INTERACT_NAMES.indexOf(decision.action)
      }
    }
    if (!decision) {
      this._perceiveWorldSense(sense)
      return
    }
    // 3. 行为请求
    let actRes = { ok: false }
    let reward = 0
    if (decision.action === 'move') {
      actRes = await this.worldAct('move', { dir: decision.dir })
      if (actRes.ok) {
        this.body.applyPos(actRes.newPos)
        this.body.steps++
        this.body.speed++
      }
    } else if (decision.action === 'jump') {
      actRes = await this.worldAct('jump', {})
      if (actRes.ok) {
        this.body.applyPos(actRes.newPos)
        this.body.steps++
      }
    } else if (decision.action === 'grab' && decision.target) {
      actRes = await this.worldAct('grab', { target: decision.target })
      if (actRes.ok) this.body.heldItem = decision.target
    } else if (decision.action === 'eat') {
      actRes = await this.worldAct('eat', { target: this.body.heldItem ?? decision.target })
      if (actRes.ok && actRes.effects) {
        this._applyWorldEffects(actRes.effects)
        this.body.heldItem = null
        this.body.eats++
        reward = (actRes.effects.dopamine ?? 0) + (actRes.effects.satiety ?? 0)
        this.workspace?.log(`world: ate → ${JSON.stringify(actRes.result ?? {})}`)
      }
    } else {
      // rest：静止休息
      actRes = { ok: true, result: '休息中' }
      this._applyWorldEffects({ energy: 0.01 })
    }
    // 4. 效果与奖励 → 身体数据网络强化学习 + 内分泌
    if (actRes.effects) this._applyWorldEffects(actRes.effects)
    if (this.bodyNet && Number.isInteger(chosenIndex) && chosenIndex >= 0 && prevInput) {
      this.bodyNet.train(prevInput, decision.output, chosenIndex, reward, {})
    }
    // 5. 感知记录（世界信息以数据形式进入情节日志）
    this._perceiveWorldSense(sense, decision, actRes)
    // 6. 轮询世界主动反馈
    await this.worldEvents()
  }

  /** 把世界感知写入情节（限流：每 5 秒最多一条，避免日志爆炸）。 */
  _perceiveWorldSense(sense, decision = null, actRes = null) {
    const now = Date.now()
    if (this._lastWorldLogAt && now - this._lastWorldLogAt < 5000) return
    this._lastWorldLogAt = now
    const parts = []
    for (const s of sense.sights ?? []) if (s.text) parts.push(s.text)
    for (const s of sense.smells ?? []) if (s.text) parts.push(s.text)
    for (const s of sense.sounds ?? []) if (s.text) parts.push(s.text)
    const text = parts.slice(0, 3).join('；') || '四周静悄悄的。'
    try {
      this.perceive(text, { module: 'visual', source: 'world', kind: 'world-sense' })
    } catch {
      /* 感知失败不影响世界循环 */
    }
  }

  // -------------------------------------------------------------------------
  // 神经网络扩展：外部扩容（未训练网络挂载为记忆网络的扩展区）
  // -------------------------------------------------------------------------

  /** 挂载扩展网络（内部实现：不写配置）。文件已存在但不是插件格式 → 拒绝（绝不覆盖用户文件）。 */
  _mountExtensionInternal(path, opts = {}) {
    if (this.extensions.has(path)) return { path, alreadyMounted: true }
    let ext
    let created = false
    let exists = false
    try {
      exists = existsSync(path)
    } catch {
      /* 状态未知，按不存在处理 → 会尝试创建 */
    }
    if (exists) {
      let raw
      try {
        raw = readFileSync(path, 'utf8')
      } catch (error) {
        throw new Error(`扩展网络文件不可读：${path}（${error?.message ?? error}）`)
      }
      try {
        ext = MemoryNet.fromJSON(JSON.parse(raw))
      } catch {
        throw new Error(
          `该文件不是可塑性记忆网络格式（.brain.json）：${path}。` +
          '请填写一个「不存在的路径」让插件创建全新的空白网络，或填写一个本插件保存过的 .brain.json 权重文件。' +
          '（PyTorch 的 model.py / params.json / architecture.json 不能直接挂载）',
        )
      }
    } else {
      // 文件不存在 → 创建全新（未曾训练的）网络
      const seed = opts.seed ?? (Math.floor(Math.random() * 1e9) >>> 0)
      ext = new MemoryNet({ seed, baseLr: this.memory ? this.memory.baseLr : 0.008 })
      created = true
      try {
        this.workspace.saveExtensionFile(path, ext.toJSON())
      } catch {
        /* 目录不存在则留到 save() 时重试 */
      }
    }
    this.extensions.set(path, ext)
    return { path, created, seed: ext.seed }
  }

  /** 批量挂载：paths 为路径数组，逐个挂载，单个失败不影响其余。 */
  mountExtensions(pathInputs, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化：请先指定工作区文件夹')
    const mounted = []
    const errors = []
    for (const p of pathInputs) {
      try {
        mounted.push(this.mountExtension(p, opts))
      } catch (error) {
        errors.push({ path: String(p), error: error instanceof Error ? error.message : String(error) })
      }
    }
    return { mounted, errors }
  }

  /** 挂载扩展网络并持久化到工作区配置（重启自动恢复）。 */
  mountExtension(pathInput, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化：请先指定工作区文件夹')
    const path = normalizeFolder(pathInput)
    if (!path) throw new Error('扩展网络路径无效')
    const result = this._mountExtensionInternal(path, opts)
    const cfg = this.workspace.readConfig() ?? {}
    if (!Array.isArray(cfg.extensions)) cfg.extensions = []
    if (!cfg.extensions.includes(path)) {
      cfg.extensions.push(path)
      this.workspace.writeConfig(cfg)
    }
    this.workspace.log(`extension mounted: ${path} created=${result.created === true}`)
    return result
  }

  /** 卸载扩展网络（保存权重后从列表移除）。 */
  unmountExtension(pathInput) {
    if (!this.initialized) throw new Error('大脑未初始化：请先指定工作区文件夹')
    const path = normalizeFolder(pathInput)
    if (!path) throw new Error('扩展网络路径无效')
    const ext = this.extensions.get(path)
    if (!ext) throw new Error('该路径未挂载扩展网络')
    try {
      this.workspace.saveExtensionFile(path, ext.toJSON())
    } catch (error) {
      this.lastError = String(error?.message ?? error)
    }
    this.extensions.delete(path)
    const cfg = this.workspace.readConfig() ?? {}
    if (Array.isArray(cfg.extensions)) {
      cfg.extensions = cfg.extensions.filter((p) => p !== path)
      this.workspace.writeConfig(cfg)
    }
    this.workspace.log(`extension unmounted: ${path}`)
    return { path, unmounted: true }
  }

  /** 扩展网络列表摘要。 */
  extensionList() {
    return Array.from(this.extensions.entries()).map(([path, ext]) => {
      const st = ext.status()
      return {
        path,
        paramCount: st.paramCount,
        layerSizes: st.layerSizes,
        predErrorEma: Number(st.predErrorEma.toFixed(4)),
        observeCount: st.observeCount,
        tick: st.tick,
      }
    })
  }

  /** 扫描感知文件夹（每秒）。 */
  scanSenses() {
    if (!this.initialized) return { processed: 0 }
    const files = this.workspace.listPendingSenses()
    let processed = 0
    for (const file of files) {
      try {
        let raw = this.workspace.readSenseFile(file)
        // JSON 感知文件：单个 {module,text,kind,reply,moodTag} 或语料数组 [{...}]
        try {
          const parsed = JSON.parse(raw)
          if (Array.isArray(parsed)) {
            // 语料批量格式：一个文件含多组问答
            let n = 0
            for (const item of parsed) {
              if (item && typeof item === 'object' && typeof item.text === 'string') {
                this.perceive(item.text, {
                  module: typeof item.module === 'string' ? item.module : 'memory',
                  source: 'sense-file',
                  kind: typeof item.kind === 'string' ? item.kind : null,
                  reply: typeof item.reply === 'string' ? item.reply : undefined,
                  moodTag: typeof item.moodTag === 'string' ? item.moodTag : null,
                })
                n++
              }
            }
            this.workspace.archiveSense(file)
            processed++
            this.scannedFiles++
            this.workspace.log(`sense consumed: ${file} (batch=${n})`)
            continue
          }
          if (parsed && typeof parsed === 'object' && typeof parsed.text === 'string') {
            this.perceive(parsed.text, {
              module: typeof parsed.module === 'string' ? parsed.module : 'memory',
              source: 'sense-file',
              kind: typeof parsed.kind === 'string' ? parsed.kind : null,
              reply: typeof parsed.reply === 'string' ? parsed.reply : undefined,
              moodTag: typeof parsed.moodTag === 'string' ? parsed.moodTag : null,
            })
            this.workspace.archiveSense(file)
            processed++
            this.scannedFiles++
            this.workspace.log(`sense consumed: ${file}`)
            continue
          }
        } catch {
          /* 非 JSON → 按纯文本感知处理 */
        }
        this.perceive(raw, { module: 'memory', source: 'sense-file' })
        this.workspace.archiveSense(file)
        processed++
        this.scannedFiles++
        this.workspace.log(`sense consumed: ${file} (text)`)
      } catch (error) {
        this.lastError = String(error?.message ?? error)
        this.workspace.log(`sense FAILED: ${file} :: ${this.lastError}`)
      }
    }
    // 内分泌衰减 + 睡眠动力学 + 周期决策
    this.endocrine.tick(1)
    this._updateSleep(1)
    if (!this.asleep && this.mode === 'training' && this.controller.tick % 10 === 0) {
      const report = this.controller.decide(this.memory, this.endocrine)
      const outcome = this.controller.learnFromOutcome(this.memory)
      if (report.grew || report.pruned) {
        this.workspace.log(
          `structure: ${report.grew ? 'grow@L' + report.grew.layer : ''}${report.pruned ? ' prune@L' + report.pruned.layer : ''} reward=${outcome.reward.toFixed(3)}`,
        )
        this.save() // 结构变更立即落盘
      }
    }
    return { processed }
  }

  // -------------------------------------------------------------------------
  // 听 / 说 / 读 / 写
  // -------------------------------------------------------------------------

  /** 听：接收感知文本（API 或工具）。 */
  hear(text, opts = {}) {
    return this.perceive(text, { ...opts, source: opts.source ?? 'api' })
  }

  /** 说：状态感知对话引擎。
   *  · 有问话（text）：编码问话 → 从学过的对话语料中检索回答（按身体状态加权 + 轮换不重复）
   *  · 无问话：主动发言（情绪表达 + 记忆/语言联想 + 人格自述） */
  speak(text, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化')
    const wasAsleep = this.asleep
    const eff = this.endocrine.effects()
    const emo = this.memory.lastEmo
    const persona = this.memory.expressPersona(eff.socialAxis)
    const interp = interpretPersona(persona)
    const emoInterp = interpretEmotion(emo)

    // ---- 对话模式：问话 → 语料检索回答 ----
    if (text) {
      this.hear(text, { module: 'audio', source: 'speak' })
      const conv = this._answer(text)
      if (conv) {
        const statusLine = this._statusPhrase(this.endocrine)
        const wakeLine = wasAsleep ? '我刚才睡着了，被你叫醒了。' : ''
        // 缺陷人格网络加工美化（架在操控网络输出与用户之间；对话中权重固定）
        const polished = this._applyFlaw(`${wakeLine}${statusLine}${conv.reply}`)
        // V4：他说出的话带小标记（供回忆对话网络识别，自我训练时加倍）
        const marked = this._markSelfSpoken(polished)
        return {
          text: marked.displayText,
          conversation: true,
          matchedQuestion: conv.matched,
          status: statusLine,
          reply: conv.reply,
          moodTag: conv.moodTag,
          persona: interp,
          emotion: emoInterp,
          wokeFromSleep: wasAsleep,
        }
      }
      // 语料中没匹配到 → 回退主动表达
    }

    // ---- 主动发言模式（无问话或未匹配）----
    const valence = eff.valence
    const arousal = eff.arousal
    let mood
    if (valence > 0.3 && arousal > 0.45) mood = '我感到兴奋而愉快，心里很有劲头。'
    else if (valence > 0.3) mood = '我感到愉快而平静。'
    else if (valence < -0.3 && arousal > 0.45) mood = '我有些紧张不安，心里沉沉的。'
    else if (valence < -0.3) mood = '我有些低落，提不起精神。'
    else if (arousal > 0.45) mood = '我有点激动，注意力很集中。'
    else if (this.asleep) mood = '我迷迷糊糊的，快要睡着了。'
    else mood = '我感觉平静。'
    const wakeLine = wasAsleep ? '我刚才睡着了，被你叫醒了。' : ''

    // 叙事自我：最近的深刻记忆
    const episodes = this.workspace.readEpisodes(50)
    let recall = null
    if (episodes.length > 0) {
      const top = episodes.reduce((a, b) => (a.salience >= b.salience ? a : b))
      if (top && top.text) recall = top.text.slice(0, 60)
    }
    const recallLine = recall ? `我想起之前感知到的一些内容：「${recall}」。` : ''

    // 语言联想：从学过的语言素材（语文教材 kind=language）中检索与当前情绪匹配的片段，
    // 让"说出的话"来自它真正读过的内容，而非写死的模板
    const langLine = this._quoteLanguage(emoInterp, eff)

    // 人格自述（道德基础 + 社会轴，视频09/02/03）
    const traits = interp.traits ?? {}
    const moral = interp.moralFoundations ?? {}
    const moralOrder = Object.entries(moral).sort((a, b) => b[1] - a[1])
    const topMoral = moralOrder[0]
    const moralLine = topMoral
      ? `我在意${({ care: '关爱', fairness: '公平', loyalty: '忠诚', authority: '权威', sanctity: '圣洁' })[topMoral[0]] ?? topMoral[0]}。`
      : ''
    const socialLine = eff.socialAxis > 0.25
      ? '我对人亲近、信任。'
      : eff.socialAxis < -0.25
        ? '我倾向于坚持自己、直面竞争。'
        : '我在亲和与独立之间保持平衡。'

    const text2 = `${wakeLine}${mood}${langLine}${recallLine}${moralLine}${socialLine}`
    // 缺陷人格网络加工美化（主动发言同样经过修饰层）
    const polished = this._applyFlaw(text2)
    // V4：小标记
    const marked = this._markSelfSpoken(polished)
    return { text: marked.displayText, conversation: false, mood, recall, quote: langLine, persona: interp, emotion: emoInterp, wokeFromSleep: wasAsleep }
  }

  /** V4：他说出的话带上小标记（零宽空格），存入自我提示词预存（source=self）。
   *  返回剥离标记后的展示文本。 */
  _markSelfSpoken(text) {
    const marked = `${text}${SELF_MARK}`
    try {
      this.selfTrainer?.appendSelfPending({ t: Date.now(), text: marked, source: 'self', module: 'audio' })
    } catch (error) {
      this.lastError = String(error?.message ?? error)
    }
    return { displayText: stripMark(marked) }
  }

  /** 缺陷人格网络修饰层：把操控网络的输出加工美化（对话中权重固定，仅前向）。 */
  _applyFlaw(text) {
    try {
      if (!this.flaw || typeof text !== 'string' || text === '') return text
      const { latent } = this.controller.hear(text, false)
      const m = this.flaw.apply(latent, this.endocrine.levels)
      let out = text
      // 修饰参数 → 措辞映射（修订 4）
      if (m.quirk > 0.5 && !out.startsWith('嘿嘿')) out = '嘿嘿，' + out
      if (m.deflect > 0.6 && !out.startsWith('嗯')) out = '嗯……' + out
      if (m.intimacy > 0.6 && !out.includes('朋友')) out = out.replace(/^(嘿嘿，)?(嗯……)?/, (p) => p + '朋友，')
      if (m.warmth > 0.6 && !/[呀呢哦～]$/.test(out)) out = out.replace(/[。！？]?$/, '～')
      if (m.embellish > 0.6) out += '我很高兴能和你聊这些。'
      return out
    } catch {
      return text
    }
  }

  // -------------------------------------------------------------------------
  // 对话引擎：问话理解（网络编码 + 语义检索）+ 身体状态加权 + 轮换多样性
  // -------------------------------------------------------------------------

  /** 当前身体状态 → 语料 moodTag。 */
  _currentMoodTag(endocrine) {
    if (this.wakefulness < 0.4 || endocrine.get('melatonin') > 0.5) return 'tired'
    const eff = endocrine.effects()
    if (eff.valence > 0.25) return 'happy'
    if (eff.valence < -0.25 && eff.arousal > 0.45) return 'upset'
    if (eff.valence < -0.25) return 'sad'
    if (eff.arousal > 0.5) return 'excited'
    return 'neutral'
  }

  /** 状态措辞（多套变体，按 tick 轮换，不重复同一句）。 */
  _statusPhrase(endocrine) {
    let pool
    if (this.wakefulness < 0.4 || endocrine.get('melatonin') > 0.5) {
      pool = ['我有点困了，', '我迷迷糊糊的，', '我困得眼皮打架，', '我有些犯困，']
    } else {
      const eff = endocrine.effects()
      if (eff.valence > 0.3) pool = ['我心情很好，', '我现在挺开心的，', '我状态很不错，', '我心情舒畅，']
      else if (eff.valence < -0.3 && eff.arousal > 0.45) pool = ['我有点烦躁，', '我心里不太安稳，', '我有些紧张，', '我心烦意乱的，']
      else if (eff.valence < -0.3) pool = ['我有点低落，', '我没什么精神，', '我有些难过，', '我提不起劲，']
      else if (eff.arousal > 0.5) pool = ['我挺精神的，', '我注意力很集中，', '我有点兴奋，', '我劲头十足，']
      else pool = ['我挺好的，', '我还好，', '我一切正常，', '我不错，']
    }
    return pool[this.memory.tick % pool.length]
  }

  /** 问话 → 检索学过的对话语料 → 状态加权 + 轮换回答。
   *  策略：精确匹配优先（问句相同/互相包含）→ 语义检索兜底 → 当前身体状态加权 → 轮换。 */
  _answer(text) {
    try {
      const convEps = this.workspace
        .readEpisodes(500)
        .filter((e) => e && e.kind === 'conversation' && typeof e.reply === 'string' && e.reply !== '')
      if (convEps.length === 0) return null
      const t = String(text).trim()
      const moodTag = this._currentMoodTag(this.endocrine)

      // 1. 精确匹配：问句完全相同，或互相包含（长度≥2 防单字误配）
      let pool = convEps.filter(
        (e) =>
          e.text === t ||
          (t.length >= 2 && typeof e.text === 'string' && e.text.length >= 2 && (t.includes(e.text) || e.text.includes(t))),
      )

      // 2. 语义检索兜底：精确无命中时按相似度取 top-K
      if (pool.length === 0) {
        const { latent } = this.controller.hear(t, false)
        const a = new Float64Array(32)
        for (let i = 0; i < 32; i++) a[i] = latent[i]
        pool = convEps
          .map((ep) => {
            if (!Array.isArray(ep.latent32) || ep.latent32.length < 32) return { ep, score: -1 }
            const b = new Float64Array(32)
            for (let i = 0; i < 32; i++) b[i] = ep.latent32[i]
            return { ep, score: cosine(a, b) }
          })
          .filter((x) => x.score > 0.15)
          .sort((x, y) => y.score - x.score)
          .slice(0, 4)
      }
      if (pool.length === 0) return null

      // 3. 加权轮换：与当前身体状态同 moodTag 的回答权重 ×3（更常命中但保留多样性）
      const weights = pool.map((e) => ((e.ep?.moodTag ?? e.moodTag) === moodTag ? 3 : 1))
      const totalW = weights.reduce((s, w) => s + w, 0)
      const hash = fnv1a(t)
      let pos = (this.memory.tick + hash) % totalW
      let idx = 0
      for (let i = 0; i < weights.length; i++) {
        pos -= weights[i]
        if (pos < 0) {
          idx = i
          break
        }
      }
      const pick = pool[idx]
      return { matched: String(pick.ep?.text ?? pick.text).slice(0, 40), reply: pick.ep?.reply ?? pick.reply, moodTag }
    } catch {
      return null
    }
  }

  /** 语言联想：情绪 → 主题词 → 检索学过的语言素材（kind=language）→ 引用。 */
  _quoteLanguage(emoInterp, eff) {
    try {
      const langEpisodes = this.workspace.readEpisodes(400).filter((e) => e && e.kind === 'language')
      if (langEpisodes.length === 0) return ''
      // 当前主导情绪 → 检索主题词
      const EMO_TOPIC = {
        joy: '喜悦 美好 春天 希望', sadness: '思念 离愁 故乡', anger: '壮志 不屈 愤慨',
        fear: '危险 黑夜 守护', calm: '宁静 山水 田园', anxiety: '忧虑 时光 珍惜',
        trust: '友谊 信任 相伴', love: '亲情 思念 牵挂',
      }
      const names = ['joy', 'sadness', 'anger', 'fear', 'calm', 'anxiety', 'trust', 'love']
      let hot = 'calm'
      let hotV = -Infinity
      for (const n of names) {
        const v = Number(emoInterp[n] ?? 0)
        if (v > hotV) {
          hotV = v
          hot = n
        }
      }
      const topic = EMO_TOPIC[hot] ?? '美好'
      const { latent } = this.controller.hear(topic, false)
      const scored = langEpisodes
        .map((ep) => {
          if (!Array.isArray(ep.latent32) || ep.latent32.length < 32) return { ep, sim: -1 }
          const a = new Float64Array(32)
          for (let i = 0; i < 32; i++) a[i] = latent[i]
          const b = new Float64Array(32)
          for (let i = 0; i < 32; i++) b[i] = ep.latent32[i]
          return { ep, sim: cosine(a, b) }
        })
        .filter((x) => x.sim > 0.1)
        .sort((a, b) => b.sim - a.sim)
      const top = scored[0]
      if (top && top.ep && top.ep.text) {
        const quote = top.ep.text.slice(0, 70)
        return `我还记得读到过的一段文字：「${quote}」。`
      }
    } catch {
      /* 语言联想失败不影响表达 */
    }
    return ''
  }

  /** 读：联想检索（记忆区激活 + 情节日志相似度 + 扩展网络联想）。 */
  read(query, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化')
    const { latent } = this.controller.hear(String(query), false)
    const rec = this.memory.recall(latent, { module: 'memory', salience: 0.6 })
    const episodes = this.workspace.readEpisodes(200)
    const scored = episodes
      .map((ep) => {
        if (!Array.isArray(ep.latent32) || ep.latent32.length < 32) return { ep, sim: -1 }
        const a = new Float64Array(32)
        for (let i = 0; i < 32; i++) a[i] = latent[i]
        const b = new Float64Array(32)
        for (let i = 0; i < 32; i++) b[i] = ep.latent32[i]
        return { ep, sim: cosine(a, b) }
      })
      .filter((x) => x.sim > 0.2)
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 5)
    // 扩展网络联想状态（扩容网络各自前向，返回记忆区激活）
    const extensionStates = []
    for (const [path, ext] of this.extensions) {
      try {
        const erec = ext.recall(latent, { module: 'memory', salience: 0.6 })
        extensionStates.push({
          path,
          activeUnits: erec.memAct.length,
          meanActivation: Number((erec.memAct.reduce((s, v) => s + Math.abs(v), 0) / erec.memAct.length).toFixed(4)),
        })
      } catch (error) {
        this.lastError = String(error?.message ?? error)
      }
    }
    return {
      query,
      memoryState: {
        activeUnits: rec.memAct.length,
        meanActivation: Number((rec.memAct.reduce((s, v) => s + Math.abs(v), 0) / rec.memAct.length).toFixed(4)),
      },
      matches: scored.map(({ ep, sim }) => ({
        similarity: Number(sim.toFixed(3)),
        text: ep.text,
        module: ep.module,
        salience: ep.salience,
        at: new Date(ep.t).toISOString(),
      })),
      extensions: extensionStates,
    }
  }

  /** 写：强化写入（目标人格向量 × 多 epochs → 人格固化）。 */
  write(text, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化')
    const module = opts.module ?? 'persona'
    const epochs = Math.max(1, Math.min(40, opts.epochs ?? 8))
    // 文本 → 编码 → 以人格区为目标：先观察（正常感知），再按感知内容刻写人格
    const { latent } = this.controller.hear(String(text))
    const eff = this.endocrine.effects()
    // 目标人格 = 观察该内容后的人格输出（把内容"内化"为自我）
    const first = this.memory.observe(latent, {
      module,
      salience: eff.memorySalience,
      lrScale: eff.learningRateMod,
      emotionTarget: eff.emotionTarget,
      train: this.mode === 'training',
    })
    const target = new Float64Array(first.persona)
    const res = this.mode === 'training'
      ? this.memory.writePersona(target, { epochs, lrScale: 1.5 })
      : { loss: 0, epochs: 0, skipped: 'frozen' }
    this.workspace.appendEpisode({
      t: Date.now(),
      source: 'write',
      module,
      text: String(text).slice(0, 100),
      cues: 0,
      salience: 1,
      emotion: { valence: 0, arousal: 0, joy: 0, sadness: 0, fear: 0, anxiety: 0 },
      latent32: Array.from(latent.slice(0, 32)).map((v) => Number(v.toFixed(2))),
    })
    this.save()
    return { module, epochs, ...res }
  }

  // -------------------------------------------------------------------------
  // 生命周期控制
  // -------------------------------------------------------------------------

  freeze() {
    if (!this.initialized) throw new Error('大脑未初始化')
    const at = this.workspace.freeze(this.memory)
    this.mode = 'frozen'
    if (this.flaw) this.flaw.frozen = true // 冻结态：缺陷人格网络权重固定
    this.save()
    this.workspace.log(`brain frozen at ${at}`)
    return { frozen: true, at }
  }

  thaw() {
    if (!this.initialized) throw new Error('大脑未初始化')
    this.workspace.thaw()
    this.mode = 'training'
    if (this.flaw) this.flaw.frozen = false // 训练态：缺陷人格网络可被用户 API 训练
    this.workspace.log('brain thawed → training')
    return { frozen: false }
  }

  /** 返祖：记忆网络权重恢复为出厂基因（先天），清空后天经验；体液回到基线。 */
  resetGenome() {
    if (!this.initialized) throw new Error('大脑未初始化')
    const genome = this.workspace.loadGenome()
    if (!genome) throw new Error('工作区没有出厂基因快照')
    this.memory = MemoryNet.fromJSON(genome)
    this.endocrine = new EndocrineSystem() // 身体状态一并返祖
    this.mode = 'training'
    this.workspace.thaw()
    this.save()
    this.workspace.log('brain reset to genome')
    return { reset: true }
  }

  /** 直接操控接口：操控网络增删记忆网络神经元。 */
  structuralCommand(cmd, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化')
    if (cmd === 'grow') {
      const layer = opts.layer ?? 0
      const r = this.memory.growNeuron(layer, { mutation: opts.mutation ?? 0.2 })
      if (!r) throw new Error(`层 ${layer} 不可增（不可变或不存在）`)
      this.save()
      return r
    }
    if (cmd === 'prune') {
      const layer = opts.layer ?? 0
      const idx = Number.isInteger(opts.index) ? opts.index : this.memory.findPrunable(layer)?.index
      if (idx === undefined || idx === null) throw new Error('没有可剪的神经元')
      const r = this.memory.pruneNeuron(layer, idx)
      if (!r) throw new Error('剪除失败')
      this.save()
      return r
    }
    if (cmd === 'adjustLr') {
      const factor = Math.max(0.1, Math.min(10, Number(opts.factor ?? 1)))
      this.memory.baseLr *= factor
      this.save()
      return { baseLr: this.memory.baseLr }
    }
    throw new Error(`未知结构命令: ${cmd}（支持 grow/prune/adjustLr）`)
  }

  /** 内分泌手动调节。 */
  adjustMolecules(map) {
    if (!this.initialized) throw new Error('大脑未初始化')
    this.endocrine.releaseMany(map ?? {})
    return this.endocrine.snapshot()
  }

  /** 自动生成神经网络（合并自 AI 模型构建插件，生成可塑性记忆自己的网络类型）。
   *  生成 MemoryNet 初始化权重保存到 path 并自动挂载为外部扩容。 */
  generateNetwork(pathInput, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化')
    const path = normalizeFolder(pathInput)
    if (!path) throw new Error('网络保存路径无效')
    if (existsSync(path)) {
      throw new Error(`路径已存在文件：${path}。请填一个不存在的路径（自动创建全新网络）。`)
    }
    const seed = Number.isInteger(opts.seed) ? opts.seed : (Math.floor(Math.random() * 1e9) >>> 0)
    const net = new MemoryNet({ seed, baseLr: this.memory.baseLr })
    this.workspace.saveExtensionFile(path, net.toJSON())
    const mounted = this.mountExtension(path)
    this.workspace.log(`network generated: ${path} seed=${seed} params=${net.paramCount}`)
    return { path, seed, paramCount: net.paramCount, layerSizes: net.layerSizes, mounted: !mounted.alreadyMounted, created: true }
  }

  /** 用户训练缺陷人格网络（唯一训练入口；仅训练态可用；对话中权重保持固定）。 */
  trainFlaw(text, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化')
    if (this.mode !== 'training') {
      return { loss: 0, epochs: 0, skipped: 'frozen' }
    }
    if (!this.flaw) throw new Error('缺陷人格网络未初始化')
    const epochs = Math.max(1, Math.min(40, opts.epochs ?? 8))
    const result = this.flaw.train(String(text), { epochs, endoLevels: opts.endoLevels })
    this.save()
    this.workspace.log(`flaw trained: epochs=${epochs} loss=${result.loss?.toFixed?.(4) ?? result.loss}`)
    return result
  }

  /** 分析存储器最近分析记录。 */
  analysisStatus(limit = 20) {
    if (!this.initialized || !this.analysisStore) return { initialized: false }
    return {
      initialized: true,
      count: this.analysisStore.count(),
      recent: this.analysisStore.recent(Math.min(100, limit)),
      network: this.analysisNet ? this.analysisNet.status() : null,
    }
  }

  // -------------------------------------------------------------------------
  // 导出
  // -------------------------------------------------------------------------

  /** 人格化数据导出（低 token，供外部模型/agent 注入使用）。 */
  persona(opts = {}) {
    if (!this.initialized) return { initialized: false }
    const eff = this.endocrine.effects()
    const personaVec = this.memory.expressPersona(eff.socialAxis)
    const episodes = this.workspace.readEpisodes(opts.recent ?? 6)
    return {
      initialized: true,
      name: this.workspace.readConfig()?.name ?? '数字生命',
      mode: this.mode,
      updatedAt: new Date().toISOString(),
      tick: this.memory.tick,
      predictions: this.memory.observeCount,
      persona: interpretPersona(personaVec),
      emotion: interpretEmotion(this.memory.lastEmo),
      state: {
        valence: Number(eff.valence.toFixed(3)),
        arousal: Number(eff.arousal.toFixed(3)),
        socialAxis: Number(eff.socialAxis.toFixed(3)),
        learningRateMod: Number(eff.learningRateMod.toFixed(3)),
      },
      recent: episodes.map((ep) => ({
        text: (ep.text ?? '').slice(0, 80),
        module: ep.module,
        salience: ep.salience,
        at: ep.t ? new Date(ep.t).toISOString() : undefined,
      })),
    }
  }

  /** 完整状态（UI 仪表盘 / API /status）。 */
  status() {
    if (!this.initialized) {
      return { initialized: false, running: false }
    }
    const eff = this.endocrine.effects()
    return {
      initialized: true,
      running: this.running,
      mode: this.mode,
      workspace: this.workspace.folder,
      name: this.workspace.readConfig()?.name ?? '数字生命',
      seed: this.workspace.readConfig()?.seed ?? null,
      bootedAt: this.bootedAt,
      scannedFiles: this.scannedFiles,
      perceptions: this.perceptions,
      lastError: this.lastError,
      memory: this.memory.status(),
      controller: this.controller.status(),
      endocrine: {
        molecules: this.endocrine.snapshot(),
        effects: {
          learningRateMod: Number(eff.learningRateMod.toFixed(3)),
          exploration: Number(eff.exploration.toFixed(3)),
          arousal: Number(eff.arousal.toFixed(3)),
          valence: Number(eff.valence.toFixed(3)),
          socialAxis: Number(eff.socialAxis.toFixed(3)),
        },
      },
      emotion: interpretEmotion(this.memory.lastEmo),
      persona: interpretPersona(this.memory.personaEma),
      pendingSenses: this.workspace.listPendingSenses().length,
      extensions: this.extensionList(),
      pendingCount: this.pendingCount,
      analysis: {
        initialized: !!(this.analysisNet && this.analysisStore),
        storeCount: this.analysisStore ? this.analysisStore.count() : 0,
        network: this.analysisNet ? this.analysisNet.status() : null,
      },
      mirror: {
        synced: this.trainer ? this.trainer.loadMirror() !== null : false,
      },
      flaw: {
        initialized: !!this.flaw,
        frozen: this.flaw ? this.flaw.frozen : null,
      },
      sleep: {
        asleep: this.asleep,
        wakefulness: Number(this.wakefulness.toFixed(3)),
        sleepCount: this.sleepCount,
        sleptAt: this.sleepStartedAt,
        lastWakeAt: this.lastWakeAt,
        idleSeconds: Math.round((Date.now() - this.lastPerceptionAt) / 1000),
        melatoninProgress: Number(this._melatoninFromPending().toFixed(3)),
      },
      // 大脑架构 V3：世界 / 基因 / 身体
      world: {
        online: this.worldOnline,
        url: this.worldUrl,
      },
      gene: this.gene ? { length: this.gene.length, full: this.gene, head: this.gene.slice(0, 12), summary: geneSummary(this.gene) } : null,
      body: this.body
        ? {
            pos: this.body.pos,
            dir: this.body.dir,
            energy: Number(this.body.energy.toFixed(3)),
            pain: Number(this.body.pain.toFixed(3)),
            satiety: Number(this.body.satiety.toFixed(3)),
            heldItem: this.body.heldItem,
            steps: this.body.steps,
            eats: this.body.eats,
          }
        : null,
      bodyNet: this.bodyNet ? this.bodyNet.status() : null,
      // 大脑架构 V4：自我训练系统
      self: {
        net: this.selfNet ? { params: this.selfNet.paramCount, ticks: this.selfNet.tick, observeCount: this.selfNet.observeCount } : null,
        recall: this.recallNet ? this.recallNet.status() : null,
        trainer: this.selfTrainer ? this.selfTrainer.status() : null,
      },
    }
  }
}
