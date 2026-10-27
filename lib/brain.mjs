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
import { cosine } from './nn/core.mjs'
import { readFileSync, existsSync } from 'node:fs'

export class PlasticBrain {
  constructor() {
    this.workspace = null
    this.memory = null
    this.controller = null
    this.endocrine = null
    this.mode = 'training' // training | frozen
    this.scanInterval = null
    this.saveInterval = null
    this.running = false
    this.scannedFiles = 0
    this.perceptions = 0
    this.lastError = null
    this.bootedAt = null
    this.extensions = new Map() // 外部扩容网络：path → MemoryNet
    // 休眠状态：无聊积累 → 清醒度衰减 + 褪黑素上升 → 自主入睡；
    // 任何感知/对话（hear/speak/read/write/感知文件）即唤醒
    this.asleep = false
    this.wakefulness = 1 // 0..1 清醒度
    this.sleepCount = 0
    this.sleepStartedAt = null
    this.lastPerceptionAt = Date.now()
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

    // 出厂基因（视频01：自私的基因 —— 先天权重，永不被训练修改）
    ws.saveGenome(this.memory.toJSON())
    this.save()

    ws.log(`brain initialized: folder=${folder} seed=${seed} name=${name}`)
    this.bootedAt = new Date().toISOString()
    this.start()
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
    return this.status()
  }

  /** 切换工作区（保存当前，加载目标）。 */
  switch(folderInput) {
    if (this.workspace) this.save()
    return this.load(folderInput)
  }

  save() {
    if (!this.initialized) return false
    this.workspace.saveMemory(this.memory)
    this.workspace.saveController(this.controller, this.endocrine)
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
    // 让 timer 不阻止进程退出
    this.scanInterval.unref?.()
    this.saveInterval.unref?.()
  }

  stop() {
    this.running = false
    if (this.scanInterval) clearInterval(this.scanInterval)
    if (this.saveInterval) clearInterval(this.saveInterval)
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

    // 0. 任何感知都算"打扰"：若在休眠则唤醒（用户对他说话即唤醒）
    this.lastPerceptionAt = Date.now()
    const wokeFrom = this.asleep ? 'perception' : null
    if (this.asleep) this._wake(source === 'speak' ? 'speak' : 'perception')

    // 1. 操控网络"听"：编码 + 编码器在线学习
    const { latent } = this.controller.hear(String(text))
    // 2. 身体生理反应：情绪线索 → 分子释放
    const cues = this.controller.react(String(text), this.endocrine)
    // 3. 当前体液状态 → 调制参数
    const eff = this.endocrine.effects()
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
    }
  }

  // -------------------------------------------------------------------------
  // 休眠（自主入睡 + 说话唤醒）
  // -------------------------------------------------------------------------

  /** 入睡（自主或手动）。 */
  _sleep() {
    if (this.asleep) return false
    this.asleep = true
    this.sleepStartedAt = Date.now()
    this.sleepCount++
    this.endocrine.release('gaba', 0.4)
    this.workspace?.log(`brain fell asleep (#${this.sleepCount})`)
    return true
  }

  /** 唤醒。 */
  _wake(source = 'perception') {
    if (!this.asleep) return false
    this.asleep = false
    this.wakefulness = 1
    this.endocrine.set('melatonin', 0.12) // 褪黑素骤降
    this.endocrine.release('adrenaline', 0.25) // 惊醒
    this.workspace?.log(`brain woke up (source=${source})`)
    return true
  }

  /** 每秒心跳的睡眠动力学：清醒度衰减 + 褪黑素积累 → 自主入睡；睡饱自然醒。 */
  _updateSleep(dtSeconds = 1) {
    if (!this.initialized) return
    const idleSeconds = (Date.now() - this.lastPerceptionAt) / 1000
    if (this.asleep) {
      // 睡眠中：清醒度缓慢恢复，睡满 10 分钟自然醒（"睡饱了"）
      this.wakefulness = Math.min(1, this.wakefulness + dtSeconds / 600)
      if (this.wakefulness >= 1 && Date.now() - this.sleepStartedAt >= 10 * 60 * 1000) {
        this._wake('rested')
      }
      return
    }
    if (idleSeconds > 60) {
      // 无聊：清醒度衰减（约 4 分钟无聊 → 归零）+ 褪黑素积累（困意）
      this.wakefulness = Math.max(0, this.wakefulness - dtSeconds / 240)
      this.endocrine.release('melatonin', 0.006 * dtSeconds)
      if (this.wakefulness <= 0.05 && this.endocrine.get('melatonin') > 0.45) {
        this._sleep()
      }
    } else {
      // 有感知活动：恢复清醒
      this.wakefulness = Math.min(1, this.wakefulness + dtSeconds / 60)
    }
  }

  /** 手动入睡（API/工具）。 */
  sleepNow() {
    if (!this.initialized) throw new Error('大脑未初始化')
    const did = this._sleep()
    return { asleep: true, slept: did }
  }

  /** 手动唤醒（API/工具）。 */
  wakeNow() {
    if (!this.initialized) throw new Error('大脑未初始化')
    const did = this._wake('manual')
    return { asleep: false, woke: did }
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
        let module = 'memory'
        let text = raw
        let kind = null
        // JSON 感知文件支持 {module, text, kind} 结构化格式
        try {
          const parsed = JSON.parse(raw)
          if (parsed && typeof parsed === 'object' && typeof parsed.text === 'string') {
            text = parsed.text
            if (typeof parsed.module === 'string') module = parsed.module
            if (typeof parsed.kind === 'string') kind = parsed.kind
          }
        } catch {
          /* 纯文本感知 */
        }
        this.perceive(text, { module, source: 'sense-file', kind })
        this.workspace.archiveSense(file)
        processed++
        this.scannedFiles++
        this.workspace.log(`sense consumed: ${file} (module=${module})`)
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

  /** 说：本地人格化表达（不依赖外部 LLM）。返回 text + 人格/情绪数据。 */
  speak(text, opts = {}) {
    if (!this.initialized) throw new Error('大脑未初始化')
    // 可选：先听用户说的话（若在休眠，说话会把它唤醒）
    const wasAsleep = this.asleep
    if (text) this.hear(text, { module: 'audio', source: 'speak' })
    const eff = this.endocrine.effects()
    const emo = this.memory.lastEmo
    const persona = this.memory.expressPersona(eff.socialAxis)
    const interp = interpretPersona(persona)
    const emoInterp = interpretEmotion(emo)

    // 情绪 → 措辞（情绪构建论的输出端：当前构建的情绪状态）
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
    return { text: text2, mood, recall, quote: langLine, persona: interp, emotion: emoInterp, wokeFromSleep: wasAsleep }
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
    this.save()
    this.workspace.log(`brain frozen at ${at}`)
    return { frozen: true, at }
  }

  thaw() {
    if (!this.initialized) throw new Error('大脑未初始化')
    this.workspace.thaw()
    this.mode = 'training'
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
      sleep: {
        asleep: this.asleep,
        wakefulness: Number(this.wakefulness.toFixed(3)),
        sleepCount: this.sleepCount,
        sleptAt: this.sleepStartedAt,
        idleSeconds: Math.round((Date.now() - this.lastPerceptionAt) / 1000),
      },
    }
  }
}
