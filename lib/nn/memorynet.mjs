// ============================================================================
// dsh-plastic-memory —— 记忆神经网络（MemoryNet）＝"灵魂"
// ----------------------------------------------------------------------------
// 结构对应人脑功能分区（人性矩阵系列视频的机制落地）：
//
//   输入层（256 维格式化数据）＝ 操控神经网络的唯一供给：
//     · 语义潜向量 latent 128（操控网络编码器输出）
//     · 模态特征 modal 64（视觉/听觉/触觉/记忆/人格 one-hot 通道）
//     · 循环上下文 context 64（上一时刻记忆区状态经 U 投影）
//
//   层1 分区层（204 神经元，可增删）：
//     visual [0..40)   视觉区      —— 感知通道特化（分区增益）
//     audio  [40..80)  听觉区
//     touch  [80..120) 触觉区
//     memory [120..164)记忆区      —— 带循环连接（Elman 式，U 由 Hebbian 学习）
//     persona[164..204)人格区      —— 学习率 ×3、权重衰减更低 → 过拟合倾向，
//                                    高频重复输入在此形成强吸引子（人格固化，
//                                    即视频「态度/意识形态/叙事自我」的机制）
//
//   层2 联想层（128 神经元，可增删）：跨区联想整合
//
//   输出（332 维）：
//     prediction 256 —— 对下一时刻输入的全脑预测
//                       （视频05「大脑：节俭的预测机器」：预测误差驱动学习）
//     persona    64  —— 人格特质向量（亲和/竞争轴、道德五基础、大五倾向、叙事）
//                       （视频02/03/09/12）
//     emoConcept 12  —— 情绪的概念贡献（视频04「情绪构建论」：情绪＝概念＋生理
//                       共同构建；生理贡献由 W_emo2 从内分泌目标映射）
//
// 学习信号（全部是真实误差反传，不是规则打分）：
//   1. 预测误差：pred(t-1) − x(t)            （预测加工）
//   2. 生理对齐：emo(t-1) − 内分泌目标(t)     （情绪构建的生理面）
//   3. 叙事一致：persona(t-1) − 自我EMA        （叙事自我的稳定性，视频12）
//   4. 循环学习：记忆区误差 ⊗ 上一时刻记忆态   （Hebbian，记忆巩固）
//   5. 强化写入：目标人格向量 × 多 epochs      （人格/记忆刻写，倾向过拟合）
// ============================================================================

import { PlasticNetwork } from './network.mjs'
import { zeros, randMat, matVecMul, axpy, cosine, arrToPlain, arrFromPlain, mulberry32 } from './core.mjs'

// 分区布局（层1）
export const SEGMENTS = {
  visual: [0, 40],
  audio: [40, 80],
  touch: [80, 120],
  memory: [120, 164],
  persona: [164, 204],
}
export const L1_SIZE = 204

export const DIMS = {
  latent: 128,
  modal: 64,
  context: 64,
  input: 256,
  l2: 128,
  pred: 256,
  persona: 64,
  emo: 12,
  output: 332, // pred + persona + emo
}

export const MODULES = ['visual', 'audio', 'touch', 'memory', 'persona']

export const EMOTION_NAMES = ['valence', 'arousal', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'calm', 'anxiety', 'trust', 'love']

// 人格向量的可解释标签
export const PERSONA_LABELS = [
  '亲和轴', '竞争轴',
  '关爱', '公平', '忠诚', '权威', '圣洁', // 道德基础五原则（视频09）
  '开放性', '尽责性', '外向性', '宜人性', '情绪稳定性', '叙事一致性', '成长性',
  // 其余为自由表征维度（网络自主形成的表征空间）
]

export const MORAL_INDEX = { care: 2, fairness: 3, loyalty: 4, authority: 5, sanctity: 6 }

/** 模块 → 64 维模态特征向量（one-hot 通道块 + 强度/时间编码）。 */
export function makeModalVec(module, { salience = 0.5, intensity = 1, t = 0 } = {}) {
  const v = new Float64Array(DIMS.modal)
  const idx = MODULES.indexOf(module)
  if (idx >= 0) {
    for (let i = 0; i < 12; i++) v[idx * 12 + i] = 1
  }
  // 尾部 4 维：显著性 / 强度 / 时间相位（sin,cos）
  v[60] = Math.max(0, Math.min(1, salience))
  v[61] = Math.max(0, Math.min(1, intensity))
  v[62] = Math.sin(t / 7)
  v[63] = Math.cos(t / 11)
  return v
}

export class MemoryNet {
  constructor({ seed = 42, baseLr = 0.008 } = {}) {
    this.seed = seed
    this.baseLr = baseLr
    const rng = mulberry32(seed)
    this.net = new PlasticNetwork({
      inputSize: DIMS.input,
      momentum: 0.9,
      rng,
      layers: [
        { size: L1_SIZE, activation: 'tanh', lr: 1, decay: 1e-5, mutable: true },
        { size: DIMS.l2, activation: 'tanh', lr: 1, decay: 1e-5, mutable: true },
        { size: DIMS.output, activation: 'tanh', lr: 0.8, decay: 1e-5, mutable: false },
      ],
    })
    // 循环投影 U：上一时刻记忆区激活(44) → context(64)
    this.U = randMat(DIMS.context, SEGMENTS.memory[1] - SEGMENTS.memory[0], rng)
    // 情绪生理映射 W_emo2：内分泌情绪目标(12) → 概念空间贡献(12)
    this.Wemo2 = randMat(DIMS.emo, DIMS.emo, rng)
    this.lrHebb = 0.003
    this.lrWemo2 = 0.004

    this.personaEma = new Float64Array(DIMS.persona) // 叙事自我 EMA
    this.lastPred = null // 上一时刻预测（用于预测误差）
    this.lastEmo = new Float64Array(DIMS.emo)
    this.contextVec = new Float64Array(DIMS.context)
    this.memStatePrev = new Float64Array(SEGMENTS.memory[1] - SEGMENTS.memory[0])
    this.tick = 0
    this.predErrorEma = 1
    this.observeCount = 0
    this.lastFwd = null
  }

  // -------------------------------------------------------------------------
  // 组装与掩码
  // -------------------------------------------------------------------------

  _gainMask1(module, salience) {
    const gain = new Float64Array(this.net.layers[0].size).fill(1)
    if (MODULES.includes(module)) {
      const [a, b] = SEGMENTS[module]
      for (let i = a; i < b && i < gain.length; i++) gain[i] = 1.6
    }
    // 记忆区随唤醒显著性增强（皮质醇固化：高唤醒记得深）
    const [ma, mb] = SEGMENTS.memory
    const memGain = 0.8 + 0.5 * salience
    for (let i = ma; i < mb && i < gain.length; i++) gain[i] *= memGain
    return gain
  }

  /** 人格区 delta 放大掩码（等效人格区学习率 ×3 → 过拟合倾向）。 */
  _deltaScale1() {
    const size = this.net.layers[0].size
    const scale = new Float64Array(size).fill(1)
    const [a, b] = SEGMENTS.persona
    for (let i = a; i < b && i < size; i++) scale[i] = 3.0
    return scale
  }

  // -------------------------------------------------------------------------
  // 感知注入（每个时刻一次：先以预测误差训练，再前向）
  // -------------------------------------------------------------------------

  /**
   * 观察一个格式化感知向量（唯一入口；数据必须来自操控神经网络的格式化输出）。
   * @param {Float64Array} latent 语义潜向量（128）
   * @param {object} opts {module, salience, lrScale, train:true}
   * @returns 状态摘要 {predError, persona, emotion, loss}
   */
  observe(latent, opts = {}) {
    const module = opts.module ?? 'memory'
    const salience = opts.salience ?? 0.5
    const lrScale = opts.lrScale ?? 1
    const train = opts.train ?? true
    this.tick++

    // ---- 组装输入 ----
    const modalVec = makeModalVec(module, { salience, t: this.tick })
    const input = new Float64Array(DIMS.input)
    input.set(latent, 0)
    input.set(modalVec, DIMS.latent)
    input.set(this.contextVec, DIMS.latent + DIMS.modal)

    // ---- 预测误差训练（视频05：预测机器，误差驱动学习）----
    let predError = 0
    let personaErr = 0
    let emoErr = 0
    if (train && this.lastPred) {
      const outputDelta = new Float64Array(DIMS.output)
      for (let i = 0; i < DIMS.pred; i++) {
        const e = this.lastPred[i] - input[i]
        outputDelta[i] = e
        predError += e * e
      }
      // 叙事一致性：人格输出被自我 EMA 温和拉回（λ=0.15）
      const pOff = DIMS.pred
      for (let i = 0; i < DIMS.persona; i++) {
        const e = (this.lastPersona[i] - this.personaEma[i]) * 0.15
        outputDelta[pOff + i] = e
        personaErr += e * e
      }
      // 生理对齐：上一时刻情绪 vs 当前内分泌目标（buildEmotion 时已记录 lastEmoTarget）
      const eOff = DIMS.pred + DIMS.persona
      for (let i = 0; i < DIMS.emo; i++) {
        const e = this.lastEmo[i] - (this.lastEmoTarget ? this.lastEmoTarget[i] : 0)
        outputDelta[eOff + i] = e * 0.5
        emoErr += e * e
      }

      const res = this.net.backward(outputDelta, {
        baseLr: this.baseLr * (0.5 + salience) * lrScale,
        deltaScale: [this._deltaScale1(), null, null],
      })

      // ---- Hebbian 循环学习：记忆区误差 ⊗ 上一时刻记忆态 ----
      if (res.deltas && res.deltas[0]) {
        const errMem = res.deltas[0]
        const [ma, mb] = SEGMENTS.memory
        for (let i = 0; i < DIMS.context; i++) {
          const off = i * this.memStatePrev.length
          for (let j = 0; j < this.memStatePrev.length; j++) {
            this.U.data[off + j] -= this.lrHebb * errMem[ma + j] * this.memStatePrev[j]
          }
        }
      }
      // ---- 情绪生理映射 W_emo2 学习：误差 ⊗ 生理目标 ----
      if (this.lastEmoTarget) {
        for (let i = 0; i < DIMS.emo; i++) {
          const off = i * DIMS.emo
          for (let j = 0; j < DIMS.emo; j++) {
            this.Wemo2.data[off + j] -=
              this.lrWemo2 * (this.lastEmo[i] - this.lastEmoTarget[i]) * this.lastEmoTarget[j]
          }
        }
      }
      this.predErrorEma = this.predErrorEma * 0.98 + 0.02 * Math.sqrt(predError / DIMS.pred)
    }

    // ---- 前向（生成新预测 / 新人格 / 新情绪）----
    const fwd = this.net.forward(input, {
      gainByLayer: [this._gainMask1(module, salience), null, null],
    })
    this.lastFwd = fwd
    const out = fwd.post[fwd.post.length - 1] // 332

    const pred = out.subarray(0, DIMS.pred)
    this.lastPred = new Float64Array(pred)
    const persona = out.subarray(DIMS.pred, DIMS.pred + DIMS.persona)
    this.lastPersona = new Float64Array(persona)
    const emoConcept = out.subarray(DIMS.pred + DIMS.persona)

    // 情绪 = 概念贡献 + 生理贡献（情绪构建论）
    if (opts.emotionTarget) this.lastEmoTarget = opts.emotionTarget
    const phys = matVecMul(this.Wemo2, this.lastEmoTarget ?? new Float64Array(DIMS.emo))
    const emo = new Float64Array(DIMS.emo)
    for (let i = 0; i < DIMS.emo; i++) {
      emo[i] = Math.tanh(emoConcept[i] + phys[i])
    }
    this.lastEmo = emo

    // 更新循环状态：记忆区激活经 U 投影 → context
    const h1 = fwd.post[1]
    const [ma, mb] = SEGMENTS.memory
    for (let j = 0; j < this.memStatePrev.length; j++) this.memStatePrev[j] = h1[ma + j]
    const ctx = matVecMul(this.U, this.memStatePrev)
    this.contextVec = ctx

    // 叙事自我 EMA 缓慢更新
    for (let i = 0; i < DIMS.persona; i++) {
      this.personaEma[i] = this.personaEma[i] * 0.995 + 0.005 * persona[i]
    }

    this.observeCount++
    return {
      predError: predError / DIMS.pred,
      personaErr,
      emoErr,
      persona: new Float64Array(persona),
      emo: new Float64Array(emo),
    }
  }

  /** 人格表达（对外）：亲和/竞争轴受内分泌社会轴调制，其余维度原样。 */
  expressPersona(socialAxis = 0) {
    const p = new Float64Array(this.personaEma)
    const a = Math.max(-1, Math.min(1, socialAxis))
    p[0] = Math.tanh(Math.tanh(p[0]) + 0.25 * a) // 亲和表达
    p[1] = Math.tanh(Math.tanh(p[1]) - 0.25 * a) // 竞争表达
    return p
  }

  /**
   * 强化写入：以目标人格向量多次训练人格输出（记忆刻写 → 固化）。
   * 人格区高学习率 + 多 epochs → 对写入内容过拟合（这是设计目标：
   * 用户反复强调的内容将沉淀为稳定人格）。
   * @param {Float64Array} targetPersona 64 维目标人格
   * @param {object} opts {epochs=8, lrScale=1.5, pushEma=true}
   */
  writePersona(targetPersona, opts = {}) {
    const epochs = Math.max(1, Math.min(40, opts.epochs ?? 8))
    const lrScale = opts.lrScale ?? 1.5
    const input = new Float64Array(DIMS.input)
    input.set(this.lastPred ?? new Float64Array(DIMS.input))
    let lastLoss = 0
    for (let e = 0; e < epochs; e++) {
      const fwd = this.net.forward(input)
      const out = fwd.post[fwd.post.length - 1]
      const delta = new Float64Array(DIMS.output)
      for (let i = 0; i < DIMS.persona; i++) {
        const err = out[DIMS.pred + i] - targetPersona[i]
        delta[DIMS.pred + i] = err
        lastLoss += err * err
      }
      const res = this.net.backward(delta, {
        baseLr: this.baseLr * lrScale,
        deltaScale: [this._deltaScale1(), null, null],
      })
      if (res.deltas && res.deltas[0]) {
        const errMem = res.deltas[0]
        const [ma, mb] = SEGMENTS.memory
        for (let i = 0; i < DIMS.context; i++) {
          const off = i * this.memStatePrev.length
          for (let j = 0; j < this.memStatePrev.length; j++) {
            this.U.data[off + j] -= this.lrHebb * errMem[ma + j] * this.memStatePrev[j]
          }
        }
      }
      // 每 epoch 后更新 personaEma 快照
      const out2 = this.net.forward(input).post
      const pOut = out2[out2.length - 1].subarray(DIMS.pred, DIMS.pred + DIMS.persona)
      for (let i = 0; i < DIMS.persona; i++) this.personaEma[i] = this.personaEma[i] * 0.7 + 0.3 * pOut[i]
    }
    // 叙事自我向目标靠近（写入即自我更新）
    if (opts.pushEma !== false) {
      for (let i = 0; i < DIMS.persona; i++) {
        this.personaEma[i] = this.personaEma[i] * 0.7 + 0.3 * targetPersona[i]
      }
    }
    return { loss: lastLoss / Math.max(1, epochs), epochs }
  }

  /**
   * 联想读取：给定查询潜向量，前向不训练，返回联想状态。
   * （brain 层据此与情节日志做相似度检索。）
   */
  recall(latent, opts = {}) {
    const module = opts.module ?? 'memory'
    const salience = opts.salience ?? 0.5
    const modalVec = makeModalVec(module, { salience, t: this.tick + 1 })
    const input = new Float64Array(DIMS.input)
    input.set(latent, 0)
    input.set(modalVec, DIMS.latent)
    input.set(this.contextVec, DIMS.latent + DIMS.modal)
    const fwd = this.net.forward(input, { gainByLayer: [this._gainMask1(module, salience), null, null] })
    const out = fwd.post[fwd.post.length - 1]
    const h1 = fwd.post[1]
    const [ma, mb] = SEGMENTS.memory
    const memAct = new Float64Array(mb - ma)
    for (let j = 0; j < memAct.length; j++) memAct[j] = h1[ma + j]
    return {
      pred: out.subarray(0, DIMS.pred),
      persona: out.subarray(DIMS.pred, DIMS.pred + DIMS.persona),
      memAct,
    }
  }

  /** 相似度检索用：把潜向量与记忆中相似度（用预测空间度量）。 */
  similarity(latentA, latentB) {
    return cosine(latentA, latentB)
  }

  // -------------------------------------------------------------------------
  // 结构操作透传（由操控网络调用）
  // -------------------------------------------------------------------------

  growNeuron(layerIdx, opts) {
    const res = this.net.addNeuron(layerIdx, opts)
    if (res && layerIdx === 0) this._syncDeltaScaleCache = null
    return res
  }

  pruneNeuron(layerIdx, idx) {
    const res = this.net.pruneNeuron(layerIdx, idx)
    if (res && layerIdx === 0) this._syncDeltaScaleCache = null
    return res
  }

  findPrunable(layerIdx, opts) {
    return this.net.findPrunable(layerIdx, opts)
  }

  readLayerWeights(layerIdx) {
    return this.net.readWeights(layerIdx)
  }

  writeLayerWeights(layerIdx, m, opts) {
    return this.net.writeWeights(layerIdx, m, opts)
  }

  scaleLayerWeights(layerIdx, factor, opts) {
    return this.net.scaleWeights(layerIdx, factor, opts)
  }

  get paramCount() {
    return this.net.paramCount + this.U.data.length + this.Wemo2.data.length
  }

  get layerSizes() {
    return this.net.layerSizes
  }

  // -------------------------------------------------------------------------
  // 状态摘要 / 序列化
  // -------------------------------------------------------------------------

  status() {
    const net = this.net
    const segments = {}
    for (const [name, [a, b]] of Object.entries(SEGMENTS)) {
      let usageSum = 0
      let n = 0
      for (let i = a; i < Math.min(b, net.layers[0].size); i++) {
        usageSum += net.layers[0].usage[i]
        n++
      }
      segments[name] = {
        size: n,
        meanUsage: n > 0 ? usageSum / n : 0,
      }
    }
    return {
      tick: this.tick,
      layerSizes: this.layerSizes,
      paramCount: this.paramCount,
      predErrorEma: this.predErrorEma,
      observeCount: this.observeCount,
      segments,
      structuralChanges: net.structuralChanges.slice(-20),
    }
  }

  toJSON() {
    return {
      seed: this.seed,
      baseLr: this.baseLr,
      net: this.net.toJSON(),
      U: arrToPlain(this.U.data),
      Wemo2: arrToPlain(this.Wemo2.data),
      lrHebb: this.lrHebb,
      lrWemo2: this.lrWemo2,
      personaEma: arrToPlain(this.personaEma),
      lastPred: this.lastPred ? arrToPlain(this.lastPred) : null,
      lastPersona: this.lastPersona ? arrToPlain(this.lastPersona) : null,
      lastEmo: arrToPlain(this.lastEmo),
      lastEmoTarget: this.lastEmoTarget ? arrToPlain(this.lastEmoTarget) : null,
      contextVec: arrToPlain(this.contextVec),
      memStatePrev: arrToPlain(this.memStatePrev),
      tick: this.tick,
      predErrorEma: this.predErrorEma,
      observeCount: this.observeCount,
    }
  }

  static fromJSON(json) {
    const m = Object.create(MemoryNet.prototype)
    m.seed = json.seed ?? 42
    m.baseLr = json.baseLr ?? 0.008
    m.net = PlasticNetwork.fromJSON(json.net)
    m.U = zeros(DIMS.context, SEGMENTS.memory[1] - SEGMENTS.memory[0])
    m.U.data.set(arrFromPlain(json.U))
    m.Wemo2 = zeros(DIMS.emo, DIMS.emo)
    m.Wemo2.data.set(arrFromPlain(json.Wemo2))
    m.lrHebb = json.lrHebb ?? 0.003
    m.lrWemo2 = json.lrWemo2 ?? 0.004
    m.personaEma = arrFromPlain(json.personaEma ?? new Array(DIMS.persona).fill(0))
    m.lastPred = json.lastPred ? arrFromPlain(json.lastPred) : null
    m.lastPersona = json.lastPersona ? arrFromPlain(json.lastPersona) : null
    m.lastEmo = arrFromPlain(json.lastEmo ?? new Array(DIMS.emo).fill(0))
    m.lastEmoTarget = json.lastEmoTarget ? arrFromPlain(json.lastEmoTarget) : null
    m.contextVec = arrFromPlain(json.contextVec ?? new Array(DIMS.context).fill(0))
    m.memStatePrev = arrFromPlain(json.memStatePrev ?? new Array(SEGMENTS.memory[1] - SEGMENTS.memory[0]).fill(0))
    m.tick = json.tick ?? 0
    m.predErrorEma = json.predErrorEma ?? 1
    m.observeCount = json.observeCount ?? 0
    m.lastFwd = null
    return m
  }
}

// ---------------------------------------------------------------------------
// 人格向量解释辅助（给 API / UI 用）
// ---------------------------------------------------------------------------

export function interpretPersona(p) {
  const out = {}
  for (let i = 0; i < Math.min(PERSONA_LABELS.length, p.length); i++) {
    out[PERSONA_LABELS[i]] = Number(p[i].toFixed(3))
  }
  const moral = {}
  for (const [name, idx] of Object.entries(MORAL_INDEX)) {
    moral[name] = Number(p[idx].toFixed(3))
  }
  return { traits: out, moralFoundations: moral }
}

export function interpretEmotion(e) {
  const out = {}
  for (let i = 0; i < EMOTION_NAMES.length; i++) {
    out[EMOTION_NAMES[i]] = Number(e[i].toFixed(3))
  }
  return out
}
