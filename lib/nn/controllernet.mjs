// ============================================================================
// dsh-plastic-memory —— 操控神经网络（ControllerNet）＝"身体"
// ----------------------------------------------------------------------------
// 第一个神经网络。职责（对应"身体"隐喻：感知信息、为灵魂获取外界反馈、行动）：
//
//   1. 感官编码：SenseEncoder 把文本感知编码为格式化潜向量（记忆网络唯一数据源）
//   2. 感知解读：检测情绪线索 → 释放内分泌分子（身体对外界的生理反应）
//   3. 结构管理：根据记忆网络健康度（误差/梯度/使用度）实时增减其神经元与参数
//      —— 决策由自身策略网络（policy MLP）给出，用简化 REINFORCE 从
//      「记忆网络预测误差下降=奖励」信号中真实地学习如何操控
//   4. 参数调制：输出学习率建议，调控记忆网络的训练强度
//
// 策略网络：48 维观察（误差 EMA/误差变化/各层梯度/层规模/15 分子浓度/效应轴）
//         → 64 隐藏 → 16 维决策（层1增/层2增/剪枝概率/学习率调制/15 分子释放建议）
// ============================================================================

import { PlasticNetwork } from './network.mjs'
import { SenseEncoder } from './encoder.mjs'
import { MOLECULES, MOLECULE_INDEX } from './endocrine.mjs'
import { arrToPlain, arrFromPlain, mulberry32 } from './core.mjs'

const OBS_SIZE = 48
const ACT_SIZE = 16
// 决策维度布局
const ACT = {
  growL1: 0,
  growL2: 1,
  prune: 2,
  lrMod: 3,
  molecules: [4, 19], // [start, end) 对应 MOLECULES 顺序
}

// 情绪线索词表（身体对外界信号的生理反应，规则先行 + 策略网络微调）
const EMOTION_CUES = [
  { re: /威胁|危险|害怕|恐惧|袭击|战争|灾难|失败|死亡|黑暗/, release: { cortisol: 0.30, adrenaline: 0.40, noradrenaline: 0.30 } },
  { re: /愤怒|生气|攻击|敌人|冲突|对抗|仇恨|侮辱/, release: { testosterone: 0.30, adrenaline: 0.25, noradrenaline: 0.20 } },
  { re: /开心|高兴|快乐|成功|胜利|喜欢|喜悦|欢笑|庆祝/, release: { dopamine: 0.30, endorphin: 0.30, oxytocin: 0.20, serotonin: 0.20 } },
  { re: /爱|拥抱|温暖|信任|朋友|陪伴|亲密|家人/, release: { oxytocin: 0.40, prolactin: 0.20, endorphin: 0.20, serotonin: 0.20 } },
  { re: /悲伤|难过|失去|离别|孤独|哭泣|遗憾/, release: { cortisol: 0.25, prolactin: 0.30 } },
  { re: /焦虑|担心|压力|紧张|截止|期限|任务繁重/, release: { cortisol: 0.35, noradrenaline: 0.20 } },
  { re: /意外|突然|惊讶|震惊|没想到|竟然/, release: { adrenaline: 0.30, glutamate: 0.20 } },
  { re: /平静|放松|休息|睡眠|安静|祥和/, release: { gaba: 0.30, melatonin: 0.20, serotonin: 0.30 } },
  { re: /学习|思考|研究|阅读|探索|好奇|发现/, release: { acetylcholine: 0.30, glutamate: 0.20, dopamine: 0.20 } },
]

export class ControllerNet {
  constructor({ seed = 7, baseLr = 0.008, policyLr = 0.01 } = {}) {
    this.seed = seed
    this.baseLr = baseLr
    this.policyLr = policyLr
    this.encoder = new SenseEncoder(seed * 31 + 1)
    const rng = mulberry32(seed)
    this.policy = new PlasticNetwork({
      inputSize: OBS_SIZE,
      momentum: 0.6,
      rng,
      layers: [
        { size: 64, activation: 'tanh', lr: 1, decay: 1e-5, mutable: false },
        { size: ACT_SIZE, activation: 'sigmoid', lr: 1, decay: 1e-5, mutable: false },
      ],
    })
    this.tick = 0
    this.decisionCount = 0
    this.lastPredError = 1
    this.lastReward = 0
    this.rewardEma = 0
    this.lastObs = null
    this.lastAction = null
    this.growCount = 0
    this.pruneCount = 0
    this.lastStructureChangeAt = -999
  }

  // -------------------------------------------------------------------------
  // 感官：听（编码）+ 生理反应
  // -------------------------------------------------------------------------

  /** 感知文本 → {latent, bag} + 在线训练编码器。 */
  hear(text, trainEncoder = true) {
    if (trainEncoder) this.encoder.train(text, 0.002)
    return this.encoder.encode(text)
  }

  /** 感知解读：情绪线索 → 内分泌释放（身体生理反应）。 */
  react(text, endocrine) {
    const t = String(text ?? '')
    let hits = 0
    for (const cue of EMOTION_CUES) {
      if (cue.re.test(t)) {
        endocrine.releaseMany(cue.release)
        hits++
      }
    }
    return hits
  }

  // -------------------------------------------------------------------------
  // 观察构建：记忆网络健康度 → 策略输入
  // -------------------------------------------------------------------------

  buildObservation(memoryNet, endocrine) {
    const obs = new Float64Array(OBS_SIZE)
    const eff = endocrine.effects()
    const m = memoryNet
    obs[0] = Math.min(1, m.predErrorEma / 2)
    const errDelta = m.predErrorEma - this.lastPredError
    obs[1] = Math.max(-1, Math.min(1, errDelta / 0.1))
    for (let i = 0; i < 3 && i < m.net.gradNormEma.length; i++) {
      obs[2 + i] = Math.min(1, m.net.gradNormEma[i] / 0.5)
    }
    const sizes = m.net.layerSizes
    const baseSizes = [204, 128, 332]
    for (let i = 0; i < 3; i++) {
      obs[5 + i] = Math.log((sizes[i] ?? baseSizes[i]) + 1) / Math.log(600)
    }
    for (let i = 0; i < MOLECULES.length; i++) obs[8 + i] = endocrine.levels[i]
    obs[23] = eff.learningRateMod / 2
    obs[24] = eff.exploration
    obs[25] = eff.arousal
    obs[26] = (eff.valence + 1) / 2
    obs[27] = Math.min(1, m.observeCount / 10000)
    return obs
  }

  // -------------------------------------------------------------------------
  // 决策：结构管理 + 参数调制
  // -------------------------------------------------------------------------

  /**
   * 周期决策（每 N 个感知调用一次）：
   *   1. 策略网络给出控制建议
   *   2. 保守规则门槛执行增/删神经元与学习率调制
   *   3. 记录 (state, action) 供奖励学习
   * @returns 决策报告
   */
  decide(memoryNet, endocrine, opts = {}) {
    this.tick++
    const obs = this.buildObservation(memoryNet, endocrine)
    const fwd = this.policy.forward(obs)
    const action = fwd.post[fwd.post.length - 1] // 16 维 sigmoid

    this.lastObs = obs
    this.lastAction = new Float64Array(action)

    const report = {
      tick: this.tick,
      growL1Prob: Number(action[ACT.growL1].toFixed(3)),
      growL2Prob: Number(action[ACT.growL2].toFixed(3)),
      pruneProb: Number(action[ACT.prune].toFixed(3)),
      lrMod: Number((0.2 + action[ACT.lrMod] * 1.8).toFixed(3)),
      molecules: {},
      grew: null,
      pruned: null,
    }

    // ---- 结构操作（保守门槛）----
    const cooldown = this.tick - this.lastStructureChangeAt
    if (cooldown >= 30 && memoryNet.predErrorEma > 0.15) {
      const maxSizes = [600, 400, 0]
      if (action[ACT.growL1] > 0.65 && memoryNet.layerSizes[0] < maxSizes[0]) {
        const r = memoryNet.growNeuron(0, { mutation: 0.2 })
        if (r) {
          this.growCount++
          this.lastStructureChangeAt = this.tick
          report.grew = { layer: 0, ...r }
        }
      } else if (action[ACT.growL2] > 0.65 && memoryNet.layerSizes[1] < maxSizes[1]) {
        const r = memoryNet.growNeuron(1, { mutation: 0.2 })
        if (r) {
          this.growCount++
          this.lastStructureChangeAt = this.tick
          report.grew = { layer: 1, ...r }
        }
      }
    }
    if (!report.grew && cooldown >= 50 && action[ACT.prune] > 0.7) {
      const cand1 = memoryNet.findPrunable(0, { usageThreshold: 0.003 })
      const cand2 = memoryNet.findPrunable(1, { usageThreshold: 0.003 })
      if (cand1) {
        const r = memoryNet.pruneNeuron(0, cand1.index)
        if (r) {
          this.pruneCount++
          this.lastStructureChangeAt = this.tick
          report.pruned = { layer: 0, ...r, usage: cand1.usage }
        }
      } else if (cand2) {
        const r = memoryNet.pruneNeuron(1, cand2.index)
        if (r) {
          this.pruneCount++
          this.lastStructureChangeAt = this.tick
          report.pruned = { layer: 1, ...r, usage: cand2.usage }
        }
      }
    }

    // ---- 内分泌建议（策略网络的微调释放：以 0.5 为基线，只有策略明确倾向释放时才释放）----
    for (let i = 0; i < MOLECULES.length; i++) {
      const amount = Math.max(0, action[ACT.molecules[0] + i] - 0.5) * 0.2
      if (amount > 0.01) {
        endocrine.release(MOLECULES[i].id, amount)
        report.molecules[MOLECULES[i].id] = Number(amount.toFixed(3))
      }
    }

    this.decisionCount++
    return report
  }

  /**
   * 元学习：奖励 = 记忆网络预测误差的下降（+ 轻微探索奖励）。
   * 简化策略梯度（高斯策略近似）：ΔW ∝ reward · (a − μ) · s。
   * 控制器由此真实地学会"怎样的操控能让记忆网络学得更好"。
   */
  learnFromOutcome(memoryNet, opts = {}) {
    if (!this.lastObs || !this.lastAction) return { reward: 0 }
    const errDelta = this.lastPredError - memoryNet.predErrorEma
    // 误差下降为正奖励；探索奖励防止策略过早停滞；
    // 分子释放有代价（防策略学会"持续高释放"）
    let reward = Math.max(-1, Math.min(1, errDelta / 0.05))
    reward += (this.lastAction[ACT.growL1] + this.lastAction[ACT.growL2]) * 0.02
    let releaseSum = 0
    for (let i = 0; i < MOLECULES.length; i++) releaseSum += this.lastAction[ACT.molecules[0] + i]
    reward -= (releaseSum / MOLECULES.length) * 0.1
    reward = Math.max(-1, Math.min(1, reward))
    this.lastPredError = memoryNet.predErrorEma
    this.lastReward = reward
    this.rewardEma = this.rewardEma * 0.95 + 0.05 * reward

    // 策略梯度更新（只更新隐藏层与输出层之间的 W，布局 out×in）
    const outLayer = this.policy.layers[1]
    const mu = 0.5 // sigmoid 基准
    const s = this.policy._lastForward.post[1] // 隐藏层激活（64）
    const lr = this.policyLr
    for (let j = 0; j < outLayer.size; j++) {
      const off = j * s.length
      const adv = reward * (this.lastAction[j] - mu)
      for (let i = 0; i < s.length; i++) {
        outLayer.W.data[off + i] += lr * adv * s[i]
      }
    }
    for (let j = 0; j < outLayer.size; j++) {
      outLayer.b[j] += lr * reward * (this.lastAction[j] - mu)
    }
    return { reward, rewardEma: this.rewardEma }
  }

  /** 当前建议的学习率调制（供 brain 训练时使用）。 */
  get lrMod() {
    if (!this.lastAction) return 1
    return 0.2 + this.lastAction[ACT.lrMod] * 1.8
  }

  // -------------------------------------------------------------------------
  // 序列化
  // -------------------------------------------------------------------------

  toJSON() {
    return {
      seed: this.seed,
      baseLr: this.baseLr,
      policyLr: this.policyLr,
      encoder: this.encoder.toJSON(),
      policy: this.policy.toJSON(),
      tick: this.tick,
      decisionCount: this.decisionCount,
      lastPredError: this.lastPredError,
      lastReward: this.lastReward,
      rewardEma: this.rewardEma,
      growCount: this.growCount,
      pruneCount: this.pruneCount,
      lastStructureChangeAt: this.lastStructureChangeAt,
    }
  }

  static fromJSON(json) {
    const c = Object.create(ControllerNet.prototype)
    c.seed = json.seed ?? 7
    c.baseLr = json.baseLr ?? 0.008
    c.policyLr = json.policyLr ?? 0.01
    c.encoder = new SenseEncoder(c.seed * 31 + 1)
    c.encoder.fromJSON(json.encoder ?? {})
    c.policy = PlasticNetwork.fromJSON(json.policy)
    c.tick = json.tick ?? 0
    c.decisionCount = json.decisionCount ?? 0
    c.lastPredError = json.lastPredError ?? 1
    c.lastReward = json.lastReward ?? 0
    c.rewardEma = json.rewardEma ?? 0
    c.growCount = json.growCount ?? 0
    c.pruneCount = json.pruneCount ?? 0
    c.lastStructureChangeAt = json.lastStructureChangeAt ?? -999
    c.lastObs = null
    c.lastAction = null
    return c
  }

  status() {
    return {
      tick: this.tick,
      decisionCount: this.decisionCount,
      growCount: this.growCount,
      pruneCount: this.pruneCount,
      rewardEma: this.rewardEma,
      lastReward: this.lastReward,
      lrMod: this.lrMod,
      encoderLoss: this.encoder.recentLoss,
      encoderTrained: this.encoder.trained,
      policyParams: this.policy.paramCount,
    }
  }
}
