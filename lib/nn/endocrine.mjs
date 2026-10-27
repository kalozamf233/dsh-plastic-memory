// ============================================================================
// dsh-plastic-memory —— 内分泌系统（EndocrineSystem）
// ----------------------------------------------------------------------------
// 参考《人性矩阵·矩阵之外03：潜意识与支配我们的15种分子》：
// 15 种分子浓度状态（0..1），随时间指数衰减回基线；事件触发释放。
// 分子状态对记忆网络施加真实调制（学习率/激活增益/探索/情绪目标），
// 而非作为提示词标签存在。
// ============================================================================

export const MOLECULES = [
  { id: 'dopamine', name: '多巴胺', baseline: 0.22, halfLife: 90 },
  { id: 'noradrenaline', name: '去甲肾上腺素', baseline: 0.18, halfLife: 60 },
  { id: 'serotonin', name: '血清素', baseline: 0.30, halfLife: 300 },
  { id: 'cortisol', name: '皮质醇', baseline: 0.15, halfLife: 600 },
  { id: 'adrenaline', name: '肾上腺素', baseline: 0.10, halfLife: 45 },
  { id: 'oxytocin', name: '催产素', baseline: 0.25, halfLife: 120 },
  { id: 'testosterone', name: '睾酮', baseline: 0.20, halfLife: 240 },
  { id: 'estrogen', name: '雌激素', baseline: 0.20, halfLife: 600 },
  { id: 'progesterone', name: '孕激素', baseline: 0.18, halfLife: 600 },
  { id: 'endorphin', name: '内啡肽', baseline: 0.20, halfLife: 90 },
  { id: 'acetylcholine', name: '乙酰胆碱', baseline: 0.22, halfLife: 30 },
  { id: 'gaba', name: 'GABA', baseline: 0.25, halfLife: 60 },
  { id: 'glutamate', name: '谷氨酸', baseline: 0.25, halfLife: 30 },
  { id: 'melatonin', name: '褪黑素', baseline: 0.12, halfLife: 120 },
  { id: 'prolactin', name: '催乳素', baseline: 0.15, halfLife: 300 },
]

export const MOLECULE_INDEX = Object.fromEntries(MOLECULES.map((m, i) => [m.id, i]))

export class EndocrineSystem {
  constructor(seedState = null) {
    this.levels = new Float64Array(MOLECULES.length)
    if (seedState) {
      this.fromJSON(seedState)
    } else {
      for (let i = 0; i < MOLECULES.length; i++) this.levels[i] = MOLECULES[i].baseline
    }
  }

  get(id) {
    const i = MOLECULE_INDEX[id]
    return i === undefined ? 0 : this.levels[i]
  }

  set(id, value) {
    const i = MOLECULE_INDEX[id]
    if (i === undefined) return false
    this.levels[i] = Math.max(0, Math.min(1, value))
    return true
  }

  /** 释放：浓度向 1 靠近 amount 比例（0..1）。 */
  release(id, amount) {
    const i = MOLECULE_INDEX[id]
    if (i === undefined) return false
    const v = this.levels[i]
    this.levels[i] = Math.max(0, Math.min(1, v + (1 - v) * amount))
    return true
  }

  /** 批量释放：{moleculeId: amount}。 */
  releaseMany(map) {
    for (const [id, amount] of Object.entries(map)) this.release(id, amount)
  }

  /** 时间推进：指数衰减回基线。 */
  tick(dtSeconds) {
    for (let i = 0; i < MOLECULES.length; i++) {
      const k = Math.exp(-dtSeconds / MOLECULES[i].halfLife)
      this.levels[i] = MOLECULES[i].baseline + (this.levels[i] - MOLECULES[i].baseline) * k
    }
  }

  // -------------------------------------------------------------------------
  // 效应映射：分子浓度 → 对网络的真实调制参数
  // -------------------------------------------------------------------------

  /**
   * 计算当前分子状态对记忆网络与操控网络的调制。
   * 返回：
   *   learningRateMod  全局学习率乘数（皮质醇/去甲肾上腺素/谷氨酸升，GABA/褪黑素降）
   *   exploration      结构探索度（多巴胺升 → 操控网络更倾向新增神经元）
   *   arousal          唤醒度 0..1（急性应激/兴奋）
   *   valence          效价 -1..1（愉悦↔痛苦）
   *   socialAxis       社会轴 -1..1（催产素亲和 ↔ 睾酮竞争）
   *   memorySalience   记忆显著性（高唤醒事件记得更深）
   *   emotionTarget    期望情绪目标向量（情绪构建论的生理输入）
   */
  effects() {
    const L = this.levels
    const g = (id) => L[MOLECULE_INDEX[id]]

    const learningRateMod =
      1 +
      g('cortisol') * 0.5 +
      g('noradrenaline') * 0.4 +
      g('glutamate') * 0.25 +
      g('acetylcholine') * 0.15 -
      g('gaba') * 0.3 -
      g('melatonin') * 0.35

    const exploration =
      Math.max(0, Math.min(1, g('dopamine') * 0.8 + g('glutamate') * 0.35 - g('serotonin') * 0.45))

    const arousal = Math.max(
      0,
      Math.min(
        1,
        0.35 * g('noradrenaline') +
          0.35 * g('adrenaline') +
          0.2 * g('cortisol') +
          0.15 * g('glutamate') -
          0.3 * g('melatonin') -
          0.2 * g('gaba'),
      ),
    )

    const valence = Math.max(
      -1,
      Math.min(
        1,
        0.4 * g('dopamine') +
          0.35 * g('endorphin') +
          0.3 * g('oxytocin') +
          0.2 * g('serotonin') -
          0.45 * g('cortisol') -
          0.3 * g('adrenaline'),
      ),
    )

    const socialAxis = Math.max(-1, Math.min(1, g('oxytocin') - g('testosterone')))

    const memorySalience = 0.5 + 0.5 * arousal

    // 情绪目标（构建情绪 12 维的目标输入，见 memorynet EMOTION_DIMS）：
    // 维序 [valence, arousal, joy, sadness, anger, fear, disgust, surprise, calm, anxiety, trust, love]
    const joy = Math.max(0, Math.min(1, 0.6 * g('dopamine') + 0.4 * g('endorphin')))
    const sadness = Math.max(0, Math.min(1, 0.6 * (1 - g('dopamine')) + 0.4 * g('prolactin')))
    const anger = Math.max(0, Math.min(1, 0.5 * g('testosterone') + 0.3 * g('adrenaline') - 0.4 * g('serotonin')))
    const fear = Math.max(0, Math.min(1, 0.55 * g('cortisol') + 0.45 * g('adrenaline')))
    const disgust = Math.max(0, Math.min(1, 0.4 * (1 - g('serotonin')) + 0.2 * g('cortisol')))
    const surprise = Math.max(0, Math.min(1, 0.5 * g('adrenaline') + 0.5 * g('glutamate')))
    const calm = Math.max(0, Math.min(1, 0.5 * g('gaba') + 0.3 * g('serotonin') + 0.2 * g('melatonin')))
    const anxiety = Math.max(0, Math.min(1, 0.7 * g('cortisol') + 0.2 * g('noradrenaline')))
    const trust = Math.max(0, Math.min(1, 0.6 * g('oxytocin') + 0.2 * g('estrogen') + 0.2 * g('serotonin')))
    const love = Math.max(0, Math.min(1, 0.5 * g('oxytocin') + 0.3 * g('prolactin') + 0.2 * g('endorphin')))
    const emotionTarget = new Float64Array([
      (valence + 1) / 2, arousal, joy, sadness, anger, fear, disgust, surprise, calm, anxiety, trust, love,
    ])

    return { learningRateMod, exploration, arousal, valence, socialAxis, memorySalience, emotionTarget }
  }

  // -------------------------------------------------------------------------
  // 序列化
  // -------------------------------------------------------------------------

  toJSON() {
    return Array.from(this.levels)
  }

  fromJSON(arr) {
    for (let i = 0; i < MOLECULES.length && i < arr.length; i++) {
      this.levels[i] = Math.max(0, Math.min(1, arr[i]))
    }
  }

  /** 人类可读快照。 */
  snapshot() {
    const out = {}
    for (let i = 0; i < MOLECULES.length; i++) {
      out[MOLECULES[i].id] = {
        name: MOLECULES[i].name,
        level: Number(this.levels[i].toFixed(3)),
        baseline: MOLECULES[i].baseline,
      }
    }
    return out
  }
}
