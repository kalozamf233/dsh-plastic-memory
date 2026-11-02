// ============================================================================
// 身体状态：空间坐标（xyz）、朝向、能量、痛感、饱腹、手持物
// 出生自带坐标（世界中心地面）；所有身体行为由身体数据网络输出驱动，
// 世界负责物理校验（重力/步长/交互距离）。
// ============================================================================

export const BIRTH_POS = { x: 20, y: 20, z: 0 }
export const DIRS = ['n', 's', 'e', 'w']

const DIR_DELTA = {
  n: { x: 0, y: -1 },
  s: { x: 0, y: 1 },
  e: { x: 1, y: 0 },
  w: { x: -1, y: 0 },
}

export class Body {
  constructor(seed = 0) {
    this.pos = { ...BIRTH_POS } // 世界内坐标（以世界返回为准）
    this.dir = 'n' // 朝向
    this.energy = 0.8 // 能量 0..1
    this.pain = 0 // 痛感 0..1
    this.satiety = 0.5 // 饱腹 0..1
    this.heldItem = null // 手持物 id
    this.speed = 0 // 最近一次移动步数
    this.steps = 0 // 累计步数
    this.eats = 0 // 累计进食次数
    this.bornAt = Date.now()
    this.seed = seed
  }

  /** 朝向向量（世界坐标 y 轴方向约定：n=-y, s=+y, e=+x, w=-x）。 */
  get dirDelta() {
    return DIR_DELTA[this.dir] ?? { x: 0, y: 0 }
  }

  /** 归一化身体状态 8 维（供身体数据网络输入）。 */
  encode() {
    return new Float64Array([
      this.pos.x / 40,
      this.pos.y / 40,
      this.pos.z / 8,
      Math.max(0, DIRS.indexOf(this.dir)) / 3,
      clamp01(this.energy),
      clamp01(this.pain),
      clamp01(this.satiety),
      clamp01(Math.min(1, this.speed / 5)),
    ])
  }

  /** 更新位置（世界返回的 newPos）。 */
  applyPos(newPos) {
    if (newPos && Number.isFinite(newPos.x) && Number.isFinite(newPos.y) && Number.isFinite(newPos.z)) {
      this.pos = { x: Math.round(newPos.x), y: Math.round(newPos.y), z: Math.round(newPos.z) }
    }
    return this.pos
  }

  applyEffect(effects = {}) {
    if (typeof effects.energy === 'number') this.energy = clamp01(this.energy + effects.energy)
    if (typeof effects.pain === 'number') this.pain = clamp01(this.pain + effects.pain)
    if (typeof effects.satiety === 'number') this.satiety = clamp01(this.satiety + effects.satiety)
  }

  /** 每秒代谢：能量缓慢消耗、痛感自然消退、饱腹缓慢下降。 */
  tick(dtSeconds = 1) {
    this.energy = clamp01(this.energy - 0.002 * dtSeconds)
    this.pain = clamp01(this.pain - 0.01 * dtSeconds)
    this.satiety = clamp01(this.satiety - 0.001 * dtSeconds)
    this.speed = 0
  }

  toJSON() {
    return {
      pos: this.pos,
      dir: this.dir,
      energy: this.energy,
      pain: this.pain,
      satiety: this.satiety,
      heldItem: this.heldItem,
      steps: this.steps,
      eats: this.eats,
      bornAt: this.bornAt,
      seed: this.seed,
    }
  }

  static fromJSON(data) {
    const b = new Body()
    if (data && typeof data === 'object') {
      b.pos = { ...BIRTH_POS, ...(data.pos ?? {}) }
      b.dir = DIRS.includes(data.dir) ? data.dir : 'n'
      b.energy = clamp01(Number(data.energy ?? 0.8))
      b.pain = clamp01(Number(data.pain ?? 0))
      b.satiety = clamp01(Number(data.satiety ?? 0.5))
      b.heldItem = typeof data.heldItem === 'string' ? data.heldItem : null
      b.steps = Number(data.steps ?? 0)
      b.eats = Number(data.eats ?? 0)
      b.bornAt = Number(data.bornAt ?? Date.now())
      b.seed = Number(data.seed ?? 0)
    }
    return b
  }
}

function clamp01(v) {
  if (!Number.isFinite(v)) return 0
  return Math.max(0, Math.min(1, v))
}
