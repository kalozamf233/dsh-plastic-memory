// ============================================================================
// 神格（Godhood）——控制「世界」的权限系统
// 等级：
//   0 凡人（mortal）：只能被动感知，不能发出任何行为请求
//   1 观察者（observer）：只能旁观世界、记录故事，无法插手（不能 act）
//   2 干涉者（actor）：可在世界中行动（移动/拿取/吃/跳）
//   3 神（god）：除行动外，可创造物品、修改世界（天气/地形/时间）
// 王楚门与创世者签订契约后神格为「观察者」——旁观并记录，在我创造的
// 崭新空白世界中再现故事。
// ============================================================================

export const GODHOOD_TIERS = {
  mortal: 0,
  observer: 1,
  actor: 2,
  god: 3,
}

export const GODHOOD_NAMES = ['凡人', '观察者', '干涉者', '神']

/** 某神格可执行的动作集合（世界行为）。 */
export function actionsAllowed(tier) {
  if (tier <= GODHOOD_TIERS.mortal) return []
  if (tier <= GODHOOD_TIERS.observer) return [] // 观察者：无法插手
  const acts = ['move', 'jump', 'grab', 'eat', 'drop', 'rest']
  return acts
}

/** 世界管理能力（仅神）。 */
export function godPowersAllowed(tier) {
  return tier >= GODHOOD_TIERS.god
    ? ['spawnItem', 'removeItem', 'setWeather', 'setTime', 'listItems']
    : []
}

/** 观察者职责（神格 1 的契约义务）。 */
export function observerDuties(tier) {
  if (tier <= GODHOOD_TIERS.mortal) return []
  if (tier >= GODHOOD_TIERS.god) return ['record', 'reproduce'] // 神也保留记录能力
  if (tier === GODHOOD_TIERS.observer) return ['record', 'reproduce']
  return []
}

/** 校验一个世界行为：返回 {ok, error}。 */
export function checkWorldAction(tier, action) {
  if (tier <= GODHOOD_TIERS.mortal) {
    return { ok: false, error: '你没有神格，无法与世界交互' }
  }
  if (tier === GODHOOD_TIERS.observer) {
    return { ok: false, error: '你已签订契约：只能旁观，无法插手这个世界' }
  }
  if (!actionsAllowed(tier).includes(action)) {
    return { ok: false, error: `当前神格（${GODHOOD_NAMES[tier]}）不允许动作 ${action}` }
  }
  return { ok: true }
}

/** 校验世界管理能力：返回 {ok, error}。 */
export function checkGodPower(tier, power) {
  if (tier < GODHOOD_TIERS.god) {
    return { ok: false, error: '只有「神」格才能修改世界' }
  }
  if (!godPowersAllowed(tier).includes(power)) {
    return { ok: false, error: `未知的神力：${power}` }
  }
  return { ok: true }
}
