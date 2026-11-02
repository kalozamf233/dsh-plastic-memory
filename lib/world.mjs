// ============================================================================
// dsh-plastic-memory —— 世界核心模拟器（World）
// ----------------------------------------------------------------------------
// 独立于数字生命进程的"固定世界"：地图 40×40（x,y）+ 垂直轴 z∈[0,8]。
//   · 物品固定摆放：seed 决定布局（蛋糕×2 距出生点 6-10、花朵×3、臭泥×2、
//     果树×2、水×1、石头×3，共 13 件），摆放一次后保持不变
//   · 物理 tick（每秒）：重力坠落 + 事件引擎（每 20-40 秒随机世界反馈事件：
//     石头坠落（可砸中）/ 花香飘散 / 鸟鸣）
//   · 常驻感知 sense()：六感（视觉=四向正向 90° 扇形；嗅觉/听觉=全向；触觉=接触）
//   · 行为交互 act()：move 步长 1 / jump +2（上限 8）/ grab 距离 ≤2 /
//     eat（heldBy 且 edible → 效果）/ drop；每次 act 先做重力校正
//   · 世界反馈 eventsFor(entityId, sinceTs)：主动事件队列（ts 过滤幂等领取）
// 实体位置由世界侧维护（pos 以世界为准，caller 的 pos 仅做步长一致性校验）。
// 零外部依赖（node:fs / node:path + lib/nn/core.mjs 的 mulberry32）。
// ============================================================================

import { mulberry32 } from './nn/core.mjs'
import { mkdirSync, writeFileSync, renameSync } from 'node:fs'
import { dirname } from 'node:path'

// ---------------------------------------------------------------------------
// 常量与物品定义
// ---------------------------------------------------------------------------

export const MAP_SIZE = 40
export const SPAWN = { x: 20, y: 20, z: 0 }
export const MAX_Z = 8
export const DIRS = ['n', 's', 'e', 'w']
export const DIR_VEC = { n: { x: 0, y: -1 }, s: { x: 0, y: 1 }, e: { x: 1, y: 0 }, w: { x: -1, y: 0 } }
export const DIR_NAME = { n: '北方', s: '南方', e: '东方', w: '西方' }

export const ITEM_DEFS = {
  cake:   { name: '蛋糕', sight: '一块蛋糕', smell: '香甜', sound: null, taste: '甜', edible: true },
  flower: { name: '花朵', sight: '一朵花',   smell: '花香', sound: '鸟鸣', taste: null, edible: false },
  rotten: { name: '臭泥', sight: '一滩臭泥', smell: '腐臭', sound: null, taste: null, edible: false },
  stone:  { name: '石头', sight: '一块石头', smell: null,   sound: null, taste: null, edible: false },
  fruit:  { name: '果树', sight: '一棵果树', smell: null,   sound: null, taste: '甜', edible: true },
  water:  { name: '水',   sight: '一汪清水', smell: null,   sound: null, taste: '淡', edible: true },
}

/** 固定布局计划（seed 决定具体坐标）。 */
const LAYOUT_PLAN = [
  { type: 'cake', around: SPAWN, distMin: 6, distMax: 10 },
  { type: 'cake', around: SPAWN, distMin: 6, distMax: 10 },
  { type: 'flower' },
  { type: 'flower' },
  { type: 'flower' },
  { type: 'rotten' },
  { type: 'rotten' },
  { type: 'fruit' },
  { type: 'fruit' },
  { type: 'water' },
  { type: 'stone' },
  { type: 'stone' },
  { type: 'stone' },
]

/** seed 决定物品摆放（确定性，可复现）。 */
function buildLayout(seed, mapSize) {
  const rng = mulberry32(seed >>> 0)
  const items = []
  const used = new Set([`${SPAWN.x},${SPAWN.y}`]) // 出生点不放物品
  let seq = 0
  const push = (type, x, y) => {
    seq++
    items.push({ id: `item-${type}-${seq}`, type, ...ITEM_DEFS[type], pos: { x, y, z: 0 }, radius: 1, heldBy: null })
    used.add(`${x},${y}`)
  }
  for (const plan of LAYOUT_PLAN) {
    let placed = false
    for (let attempt = 0; attempt < 80 && !placed; attempt++) {
      let x
      let y
      if (plan.around) {
        // 蛋糕：距出生点 8/9 格（取整误差 ≤1 后仍落在 6..10 契约区间）
        const ang = rng() * Math.PI * 2
        const dist = 8 + (rng() < 0.5 ? 0 : 1)
        x = Math.round(plan.around.x + Math.cos(ang) * dist)
        y = Math.round(plan.around.y + Math.sin(ang) * dist)
      } else {
        x = Math.floor(rng() * mapSize)
        y = Math.floor(rng() * mapSize)
      }
      if (x < 0 || x >= mapSize || y < 0 || y >= mapSize) continue
      if (used.has(`${x},${y}`)) continue
      push(plan.type, x, y)
      placed = true
    }
    if (!placed) {
      // 兜底：确定性全图扫描空位
      for (let y = 0; y < mapSize && !placed; y++) {
        for (let x = 0; x < mapSize; x++) {
          if (used.has(`${x},${y}`)) continue
          push(plan.type, x, y)
          placed = true
          break
        }
      }
    }
  }
  return items
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v))
}

function dist3(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
}

// ============================================================================
// World
// ============================================================================

export class World {
  constructor({ seed = 20261102, mapSize = MAP_SIZE } = {}) {
    this.seed = seed >>> 0
    this.mapSize = mapSize
    this.items = buildLayout(this.seed, mapSize)
    this.entities = new Map() // id → {id, pos:{x,y,z}, dir, held}
    this._events = new Map() // id → 主动反馈事件数组
    this.worldEvents = [] // 全局事件日志（观察界面用）
    this.tickCount = 0
    this.rng = mulberry32((this.seed ^ 0x9e3779b9) >>> 0)
    this.nextEventIn = this._rollEventDelay()
    this._lastEventTs = 0 // 事件时钟：保证事件 ts 严格递增（轮询 since 过滤不丢同毫秒事件）
  }

  _rollEventDelay() {
    return 20 + Math.floor(this.rng() * 21) // 20..40 秒
  }

  /** 严格递增的事件时间戳（≥ Date.now() 且恒大于上一事件）。 */
  _eventNow() {
    const now = Date.now()
    const ts = Math.max(now, this._lastEventTs + 1)
    this._lastEventTs = ts
    return ts
  }

  // -------------------------------------------------------------------------
  // 实体登记（世界侧为准）
  // -------------------------------------------------------------------------

  _clampPos(pos = {}) {
    return {
      x: clamp(Math.floor(Number(pos.x ?? SPAWN.x)), 0, this.mapSize - 1),
      y: clamp(Math.floor(Number(pos.y ?? SPAWN.y)), 0, this.mapSize - 1),
      z: clamp(Math.floor(Number(pos.z ?? 0)), 0, MAX_Z),
    }
  }

  registerEntity(id, pos = {}) {
    const p = this._clampPos(pos)
    const existing = this.entities.get(id)
    if (existing) {
      existing.pos = p // 恢复/同步位置
      return { ok: true, entity: this._entityView(existing) }
    }
    const e = { id, pos: p, dir: 'n', held: null }
    this.entities.set(id, e)
    return { ok: true, entity: this._entityView(e) }
  }

  _entityView(e) {
    return { id: e.id, pos: { ...e.pos }, dir: e.dir, held: e.held }
  }

  // -------------------------------------------------------------------------
  // 物理 tick（每秒调用）
  // -------------------------------------------------------------------------

  tick() {
    this.tickCount++
    // 重力：所有实体 z>0 则 z-1（坠落）
    for (const e of this.entities.values()) {
      if (e.pos.z > 0) e.pos.z -= 1
    }
    // 事件引擎：每 20-40 秒随机一个世界反馈事件
    this.nextEventIn -= 1
    if (this.nextEventIn <= 0) {
      this._generateRandomEvent()
      this.nextEventIn = this._rollEventDelay()
    }
    return this.tickCount
  }

  _generateRandomEvent() {
    const roll = this.rng()
    if (roll < 0.45) return this._spawnStoneFall()
    if (roll < 0.75) return this._spawnSmellDrift()
    return this._spawnBirdSing()
  }

  // ---- 事件生成（内部方法；_spawnStoneFall 支持显式落点，供自测/定向使用）----

  _spawnStoneFall(opts = {}) {
    let lx
    let ly
    if (Number.isInteger(opts.x) && Number.isInteger(opts.y)) {
      lx = opts.x
      ly = opts.y
    } else {
      const ents = Array.from(this.entities.values())
      if (ents.length > 0 && this.rng() < 0.5) {
        // 一半概率瞄准某个实体附近，让"砸中"真实可发生
        const t = ents[Math.floor(this.rng() * ents.length)]
        lx = clamp(t.pos.x + Math.floor(this.rng() * 3) - 1, 0, this.mapSize - 1)
        ly = clamp(t.pos.y + Math.floor(this.rng() * 3) - 1, 0, this.mapSize - 1)
      } else {
        lx = Math.floor(this.rng() * this.mapSize)
        ly = Math.floor(this.rng() * this.mapSize)
      }
    }
    const ts = this._eventNow()
    let hit = false
    for (const e of this.entities.values()) {
      // 水平切比雪夫距离 ≤1 且 z≤1 → 砸中（跳起来 z≥2 可躲开）
      const hd = Math.max(Math.abs(e.pos.x - lx), Math.abs(e.pos.y - ly))
      if (hd <= 1 && e.pos.z <= 1) {
        this._pushEvent(e.id, {
          ts, type: 'stone_hit', text: '一颗石子掉下来砸到了你', pain: 0.6,
          sights: ['石子掉下来'], sounds: ['石子掉下来的声音'],
        })
        hit = true
      } else if (Math.hypot(e.pos.x - lx, e.pos.y - ly) <= 8) {
        this._pushEvent(e.id, { ts, type: 'stone_fall', text: '你看到一颗石子掉在不远处', sights: ['石子掉在不远处'] })
      }
    }
    this._logWorld({
      ts,
      type: hit ? 'stone_hit' : 'stone_fall',
      text: hit ? `一颗石子砸中了某个生命（${lx},${ly}）` : `一颗石子坠落在（${lx},${ly}）`,
      at: { x: lx, y: ly },
    })
    return { type: 'stone_fall', at: { x: lx, y: ly }, hit }
  }

  _spawnSmellDrift() {
    const candidates = this.items.filter((it) => it.smell && it.heldBy == null)
    if (candidates.length === 0) return null
    const item = candidates[Math.floor(this.rng() * candidates.length)]
    const ts = this._eventNow()
    for (const e of this.entities.values()) {
      if (Math.hypot(e.pos.x - item.pos.x, e.pos.y - item.pos.y) <= 8) {
        this._pushEvent(e.id, { ts, type: 'smell_drift', text: `一阵${item.smell}的气味飘了过来`, smells: [item.smell] })
      }
    }
    this._logWorld({ ts, type: 'smell_drift', text: `一阵${item.smell}的气味从（${item.pos.x},${item.pos.y}）飘散`, at: { x: item.pos.x, y: item.pos.y } })
    return { type: 'smell_drift' }
  }

  _spawnBirdSing() {
    const ts = this._eventNow()
    for (const e of this.entities.values()) {
      this._pushEvent(e.id, { ts, type: 'bird_sing', text: '远处传来鸟鸣', sounds: ['鸟鸣'] })
    }
    this._logWorld({ ts, type: 'bird_sing', text: '远处传来鸟鸣' })
    return { type: 'bird_sing' }
  }

  _pushEvent(entityId, event) {
    let list = this._events.get(entityId)
    if (!list) {
      list = []
      this._events.set(entityId, list)
    }
    list.push(event)
    if (list.length > 200) list.splice(0, list.length - 200)
  }

  _logWorld(event) {
    this.worldEvents.push(event)
    if (this.worldEvents.length > 200) this.worldEvents.splice(0, this.worldEvents.length - 200)
  }

  /** 世界主动反馈事件：返回 ts > sinceTs 的事件（不删除，ts 过滤幂等领取）。 */
  eventsFor(entityId, sinceTs = 0) {
    const list = this._events.get(entityId) ?? []
    return list.filter((ev) => ev.ts > sinceTs)
  }

  // -------------------------------------------------------------------------
  // 常驻感知（六感）
  // -------------------------------------------------------------------------

  /**
   * 六感感知：按坐标 + 四向过滤物品。
   * 视觉 = 正向 90° 扇形（range 默认 5）；嗅觉/听觉 = 全向（range 内）；
   * 触觉 = 3D 接触（距离 ≤1.5）。每项含相对方位与距离。
   * senses 可选值：'sight' | 'hearing' | 'smell' | 'touch'。
   */
  sense(entityId, { pos, dir = 'n', range = 5, senses = ['sight', 'hearing', 'smell', 'touch'] } = {}) {
    const stored = this.entities.get(entityId)
    const p = stored ? stored.pos : (pos ? this._clampPos(pos) : null)
    if (!p) return { sights: [], sounds: [], smells: [], touches: [] }
    const f = DIRS.includes(dir) ? dir : 'n'
    const rngDist = Number.isFinite(Number(range)) ? Number(range) : 5
    const senseSet = new Set(Array.isArray(senses) ? senses : [])
    const out = { sights: [], sounds: [], smells: [], touches: [] }
    for (const item of this.items) {
      if (item.heldBy != null) continue // 被拿起的物品不在世界中呈现
      const dx = item.pos.x - p.x
      const dy = item.pos.y - p.y
      const dz = item.pos.z - p.z
      const dist2d = Math.hypot(dx, dy)
      const dirRel = this._relDir(dx, dy)
      const grid = Math.max(0, Math.round(dist2d))
      // 触觉：接触（3D 距离 ≤1.5）
      if (senseSet.has('touch') && Math.hypot(dx, dy, dz) <= 1.5) {
        out.touches.push({
          itemId: item.id, type: item.type, name: item.name, dir: dirRel, dist: Number(dist2d.toFixed(1)),
          text: `你碰到了${item.sight}`,
        })
      }
      if (dist2d > rngDist) continue
      // 视觉：正向 90° 扇形
      if (senseSet.has('sight') && this._inFan(f, dx, dy)) {
        out.sights.push({
          itemId: item.id, type: item.type, name: item.name, dir: dirRel, dist: Number(dist2d.toFixed(1)),
          text: grid === 0 ? `你看到${item.sight}，就在你身边` : `你看到${item.sight}，在${DIR_NAME[dirRel]}约${grid}格处`,
        })
      }
      // 嗅觉：全向
      if (senseSet.has('smell') && item.smell) {
        out.smells.push({
          itemId: item.id, type: item.type, name: item.name, dir: dirRel, dist: Number(dist2d.toFixed(1)),
          text: grid === 0 ? `你闻到${item.smell}的气味，就在你身边` : `你闻到${item.smell}的气味，来自${DIR_NAME[dirRel]}约${grid}格处`,
        })
      }
      // 听觉：全向
      if (senseSet.has('hearing') && item.sound) {
        out.sounds.push({
          itemId: item.id, type: item.type, name: item.name, dir: dirRel, dist: Number(dist2d.toFixed(1)),
          text: grid === 0 ? `你听到${item.sound}，就在你身边` : `你听到${item.sound}，来自${DIR_NAME[dirRel]}约${grid}格处`,
        })
      }
    }
    return out
  }

  /** 相对方位（主导轴）：dx/dy → 'n'|'s'|'e'|'w'。 */
  _relDir(dx, dy) {
    if (Math.abs(dx) >= Math.abs(dy)) {
      if (dx > 0) return 'e'
      if (dx < 0) return 'w'
      return dy > 0 ? 's' : 'n'
    }
    return dy > 0 ? 's' : 'n'
  }

  /** 正向 90° 扇形判定。 */
  _inFan(f, dx, dy) {
    switch (f) {
      case 'n': return dy <= 0 && Math.abs(dx) <= Math.abs(dy)
      case 's': return dy >= 0 && Math.abs(dx) <= Math.abs(dy)
      case 'e': return dx >= 0 && Math.abs(dy) <= Math.abs(dx)
      case 'w': return dx <= 0 && Math.abs(dy) <= Math.abs(dx)
      default: return false
    }
  }

  // -------------------------------------------------------------------------
  // 行为交互
  // -------------------------------------------------------------------------

  /**
   * 行为交互。每次 act 先执行重力校正（z>0 则 z-1）。
   * 返回 {ok, error?, newPos, result?, effects?}（newPos 恒返回，供 caller 同步）。
   */
  act(entityId, { pos, action, dir, target } = {}) {
    // 实体位置世界侧维护：未注册则用请求 pos 兜底注册
    let e = this.entities.get(entityId)
    if (!e) {
      if (!pos) return { ok: false, error: `实体 ${entityId} 未注册` }
      this.registerEntity(entityId, pos)
      e = this.entities.get(entityId)
    }
    if (e.pos.z > 0) e.pos.z -= 1 // 重力校正
    if (DIRS.includes(dir)) e.dir = dir
    let res
    switch (action) {
      case 'move': res = this._actMove(e, pos, dir); break
      case 'jump': res = this._actJump(e); break
      case 'grab': res = this._actGrab(e, target); break
      case 'eat': res = this._actEat(e, target); break
      case 'drop': res = this._actDrop(e); break
      default: return { ok: false, error: `未知行为: ${action}`, newPos: { ...e.pos } }
    }
    return {
      ok: res.ok,
      ...(res.error ? { error: res.error } : {}),
      newPos: { ...e.pos },
      ...(res.result ? { result: res.result } : {}),
      ...(res.effects ? { effects: res.effects } : {}),
    }
  }

  _actMove(e, claimedPos, dir) {
    // 步长校验：caller 声称的位置与世界侧位置偏差 >1 → 拒绝（pos 以世界为准）
    if (
      claimedPos &&
      (Math.abs(claimedPos.x - e.pos.x) > 1 ||
        Math.abs(claimedPos.y - e.pos.y) > 1 ||
        Math.abs(Number(claimedPos.z ?? e.pos.z) - e.pos.z) > 1)
    ) {
      return { ok: false, error: '步长必须为 1：pos 与世界不同步，请以服务器返回的 newPos 为准' }
    }
    const d = DIRS.includes(dir) ? dir : e.dir
    const v = DIR_VEC[d]
    const nx = e.pos.x + v.x
    const ny = e.pos.y + v.y
    if (nx < 0 || nx >= this.mapSize || ny < 0 || ny >= this.mapSize) {
      return { ok: false, error: `越界：无法向${DIR_NAME[d]}移动（地图边缘）` }
    }
    const from = { ...e.pos }
    e.pos.x = nx
    e.pos.y = ny
    return { ok: true, result: { moved: true, dir: d, from, to: { ...e.pos } } }
  }

  _actJump(e) {
    const nz = e.pos.z + 2
    if (nz > MAX_Z) {
      return { ok: false, error: `跳得太高了：z 上限 ${MAX_Z}` }
    }
    e.pos.z = nz
    return { ok: true, result: { jumped: true, z: e.pos.z } }
  }

  _actGrab(e, target) {
    if (e.held) return { ok: false, error: '手里已经有东西了，先 drop 再拿' }
    const item = this.items.find((it) => it.id === target)
    if (!item) return { ok: false, error: '目标不存在' }
    if (item.heldBy != null) return { ok: false, error: '目标已被拿取' }
    const d = dist3(e.pos, item.pos)
    if (d > 2) return { ok: false, error: `太远了：距离 ${d.toFixed(1)} 超过 2` }
    item.heldBy = e.id
    e.held = item.id
    return {
      ok: true,
      result: { grabbed: true, item: { id: item.id, type: item.type, name: item.name, taste: item.taste, edible: item.edible } },
    }
  }

  _actEat(e, target) {
    const item = this.items.find((it) => it.id === target)
    if (!item) return { ok: false, error: '目标不存在' }
    if (item.heldBy !== e.id) return { ok: false, error: '你没有拿着这个物品' }
    if (!item.edible) return { ok: false, error: `${item.name}不能吃` }
    this.items = this.items.filter((it) => it.id !== target)
    e.held = null
    this._logWorld({ ts: this._eventNow(), type: 'eat', text: `有生命吃掉了一件物品（${item.name}）`, at: { x: item.pos.x, y: item.pos.y } })
    return {
      ok: true,
      result: { ate: true, item: { id: item.id, type: item.type, name: item.name, taste: item.taste } },
      effects: { dopamine: 0.5, endorphin: 0.4, satiety: 0.6 },
    }
  }

  _actDrop(e) {
    if (!e.held) return { ok: false, error: '手里没有东西' }
    const item = this.items.find((it) => it.id === e.held)
    if (item) item.heldBy = null // 放回原坐标（grab 未移动物品，坐标不变）
    e.held = null
    return { ok: true, result: { dropped: true } }
  }

  // -------------------------------------------------------------------------
  // 状态序列化
  // -------------------------------------------------------------------------

  /** 世界全量状态对象（JSON 可序列化；/state 与存档共用）。 */
  loadState() {
    return {
      version: 1,
      seed: this.seed,
      mapSize: this.mapSize,
      tick: this.tickCount,
      nextEventIn: this.nextEventIn,
      items: this.items.map((it) => ({ ...it, pos: { ...it.pos } })),
      entities: Object.fromEntries(
        Array.from(this.entities.entries()).map(([id, e]) => [id, this._entityView(e)]),
      ),
      events: Object.fromEntries(
        Array.from(this._events.entries()).map(([id, list]) => [id, list.slice(-200)]),
      ),
      worldEvents: this.worldEvents.slice(-200),
    }
  }

  /** 原子写状态文件（tmp + rename）。成功 true，异常 false。 */
  saveState(file) {
    try {
      const dir = dirname(file)
      if (dir) mkdirSync(dir, { recursive: true })
      const tmp = `${file}.tmp`
      writeFileSync(tmp, JSON.stringify(this.loadState(), null, 2), 'utf8')
      renameSync(tmp, file)
      return true
    } catch {
      return false
    }
  }

  /** 从状态对象恢复世界（server 启动时用）。损坏/缺字段 → null。 */
  static fromState(st) {
    if (!st || typeof st !== 'object' || !Array.isArray(st.items)) return null
    const w = new World({ seed: st.seed ?? 20261102, mapSize: st.mapSize ?? MAP_SIZE })
    w.items = st.items.map((it) => ({ ...it, pos: { ...it.pos }, heldBy: it.heldBy ?? null }))
    w.tickCount = Number(st.tick ?? 0)
    w.nextEventIn = Number.isFinite(st.nextEventIn) ? st.nextEventIn : w._rollEventDelay()
    w.worldEvents = Array.isArray(st.worldEvents) ? st.worldEvents.slice(-200) : []
    w.entities = new Map()
    for (const [id, e] of Object.entries(st.entities ?? {})) {
      w.entities.set(id, { id: e.id ?? id, pos: { ...(e.pos ?? SPAWN) }, dir: e.dir ?? 'n', held: e.held ?? null })
    }
    w._events = new Map()
    for (const [id, list] of Object.entries(st.events ?? {})) {
      w._events.set(id, Array.isArray(list) ? list.slice(-200) : [])
    }
    return w
  }
}
