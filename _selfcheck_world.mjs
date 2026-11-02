// ============================================================================
// dsh-plastic-memory —— World 自测（临时脚本，零依赖）
// 运行：<dsh-runtime>/dependencies/node/bin/node.exe _selfcheck_world.mjs
// 覆盖：布局确定性 / 重力 / move 步长校验 / 跳跃上限 / 交互距离 / eat 效果 /
//       不可食用拒绝 / 事件引擎（stone_hit + eventsFor）/ 六感文案 / 越界 /
//       状态序列化往返 / 世界服务冒烟（/state /register /sense /act /）。
// ============================================================================

import { mkdtempSync, rmSync, readFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'
import { World, SPAWN } from './lib/world.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const results = []
function check(name, cond, detail = '') {
  results.push({ name, pass: !!cond, detail })
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`)
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// 找一个远离物品的自由格子（供实体活动）
function freeCell(world, minItemDist = 3) {
  for (let y = 2; y < 38; y++) {
    for (let x = 2; x < 38; x++) {
      const busy = world.items.some((it) => it.heldBy == null && Math.hypot(it.pos.x - x, it.pos.y - y) < minItemDist)
      if (!busy) return { x, y, z: 0 }
    }
  }
  throw new Error('找不到自由格子')
}

const world = new World({ seed: 12345 })
const F = freeCell(world)

// ============================================================================
// ① 布局与确定性
// ============================================================================
check('①a 物品 ≥10 件', world.items.length >= 10, `items=${world.items.length}`)
const cakes = world.items.filter((it) => it.type === 'cake')
check('①b 蛋糕 ×2', cakes.length === 2)
check(
  '①c 蛋糕距出生点 6-10',
  cakes.every((c) => {
    const d = Math.hypot(c.pos.x - SPAWN.x, c.pos.y - SPAWN.y)
    return d >= 6 - 1e-9 && d <= 10 + 1e-9
  }),
  cakes.map((c) => Math.hypot(c.pos.x - SPAWN.x, c.pos.y - SPAWN.y).toFixed(1)).join(','),
)
check('①d 物品字段完整', world.items.every((it) => ['id', 'type', 'name', 'pos', 'sight', 'smell', 'sound', 'taste', 'edible', 'radius'].every((k) => k in it)))
const w2 = new World({ seed: 12345 })
const w3 = new World({ seed: 999 })
check('①e 同 seed 布局一致', JSON.stringify(w2.items) === JSON.stringify(world.items))
check('①f 异 seed 布局不同', JSON.stringify(w3.items) !== JSON.stringify(world.items))

// ============================================================================
// ② 移动 / 重力 / 跳跃
// ============================================================================
check('②a registerEntity ok', world.registerEntity('life', F).ok === true)
const j1 = world.act('life', { pos: { ...F }, action: 'jump' })
check('②b jump z+2', j1.ok && j1.newPos.z === 2, JSON.stringify(j1))
world.tick()
check('②c tick 重力 z-1', world.entities.get('life').pos.z === 1)
world.tick()
check('②d 再 tick 落地 z=0', world.entities.get('life').pos.z === 0)
world.act('life', { pos: { ...F, z: 0 }, action: 'jump' }) // z=2
const m1 = world.act('life', { pos: { x: F.x, y: F.y, z: 2 }, action: 'move', dir: 'e' })
check('②e act 先重力再移动（z=1）', m1.ok && m1.newPos.z === 1 && m1.newPos.x === F.x + 1, JSON.stringify(m1))
world.registerEntity('life', { ...F }) // 复位
const bad = world.act('life', { pos: { x: F.x + 5, y: F.y }, action: 'move', dir: 'e' })
check('②f move 步长 >1 拒绝', bad.ok === false && /步长|同步/.test(bad.error ?? ''), JSON.stringify(bad))
const good = world.act('life', { pos: { ...F }, action: 'move', dir: 'e' })
check('②g move 1 步成功', good.ok && good.newPos.x === F.x + 1 && good.newPos.z === 0, JSON.stringify(good))
world.registerEntity('life', { x: 0, y: F.y, z: 0 })
const edge = world.act('life', { pos: { x: 0, y: F.y }, action: 'move', dir: 'w' })
check('②h 地图越界拒绝', edge.ok === false && /越界/.test(edge.error ?? ''), JSON.stringify(edge))
world.registerEntity('life', { x: F.x, y: F.y, z: 8 })
const jc = world.act('life', { pos: { x: F.x, y: F.y, z: 8 }, action: 'jump' })
check('②i jump 超上限拒绝（重力后 z=7）', jc.ok === false && jc.newPos.z === 7, JSON.stringify(jc))

// ============================================================================
// ③ 六感感知
// ============================================================================
const flowers = world.items.filter((it) => it.type === 'flower')
const flower1 = flowers[0]
const flower2 = flowers[1]
const stone1 = world.items.find((it) => it.type === 'stone')
world.registerEntity('life', { x: flower1.pos.x, y: Math.min(39, flower1.pos.y + 4), z: 0 })
const s1 = world.sense('life', { dir: 'n', senses: ['sight', 'smell', 'hearing'] })
const sightFlower = s1.sights.find((s) => s.itemId === flower1.id)
check('③a 视觉：正向扇形 + 方位文案', !!sightFlower && sightFlower.dir === 'n' && /北方|身边/.test(sightFlower.text) && sightFlower.text.includes('一朵花'), sightFlower?.text)
check('③b 嗅觉文案', s1.smells.some((s) => s.itemId === flower1.id && s.text.includes('花香')), s1.smells.find((s) => s.itemId === flower1.id)?.text)
check('③c 听觉文案（鸟鸣）', s1.sounds.some((s) => s.itemId === flower1.id && s.text.includes('鸟鸣')))
const sBack = world.sense('life', { dir: 's', senses: ['sight'] })
check('③d 背向不可见（扇形外）', !sBack.sights.some((s) => s.itemId === flower1.id))
world.registerEntity('life', { x: flower2.pos.x, y: Math.min(39, flower2.pos.y + 7), z: 0 })
const sFar = world.sense('life', { dir: 'n', senses: ['sight', 'smell'] })
check('③e 默认 range=5 之外不可见/不可闻', !sFar.sights.some((s) => s.itemId === flower2.id) && !sFar.smells.some((s) => s.itemId === flower2.id))
world.registerEntity('life', { x: Math.min(39, stone1.pos.x + 1), y: stone1.pos.y, z: 0 })
const sTouch = world.sense('life', { senses: ['touch'] })
check('③f 触觉接触反馈', sTouch.touches.some((t) => t.itemId === stone1.id && t.text.includes('石头')), sTouch.touches.find((t) => t.itemId === stone1.id)?.text)

// ============================================================================
// ④ 交互：grab / eat / drop
// ============================================================================
const cake = world.items.find((it) => it.type === 'cake')
const sideDx = cake.pos.x >= 3 ? -3 : 3
world.registerEntity('life', { x: cake.pos.x + sideDx, y: cake.pos.y, z: 0 })
const gFar = world.act('life', { pos: { x: cake.pos.x + sideDx, y: cake.pos.y }, action: 'grab', target: cake.id })
check('④a grab 距离 >2 拒绝', gFar.ok === false && /太远/.test(gFar.error ?? ''), JSON.stringify(gFar))
const nearDx = sideDx > 0 ? 2 : -2
world.registerEntity('life', { x: cake.pos.x + nearDx, y: cake.pos.y, z: 0 })
const gOk = world.act('life', { pos: { x: cake.pos.x + nearDx, y: cake.pos.y }, action: 'grab', target: cake.id })
check('④b grab 距离 =2 成功', gOk.ok && cake.heldBy === 'life' && world.entities.get('life').held === cake.id, JSON.stringify(gOk))
world.registerEntity('other', { x: cake.pos.x - nearDx, y: cake.pos.y, z: 0 })
const g2 = world.act('other', { pos: { x: cake.pos.x - nearDx, y: cake.pos.y }, action: 'grab', target: cake.id })
check('④c 已被拿取拒绝', g2.ok === false && /拿取/.test(g2.error ?? ''), JSON.stringify(g2))
const eNotHeld = world.act('other', { pos: { x: cake.pos.x - nearDx, y: cake.pos.y }, action: 'eat', target: cake.id })
check('④d 未持有不能吃', eNotHeld.ok === false && /没有拿着/.test(eNotHeld.error ?? ''))
const nBefore = world.items.length
const eOk = world.act('life', { pos: { x: cake.pos.x + nearDx, y: cake.pos.y }, action: 'eat', target: cake.id })
check(
  '④e eat 成功：效果 + 物品移除',
  eOk.ok && eOk.effects?.dopamine === 0.5 && eOk.effects?.endorphin === 0.4 && eOk.effects?.satiety === 0.6 &&
    !world.items.some((it) => it.id === cake.id) && world.items.length === nBefore - 1 && world.entities.get('life').held === null,
  JSON.stringify(eOk),
)
const flowerX = flower2.pos.x >= 2 ? -2 : 2
world.registerEntity('life', { x: flower2.pos.x + flowerX, y: flower2.pos.y, z: 0 })
world.act('life', { pos: { x: flower2.pos.x + flowerX, y: flower2.pos.y }, action: 'grab', target: flower2.id })
const eBad = world.act('life', { pos: { x: flower2.pos.x + flowerX, y: flower2.pos.y }, action: 'eat', target: flower2.id })
check('④f 不可食用拒绝且物品保留', eBad.ok === false && /不能吃/.test(eBad.error ?? '') && world.items.some((it) => it.id === flower2.id), JSON.stringify(eBad))
// 花朵仍拿在手里：先放下，再测拿石头
const dFlower = world.act('life', { pos: { x: flower2.pos.x + flowerX, y: flower2.pos.y }, action: 'drop' })
check('④g 放下不可食用的花朵（放回原坐标）', dFlower.ok && world.items.find((it) => it.id === flower2.id).heldBy == null && world.entities.get('life').held === null, JSON.stringify(dFlower))
const stoneX = stone1.pos.x >= 2 ? -2 : 2
world.registerEntity('life', { x: stone1.pos.x + stoneX, y: stone1.pos.y, z: 0 })
const gStone = world.act('life', { pos: { x: stone1.pos.x + stoneX, y: stone1.pos.y }, action: 'grab', target: stone1.id })
check('④h 拿石头成功', gStone.ok && stone1.heldBy === 'life')
const dOk = world.act('life', { pos: { x: stone1.pos.x + stoneX, y: stone1.pos.y }, action: 'drop' })
check('④i drop 放回原坐标', dOk.ok && stone1.heldBy == null && world.entities.get('life').held === null, JSON.stringify(dOk))
const dEmpty = world.act('life', { pos: { x: stone1.pos.x + stoneX, y: stone1.pos.y }, action: 'drop' })
check('④j 空手 drop 拒绝', dEmpty.ok === false && /没有东西/.test(dEmpty.error ?? ''))

// ============================================================================
// ⑤ 事件引擎（石头坠落 / 领取过滤 / tick 周期）
// ============================================================================
world.registerEntity('other', { x: 39, y: 39, z: 0 }) // 移远避免干扰
world.registerEntity('life', { ...F })
world._spawnStoneFall({ x: F.x, y: F.y }) // 显式落点=实体脚下 → 砸中
const hitEvents = world.eventsFor('life', 0)
const hit = hitEvents.find((ev) => ev.type === 'stone_hit')
check(
  '⑤a stone_hit 事件（痛感反馈）',
  !!hit && hit.pain === 0.6 && hit.text.includes('砸到了你') && hit.sights?.includes('石子掉下来') && hit.sounds?.includes('石子掉下来的声音'),
  JSON.stringify(hit),
)
check('⑤b eventsFor since 过滤幂等', !world.eventsFor('life', hit.ts).some((ev) => ev.type === 'stone_hit'))
const fx2 = F.x + 5 < 40 ? F.x + 5 : F.x - 5
world._spawnStoneFall({ x: fx2, y: F.y }) // 落点 5 格外 → 未砸中
const fallEvents = world.eventsFor('life', hit.ts)
check('⑤c stone_fall 未砸中事件', fallEvents.some((ev) => ev.type === 'stone_fall' && ev.text.includes('不远处')), JSON.stringify(fallEvents[fallEvents.length - 1]))
const wlBefore = world.worldEvents.length
world.nextEventIn = 1
world.tick()
check('⑤d tick 事件引擎触发并重置周期', world.worldEvents.length > wlBefore && world.nextEventIn >= 20 && world.nextEventIn <= 40, `next=${world.nextEventIn}`)

// ============================================================================
// ⑥ 状态序列化往返
// ============================================================================
const tmpDir = mkdtempSync(join(tmpdir(), 'world-selfcheck-'))
const stateFile = join(tmpDir, 'world.json')
check('⑥a saveState 原子写成功', world.saveState(stateFile) === true)
const restored = World.fromState(JSON.parse(readFileSync(stateFile, 'utf8')))
check('⑥b fromState 恢复物品一致', !!restored && restored.items.length === world.items.length && JSON.stringify(restored.items.map((i) => i.id)) === JSON.stringify(world.items.map((i) => i.id)))
const lifePos = world.entities.get('life').pos
const lifePos2 = restored?.entities.get('life')?.pos
check('⑥c 实体位置恢复一致', !!lifePos2 && lifePos2.x === lifePos.x && lifePos2.y === lifePos.y && lifePos2.z === lifePos.z)
check('⑥d tick/nextEventIn 恢复一致', restored?.tickCount === world.tickCount && restored?.nextEventIn === world.nextEventIn)
const st = world.loadState()
check('⑥e loadState 全量状态', st.seed === world.seed && st.items.length === world.items.length && Array.isArray(st.worldEvents))

// ============================================================================
// ⑦ 世界服务冒烟（spawn + HTTP 轮询；结束即清理）
// ============================================================================
if (typeof fetch === 'function') {
  const port = 19371 + Math.floor(Math.random() * 1000)
  const dataDir = join(__dirname, 'server', 'world-data')
  const dataExisted = existsSync(dataDir)
  const child = spawn(process.execPath, [join(__dirname, 'server', 'world-server.mjs')], {
    env: { ...process.env, PORT_WORLD: String(port) },
    stdio: 'ignore',
  })
  let up = false
  for (let i = 0; i < 40 && !up; i++) {
    await sleep(250)
    try {
      const r = await fetch(`http://127.0.0.1:${port}/state`)
      if (r.ok) up = true
    } catch {
      /* 未就绪 */
    }
  }
  check(`⑦a 服务启动 /state 可访问（port ${port}）`, up)
  if (up) {
    const stj = await (await fetch(`http://127.0.0.1:${port}/state`)).json()
    check('⑦b /state 物品 ≥10', stj.items.length >= 10)
    const html = await (await fetch(`http://127.0.0.1:${port}/`)).text()
    check('⑦c GET / 返回观察界面', html.includes('世界观察'))
    const reg = await (await fetch(`http://127.0.0.1:${port}/register`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ entity: 'smoke', pos: { x: 20, y: 20, z: 0 } }) })).json()
    check('⑦d POST /register ok', reg.ok === true)
    const sn = await (await fetch(`http://127.0.0.1:${port}/sense`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ entity: 'smoke', pos: { x: 20, y: 20 }, dir: 'n', senses: ['sight', 'smell'] }) })).json()
    check('⑦e POST /sense 六感结构', Array.isArray(sn.sights) && Array.isArray(sn.smells))
    const ac = await (await fetch(`http://127.0.0.1:${port}/act`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ entity: 'smoke', pos: { x: 20, y: 20, z: 0 }, action: 'jump' }) })).json()
    check('⑦f POST /act jump ok', ac.ok === true && ac.newPos.z === 2, JSON.stringify(ac))
  }
  child.kill()
  await sleep(300)
  if (!dataExisted) {
    try {
      rmSync(dataDir, { recursive: true, force: true })
    } catch {
      /* ignore */
    }
  }
} else {
  console.log('SKIP  ⑦ 服务冒烟（当前 node 无全局 fetch）')
}

// ---- 清理 ----
try {
  rmSync(tmpDir, { recursive: true, force: true })
} catch {
  /* ignore */
}

const failed = results.filter((r) => !r.pass).length
console.log(`\n${results.length - failed}/${results.length} 通过${failed ? `，${failed} 失败` : '，全部通过 ✅'}`)
process.exit(failed ? 1 : 0)
