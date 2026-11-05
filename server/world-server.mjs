// ============================================================================
// dsh-plastic-memory —— 世界独立服务（world-server.mjs）
// ----------------------------------------------------------------------------
// 端口 9631（PORT_WORLD 环境变量可覆盖）。数字生命 9630 服务经 HTTP 调用，
// CORS 全开。接口：
//   GET  /             → server/world-ui/index.html（观察界面）
//   GET  /state        → 世界全量状态
//   GET  /state-simple → 纯文本行：首行 `temp <温度> <tick>`，其后 `item|entity x y z name`
//                        （raylib3D 渲染程序每 1 秒经 WinHTTP 轮询）
//   GET  /events       ?entity=xx&since=ts → 世界主动反馈事件
//   POST /register     {entity, pos} → 注册/恢复实体
//   POST /sense        {entity, pos, dir, range, senses} → 六感（含温度/温度文本/spoken）
//   POST /act          {entity, pos, action, dir, target} → 行为结果
//   POST /speak        {entity, text} → 把文字放到实体位置（下次 sense 的 spoken 领取）
// 心跳：每秒 tick（重力/事件引擎）；每 5 秒原子写 server/world-data/world.json。
// 零外部依赖（node:http / node:fs / node:path + lib/world.mjs）。
// ============================================================================

import http from 'node:http'
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { World, temperatureAt } from '../lib/world.mjs'
import { checkWorldAction, checkGodPower, GODHOOD_NAMES } from '../lib/godhood.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PORT = Number(process.env.PORT_WORLD || 9631)
const STATE_FILE = join(__dirname, 'world-data', 'world.json')
const UI_FILE = join(__dirname, 'world-ui', 'index.html')

// ---- 启动：优先恢复存档，否则新建固定布局世界 ----
let world = null
try {
  world = World.fromState(JSON.parse(readFileSync(STATE_FILE, 'utf8')))
} catch {
  /* 存档不存在/损坏 → 新建 */
}
if (!world) {
  const seed = Number(process.env.WORLD_SEED) || 20261102
  world = new World({ seed })
}
console.log(`[world] seed=${world.seed} items=${world.items.length} port=${PORT}`)

// ---------------------------------------------------------------------------
// HTTP 工具
// ---------------------------------------------------------------------------

function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
}

function json(res, code, data) {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' })
  res.end(JSON.stringify(data))
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0
    const chunks = []
    req.on('data', (c) => {
      size += c.length
      if (size > 1024 * 1024) {
        reject(new Error('body 过大'))
        req.destroy()
        return
      }
      chunks.push(c)
    })
    req.on('end', () => {
      try {
        resolve(chunks.length > 0 ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {})
      } catch {
        reject(new Error('body 不是合法 JSON'))
      }
    })
    req.on('error', reject)
  })
}

// ---------------------------------------------------------------------------
// 路由
// ---------------------------------------------------------------------------

const server = http.createServer(async (req, res) => {
  setCors(res)
  if (req.method === 'OPTIONS') {
    res.writeHead(204)
    res.end()
    return
  }
  const url = new URL(req.url ?? '/', 'http://127.0.0.1')
  try {
    // 观察界面
    if (req.method === 'GET' && url.pathname === '/') {
      const html = readFileSync(UI_FILE, 'utf8')
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
      res.end(html)
      return
    }
    // 世界全量状态
    if (req.method === 'GET' && url.pathname === '/state') {
      json(res, 200, world.loadState())
      return
    }
    // raylib3D 渲染程序用的纯文本简化状态：
    //   首行 `temp <温度> <tick>`；其后每行 `item|entity x y z name`
    if (req.method === 'GET' && url.pathname === '/state-simple') {
      const st = world.loadState()
      const ents = Object.values(st.entities ?? {})
      const anchor = ents[0]?.pos ?? { x: 20, y: 20, z: 0 }
      const temp = temperatureAt(anchor.x, anchor.y, anchor.z)
      const lines = [`temp ${temp} ${st.tick}`]
      for (const it of st.items) {
        lines.push(`item ${it.pos.x} ${it.pos.y} ${it.pos.z} ${String(it.name ?? it.type)}`)
      }
      for (const e of ents) {
        const name = String(e.id ?? 'life').replace(/\s+/g, '_')
        lines.push(`entity ${e.pos.x} ${e.pos.y} ${e.pos.z} ${name}`)
      }
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' })
      res.end(lines.join('\n') + '\n')
      return
    }
    // 世界主动反馈事件
    if (req.method === 'GET' && url.pathname === '/events') {
      const entity = url.searchParams.get('entity')
      if (!entity) return json(res, 400, { ok: false, error: '缺少 entity 参数' })
      const since = Number(url.searchParams.get('since') ?? 0) || 0
      json(res, 200, { entity, since, events: world.eventsFor(entity, since) })
      return
    }
    if (req.method === 'POST' && url.pathname === '/register') {
      const body = await readBody(req)
      if (!body.entity) return json(res, 400, { ok: false, error: '缺少 entity' })
      const tier = Number.isInteger(body.tier) ? body.tier : 2 // 默认干涉者
      json(res, 200, world.registerEntity(String(body.entity), body.pos ?? {}, { tier }))
      return
    }
    if (req.method === 'POST' && url.pathname === '/sense') {
      const body = await readBody(req)
      if (!body.entity) return json(res, 400, { ok: false, error: '缺少 entity' })
      json(res, 200, world.sense(String(body.entity), body))
      return
    }
    if (req.method === 'POST' && url.pathname === '/act') {
      const body = await readBody(req)
      if (!body.entity) return json(res, 400, { ok: false, error: '缺少 entity' })
      // 神格校验：观察者只能旁观，无法插手
      const tier = world.entityTier(String(body.entity))
      const check = checkWorldAction(tier, body.action ?? '')
      if (!check.ok) return json(res, 403, { ok: false, error: check.error, tier, tierName: GODHOOD_NAMES[tier] ?? '未知' })
      json(res, 200, world.act(String(body.entity), body))
      return
    }
    if (req.method === 'POST' && url.pathname === '/speak') {
      const body = await readBody(req)
      if (!body.entity) return json(res, 400, { ok: false, error: '缺少 entity' })
      json(res, 200, world.speak(String(body.entity), body.text))
      return
    }
    // ---- 神力端点（仅神格「神」可用）----
    if (req.method === 'POST' && url.pathname === '/god') {
      const body = await readBody(req)
      if (!body.entity) return json(res, 400, { ok: false, error: '缺少 entity' })
      const tier = world.entityTier(String(body.entity))
      const check = checkGodPower(tier, body.power ?? '')
      if (!check.ok) return json(res, 403, { ok: false, error: check.error, tier, tierName: GODHOOD_NAMES[tier] ?? '未知' })
      const result = world.godPower(String(body.entity), body.power, body.args ?? {})
      json(res, 200, { ok: true, ...result })
      return
    }
    // ---- 观察者故事：世界发生的故事（记录职责）----
    if (req.method === 'GET' && url.pathname === '/story') {
      json(res, 200, { ok: true, stories: world.stories() })
      return
    }
    json(res, 404, { ok: false, error: `未找到接口: ${req.method} ${url.pathname}` })
  } catch (error) {
    json(res, 500, { ok: false, error: String(error?.message ?? error) })
  }
})

server.listen(PORT, () => {
  console.log(`[world] 世界服务已启动: http://127.0.0.1:${PORT}/ （观察界面）`)
})

// ---------------------------------------------------------------------------
// 心跳与持久化
// ---------------------------------------------------------------------------

setInterval(() => {
  try {
    world.tick()
  } catch (error) {
    console.error('[world] tick 异常:', error?.message ?? error)
  }
}, 1000)

setInterval(() => {
  try {
    world.saveState(STATE_FILE)
  } catch (error) {
    console.error('[world] 保存异常:', error?.message ?? error)
  }
}, 5000)

// 启动即落盘一次（确保 world-data/world.json 存在）
try {
  world.saveState(STATE_FILE)
} catch (error) {
  console.error('[world] 初始保存异常:', error?.message ?? error)
}
