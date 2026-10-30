// ============================================================================
// 临时自测：聊天窗口 UI（server/ui/index.html）
//   · 起 mock HTTP 服务（node:http，端口 19630）返回契约形状假数据，
//     并记录 /api/controller 请求
//   · 用 node 读 index.html 做字符串断言：API 路径、POST 调用、
//     气泡渲染关键函数名等与契约一致（无需真实浏览器）
// 运行：node _selfcheck_ui.mjs
// ============================================================================
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const HTML = path.join(__dirname, 'server', 'ui', 'index.html')
const PORT = 19630
const BASE = 'http://127.0.0.1:' + PORT

let pass = 0
let fail = 0
const failures = []
function check(name, cond, extra = '') {
  if (cond) {
    pass++
    console.log(`  PASS  ${name}`)
  } else {
    fail++
    failures.push(name + (extra ? ` | ${extra}` : ''))
    console.log(`  FAIL  ${name}  ${extra}`)
  }
}

/* ===== mock 服务：契约形状假数据 + 记录 /api/controller 请求 ===== */
const controllerRequests = []
const fakeStatus = {
  ok: true,
  initialized: true,
  mode: 'train',
  name: '小忆',
  sleep: { asleep: false, wakefulness: 0.82, sleepCount: 3, lastWakeAt: Date.now(), melatoninProgress: 0.15 },
  pendingCount: 2,
  emotion: { valence: 0.42, arousal: 0.31, joy: 0.5, sadness: 0.1, anger: 0.05 },
  persona: { traits: { agreeableness: 0.7, openness: 0.6 }, moralFoundations: { care: 0.8, fairness: 0.7 } },
  endocrine: {
    molecules: {
      dopamine: { level: 0.55 }, cortisol: { level: 0.2 }, oxytocin: { level: 0.6 },
      testosterone: { level: 0.3 }, serotonin: { level: 0.5 }, melatonin: { level: 0.15 },
      noradrenaline: { level: 0.18 }, adrenaline: { level: 0.1 }, estrogen: { level: 0.2 },
      progesterone: { level: 0.18 }, endorphin: { level: 0.2 }, acetylcholine: { level: 0.22 },
      gaba: { level: 0.25 }, glutamate: { level: 0.25 }, prolactin: { level: 0.15 },
    },
    effects: { valence: 0.4, arousal: 0.3 },
  },
  memory: { predErrorEma: 0.021, observeCount: 128, paramCount: 53210, layerSizes: [204, 128] },
  controller: { actions: ['hear', 'speak', 'sleep', 'wake', 'freeze', 'thaw'] },
  mirror: { synced: true },
  flaw: { frozen: false },
  analysis: { storeCount: 42 },
  extensions: [{ path: '/ext/a', paramCount: 1000, layerSizes: [64], predErrorEma: 0.01, observeCount: 10 }],
  pendingSenses: 2,
}

const server = http.createServer((req, res) => {
  const u = new URL(req.url, BASE)
  if (req.method === 'GET' && u.pathname === '/api/status') {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(fakeStatus))
  } else if (req.method === 'POST' && u.pathname === '/api/controller') {
    let body = ''
    req.on('data', (c) => { body += c })
    req.on('end', () => {
      let parsed = null
      try { parsed = JSON.parse(body) } catch { /* ignore */ }
      if (parsed) controllerRequests.push(parsed)
      let result = null
      if (parsed && parsed.action === 'speak') {
        result = {
          text: '你好，我是「小忆」，很高兴见到你。',
          conversation: [], status: 'ok', reply: '你好，我是小忆。',
          moodTag: 'calm', persona: fakeStatus.persona, emotion: fakeStatus.emotion,
        }
      } else if (parsed && parsed.action === 'sleep') {
        fakeStatus.sleep.asleep = true
        result = { asleep: true }
      } else if (parsed && parsed.action === 'wake') {
        fakeStatus.sleep.asleep = false
        fakeStatus.sleep.lastWakeAt = Date.now()
        result = { asleep: false }
      }
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ ok: true, action: parsed ? parsed.action : null, result }))
    })
  } else if (req.method === 'GET' && (u.pathname === '/' || u.pathname === '/index.html')) {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
    res.end(fs.readFileSync(HTML))
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' })
    res.end('not found')
  }
})

await new Promise((resolve, reject) => {
  server.once('error', reject)
  server.listen(PORT, '127.0.0.1', resolve)
})

try {
  console.log('== mock 服务契约形状 ==')
  const st = await (await fetch(BASE + '/api/status')).json()
  check('GET /api/status 返回 ok:true', st.ok === true)
  check('status 含 initialized/name/mode', 'initialized' in st && 'name' in st && 'mode' in st)
  check(
    'status 含 sleep 形状（asleep/wakefulness/sleepCount/lastWakeAt/melatoninProgress）',
    st.sleep && 'asleep' in st.sleep && 'wakefulness' in st.sleep && 'sleepCount' in st.sleep && 'lastWakeAt' in st.sleep && 'melatoninProgress' in st.sleep,
  )
  check('status 含 emotion（valence/arousal）', st.emotion && typeof st.emotion.valence === 'number' && typeof st.emotion.arousal === 'number')
  check('status 含 endocrine.molecules.dopamine.level', st.endocrine && st.endocrine.molecules && typeof st.endocrine.molecules.dopamine.level === 'number')
  check('status 含 memory（predErrorEma/observeCount）', st.memory && typeof st.memory.predErrorEma === 'number' && typeof st.memory.observeCount === 'number')
  check('status 含 analysis.storeCount', st.analysis && typeof st.analysis.storeCount === 'number')
  check('status 含 extensions 数组', Array.isArray(st.extensions))
  check('status 含 pendingCount', typeof st.pendingCount === 'number')

  console.log('== /api/controller 请求记录 + speak 形状 ==')
  const sp = await (
    await fetch(BASE + '/api/controller', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'speak', payload: {} }),
    })
  ).json()
  check('POST /api/controller 返回 {ok, action, result}', sp.ok === true && sp.action === 'speak' && sp.result && typeof sp.result.text === 'string')
  check(
    'controller 请求已记录（speak + 空 payload）',
    controllerRequests.length === 1 && controllerRequests[0].action === 'speak' && controllerRequests[0].payload && Object.keys(controllerRequests[0].payload).length === 0,
  )
  const wk = await (
    await fetch(BASE + '/api/controller', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'wake', payload: {} }),
    })
  ).json()
  check('wake action 可用（lastWakeAt 更新、asleep=false）', wk.ok === true && fakeStatus.sleep.asleep === false)

  console.log('== index.html 可服务 ==')
  const page = await fetch(BASE + '/index.html')
  check('index.html 200 且 Content-Type text/html', page.status === 200 && (page.headers.get('content-type') || '').includes('text/html'))

  console.log('== index.html 契约字符串断言 ==')
  const html = fs.readFileSync(HTML, 'utf8')
  const asserts = [
    ['引用 fetch /api/status', html.includes("fetch('/api/status')")],
    ['引用 fetch /api/controller', html.includes("fetch('/api/controller'")],
    ['POST 调用（method: POST）', html.includes("method: 'POST'")],
    ['POST body JSON.stringify({action, payload})', html.includes('JSON.stringify({ action, payload')],
    ['初始主动发言 controller("speak", {})', html.includes("controller('speak', {})")],
    ['发送时 speak 携带用户文本 payload', html.includes("controller('speak', { text })")],
    ['气泡渲染 addMessage', html.includes('function addMessage(')],
    ['用户气泡 addUserMessage', html.includes('function addUserMessage(')],
    ['生命气泡 addLifeMessage', html.includes('function addLifeMessage(')],
    ['思考中占位 showThinking', html.includes('function showThinking(')],
    ['移除思考 hideThinking', html.includes('function hideThinking(')],
    ['发送 sendMessage', html.includes('function sendMessage(')],
    ['状态渲染 renderStatus', html.includes('function renderStatus(')],
    ['状态拉取 fetchStatus', html.includes('function fetchStatus(')],
    ['启动 boot', html.includes('function boot(')],
    ['提示条 showToast', html.includes('function showToast(')],
    ['controller 封装', html.includes('function controller(')],
    ['初始主动发言 runIntro', html.includes('function runIntro(')],
    ['2 秒轮询 setInterval(..., 2000)', html.includes('setInterval(') && html.includes('2000')],
    ['深色主题 #1e2129', html.includes('#1e2129')],
    ['侧栏宽度 260px', html.includes('260px')],
    ['醒来提示「☀ 它醒来了」', html.includes('☀ 它醒来了')],
    ['lastWakeAt 变化跟踪', html.includes('lastWakeAt')],
    ['褪黑素进度 melatoninProgress', html.includes('melatoninProgress')],
    ['清醒度 wakefulness', html.includes('wakefulness')],
    ['预测误差 predErrorEma', html.includes('predErrorEma')],
    ['6 种主要分子', ['dopamine', 'cortisol', 'oxytocin', 'testosterone', 'serotonin', 'melatonin'].every((k) => html.includes(k))],
    ['其余 9 种分子', ['noradrenaline', 'adrenaline', 'estrogen', 'progesterone', 'endorphin', 'acetylcholine', 'gaba', 'glutamate', 'prolactin'].every((k) => html.includes(k))],
    ['分析存档 storeCount', html.includes('storeCount')],
    ['感知观察 observeCount', html.includes('observeCount')],
    ['待处理感知 pendingCount', html.includes('pendingCount')],
    ['思考中文案', html.includes('思考中')],
    ['头像 🧠', html.includes('🧠')],
    ['Enter 发送', html.includes("e.key === 'Enter'")],
    ['未初始化引导（initialized + 初始化文案）', html.includes('initialized') && html.includes('初始化')],
    ['镜像同步 mirror.synced', html.includes('synced')],
    ['缺陷人格 flaw.frozen', html.includes('frozen')],
    ['情绪效价 valence', html.includes('valence')],
    ['情绪唤醒 arousal', html.includes('arousal')],
  ]
  for (const [name, cond] of asserts) check(name, cond)
} finally {
  try { server.closeAllConnections() } catch { /* ignore */ }
  await new Promise((r) => server.close(r))
}

console.log('')
console.log(`结果：PASS ${pass} / FAIL ${fail}`)
if (fail > 0) {
  console.log('失败项：')
  for (const f of failures) console.log('  - ' + f)
  process.exit(1)
}
process.exit(0)
