// ============================================================================
// dsh-plastic-memory —— 独立软件服务（数字生命独立运行）
// ----------------------------------------------------------------------------
// 数字生命从 DSH 插件中独立出来，作为常驻服务运行：
//   · 数据接口：GET /api/status、POST /api/controller、GET /api/persona、
//     POST /api/sense、POST /api/init（DSH 侧边栏插件全部经此调用）
//   · AI 大模型接入：OpenAI 兼容 /v1/chat/completions（DSH 添加自定义模型即可
//     以"AI 大模型"形式调用；响应不含 tool_calls → 与该 AI 聊天不调用其他工具）
//   · 独立窗口 UI：GET /ui（聊天窗口，见 server/ui/index.html）
//
// 启动：node server/server.mjs（端口 9630，PORT 环境变量可覆盖）
// 工作区：优先 PLASTIC_WORKSPACE 环境变量，其次 DSH 上次工作区记录
// ============================================================================

import { createServer } from 'node:http'
import { readFileSync, existsSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PlasticBrain } from '../lib/brain.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PLUGIN_ROOT = resolve(__dirname, '..')
const PORT = Number(process.env.PORT ?? 9630)
const HOST = '127.0.0.1'

// DSH_HOME：PLUGIN_ROOT = <DSH_HOME>/plugins/dsh-plastic-memory → 上两层即 DSH_HOME
const DSH_HOME = process.env.DSH_HOME ?? resolve(PLUGIN_ROOT, '..', '..')
const LAST_WORKSPACE_FILE = join(DSH_HOME, 'data', 'dsh-plastic-memory', 'last-workspace.json')

// ---------------------------------------------------------------------------
// 大脑实例：优先环境变量工作区，其次 DSH 上次记录
// ---------------------------------------------------------------------------

const brain = new PlasticBrain()

function resolveWorkspace() {
  if (process.env.PLASTIC_WORKSPACE) return process.env.PLASTIC_WORKSPACE
  try {
    return JSON.parse(readFileSync(LAST_WORKSPACE_FILE, 'utf8')).workspace ?? null
  } catch {
    return null
  }
}

function bootBrain() {
  const ws = resolveWorkspace()
  if (ws) {
    try {
      brain.load(ws)
      return { loaded: ws }
    } catch (error) {
      return { error: String(error?.message ?? error) }
    }
  }
  return { loaded: null }
}

bootBrain()

// ---------------------------------------------------------------------------
// HTTP 小工具
// ---------------------------------------------------------------------------

function json(res, status, body, extra = {}) {
  res.statusCode = status
  res.setHeader('content-type', 'application/json; charset=utf-8')
  res.setHeader('access-control-allow-origin', '*')
  res.setHeader('access-control-allow-headers', 'content-type, authorization')
  res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS')
  res.setHeader('cache-control', 'no-store')
  for (const [k, v] of Object.entries(extra)) res.setHeader(k, v)
  res.end(JSON.stringify(body))
}

async function readBody(req, limit = 512 * 1024) {
  const chunks = []
  let total = 0
  try {
    for await (const chunk of req) {
      total += chunk.length
      if (total > limit) return null
      chunks.push(chunk)
    }
  } catch {
    return null
  }
  return Buffer.concat(chunks).toString('utf8')
}

async function parseBody(req, res) {
  const raw = await readBody(req)
  if (raw === null) {
    json(res, 413, { error: '请求体过大' })
    return null
  }
  try {
    return JSON.parse(raw || '{}')
  } catch {
    json(res, 400, { error: '请求体不是合法 JSON' })
    return null
  }
}

// ---------------------------------------------------------------------------
// 路由
// ---------------------------------------------------------------------------

const CONTROLLER_ACTIONS = new Set([
  'hear', 'speak', 'read', 'write', 'adjust', 'freeze', 'thaw', 'reset', 'save',
  'grow', 'prune', 'adjustLr', 'scan', 'mountExtension', 'unmountExtension',
  'sleep', 'wake', 'generateNetwork', 'trainFlaw', 'analysisStatus',
])

async function handleController(req, res, body) {
  const action = typeof body.action === 'string' ? body.action : ''
  const payload = body.payload && typeof body.payload === 'object' ? body.payload : {}
  if (!CONTROLLER_ACTIONS.has(action)) {
    json(res, 400, { ok: false, error: `未知 action：${action}。支持：${[...CONTROLLER_ACTIONS].join('/')}` })
    return
  }
  try {
    let result
    switch (action) {
      case 'hear':
        result = brain.hear(payload.text ?? '', { module: payload.module, source: 'standalone-api' })
        break
      case 'speak':
        result = brain.speak(payload.text ?? '', { module: payload.module })
        break
      case 'read':
        result = brain.read(payload.query ?? '', {})
        break
      case 'write':
        result = brain.write(payload.text ?? '', { module: payload.module, epochs: payload.epochs })
        break
      case 'adjust':
        result = brain.adjustMolecules(payload.molecules ?? {})
        break
      case 'freeze':
        result = brain.freeze()
        break
      case 'thaw':
        result = brain.thaw()
        break
      case 'reset':
        result = brain.resetGenome()
        break
      case 'save':
        brain.save()
        result = { saved: true }
        break
      case 'grow':
        result = brain.structuralCommand('grow', { layer: payload.layer })
        break
      case 'prune':
        result = brain.structuralCommand('prune', { layer: payload.layer, index: payload.index })
        break
      case 'adjustLr':
        result = brain.structuralCommand('adjustLr', { factor: payload.factor })
        break
      case 'scan':
        result = brain.scanSenses()
        break
      case 'mountExtension': {
        if (Array.isArray(payload.paths)) result = brain.mountExtensions(payload.paths)
        else result = brain.mountExtension(payload.path ?? '')
        break
      }
      case 'unmountExtension':
        result = brain.unmountExtension(payload.path ?? '')
        break
      case 'sleep':
        result = brain.sleepNow()
        break
      case 'wake':
        result = brain.wakeNow()
        break
      case 'generateNetwork':
        result = brain.generateNetwork(payload.path ?? '', { seed: Number.isInteger(payload.seed) ? payload.seed : undefined })
        break
      case 'trainFlaw':
        result = brain.trainFlaw(payload.text ?? '', { epochs: payload.epochs })
        break
      case 'analysisStatus':
        result = brain.analysisStatus(payload.limit ?? 20)
        break
      default:
        result = null
    }
    json(res, 200, { ok: true, action, result })
  } catch (error) {
    json(res, 500, { ok: false, error: error instanceof Error ? error.message : String(error) })
  }
}

/** OpenAI 兼容：DSH 以 AI 大模型形式调用数字生命。响应不含 tool_calls。 */
async function handleChatCompletions(req, res, body) {
  const messages = Array.isArray(body.messages) ? body.messages : []
  // 取最后一条用户消息（忽略 system；数字生命用自己的网络状态回应）
  let userText = ''
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m && m.role === 'user' && typeof m.content === 'string') {
      userText = m.content
      break
    }
    if (m && m.role === 'user' && Array.isArray(m.content)) {
      const textPart = m.content.find((p) => p && p.type === 'text')
      if (textPart) userText = String(textPart.text ?? '')
      break
    }
  }
  if (!brain.initialized) {
    json(res, 503, {
      error: { message: '数字生命尚未初始化：请先在 DSH 侧边栏「可塑性记忆」面板初始化工作区，或 POST /api/init {workspace}', type: 'not_initialized', code: 'not_initialized' },
    })
    return
  }
  try {
    const said = brain.speak(userText, { module: 'audio' })
    const content = typeof said?.text === 'string' ? said.text : '（它沉默了一会儿）'
    const model = body.model ?? 'digital-life'
    const id = `chatcmpl-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const created = Math.floor(Date.now() / 1000)

    // 流式（SSE）：DSH 的 chatCompletions 协议默认要求流式
    if (body.stream === true) {
      res.statusCode = 200
      res.setHeader('content-type', 'text/event-stream; charset=utf-8')
      res.setHeader('cache-control', 'no-store')
      res.setHeader('connection', 'keep-alive')
      res.setHeader('access-control-allow-origin', '*')
      const sse = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
      // 按小段流式输出，模拟逐字生成（每段 6 字，间隔 20ms）
      const chunkSize = 6
      for (let i = 0; i < content.length; i += chunkSize) {
        const piece = content.slice(i, i + chunkSize)
        sse({
          id,
          object: 'chat.completion.chunk',
          created,
          model,
          choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
        })
        await new Promise((r) => setTimeout(r, 20))
      }
      sse({
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      })
      res.write('data: [DONE]\n\n')
      res.end()
      return
    }

    // 非流式
    json(res, 200, {
      id,
      object: 'chat.completion',
      created,
      model,
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content },
          finish_reason: 'stop',
        },
      ],
      usage: { prompt_tokens: Math.round(userText.length / 2), completion_tokens: Math.round(content.length / 2), total_tokens: 0 },
    })
  } catch (error) {
    json(res, 500, { error: { message: error instanceof Error ? error.message : String(error), type: 'internal' } })
  }
}

/** OpenAI Responses 协议（DSH 的 openAI responses）：支持非流式与流式。 */
async function handleResponses(req, res, body) {
  // 提取输入文本：input 为字符串或消息数组
  let userText = ''
  const input = body.input
  if (typeof input === 'string') {
    userText = input
  } else if (Array.isArray(input)) {
    for (let i = input.length - 1; i >= 0; i--) {
      const m = input[i]
      if (m && m.role === 'user' && typeof m.content === 'string') {
        userText = m.content
        break
      }
      if (m && m.role === 'user' && Array.isArray(m.content)) {
        const textPart = m.content.find((p) => p && p.type === 'text' && typeof p.text === 'string')
        if (textPart) {
          userText = textPart.text
          break
        }
      }
      if (m && m.type === 'message' && m.role === 'user') {
        const c = m.content
        if (typeof c === 'string') {
          userText = c
          break
        }
        if (Array.isArray(c)) {
          const tp = c.find((p) => p && p.type === 'output_text')
          if (tp) {
            userText = tp.text
            break
          }
        }
      }
    }
  }
  if (!brain.initialized) {
    json(res, 503, {
      error: { message: '数字生命尚未初始化', type: 'not_initialized', code: 'not_initialized' },
    })
    return
  }
  try {
    const said = brain.speak(userText, { module: 'audio' })
    const content = typeof said?.text === 'string' ? said.text : '（它沉默了一会儿）'
    const model = body.model ?? 'digital-life'
    const respId = `resp_${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const outputItem = {
      type: 'message',
      role: 'assistant',
      content: [{ type: 'output_text', text: content }],
    }

    if (body.stream === true) {
      res.statusCode = 200
      res.setHeader('content-type', 'text/event-stream; charset=utf-8')
      res.setHeader('cache-control', 'no-store')
      res.setHeader('connection', 'keep-alive')
      res.setHeader('access-control-allow-origin', '*')
      res.write(`event: response.created\ndata: ${JSON.stringify({ type: 'response.created', response: { id: respId, object: 'response', status: 'in_progress', output: [] } })}\n\n`)
      const chunkSize = 6
      for (let i = 0; i < content.length; i += chunkSize) {
        const piece = content.slice(i, i + chunkSize)
        res.write(`event: response.output_text.delta\ndata: ${JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: piece })}\n\n`)
        await new Promise((r) => setTimeout(r, 20))
      }
      res.write(`event: response.completed\ndata: ${JSON.stringify({ type: 'response.completed', response: { id: respId, object: 'response', status: 'completed', output: [outputItem] } })}\n\n`)
      res.end()
      return
    }

    json(res, 200, {
      id: respId,
      object: 'response',
      created_at: Math.floor(Date.now() / 1000),
      status: 'completed',
      model,
      output: [outputItem],
      usage: { input_tokens: Math.round(userText.length / 2), output_tokens: Math.round(content.length / 2), total_tokens: 0 },
    })
  } catch (error) {
    json(res, 500, { error: { message: error instanceof Error ? error.message : String(error), type: 'internal' } })
  }
}

const server = createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    res.statusCode = 204
    res.setHeader('access-control-allow-origin', '*')
    res.setHeader('access-control-allow-headers', 'content-type, authorization')
    res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS')
    res.end()
    return
  }
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? HOST}`)
  const path = url.pathname

  try {
    // ---- 数据接口 ----
    if (path === '/api/status' && req.method === 'GET') {
      json(res, 200, { ok: true, ...brain.status(), workspace: resolveWorkspace() })
      return
    }
    if (path === '/api/init' && req.method === 'POST') {
      const body = await parseBody(req, res)
      if (body === null) return
      if (typeof body.workspace !== 'string' || body.workspace.trim() === '') {
        json(res, 400, { ok: false, error: '初始化必须由用户指定一个文件夹：请提供 workspace 字段' })
        return
      }
      const status = brain.init(body.workspace, {
        seed: Number.isInteger(body.seed) ? body.seed : undefined,
        name: typeof body.name === 'string' ? body.name : undefined,
      })
      json(res, 200, { ok: true, workspace: body.workspace, status })
      return
    }
    if (path === '/api/controller' && req.method === 'POST') {
      const body = await parseBody(req, res)
      if (body === null) return
      await handleController(req, res, body)
      return
    }
    if (path === '/api/persona' && req.method === 'GET') {
      json(res, 200, { ok: true, ...brain.persona({ recent: 6 }) })
      return
    }
    if (path === '/api/sense' && req.method === 'POST') {
      const body = await parseBody(req, res)
      if (body === null) return
      if (typeof body.text !== 'string' || body.text === '') {
        json(res, 400, { ok: false, error: 'text 字段必填' })
        return
      }
      if (!brain.initialized) {
        json(res, 400, { ok: false, error: '大脑未初始化' })
        return
      }
      const { writeFileSync } = await import('node:fs')
      const file = join(brain.workspace.paths.senses, `sense-${Date.now()}.txt`)
      writeFileSync(file, body.module ? JSON.stringify({ module: body.module, text: body.text }) : body.text, 'utf8')
      json(res, 200, { ok: true, file })
      return
    }

    // ---- AI 大模型接入（OpenAI 兼容：chatCompletions + responses）----
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await parseBody(req, res)
      if (body === null) return
      await handleChatCompletions(req, res, body)
      return
    }
    if (path === '/v1/responses' && req.method === 'POST') {
      const body = await parseBody(req, res)
      if (body === null) return
      await handleResponses(req, res, body)
      return
    }
    if (path === '/v1/models' && req.method === 'GET') {
      json(res, 200, {
        object: 'list',
        data: [
          { id: 'digital-life', object: 'model', created: 0, owned_by: 'plastic-memory' },
        ],
      })
      return
    }

    // ---- 独立窗口 UI ----
    if (path === '/ui' || path === '/') {
      const uiFile = join(__dirname, 'ui', 'index.html')
      if (existsSync(uiFile)) {
        res.statusCode = 200
        res.setHeader('content-type', 'text/html; charset=utf-8')
        res.setHeader('cache-control', 'no-store')
        res.end(readFileSync(uiFile, 'utf8'))
      } else {
        json(res, 404, { error: 'UI 未就绪' })
      }
      return
    }

    json(res, 404, { ok: false, error: `未知路径：${path}` })
  } catch (error) {
    json(res, 500, { ok: false, error: error instanceof Error ? error.message : String(error) })
  }
})

server.listen(PORT, HOST, () => {
  console.log(`[plastic-memory] 独立软件已启动：http://${HOST}:${PORT}`)
  console.log(`[plastic-memory] 聊天窗口 UI：http://${HOST}:${PORT}/ui`)
  console.log(`[plastic-memory] OpenAI 兼容接口：http://${HOST}:${PORT}/v1/chat/completions`)
  console.log(`[plastic-memory] 工作区：${resolveWorkspace() ?? '（未初始化，POST /api/init 指定）'}`)
})

server.on('error', (error) => {
  if (error && error.code === 'EADDRINUSE') {
    console.log('[plastic-memory] 服务已在运行（端口占用），本实例退出')
  } else {
    console.error('[plastic-memory] 启动失败：', error?.message ?? error)
  }
  process.exit(0)
})

process.on('SIGINT', () => {
  brain.dispose()
  process.exit(0)
})
process.on('SIGTERM', () => {
  brain.dispose()
  process.exit(0)
})
