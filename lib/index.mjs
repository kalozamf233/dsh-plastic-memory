// ============================================================================
// dsh-plastic-memory —— Node half（薄客户端）
// ----------------------------------------------------------------------------
// 架构变更：数字生命已独立为常驻服务（server/server.mjs，默认端口 9630）。
// 本插件不再内嵌大脑，职责变为：
//   1. 启动时确保独立服务在运行（未运行则自动拉起 detached 进程）
//   2. webServer 代理路由：DSH 侧边栏面板的所有数据与操作经同源代理转发到
//      独立服务的数据接口（/api/*）——侧边栏插件数据全部通过数据接口调取
//   3. DSH 工具：plastic_memory_* 全部改为 HTTP 调用独立服务
//   4. AI 大模型接入：独立服务暴露 OpenAI 兼容 /v1/chat/completions，
//      在 DSH 中添加自定义模型（baseURL=http://127.0.0.1:9630/v1，
//      model=digital-life）即可像大模型一样与它对话；响应不含 tool_calls，
//      与该 AI 聊天时不会调用其他 agent 工具
// ============================================================================

import { join, resolve } from 'node:path'
import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'

export const name = 'dsh-plastic-memory'
export const inject = ['webServer', 'tools']

export const ROUTE_PREFIX = '/plastic-memory'
export const STATUS_PATH = `${ROUTE_PREFIX}/status`
export const INIT_PATH = `${ROUTE_PREFIX}/init`
export const LOAD_PATH = `${ROUTE_PREFIX}/load`
export const CONTROLLER_PATH = `${ROUTE_PREFIX}/controller`
export const MEMORY_PATH = `${ROUTE_PREFIX}/memory`
export const PERSONA_PATH = `${ROUTE_PREFIX}/persona`
export const SENSE_PATH = `${ROUTE_PREFIX}/sense`

const SERVICE = process.env.PLASTIC_SERVICE ?? 'http://127.0.0.1:9630'
const PLUGIN_DIR = resolve(import.meta.dirname, '..')
const DSH_HOME = process.env.DSH_HOME ?? resolve(import.meta.dirname, '../../..')

function bootLog(message) {
  try {
    const dir = join(DSH_HOME, 'data', 'dsh-plastic-memory')
    mkdirSync(dir, { recursive: true })
    writeFileSync(join(dir, 'boot.log'), `[${new Date().toISOString()}] ${message}\n`, { flag: 'a' })
  } catch {
    /* ignore */
  }
}

/** 探测独立服务；未运行则自动拉起（detached，插件退出不杀服务）。 */
async function ensureService() {
  try {
    const res = await fetch(`${SERVICE}/api/status`, { signal: AbortSignal.timeout(1500) })
    if (res.ok) {
      bootLog('standalone service already running')
      return true
    }
  } catch {
    /* 未运行 → 拉起 */
  }
  const candidates = [
    join(DSH_HOME, 'dsh-runtimes', 'dsh-primary-runtime', 'dependencies', 'node', 'bin', 'node.exe'),
    'C:/Users/wsp/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/node/bin/node.exe',
  ]
  let nodeExe = null
  for (const c of candidates) {
    try {
      const { existsSync } = await import('node:fs')
      if (existsSync(c)) {
        nodeExe = c
        break
      }
    } catch {
      /* continue */
    }
  }
  if (!nodeExe) {
    bootLog('standalone service NOT started: node runtime not found')
    return false
  }
  try {
    // 显式传递工作区（保证独立服务继承 DSH 侧边栏初始化的工作区数据）
    let workspaceEnv = process.env.PLASTIC_WORKSPACE ?? ''
    if (!workspaceEnv) {
      try {
        const { readFileSync } = await import('node:fs')
        const lastFile = join(DSH_HOME, 'data', 'dsh-plastic-memory', 'last-workspace.json')
        workspaceEnv = JSON.parse(readFileSync(lastFile, 'utf8')).workspace ?? ''
      } catch {
        /* 无记录则让服务自行判断 */
      }
    }
    const child = spawn(nodeExe, [join(PLUGIN_DIR, 'server', 'server.mjs')], {
      detached: true,
      stdio: 'ignore',
      cwd: PLUGIN_DIR,
      windowsHide: true,
      env: { ...process.env, PLASTIC_WORKSPACE: workspaceEnv, DSH_HOME: DSH_HOME },
    })
    child.unref()
    bootLog('standalone service spawned (workspace=' + workspaceEnv + ')')
  } catch (error) {
    bootLog('standalone service spawn FAILED: ' + (error?.message ?? error))
  }
  return false
}

/** 同源代理：把 DSH 侧边栏请求转发到独立服务的数据接口。 */
async function proxy(path, method = 'GET', body = undefined) {
  const init = { method, headers: { 'content-type': 'application/json' } }
  if (body !== undefined) init.body = JSON.stringify(body)
  const res = await fetch(`${SERVICE}${path}`, init)
  const text = await res.text()
  let data
  try {
    data = JSON.parse(text)
  } catch {
    data = { ok: false, error: text.slice(0, 200) }
  }
  return { status: res.status, data }
}

function json(res, status, body, extraHeaders = {}) {
  res.statusCode = status
  res.setHeader('content-type', 'application/json; charset=utf-8')
  res.setHeader('cache-control', 'no-store')
  for (const [k, v] of Object.entries(extraHeaders)) res.setHeader(k, v)
  res.end(JSON.stringify(body))
}

async function readBody(req, limit = 256 * 1024) {
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

function isCrossOrigin(headers, host) {
  const origin = headers.origin
  if (origin === undefined || origin === 'null') return false
  try {
    return new URL(origin).host !== host
  } catch {
    return true
  }
}

async function parseBody(req, res) {
  if (isCrossOrigin(req.headers, req.headers.host)) {
    json(res, 403, { ok: false, error: '跨源请求被拒绝' })
    return null
  }
  const raw = await readBody(req)
  if (raw === null) {
    json(res, 413, { ok: false, error: '请求体过大' })
    return null
  }
  let body = {}
  try {
    body = JSON.parse(raw || '{}')
  } catch {
    json(res, 400, { ok: false, error: '请求体不是合法 JSON' })
    return null
  }
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    json(res, 400, { ok: false, error: '请求体必须是 JSON 对象' })
    return null
  }
  return body
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

export function apply(ctx) {
  bootLog('apply entered (thin client)')
  const webServer = typeof ctx.get === 'function' ? ctx.get('webServer') : undefined
  const tools = typeof ctx.get === 'function' ? ctx.get('tools') : undefined
  const dispose = typeof ctx.on === 'function' ? ctx.on('dispose', () => {}) : () => {}

  // 启动时确保独立软件在运行（数字生命由独立服务承载）
  ensureService().catch(() => {})

  const register = (path, handler) => {
    if (webServer === undefined) return
    webServer.register({
      kind: 'exact',
      path,
      handler: async (req, res) => {
        try {
          await handler(req, res)
        } catch (error) {
          json(res, 502, { ok: false, error: `独立服务调用失败：${error instanceof Error ? error.message : String(error)}（服务地址 ${SERVICE}）` })
        }
      },
    })
    bootLog('proxy route registered: ' + path)
  }

  // ---- 数据接口代理（侧边栏插件数据全部经此调取/操作）----
  register(STATUS_PATH, async (req, res) => {
    if (req.method !== 'GET') {
      json(res, 405, { ok: false, error: '仅支持 GET' }, { allow: 'GET' })
      return
    }
    const r = await proxy('/api/status')
    json(res, r.status, r.data)
  })

  register(INIT_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    if (typeof body.workspace !== 'string' || body.workspace.trim() === '') {
      json(res, 400, { ok: false, error: '初始化必须由用户指定一个文件夹：请提供 workspace 字段（工作区绝对路径）' })
      return
    }
    const r = await proxy('/api/init', 'POST', body)
    json(res, r.status, r.data)
  })

  register(LOAD_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    // 独立服务按 PLASTIC_WORKSPACE/上次记录恢复；此处转发 init 以切换工作区
    const r = await proxy('/api/init', 'POST', body)
    json(res, r.status, r.data)
  })

  register(CONTROLLER_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    const r = await proxy('/api/controller', 'POST', body)
    json(res, r.status, r.data)
  })

  register(MEMORY_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    // 经操控网络间接操控记忆网络：转交独立服务（服务内做格式化校验）
    const r = await proxy('/api/controller', 'POST', {
      action: 'hear',
      payload: { text: body.text ?? '', module: body.module, source: 'memory-api' },
    })
    json(res, r.status, r.data)
  })

  register(PERSONA_PATH, async (req, res) => {
    if (req.method !== 'GET') {
      json(res, 405, { ok: false, error: '仅支持 GET' }, { allow: 'GET' })
      return
    }
    const r = await proxy('/api/persona')
    json(res, r.status, r.data)
  })

  register(SENSE_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    const r = await proxy('/api/sense', 'POST', body)
    json(res, r.status, r.data)
  })

  // -------------------------------------------------------------------------
  // DSH 工具（全部经数据接口调用独立软件）
  // -------------------------------------------------------------------------

  const registerTool = (def) => {
    if (tools === undefined) return
    try {
      ctx.effect(() => tools.register(def))
    } catch (error) {
      bootLog('tool register FAILED: ' + def.name + ' :: ' + (error?.message ?? error))
    }
  }

  const toolOut = (description) => ({
    schema: { type: 'object', description },
    render(args, value) {
      return [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 2) }]
    },
  })
  const params = (properties, required = []) => ({ type: 'object', properties, required })
  const wrap = (fn) => async (args, exec) => {
    try {
      return await fn(args, exec)
    } catch (e) {
      return { error: e && e.message ? e.message : String(e) }
    }
  }
  const ctrl = async (action, payload = {}) => {
    const r = await proxy('/api/controller', 'POST', { action, payload })
    if (!r.data || r.data.ok !== true) {
      return { error: r.data?.error ?? `HTTP ${r.status}` }
    }
    return r.data.result
  }

  // 1. 状态速查
  registerTool({
    name: 'plastic_memory_status',
    description:
      '查看可塑性记忆数字生命的大脑状态（独立软件，经数据接口调取）：多网络（操控/记忆/分析/镜像/缺陷人格）的神经元与参数、训练/冻结模式、预测误差、内分泌分子浓度、情绪、人格向量、休眠与预存训练进度。进入任何与该生命体相关的操作前先调用以了解其当前状态。',
    parameters: params({}, []),
    output: toolOut('数字生命完整状态'),
    execute: wrap(async () => {
      const r = await proxy('/api/status')
      return r.data
    }),
  })

  // 2. 人格化数据（低 token 注入）
  registerTool({
    name: 'plastic_memory_persona',
    description:
      '读取数字生命的低 token 人格化数据：人格特质向量（亲和/竞争轴、道德五基础、大五倾向）、当前构建情绪、内分泌状态、近期重要感知记忆摘要。用于在会话开始时注入人格上下文，或需要了解"灵魂"当前状态时。数据量约 2-4KB。',
    parameters: params({}, []),
    output: toolOut('人格化数据摘要'),
    execute: wrap(async () => {
      const r = await proxy('/api/persona')
      return r.data
    }),
  })

  // 3. 听（感知输入）
  registerTool({
    name: 'plastic_memory_hear',
    description:
      '向数字生命输入感知（听）：文本会经操控神经网络编码为格式化数据后注入记忆神经网络，实时影响其权重（训练态），并进入预存文件（休眠自训练数据源）。module 可选 visual/audio/touch/memory/persona。用于让它"经历"对话内容或外部信息。',
    parameters: params(
      {
        text: { type: 'string', description: '感知内容（文本）' },
        module: { type: 'string', description: '感知通道：visual/audio/touch/memory/persona，默认 memory' },
      },
      ['text'],
    ),
    output: toolOut('感知注入结果（预测误差/情绪/内分泌线索）'),
    execute: wrap((args) => ctrl('hear', { text: args.text, module: args.module })),
  })

  // 4. 说（人格化反馈）
  registerTool({
    name: 'plastic_memory_speak',
    description:
      '让数字生命"说话"：返回基于其当前神经网络状态生成的人格化表达（情绪措辞、语料回答、语言素材联想、道德立场、社会倾向），经缺陷人格网络加工美化。可先传入 text 让它听见并回应。输出可作为人格化反馈嵌入回复。',
    parameters: params(
      {
        text: { type: 'string', description: '可选：说给它的内容（先听再说）' },
        module: { type: 'string', description: '听入通道，默认 audio' },
      },
      [],
    ),
    output: toolOut('人格化表达与人格/情绪数据'),
    execute: wrap((args) => ctrl('speak', { text: args.text ?? '', module: args.module })),
  })

  // 5. 读（联想检索）
  registerTool({
    name: 'plastic_memory_read',
    description:
      '检索数字生命的记忆：query 经操控网络编码后触发记忆神经网络的联想激活，返回记忆中与查询语义最相关的既往感知片段（含相似度/时间/情绪显著性）。用于了解它"记得什么、怎么想的"。',
    parameters: params(
      { query: { type: 'string', description: '检索内容/主题' } },
      ['query'],
    ),
    output: toolOut('联想检索结果（记忆片段按相似度排序）'),
    execute: wrap((args) => ctrl('read', { query: args.query })),
  })

  // 6. 写（强化写入）
  registerTool({
    name: 'plastic_memory_write',
    description:
      '向数字生命强化写入内容：文本经编码后以目标人格向量多次迭代训练记忆网络（默认 8 epochs），使内容内化为稳定人格/长期记忆。人格区学习率更高、倾向过拟合——反复写入同一内容会沉淀为其性格的一部分。冻结态下跳过训练。',
    parameters: params(
      {
        text: { type: 'string', description: '要写入的内容（信念/记忆/特质描述）' },
        module: { type: 'string', description: '写入通道，默认 persona' },
        epochs: { type: 'integer', description: '训练轮数 1-40，默认 8' },
      },
      ['text'],
    ),
    output: toolOut('写入结果（训练轮数与损失）'),
    execute: wrap((args) => ctrl('write', { text: args.text, module: args.module, epochs: args.epochs })),
  })

  // 7. 生命周期控制
  registerTool({
    name: 'plastic_memory_control',
    description:
      '控制数字生命生命周期（独立软件，经数据接口操作）：freeze（固定记忆网络权重为冻结快照）、thaw（恢复训练态）、reset（返祖到出厂基因权重）、save（立即保存权重）、grow（操控网络为记忆网络新增神经元）、prune（剪除低使用度神经元）、adjust（手动调节内分泌分子浓度）、mountExtension（挂载外部扩容神经网络）、unmountExtension（卸载外部扩容网络）、sleep（让它入睡，开始休眠自训练）、wake（唤醒它，先同步镜像再苏醒）、generateNetwork（自动生成可塑性记忆格式的新网络并挂载）、trainFlaw（用户训练缺陷人格网络，仅训练态可用）、analysisStatus（查看分析存储器）。',
    parameters: params(
      {
        action: { type: 'string', description: 'freeze/thaw/reset/save/grow/prune/adjust/mountExtension/unmountExtension/sleep/wake/generateNetwork/trainFlaw/analysisStatus' },
        layer: { type: 'integer', description: 'grow/prune 的目标层：0=分区层 1=联想层' },
        index: { type: 'integer', description: 'prune 的神经元下标（省略自动选最低使用度）' },
        molecules: { type: 'object', description: 'adjust 时释放的分子映射 {分子id: 量0-1}' },
        path: { type: 'string', description: 'mountExtension/unmountExtension/generateNetwork 的网络文件路径' },
        text: { type: 'string', description: 'trainFlaw 的训练文本' },
        epochs: { type: 'integer', description: 'trainFlaw 训练轮数 1-40，默认 8' },
      },
      ['action'],
    ),
    output: toolOut('控制操作结果'),
    execute: wrap((args) => ctrl(args.action, { layer: args.layer, index: args.index, molecules: args.molecules, path: args.path, text: args.text, epochs: args.epochs })),
  })

  bootLog('apply completed (thin client)')
  return dispose
}
