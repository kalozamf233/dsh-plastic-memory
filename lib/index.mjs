// ============================================================================
// dsh-plastic-memory —— Node half（Cordis bundle 插件）
// ----------------------------------------------------------------------------
// 职责：
//   1. 持有 PlasticBrain 单例（双神经网络），启动时自动恢复上次工作区；
//   2. webServer 路由（前缀 /plastic-memory）：API 接口
//      · 直接操控【操控神经网络】（/controller）
//      · 经操控网络间接操控【记忆神经网络】（/memory，仅接受格式化数据）
//      · 初始化必须由用户指定工作区文件夹（/init）
//   3. 注册 DSH 工具（会话内调用）：status/persona/hear/speak/read/write/control
//   4. dispose 时保存权重并停心跳。
//
// 零外部依赖：只 import node: 内置模块与本地 nn 核心。
// ============================================================================

import { join, resolve } from 'node:path'
import { mkdirSync, readFileSync, writeFileSync, renameSync } from 'node:fs'
import { PlasticBrain } from './brain.mjs'
import { interpretEmotion } from './nn/memorynet.mjs'

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

const DSH_HOME = process.env.DSH_HOME ?? resolve(import.meta.dirname, '../../..')
const DATA_DIR = join(DSH_HOME, 'data', 'dsh-plastic-memory')
const LAST_WORKSPACE_FILE = join(DATA_DIR, 'last-workspace.json')
const BOOT_LOG = join(DATA_DIR, 'boot.log')

// ---------------------------------------------------------------------------
// 单例大脑
// ---------------------------------------------------------------------------

const brain = new PlasticBrain()

function bootLog(message) {
  try {
    mkdirSync(DATA_DIR, { recursive: true })
    writeFileSync(BOOT_LOG, `[${new Date().toISOString()}] ${message}\n`, { flag: 'a' })
  } catch {
    /* ignore */
  }
}

function readLastWorkspace() {
  try {
    return JSON.parse(readFileSync(LAST_WORKSPACE_FILE, 'utf8')).workspace ?? null
  } catch {
    return null
  }
}

function writeLastWorkspace(workspace) {
  try {
    mkdirSync(DATA_DIR, { recursive: true })
    const tmp = `${LAST_WORKSPACE_FILE}.tmp`
    writeFileSync(tmp, JSON.stringify({ workspace, at: new Date().toISOString() }))
    renameSync(tmp, LAST_WORKSPACE_FILE)
  } catch {
    /* ignore */
  }
}

// ---------------------------------------------------------------------------
// HTTP 小工具
// ---------------------------------------------------------------------------

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

/** 解析 JSON 请求体；失败时已写响应，返回 null。 */
async function parseBody(req, res) {
  if (isCrossOrigin(req.headers, req.headers.host)) {
    json(res, 403, { ok: false, error: '跨源请求被拒绝' })
    return null
  }
  const raw = await readBody(req)
  if (raw === null) {
    json(res, 413, { ok: false, error: '请求体过大（上限 256KB）' })
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

function err(res, error) {
  json(res, 500, { ok: false, error: error instanceof Error ? error.message : String(error) })
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

export function apply(ctx) {
  bootLog('apply entered')
  const webServer = typeof ctx.get === 'function' ? ctx.get('webServer') : undefined
  const tools = typeof ctx.get === 'function' ? ctx.get('tools') : undefined
  const dispose = typeof ctx.on === 'function' ? ctx.on('dispose', () => brain.dispose()) : () => {}

  // 启动时自动恢复上次工作区（首次使用需 /init 指定文件夹）
  const last = readLastWorkspace()
  if (last) {
    try {
      brain.load(last)
      bootLog(`auto-loaded workspace: ${last}`)
    } catch (error) {
      bootLog(`auto-load FAILED: ${error?.message ?? error}`)
    }
  }

  // -------------------------------------------------------------------------
  // API 路由
  // -------------------------------------------------------------------------

  const register = (path, handler) => {
    if (webServer === undefined) return
    webServer.register({
      kind: 'exact',
      path,
      handler: async (req, res) => {
        try {
          await handler(req, res)
        } catch (error) {
          err(res, error)
        }
      },
    })
    bootLog('route registered: ' + path)
  }

  // ---- GET /status：双网络完整状态 ----
  register(STATUS_PATH, (req, res) => {
    if (req.method !== 'GET') {
      json(res, 405, { ok: false, error: '仅支持 GET' }, { allow: 'GET' })
      return
    }
    json(res, 200, { ok: true, ...brain.status() })
  })

  // ---- POST /init：初始化（用户必须指定工作区文件夹） ----
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
    const status = brain.init(body.workspace, {
      seed: Number.isInteger(body.seed) ? body.seed : undefined,
      name: typeof body.name === 'string' && body.name.trim() !== '' ? body.name : undefined,
    })
    writeLastWorkspace(body.workspace)
    json(res, 200, { ok: true, workspace: body.workspace, status })
  })

  // ---- POST /load：恢复 / 切换工作区 ----
  register(LOAD_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    if (typeof body.workspace !== 'string' || body.workspace.trim() === '') {
      json(res, 400, { ok: false, error: 'workspace 字段必填（工作区文件夹路径）' })
      return
    }
    const status = brain.load(body.workspace)
    writeLastWorkspace(body.workspace)
    json(res, 200, { ok: true, workspace: body.workspace, status })
  })

  // ---- POST /controller：直接操控【操控神经网络】 ----
  register(CONTROLLER_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    const action = typeof body.action === 'string' ? body.action : ''
    const payload = body.payload && typeof body.payload === 'object' ? body.payload : {}
    let result
    switch (action) {
      case 'hear':
        result = brain.hear(payload.text ?? '', { module: payload.module, source: 'controller-api' })
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
        if (Array.isArray(payload.paths)) {
          result = brain.mountExtensions(payload.paths)
        } else {
          result = brain.mountExtension(payload.path ?? '', { seed: Number.isInteger(payload.seed) ? payload.seed : undefined })
        }
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
      default:
        json(res, 400, {
          ok: false,
          error: '未知 action。支持：hear/speak/read/write/adjust/freeze/thaw/reset/save/grow/prune/adjustLr/scan/mountExtension/unmountExtension/sleep/wake',
        })
        return
    }
    json(res, 200, { ok: true, action, result })
  })

  // ---- POST /memory：经操控网络间接操控【记忆神经网络】 ----
  // 记忆网络永远只接受由操控网络提供的格式化数据：
  //   · text 路径：文本先经操控网络编码成格式化向量，再注入
  //   · vector 路径：必须是 128 维格式化向量（格式校验后直接注入）
  register(MEMORY_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    const module = ['visual', 'audio', 'touch', 'memory', 'persona'].includes(body.module)
      ? body.module
      : 'memory'

    if (typeof body.text === 'string' && body.text !== '') {
      // 文本 → 操控网络编码 → 格式化数据 → 记忆网络
      const r = brain.hear(body.text, { module, source: 'memory-api' })
      json(res, 200, { ok: true, via: 'controller-encoder', module, result: r })
      return
    }
    if (Array.isArray(body.vector)) {
      if (body.vector.length !== 128 || body.vector.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
        json(res, 400, {
          ok: false,
          error: '记忆神经网络只接受由操控神经网络提供的格式化数据：vector 必须是 128 维数值数组（或提供 text 由操控网络编码）',
        })
        return
      }
      const latent = new Float64Array(body.vector)
      const eff = brain.endocrine.effects()
      const train = brain.mode === 'training'
      const result = brain.memory.observe(latent, {
        module,
        salience: eff.memorySalience,
        lrScale: eff.learningRateMod * brain.controller.lrMod,
        emotionTarget: eff.emotionTarget,
        train,
      })
      brain.perceptions++
      json(res, 200, {
        ok: true,
        via: 'formatted-vector',
        module,
        train,
        result: {
          predError: Number(result.predError.toFixed(5)),
          emotion: interpretEmotion(result.emo),
        },
      })
      return
    }
    json(res, 400, { ok: false, error: '需要 text（经操控网络编码）或 vector（128 维格式化数据）' })
  })

  // ---- GET /persona：人格化数据（低 token，供外部模型接入） ----
  register(PERSONA_PATH, (req, res) => {
    if (req.method !== 'GET') {
      json(res, 405, { ok: false, error: '仅支持 GET' }, { allow: 'GET' })
      return
    }
    json(res, 200, { ok: true, ...brain.persona({ recent: 6 }) })
  })

  // ---- POST /sense：向感知文件夹写入外部世界信号 ----
  register(SENSE_PATH, async (req, res) => {
    if (req.method !== 'POST') {
      json(res, 405, { ok: false, error: '仅支持 POST' }, { allow: 'POST' })
      return
    }
    const body = await parseBody(req, res)
    if (body === null) return
    if (typeof body.text !== 'string' || body.text === '') {
      json(res, 400, { ok: false, error: 'text 字段必填（感知内容）' })
      return
    }
    if (!brain.initialized) {
      json(res, 400, { ok: false, error: '大脑未初始化：请先 POST /init 指定工作区文件夹' })
      return
    }
    const sensesDir = brain.workspace.paths.senses
    const file = join(sensesDir, `sense-${Date.now()}-${Math.floor(Math.random() * 1e4)}.txt`)
    const content = body.module
      ? JSON.stringify({ module: body.module, text: body.text })
      : body.text
    writeFileSync(file, content, 'utf8')
    json(res, 200, { ok: true, file, note: '已写入感知文件夹，将在下一秒心跳被操控网络扫描并传入记忆网络' })
  })

  // -------------------------------------------------------------------------
  // DSH 工具（会话内接入：为模型提供低 token 长期人格化数据）
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

  // 1. 状态速查
  registerTool({
    name: 'plastic_memory_status',
    description:
      '查看可塑性记忆数字生命的大脑状态：两个神经网络（操控网络/记忆网络）的神经元数与参数、训练/冻结模式、预测误差、内分泌分子浓度、情绪、人格向量、感知扫描统计。进入任何与该生命体相关的操作前先调用以了解其当前状态。',
    parameters: params({}, []),
    output: toolOut('双神经网络完整状态'),
    execute: wrap(() => brain.status()),
  })

  // 2. 人格化数据（低 token 注入）
  registerTool({
    name: 'plastic_memory_persona',
    description:
      '读取数字生命的低 token 人格化数据：人格特质向量（亲和/竞争轴、道德五基础、大五倾向）、当前构建情绪、内分泌状态、近期重要感知记忆摘要。用于在会话开始时注入人格上下文，或需要了解"灵魂"当前状态时。数据量约 2-4KB。',
    parameters: params({}, []),
    output: toolOut('人格化数据摘要'),
    execute: wrap(() => brain.persona({ recent: 6 })),
  })

  // 3. 听（感知输入）
  registerTool({
    name: 'plastic_memory_hear',
    description:
      '向数字生命输入感知（听）：文本会经操控神经网络编码为格式化数据后注入记忆神经网络，实时影响其权重（训练态）。module 可选 visual/audio/touch/memory/persona，对应大脑不同感知分区。用于让它"经历"对话内容或外部信息。',
    parameters: params(
      {
        text: { type: 'string', description: '感知内容（文本）' },
        module: { type: 'string', description: '感知通道：visual/audio/touch/memory/persona，默认 memory' },
      },
      ['text'],
    ),
    output: toolOut('感知注入结果（预测误差/情绪/内分泌线索）'),
    execute: wrap((args) => brain.hear(args.text, { module: args.module, source: 'tool' })),
  })

  // 4. 说（人格化反馈）
  registerTool({
    name: 'plastic_memory_speak',
    description:
      '让数字生命"说话"：返回基于其当前神经网络状态生成的人格化表达（情绪措辞、深刻记忆回想、道德立场、社会倾向），以及完整人格/情绪数据。可先传入 text 让它听见并回应。输出可作为人格化反馈嵌入回复。',
    parameters: params(
      {
        text: { type: 'string', description: '可选：说给它的内容（先听再说）' },
        module: { type: 'string', description: '听入通道，默认 audio' },
      },
      [],
    ),
    output: toolOut('人格化表达与人格/情绪数据'),
    execute: wrap((args) => brain.speak(args.text ?? '', { module: args.module })),
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
    execute: wrap((args) => brain.read(args.query, {})),
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
    execute: wrap((args) => brain.write(args.text, { module: args.module, epochs: args.epochs })),
  })

  // 7. 生命周期控制
  registerTool({
    name: 'plastic_memory_control',
    description:
      '控制数字生命生命周期：freeze（训练完毕，固定记忆网络权重为冻结快照）、thaw（恢复训练态）、reset（记忆网络返祖到出厂基因权重）、save（立即保存权重到工作区）、grow（操控网络为记忆网络新增神经元）、prune（剪除低使用度神经元）、adjust（手动调节内分泌分子浓度）、mountExtension（挂载外部扩容神经网络，path 指向未训练网络文件）、unmountExtension（卸载外部扩容网络）、sleep（让它入睡）、wake（唤醒它）。',
    parameters: params(
      {
        action: { type: 'string', description: 'freeze/thaw/reset/save/grow/prune/adjust/mountExtension/unmountExtension/sleep/wake' },
        layer: { type: 'integer', description: 'grow/prune 的目标层：0=分区层 1=联想层' },
        index: { type: 'integer', description: 'prune 的神经元下标（省略自动选最低使用度）' },
        molecules: { type: 'object', description: 'adjust 时释放的分子映射 {分子id: 量0-1}' },
        path: { type: 'string', description: 'mountExtension/unmountExtension 的扩展网络文件路径' },
      },
      ['action'],
    ),
    output: toolOut('控制操作结果'),
    execute: wrap((args) => {
      switch (args.action) {
        case 'freeze':
          return brain.freeze()
        case 'thaw':
          return brain.thaw()
        case 'reset':
          return brain.resetGenome()
        case 'save':
          brain.save()
          return { saved: true }
        case 'grow':
          return brain.structuralCommand('grow', { layer: args.layer })
        case 'prune':
          return brain.structuralCommand('prune', { layer: args.layer, index: args.index })
        case 'adjust':
          return brain.adjustMolecules(args.molecules ?? {})
        case 'mountExtension':
          return brain.mountExtension(args.path ?? '')
        case 'unmountExtension':
          return brain.unmountExtension(args.path ?? '')
        case 'sleep':
          return brain.sleepNow()
        case 'wake':
          return brain.wakeNow()
        default:
          return { error: '未知 action：freeze/thaw/reset/save/grow/prune/adjust/mountExtension/unmountExtension/sleep/wake' }
      }
    }),
  })

  bootLog('apply completed')
  return dispose
}
