// 插件激活后端到端验证：通过 DSH 中运行的插件 API 驱动数字生命
const BASE = 'http://127.0.0.1:57815/plastic-memory'

async function post(path, body) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  return r.json()
}
async function get(path) {
  const r = await fetch(BASE + path)
  return r.json()
}

const log = (k, v) => console.log(`  ${k}: ${v}`)
let failures = 0
function check(name, cond, detail = '') {
  if (cond) console.log(`  ✔ ${name}`)
  else {
    failures++
    console.log(`  ✘ ${name} ${detail}`)
  }
}

console.log('== 1. 听（感知注入）==')
const hear1 = await post('/controller', { action: 'hear', payload: { text: '你好，我是你的朋友，很高兴认识你，我们一起学习成长', module: 'audio' } })
log('响应 ok', hear1.ok)
check('hear 成功', hear1.ok === true)
check('返回预测误差', typeof hear1.result?.predError === 'number', JSON.stringify(hear1).slice(0, 200))

console.log('== 2. 情绪线索 → 内分泌反应 ==')
const hear2 = await post('/controller', { action: 'hear', payload: { text: '危险！前方有敌人来袭，我感到非常恐惧和紧张', module: 'visual' } })
const st1 = await get('/status')
const cortisol = st1.endocrine?.molecules?.cortisol?.level
const adrenaline = st1.endocrine?.molecules?.adrenaline?.level
log('皮质醇', cortisol, '肾上腺素', adrenaline)
check('威胁词触发皮质醇升高', cortisol > 0.15, `cortisol=${cortisol}`)
check('情绪被构建（非全零）', st1.emotion && (st1.emotion.valence !== 0 || st1.emotion.arousal !== 0), JSON.stringify(st1.emotion))

console.log('== 3. 重复感知 → 预测误差下降（学习）==')
const text3 = '这是一个每天都会重复出现的规律性模式句子'
const e1 = (await post('/controller', { action: 'hear', payload: { text: text3, module: 'memory' } })).result.predError
for (let i = 0; i < 30; i++) {
  await post('/controller', { action: 'hear', payload: { text: text3, module: 'memory' } })
}
const e2 = (await post('/controller', { action: 'hear', payload: { text: text3, module: 'memory' } })).result.predError
log('首次误差', e1, '→ 30 次后', e2)
check('预测误差下降（网络真的学会了）', e2 < e1, `${e1} -> ${e2}`)

console.log('== 4. 感知文件夹扫描（每秒心跳）==')
await post('/sense', { text: '傍晚的风很温柔，我听见窗外的鸟鸣，感到平静', module: 'audio' })
await new Promise((r) => setTimeout(r, 2600))
const st2 = await get('/status')
log('扫描文件数', st2.scannedFiles, '待处理', st2.pendingSenses)
check('感知文件夹被扫描', st2.scannedFiles >= 1, `scannedFiles=${st2.scannedFiles}`)

console.log('== 5. 说（人格化表达）==')
const said = await post('/controller', { action: 'speak', payload: {} })
log('表达', said.result?.text)
check('生成人格化文本', typeof said.result?.text === 'string' && said.result.text.length > 5)

console.log('== 6. 读（联想检索）与写（人格刻写）==')
const read = await post('/controller', { action: 'read', payload: { query: '朋友' } })
log('检索条数', read.result?.matches?.length)
check('read 返回匹配', Array.isArray(read.result?.matches))
const wrote = await post('/controller', { action: 'write', payload: { text: '我坚信公平与正义是最重要的价值', module: 'persona', epochs: 8 } })
check('write 执行', wrote.ok === true && wrote.result?.epochs === 8)
const st3 = await get('/status')
const fairness = st3.persona?.moralFoundations?.fairness
log('写入后公平值', fairness)
check('人格刻写生效（公平偏离 0）', Math.abs(fairness) > 0.01, `fairness=${fairness}`)

console.log('== 7. 低 token 人格导出 ==')
const persona = await get('/persona')
const kb = JSON.stringify(persona).length / 1024
log('persona 大小', kb.toFixed(1) + 'KB')
check('persona 导出（<8KB）', kb < 8)

console.log('== 8. 间接操控记忆网络（格式化数据校验）==')
const bad = await post('/memory', { module: 'memory', vector: [1, 2, 3] })
check('非法 vector 被拒绝', bad.ok === false && /128/.test(bad.error || ''), JSON.stringify(bad).slice(0, 150))
const okVec = await post('/memory', { module: 'memory', vector: Array(128).fill(0.1) })
check('合法 128 维 vector 被接受', okVec.ok === true && okVec.via === 'formatted-vector')

console.log('== 9. 神经元增删（操控网络权限）==')
const before = (await get('/status')).memory.layerSizes
const grow = await post('/controller', { action: 'grow', payload: { layer: 0 } })
const after = (await get('/status')).memory.layerSizes
log('层大小', JSON.stringify(before), '→', JSON.stringify(after))
check('操控网络增神经元', after[0] === before[0] + 1)
const prune = await post('/controller', { action: 'prune', payload: { layer: 0, index: after[0] - 1 } })
const afterPrune = (await get('/status')).memory.layerSizes
check('剪除神经元', afterPrune[0] === before[0], JSON.stringify(afterPrune))

console.log('== 10. freeze/thaw 生命周期 ==')
const froze = await post('/controller', { action: 'freeze', payload: {} })
check('freeze 固定权重', froze.ok === true && froze.result?.frozen === true)
const frozenSt = await get('/status')
check('冻结态生效', frozenSt.mode === 'frozen')
const thawed = await post('/controller', { action: 'thaw', payload: {} })
check('thaw 恢复训练态', thawed.ok === true)
const thawSt = await get('/status')
check('训练态恢复', thawSt.mode === 'training')

console.log('')
console.log(failures === 0 ? '★ 全部通过：数字生命端到端运行正常' : `✘ ${failures} 项失败`)
process.exitCode = failures > 0 ? 1 : 0
