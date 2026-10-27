// 最终验收：修复后的数字生命端到端验证（DSH 运行实例）
const BASE = 'http://127.0.0.1:54042/plastic-memory'
const post = async (p, b) => (await fetch(BASE + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) })).json()
const get = async (p) => (await fetch(BASE + p)).json()

let failures = 0
const check = (name, cond, detail = '') => {
  if (cond) console.log(`  ✔ ${name}`)
  else { failures++; console.log(`  ✘ ${name} ${detail}`) }
}

console.log('== 0. 返祖出厂基因（清测试数据）==')
const reset = await post('/controller', { action: 'reset', payload: {} })
check('resetGenome', reset.ok === true)

console.log('== 1. 恒定感知 → 预测误差 EMA 收敛（核心学习验证）==')
const s0 = await get('/status')
const t = '这是数字生命的规律感知模式，用于验证预测加工学习'
const eFirst = (await post('/controller', { action: 'hear', payload: { text: t, module: 'memory' } })).result?.predError
for (let i = 0; i < 120; i++) await post('/controller', { action: 'hear', payload: { text: t, module: 'memory' } })
const s1 = await get('/status')
console.log(`  EMA: ${s0.memory.predErrorEma.toFixed(4)} → ${s1.memory.predErrorEma.toFixed(4)}（首步误差 ${eFirst?.toFixed(4)}）`)
check('预测误差 EMA 显著下降（学习真实发生）', s1.memory.predErrorEma < s0.memory.predErrorEma * 0.8, `${s0.memory.predErrorEma.toFixed(4)} -> ${s1.memory.predErrorEma.toFixed(4)}`)

console.log('== 2. 内分泌在无威胁时保持基线（修复验证）==')
const s2 = await get('/status')
const cort = s2.endocrine.molecules.cortisol.level
const lrMod = s2.endocrine.effects.learningRateMod
console.log(`  皮质醇 ${cort}，学习率调制 ${lrMod}`)
check('学习率调制不失控（<1.4）', lrMod < 1.4, `lrMod=${lrMod}`)

console.log('== 3. 情绪线索仍正常触发（威胁感知）==')
await post('/controller', { action: 'hear', payload: { text: '危险！有敌人靠近，我感到恐惧', module: 'visual' } })
const s3 = await get('/status')
check('威胁词仍能触发皮质醇', s3.endocrine.molecules.cortisol.level > 0.25, `cortisol=${s3.endocrine.molecules.cortisol.level}`)
check('情绪被构建', (s3.emotion.fear || 0) + (s3.emotion.anxiety || 0) > 0 || s3.emotion.arousal > 0)

console.log('== 4. 感知文件夹心跳扫描 ==')
await post('/sense', { text: '清晨的阳光照进房间，我听见鸟鸣', module: 'audio' })
await new Promise((r) => setTimeout(r, 2600))
const s4 = await get('/status')
check('感知文件被扫描', s4.scannedFiles >= 1, `scannedFiles=${s4.scannedFiles}`)
check('待处理清空', s4.pendingSenses === 0)

console.log('== 5. 人格刻写与表达 ==')
const w = await post('/controller', { action: 'write', payload: { text: '我坚信诚实和善良是最重要的品质', module: 'persona', epochs: 8 } })
check('write 刻写', w.ok === true)
const said = await post('/controller', { action: 'speak', payload: {} })
console.log('  它说:', said.result?.text)
check('speak 人格化表达', typeof said.result?.text === 'string' && said.result.text.length > 10)

console.log('== 6. 低 token 人格导出 ==')
const persona = await get('/persona')
check('persona < 8KB', JSON.stringify(persona).length / 1024 < 8)

console.log('== 7. 保存与状态健康 ==')
await post('/controller', { action: 'save', payload: {} })
const sf = await get('/status')
check('模式为训练态（发布态）', sf.mode === 'training')
check('心跳运行', sf.running === true)
check('无运行错误', sf.lastError === null || sf.lastError === undefined)
check('神经元结构完整', sf.memory.layerSizes[0] >= 204 && sf.memory.layerSizes[1] >= 128)

console.log('')
console.log(failures === 0 ? '★ 最终验收全部通过' : `✘ ${failures} 项失败`)
process.exitCode = failures > 0 ? 1 : 0
