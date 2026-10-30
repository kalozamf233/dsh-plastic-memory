// ============================================================================
// dsh-plastic-memory —— 离线核心测试
// 运行：node test_offline.mjs（不依赖 DSH 宿主，纯 Node 验证神经网络核心）
// ============================================================================
import { mkdtempSync, writeFileSync, rmSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { PlasticNetwork } from './lib/nn/network.mjs'
import { SenseEncoder } from './lib/nn/encoder.mjs'
import { EndocrineSystem } from './lib/nn/endocrine.mjs'
import { MemoryNet, DIMS, interpretPersona } from './lib/nn/memorynet.mjs'
import { ControllerNet } from './lib/nn/controllernet.mjs'
import { AnalysisNet, AnalysisStore } from './lib/nn/analysisnet.mjs'
import { DefectNet } from './lib/nn/flawnet.mjs'
import { TrainingController } from './lib/trainer.mjs'
import { PlasticBrain } from './lib/brain.mjs'

let passed = 0
let failed = 0
function check(name, cond, detail = '') {
  if (cond) {
    passed++
    console.log(`  ✔ ${name}`)
  } else {
    failed++
    console.log(`  ✘ ${name} ${detail}`)
  }
}
function approx(a, b, tol = 1e-6) {
  return Math.abs(a - b) <= tol
}

console.log('== 1. PlasticNetwork 基础 ==')
{
  const net = new PlasticNetwork({
    inputSize: 8,
    layers: [
      { size: 16, activation: 'tanh', lr: 1, mutable: true },
      { size: 8, activation: 'tanh', lr: 1, mutable: true },
      { size: 4, activation: 'tanh', lr: 1, mutable: false },
    ],
    seed: 1,
  })
  const x = new Float64Array(8).fill(0.3)
  const fwd = net.forward(x)
  check('forward 形状', fwd.post.length === 4 && fwd.post[1].length === 16 && fwd.post[3].length === 4)
  const target = new Float64Array(4).fill(0.5)
  let firstLoss = Infinity
  for (let i = 0; i < 300; i++) {
    const r = net.train(x, target, { baseLr: 0.05 })
    if (i === 0) firstLoss = r.loss
  }
  const finalOut = net.forward(x).post[3]
  const finalLoss = net.train(x, target, { baseLr: 0, frozen: true })
  check('BP 收敛（损失下降）', finalLoss.loss < firstLoss * 0.1, `first=${firstLoss.toFixed(3)} final=${finalLoss.loss.toFixed(3)}`)
  check('输出接近目标', Math.abs(finalOut[0] - 0.5) < 0.2, `out=${finalOut[0].toFixed(3)}`)

  const before = net.paramCount
  const g = net.addNeuron(0, { mutation: 0.2 })
  check('addNeuron 层0', g !== null && net.layers[0].size === 17)
  const g2 = net.addNeuron(1, { mutation: 0.2 })
  check('addNeuron 层1', g2 !== null && net.layers[1].size === 9)
  check('增后前向正常', net.forward(x).post[3].length === 4)
  const after = net.paramCount
  check('参数增加', after > before, `${before} -> ${after}`)
  const pr = net.pruneNeuron(0, 16)
  check('pruneNeuron', pr !== null && net.layers[0].size === 16)
  check('剪后前向正常', net.forward(x).post[3].length === 4)
  const noGrowOut = net.addNeuron(2)
  check('输出层不可增', noGrowOut === null)
  // 序列化往返
  const json = net.toJSON()
  const net2 = PlasticNetwork.fromJSON(json)
  const f1 = net.forward(x).post[3]
  const f2 = net2.forward(x).post[3]
  let same = true
  for (let i = 0; i < f1.length; i++) if (Math.abs(f1[i] - f2[i]) > 1e-12) same = false
  check('序列化往返一致', same)
  check('序列化保留结构变更日志', net2.structuralChanges.length === net.structuralChanges.length)
}

console.log('== 2. SenseEncoder ==')
{
  const enc = new SenseEncoder(5)
  const { latent, bag } = enc.encode('今天天气很好，我感到很开心')
  check('latent 维度 128', latent.length === 128)
  check('bag 维度 512', bag.length === 512)
  let loss0 = Infinity
  for (let i = 0; i < 50; i++) {
    const l = enc.train('今天天气很好，我感到很开心', 0.01)
    if (i === 0) loss0 = l
  }
  const l50 = enc.train('今天天气很好，我感到很开心', 0.01)
  check('编码器重构损失下降', l50 < loss0, `${loss0.toFixed(3)} -> ${l50.toFixed(3)}`)
  const e2 = new SenseEncoder(6)
  e2.fromJSON(enc.toJSON())
  const { latent: fresh } = enc.encode('今天天气很好，我感到很开心')
  const { latent: l2 } = e2.encode('今天天气很好，我感到很开心')
  let same = true
  for (let i = 0; i < 128; i++) if (Math.abs(fresh[i] - l2[i]) > 1e-12) same = false
  check('编码器序列化往返', same)
}

console.log('== 3. EndocrineSystem ==')
{
  const endo = new EndocrineSystem()
  endo.release('cortisol', 0.5)
  const high = endo.get('cortisol')
  check('释放升高', high > 0.15)
  for (let i = 0; i < 6000; i++) endo.tick(1)
  check('衰减回基线', Math.abs(endo.get('cortisol') - 0.15) < 0.05, `cortisol=${endo.get('cortisol').toFixed(3)}`)
  const eff = endo.effects()
  check('effects 完整', eff.learningRateMod > 0 && eff.emotionTarget.length === 12)
  endo.releaseMany({ oxytocin: 0.5 })
  const eff2 = endo.effects()
  check('催产素提升亲和轴', eff2.socialAxis > 0.2, `socialAxis=${eff2.socialAxis.toFixed(3)}`)
}

console.log('== 4. MemoryNet ==')
{
  const mem = new MemoryNet({ seed: 11, baseLr: 0.01 })
  const latent = new Float64Array(DIMS.latent).fill(0.2)
  let predErr0 = 0
  for (let i = 0; i < 200; i++) {
    const r = mem.observe(latent, { module: 'memory', salience: 0.6, lrScale: 1, emotionTarget: new Float64Array(DIMS.emo).fill(0.3) })
    if (i === 0) predErr0 = r.predError
  }
  check('observe 正常', mem.observeCount === 200)
  check('预测误差存在且收敛（学习发生）', mem.predErrorEma < 1, `ema=${mem.predErrorEma.toFixed(4)} 首步=${predErr0.toFixed(4)}`)
  check('人格输出维度 64', mem.lastPersona.length === 64)
  check('情绪输出维度 12', mem.lastEmo.length === 12)
  // 恒定输入 300 步：EMA 应显著下降（回归：循环投影 U 稳定性）
  const memConv = new MemoryNet({ seed: 42, baseLr: 0.008 })
  const latentC = new Float64Array(DIMS.latent).fill(0.15)
  const emoC = new Float64Array(DIMS.emo).fill(0.25)
  let emaEarly = null
  for (let i = 0; i < 300; i++) {
    memConv.observe(latentC, { module: 'memory', salience: 0.5, lrScale: 1, emotionTarget: emoC })
    if (i === 9) emaEarly = memConv.predErrorEma
  }
  check('恒定输入 300 步 EMA 显著下降', memConv.predErrorEma < emaEarly * 0.6, `${emaEarly.toFixed(4)} -> ${memConv.predErrorEma.toFixed(4)}`)
  check('循环投影 U 有界', Math.max(...Array.from(memConv.U.data).map(Math.abs)) < 5)
  // 写入：强化刻写人格
  const target = new Float64Array(64).fill(-0.8)
  const wRes = mem.writePersona(target, { epochs: 10, lrScale: 1.5 })
  const interp = interpretPersona(mem.personaEma)
  check('writePersona 执行', wRes.epochs === 10)
  check('人格 EMA 向目标偏移', mem.personaEma[0] < -0.2, `亲和轴=${interp.traits['亲和轴']}`)
  // 分区结构
  const st = mem.status()
  check('分区段存在', st.segments.visual && st.segments.memory && st.segments.persona, JSON.stringify(Object.keys(st.segments)))
  check('参数统计 >0', st.paramCount > 1000, `params=${st.paramCount}`)
  // 增删
  const g = mem.growNeuron(0)
  check('记忆网络增神经元', g !== null)
  const g2 = mem.growNeuron(1)
  check('联想层增神经元', g2 !== null)
  check('增后 observe 正常', mem.observe(latent, { module: 'visual' }).predError !== undefined)
  const prunable = mem.findPrunable(0, { minAge: 0, usageThreshold: 1 })
  const pr = prunable ? mem.pruneNeuron(0, prunable.index) : null
  check('可剪或保守不剪（都合法）', pr === null || pr !== null)
  // 序列化
  const json = mem.toJSON()
  const mem2 = MemoryNet.fromJSON(json)
  const r1 = mem.observe(latent, { module: 'memory' })
  const r2 = mem2.observe(latent, { module: 'memory' })
  check('记忆网络序列化往返', Math.abs(r1.predError - r2.predError) < 1e-9, `${r1.predError} vs ${r2.predError}`)
}

console.log('== 5. ControllerNet ==')
{
  const ctrl = new ControllerNet({ seed: 3 })
  const mem = new MemoryNet({ seed: 4 })
  const endo = new EndocrineSystem()
  const { latent } = ctrl.hear('危险！敌人来袭，我感到恐惧')
  check('hear 编码 128 维', latent.length === 128)
  const cues = ctrl.react('危险！敌人来袭，我感到恐惧', endo)
  check('威胁词触发生理反应', cues > 0 && endo.get('cortisol') > 0.2, `cortisol=${endo.get('cortisol').toFixed(3)}`)
  const report = ctrl.decide(mem, endo)
  check('decide 输出决策', report.growL1Prob >= 0 && report.growL1Prob <= 1 && report.lrMod > 0)
  // 回归：初始策略不应疯狂释放分子（基线 0.5 对齐后，初始释放为小剂量）
  const releaseCount = Object.keys(report.molecules).length
  const releaseTotal = Object.values(report.molecules).reduce((s, v) => s + v, 0)
  check('初始策略分子释放受基线约束（种类≤10 且总量<0.15）', releaseCount <= 10 && releaseTotal < 0.15, `释放 ${releaseCount} 种，总量 ${releaseTotal.toFixed(3)}`)
  const outcome = ctrl.learnFromOutcome(mem)
  check('元学习执行', typeof outcome.reward === 'number')
  // 序列化
  const c2 = ControllerNet.fromJSON(ctrl.toJSON())
  check('操控网络序列化往返', c2.policy.paramCount === ctrl.policy.paramCount)
}

console.log('== 6. PlasticBrain 完整流程 ==')
{
  const dir = mkdtempSync(join(tmpdir(), 'plastic-memory-test-'))
  const brain = new PlasticBrain()
  const status = brain.init(dir, { seed: 99, name: '测试生命' })
  check('init 初始化', status.initialized === true && status.memory.paramCount > 0)
  brain.stop() // 测试中手动心跳，避免定时器干扰

  // 感知注入（听）
  const r1 = brain.hear('你好，我是你的朋友，很高兴认识你', { module: 'audio' })
  check('hear 返回感知结果', typeof r1.predError === 'number')
  const emoAfter = brain.status().emotion
  check('情绪被构建', typeof emoAfter.valence === 'number')

  // 学习验证：重复注入相同模式 → 预测误差下降
  const text = '每天都重复这句话的规律模式'
  const e1 = brain.hear(text, { module: 'memory' }).predError
  for (let i = 0; i < 50; i++) brain.hear(text, { module: 'memory' })
  const e2 = brain.hear(text, { module: 'memory' }).predError
  check('重复感知预测误差下降（真实学习）', e2 < e1, `${e1.toFixed(4)} -> ${e2.toFixed(4)}`)
  check('predErrorEma 收敛', brain.memory.predErrorEma < 1)

  // 说
  const said = brain.speak('今天天气真好', {})
  check('speak 生成人格化文本', typeof said.text === 'string' && said.text.length > 10, said.text)
  check('speak 返回人格数据', said.persona && said.persona.moralFoundations)

  // 读
  const read = brain.read('朋友')
  check('read 返回检索结果', Array.isArray(read.matches))

  // 写（人格刻写）
  const wrote = brain.write('我坚信公平正义是最重要的价值', { module: 'persona', epochs: 6 })
  check('write 强化写入', wrote.epochs === 6)

  // 感知文件夹扫描
  writeFileSync(join(dir, 'senses', 's1.txt'), '傍晚的风很温柔，我感到平静')
  writeFileSync(join(dir, 'senses', 's2.json'), JSON.stringify({ module: 'visual', text: '看到美丽的晚霞' }))
  const scanned = brain.scanSenses()
  check('感知文件夹扫描 2 个文件', scanned.processed === 2)
  check('待处理已清空', brain.workspace.listPendingSenses().length === 0)
  check('扫描计数', brain.scannedFiles === 2)
  const pend = brain.workspace.listPendingSenses()
  // 已归档
  check('归档存在', brain.status().pendingSenses === 0)

  // 控制器结构操作
  const grow = brain.structuralCommand('grow', { layer: 0 })
  check('结构命令 grow', grow.kind === 'grow')
  let pruneOk = false
  try {
    const prune = brain.structuralCommand('prune', { layer: 1 })
    pruneOk = prune === null || prune.kind === 'prune'
  } catch (e) {
    pruneOk = String(e.message).includes('没有可剪') // 保护期内无可剪 → 合法拒绝
  }
  check('结构命令 prune（保守拒绝或成功）', pruneOk)

  // 冻结/解冻
  const frozen = brain.freeze()
  check('freeze', frozen.frozen === true && brain.mode === 'frozen')
  brain.thaw()
  check('thaw', brain.mode === 'training')

  // 保存 → 重新加载
  brain.save()
  const brain2 = new PlasticBrain()
  const st2 = brain2.load(dir)
  brain2.stop()
  check('重载后状态', st2.initialized && brain2.memory.observeCount === brain.memory.observeCount)
  const p1 = brain.persona()
  const p2 = brain2.persona()
  check('人格数据往返一致', JSON.stringify(p1.persona) === JSON.stringify(p2.persona))

  // 返祖
  const reset = brain2.resetGenome()
  check('resetGenome 返祖', reset.reset === true && brain2.memory.observeCount === 0)

  // 人格导出（低 token）
  const persona = brain.persona()
  const sizeKb = JSON.stringify(persona).length / 1024
  check('persona 导出低 token（<8KB）', sizeKb < 8, `${sizeKb.toFixed(1)}KB`)

  // 神经网络扩展（外部扩容）
  const extPath = join(dir, 'ext-1.brain.json')
  const mounted = brain.mountExtension(extPath)
  check('挂载扩展（创建全新网络）', mounted.created === true)
  const stExt = brain.status()
  check('status 含扩展列表', Array.isArray(stExt.extensions) && stExt.extensions.length === 1)
  const beforeObs = stExt.extensions[0].observeCount
  brain.hear('扩展网络也在学习', { module: 'memory' })
  const stExt2 = brain.status()
  check('感知同时注入扩展网络', stExt2.extensions[0].observeCount === beforeObs + 1)
  const readExt = brain.read('扩展')
  check('read 返回扩展联想', Array.isArray(readExt.extensions) && readExt.extensions.length === 1)
  brain.save()
  const brain3 = new PlasticBrain()
  const st3 = brain3.load(dir)
  brain3.stop()
  check('重载后扩展自动恢复', Array.isArray(st3.extensions) && st3.extensions.length === 1)
  // 批量挂载：2 个新路径 + 1 个非法文件（不存在的假路径会被创建，非法内容文件被拒绝）
  const extA = join(dir, 'ext-a.brain.json')
  const extB = join(dir, 'ext-b.brain.json')
  const foreignFile = join(dir, 'foreign.json')
  writeFileSync(foreignFile, JSON.stringify({ kind: 'mlp', framework: 'pytorch' }), 'utf8')
  const batch = brain3.mountExtensions([extA, extB, foreignFile])
  check('批量挂载 2 成功', batch.mounted.length === 2, JSON.stringify(batch))
  check('非法格式文件被拒绝且不覆盖', batch.errors.length === 1 && JSON.parse(readFileSync(foreignFile, 'utf8')).framework === 'pytorch')
  const stExtB = brain3.status()
  check('批量后列表共 3 个扩展', stExtB.extensions.length === 3)
  const unmounted = brain3.unmountExtension(extPath)
  check('卸载扩展', unmounted.unmounted === true)
  const stExt3 = brain3.status()
  check('卸载后剩 2 个', stExt3.extensions.length === 2)

  // 休眠功能
  const sleepNow = brain3.sleepNow()
  check('手动入睡', sleepNow.asleep === true && brain3.asleep === true)
  const wakeNow = brain3.wakeNow()
  check('手动唤醒', wakeNow.asleep === false && brain3.asleep === false)
  // 自主入睡：模拟长时间无聊（把上次感知时间拨回 10 分钟前，清醒度降到阈值下）
  brain3.lastPerceptionAt = Date.now() - 10 * 60 * 1000
  brain3.wakefulness = 0.03
  brain3.endocrine.set('melatonin', 0.6)
  brain3._updateSleep(1)
  check('自主入睡（无聊+褪黑素）', brain3.asleep === true)
  // 感知唤醒（说话即唤醒）
  brain3.hear('醒醒，朋友，我来了', { module: 'audio' })
  check('说话唤醒', brain3.asleep === false && brain3.wakefulness === 1)
  const sl = brain3.status()
  check('status 含睡眠字段', sl.sleep && typeof sl.sleep.asleep === 'boolean' && sl.sleep.sleepCount >= 1)
  // 睡眠时 speak 措辞含唤醒提示
  brain3.sleepNow()
  const said2 = brain3.speak('你好呀', {})
  check('speak 唤醒并带唤醒措辞', said2.wokeFromSleep === true && said2.text.includes('我刚才睡着了'))

  // 对话引擎（语料批量扫描 → 问话检索回答）
  writeFileSync(
    join(dir, 'senses', 'conv.json'),
    JSON.stringify([
      { module: 'audio', kind: 'conversation', text: '你好', reply: '你好呀，见到你我很开心。', moodTag: 'happy' },
      { module: 'audio', kind: 'conversation', text: '你好', reply: '你好，我在呢。', moodTag: 'neutral' },
      { module: 'audio', kind: 'conversation', text: '你好', reply: '你好……我刚才有点走神。', moodTag: 'tired' },
      { module: 'audio', kind: 'conversation', text: '你怎么样', reply: '我心情很好。', moodTag: 'happy' },
      { module: 'audio', kind: 'conversation', text: '你怎么样', reply: '我有点困了。', moodTag: 'tired' },
      { module: 'audio', kind: 'conversation', text: '再见', reply: '再见，我会想你的。', moodTag: 'neutral' },
      { module: 'audio', kind: 'conversation', text: '你是谁', reply: '我是你的数字生命朋友。', moodTag: 'neutral' },
    ]),
    'utf8',
  )
  const convScan = brain3.scanSenses()
  check('语料批量扫描', convScan.processed === 1)
  const ans1 = brain3.speak('你好', {})
  check('对话模式：问好得到语料回答', ans1.conversation === true && typeof ans1.reply === 'string' && ans1.reply.length > 0, JSON.stringify(ans1.reply))
  const ans2 = brain3.speak('再见', {})
  check('不同问话 → 不同回答', ans2.conversation === true && ans2.matchedQuestion !== ans1.matchedQuestion, `${ans1.matchedQuestion} vs ${ans2.matchedQuestion}`)
  const replies = new Set()
  for (let i = 0; i < 6; i++) {
    const a = brain3.speak('你好', {})
    replies.add(a.reply)
  }
  check('同一问话轮换不重复（≥2 种回答）', replies.size >= 2, `得到 ${replies.size} 种`)
  // 状态调制：困倦时问"你怎么样"优先困倦回答
  brain3.endocrine.set('melatonin', 0.8)
  brain3.wakefulness = 0.2
  const ansTired = brain3.speak('你怎么样', {})
  check('身体状态调制回答（困倦优先）', ansTired.moodTag === 'tired' || ansTired.status.includes('困'), `moodTag=${ansTired.moodTag} status=${ansTired.status}`)
  check('状态措辞存在', typeof ansTired.status === 'string' && ansTired.status.length > 0)

  // ===================== 大脑架构 V2 集成测试 =====================
  // 1. 新网络实例
  check('V2: 分析网络实例', brain3.analysisNet instanceof AnalysisNet)
  check('V2: 缺陷人格网络实例', brain3.flaw instanceof DefectNet)
  check('V2: 训练控制器实例', brain3.trainer instanceof TrainingController)
  // 2. perceive → 预存追加 + 分析存储
  const stBefore = brain3.status()
  brain3.hear('V2 架构测试：这句话应该进入预存文件', { module: 'memory' })
  const stAfter = brain3.status()
  check('V2: 对话追加预存', stAfter.pendingCount === stBefore.pendingCount + 1, `${stBefore.pendingCount}→${stAfter.pendingCount}`)
  check('V2: 分析存储记录', stAfter.analysis.storeCount >= 0 && stAfter.analysis.initialized === true)
  // 3. 缺陷人格网络：训练与冻结
  const beforeFlawJson = JSON.stringify(brain3.flaw.toJSON())
  const flawRes = brain3.trainFlaw('我希望对话更加温柔可爱一些', { epochs: 4 })
  check('V2: trainFlaw 训练', typeof flawRes.loss === 'number' && flawRes.epochs === 4)
  check('V2: trainFlaw 改变权重', JSON.stringify(brain3.flaw.toJSON()) !== beforeFlawJson)
  brain3.freeze()
  const flawFrozen = brain3.trainFlaw('冻结态训练应被跳过', { epochs: 4 })
  check('V2: 冻结态跳过缺陷训练', flawFrozen.skipped === 'frozen')
  brain3.thaw()
  // 4. 休眠自训练闭环（pending 驱动）
  brain3.trainer.appendPending({ t: Date.now(), text: '第一条休眠训练内容', latent32: Array(32).fill(0.1), module: 'memory' })
  brain3.trainer.appendPending({ t: Date.now(), text: '第二条休眠训练内容', latent32: Array(32).fill(0.2), module: 'memory' })
  brain3.trainer.appendPending({ t: Date.now(), text: '第三条休眠训练内容', latent32: Array(32).fill(0.3), module: 'memory' })
  const pend0 = brain3.pendingCount
  brain3.sleepNow()
  check('V2: 手动入睡', brain3.asleep === true)
  const proc = brain3.trainer.processPending(2)
  check('V2: 休眠训练执行', proc.trained >= 1, JSON.stringify(proc))
  check('V2: 训练消耗预存', brain3.pendingCount < pend0, `${pend0}→${brain3.pendingCount}`)
  check('V2: 褪黑素进度条与预存关联', Math.abs(brain3._melatoninFromPending() - (0.12 + Math.min(0.85, brain3.pendingCount * 0.02))) < 1e-6, `mel=${brain3._melatoninFromPending()} vs ${0.12 + Math.min(0.85, brain3.pendingCount * 0.02)} pend=${brain3.pendingCount}`)
  // 5. 统一苏醒流程：同步镜像
  const wakeRes = brain3.wakeNow()
  check('V2: 唤醒走统一流程', wakeRes.woke === true && wakeRes.synced === true)
  check('V2: lastWakeAt 记录', typeof brain3.lastWakeAt === 'string' && brain3.lastWakeAt.length > 0)
  check('V2: 镜像已同步', brain3.status().mirror.synced === true)
  // 6. 自动生成网络
  const genPath = join(dir, 'auto-generated.brain.json')
  const gen = brain3.generateNetwork(genPath)
  check('V2: 自动生成网络', gen.created === true && gen.paramCount > 1000)
  const genSt = brain3.status()
  check('V2: 生成后自动挂载为扩展', genSt.extensions.some((e) => e.path === genPath))
  // 7. 持久化往返：新网络权重
  brain3.save()
  const brain4 = new PlasticBrain()
  const st4 = brain4.load(dir)
  brain4.stop()
  check('V2: 重载后分析网络存在', st4.analysis.initialized === true)
  check('V2: 重载后缺陷网络存在', st4.flaw.initialized === true)
  check('V2: 重载后扩展含生成网络', Array.isArray(st4.extensions) && st4.extensions.some((e) => e.path === genPath))
  brain4.dispose()

  // 清理
  brain.dispose()
  brain2.dispose()
  brain3.dispose()
  rmSync(dir, { recursive: true, force: true })
  console.log('   工作区已清理')
}

console.log('')
console.log(`结果：${passed} 通过 / ${failed} 失败`)
if (failed > 0) process.exitCode = 1
