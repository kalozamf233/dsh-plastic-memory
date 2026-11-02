// ============================================================================
// 临时自测：BodyNet + WorldEncoder（dsh-plastic-memory V3）
// 运行：node _selfcheck_bodynet.mjs
// ============================================================================
import { BodyNet, WorldEncoder, desireFromEndocrine } from './lib/bodynet.mjs'
import { MOLECULE_INDEX } from './lib/nn/endocrine.mjs'

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

const sum = (arr) => { let s = 0; for (const v of arr) s += v; return s }
function weightsEqual(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

const sense = {
  sights: [{ text: '一棵苹果树', type: 'tree' }, { text: '红色花朵', type: 'flower' }],
  sounds: [{ text: '鸟鸣声', type: 'bird' }],
  smells: [{ text: '香甜的蛋糕', type: 'cake', dir: 'e', id: 'cake-1' }],
  touches: [],
}
const body0 = { x: 20, y: 20, z: 0, dir: 0, energy: 0.8, pain: 0, satiety: 0.5, speed: 0 }
const desireAppetite = new Float64Array(16)
desireAppetite[2] = 1

console.log('== WorldEncoder ==')
const enc = new WorldEncoder()
const v1 = enc.encode(sense, { x: 20, y: 20, z: 0, dir: 2, energy: 0.8, pain: 0.1, satiety: 0.5, speed: 0 })
check('encode 返回 Float64Array(152)', v1 instanceof Float64Array && v1.length === 152)
check('哈希词袋非零且全有限', v1.every(Number.isFinite) && v1.subarray(0, 128).some((v) => v > 0))
check(
  'body8 归一化（x/40、z/8、dir/3、能量/痛感/饱腹）',
  Math.abs(v1[144] - 0.5) < 1e-12 && Math.abs(v1[145] - 0.5) < 1e-12 && Math.abs(v1[146] - 0) < 1e-12 &&
    Math.abs(v1[147] - 2 / 3) < 1e-12 && Math.abs(v1[148] - 0.8) < 1e-12 &&
    Math.abs(v1[149] - 0.1) < 1e-12 && Math.abs(v1[150] - 0.5) < 1e-12,
)
check('词袋桶值 ≤ 1（计数/4 归一）', v1.subarray(0, 128).every((v) => v <= 1))
const vEmpty = enc.encode({}, null)
check('空感知不抛错且前 128 全 0', vEmpty.subarray(0, 128).every((v) => v === 0))
const vZh = enc.encode({}, { x: 40, y: 40, z: 8, dir: 3, 能量: 1, 痛感: 1, 饱腹: 0.5, 速度: 0.25 })
check(
  '身体状态中文字段兼容',
  Math.abs(vZh[144] - 1) < 1e-12 && Math.abs(vZh[145] - 1) < 1e-12 && Math.abs(vZh[146] - 1) < 1e-12 &&
    Math.abs(vZh[147] - 1) < 1e-12 && Math.abs(vZh[148] - 1) < 1e-12 && Math.abs(vZh[149] - 1) < 1e-12 &&
    Math.abs(vZh[150] - 0.5) < 1e-12 && Math.abs(vZh[151] - 0.25) < 1e-12,
)

console.log('== desireFromEndocrine ==')
const d0 = desireFromEndocrine(new Float64Array(15))
check('desire 维度 16 且值域 [0,1]', d0 instanceof Float64Array && d0.length === 16 && d0.every((v) => Number.isFinite(v) && v >= 0 && v <= 1))
const mkEndo = (id) => { const e = new Float64Array(15); e[MOLECULE_INDEX[id]] = 1; return e }
check('多巴胺 → 探索欲高', desireFromEndocrine(mkEndo('dopamine'))[0] > 0.7)
check('褪黑素 → 休息欲高', desireFromEndocrine(mkEndo('melatonin'))[3] > 0.7)
check('催产素 → 社交欲高', desireFromEndocrine(mkEndo('oxytocin'))[4] > 0.6)
check('皮质醇 → 回避欲高', desireFromEndocrine(mkEndo('cortisol'))[1] > 0.6)
check('null 内分泌不抛错', (() => { try { return desireFromEndocrine(null).length === 16 } catch { return false } })())

console.log('== BodyNet 前向 / decide ==')
const net1 = new BodyNet({ seed: 42 })
const d1 = net1.decide(sense, body0, null)
check('输出 17 维且无 NaN', d1.output instanceof Float64Array && d1.output.length === 17 && d1.output.every(Number.isFinite))
check('probs 6 维、有限、和≈1', d1.probs.length === 6 && d1.probs.every(Number.isFinite) && Math.abs(sum(d1.probs) - 1) < 1e-9)
check('action 合法', ['move', 'jump', 'grab', 'eat', 'rest'].includes(d1.action), d1.action)
check('move 时 dir 必非空', d1.action !== 'move' || ['n', 's', 'e', 'w'].includes(d1.dir))
check('lastAction 已记录', net1.lastAction && net1.lastAction.action === d1.action)

console.log('== 欲望 / 痛感偏置 ==')
const cakeSense = { smells: [{ text: '香甜的蛋糕', type: 'cake', dir: 'e', id: 'cake-1' }] }
const bnBias = new BodyNet({ seed: 7 })
const dBias = bnBias.decide(cakeSense, body0, desireAppetite, {})
check('食欲偏置：东向概率 ≥ 0.5（0.5 权重）', dBias.probs[3] >= 0.5, String(dBias.probs[3]))
let eastCount = 0
for (let i = 0; i < 30; i++) {
  const r = bnBias.decide(cakeSense, body0, desireAppetite, {})
  if (r.action === 'move' && r.dir === 'e') eastCount++
}
check('30 轮采样至少 3 次朝目标方向', eastCount >= 3, `eastCount=${eastCount}`)
const painBody = { x: 20, y: 20, z: 0, dir: 0, energy: 0.8, pain: 1, satiety: 0.5, speed: 0 }
const dPain = new BodyNet({ seed: 9 }).decide(sense, painBody, null)
check('痛感高 → 静止概率 ≥ 0.28（+0.4 再归一）', dPain.probs[5] >= 0.28, String(dPain.probs[5]))

console.log('== REINFORCE 训练 ==')
const bn = new BodyNet({ seed: 11 })
const trainInput = BodyNet.assembleInput(enc.encode(sense, body0), desireAppetite)
const out17 = bn.decide(sense, body0, desireAppetite).output
const wBefore = bn.net.layers[0].W.data.slice()
const tr = bn.train(trainInput, out17, 3, 1.0, { epochs: 2 })
check('train 返回有限非负 loss', Number.isFinite(tr.loss) && tr.loss >= 0, String(tr.loss))
check('baseline EMA 更新（0 → 0.1）', Math.abs(tr.baseline - 0.1) < 1e-12)
check('advantage = reward − 旧 baseline（=1）', tr.advantage === 1)
check('train 后权重变化', !weightsEqual(wBefore, bn.net.layers[0].W.data))
check('train 后权重无 NaN', bn.net.layers.every((L) => L.W.data.every(Number.isFinite)))
const trNeg = bn.train(trainInput, out17, 3, -1, { epochs: 1 })
check('负奖励 advantage < 0', trNeg.advantage < 0)
const trFlat = bn.train(trainInput, out17, 3, bn.baseline, { epochs: 1 })
check('advantage≈0 时跳过（skipped=flat）', trFlat.skipped === 'flat' && trFlat.loss === 0)

console.log('== 序列化往返 ==')
const a = new BodyNet({ seed: 99 })
const json = JSON.parse(JSON.stringify(a.toJSON()))
const b = BodyNet.fromJSON(json)
check('往返 seed/baseline/rngCalls 一致', b.seed === a.seed && b.baseline === a.baseline && b.rngCalls === a.rngCalls)
const dA = a.decide(sense, body0, desireAppetite)
const dB = b.decide(sense, body0, desireAppetite)
check(
  '往返 decide 完全一致',
  dB.action === dA.action && dB.dir === dA.dir &&
    dB.output.every((v, i) => Math.abs(v - dA.output[i]) < 1e-12) &&
    dB.probs.every((v, i) => Math.abs(v - dA.probs[i]) < 1e-12),
)
check('往返 lastAction 一致', JSON.stringify(b.lastAction) === JSON.stringify(a.lastAction))
const st = a.status()
check(
  'status() = {params, baseline, lastAction}',
  typeof st.params === 'number' && st.params > 0 && typeof st.baseline === 'number' && (st.lastAction === null || typeof st.lastAction === 'object'),
)

console.log('== 边界输入 ==')
check('空感知 decide 不抛错', (() => { try { const r = new BodyNet({ seed: 3 }).decide({}, body0, null); return !!r && r.output.length === 17 } catch { return false } })())
check('null 感知/身体/欲望 不抛错', (() => { try { const r = new BodyNet({ seed: 3 }).decide(null, null, null); return !!r && !!r.action } catch { return false } })())
const asm = BodyNet.assembleInput(enc.encode(sense, body0), desireAppetite, new Float64Array(8))
check('assembleInput 组装 152 维', asm instanceof Float64Array && asm.length === 152)
const asm2 = BodyNet.assembleInput(enc.encode({}, null), desireAppetite)
check('欲望填入 [128..144)：食欲位=1', asm2.length === 152 && asm2[128] === 0 && asm2[130] === 1 && asm2[143] === 0)

console.log('')
console.log(`结果：PASS ${pass} / FAIL ${fail}`)
if (fail > 0) {
  console.log('失败项：')
  for (const f of failures) console.log('  - ' + f)
  process.exit(1)
}
process.exit(0)
