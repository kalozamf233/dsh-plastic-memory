// ============================================================================
// 临时自测：PersonaBuilder 人格构造器（dsh-plastic-memory V5）
// 运行：node _selfcheck_persona_builder.mjs
// mock brain 用真实类构造（MemoryNet / RecallNet / DefectNet），
// 接口与 brain.memory / brain.trainer.loadMirror / brain.recallNet / brain.flaw 一致。
// ============================================================================
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { PersonaBuilder } from './lib/personabuilder.mjs'
import { MemoryNet } from './lib/nn/memorynet.mjs'
import { DefectNet } from './lib/nn/flawnet.mjs'
import { RecallNet } from './lib/recallnet.mjs'
import { mulberry32 } from './lib/nn/core.mjs'

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

function weightsEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

/** mock brain：用真实类构造，保证与 brain 接口一致。 */
function makeMockBrain(seedBase = 1) {
  const memory = new MemoryNet({ seed: seedBase })
  const rng = mulberry32(seedBase)
  const persona = new Float64Array(64)
  for (let i = 0; i < 64; i++) persona[i] = (rng() * 2 - 1) * 0.5
  memory.writePersona(persona, { epochs: 2 })
  const mirrorMem = new MemoryNet({ seed: seedBase + 100 })
  mirrorMem.writePersona(persona, { epochs: 1 })
  const recallNet = new RecallNet({ seed: seedBase + 200 })
  const flaw = new DefectNet({ seed: seedBase + 300 })
  flaw.train('温柔的话语', { epochs: 2 })
  const trainer = { loadMirror: () => mirrorMem }
  return { memory, trainer, recallNet, flaw }
}

console.log('== 构造 ==')
const brain = makeMockBrain(1)
const builder = new PersonaBuilder(brain, {})
check('构造网络为 DefectNet（143→64→8）', builder.net instanceof DefectNet && builder.net.net.inputSize === 143 && builder.net.net.layerSizes.join(',') === '64,8')

console.log('== collectSources 四源采样 ==')
const c = builder.collectSources()
check('四源样本存在', Array.isArray(c.sources) && c.sources.length === 4)
check('源维度 [64,64,8,64]', c.sources.map((s) => s.length).join(',') === '64,64,8,64')
check('四源拼接 200 维', c.concat instanceof Float64Array && c.concat.length === 200)
check('合成目标 128 维（存内部）', c.target instanceof Float64Array && c.target.length === 128)
check('四源平均 8 维', c.fourAvg instanceof Float64Array && c.fourAvg.length === 8)
check('样本全有限', c.concat.every(Number.isFinite) && c.target.every(Number.isFinite) && c.fourAvg.every(Number.isFinite))
check('记忆源非零（已 writePersona）', c.sources[0].some((v) => v !== 0))
check('缺陷人格源非零（已训练）', c.sources[3].some((v) => v !== 0))

console.log('== build 训练 ==')
const w0 = builder.net.net.layers[0].W.data.slice()
const r1 = builder.build({ epochs: 5 })
check('build 返回有限非负 loss', Number.isFinite(r1.loss) && r1.loss >= 0, String(r1.loss))
check('builds 计数 = 1', builder.builds === 1)
check('build 后权重变化', !weightsEqual(w0, builder.net.net.layers[0].W.data))
check('build 后权重无 NaN', builder.net.net.layers.every((L) => L.W.data.every(Number.isFinite)))
const probe = new Float64Array(143)
probe[0] = 0.5
const bOut = builder.net.net.forward(probe).post.slice(-1)[0]
const fOut = brain.flaw.net.forward(probe).post.slice(-1)[0]
check('构造网络输出 8 维', bOut.length === 8 && bOut.every(Number.isFinite))
check('与缺陷人格输出空间同维度（8）', fOut.length === 8 && bOut.length === fOut.length)
const t0 = builder.net.net.tick
builder.build({ epochs: 3 })
check('epochs 控制训练轮数（tick +3）', builder.net.net.tick - t0 === 3, String(builder.net.net.tick - t0))
builder.build({ epochs: 7 })
check('epochs 控制训练轮数（tick +7）', builder.net.net.tick - (t0 + 3) === 7)
builder.net.frozen = true
const fr = builder.build({ epochs: 2 })
check('构造网络 frozen 时 build 跳过', fr.skipped === 'frozen' && fr.epochs === 0)
builder.net.frozen = false

console.log('== commitReplace ==')
const cr = builder.commitReplace()
check('返回 {replaced:true, before, after}', cr.replaced === true && cr.before !== null && !!cr.after)
check('after 为可加载 JSON（DefectNet.fromJSON → 8 维输出）', (() => {
  try {
    const d = DefectNet.fromJSON(JSON.parse(JSON.stringify(cr.after)))
    const o = d.net.forward(probe).post.slice(-1)[0]
    return o.length === 8 && o.every(Number.isFinite)
  } catch { return false }
})())
check('before = 当前缺陷人格 JSON', typeof cr.before === 'object' && !!cr.before.net)
check('after ≠ before（构造出的新网络）', JSON.stringify(cr.after.net) !== JSON.stringify(cr.before.net))
check('commits 计数 = 1', builder.commits === 1)

console.log('== toJSON / fromJSON 往返 ==')
const json = JSON.parse(JSON.stringify(builder.toJSON()))
const b2 = PersonaBuilder.fromJSON(json, brain)
check('往返 builds/lastLoss 一致', b2.builds === builder.builds && b2.lastLoss === builder.lastLoss)
check('往返构造网络权重一致', weightsEqual(builder.net.net.layers[0].W.data, b2.net.net.layers[0].W.data))
const c1 = builder.collectSources()
const c2 = b2.collectSources()
check('往返后重采样一致（确定性）', c1.concat.every((v, i) => Math.abs(v - c2.concat[i]) < 1e-12))

console.log('== 空源容错 ==')
const emptyBuilder = new PersonaBuilder({}, {})
const ec = emptyBuilder.collectSources()
check('空 brain 不抛错且四源为零向量', ec.sources.length === 4 && ec.sources.every((s) => s.every((v) => v === 0)))
check('空源合成目标全零', ec.target.every((v) => v === 0) && ec.fourAvg.every((v) => v === 0))
const eb = emptyBuilder.build({ epochs: 2 })
check('空源 build 可用（有限 loss）', Number.isFinite(eb.loss) && eb.loss >= 0, String(eb.loss))
check('未 collectSources 直接 build 自动采样', (() => {
  const p = new PersonaBuilder({}, {})
  p.build({ epochs: 1 })
  return p.collected !== null && p.builds === 1
})())

console.log('== 独立文件存储（persona-builder.brain.json）==')
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-pb-'))
try {
  const file = path.join(tmpDir, 'persona-builder.brain.json')
  const p1 = new PersonaBuilder(brain, { builderFile: file })
  p1.collectSources()
  p1.build({ epochs: 3 })
  check('save() 写入独立文件', p1.save() === true && fs.existsSync(file))
  const p2 = new PersonaBuilder(brain, { builderFile: file }) // 构造时自动加载
  check(
    '文件加载恢复状态（builds/lastLoss/权重）',
    p2.builds === p1.builds && p2.lastLoss === p1.lastLoss && weightsEqual(p1.net.net.layers[0].W.data, p2.net.net.layers[0].W.data),
  )
} finally {
  fs.rmSync(tmpDir, { recursive: true, force: true })
}

console.log('== status ==')
const st = builder.status()
check(
  'status() 形状',
  typeof st.params === 'number' && st.params > 0 && typeof st.builds === 'number' && typeof st.commits === 'number' && typeof st.hasCollected === 'boolean',
)

console.log('')
console.log(`结果：PASS ${pass} / FAIL ${fail}`)
if (fail > 0) {
  console.log('失败项：')
  for (const f of failures) console.log('  - ' + f)
  process.exit(1)
}
process.exit(0)
