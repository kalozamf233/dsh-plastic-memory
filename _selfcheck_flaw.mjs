// ============================================================================
// 临时自测：DefectNet（dsh-plastic-memory）
// 运行：node _selfcheck_flaw.mjs
// ============================================================================
import { DefectNet } from './lib/nn/flawnet.mjs'
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

const rng = mulberry32(7)
const latent = new Float64Array(128)
for (let i = 0; i < 128; i++) latent[i] = rng() * 2 - 1
const endo = new Float64Array(15)
for (let i = 0; i < 15; i++) endo[i] = rng()

const KEYS = ['warmth', 'intimacy', 'embellish', 'deflect', 'quirk']
const inRange = (o) => KEYS.every((k) => Number.isFinite(o[k]) && o[k] >= 0 && o[k] <= 1)
const weightsEqual = (a, b) => {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

console.log('== DefectNet ==')
const net = new DefectNet()
check('构造后 frozen = false', net.frozen === false)
check('结构 143 → 64 → 8', net.net.inputSize === 143 && net.net.layerSizes.join(',') === '64,8')

const out = net.apply(latent, endo)
check('apply 返回 5 个参数', KEYS.every((k) => typeof out[k] === 'number'))
check('apply 参数均在 [0,1]', inRange(out), JSON.stringify(out))

// apply 不更新权重
const wBefore = net.net.layers[0].W.data.slice()
for (let i = 0; i < 5; i++) net.apply(latent, endo)
check('apply 只前向、不更新权重', weightsEqual(wBefore, net.net.layers[0].W.data))

// endoLevels 缺省
const outNoEndo = net.apply(latent)
check('apply 缺省 endoLevels 可用', inRange(outNoEndo))

// train 更新权重
const w0 = net.net.layers[0].W.data.slice()
const tr = net.train('今天天气真好，心里充满了喜悦！', { epochs: 8 })
check('train 返回 {loss, epochs}', typeof tr.loss === 'number' && tr.epochs === 8)
check('train loss 有限且 ≥0', Number.isFinite(tr.loss) && tr.loss >= 0, String(tr.loss))
check('train 后权重变化', !weightsEqual(w0, net.net.layers[0].W.data))

// 不同情绪文本均可训练
for (const txt of ['悲伤的消息让人难过', '愤怒的我生气了', '你真是又可爱又温柔', '']) {
  const r = net.train(txt, { epochs: 2 })
  check(`train("${txt || '(空)'}") 可用`, Number.isFinite(r.loss) && r.loss >= 0)
}

// frozen：train 跳过、权重不变
net.frozen = true
const w1 = net.net.layers[0].W.data.slice()
const fsr = net.train('再来一次训练', { epochs: 4 })
check(
  'frozen 时 train 跳过 {loss:0, epochs:0, skipped:"frozen"}',
  fsr.skipped === 'frozen' && fsr.loss === 0 && fsr.epochs === 0,
  JSON.stringify(fsr),
)
check('frozen 时权重不变', weightsEqual(w1, net.net.layers[0].W.data))
check('frozen 时 apply 仍可用', inRange(net.apply(latent, endo)))
net.frozen = false

// 序列化往返
const clone = DefectNet.fromJSON(JSON.parse(JSON.stringify(net.toJSON())))
check('往返 frozen 标志一致', clone.frozen === net.frozen)
const oa = net.apply(latent, endo)
const ob = clone.apply(latent, endo)
check('往返 apply 输出一致', KEYS.every((k) => Math.abs(oa[k] - ob[k]) < 1e-12))
const la = net.train('温柔亲密的话语', { epochs: 3 })
const lb = clone.train('温柔亲密的话语', { epochs: 3 })
check('往返 train loss 一致', Math.abs(la.loss - lb.loss) < 1e-9, `${la.loss} vs ${lb.loss}`)

net.frozen = true
const clone2 = DefectNet.fromJSON(JSON.parse(JSON.stringify(net.toJSON())))
check('frozen=true 序列化保留', clone2.frozen === true)
net.frozen = false

console.log('')
console.log(`结果：PASS ${pass} / FAIL ${fail}`)
if (fail > 0) {
  console.log('失败项：')
  for (const f of failures) console.log('  - ' + f)
  process.exit(1)
}
process.exit(0)
