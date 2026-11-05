// ============================================================================
// dsh-plastic-memory —— NeuroController 自测（临时脚本，零依赖）
// 运行：<dsh-runtime>/dependencies/node/bin/node.exe _selfcheck_neuro.mjs
// 覆盖：geneThreshold 固定计算（同基因同值/异基因异值/精确值/范围）+
//       mergeSimilarNeurons（相近神经元对合并减少、权重取平均、跨层列同步、
//       可 fromJSON 往返加载、入参不被修改、阈值 0 不合并、阈值 1 大量合并不崩、
//       输入/输出层不动）。
// ============================================================================

import { geneThreshold, mergeSimilarNeurons } from './lib/neurocontroller.mjs'
import { PlasticNetwork } from './lib/nn/network.mjs'

const results = []
function check(name, cond, detail = '') {
  results.push({ name, pass: !!cond, detail })
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`)
}

// ---------------------------------------------------------------------------
// ① geneThreshold
// ---------------------------------------------------------------------------
const thA = geneThreshold('a'.repeat(100)) // 'a'=97&31=1 → '00001' → 前100位 20 个 1 → p=0.2
check('①a 全 a 基因阈值 = 0.024', Math.abs(thA - 0.024) < 1e-9, `th=${thA}`)
const thZ = geneThreshold('z'.repeat(100)) // 'z'=122&31=26 → '11010' → 3 个 1/字符 → p=0.6
check('①b 全 z 基因阈值 = 0.062', Math.abs(thZ - 0.062) < 1e-9, `th=${thZ}`)
check('①c 同基因同值（确定性）', geneThreshold('a'.repeat(100)) === thA)
check('①d 异基因异值', thA !== thZ)
check('①e 空基因阈值 = 0.005', Math.abs(geneThreshold('') - 0.005) < 1e-9)
check('①f 只有前 100 位参与（a×20+z×80）', Math.abs(geneThreshold('a'.repeat(20) + 'z'.repeat(80)) - 0.024) < 1e-9)
check('①g 只有前 100 位参与（z×20+a×80）', Math.abs(geneThreshold('z'.repeat(20) + 'a'.repeat(80)) - 0.062) < 1e-9)
let rangeOk = true
let minT = Infinity
let maxT = -Infinity
for (let i = 0; i < 300; i++) {
  let g = ''
  for (let k = 0; k < 100; k++) g += String.fromCharCode(97 + Math.floor(Math.random() * 26))
  const t = geneThreshold(g)
  if (t < minT) minT = t
  if (t > maxT) maxT = t
  if (t < 0.005 - 1e-9 || t > 0.1 + 1e-9) rangeOk = false
}
check('①h 随机基因阈值恒在 [0.005, 0.1]', rangeOk, `min=${minT.toFixed(4)} max=${maxT.toFixed(4)}`)

// ---------------------------------------------------------------------------
// ② mergeSimilarNeurons
// ---------------------------------------------------------------------------

/** 手工构造的可控 3 层网络 JSON（4→4→3→2，层0 神经元 0/1 权重完全相同）。 */
function mkNetJson() {
  return {
    inputSize: 4,
    tick: 0,
    momentum: 0.9,
    gradNormEma: [0, 0, 0],
    structuralChanges: [],
    layers: [
      {
        size: 4, activation: 'tanh', lr: 1, decay: 0, mutable: true,
        W: [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        b: [0.1, 0.2, 0.3, 0.4], vW: new Array(16).fill(0), vb: [0, 0, 0, 0],
        usage: [0.5, 0.5, 0.5, 0.5], bornAt: [0, 10, 20, 30],
      },
      {
        size: 3, activation: 'tanh', lr: 1, decay: 0, mutable: true,
        W: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        b: [0, 0, 0], vW: new Array(12).fill(0), vb: [0, 0, 0],
        usage: [0.5, 0.5, 0.5], bornAt: [1, 2, 3],
      },
      {
        size: 2, activation: 'tanh', lr: 1, decay: 0, mutable: false,
        W: [1, 2, 3, 4, 5, 6],
        b: [0, 0], vW: new Array(6).fill(0), vb: [0, 0],
        usage: [0.5, 0.5], bornAt: [0, 0],
      },
    ],
  }
}

// ---- 阈值 0.01：只合并完全相同的一对 ----
const netJson = mkNetJson()
const before = JSON.stringify(netJson)
const r = mergeSimilarNeurons(netJson, 0.01)
check('②a 相同相邻神经元对被合并（4→3）', r.mergedCount === 1 && r.newJson.layers[0].size === 3, `merged=${r.mergedCount}`)
check('②b pairs 记录 {layer:0,i:0,j:1,dist:0}', r.pairs.length === 1 && r.pairs[0].layer === 0 && r.pairs[0].i === 0 && r.pairs[0].j === 1 && r.pairs[0].dist === 0, JSON.stringify(r.pairs))
check('②c 合并后权重 = 两者平均', JSON.stringify(r.newJson.layers[0].W.slice(0, 4)) === JSON.stringify([1, 2, 3, 4]), JSON.stringify(r.newJson.layers[0].W.slice(0, 4)))
check('②d 偏置取平均', Math.abs(r.newJson.layers[0].b[0] - 0.15) < 1e-9, `b0=${r.newJson.layers[0].b[0]}`)
check('②e 出生取较早', r.newJson.layers[0].bornAt[0] === 0)
check('②f 下一层输入列同步合并', JSON.stringify(r.newJson.layers[1].W.slice(0, 3)) === JSON.stringify([1.5, 3, 4]), JSON.stringify(r.newJson.layers[1].W.slice(0, 3)))
check('②g 输出层神经元不动', r.newJson.layers[2].size === 2)
check('②h 输入层（inputSize）不动', r.newJson.inputSize === 4)
check('②i 入参 netJson 不被修改', JSON.stringify(netJson) === before)

// ---- 往返加载（真实 PlasticNetwork.fromJSON + 前向）----
const loaded = PlasticNetwork.fromJSON(r.newJson)
check('②j fromJSON 可加载且层尺寸 [3,3,2]', loaded.layerSizes.join(',') === '3,3,2', loaded.layerSizes.join(','))
check('②k 加载后前向可用', (() => {
  try {
    const x = new Float64Array([0.1, -0.2, 0.3, -0.4])
    const fwd = loaded.forward(x)
    return fwd.post[fwd.post.length - 1].length === 2
  } catch {
    return false
  }
})())
check('②l 往返再序列化一致', JSON.stringify(loaded.toJSON().layers.map((l) => l.size)) === JSON.stringify([3, 3, 2]))

// ---- 阈值 0：不合并 ----
const r0 = mergeSimilarNeurons(mkNetJson(), 0)
check('②m 阈值 0 不合并', r0.mergedCount === 0 && r0.newJson.layers[0].size === 4 && r0.pairs.length === 0)

// ---- 阈值 1：大量合并不崩（隐藏层压到 1，输出层仍 2，可加载可前向）----
const r1 = mergeSimilarNeurons(mkNetJson(), 1)
check('②n 阈值 1 大量合并', r1.mergedCount >= 3 && r1.newJson.layers[0].size === 1 && r1.newJson.layers[1].size === 1, `merged=${r1.mergedCount}`)
check('②o 输出层始终不动', r1.newJson.layers[2].size === 2)
const loaded1 = PlasticNetwork.fromJSON(r1.newJson)
check('②p 大量合并后仍可加载前向', (() => {
  try {
    const fwd = loaded1.forward(new Float64Array([0.5, 0.1, -0.3, 0.2]))
    return fwd.post[fwd.post.length - 1].length === 2
  } catch {
    return false
  }
})())

// ---- 单层网络（无隐藏层）不合并 ----
const single = { inputSize: 2, layers: [{ size: 2, W: [1, 2, 3, 4], b: [0, 0], vW: [0, 0, 0, 0], vb: [0, 0], usage: [0.5, 0.5], bornAt: [0, 0] }] }
const rs = mergeSimilarNeurons(single, 1)
check('②q 单层网络（纯输出层）不合并', rs.mergedCount === 0 && rs.newJson.layers[0].size === 2)

// ---- 非法入参容错 ----
const rNull = mergeSimilarNeurons(null, 0.05)
const rBad = mergeSimilarNeurons({}, 0.05)
check('②r 非法入参容错', rNull.mergedCount === 0 && rNull.pairs.length === 0 && rBad.mergedCount === 0)

// ---- 真实 PlasticNetwork 权重（随机初始化）上跑一遍不崩 ----
const realNet = new PlasticNetwork({
  inputSize: 16,
  momentum: 0.9,
  layers: [
    { size: 12, activation: 'tanh', mutable: true },
    { size: 8, activation: 'tanh', mutable: true },
    { size: 4, activation: 'tanh', mutable: false },
  ],
})
const rr = mergeSimilarNeurons(realNet.toJSON(), 0.2)
const loadedReal = PlasticNetwork.fromJSON(rr.newJson)
check('②s 真实 PlasticNetwork JSON 合并后可加载', rr.mergedCount >= 0 && loadedReal.layerSizes.every((s, i) => s <= [12, 8, 4][i]), `merged=${rr.mergedCount} sizes=${loadedReal.layerSizes.join(',')}`)

const failed = results.filter((x) => !x.pass).length
console.log(`\n${results.length - failed}/${results.length} 通过${failed ? `，${failed} 失败` : '，全部通过 ✅'}`)
process.exit(failed ? 1 : 0)
