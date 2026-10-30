// ============================================================================
// 临时自测：AnalysisNet + AnalysisStore（dsh-plastic-memory）
// 运行：node _selfcheck_analysis.mjs
// ============================================================================
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { AnalysisNet, AnalysisStore } from './lib/nn/analysisnet.mjs'
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

function randVec(n, seed) {
  const rng = mulberry32(seed)
  const v = new Float64Array(n)
  for (let i = 0; i < n; i++) v[i] = rng() * 2 - 1
  return v
}

function countRawLines(file) {
  return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).length
}

console.log('== AnalysisNet ==')
const net = new AnalysisNet()
const r1 = net.analyze(randVec(128, 1), {})
check('analyze 返回 16 维 Float64Array', r1.vector instanceof Float64Array && r1.vector.length === 16)
check('vector 全部有限', r1.vector.every((v) => Number.isFinite(v)))
check('topicScore ∈ [0,1]', r1.topicScore >= 0 && r1.topicScore <= 1, String(r1.topicScore))
check('novelty ∈ [0,1]', r1.novelty >= 0 && r1.novelty <= 1, String(r1.novelty))
check('首帧 novelty = 0.5', r1.novelty === 0.5)

const rNone = net.analyze(randVec(128, 5)) // effects 完全缺省
check('effects 缺省可用（全有限）', rNone.vector.every((v) => Number.isFinite(v)))

const r2 = net.analyze(randVec(128, 2), {})
const r3 = net.analyze(randVec(128, 3), {})
const novs = [r1.novelty, r2.novelty, r3.novelty].map((x) => Number(x.toFixed(4)))
check('3 次不同输入 novelty 变化', new Set(novs).size > 1, `novelties=${novs.join(',')}`)

// 序列化往返（同一输入 → 相同输出）
const a = new AnalysisNet({ seed: 20261030 })
const b = AnalysisNet.fromJSON(JSON.parse(JSON.stringify(a.toJSON())))
const x = randVec(128, 42)
const fx = { arousal: 0.7, valence: 0.3 }
const ra = a.analyze(x, fx)
const rb = b.analyze(x, fx)
check('往返 vector 一致', ra.vector.every((v, i) => Math.abs(v - rb.vector[i]) < 1e-12))
check('往返 topicScore 一致', Math.abs(ra.topicScore - rb.topicScore) < 1e-12)
check('往返 novelty 一致', Math.abs(ra.novelty - rb.novelty) < 1e-12)
const st = a.status()
check(
  'status() = {params, ticks, lastNovelty}',
  typeof st.params === 'number' && st.params > 0 && typeof st.ticks === 'number' && st.ticks > 0 && typeof st.lastNovelty === 'number',
)

console.log('== AnalysisStore：缓冲/落盘/合并/坏行 ==')
const dir1 = path.join(os.tmpdir(), `dsh-am-selfcheck-${process.pid}-${Date.now()}-a`)
try {
  const store = new AnalysisStore(dir1)
  const file1 = path.join(dir1, 'records.jsonl')
  check('文件不存在时 count = 0', store.count() === 0)
  store.append({ t: 1, text: 'x'.repeat(150), module: 'audio', vector: randVec(16, 10), topicScore: 0.5, novelty: 0.2 })
  store.append({ t: 2, text: 'two', module: 'visual', vector: new Float64Array(16), topicScore: 0.1, novelty: 0.9 })
  store.append({ t: 3, text: 'three', module: 'memory', vector: Array.from({ length: 16 }, (_, i) => i / 16), topicScore: 0.9, novelty: 0.1 })
  check('append 只进缓冲、不落盘', !fs.existsSync(file1))
  check('count = 3（纯缓冲）', store.count() === 3, String(store.count()))
  const recents = store.recent()
  check('recent 返回 3 条且新在后', recents.length === 3 && recents[0].t === 1 && recents[2].t === 3)
  check('text 截断到 ≤100', recents[0].text.length === 100, `len=${recents[0].text.length}`)
  check('vector 存为 16 元数组', Array.isArray(recents[0].vector) && recents[0].vector.length === 16)

  // flush：缓冲落盘并清空
  const flushed = store.flush()
  check('flush 落盘 3 条且文件 3 行', flushed === 3 && fs.existsSync(file1) && countRawLines(file1) === 3)
  check('flush 后缓冲清空、count 仍 = 3（读文件）', store.count() === 3, String(store.count()))
  check('flush 空缓冲安全（返回 0）', store.flush() === 0)

  // 文件 + 缓冲合并
  store.append({ t: 4, text: 'four', module: 'audio', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
  store.append({ t: 5, text: 'five', module: 'audio', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
  check('count = 5（文件3+缓冲2）', store.count() === 5, String(store.count()))
  const r3 = store.recent(3)
  check('recent(3) 合并文件尾+缓冲', r3.length === 3 && r3[0].t === 3 && r3[2].t === 5)

  // 坏行：非 JSON + 缺 vector（直接写入文件）
  fs.appendFileSync(file1, 'this is not json\n')
  fs.appendFileSync(file1, '{"t":99,"vector":"nope"}\n')
  check('坏行跳过（recent）', store.recent().length === 5, String(store.recent().length))
  check('坏行跳过（count）', store.count() === 5, String(store.count()))

  // 环形缓冲容量 200：追加 250 条不 flush
  for (let i = 1; i <= 250; i++) {
    store.append({ t: 9000 + i, text: 'ring', module: 'audio', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
  }
  check('缓冲容量 200（丢最旧）', store.count() === 3 + 200, String(store.count()))
  check('缓冲保留最后 200 条', store.recent(1)[0].t === 9250, String(store.recent(1)[0].t))
  store.flush()
  check('flush 后 count = 203、原始行 = 205', store.count() === 203 && countRawLines(file1) === 205)
} finally {
  fs.rmSync(dir1, { recursive: true, force: true })
}

console.log('== AnalysisStore：截断 ==')
const dir2 = path.join(os.tmpdir(), `dsh-am-selfcheck-${process.pid}-${Date.now()}-b`)
try {
  const store2 = new AnalysisStore(dir2)
  const file2 = path.join(dir2, 'records.jsonl')
  // 环形缓冲验证（flush 前）：250 条只留 200
  for (let i = 1; i <= 250; i++) {
    store2.append({ t: i, text: 'ring', module: 'm', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
  }
  check('截断场景：缓冲容量 200（丢最旧）', store2.count() === 200 && store2.recent(1)[0].t === 250)
  store2.flush() // 文件 200 行（t=51..250）
  // 5 批 × 200 条（t = 1000+b*200+i），每批 flush
  for (let b = 0; b < 5; b++) {
    for (let i = 1; i <= 200; i++) {
      store2.append({ t: 1000 + b * 200 + i, text: 'fill', module: 'm', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
    }
    store2.flush()
  }
  check('超 1000 行截断后 count = 800', store2.count() === 800, String(store2.count()))
  check('原始文件行数 = 800', countRawLines(file2) === 800, String(countRawLines(file2)))
  const all2 = store2.recent(1000)
  check(
    '保留的恰是最后 800 条',
    all2.length === 800 && all2[0].t === 1201 && all2[all2.length - 1].t === 2000,
    `first=${all2[0].t}, last=${all2[all2.length - 1].t}`,
  )
  // 截断后追加 → 文件尾 + 缓冲合并
  store2.append({ t: 3001, text: 'z1', module: 'm', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
  store2.append({ t: 3002, text: 'z2', module: 'm', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
  store2.append({ t: 3003, text: 'z3', module: 'm', vector: new Float64Array(16), topicScore: 0, novelty: 0 })
  check('截断后 count = 803', store2.count() === 803, String(store2.count()))
  const r5 = store2.recent(5)
  check('recent(5) = 文件尾+缓冲', r5.length === 5 && r5[0].t === 1999 && r5[4].t === 3003, `first=${r5[0].t}, last=${r5[4].t}`)
} finally {
  fs.rmSync(dir2, { recursive: true, force: true })
}

console.log('')
console.log(`结果：PASS ${pass} / FAIL ${fail}`)
if (fail > 0) {
  console.log('失败项：')
  for (const f of failures) console.log('  - ' + f)
  process.exit(1)
}
process.exit(0)
