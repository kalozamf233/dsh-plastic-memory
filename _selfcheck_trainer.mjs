// ============================================================================
// dsh-plastic-memory —— TrainingController 自测（临时脚本，零依赖）
// 运行：<dsh-runtime>/dependencies/node/bin/node.exe _selfcheck_trainer.mjs
// 用真实 MemoryNet / ControllerNet / EndocrineSystem + 临时目录构造最小 mock
// brain（brain.workspace = null，镜像/队列路径通过构造参数显式传入）。
// ============================================================================

import { mkdtempSync, rmSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { TrainingController, MIRROR_FILE_NAME, PENDING_FILE_NAME } from './lib/trainer.mjs'
import { MemoryNet } from './lib/nn/memorynet.mjs'
import { ControllerNet } from './lib/nn/controllernet.mjs'
import { EndocrineSystem } from './lib/nn/endocrine.mjs'

const results = []
function check(name, cond, detail = '') {
  results.push({ name, pass: !!cond, detail })
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`)
}

// ---- 临时目录与最小 mock brain ----
const dir = mkdtempSync(join(tmpdir(), 'trainer-selfcheck-'))
const mirrorFile = join(dir, MIRROR_FILE_NAME)
const pendingFile = join(dir, PENDING_FILE_NAME)

const brain = {
  memory: new MemoryNet({ seed: 42, baseLr: 0.008 }),
  controller: new ControllerNet({ seed: 7 }),
  endocrine: new EndocrineSystem(),
  workspace: null, // paths.mirror/paths.pending 不存在 → 走显式路径参数
  mode: 'training', // 新契约：train = brain.mode === 'training'
}

const trainer = new TrainingController(brain, { mirrorFile, pendingFile })
const latentOf = (t) => brain.controller.hear(t, false).latent

// ---- 文本集 ----
const T1 = '今晚的月亮很圆，风从海面吹来，带着咸味。'
const T2 = '我在图书馆读了一下午的神经科学论文。'
const T3 = '训练完成后要好好睡觉，睡眠巩固记忆。'
const T4 = '庭院里的桂花开了，香气很浓。'
const T5 = '模型需要周期性地保存权重，防止意外丢失。'
const S = '睡眠中记忆会被重新整理，褪黑素上升促进巩固。' // ② 的巩固文本
const U = '无关的噪声文本，用来验证镜像不命中路径。'

// ============================================================================
// ① appendPending ×5 → readPending 5 → processPending(3) → trained≥1、remaining 减少
// ============================================================================
for (const t of [T1, T2, T3, T4, T5]) {
  trainer.appendPending({ text: t, latent: latentOf(t), module: 'memory' })
}
check('①a appendPending×5 后 readPending=5', trainer.readPending().length === 5)
brain.endocrine.set('melatonin', 0.3) // 新契约：processPending 不得覆写褪黑素
const r1 = trainer.processPending(3)
check('①b processPending(3) trained=3', r1.trained === 3, JSON.stringify(r1))
check('①c trained>=1', r1.trained >= 1)
check('①d remaining 减少到 2', r1.remaining === 2)
check('①e 无镜像时 doubled=0', r1.doubled === 0)
check('①f stopped=false', r1.stopped === false)
check('①g processPending 不更新褪黑素（仍=0.3）', Math.abs(brain.endocrine.get('melatonin') - 0.3) < 1e-9)

// ============================================================================
// ② 镜像存在与否影响加倍：先 syncMirror → 再次 processPending 时镜像命中
// ============================================================================
// 先把文本 S 巩固进 brain.memory（镜像快照将包含这些权重）。
// 注意：巩固 salience 与 mirrorRecall 的查询条件（module:'memory', salience:0.6）
// 保持一致，预测吻合度（熟悉度）才会真实升高（salience 错配会使 predSim 上不去）。
for (let i = 0; i < 120; i++) {
  brain.memory.observe(latentOf(S), {
    module: 'memory',
    salience: 0.6,
    lrScale: 1.2,
    emotionTarget: brain.endocrine.effects().emotionTarget,
    train: true,
  })
}
check('②a syncMirror=true', trainer.syncMirror() === true)
check('②b 镜像文件存在且 >0', existsSync(mirrorFile) && readFileSync(mirrorFile, 'utf8').length > 0)
const mrS = trainer.mirrorRecall(latentOf(S))
const mrU = trainer.mirrorRecall(latentOf(U))
console.log(`      mirrorRecall(S) = ${JSON.stringify(mrS)}   mirrorRecall(U) = ${JSON.stringify(mrU)}`)
check('②c similarity(S) ∈ [-1,1] 且 found=true', mrS.similarity >= -1 && mrS.similarity <= 1 && mrS.found === true, `sim=${mrS.similarity} actSim=${mrS.actSim}`)
check('②d actSim 附带字段 ∈ [0,1]', mrS.actSim >= 0 && mrS.actSim <= 1, `actSim=${mrS.actSim}`)
check('②e 无关文本不命中 found=false（单倍语义）', mrU.found === false && mrU.similarity >= -1 && mrU.similarity <= 1, `sim=${mrU.similarity} actSim=${mrU.actSim}`)

// 清掉 ① 遗留的 T4、T5，再验证 S 的加倍路径
const r2a = trainer.processPending(2)
check('②f 清空旧队列 trained=2、remaining=0', r2a.trained === 2 && r2a.remaining === 0, JSON.stringify(r2a))
trainer.appendPending({ text: S, latent: latentOf(S), module: 'memory' })
const r2 = trainer.processPending(1)
check('②g 镜像命中 → doubled=1', r2.doubled === 1, JSON.stringify(r2))
check('②h trained=1、remaining=0', r2.trained === 1 && r2.remaining === 0)

// ============================================================================
// ③ stop 后 processPending 返回 stopped
// ============================================================================
trainer.appendPending({ text: '清晨的露水打湿了台阶。', latent: latentOf('清晨的露水打湿了台阶。') })
trainer.appendPending({ text: '傍晚的云烧成了橘红色。', latent: latentOf('傍晚的云烧成了橘红色。') })
trainer.stop()
const r3 = trainer.processPending(1)
check('③a stop 后 stopped=true', r3.stopped === true, JSON.stringify(r3))
check('③b trained=0/doubled=0/remaining=2', r3.trained === 0 && r3.doubled === 0 && r3.remaining === 2)
trainer.resume()
check('③c resume 后 stopped=false', trainer.stopped === false)

// ============================================================================
// ④ 坏行跳过
// ============================================================================
const raw = readFileSync(pendingFile, 'utf8')
writeFileSync(pendingFile, '{这是坏行,不是合法JSON}\n' + raw, 'utf8')
check('④a 坏行后 readPending 仍=2', trainer.readPending().length === 2)
const r4 = trainer.processPending(1)
const after = readFileSync(pendingFile, 'utf8')
check('④b 训练正常 trained=1、remaining=1', r4.trained === 1 && r4.remaining === 1, JSON.stringify(r4))
check('④c 坏行被清理且队列剩 1 条', !after.includes('坏行') && trainer.readPending().length === 1)

// ============================================================================
// ⑤ syncMirror 后读回可 MemoryNet.fromJSON（往返一致）
// ============================================================================
check('⑤a 再次 syncMirror=true', trainer.syncMirror() === true)
const mirror = trainer.loadMirror()
check('⑤b loadMirror 返回 MemoryNet 实例', mirror instanceof MemoryNet)
check('⑤c 参数规模与 brain.memory 一致', !!mirror && mirror.paramCount === brain.memory.paramCount, mirror ? `${mirror.paramCount} params` : 'null')
const rt = mirror ? MemoryNet.fromJSON(mirror.toJSON()) : null
check('⑤d toJSON/fromJSON 往返参数一致', !!rt && rt.paramCount === brain.memory.paramCount)
check(
  '⑤e 往返后 observe 可用',
  (() => {
    try {
      rt.observe(latentOf('往返校验'), { module: 'memory', train: false })
      return true
    } catch {
      return false
    }
  })(),
  rt ? `observeCount=${rt.status().observeCount}` : 'null',
)

// ============================================================================
// ⑥ 构造契约：workspace.paths.mirror/pending 缺失且未显式传参 → 抛错提示
// ============================================================================
let threw = false
try {
  new TrainingController({ memory: new MemoryNet(), controller: brain.controller, endocrine: brain.endocrine, workspace: null })
} catch {
  threw = true
}
check('⑥ 无镜像/队列路径时构造抛错', threw)

// ---- 清理 ----
try {
  rmSync(dir, { recursive: true, force: true })
} catch {
  /* ignore */
}

const failed = results.filter((r) => !r.pass).length
console.log(`\n${results.length - failed}/${results.length} 通过${failed ? `，${failed} 失败` : '，全部通过 ✅'}`)
process.exit(failed ? 1 : 0)
