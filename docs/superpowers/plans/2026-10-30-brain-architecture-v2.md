# 大脑架构 V2 实现计划

> **For agentic workers:** 本计划由 Team Lead 与队友并行实施（subagent-driven-development 模式）。设计文档：`docs/superpowers/specs/2026-10-30-brain-architecture-v2-design.md`

**Goal:** 把可塑性记忆升级为多网络架构：分析网络+分析存储、镜像网络、缺陷人格网络、训练控制器+同步器、休眠自训练闭环（褪黑素=预存容量进度条）、AI 模型构建子页面（自动生成 MemoryNet 网络）、苏醒提示框。

**Architecture:** 新增三个纯 nn 模块（analysisnet/flawnet/trainer），brain.mjs 集成编排，index.mjs 暴露新 API，client.js 加子页面与苏醒提示框。全部纯 JS 零依赖。

**Tech Stack:** Node.js ESM、零外部依赖、现有 PlasticNetwork/MemoryNet 核心。

**Spec:** `docs/superpowers/specs/2026-10-30-brain-architecture-v2-design.md`

## Global Constraints

- 纯 JS 零依赖；心跳每秒 tick 必须毫秒级返回（休眠训练每 tick ≤3 条）
- 训练后校验权重有限性，NaN 回滚
- 新文件原子写（tmp+rename）
- 旧工作区自动初始化新文件，不破坏已有数据
- 接口契约（队友按此实现）：
  - `AnalysisNet({seed}).analyze(latent128, effects) → {vector:Float64Array(16), topicScore, novelty}`；`toJSON()/fromJSON()`
  - `AnalysisStore(dir).append(record)/recent(n)/count()`
  - `DefectNet({seed}).apply(replyLatent128, endoLevels15) → {warmth, intimacy, embellish, deflect}`；`train(text, epochs, lr)`；`frozen` 字段；`toJSON()/fromJSON()`
  - `TrainingController(brain).processPending(max=3) → {trained, doubled, remaining}`；`syncMirror() → boolean`；`stop()`；`mirrorRecall(latent) → {found, similarity}`

## Review Focus

1. 休眠训练期间用户发消息（并发）→ perceive 与 processPending 不共享可变缓冲，各自用快照
2. 同步失败（写盘异常）→ syncMirror 返回 false，苏醒流程兜底仍苏醒但记录 lastError
3. 预存文件损坏行 → 读取跳过坏行
4. 褪黑素封顶与归零边界（pendingCount 0 时不得负值）
5. 缺陷网络冻结态训练请求 → 仅 trainFlaw API 可写，其余路径观察不更新权重

---

## Task 1: 分析网络 + 分析存储（队友 nn-modules）

**Files:** Create `lib/nn/analysisnet.mjs`；自测脚本 `_selfcheck_analysis.mjs`（临时）

- [ ] 实现 AnalysisNet（两层：128+8→64→16，tanh），analyze 输出分析向量与 {topicScore, novelty}
- [ ] 实现 AnalysisStore（append/recent/count，JSONL 原子追加，损坏行跳过）
- [ ] 自测：构造→分析 3 条不同文本→断言向量有限、store 计数正确、坏行跳过
- [ ] 报告结果给 lead

## Task 2: 缺陷人格网络（队友 nn-modules）

**Files:** Create `lib/nn/flawnet.mjs`；自测脚本 `_selfcheck_flaw.mjs`（临时）

- [ ] 实现 DefectNet（143→64→8：输入=replyLatent128+endoLevels15，输出修饰参数）
- [ ] frozen 冻结逻辑（forward 只前向不反传）+ train（仅显式调用可更新权重）
- [ ] 序列化往返一致
- [ ] 自测通过后报告 lead

## Task 3: 训练控制器 + 同步器 + 镜像（队友 trainer）

**Files:** Create `lib/trainer.mjs`；自测脚本 `_selfcheck_trainer.mjs`（临时）

- [ ] 实现 TrainingController：processPending(max) 逐条编码→镜像 recall（相似度>0.6 命中）→ 1x/2x epochs 训练记忆网络→移除条目→更新褪黑素
- [ ] 实现 syncMirror（记忆网络 toJSON → mirror.brain.json 原子写）与 mirrorRecall
- [ ] 实现 pending 文件读写（appendPending/readPending/removePending，JSONL）
- [ ] 自测：mock brain（构造最小 brain 对象）跑通训练/加倍/同步流程
- [ ] 报告结果给 lead

## Task 4: brain.mjs 集成 + workspace + API（lead）

- [ ] workspace.mjs：pending-training.jsonl / mirror.brain.json / analysis/ 路径与读写
- [ ] brain.mjs：实例化四网；perceive 追加预存+分析网络分析+分析存储；休眠训练编排（调 trainer.processPending）；褪黑素容量联动公式；苏醒流程顺序（停训→回传→同步→唤醒+正面激素增值）；缺陷网络应用到 speak
- [ ] index.mjs：action generateNetwork/trainFlaw/analysisStatus；status 加 analysis/mirror/pendingCount/lastWakeAt
- [ ] test_offline.mjs 集成测试（全部模块）
- [ ] 62→N 测试全绿

## Task 5: client.js UI（lead）

- [ ] 面板 tab 切换：仪表盘 / 模型构建（自动生成 .brain.json + 挂载）
- [ ] 右上角苏醒提示框：轮询 status 检测 sleep.lastWakeAt 变化→toast「它醒来了」
- [ ] check_client.mjs 通过

## Task 6: 安装与验证（lead）

- [ ] pnpm install 无需重跑（link 不变）；重启 DSH
- [ ] API 冒烟：generateNetwork/trainFlaw/休眠训练/苏醒提示
- [ ] 提交 git
