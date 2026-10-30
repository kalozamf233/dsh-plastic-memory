# 可塑性记忆 · 大脑架构 V2 设计文档

> 日期：2026-10-30　状态：待用户批准

## 意图

把「可塑性记忆」数字生命从双网络架构升级为多网络协作架构：自动生成神经网络（替代 AI 模型构建插件的 PyTorch 生成）、分析网络+分析存储、镜像网络、缺陷人格网络、训练控制器与同步器，以及以「褪黑素=预存文件容量进度条」为核心的休眠自训练闭环。全部在 DSH 中生效。

## 目标 / 非目标 / 反目标

**目标**
1. dsh-ai-model-builder 合并为「可塑性记忆」面板的子页面；手动生成改为自动；生成物为可塑性记忆自己的网络格式（MemoryNet .brain.json，可直接挂载为外部扩容）
2. 分析型神经网络 + 分析存储器：感知流实时分析，结果结构化保存
3. 镜像神经网络：记忆网络权重的完整副本
4. 操控网络：提示词预存文件 + 休眠自训练 + 褪黑素容量联动 + 苏醒提示框
5. 缺陷人格神经网络：架在操控网络与用户聊天之间加工美化输出；对话中权重冻结；仅训练态、仅用户 API 可训练
6. 训练控制器 + 同步器：休眠训练执行、镜像查上下文（命中加倍）、苏醒前同步

**非目标**
- 不引入任何外部依赖（继续纯 JS 零依赖）
- 不替换现有对话引擎（语料检索对话继续工作，缺陷网络是叠加的修饰层）
- 不做真实文本生成模型（网络仍是潜向量级表征）

**反目标（即使功能可用也算失败）**
- 休眠训练拖垮心跳实时性（每秒 tick 必须毫秒级返回）
- 训练导致权重 NaN/损坏（每次训练后校验，异常回滚）
- 镜像与记忆网络永久不一致（同步必须原子）
- 苏醒流程死锁（训练停不下来/同步失败时必须有兜底苏醒路径）

## 约束

- 纯 JS 零依赖，复用 `lib/nn/*`（PlasticNetwork/MemoryNet/encoder/endocrine）
- 工作区向后兼容：旧工作区无新文件时自动初始化，不破坏已有权重
- 心跳 1 秒：休眠训练每 tick 最多训练 N 条（默认 3 条/秒），长文本可跨 tick 分批
- 所有新权重文件原子写（tmp+rename），沿用 Workspace 约定

## 方法（Approaches）

**A. 分析网络架构** —— 候选：①复用 MemoryNet 小型化（分区结构对分析无用）②专用两层小网络（input=latent128+内分泌效应8 → 64 → 输出分析向量16：主题/情绪/新颖度/重要性）+ 规则辅助。**选②**：职责清晰、参数少、每秒可跑；分析存储器 = 工作区 `analysis/` 目录（每分析一条 JSON：时间/文本摘要/分析向量/情绪），内存保留最近 200 条索引。

**B. 镜像网络** —— 记忆网络权重 JSON 的独立副本（`mirror.brain.json`）。同步 = 序列化复制（同步器负责）。查询上下文 = 用条目 latent 对镜像做 recall（记忆区激活 + 相似度），相似度 > 阈值（0.6）判定"找到"。

**C. 缺陷人格网络** —— 不能做文本生成（违背约束），设计为**对话修饰层**：DefectNet（两层小网络）输入 = 回复内容 latent(128) + 内分泌状态(15) → 输出 = 修饰参数向量（语气软化度、亲密度、口头禅选择、回避/热情方向）。speak 组装回复时应用修饰参数（选词/句式变体由参数驱动）。权重冻结逻辑：`flaw.frozen=true` 时 observe 只前向不反传；只有 `/controller action=trainFlaw`（用户 API）可训练。

**D. 训练控制器** —— 休眠期间每 tick 从 `pending-training.jsonl` 取最多 3 条：逐条编码 → 镜像网络查上下文（recall 相似度 > 0.6 = 找到）→ 找到则 2×epochs 训练记忆网络，未找到 1×epochs → 从预存文件移除该条 → 更新褪黑素。训练用记忆网络的 observe/writePersona 既有接口，trainer 只做编排。

**E. 褪黑素容量联动** —— 公式：`melatonin = baseline + min(0.85, pendingCount × 0.02)`（50 条封顶）。预存条目通过感知与用户消息持续追加；休眠训练消耗条目 → 褪黑素随之下滑；pendingCount=0 → 自然苏醒。苏醒时正面情绪激素增值：dopamine +0.6、endorphin +0.6、serotonin +0.5（一次性大剂量释放）。

**F. 苏醒流程（顺序不可变）** —— ①trainer 停止（不再取新条目）②剩余条目回写预存文件 ③同步器把记忆网络权重同步到镜像 ④执行 _wake ⑤client 检测 `sleep.lastWakeAt` 变化 → 右上角弹「它醒来了」提示框。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `lib/nn/analysisnet.mjs`（新建） | AnalysisNet + AnalysisStore |
| `lib/nn/flawnet.mjs`（新建） | DefectNet（修饰层）+ 序列化 |
| `lib/trainer.mjs`（新建） | TrainingController + Syncer + 镜像管理 |
| `lib/brain.mjs`（改） | 集成四网、预存、褪黑素联动、苏醒流程 |
| `lib/workspace.mjs`（改） | pending-training.jsonl / mirror.brain.json / analysis/ |
| `lib/index.mjs`（改） | 新 API action：generateNetwork/trainFlaw/analysisStatus |
| `lib/client.js`（改） | 子页面 tab、右上角苏醒提示框 |
| `test_offline.mjs`（改） | 新增各模块测试 |

## API 增量

- `POST /controller {action:'generateNetwork', payload:{path, name?}}` → 自动生成 .brain.json（MemoryNet 初始化权重）并可选挂载为扩展
- `POST /controller {action:'trainFlaw', payload:{text, epochs?}}` → 用户训练缺陷人格网络（唯一训练入口）
- `POST /controller {action:'analysisStatus'}` → 分析存储器摘要
- `GET /status` 增加：`analysis`、`mirror:{syncedAt,params}`、`pendingCount`、`sleep.lastWakeAt`

## 修订记录（builder check 后，2026-10-30）

1. **苏醒状态机统一**：所有唤醒路径（感知交互/手动 wake/自然醒）一律走 `_wakeFlow(source)`：①trainer.stop()（处理中条目原子保留在 pending 文件=剩余上下文已回传）②syncMirror() ③_wake ④自然醒额外释放正面激素（dopamine+0.6/endorphin+0.6/serotonin+0.5，交互唤醒仅小剂量 0.2）。同步失败不阻塞苏醒，记录 lastError。
2. **镜像相似度度量**：`mirrorRecall(latent)` = 镜像网络 recall 输出 pred 与输入 latent 的 **cosine（预测吻合度=熟悉度）** 与 memAct 均值×2 取较大者，>0.6 判定"找到"。新增 MemoryNet 无需改动（pred 已返回）。
3. **加倍训练语义**：observe 是单步预测加工训练，"加倍"=同一条目 observe 2 次（lrScale 1.2 固定由 trainer 传入，绕过内分泌 melatonin 抑制）。writePersona 仅用于用户主动 write。
4. **DefectNet 训练目标与修饰映射**：train 目标由文本情绪关键词驱动（含喜悦/爱→warmth 0.8；悲伤→intimacy 0.7；愤怒→deflect 0.6；默认温和基线）。apply 输出 5 参数 → speak 措辞映射：warmth>0.6 加亲近语气词（"呀/呢"）；embellish>0.6 追加修饰句；deflect>0.6 用回避式措辞变体；quirk>0.5 追加口头禅。映射表实现于 brain.speak。
5. **新网络持久化**：flaw.brain.json、analysis.brain.json、mirror.brain.json 三个文件；save() 全量保存、load() 恢复（缺失则新建）；lastWakeAt 随 controller.brain.json 持久化。
6. **入睡条件（pending 驱动）**：`pendingCount ≥ 10 且 wakefulness ≤ 0.5` 或 `pendingCount ≥ 30`（无视清醒度）→ 入睡。重启后 pendingCount 从文件恢复 → 褪黑素恢复进度条值 → 满足条件自动入睡继续训练（修复死循环）。
7. **褪黑素双身份裁决**：休眠期 melatonin 分子值 = 进度条公式（0.12+min(0.85, count×0.02)）每小时心跳覆写；非休眠期由内分泌正常衰减。休眠训练 lrScale 由 trainer 显式传参，不受 effects 的 melatonin 抑制。
8. **pending 规则**：用户对他说的所有话（hear/speak 输入，≥2 字）自动追加（清醒期），条目格式 {t,text,latent32,module}，上限 50 条（超出丢最旧）。每 tick 训练 ≤3 条。
9. **插件合并部署**：desktop profile 移除 dsh-ai-model-builder（dependencies+bundles），pnpm install；其功能由「模型构建」子页面替代（自动生成 .brain.json，不再生成 PyTorch、不再开训练会话）。
10. **苏醒提示框范围**：面板挂载时轮询检测 sleep.lastWakeAt 变化弹右上角提示；lastWakeAt 持久化，面板重开时若 sleep.awakeEventId 未消费则补弹。DSH 无全局通知通道，此为插件内最优解（如实告知用户）。
11. **冻结态×休眠**：mode=frozen 时 trainer 不训练（pending 保留）；flaw 训练条件 = mode==='training'（freeze 即冻结缺陷网络）。
12. **分析存储性能**：内存环形缓冲 200 条，每 30 秒批量落盘 analysis/ 单文件追加，避免批量感知文件爆炸。
