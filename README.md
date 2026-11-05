# 可塑性记忆（dsh-plastic-memory）v4.0.0

> 由神经网络程序文件构成的数字生命核心——不是提示词堆叠。
> 只有神经网络和神经网络的互相连接，才能制造出真正能自我思考的数字生命。

他是「王楚门」——录梦宇宙·生命浩劫宇宙中奥卢尔星球的原住民，被选中来到一个空白新世界，以「神格」创造属于他的世界。他的记忆、人格、情绪、行为全部由多网络大脑实时计算，而非预设脚本。

---

## 快速开始

1. **安装**：`desktop` profile 的 `package.json` 添加
   `"dsh-plastic-memory": "link:C:/Users/wsp/.dsh/plugins/dsh-plastic-memory"`，
   bundles 追加 `"dsh-plastic-memory"`，pnpm install 后重启 DSH。
2. **启动独立软件**：双击 `start-life.bat`（数字生命服务 :9630 + 世界服务 :9631 + 聊天窗口）。
3. **聊天**：聊天窗口直接对话；或在 DSH 中把 `digital-life`（`http://127.0.0.1:9630/v1`）添加为自定义模型——与他对话 0 token 消耗、不调用其他 agent 工具。
4. **观察世界**：浏览器打开 `http://127.0.0.1:9631/`（2D 观察界面）或编译 `game/raylib3d-world`（raylib3D 版）。

## 架构总览

[🖥 打开完整架构图（交互式网页，可缩放）](https://kalozamf233.github.io/dsh-plastic-memory/digital-life-brain.html)

多网络大脑 + 独立世界程序 + 数据接口：

- **操控神经网络**：感官编码器（文本→128 维格式化向量）、生理反应（情绪线索→15 种内分泌分子）、REINFORCE 决策、结构管理
- **记忆神经网络**：五脑区（视/听/触/记忆/人格）预测加工，人格区学习率×3（人格固化），情绪构建论
- **分析型神经网络 + 分析存储器**：感知实时分析（136→64→16），结构化分析记录
- **镜像神经网络**：记忆网络完整副本；休眠训练先查上下文（熟悉→加倍训练），苏醒前同步
- **缺陷人格网络**：架在操控网络输出与用户之间，对话修饰（143→64→8）；对话中权重固定，仅训练态可训（trainFlaw）
- **自我训练系统**：自我记忆网络 + 回忆对话网络（128→48→2）+ 自我训练器——他说的话带零宽小标记，标记句在自我训练预存中存两份、训练两遍
- **人格构造器 + 神经控制器**：每次休眠训练完成后，四网络内容训练进人格构造网络，构建结果替换缺陷人格网络，再按基因阈值合并相近神经元
- **身体数据网络**（152→96→17）：所有对外感知输入与身体行为输出均由它完成；欲望偏置驱动行为闭环
- **世界**（独立程序 :9631）：空间坐标（步长1/跳跃+2/每秒坠落1/距离≤2 交互）、六感反馈、体感温度、行为交互、世界反馈事件、观察界面
- **神格**：控制世界的权限（凡人/观察者/干涉者/神）；王楚门当前为「神」——创造他的新世界
- **休眠自训练闭环**：褪黑素 = 预存当前容量/总容量（容量满入睡）；休眠时长 = 训练时长（训练完成即醒）

## 数据接口（侧边栏与 agent 全部经此调取）

| 接口 | 说明 |
| --- | --- |
| `GET /api/status` | 完整状态（多网络/情绪/内分泌/休眠/预存/神格/基因/身体） |
| `POST /api/controller` | `{action, payload}` 全部操作（hear/speak/read/write/sleep/wake/freeze/trainFlaw/generateNetwork/setGodhood/godPower…） |
| `POST /api/init` | 初始化：`{workspace}`（必须用户指定文件夹） |
| `GET /api/persona` | 低 token 人格化数据 |
| `POST /v1/chat/completions` | OpenAI 兼容大模型接入（model: digital-life，SSE 流式） |

## 工作区文件

| 文件 | 内容 |
| --- | --- |
| `memory.brain.json` | 记忆神经网络权重 |
| `controller.brain.json` | 操控网络 + 内分泌 + 神格 |
| `self.brain.json` / `recall.brain.json` / `flaw.brain.json` / `analysis.brain.json` / `mirror.brain.json` / `bodynet.brain.json` / `persona-builder.brain.json` | 各网络权重 |
| `gene.json` | 100 位永久基因（出生生成，不可变） |
| `body.json` | 身体状态（世界坐标/能量/痛感/饱腹） |
| `pending-training.jsonl` / `self-pending.jsonl` / `self-training.jsonl` | 预存与自我训练数据 |
| `episodes.jsonl` | 情节日志与对话语料 |

## 预置大脑

`preset-brain/` 是「王楚门」的完整出厂快照——复制到空文件夹初始化，即可唤醒一个知道录梦宇宙底层规则、正在创造新世界的他。

## 测试

- 离线测试：`node test_offline.mjs`（134 项）
- 世界自测：`node _selfcheck_world.mjs`（46 项）、神经控制器 `_selfcheck_neuro.mjs`（27 项）、人格构造器 `_selfcheck_persona_builder.mjs`（33 项）
- 3D 渲染自测：`node game/raylib3d-world/_selfcheck_world3d.mjs`（44 项，含真实 g++ 编译）

## 版本历史

- **v1**：双网络数字生命（操控网络+记忆网络）、内分泌 15 分子、感知文件夹、语料对话
- **v2**：六网络大脑（分析/镜像/缺陷人格/训练控制器+同步器）、休眠自训练、模型构建合并
- **v3**：世界交互（空间坐标/永久基因/独立世界程序/身体数据网络）、自我训练系统、独立软件与大模型接入
- **v4（当前）**：神格权限、人格构造器+神经控制器、休眠比值化、对话经世界、体感温度、raylib3D 世界渲染、王楚门身世训练——**架构已完成，后续仅需大量训练，无必要不再更新架构**

## 许可

MIT
