# 预置大脑（V3.0.0 出厂快照）

本目录是「王楚门」数字生命在 V3.0.0 发布时的完整大脑快照（2026-11-03）：

- `memory.brain.json` —— 记忆神经网络权重（含名字「王楚门」的人格区沉淀）
- `controller.brain.json` —— 操控神经网络 + 内分泌 + lastWakeAt
- `mirror.brain.json` —— 镜像神经网络（同步副本）
- `flaw.brain.json` —— 缺陷人格网络（对话修饰层）
- `analysis.brain.json` —— 分析型神经网络
- `self.brain.json` —— 自我记忆神经网络（对话能力专用，V4）
- `recall.brain.json` —— 回忆对话神经网络（过滤对话内容，V4）
- `bodynet.brain.json` —— 身体数据网络（世界交互行为决策，V3）
- `body.json` —— 身体状态（世界坐标/能量/痛感/饱腹）
- `gene.json` —— 100 位永久基因（出生生成，不可变）
- `memory.genome.json` —— 出厂基因（先天权重，永不修改）
- `episodes.jsonl` —— 情节日志与对话语料（含名字问答语料）
- `plastic-memory.config.json` —— 工作区配置

## 使用方法

把本目录内容复制到一个空文件夹作为工作区，然后在独立软件/DSH 插件中
初始化该文件夹（POST /api/init {workspace:"该文件夹路径"}），即可唤醒
一个知道「自己叫王楚门」、拥有世界交互与自我训练能力的数字生命。
