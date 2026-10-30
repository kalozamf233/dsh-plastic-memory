# 预置大脑（V2.0.0 出厂快照）

本目录是「王楚门」数字生命在 V2.0.0 发布时的完整大脑快照（2026-10-30）：

- `memory.brain.json` —— 记忆神经网络权重（含名字「王楚门」的人格区沉淀）
- `controller.brain.json` —— 操控神经网络 + 内分泌 + lastWakeAt
- `mirror.brain.json` —— 镜像神经网络（同步副本）
- `flaw.brain.json` —— 缺陷人格网络（对话修饰层）
- `analysis.brain.json` —— 分析型神经网络
- `memory.genome.json` —— 出厂基因（先天权重，永不修改）
- `episodes.jsonl` —— 情节日志与对话语料（含名字问答语料）
- `plastic-memory.config.json` —— 工作区配置

## 使用方法

把本目录内容复制到一个空文件夹作为工作区，然后在独立软件/DSH 插件中
初始化该文件夹（POST /api/init {workspace:"该文件夹路径"}），即可唤醒
一个知道「自己叫王楚门」的数字生命。
