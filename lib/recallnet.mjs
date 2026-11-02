// ============================================================================
// 回忆对话神经网络（RecallNet）
// 作用：在休眠时把「自我提示词预存文件」的内容过滤出对话内容，
//       存入「自我训练预存文件」；遇到带小标记的语句（他自己说过的话）
//       在自我训练预存文件里存两份。
// 结构：PlasticNetwork 128 → 48 → 2（[对话度, 自我标记度]），在线学习。
// ============================================================================

import { PlasticNetwork } from './nn/network.mjs'
import { fnv1a } from './nn/encoder.mjs'

/** 小标记：零宽空格（肉眼不可见），他说出的话会带上它。 */
export const SELF_MARK = '\u200b'

/** 去掉小标记（对外展示前使用）。 */
export function stripMark(text) {
  if (typeof text !== 'string') return text
  return text.replaceAll(SELF_MARK, '')
}

export function hasMark(text) {
  return typeof text === 'string' && text.includes(SELF_MARK)
}

/** 对话特征规则（兜底与训练标签共用）：问句/称呼/语气词等。 */
export function looksLikeDialogue(text) {
  if (typeof text !== 'string' || text.length < 2) return false
  return /[？?！!。]|你好|再见|谢谢|名字|什么|怎么|为什么|能不能|可以吗|吗$/.test(text)
}

export class RecallNet {
  constructor(opts = {}) {
    this.seed = opts.seed ?? (Math.floor(Math.random() * 1e9) >>> 0)
    this.baseLr = opts.baseLr ?? 0.01
    this.net = new PlasticNetwork({
      inputSize: 128,
      layers: [
        { size: 48, activation: 'tanh', lr: 1, mutable: false },
        { size: 2, activation: 'tanh', lr: 1, mutable: false },
      ],
      seed: this.seed,
    })
    this.frozen = false
    this.filters = 0
    this.doubled = 0
    this.lastOut = null
  }

  /** 判定一条预存内容：返回 { isDialogue, isMarked, dialogueScore, markedScore }。 */
  judge(text) {
    const t = String(text ?? '')
    const input = new Float64Array(128)
    // 字符 n-gram 哈希词袋编码
    const push = (s) => {
      const h = fnv1a(s) % 128
      input[h] += 1
    }
    for (let i = 0; i < t.length; i++) push(t[i])
    for (let i = 0; i < t.length - 1; i++) push(t.slice(i, i + 2))
    const { post } = this.net.forward(input)
    const out = post[post.length - 1]
    const dialogueScore = 1 / (1 + Math.exp(-out[0])) // sigmoid
    const markedScore = 1 / (1 + Math.exp(-out[1]))
    const isMarked = hasMark(t)
    // 网络判定 + 规则兜底（标记检测以规则为准，保证不漏）
    const isDialogue = dialogueScore > 0.5 || looksLikeDialogue(t)
    this.lastOut = { dialogueScore, markedScore, isDialogue, isMarked }
    return this.lastOut
  }

  /** 在线学习：以规则标签训练（对话度/标记度），让网络逐步学会过滤。 */
  train(text, labels = null) {
    if (this.frozen) return { skipped: 'frozen' }
    const t = String(text ?? '')
    const isMarked = hasMark(t)
    const isDialogue = labels?.isDialogue ?? looksLikeDialogue(stripMark(t))
    const input = new Float64Array(128)
    const push = (s) => {
      const h = fnv1a(s) % 128
      input[h] += 1
    }
    for (let i = 0; i < t.length; i++) push(t[i])
    for (let i = 0; i < t.length - 1; i++) push(t.slice(i, i + 2))
    const target = new Float64Array(2)
    target[0] = isDialogue ? 0.8 : -0.8
    target[1] = isMarked ? 0.8 : -0.8
    const { loss } = this.net.train(input, target, { epochs: 1 })
    return { loss }
  }

  toJSON() {
    return { seed: this.seed, baseLr: this.baseLr, net: this.net.toJSON(), frozen: this.frozen, filters: this.filters, doubled: this.doubled }
  }

  static fromJSON(data) {
    const r = new RecallNet({ seed: data?.seed, baseLr: data?.baseLr })
    if (data?.net) r.net = PlasticNetwork.fromJSON(data.net)
    r.frozen = !!(data?.frozen ?? false)
    r.filters = Number(data?.filters ?? 0)
    r.doubled = Number(data?.doubled ?? 0)
    return r
  }

  status() {
    return { params: this.net.paramCount, frozen: this.frozen, filters: this.filters, doubled: this.doubled }
  }
}
