// ============================================================================
// dsh-plastic-memory —— 可塑性神经网络（PlasticNetwork）
// ----------------------------------------------------------------------------
// 前馈网络核心。**矩阵方向约定**：W 为 out×in（rows=输出数, cols=输入数），
// 行主序 W.data[o*in + i]，z = W·x 由 matVecMul 直接实现。
//
//   · 在线反向传播（SGD + 动量 + 每层独立学习率/权重衰减）
//   · 动态结构可塑性：隐藏层神经元可增（分化自高贡献神经元）/ 可删（低使用度修剪）
//   · 神经元使用度追踪（激活绝对值 EMA），供操控网络做结构决策
//   · 权重可直接读写（操控网络有权限直接改第二个网络的参数）
//   · JSON 序列化 / 反序列化（工作区持久化）
// ============================================================================

import {
  zeros, randMat, matVecMul, matTVecMul,
  applyActivation, ACT, arrToPlain, arrFromPlain, splitRng,
} from './core.mjs'

const FORMAT_VERSION = 1

export class PlasticNetwork {
  /**
   * @param {object} spec
   *   inputSize: 输入维度
   *   layers: [{size, activation, lr, decay, mutable}]
   *     lr: 该层相对学习率乘数（默认 1）
   *     decay: L2 权重衰减系数（0=不衰减；遗忘控制）
   *     mutable: 神经元是否允许增删（仅隐藏层）
   *   seed: 初始化随机种子
   */
  constructor(spec) {
    this.inputSize = spec.inputSize
    this.tick = 0
    this.momentum = spec.momentum ?? 0.9
    this.rng = spec.rng ?? null
    this.gradNormEma = new Float64Array(spec.layers.length)
    this.layers = spec.layers.map((layerSpec, i) => {
      const inSize = i === 0 ? spec.inputSize : spec.layers[i - 1].size
      const layer = {
        size: layerSpec.size,
        activation: layerSpec.activation ?? 'tanh',
        lr: layerSpec.lr ?? 1,
        decay: layerSpec.decay ?? 0,
        mutable: layerSpec.mutable ?? false,
        bornAt: new Float64Array(layerSpec.size),
        W: randMat(layerSpec.size, inSize, this._rng()), // out × in
        b: new Float64Array(layerSpec.size),
        vW: zeros(layerSpec.size, inSize),
        vb: new Float64Array(layerSpec.size),
        usage: new Float64Array(layerSpec.size),
      }
      return layer
    })
    this._lastForward = null
    this.structuralChanges = []
  }

  _rng() {
    if (this.rng === null) {
      this.rng = (function () {
        let s = 1234567
        return function () {
          s = (Math.imul(s, 1664525) + 1013904223) >>> 0
          return s / 4294967296
        }
      })()
    }
    return splitRng(this.rng)
  }

  get paramCount() {
    let total = 0
    let prev = this.inputSize
    for (const L of this.layers) {
      total += prev * L.size + L.size
      prev = L.size
    }
    return total
  }

  get layerSizes() {
    return this.layers.map((L) => L.size)
  }

  // -------------------------------------------------------------------------
  // 前向传播
  // -------------------------------------------------------------------------

  /**
   * 前向：x(输入) → 各层 pre/post 激活。
   * @param {Float64Array} x 输入向量（长度 = inputSize）
   * @param {object} opts
   *   gainByLayer: 可选，每层增益掩码（长度=层神经元数），用于分区通道增强
   * @returns {{pre: Float64Array[], post: Float64Array[]}}
   *   post[0]=x, post[i+1]=第 i 层激活
   */
  forward(x, opts = {}) {
    const pre = []
    const post = [x]
    let cur = x
    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i]
      const z = matVecMul(L.W, cur)
      const a = new Float64Array(L.size)
      const gain = opts.gainByLayer && opts.gainByLayer[i] ? opts.gainByLayer[i] : null
      applyActivation(z, L.b, a, ACT[L.activation].f, gain)
      for (let j = 0; j < L.size; j++) {
        L.usage[j] = L.usage[j] * 0.999 + 0.001 * Math.abs(a[j])
      }
      pre.push(z)
      post.push(a)
      cur = a
    }
    this.tick++
    this._lastForward = { pre, post, x }
    return { pre, post }
  }

  // -------------------------------------------------------------------------
  // 反向传播 + 参数更新
  // -------------------------------------------------------------------------

  /**
   * 反向传播。输入输出层误差 delta（∂E/∂post_out，即"目标误差×导数"直接给出）。
   * @param {Float64Array} outputDelta 长度 = 最后一层 size（多输出块拼接后传误差）
   * @param {object} opts
   *   lrScale: 全局学习率缩放（内分泌调制）
   *   baseLr:  基础学习率
   *   frozen:  只计算梯度不更新权重（评估用）
   *   deltaScale: 每层 delta 缩放掩码数组（神经元级学习率调制，如人格区 ×3）
   * @returns {{loss: number, gradNorms: number[], deltas: Float64Array[]}}
   */
  backward(outputDelta, opts = {}) {
    const fwd = this._lastForward
    if (!fwd) return { loss: 0, gradNorms: [], deltas: [] }
    const { post } = fwd
    const L = this.layers.length
    const baseLr = opts.baseLr ?? 0.01
    const lrScale = opts.lrScale ?? 1
    const frozen = opts.frozen ?? false

    let loss = 0
    for (let i = 0; i < outputDelta.length; i++) loss += outputDelta[i] * outputDelta[i]

    const deltas = new Array(L)
    let delta = outputDelta
    const gradNorms = new Array(L)
    for (let i = L - 1; i >= 0; i--) {
      const layer = this.layers[i]
      const y = post[i + 1]
      const dPre = new Float64Array(layer.size)
      const act = ACT[layer.activation]
      const scale = opts.deltaScale && opts.deltaScale[i] ? opts.deltaScale[i] : null
      for (let j = 0; j < layer.size; j++) {
        let d = delta[j] * act.df(y[j])
        if (scale) d *= scale[j]
        dPre[j] = d
      }
      deltas[i] = dPre
      gradNorms[i] = norm(dPre)

      // 参数更新（SGD + 动量 + 每层 lr/decay）
      // W: out×in（rows=out, cols=in），ΔW[o,i] ∝ dPre[o]·x[i]
      const inputVec = post[i]
      const lr = baseLr * lrScale * layer.lr
      const momentum = this.momentum
      const decay = layer.decay
      const W = layer.W, vW = layer.vW, b = layer.b, vb = layer.vb
      const outSize = W.rows, inSize = W.cols
      const w = W.data, vw = vW.data
      for (let o = 0; o < outSize; o++) {
        const d = dPre[o]
        const wOff = o * inSize
        for (let i2 = 0; i2 < inSize; i2++) {
          const g = d * inputVec[i2]
          const idx = wOff + i2
          const v = vw[idx] = momentum * vw[idx] - lr * g
          if (!frozen) w[idx] += v - lr * decay * w[idx]
        }
      }
      for (let j = 0; j < outSize; j++) {
        const v = vb[j] = momentum * vb[j] - lr * dPre[j]
        if (!frozen) b[j] += v
      }

      // 回传到上一层：delta_prev = Wᵀ · dPre
      if (i > 0) delta = matTVecMul(W, dPre)
    }

    for (let i = 0; i < L; i++) {
      this.gradNormEma[i] = this.gradNormEma[i] * 0.95 + 0.05 * gradNorms[i]
    }
    return { loss, gradNorms, deltas }
  }

  // -------------------------------------------------------------------------
  // 便捷训练接口：train(x, target, lr) —— 目标输出 + MSE
  // -------------------------------------------------------------------------

  train(x, target, opts = {}) {
    const fwd = this.forward(x, opts.forwardOpts ?? {})
    const out = fwd.post[fwd.post.length - 1]
    const n = Math.min(out.length, target.length)
    const delta = new Float64Array(out.length)
    let loss = 0
    for (let i = 0; i < n; i++) {
      const e = out[i] - target[i]
      delta[i] = e
      loss += e * e
    }
    const res = this.backward(delta, opts)
    res.loss = loss
    return res
  }

  // -------------------------------------------------------------------------
  // 结构可塑性：增 / 删神经元
  // -------------------------------------------------------------------------

  /**
   * 新增神经元：复制当前使用度最高神经元（"源"）的输入/输出连接并加噪声变异。
   * 新神经元出生时行为与源几乎一致（无害影子），随后在训练中分化。
   * @returns {{tick, kind, layer, index, source}|null}
   */
  addNeuron(layerIdx, opts = {}) {
    const layer = this.layers[layerIdx]
    if (!layer || !layer.mutable) return null
    const next = this.layers[layerIdx + 1]
    if (!next) return null
    const mutation = opts.mutation ?? 0.15
    const rng = this._rng()

    // 源神经元：使用度最高
    let source = 0
    for (let j = 1; j < layer.size; j++) {
      if (layer.usage[j] > layer.usage[source]) source = j
    }

    const inSize = layer.W.cols
    const nextOut = next.W.rows
    const newIdx = layer.size

    // 本层 W：(out × in) → ((out+1) × in)：追加一行 = 源行 ×(1±noise)
    const W2 = zeros(layer.size + 1, inSize)
    const vW2 = zeros(layer.size + 1, inSize)
    W2.data.set(layer.W.data)
    vW2.data.set(layer.vW.data)
    const srcOff = source * inSize
    for (let i = 0; i < inSize; i++) {
      W2.data[newIdx * inSize + i] = layer.W.data[srcOff + i] * (1 + (rng() * 2 - 1) * mutation)
    }

    // 下一层 W：(nextOut × out) → (nextOut × (out+1))：插入一列 = 源列 ×(1±noise)
    const N2 = zeros(nextOut, layer.size + 1)
    const vN2 = zeros(nextOut, layer.size + 1)
    for (let o = 0; o < nextOut; o++) {
      for (let i = 0; i < layer.size; i++) {
        N2.data[o * (layer.size + 1) + i] = next.W.data[o * layer.size + i]
        vN2.data[o * (layer.size + 1) + i] = next.vW.data[o * layer.size + i]
      }
      N2.data[o * (layer.size + 1) + newIdx] =
        next.W.data[o * layer.size + source] * (1 + (rng() * 2 - 1) * mutation)
    }

    const b2 = new Float64Array(layer.size + 1)
    b2.set(layer.b)
    b2[newIdx] = layer.b[source] * (1 + (rng() * 2 - 1) * mutation)
    const vb2 = new Float64Array(layer.size + 1)
    vb2.set(layer.vb)
    const usage2 = new Float64Array(layer.size + 1)
    usage2.set(layer.usage)
    usage2[newIdx] = layer.usage[source]
    const bornAt2 = new Float64Array(layer.size + 1)
    bornAt2.set(layer.bornAt)
    bornAt2[newIdx] = this.tick

    layer.W = W2
    layer.vW = vW2
    layer.b = b2
    layer.vb = vb2
    layer.usage = usage2
    layer.bornAt = bornAt2
    layer.size++
    next.W = N2
    next.vW = vN2
    // 结构变更 → 旧前向缓存形状失效（避免 backward 越界读）
    this._lastForward = null

    const record = { tick: this.tick, kind: 'grow', layer: layerIdx, index: newIdx, source }
    this.structuralChanges.push(record)
    if (this.structuralChanges.length > 200) this.structuralChanges.shift()
    return record
  }

  /**
   * 删除神经元：移除本层 W 的一行与下一层 W 的一列。
   * @returns {{tick, kind, layer, index}|null}
   */
  pruneNeuron(layerIdx, idx) {
    const layer = this.layers[layerIdx]
    if (!layer || !layer.mutable) return null
    const next = this.layers[layerIdx + 1]
    if (!next || layer.size <= 1) return null
    if (idx < 0 || idx >= layer.size) return null

    const inSize = layer.W.cols
    const nextOut = next.W.rows

    // 本层 W：(out × in) → ((out-1) × in)：删行
    const W2 = zeros(layer.size - 1, inSize)
    const vW2 = zeros(layer.size - 1, inSize)
    let wRow = 0
    for (let o = 0; o < layer.size; o++) {
      if (o === idx) continue
      W2.data.set(layer.W.data.subarray(o * inSize, (o + 1) * inSize), wRow * inSize)
      vW2.data.set(layer.vW.data.subarray(o * inSize, (o + 1) * inSize), wRow * inSize)
      wRow++
    }

    // 下一层 W：(nextOut × out) → (nextOut × (out-1))：删列
    const N2 = zeros(nextOut, layer.size - 1)
    const vN2 = zeros(nextOut, layer.size - 1)
    for (let o = 0; o < nextOut; o++) {
      let wCol = 0
      for (let i = 0; i < layer.size; i++) {
        if (i === idx) continue
        N2.data[o * (layer.size - 1) + wCol] = next.W.data[o * layer.size + i]
        vN2.data[o * (layer.size - 1) + wCol] = next.vW.data[o * layer.size + i]
        wCol++
      }
    }

    const pick = (arr) => {
      const out = new Float64Array(arr.length - 1)
      let w = 0
      for (let i = 0; i < arr.length; i++) {
        if (i === idx) continue
        out[w++] = arr[i]
      }
      return out
    }
    layer.W = W2
    layer.vW = vW2
    layer.b = pick(layer.b)
    layer.vb = pick(layer.vb)
    layer.usage = pick(layer.usage)
    layer.bornAt = pick(layer.bornAt)
    layer.size--
    next.W = N2
    next.vW = vN2
    // 结构变更 → 旧前向缓存形状失效
    this._lastForward = null

    const record = { tick: this.tick, kind: 'prune', layer: layerIdx, index: idx }
    this.structuralChanges.push(record)
    if (this.structuralChanges.length > 200) this.structuralChanges.shift()
    return record
  }

  /** 候选修剪：使用度最低且过保护期（出生 500 tick 后）且低于阈值的神经元。 */
  findPrunable(layerIdx, { minAge = 500, usageThreshold = 0.004 } = {}) {
    const layer = this.layers[layerIdx]
    if (!layer || !layer.mutable) return null
    let best = -1
    let bestScore = Infinity
    for (let j = 0; j < layer.size; j++) {
      if (this.tick - layer.bornAt[j] < minAge) continue
      if (layer.usage[j] >= usageThreshold) continue
      const score = layer.usage[j]
      if (score < bestScore) {
        bestScore = score
        best = j
      }
    }
    return best >= 0 ? { index: best, usage: bestScore } : null
  }

  // -------------------------------------------------------------------------
  // 权重直接读写（操控网络权限）
  // -------------------------------------------------------------------------

  /** 读取某层权重矩阵的深拷贝。 */
  readWeights(layerIdx) {
    const L = this.layers[layerIdx]
    if (!L) return null
    const m = zeros(L.W.rows, L.W.cols)
    m.data.set(L.W.data)
    return m
  }

  /** 直接覆写某层权重（形状必须一致）。返回是否成功。 */
  writeWeights(layerIdx, matrix, opts = {}) {
    const L = this.layers[layerIdx]
    if (!L) return false
    if (matrix.rows !== L.W.rows || matrix.cols !== L.W.cols) return false
    if (opts.resetMomentum) {
      L.vW.data.fill(0)
      L.vb.fill(0)
    }
    L.W.data.set(matrix.data)
    return true
  }

  /** 缩放某层权重（操控网络调整"连接强度"）。 */
  scaleWeights(layerIdx, factor, opts = {}) {
    const L = this.layers[layerIdx]
    if (!L) return false
    if (opts.resetMomentum) {
      L.vW.data.fill(0)
      L.vb.fill(0)
    }
    const w = L.W.data
    for (let i = 0; i < w.length; i++) w[i] *= factor
    return true
  }

  // -------------------------------------------------------------------------
  // 序列化
  // -------------------------------------------------------------------------

  toJSON() {
    return {
      formatVersion: FORMAT_VERSION,
      inputSize: this.inputSize,
      tick: this.tick,
      momentum: this.momentum,
      gradNormEma: arrToPlain(this.gradNormEma),
      structuralChanges: this.structuralChanges,
      layers: this.layers.map((L) => ({
        size: L.size,
        activation: L.activation,
        lr: L.lr,
        decay: L.decay,
        mutable: L.mutable,
        W: arrToPlain(L.W.data),
        b: arrToPlain(L.b),
        vW: arrToPlain(L.vW.data),
        vb: arrToPlain(L.vb),
        usage: arrToPlain(L.usage),
        bornAt: arrToPlain(L.bornAt),
      })),
    }
  }

  static fromJSON(json) {
    const net = Object.create(PlasticNetwork.prototype)
    net.inputSize = json.inputSize
    net.tick = json.tick ?? 0
    net.momentum = json.momentum ?? 0.9
    net.rng = null
    net.gradNormEma = arrFromPlain(json.gradNormEma ?? [])
    net.structuralChanges = json.structuralChanges ?? []
    net._lastForward = null
    net.layers = []
    for (const L of json.layers) {
      const prevSize = net.layers.length === 0 ? net.inputSize : net.layers[net.layers.length - 1].size
      const W = zeros(L.size, prevSize) // out × in
      W.data.set(arrFromPlain(L.W))
      const vW = zeros(L.size, prevSize)
      vW.data.set(arrFromPlain(L.vW))
      net.layers.push({
        size: L.size,
        activation: L.activation ?? 'tanh',
        lr: L.lr ?? 1,
        decay: L.decay ?? 0,
        mutable: L.mutable ?? false,
        W,
        b: arrFromPlain(L.b),
        vW,
        vb: arrFromPlain(L.vb),
        usage: arrFromPlain(L.usage),
        bornAt: arrFromPlain(L.bornAt),
      })
    }
    return net
  }
}

function norm(v) {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s)
}
