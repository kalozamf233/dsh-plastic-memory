// ============================================================================
// dsh-plastic-memory —— 神经网络核心数学库（零依赖）
// ----------------------------------------------------------------------------
// 矩阵/向量运算：全部基于 Float64Array，行主序存储。
// 规模（~300 层内 × 几百维）下纯 JS 循环即可毫秒级完成，无需 BLAS。
// 同时提供可复现的伪随机数发生器（工作区"基因"需要确定性初始化）。
// ============================================================================

/** mulberry32：可复现 PRNG，返回 [0,1) 均匀分布。 */
export function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 从 seed 派生出新的 PRNG（每层独立随机流，保证增删神经元后其余权重不变）。 */
export function splitRng(rng) {
  return mulberry32(Math.floor(rng() * 0xffffffff))
}

// ---------------------------------------------------------------------------
// 矩阵构造
// ---------------------------------------------------------------------------

/** 全零矩阵 rows×cols。 */
export function zeros(rows, cols) {
  return { rows, cols, data: new Float64Array(rows * cols) }
}

/** 随机初始化矩阵：Xavier 均匀分布 scale = sqrt(6/(fanIn+fanOut))。 */
export function randMat(rows, cols, rng) {
  const m = zeros(rows, cols)
  const scale = Math.sqrt(6 / (rows + cols))
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = (rng() * 2 - 1) * scale
  }
  return m
}

export function zerosVec(n) {
  return new Float64Array(n)
}

export function randVec(n, rng, scale = 1) {
  const v = new Float64Array(n)
  for (let i = 0; i < n; i++) v[i] = (rng() * 2 - 1) * scale
  return v
}

// ---------------------------------------------------------------------------
// 基本运算
// ---------------------------------------------------------------------------

/** C = A · B（A: rows×cols, B: cols×n → rows×n）。 */
export function matMul(A, B) {
  const C = zeros(A.rows, B.cols)
  const a = A.data, b = B.data, c = C.data
  const inner = A.cols
  for (let i = 0; i < A.rows; i++) {
    const aOff = i * inner
    const cOff = i * B.cols
    for (let k = 0; k < inner; k++) {
      const av = a[aOff + k]
      if (av === 0) continue
      const bOff = k * B.cols
      for (let j = 0; j < B.cols; j++) {
        c[cOff + j] += av * b[bOff + j]
      }
    }
  }
  return C
}

/** u = A · v（A: rows×cols, v: cols）。 */
export function matVecMul(A, v) {
  const u = new Float64Array(A.rows)
  const a = A.data
  for (let i = 0; i < A.rows; i++) {
    let s = 0
    const off = i * A.cols
    for (let j = 0; j < A.cols; j++) s += a[off + j] * v[j]
    u[i] = s
  }
  return u
}

/** u = Aᵀ · v（A: rows×cols, v: rows → cols 维结果）。 */
export function matTVecMul(A, v) {
  const u = new Float64Array(A.cols)
  const a = A.data
  for (let i = 0; i < A.rows; i++) {
    const av = v[i]
    if (av === 0) continue
    const off = i * A.cols
    for (let j = 0; j < A.cols; j++) u[j] += a[off + j] * av
  }
  return u
}

/** dst += alpha * src（同长度向量）。 */
export function axpy(dst, src, alpha = 1) {
  for (let i = 0; i < dst.length; i++) dst[i] += alpha * src[i]
  return dst
}

/** 向量点积。 */
export function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

/** L2 范数。 */
export function norm2(v) {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s)
}

/** 余弦相似度。 */
export function cosine(a, b) {
  const na = norm2(a), nb = norm2(b)
  if (na === 0 || nb === 0) return 0
  return dot(a, b) / (na * nb)
}

/** 逐元素：dst[i] = f(dst[i] + bias[i])（含增益 mask，供分区掩码使用）。 */
export function applyActivation(src, bias, dst, f, gain = null) {
  for (let i = 0; i < src.length; i++) {
    let v = f(src[i] + bias[i])
    if (gain !== null) v *= gain[i]
    dst[i] = v
  }
}

// ---------------------------------------------------------------------------
// 激活函数（f 与导数 df 都接收"激活后值 y"，便于缓存计算）
// ---------------------------------------------------------------------------

export const ACT = {
  tanh: { f: Math.tanh, df: (y) => 1 - y * y },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), df: (y) => y * (1 - y) },
  relu: { f: (x) => (x > 0 ? x : 0), df: (y) => (y > 0 ? 1 : 0) },
  linear: { f: (x) => x, df: () => 1 },
}

// ---------------------------------------------------------------------------
// 序列化辅助
// ---------------------------------------------------------------------------

export function arrToPlain(v) {
  return Array.from(v)
}

export function arrFromPlain(a) {
  return new Float64Array(a)
}
