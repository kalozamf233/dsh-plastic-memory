// ============================================================================
// dsh-plastic-memory —— 感知编码器（SenseEncoder）
// ----------------------------------------------------------------------------
// 操控神经网络的"感官"：把文本感知信号编码为格式化向量。
//   1. 哈希词袋（字符 2-gram/3-gram + 词，512 维稀疏）
//   2. 可训练线性投影 bag(512) → latent(128)，tanh 归一
//   3. 可训练反投影 latent → bag 重构，以重构误差在线训练投影矩阵
//      （无监督自编码式学习：编码器随感知流真实地学习表征，而非固定哈希）
// 编码结果是唯一被允许进入记忆神经网络的"格式化数据"形态之一。
// ============================================================================

import { zeros, randMat, matVecMul, matTVecMul, arrToPlain, arrFromPlain, mulberry32 } from './core.mjs'

export const BAG_SIZE = 512
export const LATENT_SIZE = 128

export const SENSE_MODULES = ['visual', 'audio', 'touch', 'memory', 'persona']

/** FNV-1a 风格 32 位哈希。 */
export function fnv1a(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

export class SenseEncoder {
  constructor(seed = 20240607) {
    const rng = mulberry32(seed)
    this.proj = randMat(LATENT_SIZE, BAG_SIZE, rng) // 512 → 128（行主序：128×512）
    this.unproj = randMat(BAG_SIZE, LATENT_SIZE, rng) // 128 → 512 重构
    this.recentLoss = 1
    this.trained = 0
  }

  /** 文本 → 哈希词袋（512 维，归一化到单位 L2）。 */
  bagOf(text) {
    const bag = new Float64Array(BAG_SIZE)
    if (!text) return bag
    const t = String(text).toLowerCase().trim()
    // 词（按空白/标点切分）
    const words = t.split(/[^\p{L}\p{N}_]+/u).filter(Boolean)
    for (const w of words) {
      bag[fnv1a('w:' + w) % BAG_SIZE] += 1
      if (w.length > 4) {
        // 词首尾 n-gram（形态特征，兼顾中英）
        const head = w.slice(0, 2)
        const tail = w.slice(-2)
        bag[fnv1a('wh:' + head) % BAG_SIZE] += 0.5
        bag[fnv1a('wt:' + tail) % BAG_SIZE] += 0.5
      }
    }
    // 字符 2-gram / 3-gram（对中文单字敏感）
    const chars = Array.from(t.replace(/\s+/g, ''))
    for (let i = 0; i < chars.length - 1; i++) {
      bag[fnv1a('c2:' + chars[i] + chars[i + 1]) % BAG_SIZE] += 0.8
      if (i < chars.length - 2) {
        bag[fnv1a('c3:' + chars[i] + chars[i + 1] + chars[i + 2]) % BAG_SIZE] += 0.4
      }
    }
    // 单字符
    for (const c of chars) bag[fnv1a('c1:' + c) % BAG_SIZE] += 0.3
    // L2 归一
    let n = 0
    for (let i = 0; i < BAG_SIZE; i++) n += bag[i] * bag[i]
    if (n > 0) {
      n = Math.sqrt(n)
      for (let i = 0; i < BAG_SIZE; i++) bag[i] /= n
    }
    return bag
  }

  /** 编码：bag → latent（128 维 tanh，即"格式化数据"）。 */
  encode(text) {
    const bag = this.bagOf(text)
    const latent = matVecMul(this.proj, bag)
    for (let i = 0; i < latent.length; i++) latent[i] = Math.tanh(latent[i])
    return { latent, bag }
  }

  /** 已知 latent 的反投影重构。 */
  decode(latent) {
    return matVecMul(this.unproj, latent)
  }

  /**
   * 在线训练：以重构误差（MSE）更新投影/反投影矩阵。
   * 使编码器随感知流持续学习（无监督）。
   */
  train(text, lr = 0.002) {
    const { latent, bag } = this.encode(text)
    const recon = this.decode(latent)
    const err = new Float64Array(BAG_SIZE)
    let loss = 0
    for (let i = 0; i < BAG_SIZE; i++) {
      const e = recon[i] - bag[i]
      err[i] = e
      loss += e * e
    }
    // dL/dunproj = err ⊗ latent
    for (let i = 0; i < BAG_SIZE; i++) {
      const ei = err[i]
      if (ei === 0) continue
      const off = i * LATENT_SIZE
      for (let j = 0; j < LATENT_SIZE; j++) {
        this.unproj.data[off + j] -= lr * ei * latent[j]
      }
    }
    // dL/dproj：经 tanh 导数（1 - latent²）与 unproj 反传
    const gLatent = matTVecMul(this.unproj, err)
    const dProj = new Float64Array(LATENT_SIZE)
    for (let i = 0; i < LATENT_SIZE; i++) {
      dProj[i] = gLatent[i] * (1 - latent[i] * latent[i])
    }
    // Δproj = dProj ⊗ bag
    for (let i = 0; i < LATENT_SIZE; i++) {
      const gi = dProj[i]
      if (gi === 0) continue
      const off = i * BAG_SIZE
      for (let j = 0; j < BAG_SIZE; j++) {
        this.proj.data[off + j] -= lr * gi * bag[j]
      }
    }
    this.recentLoss = this.recentLoss * 0.95 + 0.05 * loss
    this.trained++
    return loss
  }

  toJSON() {
    return {
      proj: arrToPlain(this.proj.data),
      unproj: arrToPlain(this.unproj.data),
      recentLoss: this.recentLoss,
      trained: this.trained,
    }
  }

  fromJSON(json) {
    if (json.proj) this.proj.data.set(arrFromPlain(json.proj))
    if (json.unproj) this.unproj.data.set(arrFromPlain(json.unproj))
    this.recentLoss = json.recentLoss ?? 1
    this.trained = json.trained ?? 0
  }
}
