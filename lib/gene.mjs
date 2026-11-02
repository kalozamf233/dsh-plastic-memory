// ============================================================================
// 基因系统：100 位 26 字母随机字符串，出生时生成，永久不可变
// ============================================================================

import { randomBytes } from 'node:crypto'

const GENE_LEN = 100
const ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

/** 生成 100 位随机基因字符串。 */
export function generateGene() {
  const bytes = randomBytes(GENE_LEN)
  let gene = ''
  for (let i = 0; i < GENE_LEN; i++) {
    gene += ALPHABET[bytes[i] % 26]
  }
  return gene
}

/** 基因摘要（用于低 token 展示）：首 12 位 + 统计。 */
export function geneSummary(gene) {
  if (typeof gene !== 'string' || gene.length === 0) return { length: 0 }
  const counts = {}
  for (const ch of gene) counts[ch] = (counts[ch] ?? 0) + 1
  const top = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([ch, n]) => `${ch}×${n}`)
  return {
    length: gene.length,
    head: gene.slice(0, 12),
    tail: gene.slice(-8),
    topLetters: top,
  }
}

/** 基因确定性派生值（0..1），可作种子/特质参考（不改变基因本身）。 */
export function geneHash(gene) {
  let h = 2166136261
  for (let i = 0; i < gene.length; i++) {
    h ^= gene.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

export const GENE_LENGTH = GENE_LEN
