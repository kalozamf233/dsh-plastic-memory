// ============================================================================
// dsh-plastic-memory —— 神经控制器（NeuroController）
// ----------------------------------------------------------------------------
// 基因驱动的网络调控工具（纯函数，零依赖，只操作 PlasticNetwork 权重 JSON）：
//   · geneThreshold(gene)：100 位 a-z 基因 → 固定计算方式 → 合并阈值（0.005~0.1）
//   · mergeSimilarNeurons(netJson, threshold)：合并隐藏层相邻相似神经元
//
// 权重 JSON 格式 = PlasticNetwork.toJSON()（layers[i] = {size, W:out×in 行主序,
// b, vW, vb, usage, bornAt, ...}，inputSize 为隐式输入层维度）。
// 只合并隐藏层（除最后一层外的所有层），输入层（隐式）/输出层（最后一层）不动。
// 合并后 JSON 保持可被 PlasticNetwork.fromJSON 直接加载。
// ============================================================================

// ---------------------------------------------------------------------------
// 基因 → 阈值
// ---------------------------------------------------------------------------

/**
 * 基因 100 位 a-z → 合并阈值（固定计算方式，同基因永远同阈值）。
 * 每字符取 charCode 低 5 位（MSB 在前）拼接为 500 位二进制串，
 * 取前 100 位的 1 占比 p，阈值 = 0.005 + p * 0.095（范围约 0.005~0.1）。
 */
export function geneThreshold(gene) {
  const g = String(gene ?? '').slice(0, 100)
  let bits = ''
  for (const ch of g) {
    bits += (ch.charCodeAt(0) & 0x1f).toString(2).padStart(5, '0')
  }
  const first100 = bits.slice(0, 100)
  let ones = 0
  for (const b of first100) {
    if (b === '1') ones++
  }
  const p = ones / 100
  return 0.005 + p * 0.095
}

// ---------------------------------------------------------------------------
// 神经元合并
// ---------------------------------------------------------------------------

/**
 * 同层相邻神经元（i, i+1）的归一化权重距离：
 * 各自权重向量除以自身最大幅值（max=0 时保持零向量）后取欧氏距离。
 */
function neuronDist(W, cols, a, b) {
  const va = W.slice(a * cols, (a + 1) * cols)
  const vb = W.slice(b * cols, (b + 1) * cols)
  let ma = 0
  let mb = 0
  for (const v of va) if (Math.abs(v) > ma) ma = Math.abs(v)
  for (const v of vb) if (Math.abs(v) > mb) mb = Math.abs(v)
  let s = 0
  for (let k = 0; k < cols; k++) {
    const d = (ma === 0 ? va[k] : va[k] / ma) - (mb === 0 ? vb[k] : vb[k] / mb)
    s += d * d
  }
  return Math.sqrt(s)
}

/** 把 layer L 的相邻神经元 i、i+1 合并（权重/动量/偏置/使用度取平均，出生取较早）。 */
function mergeAdjacent(net, L, i) {
  const layer = net.layers[L]
  const next = net.layers[L + 1]
  const cols = Math.floor(layer.W.length / layer.size)
  const newSize = layer.size - 1

  // 本层行合并（W/vW：rows×cols；b/vb/usage/bornAt：rows）
  const mergeRows = (arr) => {
    const out = new Array(newSize * cols)
    for (let r = 0; r < newSize; r++) {
      const src = r < i ? r : r === i ? i : r + 1
      for (let c = 0; c < cols; c++) {
        let v = arr[src * cols + c]
        if (r === i) v = (arr[i * cols + c] + arr[(i + 1) * cols + c]) / 2
        out[r * cols + c] = v
      }
    }
    return out
  }
  const mergeVec = (arr, avg) => {
    const out = new Array(newSize)
    for (let r = 0; r < newSize; r++) {
      const src = r < i ? r : r === i ? i : r + 1
      let v = arr[src]
      if (r === i) v = avg ? (arr[i] + arr[i + 1]) / 2 : Math.min(arr[i], arr[i + 1])
      out[r] = v
    }
    return out
  }
  layer.W = mergeRows(layer.W)
  layer.vW = mergeRows(layer.vW)
  layer.b = mergeVec(layer.b, true)
  layer.vb = mergeVec(layer.vb, true)
  layer.usage = mergeVec(layer.usage, true)
  layer.bornAt = mergeVec(layer.bornAt, false)
  layer.size = newSize

  // 下一层列合并（W/vW：out×oldSize → out×newSize）
  const oldCols = newSize + 1
  const outRows = next.W.length / oldCols
  const mergeCols = (arr) => {
    const out = new Array(outRows * newSize)
    for (let o = 0; o < outRows; o++) {
      for (let k = 0; k < newSize; k++) {
        const src = k < i ? k : k === i ? i : k + 1
        let v = arr[o * oldCols + src]
        if (k === i) v = (arr[o * oldCols + i] + arr[o * oldCols + i + 1]) / 2
        out[o * newSize + k] = v
      }
    }
    return out
  }
  next.W = mergeCols(next.W)
  next.vW = mergeCols(next.vW)
}

/**
 * 合并隐藏层相邻相似神经元：对每个隐藏层（除最后一层外的所有层），
 * 按相邻索引对 (i, i+1) 计算归一化权重距离（欧氏距离 / 最大幅值归一化），
 * dist < threshold → 合并为一个神经元（权重取两者平均）。
 * 输入层（隐式）/输出层（最后一层）不动；合并后 JSON 可 fromJSON 加载。
 * @param {object} netJson PlasticNetwork.toJSON() 格式的权重 JSON（不修改入参）
 * @param {number} threshold 合并阈值（dist < threshold 才合并；0 = 不合并）
 * @returns {{mergedCount:number, newJson:object, pairs:Array<{layer,i,j,dist}>}}
 */
export function mergeSimilarNeurons(netJson, threshold = 0.05) {
  if (!netJson || typeof netJson !== 'object' || !Array.isArray(netJson.layers)) {
    return { mergedCount: 0, newJson: netJson ?? {}, pairs: [] }
  }
  const net = JSON.parse(JSON.stringify(netJson)) // 深拷贝，不修改入参
  const th = Number.isFinite(Number(threshold)) ? Number(threshold) : 0.05
  const pairs = []
  let mergedCount = 0
  // 只合并隐藏层：layers[0 .. n-2]（最后一层 = 输出层，不动）
  for (let L = 0; L < net.layers.length - 1; L++) {
    const layer = net.layers[L]
    const cols = Math.floor(layer.W.length / layer.size)
    let i = 0
    while (i < layer.size - 1) {
      const d = neuronDist(layer.W, cols, i, i + 1)
      if (d < th) {
        mergeAdjacent(net, L, i)
        pairs.push({ layer: L, i, j: i + 1, dist: Number(d.toFixed(6)) })
        mergedCount++
        // size 已减一；i 不动 → 合并后的新神经元继续与其右邻比较
      } else {
        i++
      }
    }
  }
  return { mergedCount, newJson: net, pairs }
}
