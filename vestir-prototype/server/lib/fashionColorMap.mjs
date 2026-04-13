import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

let _cache = null

function loadVocab() {
  if (_cache) return _cache
  const raw = readFileSync(join(__dirname, 'fashion_color_vocab.json'), 'utf8')
  const data = JSON.parse(raw)
  _cache = Array.isArray(data.colors) ? data.colors : []
  return _cache
}

/**
 * Map CIELAB centroid to nearest fashion color name (same vocabulary as vision-sidecar).
 * @param {number} L
 * @param {number} a
 * @param {number} b
 */
export function fashionColorNameFromLab(L, a, b) {
  const colors = loadVocab()
  if (!colors.length) return 'Taupe'
  let best = 'Taupe'
  let bestD = Infinity
  for (const row of colors) {
    const d =
      (L - row.L) ** 2 +
      (a - row.a) ** 2 +
      (b - row.b) ** 2
    if (d < bestD) {
      bestD = d
      best = row.name
    }
  }
  return best
}
