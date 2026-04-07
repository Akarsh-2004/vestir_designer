/**
 * Color pipeline helpers: gray-world, CLAHE on LAB L*, inner crop, alpha-aware LAB K-means.
 */
import sharp from 'sharp'

/** @param {Buffer} buffer */
export async function detectImageMime(buffer) {
  if (!buffer?.length) return 'image/jpeg'
  if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) return 'image/png'
  if (buffer[0] === 0xff && buffer[1] === 0xd8) return 'image/jpeg'
  return 'image/jpeg'
}

export function rgbToLab(rgbR, rgbG, rgbB) {
  let r = rgbR / 255
  let g = rgbG / 255
  let b = rgbB / 255
  r = r > 0.04045 ? ((r + 0.055) / 1.055) ** 2.4 : r / 12.92
  g = g > 0.04045 ? ((g + 0.055) / 1.055) ** 2.4 : g / 12.92
  b = b > 0.04045 ? ((b + 0.055) / 1.055) ** 2.4 : b / 12.92
  let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
  let y = r * 0.2126729 + g * 0.7151522 + b * 0.072175
  let z = r * 0.0193339 + g * 0.119192 + b * 0.9503041
  x /= 0.95047
  y /= 1.0
  z /= 1.08883
  const fx = x > 0.008856 ? Math.cbrt(x) : 7.787 * x + 16 / 116
  const fy = y > 0.008856 ? Math.cbrt(y) : 7.787 * y + 16 / 116
  const fz = z > 0.008856 ? Math.cbrt(z) : 7.787 * z + 16 / 116
  return {
    L: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz),
  }
}

export function labDistance(p, q) {
  return Math.sqrt((p.L - q.L) ** 2 + (p.a - q.a) ** 2 + (p.b - q.b) ** 2)
}

/** Gray-world illuminant normalization (same idea as server normalizeForColorAnalysis). */
export async function applyGrayWorld(buffer) {
  const { data, info } = await sharp(buffer).removeAlpha().raw().toBuffer({ resolveWithObject: true })
  const channels = info.channels
  let rSum = 0
  let gSum = 0
  let bSum = 0
  let count = 0
  for (let i = 0; i < data.length; i += channels) {
    rSum += data[i]
    gSum += data[i + 1]
    bSum += data[i + 2]
    count += 1
  }
  const avgR = rSum / Math.max(1, count)
  const avgG = gSum / Math.max(1, count)
  const avgB = bSum / Math.max(1, count)
  const gray = (avgR + avgG + avgB) / 3
  const gainR = gray / Math.max(1, avgR)
  const gainG = gray / Math.max(1, avgG)
  const gainB = gray / Math.max(1, avgB)
  const corrected = Buffer.from(data)
  for (let i = 0; i < corrected.length; i += channels) {
    corrected[i] = Math.max(0, Math.min(255, Math.round(corrected[i] * gainR)))
    corrected[i + 1] = Math.max(0, Math.min(255, Math.round(corrected[i + 1] * gainG)))
    corrected[i + 2] = Math.max(0, Math.min(255, Math.round(corrected[i + 2] * gainB)))
  }
  return sharp(corrected, { raw: { width: info.width, height: info.height, channels } }).jpeg({ quality: 92 }).toBuffer()
}

/**
 * Tone normalization before LAB K-means (approximates “help L*” without fragile custom CLAHE).
 * Disable with VESTIR_COLOR_CLAHE=0 in env from caller.
 */
export async function applyClaheLab(buffer) {
  try {
    return await sharp(buffer).normalize().jpeg({ quality: 92 }).toBuffer()
  } catch {
    return buffer
  }
}

/** Shrink crop toward center by ratio (reduces edge bleed for color stats). */
export async function innerCropBuffer(buffer, ratio = 0.12) {
  const meta = await sharp(buffer).metadata()
  const w = meta.width ?? 0
  const h = meta.height ?? 0
  if (!w || !h) return buffer
  const dx = Math.floor(w * ratio * 0.5)
  const dy = Math.floor(h * ratio * 0.5)
  const left = Math.max(0, dx)
  const top = Math.max(0, dy)
  const width = Math.max(8, w - 2 * dx)
  const height = Math.max(8, h - 2 * dy)
  return sharp(buffer).extract({ left, top, width, height }).jpeg({ quality: 92 }).toBuffer()
}

export function pointInPolygon(x, y, poly) {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x
    const yi = poly[i].y
    const xj = poly[j].x
    const yj = poly[j].y
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi
    if (intersect) inside = !inside
  }
  return inside
}

/**
 * Apply polygon mask (normalized 0–1 coords). Returns PNG buffer with alpha outside polygon.
 * @param {Buffer} buffer
 * @param {{ x: number, y: number }[]} polygonNorm
 */
export async function maskPolygonToPng(buffer, polygonNorm) {
  const meta = await sharp(buffer).metadata()
  const w = meta.width ?? 1
  const h = meta.height ?? 1
  const poly = polygonNorm.map((p) => ({
    x: p.x <= 1.01 && p.x >= 0 ? p.x * w : p.x,
    y: p.y <= 1.01 && p.y >= 0 ? p.y * h : p.y,
  }))
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const p of poly) {
    minX = Math.min(minX, p.x)
    minY = Math.min(minY, p.y)
    maxX = Math.max(maxX, p.x)
    maxY = Math.max(maxY, p.y)
  }
  minX = Math.max(0, Math.floor(minX))
  minY = Math.max(0, Math.floor(minY))
  maxX = Math.min(w - 1, Math.ceil(maxX))
  maxY = Math.min(h - 1, Math.ceil(maxY))
  const bw = Math.max(1, maxX - minX)
  const bh = Math.max(1, maxY - minY)

  const { data, info } = await sharp(buffer).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const ch = info.channels
  const stride = w * ch
  const rgba = Buffer.alloc(bw * bh * 4)
  for (let y = 0; y < bh; y += 1) {
    for (let x = 0; x < bw; x += 1) {
      const gx = minX + x
      const gy = minY + y
      const inside = pointInPolygon(gx + 0.5, gy + 0.5, poly)
      const src = (gy * w + gx) * ch
      const dst = (y * bw + x) * 4
      if (inside) {
        rgba[dst] = data[src]
        rgba[dst + 1] = data[src + 1]
        rgba[dst + 2] = data[src + 2]
        rgba[dst + 3] = ch >= 4 ? data[src + 3] : 255
      } else {
        rgba[dst] = 0
        rgba[dst + 1] = 0
        rgba[dst + 2] = 0
        rgba[dst + 3] = 0
      }
    }
  }
  return sharp(rgba, { raw: { width: bw, height: bh, channels: 4 } }).png({ compressionLevel: 6 }).toBuffer()
}

/**
 * Full color-analysis prep: inner crop → gray-world → CLAHE (optional).
 * @param {Buffer} buffer
 * @param {{ innerCrop?: number, grayWorld?: boolean, clahe?: boolean }} opts
 */
export async function prepareColorAnalysisBuffer(buffer, opts = {}) {
  const { innerCrop = 0.12, grayWorld = true, clahe = true } = opts
  let b = buffer
  if (innerCrop > 0) b = await innerCropBuffer(b, innerCrop)
  if (grayWorld) b = await applyGrayWorld(b)
  if (clahe) b = await applyClaheLab(b)
  return b
}
