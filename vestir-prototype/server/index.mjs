import cors from 'cors'
import express from 'express'
import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import { fileURLToPath } from 'node:url'
import { GoogleGenAI } from '@google/genai'
import vision from '@google-cloud/vision'
import { z } from 'zod'
import dotenv from 'dotenv'
import {
  prepareColorAnalysisBuffer,
  maskPolygonToPng,
  detectImageMime,
} from './lib/vestirColor.mjs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const rootDir = path.resolve(__dirname, '..')
dotenv.config({ path: path.resolve(rootDir, '..', '.env') })
dotenv.config({ path: path.resolve(rootDir, '.env') })

const storageDir = path.join(rootDir, 'server', 'storage')
const processedDir = path.join(storageDir, 'processed')
const embeddingsFile = path.join(storageDir, 'embeddings.json')

const app = express()
const port = Number(process.env.API_PORT ?? 8787)

app.use(cors())
app.use(express.json({ limit: '20mb' }))
app.use('/storage', express.static(storageDir))

function getGeminiApiKey() {
  const candidates = [
    { env: 'GEMINI_API_KEY', value: process.env.GEMINI_API_KEY },
    { env: 'gemini_apikey', value: process.env.gemini_apikey },
    { env: 'GEMINI_APIKEY', value: process.env.GEMINI_APIKEY },
    { env: 'GEMINI_KEY', value: process.env.GEMINI_KEY },
  ]
  const hit = candidates.find((c) => typeof c.value === 'string' && c.value.trim().length > 0)
  return hit ? { apiKey: hit.value.trim(), source: hit.env } : { apiKey: undefined, source: undefined }
}

const geminiConfig = getGeminiApiKey()
const genai = geminiConfig.apiKey ? new GoogleGenAI({ apiKey: geminiConfig.apiKey }) : null
const geminiApiKeySource = geminiConfig.source
const faceClient = process.env.GOOGLE_APPLICATION_CREDENTIALS ? new vision.ImageAnnotatorClient() : null
const ollamaBaseUrl = process.env.OLLAMA_BASE_URL ?? 'http://127.0.0.1:11434'
const ollamaModel = process.env.OLLAMA_MODEL ?? 'llama3.2:3b'
const embeddingModel = process.env.GEMINI_EMBEDDING_MODEL ?? 'gemini-embedding-001'
const visionSidecarUrl = process.env.VISION_SIDECAR_URL ?? 'http://127.0.0.1:8008'
const embeddingSidecarUrl = (process.env.EMBEDDING_SIDECAR_URL ?? '').trim()
const inferFlashModel = process.env.VESTIR_INFER_FLASH_MODEL ?? 'gemini-2.5-flash'
const inferProModel = process.env.VESTIR_INFER_PRO_MODEL ?? 'gemini-2.5-pro'
const inferConsistencySamples = Math.min(
  5,
  Math.max(1, Number.parseInt(process.env.INFER_CONSISTENCY_SAMPLES ?? '1', 10) || 1),
)
const tryonSidecarUrl = process.env.TRYON_SIDECAR_URL ?? 'http://127.0.0.1:8009'
const serpApiKey = process.env.SERPAPI_API_KEY ?? ''
/** Public origin SerpApi can reach (e.g. https://xxxx.ngrok-free.app) so /storage/... and localhost URLs can be rewritten */
const publicApiBase = (process.env.PUBLIC_API_BASE_URL ?? '').replace(/\/$/, '')

const CATEGORY_ENUM = ['Tops', 'Bottoms', 'Outerwear', 'Shoes', 'Accessories']
const PATTERN_ENUM = ['solid', 'stripe', 'plaid', 'graphic', 'floral', 'check', 'texture', 'mixed']
const FRAMING_ENUM = ['flat_lay', 'worn', 'detail']
const OCCASION_ENUM = [
  'office',
  'casual',
  'date',
  'gym',
  'beach',
  'outdoor',
  'athletic',
  'lounge',
  'streetwear',
  'travel',
  'party',
  'formal_event',
  'casual_weekend',
]
const STYLE_ENUM = ['minimalist', 'preppy', 'techwear', 'classic', 'sporty', 'street', 'boho', 'smart_casual', 'maximalist']
const LAYERING_ROLE_ENUM = ['base_layer', 'mid_layer', 'outer_layer', 'standalone']
const SEASON_ENUM = ['spring', 'summer', 'autumn', 'winter']
const FIT_ENUM = ['slim', 'regular', 'relaxed', 'oversized', 'unknown']

const hslSchema = z.object({
  h: z.number().min(0).max(360),
  s: z.number().min(0).max(1),
  l: z.number().min(0).max(1),
})

const confidenceSchema = z.number().min(0).max(1)

const qualitySchema = z.object({
  blur_score: z.number().min(0).max(1),
  lighting_score: z.number().min(0).max(1),
  framing: z.enum(FRAMING_ENUM),
  occlusion_visible_pct: z.number().min(0).max(1),
  accepted: z.boolean(),
  warnings: z.array(z.string()),
})

const structuralSchema = z.object({
  garment_type: z.string().min(1),
  subtype: z.string().min(1),
  category: z.enum(CATEGORY_ENUM),
  colors: z.array(z.object({
    name: z.string().min(1),
    hex: z.string().regex(/^#?[0-9a-fA-F]{6}$/),
    coverage_pct: z.number().min(0).max(1),
    confidence: confidenceSchema,
  })).min(1).max(5),
  pattern: z.enum(PATTERN_ENUM),
  construction_details: z.array(z.string()).max(8),
  material: z.object({
    primary: z.string().min(1),
    confidence: confidenceSchema,
  }),
  fit: z.string().optional(),
})

const semanticSchema = z.object({
  formality: z.number().min(1).max(10),
  seasons: z.object({
    spring: z.number().min(0).max(1),
    summer: z.number().min(0).max(1),
    autumn: z.number().min(0).max(1),
    winter: z.number().min(0).max(1),
  }),
  occasions: z.array(z.enum(OCCASION_ENUM)).min(1).max(6),
  style_archetype: z.enum(STYLE_ENUM),
  confidence: z.object({
    formality: confidenceSchema,
    occasions: confidenceSchema,
    style_archetype: confidenceSchema,
    seasonality: confidenceSchema,
  }),
  layering_role: z.enum(LAYERING_ROLE_ENUM).optional(),
  pairings: z.array(z.string().min(1)).min(1).max(3).optional(),
})

const inferSchema = z.object({
  item_type: z.string().min(1),
  subtype: z.string().min(1),
  category: z.enum(CATEGORY_ENUM),
  color_primary: z.string().min(1),
  color_secondary: z.string().optional(),
  color_primary_hsl: hslSchema,
  color_secondary_hsl: hslSchema.optional(),
  color_palette: z.array(z.object({
    name: z.string().min(1),
    hex: z.string().regex(/^#[0-9a-fA-F]{6}$/),
    hsl: hslSchema,
    coverage_pct: z.number().min(0).max(1),
    is_neutral: z.boolean().optional(),
  })).min(1),
  pattern: z.enum(PATTERN_ENUM),
  fit: z.string().optional(),
  material: z.string().min(1),
  material_confidence: confidenceSchema,
  formality: z.number().min(1).max(10),
  season: z.array(z.enum(SEASON_ENUM)).min(1),
  season_weights: z.object({
    spring: z.number().min(0).max(1),
    summer: z.number().min(0).max(1),
    autumn: z.number().min(0).max(1),
    winter: z.number().min(0).max(1),
  }),
  occasions: z.array(z.enum(OCCASION_ENUM)),
  style_archetype: z.enum(STYLE_ENUM),
  layering_role: z.enum(LAYERING_ROLE_ENUM).optional(),
  pairings: z.array(z.string()).optional(),
  confidence_overall: confidenceSchema,
  uncertainty: z.object({
    requires_user_confirmation: z.boolean(),
    uncertain_fields: z.array(z.string()),
  }),
  quality: qualitySchema,
})

const preprocessRequestSchema = z.object({
  imageUrl: z.string().min(1),
  /** Normalized 0–1 vertices; garment interior stays opaque in a PNG for downstream color. */
  maskPolygon: z.array(z.object({ x: z.number(), y: z.number() })).min(3).optional(),
})
const inferRequestSchema = z.object({ processedImageUrl: z.string().min(1) })
const embedRequestSchema = z.object({ item: z.any() })
const reasonRequestSchema = z.object({ item: z.any() })
const extensionMatchSchema = z.object({
  imageUrl: z.string().min(1),
  pageUrl: z.string().optional(),
  title: z.string().optional(),
  topK: z.number().int().min(1).max(20).optional(),
})
const tryonPreviewRequestSchema = z.object({
  personImageUrl: z.string().min(1),
  garmentImageUrl: z.string().min(1),
  seed: z.number().int().optional(),
})

const googleLensExperimentSchema = z.object({
  imageUrl: z.string().min(1),
  type: z.enum(['all', 'about_this_image', 'products', 'exact_matches', 'visual_matches']).optional(),
  q: z.string().optional(),
  hl: z.string().optional(),
  country: z.string().optional(),
  auto_crop: z.boolean().optional(),
  no_cache: z.boolean().optional(),
})

async function ensureStorage() {
  await fs.mkdir(processedDir, { recursive: true })
  try {
    await fs.access(embeddingsFile)
  } catch {
    await fs.writeFile(embeddingsFile, JSON.stringify([]), 'utf8')
  }
}

function parseDataUrl(dataUrl) {
  const match = dataUrl.match(/^data:(.+?);base64,(.+)$/)
  if (!match) throw new Error('Expected base64 data URL image')
  return { mime: match[1], buffer: Buffer.from(match[2], 'base64') }
}

async function resolveImageBuffer(imageUrl) {
  if (imageUrl.startsWith('data:')) {
    return parseDataUrl(imageUrl).buffer
  }
  const fetchUrl = imageUrl.startsWith('http')
    ? imageUrl
    : `http://127.0.0.1:${port}${imageUrl}`
  const response = await fetch(fetchUrl)
  if (!response.ok) throw new Error(`Could not fetch image: ${imageUrl}`)
  return Buffer.from(await response.arrayBuffer())
}

function isAlreadyProcessedUrl(imageUrl) {
  return (
    (imageUrl.startsWith('/storage/processed/') || imageUrl.includes('/storage/processed/')) &&
    !imageUrl.startsWith('data:')
  )
}

/** SerpApi Google Lens only accepts a url= that their servers can GET (not data:, not raw localhost). */
function resolveLensImageUrlForSerpApi(imageUrl) {
  const trimmed = imageUrl.trim()
  if (trimmed.startsWith('data:')) {
    throw new Error(
      'Google Lens via SerpApi needs a public HTTP(S) image URL. data: URLs are not supported. Host the image (e.g. imgur) or set PUBLIC_API_BASE_URL and use /storage/...',
    )
  }
  const localhostOriginRe = /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?/i
  if (trimmed.startsWith('/storage/')) {
    if (!publicApiBase) {
      throw new Error(
        'Relative /storage/... is not reachable from SerpApi. Set PUBLIC_API_BASE_URL to your tunnel (ngrok, cloudflared) pointing at this API.',
      )
    }
    return `${publicApiBase}${trimmed}`
  }
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
    let u
    try {
      u = new URL(trimmed)
    } catch {
      throw new Error('Invalid imageUrl')
    }
    if (localhostOriginRe.test(`${u.protocol}//${u.hostname}${u.port ? `:${u.port}` : ''}`)) {
      if (!publicApiBase) {
        throw new Error(
          'SerpApi cannot fetch localhost. Use a public image URL, or set PUBLIC_API_BASE_URL and pass /storage/processed/... instead of 127.0.0.1.',
        )
      }
      return `${publicApiBase}${u.pathname}${u.search}`
    }
    return trimmed
  }
  throw new Error('imageUrl must be https URL, http public URL, or /storage/... with PUBLIC_API_BASE_URL set')
}

function mean(values) {
  if (!values.length) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function variance(values) {
  if (!values.length) return 0
  const m = mean(values)
  return mean(values.map((value) => (value - m) ** 2))
}

async function validateInputImage(buffer) {
  const meta = await sharp(buffer).metadata()
  const width = meta.width ?? 0
  const height = meta.height ?? 0
  if (width < 224 || height < 224) {
    return { ok: false, reason: 'Photo resolution is too low. Use at least 224x224.' }
  }

  const aspectRatio = width / Math.max(1, height)
  const warnings = []
  if (aspectRatio < 0.28 || aspectRatio > 3.3) warnings.push('Unusual aspect ratio may reduce detection quality.')

  const { data, info } = await sharp(buffer).removeAlpha().resize(256, 256, { fit: 'inside' }).raw().toBuffer({ resolveWithObject: true })
  const channels = info.channels
  const lum = []
  for (let i = 0; i < data.length; i += channels) {
    lum.push(0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2])
  }
  const brightness = mean(lum)
  if (brightness < 16) return { ok: false, reason: 'Photo is too dark. Please retake in better lighting.' }
  if (brightness > 248) return { ok: false, reason: 'Photo is overexposed. Please lower brightness and retake.' }
  if (brightness < 32) warnings.push('Photo is dark; color confidence may be lower.')
  if (brightness > 232) warnings.push('Photo is bright; highlights may affect color precision.')

  // Laplacian-like focus metric (variance of local edge energy)
  const edge = []
  for (let y = 1; y < info.height - 1; y += 1) {
    for (let x = 1; x < info.width - 1; x += 1) {
      const idx = y * info.width + x
      const c = lum[idx]
      const up = lum[idx - info.width]
      const down = lum[idx + info.width]
      const left = lum[idx - 1]
      const right = lum[idx + 1]
      edge.push(Math.abs(4 * c - up - down - left - right))
    }
  }
  const blurVariance = variance(edge)
  if (blurVariance < 40) {
    return { ok: false, reason: 'Photo is too blurry. Hold steady and retake the image.' }
  }
  if (blurVariance < 75) warnings.push('Photo has mild blur; detail confidence may be lower.')

  return {
    ok: true,
    warnings,
    metrics: {
      width,
      height,
      aspect_ratio: Number(aspectRatio.toFixed(3)),
      brightness: Number(brightness.toFixed(2)),
      blur_variance: Number(blurVariance.toFixed(2)),
    },
  }
}

async function detectFaces(buffer) {
  if (!faceClient) return []
  const [result] = await faceClient.faceDetection({ image: { content: buffer } })
  return result.faceAnnotations ?? []
}

async function detectGarmentRegion(buffer) {
  if (!faceClient) return null
  const [result] = await faceClient.objectLocalization({ image: { content: buffer } })
  const objects = result.localizedObjectAnnotations ?? []
  const garmentHints = ['clothing', 'shirt', 'pants', 'trousers', 'jacket', 'coat', 'dress', 'shoe', 'sneaker', 'hoodie', 'skirt']
  const hit = objects
    .filter((obj) => garmentHints.some((hint) => (obj.name ?? '').toLowerCase().includes(hint)))
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0]
  if (!hit?.boundingPoly?.normalizedVertices?.length) return null
  return hit.boundingPoly.normalizedVertices
}

async function detectGarmentRegions(buffer) {
  if (!faceClient) return []
  const [result] = await faceClient.objectLocalization({ image: { content: buffer } })
  const objects = result.localizedObjectAnnotations ?? []
  const garmentHints = ['clothing', 'shirt', 'pants', 'trousers', 'jacket', 'coat', 'dress', 'shoe', 'sneaker', 'hoodie', 'skirt']
  return objects
    .filter((obj) => garmentHints.some((hint) => (obj.name ?? '').toLowerCase().includes(hint)))
    .map((obj) => ({
      name: (obj.name ?? '').toLowerCase(),
      score: obj.score ?? 0,
      vertices: obj.boundingPoly?.normalizedVertices ?? [],
    }))
    .filter((obj) => obj.vertices.length)
    .sort((a, b) => b.score - a.score)
}

async function callVisionSidecarAnalyze(buffer) {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 3500)
    const response = await fetch(`${visionSidecarUrl}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: buffer.toString('base64'),
        stages: ['human_detection', 'human_parsing', 'privacy_masking', 'clothing_extraction'],
      }),
      signal: controller.signal,
    })
    clearTimeout(timeout)
    if (!response.ok) return null
    const data = await response.json()
    return data && typeof data === 'object' ? data : null
  } catch {
    return null
  }
}

function verticesFromNormalizedBbox(bbox) {
  if (!bbox) return []
  const x1 = Math.max(0, Math.min(1, Number(bbox.x1 ?? 0)))
  const y1 = Math.max(0, Math.min(1, Number(bbox.y1 ?? 0)))
  const x2 = Math.max(0, Math.min(1, Number(bbox.x2 ?? 0)))
  const y2 = Math.max(0, Math.min(1, Number(bbox.y2 ?? 0)))
  if (x2 <= x1 || y2 <= y1) return []
  return [
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 },
  ]
}

function normalizedRegionCoverage(vertices) {
  if (!vertices?.length) return 0
  const xs = vertices.map((v) => Math.max(0, Math.min(1, v.x ?? 0)))
  const ys = vertices.map((v) => Math.max(0, Math.min(1, v.y ?? 0)))
  const width = Math.max(0, Math.max(...xs) - Math.min(...xs))
  const height = Math.max(0, Math.max(...ys) - Math.min(...ys))
  return width * height
}

function detectSceneTrack({ faceCount, garmentCoverage }) {
  if (faceCount > 0 && garmentCoverage > 0.22) return 'worn'
  if (faceCount === 0 && garmentCoverage > 0.18) return 'flat_lay'
  return 'ambiguous'
}

function inferFramingFromContext(faces, occlusionVisiblePct) {
  if (faces.length > 0) return 'worn'
  if (occlusionVisiblePct < 0.55) return 'detail'
  return 'flat_lay'
}

function dedupePalette(colors, limit = 3) {
  const merged = []
  for (const color of colors) {
    if (merged.length >= limit) break
    const duplicate = merged.some((existing) => rgbDistance(hexToRgb(existing.hex), hexToRgb(color.hex)) < 20)
    if (!duplicate) merged.push(color)
  }
  return merged
}

/**
 * Prefer "fabric-like" light neutrals (e.g. white t-shirt base) over dark graphic regions.
 * This matters a lot for garments with large prints/logos that can dominate simple dominant-color heuristics.
 */
function prioritizeLightNeutralsPrimary(colors, fallbackToCoverage = true) {
  if (!Array.isArray(colors) || colors.length === 0) return colors
  const candidates = colors.filter((c) => {
    const hsl = c?.hsl
    if (!hsl) return false
    const l = Number(hsl.l ?? 0)
    const s = Number(hsl.s ?? 1)
    return l >= 0.78 && s <= 0.25
  })
  if (!candidates.length) return colors

  // Pick the lightest neutral.
  candidates.sort((a, b) => Number((b.hsl?.l ?? 0) - (a.hsl?.l ?? 0)))
  const primary = candidates[0]

  const rest = colors
    .filter((c) => c !== primary)
    .sort((a, b) => (fallbackToCoverage ? Number((b.coverage_pct ?? 0) - (a.coverage_pct ?? 0)) : 0))

  return [primary, ...rest]
}

function paletteHueSpread(colors) {
  if (!colors?.length) return 0
  const hues = colors.map((c) => c.hsl?.h ?? 0)
  return Math.max(...hues) - Math.min(...hues)
}

function contradictionFlags({ subtype, material, formality, seasonWeights, pattern, colors }) {
  const flags = []
  const subtypeLc = String(subtype ?? '').toLowerCase()
  const materialLc = String(material ?? '').toLowerCase()
  if ((subtypeLc.includes('swim') || subtypeLc.includes('boardshort')) && formality >= 8) {
    flags.push('formality_context_mismatch')
  }
  if ((materialLc.includes('wool') || materialLc.includes('cashmere')) && (seasonWeights?.summer ?? 0) >= 0.85) {
    flags.push('material_season_mismatch')
  }
  const spread = paletteHueSpread(colors)
  const hasWideCoverage = colors.filter((c) => c.coverage_pct >= 0.2).length >= 3
  if (pattern === 'solid' && spread > 95 && hasWideCoverage) {
    flags.push('pattern_palette_mismatch')
  }
  return flags
}

function outerwearSignalsInText(text) {
  const t = String(text ?? '').toLowerCase()
  return /(puffer|down jacket|quilted|duvet|parka|anorak|overcoat|peacoat|trench|bomber|windbreaker|shell jacket|ski jacket|\bjacket\b|\bcoat\b|\bblazer\b|gore-tex|insulated outer|outer shell)/.test(
    t,
  )
}

function coerceCategoryForGarmentLabels(garmentType, subtype, category) {
  const blob = `${garmentType ?? ''} ${subtype ?? ''}`
  if (!outerwearSignalsInText(blob)) return { category, adjusted: false }
  if (category === 'Tops') return { category: 'Outerwear', adjusted: true }
  return { category, adjusted: false }
}

function garmentLabelLooksLikeOuterShirt(garmentType) {
  return /^\s*shirt\s*$/i.test(String(garmentType ?? '').trim())
}

async function detectFilterSignature(buffer) {
  const { data, info } = await sharp(buffer).removeAlpha().resize(128, 128, { fit: 'inside' }).raw().toBuffer({ resolveWithObject: true })
  const channels = info.channels
  let rSum = 0
  let gSum = 0
  let bSum = 0
  const satValues = []
  for (let i = 0; i < data.length; i += channels) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    rSum += r
    gSum += g
    bSum += b
    const hsl = rgbToHsl(r, g, b)
    satValues.push(hsl.s)
  }
  const pixelCount = Math.max(1, satValues.length)
  const rMean = rSum / pixelCount
  const gMean = gSum / pixelCount
  const bMean = bSum / pixelCount
  const satMean = mean(satValues)
  const warmBias = (rMean - bMean) / 255
  const greenBias = (gMean - (rMean + bMean) / 2) / 255
  const sepiaLike = warmBias > 0.16 && greenBias > -0.04
  const coolLike = warmBias < -0.14
  const oversaturated = satMean > 0.58
  const filterRisk = Math.max(
    Math.abs(warmBias) * 1.8,
    Math.abs(greenBias) * 1.6,
    oversaturated ? (satMean - 0.58) * 1.4 : 0,
  )

  return {
    risk: Math.max(0, Math.min(1, filterRisk)),
    warm_bias: Number(warmBias.toFixed(3)),
    saturation_mean: Number(satMean.toFixed(3)),
    likely_filter: sepiaLike || coolLike || oversaturated,
    kind: sepiaLike ? 'sepia_like' : coolLike ? 'cool_tint' : oversaturated ? 'vivid_filter' : 'none',
  }
}

async function normalizeForColorAnalysis(buffer) {
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

function extractBounds(faceAnnotation) {
  const vertices = faceAnnotation.boundingPoly?.vertices ?? []
  if (!vertices.length) return null
  const xs = vertices.map((v) => v.x ?? 0)
  const ys = vertices.map((v) => v.y ?? 0)
  const left = Math.max(0, Math.min(...xs))
  const top = Math.max(0, Math.min(...ys))
  const width = Math.max(1, Math.max(...xs) - left)
  const height = Math.max(1, Math.max(...ys) - top)
  return { left, top, width, height }
}

async function blurFaceRegions(buffer, regions) {
  if (!regions.length) return { output: buffer, faceDetected: false, faceBlurApplied: false }
  const base = sharp(buffer)
  const meta = await base.metadata()
  const composites = []
  for (const region of regions) {
    const left = Math.max(0, Math.floor(region.left))
    const top = Math.max(0, Math.floor(region.top))
    const width = Math.min(Math.floor(region.width), (meta.width ?? 0) - left)
    const height = Math.min(Math.floor(region.height), (meta.height ?? 0) - top)
    if (width <= 0 || height <= 0) continue
    const patch = await sharp(buffer).extract({ left, top, width, height }).blur(30).toBuffer()
    composites.push({ input: patch, left, top })
  }
  if (!composites.length) return { output: buffer, faceDetected: true, faceBlurApplied: false }
  const output = await sharp(buffer).composite(composites).jpeg({ quality: 82 }).toBuffer()
  return { output, faceDetected: true, faceBlurApplied: true }
}

async function cropToRegion(buffer, normalizedVertices) {
  if (!normalizedVertices?.length) return buffer
  const image = sharp(buffer)
  const meta = await image.metadata()
  const width = meta.width ?? 0
  const height = meta.height ?? 0
  if (!width || !height) return buffer
  const xs = normalizedVertices.map((v) => Math.max(0, Math.min(1, v.x ?? 0)) * width)
  const ys = normalizedVertices.map((v) => Math.max(0, Math.min(1, v.y ?? 0)) * height)
  let left = Math.floor(Math.min(...xs))
  let top = Math.floor(Math.min(...ys))
  let right = Math.ceil(Math.max(...xs))
  let bottom = Math.ceil(Math.max(...ys))
  if (right <= left || bottom <= top) return buffer

  // add padding so seams/details are preserved
  const padX = Math.floor((right - left) * 0.12)
  const padY = Math.floor((bottom - top) * 0.12)
  left = Math.max(0, left - padX)
  top = Math.max(0, top - padY)
  right = Math.min(width, right + padX)
  bottom = Math.min(height, bottom + padY)
  const cropW = Math.max(1, right - left)
  const cropH = Math.max(1, bottom - top)

  return sharp(buffer).extract({ left, top, width: cropW, height: cropH }).jpeg({ quality: 90 }).toBuffer()
}

function hslFromColorName(name) {
  const map = {
    navy: { h: 220, s: 0.45, l: 0.32 },
    olive: { h: 95, s: 0.34, l: 0.44 },
    cream: { h: 50, s: 0.28, l: 0.9 },
    taupe: { h: 25, s: 0.2, l: 0.52 },
    black: { h: 0, s: 0.02, l: 0.08 },
    white: { h: 0, s: 0.03, l: 0.96 },
    gray: { h: 0, s: 0.01, l: 0.5 },
  }
  return map[name.toLowerCase()] ?? { h: 32, s: 0.2, l: 0.45 }
}

function rgbToHsl(r, g, b) {
  const rn = r / 255
  const gn = g / 255
  const bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  let h = 0
  const l = (max + min) / 2
  const d = max - min
  const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1))

  if (d !== 0) {
    switch (max) {
      case rn:
        h = ((gn - bn) / d) % 6
        break
      case gn:
        h = (bn - rn) / d + 2
        break
      default:
        h = (rn - gn) / d + 4
    }
    h *= 60
    if (h < 0) h += 360
  }

  return { h, s, l }
}

function colorNameFromHsl(hsl) {
  const { h, s, l } = hsl
  if (l < 0.1) return 'Black'
  if (l > 0.94 && s < 0.08) return 'White'
  if (s < 0.1 && l < 0.35) return 'Charcoal'
  if (s < 0.1) return 'Gray'
  if (h >= 200 && h < 230 && l < 0.4) return 'Navy'
  if (h >= 200 && h < 235) return 'Slate'
  if (h >= 170 && h < 200) return 'Teal'
  // Olive drab often appears around yellow-green hues with medium/low saturation.
  if (h >= 50 && h < 155 && s >= 0.16 && l <= 0.64) return 'Olive'
  if (h >= 70 && h < 155 && s < 0.16 && l >= 0.42) return 'Khaki'
  if (h >= 55 && h < 90 && s < 0.2 && l > 0.62) return 'Khaki'
  if (h >= 45 && h < 80 && s < 0.25 && l > 0.75) return 'Ecru'
  if (h >= 35 && h < 65 && s < 0.35) return 'Camel'
  if (h >= 35 && h < 55 && s >= 0.35) return 'Mustard'
  if (h >= 18 && h < 40 && s < 0.3) return 'Tan'
  if ((h >= 0 && h < 18) || h >= 342) return 'Rust'
  if (h >= 310 && h < 342) return 'Burgundy'
  if (h >= 235 && h < 290) return 'Cobalt'
  return 'Taupe'
}

function normalizeHex(hex) {
  const trimmed = hex.startsWith('#') ? hex : `#${hex}`
  return trimmed.length === 7 ? trimmed.toUpperCase() : '#8B7355'
}

function hexToRgb(hex) {
  const normalized = normalizeHex(hex).slice(1)
  return {
    r: Number.parseInt(normalized.slice(0, 2), 16),
    g: Number.parseInt(normalized.slice(2, 4), 16),
    b: Number.parseInt(normalized.slice(4, 6), 16),
  }
}

function rgbToHex(r, g, b) {
  const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)))
  return `#${clamp(r).toString(16).padStart(2, '0')}${clamp(g).toString(16).padStart(2, '0')}${clamp(b).toString(16).padStart(2, '0')}`.toUpperCase()
}

function rgbDistance(a, b) {
  const dr = a.r - b.r
  const dg = a.g - b.g
  const db = a.b - b.b
  return Math.sqrt(dr * dr + dg * dg + db * db)
}

/** Aligns API responses with the documented multi-stage vision architecture. */
const PIPELINE_ARCHITECTURE_ID = 'multi-stage-v1'

function rgbToLab(rgbR, rgbG, rgbB) {
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
  const fx = x > 0.008856 ? Math.cbrt(x) : (7.787 * x) + 16 / 116
  const fy = y > 0.008856 ? Math.cbrt(y) : (7.787 * y) + 16 / 116
  const fz = z > 0.008856 ? Math.cbrt(z) : (7.787 * z) + 16 / 116
  return {
    L: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz),
  }
}

function labDistance(p, q) {
  return Math.sqrt((p.L - q.L) ** 2 + (p.a - q.a) ** 2 + (p.b - q.b) ** 2)
}

/**
 * Dominant colors via K-means in CIELAB (lightness separated from chroma — more stable under lighting than raw RGB).
 */
async function extractLabKMeansPalette(imageBuffer, k = 4) {
  try {
    const meta = await sharp(imageBuffer).metadata()
    const hasAlpha = Boolean(meta.hasAlpha)
    const { data, info } = await sharp(imageBuffer)
      .ensureAlpha()
      .resize(112, 112, { fit: 'inside' })
      .raw()
      .toBuffer({ resolveWithObject: true })
    const channels = info.channels
    const width = info.width
    const height = info.height
    const pixels = []
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        const idx = (y * width + x) * channels
        if (hasAlpha && channels >= 4 && data[idx + 3] < 90) continue
        const r = data[idx]
        const g = data[idx + 1]
        const bl = data[idx + 2]
        pixels.push({ r, g, b: bl, lab: rgbToLab(r, g, bl) })
      }
    }
    const kk = Math.min(k, Math.max(2, Math.floor(pixels.length / 32)))
    if (pixels.length < kk * 4) return []

    const centroids = []
    const picked = new Set()
    for (let c = 0; c < kk; c += 1) {
      let i = Math.floor(Math.random() * pixels.length)
      let guard = 0
      while (picked.has(i) && guard < 50) {
        i = Math.floor(Math.random() * pixels.length)
        guard += 1
      }
      picked.add(i)
      centroids.push({ L: pixels[i].lab.L, a: pixels[i].lab.a, b: pixels[i].lab.b })
    }

    const assignments = new Array(pixels.length).fill(0)
    for (let iter = 0; iter < 14; iter += 1) {
      for (let i = 0; i < pixels.length; i += 1) {
        let best = 0
        let bestD = Infinity
        for (let c = 0; c < centroids.length; c += 1) {
          const d = labDistance(pixels[i].lab, centroids[c])
          if (d < bestD) {
            bestD = d
            best = c
          }
        }
        assignments[i] = best
      }
      const next = centroids.map(() => ({ L: 0, a: 0, b: 0, n: 0 }))
      for (let i = 0; i < pixels.length; i += 1) {
        const c = assignments[i]
        const lab = pixels[i].lab
        next[c].L += lab.L
        next[c].a += lab.a
        next[c].b += lab.b
        next[c].n += 1
      }
      let shifted = false
      for (let c = 0; c < centroids.length; c += 1) {
        if (next[c].n === 0) continue
        const n = next[c].n
        const L = next[c].L / n
        const a = next[c].a / n
        const b = next[c].b / n
        if (labDistance({ L, a, b }, centroids[c]) > 0.4) shifted = true
        centroids[c] = { L, a, b }
      }
      if (!shifted && iter > 3) break
    }

    const rgbBuckets = centroids.map(() => ({ r: 0, g: 0, b: 0, n: 0 }))
    for (let i = 0; i < pixels.length; i += 1) {
      const c = assignments[i]
      rgbBuckets[c].r += pixels[i].r
      rgbBuckets[c].g += pixels[i].g
      rgbBuckets[c].b += pixels[i].b
      rgbBuckets[c].n += 1
    }
    const total = pixels.length
    return rgbBuckets
      .filter((bucket) => bucket.n > 0)
      .map((bucket) => {
        const r = bucket.r / bucket.n
        const g = bucket.g / bucket.n
        const b = bucket.b / bucket.n
        const hsl = rgbToHsl(r, g, b)
        return {
          name: colorNameFromHsl(hsl),
          hex: rgbToHex(r, g, b),
          hsl,
          coverage_pct: Number((bucket.n / Math.max(1, total)).toFixed(3)),
          confidence: 0.88,
          is_neutral: isNeutralHsl(hsl),
        }
      })
      .sort((a, b) => b.coverage_pct - a.coverage_pct)
      .slice(0, 3)
  } catch {
    return []
  }
}

function isNeutralHsl(hsl) {
  return hsl.s < 0.12 || hsl.l < 0.14 || hsl.l > 0.9
}

function confidenceBand(score) {
  if (score >= 0.85) return 'accepted'
  if (score >= 0.6) return 'uncertain'
  return 'needs_confirmation'
}

async function scoreImageQuality(buffer) {
  const resized = await sharp(buffer).removeAlpha().resize(256, 256, { fit: 'inside' }).raw().toBuffer({ resolveWithObject: true })
  const { data, info } = resized
  const channels = info.channels
  let luminanceSum = 0
  let luminanceSqSum = 0
  let edgeAccum = 0
  let pixelCount = 0

  for (let y = 0; y < info.height; y += 1) {
    for (let x = 0; x < info.width; x += 1) {
      const idx = (y * info.width + x) * channels
      const r = data[idx]
      const g = data[idx + 1]
      const b = data[idx + 2]
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
      luminanceSum += lum
      luminanceSqSum += lum * lum
      pixelCount += 1
      if (x > 0 && y > 0) {
        const leftIdx = (y * info.width + (x - 1)) * channels
        const topIdx = ((y - 1) * info.width + x) * channels
        const leftLum = 0.2126 * data[leftIdx] + 0.7152 * data[leftIdx + 1] + 0.0722 * data[leftIdx + 2]
        const topLum = 0.2126 * data[topIdx] + 0.7152 * data[topIdx + 1] + 0.0722 * data[topIdx + 2]
        edgeAccum += Math.abs(lum - leftLum) + Math.abs(lum - topLum)
      }
    }
  }

  const meanLum = luminanceSum / Math.max(1, pixelCount)
  const variance = luminanceSqSum / Math.max(1, pixelCount) - meanLum * meanLum
  const stdDev = Math.sqrt(Math.max(0, variance))
  const blurScore = Math.max(0, Math.min(1, edgeAccum / (pixelCount * 28)))
  const lightingCenter = 128
  const lightingScore = Math.max(0, Math.min(1, 1 - Math.abs(meanLum - lightingCenter) / lightingCenter))
  const occlusionVisiblePct = Math.max(0.2, Math.min(1, stdDev / 64))

  const warnings = []
  if (blurScore < 0.4) warnings.push('Image appears blurry; attribute confidence may drop.')
  if (lightingScore < 0.45) warnings.push('Lighting is suboptimal; color precision may be reduced.')
  if (occlusionVisiblePct < 0.5) warnings.push('Only a partial garment is visible.')
  const accepted = blurScore >= 0.25 && lightingScore >= 0.3

  return {
    blur_score: blurScore,
    lighting_score: lightingScore,
    framing: 'flat_lay',
    occlusion_visible_pct: occlusionVisiblePct,
    accepted,
    warnings,
  }
}

async function dominantColorFromImage(imageBuffer) {
  try {
    const { data, info } = await sharp(imageBuffer)
      .removeAlpha()
      .resize(64, 64, { fit: 'inside' })
      .raw()
      .toBuffer({ resolveWithObject: true })
    let r = 0
    let g = 0
    let b = 0
    let count = 0
    for (let i = 0; i < data.length; i += info.channels) {
      r += data[i]
      g += data[i + 1]
      b += data[i + 2]
      count += 1
    }
    const avgR = Math.round(r / Math.max(1, count))
    const avgG = Math.round(g / Math.max(1, count))
    const avgB = Math.round(b / Math.max(1, count))
    const hsl = rgbToHsl(avgR, avgG, avgB)
    return { name: colorNameFromHsl(hsl), hsl }
  } catch {
    return { name: 'Taupe', hsl: hslFromColorName('Taupe') }
  }
}

async function extractColorPaletteFromImage(imageBuffer) {
  try {
    const { data, info } = await sharp(imageBuffer)
      .removeAlpha()
      .resize(96, 96, { fit: 'inside' })
      .raw()
      .toBuffer({ resolveWithObject: true })

    const channels = info.channels
    const width = info.width
    const height = info.height
    const bins = new Map()
    let totalWeight = 0
    let borderR = 0
    let borderG = 0
    let borderB = 0
    let borderCount = 0

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (x !== 0 && y !== 0 && x !== width - 1 && y !== height - 1) continue
        const idx = (y * width + x) * channels
        borderR += data[idx]
        borderG += data[idx + 1]
        borderB += data[idx + 2]
        borderCount += 1
      }
    }
    const borderColor = {
      r: borderR / Math.max(1, borderCount),
      g: borderG / Math.max(1, borderCount),
      b: borderB / Math.max(1, borderCount),
    }

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = (y * width + x) * channels
        const r = data[idx]
        const g = data[idx + 1]
        const b = data[idx + 2]
        const hsl = rgbToHsl(r, g, b)

        const dx = (x - width / 2) / (width / 2)
        const dy = (y - height / 2) / (height / 2)
        const centerWeight = Math.max(0.35, 1 - Math.sqrt(dx * dx + dy * dy))
        const borderSimilarity = rgbDistance({ r, g, b }, borderColor)
        const likelyBackground = borderSimilarity < 22 && hsl.s < 0.18
        if (likelyBackground) continue

        const qr = Math.round(r / 24) * 24
        const qg = Math.round(g / 24) * 24
        const qb = Math.round(b / 24) * 24
        const key = `${qr}-${qg}-${qb}`
        const prev = bins.get(key) ?? { r: qr, g: qg, b: qb, weight: 0 }
        prev.weight += centerWeight
        bins.set(key, prev)
        totalWeight += centerWeight
      }
    }

    const top = [...bins.values()]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 3)
      .map((entry) => {
        const hsl = rgbToHsl(entry.r, entry.g, entry.b)
        return {
          name: colorNameFromHsl(hsl),
          hex: rgbToHex(entry.r, entry.g, entry.b),
          hsl,
          coverage_pct: Number((entry.weight / Math.max(1, totalWeight)).toFixed(3)),
          confidence: 0.9,
          is_neutral: isNeutralHsl(hsl),
        }
      })

    if (!top.length) {
      const dominant = await dominantColorFromImage(imageBuffer)
      return [{
        name: dominant.name,
        hex: '#8B7355',
        hsl: dominant.hsl,
        coverage_pct: 1,
        confidence: 0.65,
        is_neutral: isNeutralHsl(dominant.hsl),
      }]
    }
    return top
  } catch {
    const dominant = await dominantColorFromImage(imageBuffer)
    return [{
      name: dominant.name,
      hex: '#8B7355',
      hsl: dominant.hsl,
      coverage_pct: 1,
      confidence: 0.55,
      is_neutral: isNeutralHsl(dominant.hsl),
    }]
  }
}

async function extractVisionColorPalette(imageBuffer) {
  if (!faceClient) return []
  try {
    const [result] = await faceClient.imageProperties({ image: { content: imageBuffer } })
    const colors = result.imagePropertiesAnnotation?.dominantColors?.colors ?? []
    const top = colors.slice(0, 3)
    const sum = top.reduce((acc, c) => acc + (c.pixelFraction ?? 0), 0) || 1
    return top.map((entry) => {
      const r = entry.color?.red ?? 0
      const g = entry.color?.green ?? 0
      const b = entry.color?.blue ?? 0
      const hsl = rgbToHsl(r, g, b)
      return {
        name: colorNameFromHsl(hsl),
        hex: rgbToHex(r, g, b),
        hsl,
        coverage_pct: Number(((entry.pixelFraction ?? 0) / sum).toFixed(3)),
        confidence: Math.max(0.6, Math.min(0.98, entry.score ?? 0.8)),
        is_neutral: isNeutralHsl(hsl),
      }
    })
  } catch {
    return []
  }
}

function extractJsonObject(text) {
  const trimmed = text.trim()
  if (!trimmed) throw new Error('Gemini returned empty response')

  // Handle fenced markdown responses: ```json ... ```
  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i)
  if (fencedMatch?.[1]) return fencedMatch[1].trim()

  // Fallback to first JSON object in plain text.
  const firstBrace = trimmed.indexOf('{')
  const lastBrace = trimmed.lastIndexOf('}')
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error(`Gemini did not return JSON. Raw: ${trimmed.slice(0, 280)}`)
  }
  return trimmed.slice(firstBrace, lastBrace + 1)
}

async function runGeminiJson(model, schema, prompt, base64Image, options = {}) {
  const {
    mimeType = 'image/jpeg',
    temperature = 0.35,
    responseMimeType = 'application/json',
  } = options
  const result = await genai.models.generateContent({
    model,
    config: { responseMimeType, temperature },
    contents: [
      { text: prompt },
      {
        inlineData: {
          mimeType,
          data: base64Image,
        },
      },
    ],
  })
  const text = result.text ?? ''
  const payload = extractJsonObject(text)
  return schema.parse(JSON.parse(payload))
}

function majorityPick(samples, keyFn) {
  const counts = new Map()
  for (const s of samples) {
    const k = keyFn(s)
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  let best = null
  let bestN = 0
  for (const [k, n] of counts) {
    if (n > bestN) {
      bestN = n
      best = k
    }
  }
  return { value: best, count: bestN, total: samples.length }
}

function pickStructuralByCategoryVote(samples) {
  const { value: cat } = majorityPick(samples, (s) => s.category)
  const match = samples.filter((s) => s.category === cat)
  return match[0] ?? samples[0]
}

function inferNeedsProTiebreaker(samples) {
  if (samples.length < 2) return false
  const need = Math.ceil(samples.length / 2)
  const { count: catCount } = majorityPick(samples, (s) => s.category)
  const { count: patCount } = majorityPick(samples, (s) => s.pattern)
  return catCount < need || patCount < need
}

function buildDeterministicFallbackVector(payload, dims = 256) {
  const seed = payload.split('').reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) % 2147483647, 7)
  const vector = []
  let x = seed
  for (let i = 0; i < dims; i += 1) {
    x = (x * 48271) % 2147483647
    vector.push((x / 2147483647) * 2 - 1)
  }
  return vector
}

function buildEmbeddingPayloadFromItem(item) {
  let rawAttributes = {}
  if (typeof item?.raw_attributes === 'string') {
    try {
      rawAttributes = JSON.parse(item.raw_attributes)
    } catch {
      rawAttributes = {}
    }
  }
  const occasions = Array.isArray(rawAttributes.occasions) ? rawAttributes.occasions.join(', ') : ''
  const style = rawAttributes.style_archetype ? `style ${rawAttributes.style_archetype}` : ''
  const layeringRole = rawAttributes.layering_role ? `layers as ${rawAttributes.layering_role}` : ''
  const pairings = Array.isArray(rawAttributes.pairings) ? rawAttributes.pairings.slice(0, 3).join(', ') : ''
  const pairingHint = item.category === 'Bottoms'
    ? 'pairs with: white shirt, loafers, blazer'
    : item.category === 'Tops'
      ? 'pairs with: navy chinos, clean sneakers'
      : 'pairs with complementary neutrals'
  return [
    `${item.color_primary} ${item.item_type}`,
    rawAttributes.pattern ? `pattern ${rawAttributes.pattern}` : '',
    rawAttributes.fit ? `fit ${rawAttributes.fit}` : '',
    item.material,
    `formality ${item.formality}/10`,
    `seasons ${Array.isArray(item.season) ? item.season.join('/') : 'all'}`,
    occasions ? `occasions ${occasions}` : '',
    style,
    layeringRole,
    pairings ? `pairings ${pairings}` : '',
    pairingHint,
  ].filter(Boolean).join(', ')
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length || a.length !== b.length) return 0
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i += 1) {
    const av = Number(a[i] ?? 0)
    const bv = Number(b[i] ?? 0)
    dot += av * bv
    na += av * av
    nb += bv * bv
  }
  if (na === 0 || nb === 0) return 0
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

function normText(value) {
  return String(value ?? '').trim().toLowerCase()
}

function colorFamily(value) {
  const c = normText(value)
  if (!c) return ''
  if (c.includes('blue') || c.includes('navy') || c.includes('cobalt')) return 'blue'
  if (c.includes('olive') || c.includes('green') || c.includes('khaki')) return 'green'
  if (c.includes('white') || c.includes('cream') || c.includes('ecru')) return 'light-neutral'
  if (c.includes('black') || c.includes('charcoal')) return 'dark-neutral'
  if (c.includes('gray') || c.includes('grey')) return 'mid-neutral'
  if (c.includes('brown') || c.includes('tan') || c.includes('taupe') || c.includes('camel')) return 'brown'
  if (c.includes('red') || c.includes('burgundy') || c.includes('rust')) return 'red'
  return c
}

function inferCategoryFromLabels(labels) {
  const text = labels.join(' ').toLowerCase()
  if (text.includes('shoe') || text.includes('sneaker') || text.includes('boot')) return 'Shoes'
  if (text.includes('jacket') || text.includes('coat') || text.includes('outerwear')) return 'Outerwear'
  if (text.includes('pant') || text.includes('jean') || text.includes('trouser') || text.includes('short')) return 'Bottoms'
  if (text.includes('belt') || text.includes('bag') || text.includes('scarf') || text.includes('hat')) return 'Accessories'
  return 'Tops'
}

function inferTypeFromCategory(category) {
  const byCategory = {
    Tops: 'Shirt',
    Bottoms: 'Trousers',
    Outerwear: 'Jacket',
    Shoes: 'Sneakers',
    Accessories: 'Accessory',
  }
  return byCategory[category] ?? 'Garment'
}

async function fallbackInferFromVision(imageBuffer) {
  const started = Date.now()
  let labels = []
  if (faceClient) {
    const [labelRes] = await faceClient.labelDetection({ image: { content: imageBuffer } })
    labels = (labelRes.labelAnnotations ?? []).map((l) => (l.description ?? '').trim()).filter(Boolean)
  }
  const category = inferCategoryFromLabels(labels)
  // Dominant-color heuristics tend to get hijacked by large dark prints/logos.
  // Prefer light neutrals (white/cream/very light gray) when present.
  const paletteCandidates = await extractColorPaletteFromImage(imageBuffer)
  const ordered = prioritizeLightNeutralsPrimary(paletteCandidates)
  const primary = ordered?.[0]
  const color_primary = primary?.name ?? (await dominantColorFromImage(imageBuffer)).name
  const quality = await scoreImageQuality(imageBuffer)
  const faces = await detectFaces(imageBuffer)
  quality.framing = inferFramingFromContext(faces, quality.occlusion_visible_pct)
  const primary_hsl =
    primary?.hsl ??
    // Keep fallback behavior stable if palette extraction fails.
    (await dominantColorFromImage(imageBuffer)).hsl

  const color_palette = ordered?.length
    ? ordered.slice(0, 3).map((c) => ({
        name: c.name,
        hex: c.hex,
        hsl: c.hsl,
        coverage_pct: c.coverage_pct,
        is_neutral: c.is_neutral,
      }))
    : [
        {
          name: color_primary,
          hex: '#8B7355',
          hsl: primary_hsl,
          coverage_pct: 1,
          is_neutral: isNeutralHsl(primary_hsl),
        },
      ]
  return {
    item_type: inferTypeFromCategory(category),
    subtype: inferTypeFromCategory(category),
    category,
    color_primary,
    color_secondary: undefined,
    color_primary_hsl: primary_hsl,
    color_palette,
    pattern: 'solid',
    material: 'Cotton',
    material_confidence: 0.55,
    formality: 5,
    season: ['spring', 'autumn'],
    season_weights: { spring: 0.7, summer: 0.4, autumn: 0.8, winter: 0.5 },
    occasions: ['casual_weekend'],
    style_archetype: 'classic',
    confidence_overall: 0.58,
    uncertainty: {
      requires_user_confirmation: true,
      uncertain_fields: ['material', 'occasion', 'style_archetype'],
    },
    quality,
    metadata: {
      provider: faceClient ? 'google-vision-fallback' : 'server-fallback',
      model: faceClient ? 'labelDetection' : 'heuristic',
      latency_ms: Date.now() - started,
      version: '1.1.0',
      pipeline: {
        architecture_id: PIPELINE_ARCHITECTURE_ID,
        stages: [
          { id: 'attribute_inference', status: 'partial', detail: 'Gemini unavailable — label/heuristic fallback' },
          { id: 'post_processing', status: 'completed', detail: 'quality + uncertainty' },
        ],
      },
    },
  }
}

const GARMENT_LABELS = [
  'clothing', 'shirt', 'pants', 'trousers', 'jeans', 'jacket', 'coat',
  'dress', 'shoe', 'sneaker', 'boot', 'hoodie', 'skirt', 'shorts',
  'blazer', 'sweater', 'top', 'blouse', 'suit', 'vest', 'cardigan',
  // DeepFashion2 class tokens that appear in `df2.names`
  'outwear', 'sling',
  'bag', 'handbag', 'backpack', 'belt', 'hat', 'cap', 'scarf', 'watch',
]

function normalizedBbox(vertices) {
  const xs = vertices.map((v) => Math.max(0, Math.min(1, v.x ?? 0)))
  const ys = vertices.map((v) => Math.max(0, Math.min(1, v.y ?? 0)))
  return {
    x1: Math.min(...xs), y1: Math.min(...ys),
    x2: Math.max(...xs), y2: Math.max(...ys),
  }
}

function bboxOverlap(a, b) {
  const ix1 = Math.max(a.x1, b.x1)
  const iy1 = Math.max(a.y1, b.y1)
  const ix2 = Math.min(a.x2, b.x2)
  const iy2 = Math.min(a.y2, b.y2)
  if (ix2 <= ix1 || iy2 <= iy1) return 0
  const intersection = (ix2 - ix1) * (iy2 - iy1)
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1)
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1)
  const union = aArea + bArea - intersection
  return union > 0 ? intersection / union : 0
}

function salienceScore({ coverage, centrality, contrast }) {
  return coverage * 0.5 + centrality * 0.3 + contrast * 0.2
}

function faceCoverageRatio(faces, width, height) {
  if (!faces?.length || !width || !height) return 0
  const frameArea = width * height
  const area = faces
    .map(extractBounds)
    .filter(Boolean)
    .reduce((sum, box) => sum + box.width * box.height, 0)
  return Math.max(0, Math.min(1, area / frameArea))
}

async function cropAndIsolateGarment(buffer, vertices, meta) {
  const bbox = normalizedBbox(vertices)
  const w = meta.width ?? 1
  const h = meta.height ?? 1
  const padX = (bbox.x2 - bbox.x1) * 0.15
  const padY = (bbox.y2 - bbox.y1) * 0.15
  const left = Math.max(0, Math.floor((bbox.x1 - padX) * w))
  const top = Math.max(0, Math.floor((bbox.y1 - padY) * h))
  const right = Math.min(w, Math.ceil((bbox.x2 + padX) * w))
  const bottom = Math.min(h, Math.ceil((bbox.y2 + padY) * h))
  const cropW = Math.max(1, right - left)
  const cropH = Math.max(1, bottom - top)
  return sharp(buffer)
    .extract({ left, top, width: cropW, height: cropH })
    .jpeg({ quality: 90 })
    .toBuffer()
}

async function contrastScore(buffer) {
  const { data, info } = await sharp(buffer).removeAlpha().resize(64, 64, { fit: 'inside' }).raw().toBuffer({ resolveWithObject: true })
  const lums = []
  for (let i = 0; i < data.length; i += info.channels) {
    lums.push(0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2])
  }
  return Math.min(1, Math.sqrt(variance(lums)) / 80)
}

app.post('/api/items/detect', async (req, res) => {
  try {
    const { imageUrl } = preprocessRequestSchema.parse(req.body)
    const started = Date.now()
    const buffer = await resolveImageBuffer(imageUrl)

    const validation = await validateInputImage(buffer)
    if (!validation.ok) {
      return res.status(422).json({ error: validation.reason, code: 'INPUT_VALIDATION_FAILED' })
    }

    const meta = await sharp(buffer).metadata()
    const w = meta.width ?? 1
    const h = meta.height ?? 1

    const rawFaces = await detectFaces(buffer)
    const selfieMode = faceCoverageRatio(rawFaces, w, h) >= 0.25
    const faceRegions = rawFaces.map(extractBounds).filter(Boolean)

    const sidecar = await callVisionSidecarAnalyze(buffer)

    // Localize objects on the raw frame first; privacy masking is applied only when writing crops (detection quality drops if faces are blurred first).
    let objectResult = []
    if (Array.isArray(sidecar?.garments) && sidecar.garments.length > 0) {
      objectResult = sidecar.garments.map((g) => ({
        name: g.label ?? 'garment',
        score: Number(g.confidence ?? 0.6),
        boundingPoly: { normalizedVertices: verticesFromNormalizedBbox(g.bbox) },
      }))
    } else if (faceClient) {
      const [r] = await faceClient.objectLocalization({ image: { content: buffer } })
      objectResult = r.localizedObjectAnnotations ?? []
    }

    const privacyBuffer =
      selfieMode && faceRegions.length > 0 ? (await blurFaceRegions(buffer, faceRegions)).output : buffer

    const personLike = objectResult.filter((obj) => {
      const nameLc = (obj.name ?? '').toLowerCase()
      return /\bperson\b|people|^man$|^woman$|^boy$|^girl$|human/.test(nameLc)
    })
    const estimatedPersonCount = Math.max(
      Number(sidecar?.person_count ?? 0),
      personLike.length,
      rawFaces.length,
    )
    const multiPerson = estimatedPersonCount > 1

    const cx = 0.5
    const cy = 0.5

    const garmentObjects = objectResult
      .filter((obj) => {
        const nameLc = (obj.name ?? '').toLowerCase()
        return GARMENT_LABELS.some((hint) => nameLc.includes(hint))
      })
      .filter((obj) => (obj.score ?? 0) >= 0.45)

    // Deduplicate heavily overlapping boxes — keep higher-confidence one.
    const deduped = []
    for (const obj of garmentObjects) {
      const vertices = obj.boundingPoly?.normalizedVertices ?? []
      if (!vertices.length) continue
      const bbox = normalizedBbox(vertices)
      const overlapping = deduped.findIndex((d) => bboxOverlap(d.bbox, bbox) > 0.6)
      if (overlapping === -1) {
        deduped.push({ obj, bbox, vertices })
      } else if ((obj.score ?? 0) > (deduped[overlapping].obj.score ?? 0)) {
        deduped[overlapping] = { obj, bbox, vertices }
      }
    }

    // Score salience for each detected item.
    const scored = await Promise.all(
      deduped.map(async ({ obj, bbox, vertices }) => {
        const coverage = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        const itemCx = (bbox.x1 + bbox.x2) / 2
        const itemCy = (bbox.y1 + bbox.y2) / 2
        const dist = Math.sqrt((itemCx - cx) ** 2 + (itemCy - cy) ** 2)
        const centrality = Math.max(0, 1 - dist / 0.71)
        const cropBuffer = await cropAndIsolateGarment(privacyBuffer, vertices, meta)
        const contrast = await contrastScore(cropBuffer)
        const lowerFrameBias = selfieMode ? Math.max(0.75, itemCy) : 1
        const topwearBoost = /outerwear|outwear|jacket|coat|blazer|shirt|hoodie|sweater|vest|sling|top/i.test(
          obj.name ?? '',
        )
          ? 1.08
          : 1
        const accessoryPenalty = /watch|hat|cap|beanie|handbag|bag|belt/i.test(obj.name ?? '') ? 0.85 : 1
        const salience = salienceScore({ coverage, centrality, contrast }) * lowerFrameBias * topwearBoost * accessoryPenalty
        const partiallyVisible = coverage < 0.06

        const filename = `${crypto.randomUUID()}.jpg`
        await fs.writeFile(path.join(processedDir, filename), cropBuffer)

        return {
          obj,
          bbox,
          vertices,
          coverage,
          centrality,
          contrast,
          salience,
          partiallyVisible,
          cropUrl: `/storage/processed/${filename}`,
        }
      }),
    )

    const sorted = scored.sort((a, b) => b.salience - a.salience)
    const heroIdx = 0

    // Fallback: if Vision found nothing, return the whole image as one item.
    const fallback = sorted.length === 0
    let items = []
    if (fallback) {
      const name = `${crypto.randomUUID()}.jpg`
      await fs.writeFile(path.join(processedDir, name), privacyBuffer)
      items = [{
        id: crypto.randomUUID(),
        label: 'Garment',
        confidence: 0.5,
        crop_url: `/storage/processed/${name}`,
        coverage: 1,
        centrality: 1,
        salience: 0.5,
        is_hero: true,
        partially_visible: false,
        warning: 'Single-item fallback: garment detection unavailable.',
      }]
    } else {
      items = sorted.map((entry, idx) => ({
          id: crypto.randomUUID(),
          label: entry.obj.name ?? 'Garment',
          confidence: Number((entry.obj.score ?? 0.5).toFixed(3)),
          crop_url: entry.cropUrl,
          coverage: Number(entry.coverage.toFixed(3)),
          centrality: Number(entry.centrality.toFixed(3)),
          salience: Number(entry.salience.toFixed(3)),
          is_hero: idx === heroIdx,
          partially_visible: entry.partiallyVisible,
          warning: entry.partiallyVisible ? 'Only partial garment visible — full analysis limited.' : undefined,
        }))
    }

    const faceCount = rawFaces.length
    const maxCoverage = sorted[0]?.coverage ?? 0
    const track = faceCount > 0 && maxCoverage > 0.18
      ? 'worn'
      : faceCount === 0 && maxCoverage > 0.14
        ? 'flat_lay'
        : 'ambiguous'

    const sourceFilename = `${crypto.randomUUID()}.jpg`
    await fs.writeFile(path.join(processedDir, sourceFilename), privacyBuffer)

    const privacyApplied = privacyBuffer !== buffer
    const warnings = [
      ...(selfieMode ? ['Selfie-mode activated: face-dominant frame, garment salience re-weighted.'] : []),
      ...(multiPerson ? ['Multiple people in frame — crops are per detected garment; verify each selection.'] : []),
    ]

    res.json({
      detected: items,
      scene_track: track,
      source_image_url: `/storage/processed/${sourceFilename}`,
      warnings,
      pipeline: {
        architecture_id: PIPELINE_ARCHITECTURE_ID,
        stages: [
          { id: 'image_filtering', status: 'completed', detail: 'validateInputImage' },
          {
            id: 'human_detection',
            status: 'completed',
            detail: Array.isArray(sidecar?.garments) && sidecar.garments.length > 0
              ? 'sidecar(yolo/parse)+faces'
              : faceClient
                ? 'objectLocalization(raw)+faces'
                : 'faces-only (no Vision client)',
          },
          {
            id: 'human_parsing',
            status: Array.isArray(sidecar?.garments) && sidecar.garments.length > 0 ? 'completed' : 'skipped',
            detail: Array.isArray(sidecar?.garments) && sidecar.garments.length > 0
              ? 'sidecar garment masks/boxes'
              : 'optional SegFormer/FASHN — not wired',
          },
          {
            id: 'privacy_masking',
            status: privacyApplied ? 'completed' : 'skipped',
            detail: privacyApplied ? 'blur after localization' : undefined,
          },
          { id: 'clothing_extraction', status: fallback ? 'partial' : 'completed', detail: 'crop per garment bbox' },
          { id: 'attribute_inference', status: 'skipped', detail: 'run /preprocess + /infer on crops' },
          { id: 'post_processing', status: 'completed', detail: 'dedupe+salience' },
        ],
        estimated_person_count: estimatedPersonCount,
        multi_person: multiPerson,
      },
      metadata: {
        provider: faceClient ? 'google-vision' : 'server-fallback',
        latency_ms: Date.now() - started,
        version: '1.1.0',
      },
    })
  } catch (error) {
    res.status(400).json({ error: error instanceof Error ? error.message : 'Detection failed' })
  }
})

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    services: {
      gemini: Boolean(genai),
      gemini_api_key_source: genai ? geminiApiKeySource : undefined,
      faceDetection: Boolean(faceClient),
      visionSidecar: Boolean(visionSidecarUrl),
      tryonSidecar: Boolean(tryonSidecarUrl),
      ollama: true,
      serpApiGoogleLens: Boolean(serpApiKey),
    },
  })
})

/**
 * Experiment: proxy SerpApi Google Lens (https://serpapi.com/google-lens-api).
 * Requires SERPAPI_API_KEY. Image must be a URL SerpApi's servers can fetch.
 */
app.post('/api/experiments/google-lens', async (req, res) => {
  try {
    if (!serpApiKey) {
      return res.status(503).json({
        error: 'Set SERPAPI_API_KEY in the environment to use Google Lens (SerpApi).',
        code: 'SERPAPI_NOT_CONFIGURED',
      })
    }
    const body = googleLensExperimentSchema.parse(req.body)
    const started = Date.now()
    const urlForSerp = resolveLensImageUrlForSerpApi(body.imageUrl)

    const params = new URLSearchParams({
      engine: 'google_lens',
      api_key: serpApiKey,
      url: urlForSerp,
    })
    if (body.type) params.set('type', body.type)
    if (body.q) params.set('q', body.q)
    if (body.hl) params.set('hl', body.hl)
    if (body.country) params.set('country', body.country)
    if (typeof body.auto_crop === 'boolean') params.set('auto_crop', body.auto_crop ? 'true' : 'false')
    if (body.no_cache) params.set('no_cache', 'true')

    const serpRes = await fetch(`https://serpapi.com/search.json?${params.toString()}`)
    const data = await serpRes.json().catch(() => ({}))

    if (!serpRes.ok) {
      return res.status(502).json({
        error: typeof data.error === 'string' ? data.error : `SerpApi HTTP ${serpRes.status}`,
        code: 'SERPAPI_HTTP',
        details: data,
      })
    }
    if (data.search_metadata?.status === 'Error' && data.error) {
      return res.status(502).json({
        error: String(data.error),
        code: 'SERPAPI_LENS_ERROR',
        details: data,
      })
    }

    res.json({
      lens_url_sent_to_serpapi: urlForSerp,
      search_metadata: data.search_metadata,
      search_parameters: data.search_parameters,
      visual_matches: data.visual_matches,
      related_content: data.related_content,
      products: data.products,
      ai_overview: data.ai_overview,
      error: data.error,
      metadata: {
        provider: 'serpapi',
        engine: 'google_lens',
        latency_ms: Date.now() - started,
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Google Lens experiment failed'
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.message, code: 'VALIDAATION' })
    }
    res.status(400).json({ error: message, code: 'GOOGLE_LENS_EXPERIMENT' })
  }
})

app.post('/api/tryon/preview', async (req, res) => {
  try {
    const body = tryonPreviewRequestSchema.parse(req.body)
    const started = Date.now()
    const personBuf = await resolveImageBuffer(body.personImageUrl)
    const garmentBuf = await resolveImageBuffer(body.garmentImageUrl)

    const controller = new AbortController()
    const timeoutMs = Number(process.env.TRYON_TIMEOUT_MS ?? 120_000)
    const timeout = setTimeout(() => controller.abort(), timeoutMs)
    let sidecarResponse
    try {
      sidecarResponse = await fetch(`${tryonSidecarUrl}/tryon`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          person_image_base64: personBuf.toString('base64'),
          garment_image_base64: garmentBuf.toString('base64'),
          seed: body.seed,
        }),
        signal: controller.signal,
      })
    } catch (err) {
      clearTimeout(timeout)
      const message = err instanceof Error ? err.message : 'tryon fetch failed'
      return res.status(502).json({
        error: `Try-on sidecar unreachable (${tryonSidecarUrl}): ${message}`,
        code: 'TRYON_SIDECAR_UNREACHABLE',
      })
    }
    clearTimeout(timeout)

    if (!sidecarResponse.ok) {
      return res.status(502).json({
        error: `Try-on sidecar HTTP ${sidecarResponse.status}`,
        code: 'TRYON_SIDECAR_HTTP',
      })
    }

    const data = await sidecarResponse.json()
    if (!data?.ok) {
      return res.status(502).json({
        error: String(data?.error ?? 'Try-on sidecar error'),
        code: 'TRYON_SIDECAR',
      })
    }

    if (!data.implemented || !data.result_image_base64) {
      return res.status(503).json({
        error: String(
          data.message
            ?? 'Try-on weights not wired: set TRYON_WEIGHTS_DIR and implement _run_tryon() in tryon-sidecar/app.py',
        ),
        code: 'TRYON_NOT_CONFIGURED',
        hint: 'See vestir-prototype/tryon-sidecar/README.md and your Kaggle notebook export.',
      })
    }

    const out = Buffer.from(String(data.result_image_base64), 'base64')
    const filename = `${crypto.randomUUID()}.jpg`
    await fs.writeFile(path.join(processedDir, filename), out)
    res.json({
      previewUrl: `/storage/processed/${filename}`,
      metadata: {
        provider: 'tryon-sidecar',
        latency_ms: Date.now() - started,
        version: '1.0.0',
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Try-on preview failed'
    res.status(400).json({ error: message })
  }
})

app.post('/api/items/preprocess', async (req, res) => {
  try {
    const { imageUrl, maskPolygon } = preprocessRequestSchema.parse(req.body)
    const started = Date.now()

    // Items from /detect are already isolated crops — skip all Vision work.
    if (isAlreadyProcessedUrl(imageUrl)) {
      return res.json({
        processedImageUrl: imageUrl,
        faceDetected: false,
        faceBlurApplied: false,
        personDetected: false,
        garmentIsolated: true,
        scene_track: 'flat_lay',
        scene: { face_count: 0, garment_coverage: 1 },
        validation: {},
        warnings: [],
        metadata: {
          provider: 'passthrough',
          model: 'detect-crop',
          latency_ms: Date.now() - started,
          version: '3.1.0',
          pipeline: {
            architecture_id: PIPELINE_ARCHITECTURE_ID,
            stages: [
              { id: 'image_filtering', status: 'skipped', detail: 'already validated at detect' },
              { id: 'human_detection', status: 'skipped', detail: 'crop from /detect' },
              { id: 'human_parsing', status: 'skipped', detail: 'optional SegFormer/FASHN' },
              { id: 'privacy_masking', status: 'skipped', detail: 'applied upstream if worn' },
              { id: 'clothing_extraction', status: 'completed', detail: 'isolated garment crop' },
              { id: 'attribute_inference', status: 'skipped', detail: 'next: /infer' },
              { id: 'post_processing', status: 'skipped', detail: 'downstream' },
            ],
          },
        },
      })
    }

    const buffer = await resolveImageBuffer(imageUrl)
    const validation = await validateInputImage(buffer)
    if (!validation.ok) {
      return res.status(422).json({ error: validation.reason, code: 'INPUT_VALIDATION_FAILED' })
    }

    if (maskPolygon && maskPolygon.length >= 3) {
      const masked = await maskPolygonToPng(buffer, maskPolygon)
      const filename = `${crypto.randomUUID()}.png`
      const filepath = path.join(processedDir, filename)
      await fs.writeFile(filepath, masked)
      return res.json({
        processedImageUrl: `/storage/processed/${filename}`,
        faceDetected: false,
        faceBlurApplied: false,
        personDetected: false,
        garmentIsolated: true,
        scene_track: 'flat_lay',
        scene: { face_count: 0, garment_coverage: 1 },
        validation: validation.metrics,
        warnings: ['maskPolygon crop — downstream color uses alpha-aware stats.'],
        metadata: {
          provider: 'sharp+mask',
          model: 'polygon-alpha-png',
          latency_ms: Date.now() - started,
          version: '3.2.0',
          pipeline: {
            architecture_id: PIPELINE_ARCHITECTURE_ID,
            stages: [
              { id: 'image_filtering', status: 'completed', detail: 'validateInputImage' },
              { id: 'human_detection', status: 'skipped', detail: 'mask polygon' },
              { id: 'human_parsing', status: 'skipped', detail: 'mask polygon' },
              { id: 'privacy_masking', status: 'skipped', detail: 'user mask' },
              { id: 'clothing_extraction', status: 'completed', detail: 'maskPolygon → bbox PNG' },
              { id: 'attribute_inference', status: 'skipped', detail: 'next: /infer' },
              { id: 'post_processing', status: 'skipped', detail: 'downstream' },
            ],
          },
        },
      })
    }

    // Phase 1: understand scene on raw frame before destructive edits.
    const sidecar = await callVisionSidecarAnalyze(buffer)
    const rawFaces = await detectFaces(buffer)
    const garmentRegions = Array.isArray(sidecar?.garments) && sidecar.garments.length > 0
      ? sidecar.garments
        .map((g) => ({
          name: (g.label ?? 'garment').toLowerCase(),
          score: Number(g.confidence ?? 0.6),
          vertices: verticesFromNormalizedBbox(g.bbox),
        }))
        .filter((g) => g.vertices.length > 0)
      : await detectGarmentRegions(buffer)
    const primaryRegion = garmentRegions[0]
    const garmentCoverage = primaryRegion ? normalizedRegionCoverage(primaryRegion.vertices) : 0
    const detectedTrack = detectSceneTrack({ faceCount: rawFaces.length, garmentCoverage })
    const track = detectedTrack === 'ambiguous' ? (rawFaces.length > 0 ? 'worn' : 'flat_lay') : detectedTrack
    const sceneWarning = detectedTrack === 'ambiguous'
      ? 'Scene was ambiguous; fallback track selected automatically.'
      : undefined

    // Track A (worn): blur faces. Track B (flat lay/hanger): skip.
    const regions = rawFaces.map(extractBounds).filter(Boolean)
    const shouldBlurFaces = track === 'worn' && regions.length > 0
    const { output, faceDetected, faceBlurApplied } = shouldBlurFaces
      ? await blurFaceRegions(buffer, regions)
      : { output: buffer, faceDetected: rawFaces.length > 0, faceBlurApplied: false }

    // Crop to garment region (if available) to create clean subject for downstream color + inference.
    const garmentRegion = primaryRegion?.vertices ?? await detectGarmentRegion(output)
    const cropped = await cropToRegion(output, garmentRegion)
    const filename = `${crypto.randomUUID()}.jpg`
    const filepath = path.join(processedDir, filename)
    await fs.writeFile(filepath, cropped)
    res.json({
      processedImageUrl: `/storage/processed/${filename}`,
      faceDetected,
      faceBlurApplied,
      personDetected: Math.max(rawFaces.length, Number(sidecar?.person_count ?? 0)) > 0,
      garmentIsolated: Boolean(garmentRegion),
      scene_track: track,
      scene: {
        face_count: rawFaces.length,
        garment_coverage: Number(garmentCoverage.toFixed(3)),
      },
      validation: validation.metrics,
      warnings: [...(validation.warnings ?? []), ...(sceneWarning ? [sceneWarning] : [])],
      metadata: {
        provider: faceClient ? 'google-vision' : 'sharp-fallback',
        model: faceClient ? 'faceDetection+objectLocalization' : 'none',
        latency_ms: Date.now() - started,
        version: '3.1.0',
        pipeline: {
          architecture_id: PIPELINE_ARCHITECTURE_ID,
          stages: [
            { id: 'image_filtering', status: 'completed', detail: 'validateInputImage' },
            {
              id: 'human_detection',
              status: 'completed',
              detail: Array.isArray(sidecar?.garments) && sidecar.garments.length > 0
                ? 'faceDetection + sidecar garment parsing'
                : 'faceDetection + garment object regions',
            },
            {
              id: 'human_parsing',
              status: Array.isArray(sidecar?.garments) && sidecar.garments.length > 0 ? 'completed' : 'skipped',
              detail: Array.isArray(sidecar?.garments) && sidecar.garments.length > 0
                ? 'sidecar garment masks/boxes'
                : 'optional SegFormer/FASHN',
            },
            {
              id: 'privacy_masking',
              status: shouldBlurFaces ? 'completed' : 'skipped',
              detail: shouldBlurFaces ? 'blur faces after scene track' : undefined,
            },
            {
              id: 'clothing_extraction',
              status: garmentRegion ? 'completed' : 'partial',
              detail: 'cropToRegion from localization',
            },
            { id: 'attribute_inference', status: 'skipped', detail: 'next: /infer' },
            { id: 'post_processing', status: 'skipped', detail: 'downstream' },
          ],
        },
      },
    })
  } catch (error) {
    res.status(400).json({ error: error instanceof Error ? error.message : 'Preprocess failed' })
  }
})

app.post('/api/items/infer', async (req, res) => {
  try {
    if (!genai) return res.status(500).json({ error: 'GEMINI_API_KEY is missing' })
    const { processedImageUrl } = inferRequestSchema.parse(req.body)
    const started = Date.now()
    const imageResponse = await fetch(
      processedImageUrl.startsWith('http')
        ? processedImageUrl
        : `http://127.0.0.1:${port}${processedImageUrl}`,
    )
    if (!imageResponse.ok) throw new Error('Unable to read processed image')
    const imageBuffer = Buffer.from(await imageResponse.arrayBuffer())
    const base64Image = imageBuffer.toString('base64')
    const inferMime = await detectImageMime(imageBuffer)

    try {
      const quality = await scoreImageQuality(imageBuffer)
      const faces = await detectFaces(imageBuffer)
      quality.framing = inferFramingFromContext(faces, quality.occlusion_visible_pct)
      const filterSignature = await detectFilterSignature(imageBuffer)
      const innerCrop = Number(process.env.VESTIR_COLOR_INNER_CROP ?? '0.12')
      const useClahe = process.env.VESTIR_COLOR_CLAHE !== '0'
      const colorAnalysisBuffer = await prepareColorAnalysisBuffer(imageBuffer, {
        innerCrop: Number.isFinite(innerCrop) ? innerCrop : 0.12,
        grayWorld: true,
        clahe: useClahe,
      })
      if (filterSignature.likely_filter) {
        quality.warnings.push(`Image may use a ${filterSignature.kind} filter; color confidence reduced.`)
      }

      const structuralPrompt = `You are a fashion vision parser. Return strict JSON only for physical garment attributes.
Lighting: assume neutral white balance for color names; ignore background walls/floors; focus on fabric pixels only.
Context:
- framing: ${quality.framing}
- visible_pct_estimate: ${quality.occlusion_visible_pct.toFixed(2)}
- if framing is worn, ignore person/face; focus on clothing only.
- if framing is detail, avoid over-inferring full garment identity.

Layering (critical for worn photos):
- Identify the OUTERMOST visible garment (largest area of fabric in front of the torso: e.g. puffer, coat, blazer, cardigan worn open).
- garment_type and category MUST describe that outer layer, not an inner tee/crewneck visible only at the neck unless the frame is clearly a top-only shot.
- If a quilted/puffer/down/shell jacket or parka is visible, category is usually Outerwear and garment_type must say jacket/coat/puffer — not "shirt".
- colors: first list the dominant colors of the OUTER garment's fabric (sleeves/body), not only a small inner neckline sliver. If multiple layers share the frame, weight colors by visible garment area.
- If pattern is plaid/check/stripe, list TWO distinct garment colors when clearly visible (not background).

Schema:
{
  "garment_type":"string",
  "subtype":"string",
  "category":"${CATEGORY_ENUM.join('|')}",
  "colors":[{"name":"string","hex":"#RRGGBB","coverage_pct":0-1,"confidence":0-1}],
  "pattern":"${PATTERN_ENUM.join('|')}",
  "fit":"${FIT_ENUM.join('|')}",
  "construction_details":["string"],
  "material":{"primary":"string","confidence":0-1}
}`

      const useProFirst =
        filterSignature.risk >= 0.42 || quality.blur_score < 0.32 || quality.lighting_score < 0.32
      const temps = [0.22, 0.38, 0.52, 0.44, 0.33]
      let structural
      /** @type {{ tier: string, samples: number, pro_tiebreaker: boolean }} */
      let inferTrace = { tier: 'flash', samples: inferConsistencySamples, pro_tiebreaker: false }

      if (inferConsistencySamples <= 1) {
        const model = useProFirst ? inferProModel : inferFlashModel
        structural = await runGeminiJson(model, structuralSchema, structuralPrompt, base64Image, {
          mimeType: inferMime,
          temperature: useProFirst ? 0.22 : 0.35,
        })
        inferTrace = { tier: useProFirst ? 'pro-hard-shot' : 'flash', samples: 1, pro_tiebreaker: false }
      } else if (useProFirst) {
        structural = await runGeminiJson(inferProModel, structuralSchema, structuralPrompt, base64Image, {
          mimeType: inferMime,
          temperature: 0.22,
        })
        inferTrace = { tier: 'pro-hard-shot', samples: 1, pro_tiebreaker: false }
      } else {
        const samples = []
        for (let i = 0; i < inferConsistencySamples; i += 1) {
          samples.push(
            await runGeminiJson(inferFlashModel, structuralSchema, structuralPrompt, base64Image, {
              mimeType: inferMime,
              temperature: temps[i % temps.length],
            }),
          )
        }
        if (inferNeedsProTiebreaker(samples)) {
          structural = await runGeminiJson(inferProModel, structuralSchema, structuralPrompt, base64Image, {
            mimeType: inferMime,
            temperature: 0.2,
          })
          inferTrace = {
            tier: 'pro-tiebreaker',
            samples: inferConsistencySamples,
            pro_tiebreaker: true,
          }
        } else {
          structural = pickStructuralByCategoryVote(samples)
          inferTrace = { tier: 'flash-consensus', samples: inferConsistencySamples, pro_tiebreaker: false }
        }
      }

      const coerced = coerceCategoryForGarmentLabels(structural.garment_type, structural.subtype, structural.category)
      let resolvedGarmentType = structural.garment_type
      let resolvedCategory = coerced.category
      if (coerced.adjusted && garmentLabelLooksLikeOuterShirt(structural.garment_type)) {
        const sub = (structural.subtype ?? '').trim()
        resolvedGarmentType =
          sub && outerwearSignalsInText(sub) && /jacket|coat|puffer|parka|blazer/i.test(sub)
            ? sub
            : 'Quilted jacket'
        quality.warnings.push('Updated type and category: outer jacket/coat was misread as a basic shirt — please confirm.')
      } else if (coerced.adjusted) {
        quality.warnings.push('Category set to Outerwear (visible jacket/coat) — please confirm.')
      }
      const structuralContext = {
        ...structural,
        garment_type: resolvedGarmentType,
        category: resolvedCategory,
      }

      const semanticPrompt = `You are a fashion stylist model. Return strict JSON only for contextual meaning.
Use this structural context:
${JSON.stringify(structuralContext)}
And this quality context:
${JSON.stringify(quality)}
Schema:
{
  "formality":1-10,
  "seasons":{"spring":0-1,"summer":0-1,"autumn":0-1,"winter":0-1},
  "occasions":[${OCCASION_ENUM.map((v) => `"${v}"`).join(',')}],
  "style_archetype":"${STYLE_ENUM.join('|')}",
  "layering_role":"${LAYERING_ROLE_ENUM.join('|')}",
  "pairings":["string","string","string"],
  "confidence":{"formality":0-1,"occasions":0-1,"style_archetype":0-1,"seasonality":0-1}
}`

      const semanticModel =
        process.env.INFER_SEMANTIC_USE_PRO === '1' && useProFirst ? inferProModel : inferFlashModel
      const semantic = await runGeminiJson(semanticModel, semanticSchema, semanticPrompt, base64Image, {
        mimeType: inferMime,
        temperature: 0.35,
      })

      const modelColors = [...structural.colors]
        .sort((a, b) => b.coverage_pct - a.coverage_pct)
        .slice(0, 3)
        .map((c) => {
          const rgb = hexToRgb(c.hex)
          const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b)
          return {
            name: c.name,
            hex: normalizeHex(c.hex),
            hsl,
            coverage_pct: c.coverage_pct,
            is_neutral: isNeutralHsl(hsl),
            confidence: c.confidence,
          }
        })
      const visionColorsLocal = await extractColorPaletteFromImage(colorAnalysisBuffer)
      const visionColorsLab = await extractLabKMeansPalette(colorAnalysisBuffer)
      const visionColorsApi = await extractVisionColorPalette(colorAnalysisBuffer)
      const visionColors = [...visionColorsApi, ...visionColorsLab, ...visionColorsLocal]
      const primaryModel = modelColors[0]
      const primaryVision = visionColors[0]
      const colorMismatch = primaryModel && primaryVision
        ? rgbDistance(hexToRgb(primaryModel.hex), hexToRgb(primaryVision.hex)) > 52
        : false
      const shouldTrustVisionPrimary = !primaryModel || primaryModel.confidence < 0.85 || colorMismatch
      const wornPrefersVision = quality.framing === 'worn' && Boolean(primaryVision)

      const mergedColors =
        wornPrefersVision
          ? dedupePalette([primaryVision, ...visionColors, ...modelColors], 3)
          : shouldTrustVisionPrimary && primaryVision
            ? dedupePalette([primaryVision, ...visionColors, ...modelColors], 3)
            : dedupePalette([...modelColors, ...visionColors], 3)
      const colors = prioritizeLightNeutralsPrimary(mergedColors.length ? mergedColors : modelColors)

      const seasonEntries = Object.entries(semantic.seasons)
      const season = seasonEntries
        .filter(([, weight]) => weight >= 0.5)
        .map(([name]) => name)
      if (!season.length) season.push('spring')

      const confidenceScores = [
        structural.material.confidence,
        ...colors.map((c) => c.confidence),
        semantic.confidence.formality,
        semantic.confidence.occasions,
        semantic.confidence.style_archetype,
        semantic.confidence.seasonality,
        quality.blur_score,
        quality.lighting_score,
      ]
      const confidenceOverall = Number(
        (confidenceScores.reduce((sum, value) => sum + value, 0) / confidenceScores.length).toFixed(3),
      )
      const uncertainFields = []
      if (confidenceBand(structural.material.confidence) !== 'accepted') uncertainFields.push('material')
      if (confidenceBand(semantic.confidence.style_archetype) !== 'accepted') uncertainFields.push('style_archetype')
      if (confidenceBand(semantic.confidence.occasions) !== 'accepted') uncertainFields.push('occasions')
      if (shouldTrustVisionPrimary) uncertainFields.push('color_primary')
      if (filterSignature.risk >= 0.35) uncertainFields.push('color_filter_cast')
      if (confidenceBand(quality.blur_score) === 'needs_confirmation') uncertainFields.push('image_blur')
      if (confidenceBand(quality.lighting_score) === 'needs_confirmation') uncertainFields.push('image_lighting')
      const contradictions = contradictionFlags({
        subtype: structural.subtype,
        material: structural.material.primary,
        formality: semantic.formality,
        seasonWeights: semantic.seasons,
        pattern: structural.pattern,
        colors,
      })
      if (contradictions.length) uncertainFields.push(...contradictions)
      if (coerced.adjusted) uncertainFields.push('garment_identity')
      if (wornPrefersVision && colorMismatch) uncertainFields.push('color_primary_photo_weighted')
      const adjustedConfidenceOverall = Number(
        (confidenceOverall * (1 - Math.min(0.2, filterSignature.risk * 0.25))).toFixed(3),
      )

      const parsed = inferSchema.parse({
        item_type: resolvedGarmentType,
        subtype: structural.subtype,
        category: resolvedCategory,
        color_primary: colors[0]?.name ?? 'Taupe',
        color_secondary: colors[1]?.name,
        color_primary_hsl: colors[0]?.hsl ?? hslFromColorName('Taupe'),
        color_secondary_hsl: colors[1]?.hsl,
        color_palette: colors.map(({ confidence, ...c }) => c),
        pattern: structural.pattern,
        fit: structural.fit && structural.fit !== 'unknown' ? structural.fit : undefined,
        material: structural.material.primary,
        material_confidence: structural.material.confidence,
        formality: semantic.formality,
        season,
        season_weights: semantic.seasons,
        occasions: semantic.occasions,
        style_archetype: semantic.style_archetype,
        layering_role: semantic.layering_role ?? 'standalone',
        pairings: (semantic.pairings ?? []).slice(0, 3),
        confidence_overall: adjustedConfidenceOverall,
        uncertainty: {
          requires_user_confirmation: adjustedConfidenceOverall < 0.6 || uncertainFields.length > 0,
          uncertain_fields: [...new Set(uncertainFields)],
        },
        quality,
      })

      res.json({
        ...parsed,
        metadata: {
          provider: 'google',
          model: `gemini(${inferTrace.tier}+semantic)+lab-kmeans+grayworld`,
          latency_ms: Date.now() - started,
          version: '2.2.0',
          infer_routing: inferTrace,
          pipeline: {
            architecture_id: PIPELINE_ARCHITECTURE_ID,
            stages: [
              { id: 'image_filtering', status: 'skipped', detail: 'handled in /preprocess' },
              { id: 'human_detection', status: 'skipped', detail: 'handled in /preprocess' },
              { id: 'human_parsing', status: 'skipped', detail: 'optional SegFormer/FASHN' },
              { id: 'privacy_masking', status: 'skipped', detail: 'handled in /preprocess' },
              { id: 'clothing_extraction', status: 'skipped', detail: 'handled in /preprocess' },
              {
                id: 'attribute_inference',
                status: 'completed',
                detail: 'tiered Gemini + optional multi-sample + LAB/merge palette',
              },
              {
                id: 'post_processing',
                status: 'completed',
                detail: 'palette merge, uncertainty, contradictions',
              },
            ],
          },
        },
      })
    } catch (geminiError) {
      const fallback = await fallbackInferFromVision(imageBuffer)
      res.json({
        ...fallback,
        warning: `Gemini fallback used: ${geminiError instanceof Error ? geminiError.message : 'unknown error'}`,
      })
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Inference failed'
    res.status(400).json({ error: `Inference failed: ${message}` })
  }
})

app.post('/api/items/embed', async (req, res) => {
  try {
    if (!genai && !embeddingSidecarUrl) {
      return res.status(500).json({ error: 'Set GEMINI_API_KEY and/or EMBEDDING_SIDECAR_URL for embeddings' })
    }
    const { item } = embedRequestSchema.parse(req.body)
    const started = Date.now()
    const payload = buildEmbeddingPayloadFromItem(item)
    let values = []
    let modelUsed = embeddingModel
    let warning

    if (embeddingSidecarUrl) {
      try {
        let imageBase64
        if (item.image_url) {
          const fetchUrl = item.image_url.startsWith('http')
            ? item.image_url
            : `http://127.0.0.1:${port}${item.image_url}`
          const ir = await fetch(fetchUrl)
          if (ir.ok) imageBase64 = Buffer.from(await ir.arrayBuffer()).toString('base64')
        }
        const er = await fetch(`${embeddingSidecarUrl.replace(/\/$/, '')}/embed`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: payload, image_base64: imageBase64 }),
          signal: AbortSignal.timeout(20000),
        })
        if (er.ok) {
          const data = await er.json()
          const vec = data.vector ?? data.values ?? data.embedding
          if (Array.isArray(vec) && vec.length) {
            values = vec
            modelUsed = typeof data.model === 'string' ? data.model : 'embedding-sidecar'
          }
        } else {
          warning = `Embedding sidecar HTTP ${er.status}`
        }
      } catch (e) {
        warning = `Embedding sidecar: ${e instanceof Error ? e.message : 'unreachable'}`
      }
    }

    try {
      if (!values.length && genai) {
        const embedRes = await genai.models.embedContent({
          model: embeddingModel,
          contents: payload,
        })
        values = embedRes.embeddings?.[0]?.values ?? []
        if (!values.length) throw new Error('Embedding API returned empty vector')
      } else if (!values.length) {
        throw new Error('No embedding provider succeeded')
      }
    } catch (embedError) {
      values = buildDeterministicFallbackVector(payload, 256)
      modelUsed = 'deterministic-fallback'
      warning = `Embedding fallback used: ${embedError instanceof Error ? embedError.message : 'unknown error'}`
    }

    const vectorId = crypto.randomUUID()
    const existing = JSON.parse(await fs.readFile(embeddingsFile, 'utf8'))
    existing.push({
      id: vectorId,
      item_id: item.id,
      vector: values,
      item_snapshot: {
        item_id: item.id,
        image_url: item.image_url,
        item_type: item.item_type,
        category: item.category,
        color_primary: item.color_primary,
        material: item.material,
      },
      created_at: new Date().toISOString(),
      model: modelUsed,
    })
    await fs.writeFile(embeddingsFile, JSON.stringify(existing), 'utf8')
    const embedProvider =
      modelUsed === 'deterministic-fallback'
        ? 'server'
        : modelUsed === embeddingModel
          ? 'google'
          : 'embedding-sidecar'

    res.json({
      vector_id: vectorId,
      dimensions: values.length,
      metadata: {
        provider: embedProvider,
        model: modelUsed,
        latency_ms: Date.now() - started,
        version: '1.1.0',
      },
      warning,
    })
  } catch (error) {
    res.status(400).json({ error: error instanceof Error ? error.message : 'Embedding failed' })
  }
})

app.post('/api/extension/match', async (req, res) => {
  try {
    const body = extensionMatchSchema.parse(req.body)
    const started = Date.now()
    const topK = body.topK ?? 5
    const preprocess = await fetch(`http://127.0.0.1:${port}/api/items/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl: body.imageUrl }),
    })
    const preprocessJson = await preprocess.json()
    if (!preprocess.ok) {
      return res.status(400).json({
        error: preprocessJson?.error ?? 'Preprocess failed for extension image',
        code: 'EXTENSION_PREPROCESS_FAILED',
      })
    }
    const infer = await fetch(`http://127.0.0.1:${port}/api/items/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ processedImageUrl: preprocessJson.processedImageUrl }),
    })
    const inferJson = await infer.json()
    if (!infer.ok) {
      return res.status(400).json({
        error: inferJson?.error ?? 'Inference failed for extension image',
        code: 'EXTENSION_INFER_FAILED',
      })
    }

    const queryItem = {
      id: `ext-${crypto.randomUUID()}`,
      image_url: preprocessJson.processedImageUrl,
      item_type: inferJson.item_type ?? 'Garment',
      category: inferJson.category ?? 'Tops',
      color_primary: inferJson.color_primary ?? 'Taupe',
      material: inferJson.material ?? 'Unknown',
      formality: inferJson.formality ?? 5,
      season: Array.isArray(inferJson.season) ? inferJson.season : ['spring'],
      raw_attributes: JSON.stringify(inferJson),
    }
    const payload = buildEmbeddingPayloadFromItem(queryItem)
    const queryVector = genai
      ? (await genai.models.embedContent({
          model: embeddingModel,
          contents: payload,
        })).embeddings?.[0]?.values ?? buildDeterministicFallbackVector(payload, 256)
      : buildDeterministicFallbackVector(payload, 256)

    const entriesRaw = JSON.parse(await fs.readFile(embeddingsFile, 'utf8'))
    const entries = Array.isArray(entriesRaw) ? entriesRaw : []
    const queryDim = Array.isArray(queryVector) ? queryVector.length : 0
    const compatible = entries.filter((entry) => Array.isArray(entry.vector) && entry.vector.length === queryDim)
    const ranked = compatible
      .map((entry) => {
        const embeddingScore = cosineSimilarity(queryVector, entry.vector)
        const snapshot = entry.item_snapshot ?? null
        const categoryBoost = snapshot && normText(snapshot.category) === normText(inferJson.category) ? 0.12 : 0
        const colorBoost = snapshot && colorFamily(snapshot.color_primary) === colorFamily(inferJson.color_primary) ? 0.08 : 0
        const finalScore = Math.max(0, Math.min(1, embeddingScore * 0.8 + categoryBoost + colorBoost))
        return {
          item_id: entry.item_id,
          score: finalScore,
          score_breakdown: {
            embedding: Number(embeddingScore.toFixed(4)),
            category_boost: Number(categoryBoost.toFixed(4)),
            color_boost: Number(colorBoost.toFixed(4)),
          },
          snapshot,
        }
      })
      .filter((x) => Number.isFinite(x.score) && x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)

    res.json({
      query: {
        title: body.title ?? null,
        page_url: body.pageUrl ?? null,
        image_url: body.imageUrl,
        processed_image_url: preprocessJson.processedImageUrl,
        inferred: {
          item_type: inferJson.item_type,
          category: inferJson.category,
          color_primary: inferJson.color_primary,
          material: inferJson.material,
        },
      },
      matches: ranked,
      metadata: {
        provider: genai ? 'google' : 'deterministic-fallback',
        model: genai ? embeddingModel : 'deterministic-fallback',
        latency_ms: Date.now() - started,
        version: '1.1.0',
        query_dimensions: queryDim,
        compared_items: compatible.length,
        skipped_dimension_mismatch: Math.max(0, entries.length - compatible.length),
      },
    })
  } catch (error) {
    res.status(400).json({
      error: error instanceof Error ? error.message : 'Extension matching failed',
      code: 'EXTENSION_MATCH_FAILED',
    })
  }
})

app.post('/api/items/reason', async (req, res) => {
  try {
    const { item } = reasonRequestSchema.parse(req.body)
    const started = Date.now()
    const prompt = `Return strict JSON only.
Generate practical styling intelligence for this wardrobe item:
${JSON.stringify({
  item_type: item.item_type,
  category: item.category,
  color_primary: item.color_primary,
  material: item.material,
  season: item.season,
  formality: item.formality,
})}

Rules:
- Follow category literally: if category is Outerwear, describe a jacket/coat layer (never call it a shirt, tee, or blouse).
- If category is Tops, do not call it outerwear unless the item_type clearly indicates a jacket/coat.
- Use color_primary as the main color in the summary (natural language).

Schema:
{
  "summary":"string",
  "pairing_suggestions":["3 concise specific pairings"],
  "avoid_note":"one clash warning with reason",
  "care_context":"one care or occasion caution"
}`

    try {
      const ollamaRes = await fetch(`${ollamaBaseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: ollamaModel,
          prompt,
          stream: false,
        }),
      })
      if (!ollamaRes.ok) throw new Error(`Ollama error (${ollamaRes.status})`)
      const data = await ollamaRes.json()
      const rawResponse = String(data.response ?? '').trim()
      let parsed
      try {
        parsed = JSON.parse(extractJsonObject(rawResponse))
      } catch {
        parsed = null
      }
      res.json({
        summary: parsed?.summary ?? (rawResponse || 'No reasoning generated'),
        pairing_suggestions: Array.isArray(parsed?.pairing_suggestions) ? parsed.pairing_suggestions.slice(0, 3) : [],
        avoid_note: parsed?.avoid_note ?? 'Avoid pairing with clashing saturation levels to keep the look balanced.',
        care_context: parsed?.care_context ?? 'Use context-aware wear: avoid settings that can damage the fabric.',
        metadata: {
          provider: 'ollama',
          model: ollamaModel,
          latency_ms: Date.now() - started,
          version: '1.0.0',
        },
      })
    } catch (ollamaError) {
      res.json({
        summary: `${item.color_primary} ${item.item_type} can anchor a balanced look for ${Array.isArray(item.season) ? item.season.join(', ') : 'multiple seasons'}.`,
        pairing_suggestions: [
          `Pair with a neutral ${item.category === 'Bottoms' ? 'top' : 'bottom'} for contrast.`,
          'Add one structured layer to improve silhouette.',
          'Finish with clean, low-contrast footwear.',
        ],
        avoid_note: 'Avoid combining with equally loud prints; pattern competition weakens cohesion.',
        care_context: 'Skip high-abrasion environments if the fabric can snag or pill.',
        metadata: {
          provider: 'server-fallback',
          model: 'reasoning-template-v1',
          latency_ms: Date.now() - started,
          version: '1.0.0',
        },
        warning: `Reasoning fallback used: ${ollamaError instanceof Error ? ollamaError.message : 'unknown error'}`,
      })
    }
  } catch (error) {
    res.status(400).json({ error: error instanceof Error ? error.message : 'Reasoning failed' })
  }
})

async function startServer() {
  await ensureStorage()
  const server = app.listen(port, () => {
    console.log(`Vestir API running on http://127.0.0.1:${port}`)
  })

  await new Promise((resolve, reject) => {
    server.on('close', resolve)
    server.on('error', reject)
  })
}

startServer().catch((error) => {
  console.error('API server failed:', error)
  process.exit(1)
})
