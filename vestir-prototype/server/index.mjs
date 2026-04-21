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
import { fashionColorNameFromLab } from './lib/fashionColorMap.mjs'
import {
  prepareColorAnalysisBuffer,
  maskPolygonToPng,
  detectImageMime,
} from './lib/vestirColor.mjs'
import { runGemmaMlxVision } from './gemma-mlx-runner.mjs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const rootDir = path.resolve(__dirname, '..')
dotenv.config({ path: path.resolve(rootDir, '..', '.env') })
dotenv.config({ path: path.resolve(rootDir, '.env') })

const storageDir = path.join(rootDir, 'server', 'storage')
const processedDir = path.join(storageDir, 'processed')
const embeddingsFile = path.join(storageDir, 'embeddings.json')
const wardrobeItemsFile = path.join(storageDir, 'wardrobe_items.json')
const wardrobeCompatibilityFile = path.join(storageDir, 'wardrobe_compatibility.json')

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
const inferBackend = (process.env.VESTIR_INFER_BACKEND ?? 'hybrid_sidecar').toLowerCase()
/** gemini_on_disagreement: second Gemini pass when vision-sidecar sets attribute_disagreement (skipped when skipArbitration). */
const inferArbitrateMode = (process.env.VESTIR_INFER_ARBITRATE ?? '').trim().toLowerCase()
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
  schema_version: z.number().int().min(1).default(2),
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
  dominant_colors: z.array(z.string().regex(/^#[0-9a-fA-F]{6}$/)).min(1).max(3),
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
  source_image_stage: z.enum(['tryoff', 'blurred_fallback']).optional(),
  gemini_style_notes: z.string().optional(),
  gemini_design_tags: z.array(z.string()).optional(),
  gemini_brand_like: z.array(z.object({
    name: z.string(),
    confidence: z.number().min(0).max(1),
  })).optional(),
  quality: qualitySchema,
})

const finiteCoord = z.coerce.number().refine((n) => Number.isFinite(n), 'coordinate must be finite')
const normalizedPointSchema = z.object({ x: finiteCoord, y: finiteCoord })
const bboxCoord01 = z.coerce.number().min(0).max(1)
const normalizedBboxSchema = z.object({
  x1: bboxCoord01,
  y1: bboxCoord01,
  x2: bboxCoord01,
  y2: bboxCoord01,
})
const subjectFilterSchema = z.object({
  mode: z.enum(['keep_selected_person', 'clothing_only', 'focus_person_blur_others']).default('keep_selected_person'),
  selectedPersonIds: z.array(z.string()).optional(),
  selectedPersonBboxes: z.preprocess(
    (val) => (Array.isArray(val) && val.length === 0 ? undefined : val),
    z.array(normalizedBboxSchema).optional(),
  ),
  /** 3+ points: polygon mask; 1–2 points: axis-aligned fallback in /mannequin; SAM can return many vertices. */
  maskPolygon: z.preprocess(
    (val) => (Array.isArray(val) && val.length === 0 ? undefined : val),
    z.array(normalizedPointSchema).min(1).max(50000).optional(),
  ),
  aiAssist: z.boolean().optional(),
}).nullish()
const detectRequestSchema = z.object({
  imageUrl: z.string().min(1),
  subjectFilter: subjectFilterSchema,
  privacyPolicy: z.enum(['auto', 'strict', 'off']).optional(),
})
const preprocessRequestSchema = z.object({
  imageUrl: z.string().min(1),
  /** Normalized 0–1 vertices; garment interior stays opaque in a PNG for downstream color. */
  maskPolygon: z.array(normalizedPointSchema).min(3).optional(),
  subjectFilter: subjectFilterSchema,
  removeBackground: z.boolean().optional(),
  privacyPolicy: z.enum(['auto', 'strict', 'off']).optional(),
})
const inferRequestSchema = z.object({
  processedImageUrl: z.string().min(1),
  sourceImageStage: z.enum(['tryoff', 'blurred_fallback']).optional(),
  forceGemini: z.boolean().optional(),
  /** When true, skip Gemini disagreement arbitration (e.g. extension match latency). */
  skipArbitration: z.boolean().optional(),
})
const advancedLocalRequestSchema = z.object({
  imageUrl: z.string().min(1),
  context: z.object({
    item_type: z.string().optional(),
    category: z.string().optional(),
    color_primary: z.string().optional(),
    material: z.string().optional(),
    pattern: z.string().optional(),
  }).optional(),
})
const advancedLocalGemmaSchema = z.object({
  item_type: z.string().min(1).optional(),
  category: z.enum(CATEGORY_ENUM).optional(),
  color_primary: z.string().min(1).optional(),
  material: z.string().min(1).optional(),
  pattern: z.string().min(1).optional(),
  confidence_overall: z.number().min(0).max(1).optional(),
  design_tags: z.array(z.string()).max(20).optional(),
  style_notes: z.string().max(600).optional(),
  style_tags: z.array(z.string()).max(12).optional(),
  occasions: z.array(z.string()).max(12).optional(),
})
const embedRequestSchema = z.object({ item: z.any() })
const reasonRequestSchema = z.object({ item: z.any() })
const wardrobeItemUpsertSchema = z.object({
  item: z.any(),
  mannequinImageUrl: z.string().optional(),
  attributes: z.record(z.any()).optional(),
})
const wardrobeCompatibilitySchema = z.object({
  itemA: z.any(),
  itemB: z.any(),
  useLlm: z.boolean().optional(),
})
const wardrobeOutfitBuildSchema = z.object({
  anchorItemId: z.string().optional(),
  anchorItem: z.any().optional(),
  wardrobeItems: z.array(z.any()).optional(),
  topK: z.number().int().min(1).max(10).optional(),
  useLlm: z.boolean().optional(),
  profile: z.object({
    style_intent: z.enum(['balanced', 'formal', 'casual', 'bold']).optional(),
    aesthetic_keywords: z.array(z.string()).optional(),
  }).optional(),
  weather: z.object({
    mode: z.enum(['warm', 'cold', 'mild', 'all']).optional(),
    temperature_c: z.number().optional(),
    condition: z.string().optional(),
  }).optional(),
  requireCategories: z.array(z.enum(['Tops', 'Bottoms', 'Outerwear', 'Shoes', 'Accessories'])).optional(),
})
const styleProfileSchema = z.object({
  style_intent: z.enum(['balanced', 'formal', 'casual', 'bold']).optional(),
  aesthetic_keywords: z.array(z.string()).optional(),
}).optional()
const weatherContextSchema = z.object({
  mode: z.enum(['warm', 'cold', 'mild', 'all']).optional(),
  temperature_c: z.number().optional(),
  condition: z.string().optional(),
}).optional()
const wardrobeSuggestNextSchema = z.object({
  anchorItem: z.any(),
  wardrobeItems: z.array(z.any()).optional(),
  profile: styleProfileSchema,
  weather: weatherContextSchema,
  topK: z.number().int().min(1).max(12).optional(),
  useLlm: z.boolean().optional(),
})
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

const tryoffRequestSchema = z.object({
  imageUrl: z.string().min(1),
  garmentTarget: z
    .enum(['outfit', 'ensemble', 'tshirt', 'dress', 'pants', 'jacket'])
    .optional()
    .default('outfit'),
  seed: z.number().int().optional(),
})
const manualBlurRequestSchema = z.object({
  imageUrl: z.string().min(1),
  boxes: z.array(normalizedBboxSchema).optional(),
  maskPolygon: z.array(normalizedPointSchema).min(3).optional(),
  blurAmount: z.number().int().min(3).max(99).optional(),
  blurPreset: z.enum(['soft', 'pro', 'strong']).optional(),
}).refine((v) => (v.boxes?.length ?? 0) > 0 || (v.maskPolygon?.length ?? 0) >= 3, {
  message: 'Provide at least one blur box or a polygon mask.',
})
const mannequinContextSchema = z
  .object({
    /** Vision chip labels from the client (drives category + try-off garment_target). */
    selectedLabels: z.array(z.string()).optional(),
    /** User or upstream model hints: collar, sleeves, fabric, etc. */
    attributeHints: z.record(z.string(), z.string()).optional(),
    garmentTarget: z.enum(['outfit', 'ensemble', 'tshirt', 'dress', 'pants', 'jacket']).optional(),
  })
  .optional()
const mannequinRequestSchema = z.object({
  imageUrl: z.string().min(1),
  subjectFilter: subjectFilterSchema,
  context: mannequinContextSchema,
})
const samRefineRequestSchema = z.object({
  imageUrl: z.string().min(1),
  boxes: z.array(normalizedBboxSchema).min(1),
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
  try {
    await fs.access(wardrobeItemsFile)
  } catch {
    await fs.writeFile(wardrobeItemsFile, JSON.stringify([]), 'utf8')
  }
  try {
    await fs.access(wardrobeCompatibilityFile)
  } catch {
    await fs.writeFile(wardrobeCompatibilityFile, JSON.stringify([]), 'utf8')
  }
}

/** Load embedding cache; repair corrupt/truncated JSON so /embed never hard-fails the whole pipeline. */
async function readEmbeddingsArray() {
  try {
    const raw = (await fs.readFile(embeddingsFile, 'utf8')).trim()
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err)
    const backup = `${embeddingsFile}.corrupt.${Date.now()}.bak`
    try {
      await fs.rename(embeddingsFile, backup)
    } catch {
      /* ignore — file may be missing or busy */
    }
    await fs.writeFile(embeddingsFile, JSON.stringify([]), 'utf8')
    console.warn(`[embeddings] Invalid embeddings.json (${reason}). Reset to []. Backup attempted: ${backup}`)
    return []
  }
}

async function readJsonArrayFile(filePath) {
  try {
    const raw = (await fs.readFile(filePath, 'utf8')).trim()
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

async function writeJsonArrayFile(filePath, data) {
  await fs.writeFile(filePath, JSON.stringify(Array.isArray(data) ? data : []), 'utf8')
}

function parseRawAttributes(item) {
  if (item && typeof item.raw_attributes === 'object' && item.raw_attributes !== null) return item.raw_attributes
  if (typeof item?.raw_attributes === 'string') {
    try { return JSON.parse(item.raw_attributes) } catch { return {} }
  }
  return {}
}

function categoryCompatibilityBoost(a, b) {
  const ca = normText(a?.category)
  const cb = normText(b?.category)
  if (!ca || !cb) return 0
  const pair = `${ca}->${cb}`
  const accepted = new Set([
    'tops->bottoms', 'bottoms->tops',
    'tops->outerwear', 'outerwear->tops',
    'bottoms->shoes', 'shoes->bottoms',
    'tops->shoes', 'shoes->tops',
    'outerwear->bottoms', 'bottoms->outerwear',
  ])
  if (ca === cb) return ca === 'accessories' ? 0.03 : -0.08
  return accepted.has(pair) ? 0.14 : -0.04
}

function colorHarmonyBoost(a, b) {
  const fa = colorFamily(a?.color_primary)
  const fb = colorFamily(b?.color_primary)
  if (!fa || !fb) return 0
  if (fa === fb) return 0.08
  const neutral = new Set(['light-neutral', 'dark-neutral', 'mid-neutral', 'white', 'black', 'grey', 'gray'])
  if (neutral.has(fa) || neutral.has(fb)) return 0.1
  const complements = new Set(['blue|brown', 'brown|blue', 'green|brown', 'brown|green', 'red|blue', 'blue|red'])
  return complements.has(`${fa}|${fb}`) ? 0.06 : 0
}

function styleOccasionBoost(a, b) {
  const ar = parseRawAttributes(a)
  const br = parseRawAttributes(b)
  const aOcc = new Set(Array.isArray(ar.occasions) ? ar.occasions.map(normText) : [])
  const bOcc = new Set(Array.isArray(br.occasions) ? br.occasions.map(normText) : [])
  const overlapOcc = [...aOcc].filter((x) => bOcc.has(x)).length
  const styleA = normText(ar.style_archetype)
  const styleB = normText(br.style_archetype)
  const styleMatch = styleA && styleB && styleA === styleB ? 0.06 : 0
  return Math.min(0.12, overlapOcc * 0.03 + styleMatch)
}

function patternPenalty(a, b) {
  const ar = parseRawAttributes(a)
  const br = parseRawAttributes(b)
  const pa = normText(ar.pattern)
  const pb = normText(br.pattern)
  if (!pa || !pb) return 0
  const loud = new Set(['graphic', 'mixed', 'floral', 'plaid', 'check', 'stripe'])
  if (loud.has(pa) && loud.has(pb) && pa !== pb) return -0.07
  return 0
}

function explainHeuristicPairing(a, b, score) {
  const reasons = []
  if (categoryCompatibilityBoost(a, b) > 0.1) reasons.push('categories complement each other')
  if (colorHarmonyBoost(a, b) >= 0.08) reasons.push('colors harmonize well')
  if (styleOccasionBoost(a, b) > 0.03) reasons.push('style/occasion overlap is strong')
  if (!reasons.length) reasons.push('embedding similarity indicates a workable match')
  return `${reasons.join(', ')}.`
}

async function ensureEmbeddingForItem(item) {
  const existing = await readEmbeddingsArray()
  const latest = [...existing].reverse().find((entry) => entry?.item_id === item?.id && Array.isArray(entry?.vector) && entry.vector.length > 0)
  if (latest) return latest

  const resp = await fetch(`http://127.0.0.1:${port}/api/items/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item }),
  })
  if (!resp.ok) {
    const txt = await resp.text()
    throw new Error(`embed_failed_${resp.status}: ${txt.slice(0, 300)}`)
  }
  const json = await resp.json()
  const after = await readEmbeddingsArray()
  const byId = [...after].find((entry) => entry?.id === json?.vector_id)
  if (byId?.vector?.length) return byId
  const fallback = [...after].reverse().find((entry) => entry?.item_id === item?.id && Array.isArray(entry?.vector) && entry.vector.length > 0)
  if (fallback) return fallback
  throw new Error(`embedding_missing_for_item_${item?.id ?? 'unknown'}`)
}

function computeCompatibilityHeuristic(itemA, itemB, vectorA, vectorB) {
  const embeddingScore = cosineSimilarity(vectorA, vectorB)
  const categoryBoost = categoryCompatibilityBoost(itemA, itemB)
  const colorBoost = colorHarmonyBoost(itemA, itemB)
  const styleBoost = styleOccasionBoost(itemA, itemB)
  const pattPenalty = patternPenalty(itemA, itemB)
  const score = Math.max(0, Math.min(1, embeddingScore * 0.68 + categoryBoost + colorBoost + styleBoost + pattPenalty))
  return {
    score,
    breakdown: {
      embedding: Number(embeddingScore.toFixed(4)),
      category_boost: Number(categoryBoost.toFixed(4)),
      color_boost: Number(colorBoost.toFixed(4)),
      style_boost: Number(styleBoost.toFixed(4)),
      pattern_penalty: Number(pattPenalty.toFixed(4)),
    },
  }
}

async function llmCompatibilityAdjustment(itemA, itemB) {
  if (!genai) return null
  const prompt = `Return strict JSON only.
Rate outfit compatibility for these two wardrobe items on a 0-1 scale.

itemA: ${JSON.stringify({
  item_type: itemA?.item_type,
  category: itemA?.category,
  color_primary: itemA?.color_primary,
  material: itemA?.material,
  raw_attributes: parseRawAttributes(itemA),
})}
itemB: ${JSON.stringify({
  item_type: itemB?.item_type,
  category: itemB?.category,
  color_primary: itemB?.color_primary,
  material: itemB?.material,
  raw_attributes: parseRawAttributes(itemB),
})}

Schema:
{"score": number, "explanation": "short reason (<=18 words)"}
Rules:
- score in [0,1]
- prefer practical wearable pairings
- penalize same-role collisions (top+top, bottom+bottom) unless clearly layering`
  try {
    const out = await genai.models.generateContent({
      model: inferFlashModel,
      contents: prompt,
    })
    const raw = String(out?.text ?? '').trim()
    const parsed = JSON.parse(extractJsonObject(raw))
    const score = Math.max(0, Math.min(1, Number(parsed?.score ?? 0.5)))
    const explanation = String(parsed?.explanation ?? '').trim() || 'LLM assessed moderate compatibility.'
    return { score, explanation }
  } catch {
    return null
  }
}

function parseDataUrl(dataUrl) {
  const match = dataUrl.match(/^data:(.+?);base64,(.+)$/)
  if (!match) throw new Error('Expected base64 data URL image')
  return { mime: match[1], buffer: Buffer.from(match[2], 'base64') }
}

/**
 * Browser may send absolute URLs like http://localhost:5173/storage/... (Vite dev).
 * Node must load from this API (same process), not via the Vite host.
 */
function normalizeImageUrlForServerFetch(imageUrl) {
  if (typeof imageUrl !== 'string' || imageUrl.startsWith('data:')) return imageUrl
  if (imageUrl.startsWith('/') && !imageUrl.startsWith('//')) return imageUrl
  try {
    const u = new URL(imageUrl)
    const path = `${u.pathname}${u.search}`
    if (!path.startsWith('/storage/')) return imageUrl
    if (/^(localhost|127\.0\.0\.1)$/i.test(u.hostname)) {
      return path
    }
  } catch {
    return imageUrl
  }
  return imageUrl
}

async function resolveImageBuffer(imageUrl) {
  const normalized = normalizeImageUrlForServerFetch(imageUrl)
  if (normalized.startsWith('data:')) {
    return parseDataUrl(normalized).buffer
  }
  const fetchUrl = normalized.startsWith('http')
    ? normalized
    : `http://127.0.0.1:${port}${normalized}`
  const response = await fetch(fetchUrl)
  if (!response.ok) throw new Error(`Could not fetch image: ${normalized}`)
  return Buffer.from(await response.arrayBuffer())
}

async function downscaleImageForInference(buffer, maxEdge = 1024) {
  const meta = await sharp(buffer).metadata()
  const width = Number(meta.width ?? 0)
  const height = Number(meta.height ?? 0)
  if (!width || !height || Math.max(width, height) <= maxEdge) return buffer
  return sharp(buffer)
    .resize({ width: maxEdge, height: maxEdge, fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 90 })
    .toBuffer()
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
  const lumMin = lum.length ? Math.min(...lum) : 0
  const lumMax = lum.length ? Math.max(...lum) : 0
  const luminanceSpread = lumMax - lumMin
  if (brightness < 16) return { ok: false, reason: 'Photo is too dark. Please retake in better lighting.' }
  // Mannequin / product shots are mostly white (#fff) with a small garment — mean brightness is
  // very high but luminance still spans dark fabric ↔ paper white. Old rule (mean > 248) rejected
  // those frames and broke post-mannequin re-detect, making the pipeline look like it "didn't create" anything.
  if (brightness > 248 && luminanceSpread < 40) {
    return { ok: false, reason: 'Photo is overexposed. Please lower brightness and retake.' }
  }
  if (brightness > 248 && luminanceSpread >= 40) {
    warnings.push('Very bright image (typical for garment on white). Continuing.')
  }
  if (brightness < 32) warnings.push('Photo is dark; color confidence may be lower.')
  if (brightness > 232 && !(brightness > 248 && luminanceSpread >= 40)) {
    warnings.push('Photo is bright; highlights may affect color precision.')
  }

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
    warnings.push('Photo is very blurry; garment detection may fail.')
  } else if (blurVariance < 75) {
    warnings.push('Photo has mild blur; detail confidence may be lower.')
  }

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

/**
 * Reject face detections that look like false positives on garment text/logos
 * or on print details. Real faces have:
 *   - aspect ratio ~0.6–1.5
 *   - reasonable size (>~1.5% of frame)
 *   - not tiny clusters sitting deep in the lower torso area of a cropped garment
 * Keeping this strict avoids blurring things like "CHANEL" text or chest prints.
 */
function isLikelyTrueFaceRegion(region, frameWidth, frameHeight) {
  if (!region || !frameWidth || !frameHeight) return false
  const w = Math.max(1, Number(region.width ?? 0))
  const h = Math.max(1, Number(region.height ?? 0))
  const areaFrac = (w * h) / (frameWidth * frameHeight)
  if (areaFrac < 0.005) return false
  if (areaFrac > 0.9) return false
  const aspect = w / h
  if (aspect < 0.45 || aspect > 1.9) return false
  return true
}

function filterValidFacePixelRegions(regions, frameWidth, frameHeight) {
  return (regions ?? []).filter((r) => isLikelyTrueFaceRegion(r, frameWidth, frameHeight))
}

function filterValidFaceCandidates(candidates, frameWidth, frameHeight) {
  return (candidates ?? []).filter((candidate) => {
    const bbox = candidate?.bbox
    if (!bbox) return false
    const w = Math.max(0, (bbox.x2 - bbox.x1)) * frameWidth
    const h = Math.max(0, (bbox.y2 - bbox.y1)) * frameHeight
    return isLikelyTrueFaceRegion({ width: w, height: h }, frameWidth, frameHeight)
  })
}

function normalizeBox(box) {
  return {
    x1: Math.max(0, Math.min(1, Number(box.x1 ?? 0))),
    y1: Math.max(0, Math.min(1, Number(box.y1 ?? 0))),
    x2: Math.max(0, Math.min(1, Number(box.x2 ?? 0))),
    y2: Math.max(0, Math.min(1, Number(box.y2 ?? 0))),
  }
}

function pixelRegionFromNormalizedBbox(bbox, width, height) {
  const x1 = Math.floor(normalizeBox(bbox).x1 * width)
  const y1 = Math.floor(normalizeBox(bbox).y1 * height)
  const x2 = Math.ceil(normalizeBox(bbox).x2 * width)
  const y2 = Math.ceil(normalizeBox(bbox).y2 * height)
  const left = Math.max(0, Math.min(width - 1, x1))
  const top = Math.max(0, Math.min(height - 1, y1))
  const regionWidth = Math.max(1, Math.min(width - left, x2 - left))
  const regionHeight = Math.max(1, Math.min(height - top, y2 - top))
  return { left, top, width: regionWidth, height: regionHeight }
}

async function detectResidualFaces(buffer) {
  const meta = await sharp(buffer).metadata()
  const width = Math.max(1, meta.width ?? 1)
  const height = Math.max(1, meta.height ?? 1)

  const googleFaces = await detectFaces(buffer)
  const googleRegions = googleFaces.map(extractBounds).filter(Boolean)
  const sidecarFaces = await callVisionSidecarFaces(buffer)
  const sidecarRegions = sidecarFaces
    .map((face) => face?.bbox)
    .filter(Boolean)
    .map((bbox) => pixelRegionFromNormalizedBbox(bbox, width, height))

  const all = [...googleRegions, ...sidecarRegions]
  const normalized = all.map((region) => ({
    x1: Number((region.left / width).toFixed(4)),
    y1: Number((region.top / height).toFixed(4)),
    x2: Number(((region.left + region.width) / width).toFixed(4)),
    y2: Number(((region.top + region.height) / height).toFixed(4)),
  }))
  return {
    count: normalized.length,
    regions: normalized,
  }
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

async function callVisionSidecarFaces(buffer) {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 5000)
    const response = await fetch(`${visionSidecarUrl}/faces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: buffer.toString('base64'),
      }),
      signal: controller.signal,
    })
    clearTimeout(timeout)
    if (!response.ok) return []
    const data = await response.json()
    if (!data?.ok || !Array.isArray(data?.faces)) return []
    return data.faces
  } catch {
    return []
  }
}

async function callVisionSidecarInfer(buffer) {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 45000)
    const response = await fetch(`${visionSidecarUrl}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: buffer.toString('base64'),
      }),
      signal: controller.signal,
    })
    clearTimeout(timeout)
    if (!response.ok) return null
    const data = await response.json()
    if (!data || data.ok !== true || typeof data.result !== 'object') return null
    return data.result
  } catch {
    return null
  }
}

async function callVisionSidecarSamSegment(buffer, boxes = []) {
  if (!Array.isArray(boxes) || boxes.length === 0) return []
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 6000)
    const response = await fetch(`${visionSidecarUrl}/sam/segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: buffer.toString('base64'),
        boxes,
        balanced: true,
      }),
      signal: controller.signal,
    })
    clearTimeout(timeout)
    if (!response.ok) return []
    const data = await response.json()
    if (!data?.ok || !Array.isArray(data?.segments)) return []
    return data.segments
  } catch {
    return []
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

function normalizedToPixelRegion(bbox, width, height) {
  if (!bbox || !width || !height) return null
  const x1 = Math.max(0, Math.min(1, Number(bbox.x1 ?? 0)))
  const y1 = Math.max(0, Math.min(1, Number(bbox.y1 ?? 0)))
  const x2 = Math.max(0, Math.min(1, Number(bbox.x2 ?? 0)))
  const y2 = Math.max(0, Math.min(1, Number(bbox.y2 ?? 0)))
  if (x2 <= x1 || y2 <= y1) return null
  const left = Math.max(0, Math.floor(x1 * width))
  const top = Math.max(0, Math.floor(y1 * height))
  const right = Math.min(width, Math.ceil(x2 * width))
  const bottom = Math.min(height, Math.ceil(y2 * height))
  if (right <= left || bottom <= top) return null
  return { left, top, width: right - left, height: bottom - top }
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
  const primaryCoverage = Number(primary?.coverage_pct ?? 0)
  // Do not force a light-neutral unless it is actually prominent.
  if (primaryCoverage < 0.32) return colors

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

async function blurFaceRegions(buffer, regions, blurSigma = 30) {
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
    const patch = await sharp(buffer).extract({ left, top, width, height }).blur(blurSigma).toBuffer()
    composites.push({ input: patch, left, top })
  }
  if (!composites.length) return { output: buffer, faceDetected: true, faceBlurApplied: false }
  const output = await sharp(buffer).composite(composites).jpeg({ quality: 82 }).toBuffer()
  return { output, faceDetected: true, faceBlurApplied: true }
}

function blurSigmaForPreset(preset = 'pro') {
  if (preset === 'soft') return 14
  if (preset === 'strong') return 34
  return 24
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

async function removeBackgroundForGarment(buffer) {
  // Lightweight background suppression: trims uniform margins after garment crop.
  // This does not do full segmentation but improves cluttered borders for inference.
  const trimmed = await sharp(buffer).trim({ threshold: 18 }).toBuffer()
  return sharp(trimmed).jpeg({ quality: 90 }).toBuffer()
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
      .map((bucket, idx) => ({ bucket, lab: centroids[idx] }))
      .filter(({ bucket }) => bucket.n > 0)
      .map(({ bucket, lab }) => {
        const r = bucket.r / bucket.n
        const g = bucket.g / bucket.n
        const b = bucket.b / bucket.n
        const hsl = rgbToHsl(r, g, b)
        const name = lab ? fashionColorNameFromLab(lab.L, lab.a, lab.b) : colorNameFromHsl(hsl)
        return {
          name,
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

const arbitrateHybridSchema = z.object({
  item_type: z.string().min(1),
  category: z.enum(CATEGORY_ENUM),
  confidence: z.number().min(0).max(1).optional(),
})

/**
 * Gemini tie-breaker when SigLIP / Florence / category vote disagree (upload infer only).
 * @param {Buffer} imageBuffer
 * @param {Record<string, unknown>} sidecarInfer
 */
async function arbitrateHybridDisagreement(imageBuffer, sidecarInfer) {
  const dbg = sidecarInfer?.metadata?.debug ?? {}
  const inferMime = await detectImageMime(imageBuffer)
  const prompt = `You arbitrate garment taxonomy when vision submodels disagree. Look at the clothing crop; return JSON only.
Context (may be wrong):
- SigLIP_top3: ${JSON.stringify(dbg.siglip_top3 ?? [])}
- Florence_labels: ${JSON.stringify(dbg.florence_labels ?? [])}
- Category_vote: ${JSON.stringify(dbg.category_vote ?? '')}
- Super_category: ${JSON.stringify(dbg.super_category ?? '')}
- Current_pick: item_type=${JSON.stringify(sidecarInfer.item_type)}, category=${JSON.stringify(sidecarInfer.category)}

Schema:
{"item_type":"single lowercase token like tshirt, jeans, dress, jacket, shorts, skirt, sweater, coat, blazer",
"category":"Tops|Bottoms|Outerwear|Shoes|Accessories",
"confidence":0-1}
Describe the primary visible garment only.`
  const base64 = imageBuffer.toString('base64')
  const parsed = await runGeminiJson(inferFlashModel, arbitrateHybridSchema, prompt, base64, {
    mimeType: inferMime,
    temperature: 0.2,
  })
  const it = parsed.item_type
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/-/g, '_')
  return { item_type: it, category: parsed.category }
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

function buildEmbeddingCacheKey(item, payload) {
  const attributes = typeof item?.raw_attributes === 'string' ? item.raw_attributes : JSON.stringify(item?.raw_attributes ?? {})
  return [
    normText(item?.item_type),
    normText(item?.category),
    normText(item?.color_primary),
    normText(item?.material),
    Number(item?.formality ?? 0),
    Array.isArray(item?.season) ? item.season.map((s) => normText(s)).sort().join('|') : '',
    attributes,
    payload,
  ].join('::')
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
  if (
    text.includes('pant') ||
    text.includes('jean') ||
    text.includes('trouser') ||
    text.includes('short') ||
    text.includes('legging') ||
    text.includes('jogger') ||
    text.includes('track pant') ||
    text.includes('sweatpant') ||
    text.includes('tights')
  ) return 'Bottoms'
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

function inferBottomSubtypeFromLabels(labels) {
  const text = labels.join(' ').toLowerCase()
  if (text.includes('short')) return 'Shorts'
  if (text.includes('legging') || text.includes('tights')) return 'Leggings'
  if (text.includes('jean')) return 'Jeans'
  if (text.includes('jogger') || text.includes('sweatpant') || text.includes('track pant')) return 'Joggers'
  if (text.includes('trouser') || text.includes('pant')) return 'Trousers'
  return 'Bottoms'
}

async function fallbackInferFromVision(imageBuffer) {
  const started = Date.now()
  let labels = []
  if (faceClient) {
    const [labelRes] = await faceClient.labelDetection({ image: { content: imageBuffer } })
    labels = (labelRes.labelAnnotations ?? []).map((l) => (l.description ?? '').trim()).filter(Boolean)
    try {
      const [objRes] = await faceClient.objectLocalization({ image: { content: imageBuffer } })
      const objectLabels = (objRes.localizedObjectAnnotations ?? [])
        .map((o) => (o.name ?? '').trim())
        .filter(Boolean)
      labels = [...labels, ...objectLabels]
    } catch {
      // keep fallback resilient if object localization fails
    }
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
  const meta = await sharp(imageBuffer).metadata()
  const width = Math.max(1, meta.width ?? 1)
  const height = Math.max(1, meta.height ?? 1)
  const aspect = width / height
  let inferredType = inferTypeFromCategory(category)
  if (category === 'Bottoms') {
    inferredType = inferBottomSubtypeFromLabels(labels)
    if (inferredType === 'Bottoms' && aspect >= 0.85) inferredType = 'Shorts'
  }
  return {
    schema_version: 2,
    item_type: inferredType,
    subtype: inferredType,
    category,
    color_primary,
    color_secondary: undefined,
    color_primary_hsl: primary_hsl,
    color_palette,
    dominant_colors: color_palette.map((c) => c.hex).slice(0, 3),
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

/** Expand degenerate bboxes (single click / open polyline) so mannequin masks stay valid. */
function axisAlignedBBoxWithMinSpan(vertices, minSpan = 0.02) {
  if (!vertices?.length) return null
  const box = normalizedBbox(vertices)
  let { x1, y1, x2, y2 } = box
  if (x2 - x1 < minSpan) {
    const cx = (x1 + x2) / 2
    x1 = Math.max(0, cx - minSpan / 2)
    x2 = Math.min(1, cx + minSpan / 2)
  }
  if (y2 - y1 < minSpan) {
    const cy = (y1 + y2) / 2
    y1 = Math.max(0, cy - minSpan / 2)
    y2 = Math.min(1, cy + minSpan / 2)
  }
  if (x2 <= x1) x2 = Math.min(1, x1 + minSpan)
  if (y2 <= y1) y2 = Math.min(1, y1 + minSpan)
  return { x1, y1, x2, y2 }
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

function buildPersonCandidates(personLike = []) {
  return personLike
    .map((obj, idx) => {
      const vertices = obj.boundingPoly?.normalizedVertices ?? []
      if (!vertices.length) return null
      return {
        id: `person-${idx + 1}`,
        label: obj.name ?? 'person',
        confidence: Number((obj.score ?? 0.5).toFixed(3)),
        bbox: normalizedBbox(vertices),
      }
    })
    .filter(Boolean)
}

function selectPersonBboxes(subjectFilter, personCandidates = []) {
  const selectedIds = new Set(subjectFilter?.selectedPersonIds ?? [])
  const selectedById = personCandidates
    .filter((candidate) => selectedIds.has(candidate.id))
    .map((candidate) => candidate.bbox)
  if (selectedById.length > 0) return selectedById
  if (Array.isArray(subjectFilter?.selectedPersonBboxes) && subjectFilter.selectedPersonBboxes.length > 0) {
    return subjectFilter.selectedPersonBboxes
  }
  return []
}

function assignGarmentsToPersons(garments = [], personCandidates = []) {
  if (!personCandidates.length || !garments.length) return []
  return garments.map((entry) => {
    const garmentBbox = entry.bbox
    let best = null
    for (const person of personCandidates) {
      const overlap = bboxOverlap(garmentBbox, person.bbox)
      const gx = (garmentBbox.x1 + garmentBbox.x2) / 2
      const gy = (garmentBbox.y1 + garmentBbox.y2) / 2
      const px = (person.bbox.x1 + person.bbox.x2) / 2
      const py = (person.bbox.y1 + person.bbox.y2) / 2
      const dist = Math.hypot(gx - px, gy - py)
      const centerScore = Math.max(0, 1 - dist / 1.2)
      const confidence = overlap * 0.7 + centerScore * 0.3
      if (!best || confidence > best.confidence) {
        best = { person_id: person.id, confidence: Number(confidence.toFixed(3)) }
      }
    }
    return {
      garment_id: entry.id,
      ...(best ?? { person_id: null, confidence: 0 }),
      requires_confirmation: !best || best.confidence < 0.2,
    }
  })
}

function shouldKeepGarmentBySubjectFilter(subjectFilter, garmentBbox, selectedPersonBboxes = []) {
  if (!subjectFilter) return true
  if (Array.isArray(subjectFilter.maskPolygon) && subjectFilter.maskPolygon.length >= 3) {
    const maskBbox = normalizedBbox(subjectFilter.maskPolygon)
    return bboxOverlap(maskBbox, garmentBbox) >= 0.12
  }
  if (subjectFilter.mode === 'clothing_only') return true
  if (selectedPersonBboxes.length === 0) return true
  return selectedPersonBboxes.some((personBbox) => bboxOverlap(personBbox, garmentBbox) >= 0.05)
}

function buildMaskSvg(width, height, regions = []) {
  const rects = regions.map((bbox) => {
    const x = Math.floor(Math.max(0, Math.min(1, bbox.x1)) * width)
    const y = Math.floor(Math.max(0, Math.min(1, bbox.y1)) * height)
    const w = Math.max(1, Math.ceil((Math.max(0, Math.min(1, bbox.x2)) - Math.max(0, Math.min(1, bbox.x1))) * width))
    const h = Math.max(1, Math.ceil((Math.max(0, Math.min(1, bbox.y2)) - Math.max(0, Math.min(1, bbox.y1))) * height))
    return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="white" />`
  }).join('')
  return Buffer.from(
    // IMPORTANT: keep background transparent (not black) so downstream `dest-in`
    // composites can rely on alpha. A solid black background is still opaque,
    // which previously caused `dest-in` to keep the whole base image.
    `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${rects}</svg>`,
  )
}

function buildPolygonMaskSvg(width, height, polygons = []) {
  const shapeSvg = polygons
    .map((poly) => {
      if (!Array.isArray(poly) || poly.length < 3) return ''
      const points = poly
        .map((p) => {
          const x = Math.round(Math.max(0, Math.min(1, Number(p.x ?? 0))) * width)
          const y = Math.round(Math.max(0, Math.min(1, Number(p.y ?? 0))) * height)
          return `${x},${y}`
        })
        .join(' ')
      return `<polygon points="${points}" fill="white" />`
    })
    .join('')
  return Buffer.from(
    // Same fix as buildMaskSvg: transparent background so the polygon defines
    // opaque-white alpha for mask-based compositing.
    `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${shapeSvg}</svg>`,
  )
}

async function applySubjectFilterToBuffer(buffer, subjectFilter, personCandidates, garmentBoxes) {
  if (!subjectFilter) return { buffer, warnings: [], applied: undefined }
  if (
    subjectFilter.mode !== 'focus_person_blur_others'
    && Array.isArray(subjectFilter.maskPolygon)
    && subjectFilter.maskPolygon.length >= 3
  ) {
    // Manual polygon should visibly keep selection and blur everything else.
    const meta = await sharp(buffer).metadata()
    const width = meta.width ?? 1
    const height = meta.height ?? 1
    const keepMask = buildPolygonMaskSvg(width, height, [subjectFilter.maskPolygon])
    const keptRegion = await sharp(buffer).composite([{ input: keepMask, blend: 'dest-in' }]).png().toBuffer()
    const blurredBase = await sharp(buffer).blur(20).jpeg({ quality: 90 }).toBuffer()
    const masked = await sharp(blurredBase)
      .composite([{ input: keptRegion, blend: 'over' }])
      .jpeg({ quality: 90 })
      .toBuffer()
    return {
      buffer: masked,
      warnings: [],
      applied: { ...subjectFilter },
    }
  }
  const meta = await sharp(buffer).metadata()
  const width = meta.width ?? 1
  const height = meta.height ?? 1
  const selectedPersonBboxes = selectPersonBboxes(subjectFilter, personCandidates)
  if (subjectFilter.mode === 'focus_person_blur_others') {
    if (!selectedPersonBboxes.length) {
      return {
        buffer,
        warnings: ['Focus mode needs at least one selected person.'],
        applied: { ...subjectFilter, selectedPersonBboxes },
      }
    }
    const samSegments = await callVisionSidecarSamSegment(buffer, selectedPersonBboxes)
    const samPolygons = samSegments
      .map((seg) => (Array.isArray(seg?.polygon) ? seg.polygon : []))
      .filter((poly) => poly.length >= 3)
    const blurred = await sharp(buffer).blur(18).jpeg({ quality: 90 }).toBuffer()
    let focused = null
    if (samPolygons.length > 0) {
      const keepMask = buildPolygonMaskSvg(width, height, samPolygons)
      const keptSubject = await sharp(buffer).composite([{ input: keepMask, blend: 'dest-in' }]).png().toBuffer()
      focused = await sharp(blurred).composite([{ input: keptSubject, blend: 'over' }]).jpeg({ quality: 90 }).toBuffer()
    } else {
      const composites = []
      for (const bbox of selectedPersonBboxes) {
        const left = Math.max(0, Math.floor(Math.max(0, Math.min(1, bbox.x1)) * width))
        const top = Math.max(0, Math.floor(Math.max(0, Math.min(1, bbox.y1)) * height))
        const right = Math.max(0, Math.ceil(Math.max(0, Math.min(1, bbox.x2)) * width))
        const bottom = Math.max(0, Math.ceil(Math.max(0, Math.min(1, bbox.y2)) * height))
        const boxW = Math.max(1, right - left)
        const boxH = Math.max(1, bottom - top)
        const patch = await sharp(buffer).extract({ left, top, width: boxW, height: boxH }).toBuffer()
        composites.push({ input: patch, left, top })
      }
      focused = await sharp(blurred).composite(composites).jpeg({ quality: 90 }).toBuffer()
    }
    return {
      buffer: focused,
      warnings: [],
      applied: { ...subjectFilter, selectedPersonBboxes },
    }
  }
  const garmentRegions = garmentBoxes.map((box) => box.bbox)
  const regions = subjectFilter.mode === 'clothing_only' ? garmentRegions : selectedPersonBboxes
  if (!regions.length) {
    return {
      buffer,
      warnings: ['Subject filter requested, but no selectable regions were found.'],
      applied: { ...subjectFilter, selectedPersonBboxes },
    }
  }
  const mask = buildMaskSvg(width, height, regions)
  let masked = null
  if (subjectFilter.mode === 'keep_selected_person') {
    const kept = await sharp(buffer).composite([{ input: mask, blend: 'dest-in' }]).png().toBuffer()
    const blurred = await sharp(buffer).blur(20).jpeg({ quality: 90 }).toBuffer()
    masked = await sharp(blurred).composite([{ input: kept, blend: 'over' }]).jpeg({ quality: 90 }).toBuffer()
  } else {
    masked = await sharp(buffer)
      .composite([{ input: mask, blend: 'dest-in' }])
      .flatten({ background: '#ffffff' })
      .jpeg({ quality: 90 })
      .toBuffer()
  }
  return {
    buffer: masked,
    warnings: [],
    applied: { ...subjectFilter, selectedPersonBboxes },
  }
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
    .png()
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
    const { imageUrl, subjectFilter, privacyPolicy } = detectRequestSchema.parse(req.body)
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
    const sidecarFaces = await callVisionSidecarFaces(buffer)
    const gcvFaceRegions = rawFaces.map(extractBounds).filter(Boolean)
    const sidecarFaceRegions = sidecarFaces
      .map((f) => normalizedToPixelRegion(f.bbox, w, h))
      .filter(Boolean)
    const faceRegionsRaw = sidecarFaceRegions.length > 0 ? sidecarFaceRegions : gcvFaceRegions
    // Reject likely false positives on logos/text before blurring.
    const faceRegions = filterValidFacePixelRegions(faceRegionsRaw, w, h)
    const faceCandidatesRaw = (sidecarFaces.length > 0
      ? sidecarFaces.map((f, idx) => ({
        id: `face-${idx + 1}`,
        confidence: Number(Number(f.confidence ?? 0.7).toFixed(3)),
        bbox: f.bbox,
        source: f.source ?? 'vision-sidecar',
      }))
      : gcvFaceRegions.map((r, idx) => ({
        id: `face-${idx + 1}`,
        confidence: 0.7,
        bbox: {
          x1: Number((r.left / w).toFixed(4)),
          y1: Number((r.top / h).toFixed(4)),
          x2: Number(((r.left + r.width) / w).toFixed(4)),
          y2: Number(((r.top + r.height) / h).toFixed(4)),
        },
        source: 'google-vision',
      })))
    const faceCandidates = filterValidFaceCandidates(faceCandidatesRaw, w, h)
    const selfieMode = faceRegions.length > 0 && (faceRegions.reduce((sum, r) => sum + (r.width * r.height), 0) / (w * h)) >= 0.25

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

    const forcePrivacy = (privacyPolicy ?? 'auto') === 'strict'
    const shouldBlur = (privacyPolicy ?? 'auto') !== 'off' && faceRegions.length > 0
    const { output: privacyBuffer, faceBlurApplied } =
      shouldBlur ? await blurFaceRegions(buffer, faceRegions) : { output: buffer, faceBlurApplied: false }

    const personLike = objectResult.filter((obj) => {
      const nameLc = (obj.name ?? '').toLowerCase()
      return /\bperson\b|people|^man$|^woman$|^boy$|^girl$|human/.test(nameLc)
    })
    const personCandidates = buildPersonCandidates(personLike)
    const selectedPersonBboxes = selectPersonBboxes(subjectFilter, personCandidates)
    const estimatedPersonCount = Math.max(
      Number(sidecar?.person_count ?? 0),
      personLike.length,
      faceRegions.length,
    )
    const multiPerson = estimatedPersonCount > 1

    const cx = 0.5
    const cy = 0.5

    const garmentObjects = objectResult
      .filter((obj) => {
        const nameLc = (obj.name ?? '').toLowerCase()
        return GARMENT_LABELS.some((hint) => nameLc.includes(hint))
      })
      .filter((obj) => (obj.score ?? 0) >= 0.32)
      .filter((obj) => {
        const vertices = obj.boundingPoly?.normalizedVertices ?? []
        if (!vertices.length) return false
        return shouldKeepGarmentBySubjectFilter(subjectFilter, normalizedBbox(vertices), selectedPersonBboxes)
      })

    // Deduplicate heavily overlapping boxes — keep higher-confidence one.
    const deduped = []
    for (const obj of garmentObjects) {
      const vertices = obj.boundingPoly?.normalizedVertices ?? []
      if (!vertices.length) continue
      const bbox = normalizedBbox(vertices)
    const overlapping = deduped.findIndex((d) => bboxOverlap(d.bbox, bbox) > 0.8)
      if (overlapping === -1) {
        deduped.push({ obj, bbox, vertices })
      } else if ((obj.score ?? 0) > (deduped[overlapping].obj.score ?? 0)) {
        deduped[overlapping] = { obj, bbox, vertices }
      }
    }

    // Score salience for each detected item.
    const { buffer: filteredBuffer, warnings: subjectFilteringWarnings, applied: appliedSubjectFilter } =
      await applySubjectFilterToBuffer(privacyBuffer, subjectFilter, personCandidates, deduped)
    // Keep detection stable/fast by default; opt in explicitly for try-off-first detect.
    const enableTryoffInDetect = process.env.VESTIR_TRYOFF_FIRST === '1'
    let sourceStage = 'blurred_fallback'
    let sourceBuffer = filteredBuffer
    if (enableTryoffInDetect) {
      const tryoff = await runTryoffWithBuffer(filteredBuffer, 'outfit')
      if (tryoff.implemented && tryoff.resultBuffer) {
        sourceBuffer = tryoff.resultBuffer
        sourceStage = 'tryoff'
      } else if (tryoff.message) {
        subjectFilteringWarnings.push(`Try-off fallback used: ${tryoff.message}`)
      }
    }
    const filteredMeta = await sharp(filteredBuffer).metadata()
    const scored = await Promise.all(
      deduped.map(async ({ obj, bbox, vertices }) => {
        const coverage = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        const itemCx = (bbox.x1 + bbox.x2) / 2
        const itemCy = (bbox.y1 + bbox.y2) / 2
        const dist = Math.sqrt((itemCx - cx) ** 2 + (itemCy - cy) ** 2)
        const centrality = Math.max(0, 1 - dist / 0.71)
        const cropBuffer = await cropAndIsolateGarment(filteredBuffer, vertices, filteredMeta)
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

        const filename = `${crypto.randomUUID()}.png`
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
    if (!fallback) {
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
          bbox: {
            x1: Number(entry.bbox.x1.toFixed(4)),
            y1: Number(entry.bbox.y1.toFixed(4)),
            x2: Number(entry.bbox.x2.toFixed(4)),
            y2: Number(entry.bbox.y2.toFixed(4)),
          },
          background_removed: false,
        }))
    }
    const personAssignments = assignGarmentsToPersons(items.map((it, idx) => ({ ...it, bbox: sorted[idx]?.bbox })), personCandidates)

    const faceCount = rawFaces.length
    const maxCoverage = sorted[0]?.coverage ?? 0
    const track = faceCount > 0 && maxCoverage > 0.18
      ? 'worn'
      : faceCount === 0 && maxCoverage > 0.14
        ? 'flat_lay'
        : 'ambiguous'

    const privacyApplied = faceBlurApplied
    let autoBlurUrl = null
    if (privacyApplied) {
      const autoBlurFilename = `${crypto.randomUUID()}.jpg`
      await fs.writeFile(path.join(processedDir, autoBlurFilename), privacyBuffer)
      autoBlurUrl = `/storage/processed/${autoBlurFilename}`
    }
    const sourceFilename = `${crypto.randomUUID()}.jpg`
    await fs.writeFile(path.join(processedDir, sourceFilename), sourceBuffer)
    const warnings = [
      ...(validation.warnings ?? []),
      ...(selfieMode ? ['Selfie-mode activated: face-dominant frame, garment salience re-weighted.'] : []),
      ...(multiPerson ? ['Multiple people in frame — crops are per detected garment; verify each selection.'] : []),
      ...(personAssignments.some((a) => a.requires_confirmation)
        ? ['Some garment-to-person links are low confidence. Confirm selections in the review sheet.']
        : []),
      ...(sidecar ? [] : ['Vision sidecar unavailable — advanced garment parsing skipped.']),
      ...subjectFilteringWarnings,
      ...(fallback ? ['No garment found in this photo. Try closer framing or better lighting.'] : []),
    ]

    res.json({
      detected: items,
      person_candidates: personCandidates,
      person_assignments: personAssignments,
      face_candidates: faceCandidates,
      scene_track: track,
      source_image_url: `/storage/processed/${sourceFilename}`,
      source_image_stage: sourceStage,
      auto_blurred_image_url: autoBlurUrl,
      manual_blur_required: (forcePrivacy || faceCandidates.length > 0) && !faceBlurApplied,
      applied_subject_filter: appliedSubjectFilter,
      warnings,
      pipeline: {
        architecture_id: PIPELINE_ARCHITECTURE_ID,
        stages: [
          { id: 'image_filtering', status: 'completed', detail: 'validateInputImage' },
          {
            id: 'subject_filtering',
            status: subjectFilter ? 'completed' : 'skipped',
            detail: subjectFilter ? `mode=${subjectFilter.mode}` : 'not requested',
          },
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
            id: 'face_detection',
            status: faceCandidates.length > 0 ? 'completed' : 'partial',
            detail: faceCandidates.length > 0 ? 'retinaface(hf)/fallback' : 'no faces detected',
          },
          {
            id: 'auto_blur',
            status: privacyApplied ? 'completed' : 'skipped',
            detail: privacyApplied ? 'automatic face blur applied' : 'no blur needed',
          },
          {
            id: 'tryoff_extraction',
            status: sourceStage === 'tryoff' ? 'completed' : 'partial',
            detail: sourceStage === 'tryoff' ? 'fal virtual tryoff lora' : 'fallback to blurred source',
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

app.get('/api/health', async (_req, res) => {
  let tryonProbe = null
  if (tryonSidecarUrl) {
    try {
      const ctrl = new AbortController()
      const t = setTimeout(() => ctrl.abort(), 2500)
      const r = await fetch(`${tryonSidecarUrl}/health`, { signal: ctrl.signal })
      clearTimeout(t)
      if (r.ok) tryonProbe = await r.json()
    } catch {
      tryonProbe = null
    }
  }
  res.json({
    ok: true,
    services: {
      gemini: Boolean(genai),
      gemini_api_key_source: genai ? geminiApiKeySource : undefined,
      faceDetection: Boolean(faceClient),
      visionSidecar: Boolean(visionSidecarUrl),
      tryonSidecar: Boolean(tryonSidecarUrl),
      tryoffPipelineReady: tryonProbe?.tryoff_pipeline_ready ?? null,
      tryoffWarmupError: tryonProbe?.tryoff_warmup_error ?? null,
      tryoffMode: tryonProbe?.mode ?? null,
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

app.post('/api/items/blur-manual', async (req, res) => {
  try {
    const body = manualBlurRequestSchema.parse(req.body)
    const started = Date.now()
    const buffer = await resolveImageBuffer(body.imageUrl)
    const meta = await sharp(buffer).metadata()
    const width = meta.width ?? 0
    const height = meta.height ?? 0
    const presetSigma = blurSigmaForPreset(body.blurPreset ?? 'pro')
    const blur = body.blurAmount && body.blurAmount % 2 === 1 ? body.blurAmount : presetSigma
    const boxRegions = (body.boxes ?? [])
      .map((b) => normalizedToPixelRegion(b, width, height))
      .filter(Boolean)
    const polygonRegion =
      Array.isArray(body.maskPolygon) && body.maskPolygon.length >= 3
        ? normalizedToPixelRegion(normalizedBbox(body.maskPolygon), width, height)
        : null
    const regions = polygonRegion ? [...boxRegions, polygonRegion] : boxRegions
    const { output, faceBlurApplied } = await blurFaceRegions(buffer, regions, blur)
    const filename = `${crypto.randomUUID()}.jpg`
    await fs.writeFile(path.join(processedDir, filename), output)
    return res.json({
      blurredImageUrl: `/storage/processed/${filename}`,
      faceBlurApplied,
      regionsCount: regions.length,
      metadata: {
        provider: 'server-manual-blur',
        latency_ms: Date.now() - started,
        version: '1.0.0',
        blurPreset: body.blurPreset ?? 'pro',
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Manual blur failed'
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.message, code: 'VALIDATION' })
    }
    return res.status(400).json({ error: message })
  }
})

function garmentTargetFromMannequinLabels(labels) {
  if (!Array.isArray(labels) || labels.length === 0) return 'outfit'
  const haystack = labels.join(' ').toLowerCase()
  if (/\b(pants|jeans|trousers?)\b/.test(haystack)) return 'pants'
  if (/\b(dress|gown)\b/.test(haystack)) return 'dress'
  if (/\b(jacket|coat|blazer|hoodie|outerwear)\b/.test(haystack)) return 'jacket'
  if (/\b(shirt|tee|t-?shirt|kurta|top|blouse|sweater)\b/.test(haystack)) return 'tshirt'
  return 'outfit'
}

function rgbToApproxColorName(r, g, b) {
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 510
  const d = max - min || 1e-6
  if (l < 0.08) return 'black'
  if (l > 0.92 && d < 30) return 'white'
  if (d < 22) return l > 0.55 ? 'light gray' : 'charcoal'
  let h = 0
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6
  else if (max === g) h = ((b - r) / d + 2) / 6
  else h = ((r - g) / d + 4) / 6
  if (h < 0.02 || h >= 0.98) return l < 0.45 ? 'deep red' : 'red'
  if (h < 0.08) return 'orange'
  if (h < 0.17) return 'yellow'
  if (h < 0.22) return 'olive'
  if (h < 0.33) return 'green'
  if (h < 0.45) return 'teal'
  if (h < 0.55) return l < 0.42 ? 'navy blue' : 'blue'
  if (h < 0.7) return 'indigo'
  if (h < 0.85) return 'purple'
  return l > 0.55 ? 'pink' : 'magenta'
}

async function averageRgbFromImageBuffer(buffer) {
  try {
    const { data, info } = await sharp(buffer)
      .resize(64, 64, { fit: 'inside' })
      .flatten({ background: { r: 255, g: 255, b: 255 } })
      .raw()
      .toBuffer({ resolveWithObject: true })
    const ch = info.channels
    let r = 0
    let g = 0
    let b = 0
    const n = Math.max(1, data.length / ch)
    for (let i = 0; i < data.length; i += ch) {
      r += data[i]
      g += data[i + 1]
      b += data[i + 2]
    }
    return {
      r: Math.round(r / n),
      g: Math.round(g / n),
      b: Math.round(b / n),
    }
  } catch {
    return { r: 96, g: 96, b: 96 }
  }
}

function buildMannequinCatalogAttributes(labels, hints, rgb, colorName) {
  const category =
    (hints?.category && String(hints.category).trim()) ||
    (Array.isArray(labels) && labels[0] ? String(labels[0]).toLowerCase().replace(/\s+/g, '_') : 'garment')
  const hex = `#${[rgb.r, rgb.g, rgb.b].map((v) => v.toString(16).padStart(2, '0')).join('')}`
  const detailsRaw = hints?.details ? String(hints.details) : ''
  const details = detailsRaw
    ? detailsRaw.split(',').map((s) => s.trim()).filter(Boolean)
    : []
  return {
    category,
    color: hints?.color || colorName,
    color_hex_sample: hex,
    collar: hints?.collar ?? null,
    sleeves: hints?.sleeves ?? null,
    placket: hints?.placket ?? null,
    fit: hints?.fit ?? 'regular',
    fabric: hints?.fabric ?? null,
    details,
    vision_labels: Array.isArray(labels) ? labels : [],
  }
}

function buildMannequinGenerationPrompt(attrs) {
  const cat = attrs.category !== 'garment' ? String(attrs.category).replace(/_/g, ' ') : 'garment'
  const parts = [
    'Studio product photograph of',
    attrs.color || 'the garment',
    cat,
  ]
  if (attrs.collar) parts.push(`${attrs.collar} collar`)
  if (attrs.sleeves) parts.push(`${attrs.sleeves} sleeves`)
  if (attrs.placket) parts.push(String(attrs.placket))
  if (attrs.fabric) parts.push(`${attrs.fabric} fabric`)
  if (Array.isArray(attrs.details) && attrs.details.length) parts.push(attrs.details.join(', '))
  parts.push(
    'on an invisible neutral mannequin, symmetrical front-facing catalog pose, pure white #FFFFFF background, soft diffused studio lighting, soft shadows, high-end e-commerce, no human face or identity, no logos or readable text',
  )
  return parts.filter(Boolean).join(', ') + '.'
}

async function preprocessSourceForMannequin(buffer) {
  if (process.env.MANNEQUIN_PREPROCESS === '0') return buffer
  try {
    return await sharp(buffer)
      .removeAlpha()
      .normalize({ lower: 2, upper: 98 })
      .modulate({ saturation: 1.03, brightness: 1.01 })
      .toBuffer()
  } catch {
    try {
      return await sharp(buffer).normalize({ lower: 2, upper: 98 }).toBuffer()
    } catch {
      return buffer
    }
  }
}

async function postprocessMannequinBuffer(buffer) {
  if (process.env.MANNEQUIN_POSTPROCESS === '0') return buffer
  try {
    return await sharp(buffer)
      .sharpen({ sigma: 0.65, m1: 1.2, m2: 0.35 })
      .jpeg({ quality: 92 })
      .toBuffer()
  } catch {
    return buffer
  }
}

app.post('/api/items/mannequin', async (req, res) => {
  try {
    const parsed = mannequinRequestSchema.safeParse(req.body)
    if (!parsed.success) {
      return res.status(400).json({
        error: 'Invalid mannequin request',
        code: 'VALIDATION',
        issues: parsed.error.issues,
      })
    }
    const body = parsed.data
    const started = Date.now()
    const rawSource = await resolveImageBuffer(body.imageUrl)
    const source = await preprocessSourceForMannequin(rawSource)
    const meta = await sharp(source).metadata()
    const width = meta.width ?? 1
    const height = meta.height ?? 1
    const selectedBboxes = Array.isArray(body.subjectFilter?.selectedPersonBboxes)
      ? body.subjectFilter.selectedPersonBboxes
      : []
    const poly = Array.isArray(body.subjectFilter?.maskPolygon) ? body.subjectFilter.maskPolygon : []
    const stages = []
    stages.push({
      id: 'ingest_preprocess',
      status: process.env.MANNEQUIN_PREPROCESS === '0' ? 'skipped' : 'completed',
      detail: 'normalize + mild saturation for detection-friendly contrast',
    })
    let keepMask = null
    if (poly.length >= 3) {
      keepMask = buildPolygonMaskSvg(width, height, [poly])
    } else if (poly.length >= 1 && poly.length < 3) {
      const thin = axisAlignedBBoxWithMinSpan(poly)
      if (thin) keepMask = buildMaskSvg(width, height, [thin])
    } else if (selectedBboxes.length > 0) {
      keepMask = buildMaskSvg(width, height, selectedBboxes)
    }
    if (!keepMask) {
      return res.status(400).json({
        error:
          'Select a region/person before mannequin generation. Send subjectFilter.maskPolygon (1+ points) and/or selectedPersonBboxes.',
        code: 'MANNEQUIN_NO_REGION',
        received: {
          hasSubjectFilter: Boolean(body.subjectFilter),
          maskPointCount: poly.length,
          bboxCount: selectedBboxes.length,
        },
      })
    }
    stages.push({
      id: 'human_garment_region',
      status: 'completed',
      detail: poly.length >= 3 ? 'SAM-style polygon mask' : 'axis-aligned subject region',
    })
    const isolated = await sharp(source).composite([{ input: keepMask, blend: 'dest-in' }]).png().toBuffer()
    const cutoutRgb = await sharp({
      create: {
        width,
        height,
        channels: 3,
        background: '#ffffff',
      },
    })
      .composite([{ input: isolated, blend: 'over' }])
      .jpeg({ quality: 92 })
      .toBuffer()

    const ctx = body.context ?? {}
    const labels = ctx.selectedLabels ?? []
    const hints = ctx.attributeHints ?? {}
    const rgb = await averageRgbFromImageBuffer(isolated)
    const colorName = rgbToApproxColorName(rgb.r, rgb.g, rgb.b)
    const catalogAttributes = buildMannequinCatalogAttributes(labels, hints, rgb, colorName)
    const generationPrompt = buildMannequinGenerationPrompt(catalogAttributes)
    stages.push({
      id: 'semantic_attribute_encoding',
      status: 'completed',
      detail: 'structured catalog fields + dominant color sample from masked crop',
    })
    stages.push({
      id: 'prompt_construction',
      status: 'completed',
      detail: 'internal studio prompt for optional diffusion refine',
    })

    let outBuffer = cutoutRgb
    let diffusionRefined = false
    const useTryoff = process.env.MANNEQUIN_USE_TRYOFF?.trim() === '1'
    const garmentTarget = ctx.garmentTarget ?? garmentTargetFromMannequinLabels(labels)
    if (useTryoff) {
      stages.push({
        id: 'generative_reconstruction',
        status: 'partial',
        detail: 'calling try-on sidecar /tryoff with attribute-aware prompt',
      })
      const mannequinTryoffTimeoutMs = Number(process.env.MANNEQUIN_TRYOFF_TIMEOUT_MS ?? '90000')
      const tryoff = await runTryoffWithBuffer(cutoutRgb, garmentTarget, undefined, {
        promptOverride: generationPrompt,
        timeoutMs: Number.isFinite(mannequinTryoffTimeoutMs) && mannequinTryoffTimeoutMs >= 0
          ? mannequinTryoffTimeoutMs
          : 90000,
      })
      if (tryoff.implemented && tryoff.resultBuffer) {
        outBuffer = tryoff.resultBuffer
        diffusionRefined = true
        stages[stages.length - 1] = {
          id: 'generative_reconstruction',
          status: 'completed',
          detail: 'diffusion refine (TRYOFF) applied on white-background cutout',
        }
      } else {
        stages[stages.length - 1] = {
          id: 'generative_reconstruction',
          status: 'skipped',
          detail: String(tryoff.message ?? 'tryoff unavailable — sharp cutout retained'),
        }
      }
    } else {
      stages.push({
        id: 'generative_reconstruction',
        status: 'skipped',
        detail: 'MANNEQUIN_USE_TRYOFF not set to 1 — pixel-accurate masked composite only',
      })
    }

    stages.push({
      id: 'mannequin_composition',
      status: 'completed',
      detail: 'neutral white field, identity removed outside mask, standard front-facing crop',
    })

    const mannequin = await postprocessMannequinBuffer(outBuffer)
    stages.push({
      id: 'post_processing',
      status: process.env.MANNEQUIN_POSTPROCESS === '0' ? 'skipped' : 'completed',
      detail: 'sharpen + JPEG encode (background already #FFFFFF)',
    })

    const filename = `${crypto.randomUUID()}.jpg`
    await fs.writeFile(path.join(processedDir, filename), mannequin)
    return res.json({
      mannequinImageUrl: `/storage/processed/${filename}`,
      pipeline: {
        version: '2.0.0',
        architecture_id: PIPELINE_ARCHITECTURE_ID,
        stages,
        catalog_attributes: catalogAttributes,
        generation_prompt: generationPrompt,
        diffusion_refined: diffusionRefined,
        garment_target: garmentTarget,
        latency_ms: Date.now() - started,
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Mannequin generation failed'
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.message, code: 'VALIDATION' })
    }
    return res.status(400).json({ error: message })
  }
})

// Fully generative (text-to-image only) mannequin pipeline.
// Pipeline: image → caption → attributes → prompt → SDXL/DALL-E → rank → post.
// If attribute_confidence is low OR every candidate is rejected, the sidecar
// returns fallback_recommended=true and this route transparently reroutes to
// the legacy segmentation pipeline (/api/items/mannequin).
const mannequinV2RequestSchema = z.object({
  imageUrl: z.string().min(1),
  seed: z.number().int().optional(),
  scoreThreshold: z.number().min(0).max(1).optional(),
  numCandidates: z.number().int().min(1).max(8).optional(),
  useCache: z.boolean().optional(),
  variant: z.string().optional(),
  allowLegacyFallback: z.boolean().optional(),
})

async function callLegacyMannequinFallback(imageUrl, reason) {
  const response = await fetch(`http://127.0.0.1:${process.env.PORT || 8787}/api/items/mannequin`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUrl, fallbackReason: reason }),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`legacy_fallback_http_${response.status}: ${text.slice(0, 400)}`)
  }
  return response.json()
}

app.post('/api/items/mannequin-v2', async (req, res) => {
  const started = Date.now()
  try {
    const body = mannequinV2RequestSchema.parse(req.body)
    const allowFallback = body.allowLegacyFallback ?? true
    const buffer = await resolveImageBuffer(body.imageUrl)
    const timeoutMs = Number(process.env.MANNEQUIN_V2_TIMEOUT_MS ?? '180000')
    const controller = new AbortController()
    const timeoutId = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null
    let sidecarJson
    try {
      const response = await fetch(`${tryonSidecarUrl}/mannequin_v2`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: buffer.toString('base64'),
          seed: body.seed,
          score_threshold: body.scoreThreshold,
          num_candidates: body.numCandidates,
          use_cache: body.useCache ?? true,
          variant: body.variant ?? 'v2.1-textonly',
        }),
        signal: controller.signal,
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(`sidecar_http_${response.status}: ${text.slice(0, 400)}`)
      }
      sidecarJson = await response.json()
    } finally {
      if (timeoutId) clearTimeout(timeoutId)
    }

    console.info('[mannequin-v2] pipeline stages=%s conf=%s score=%s fallback=%s',
      JSON.stringify(sidecarJson?.stages?.map?.((s) => [s.id, s.status, s.duration_ms]) ?? []),
      sidecarJson?.attribute_confidence,
      sidecarJson?.score,
      sidecarJson?.fallback_recommended,
    )

    // Transparent fallback to legacy SAM pipeline when generative path bailed.
    if (sidecarJson?.fallback_recommended && allowFallback) {
      try {
        const legacy = await callLegacyMannequinFallback(
          body.imageUrl, sidecarJson.fallback_reason || 'v2_fallback',
        )
        return res.json({
          mannequinImageUrl: legacy.mannequinImageUrl,
          pipeline: {
            version: 'v2.0.0-generative + legacy_fallback',
            architecture_id: 'caption-normalize-prompt-sdxl-score-post',
            used_path: 'legacy_sam_fallback',
            fallback_reason: sidecarJson.fallback_reason,
            attribute_confidence: sidecarJson.attribute_confidence,
            attributes: sidecarJson.attributes,
            notes: sidecarJson.notes,
            stages: sidecarJson.stages,
            legacy_pipeline: legacy.pipeline,
            latency_ms: Date.now() - started,
          },
        })
      } catch (fallbackErr) {
        console.warn('[mannequin-v2] legacy fallback failed:', fallbackErr)
        return res.status(502).json({
          error: 'both_pipelines_failed',
          fallback_error: String(fallbackErr),
          pipeline: sidecarJson,
        })
      }
    }

    if (!sidecarJson?.has_image || !sidecarJson?.result_image_base64) {
      return res.status(502).json({
        error: sidecarJson?.fallback_reason || 'mannequin_v2_failed',
        pipeline: sidecarJson,
      })
    }

    const outBuffer = Buffer.from(sidecarJson.result_image_base64, 'base64')
    const filename = `${crypto.randomUUID()}.jpg`
    await fs.writeFile(path.join(processedDir, filename), outBuffer)

    return res.json({
      mannequinImageUrl: `/storage/processed/${filename}`,
      pipeline: {
        version: 'v2.0.0-generative',
        architecture_id: 'caption-normalize-prompt-sdxl-score-post',
        used_path: 'text_to_image',
        total_ms: sidecarJson.total_ms,
        latency_ms: Date.now() - started,
        cache_hit: sidecarJson.cache_hit,
        score: sidecarJson.score,
        attribute_confidence: sidecarJson.attribute_confidence,
        attributes: sidecarJson.attributes,
        prompt: sidecarJson.prompt,
        negative_prompt: sidecarJson.negative_prompt,
        candidates: sidecarJson.candidates,
        selected_index: sidecarJson.selected_index,
        stages: sidecarJson.stages,
        notes: sidecarJson.notes,
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Mannequin v2 generation failed'
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.message, code: 'VALIDATION' })
    }
    return res.status(500).json({ error: message })
  }
})

app.post('/api/items/sam/refine', async (req, res) => {
  try {
    const body = samRefineRequestSchema.parse(req.body)
    const buffer = await resolveImageBuffer(body.imageUrl)
    const segments = await callVisionSidecarSamSegment(buffer, body.boxes)
    return res.json({
      ok: true,
      polygons: segments
        .map((seg) => (Array.isArray(seg?.polygon) ? seg.polygon : []))
        .filter((poly) => poly.length >= 3),
      segments,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'SAM refinement failed'
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.message, code: 'VALIDATION' })
    }
    return res.status(400).json({ error: message })
  }
})

async function runTryoffWithBuffer(imageBuffer, garmentTarget = 'outfit', seed = undefined, tryoffOptions = undefined) {
  const controller = new AbortController()
  // FLUX try-off on CPU/MPS can exceed 5–20+ minutes; default 2h. Set TRYOFF_TIMEOUT_MS=0 to disable abort.
  const configuredTimeoutMs = Number(process.env.TRYOFF_TIMEOUT_MS ?? process.env.TRYON_TIMEOUT_MS ?? '7200000')
  const requestedTimeoutMs =
    Number.isFinite(Number(tryoffOptions?.timeoutMs)) && Number(tryoffOptions?.timeoutMs) >= 0
      ? Number(tryoffOptions?.timeoutMs)
      : undefined
  const timeoutMs = requestedTimeoutMs ?? configuredTimeoutMs
  const timeoutId =
    Number.isFinite(timeoutMs) && timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null
  const promptOverride =
    typeof tryoffOptions?.promptOverride === 'string' && tryoffOptions.promptOverride.trim()
      ? tryoffOptions.promptOverride.trim()
      : undefined
  try {
    const sidecarResponse = await fetch(`${tryonSidecarUrl}/tryoff`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: imageBuffer.toString('base64'),
        garment_target: garmentTarget,
        seed,
        ...(promptOverride ? { prompt_override: promptOverride } : {}),
      }),
      signal: controller.signal,
    })
    if (!sidecarResponse.ok) {
      return { implemented: false, message: `Try-off sidecar HTTP ${sidecarResponse.status}`, resultBuffer: null }
    }
    const data = await sidecarResponse.json()
    if (!data?.ok) {
      return { implemented: false, message: String(data?.error ?? 'Try-off sidecar error'), resultBuffer: null }
    }
    if (!data.implemented || !data.result_image_base64) {
      const msg = [data.message, data.debug_error].filter(Boolean).join(' | ')
      return { implemented: false, message: String(msg || 'Try-off not configured'), resultBuffer: null }
    }
    return {
      implemented: true,
      message: null,
      resultBuffer: Buffer.from(String(data.result_image_base64), 'base64'),
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : 'tryoff fetch failed'
    const timedOut =
      err instanceof Error
      && (/abort/i.test(message) || message === 'This operation was aborted')
    const hint = timedOut
      ? ' (timed out — set TRYOFF_TIMEOUT_MS=0 for no limit, or raise TRYOFF_TIMEOUT_MS in .env)'
      : ''
    return {
      implemented: false,
      message: `Try-off sidecar unreachable (${tryonSidecarUrl}): ${message}${hint}`,
      resultBuffer: null,
    }
  } finally {
    if (timeoutId) clearTimeout(timeoutId)
  }
}

app.post('/api/items/tryoff', async (req, res) => {
  try {
    const body = tryoffRequestSchema.parse(req.body)
    const started = Date.now()
    const imageBuf = await resolveImageBuffer(body.imageUrl)
    const tryoff = await runTryoffWithBuffer(imageBuf, body.garmentTarget, body.seed)
    if (!tryoff.implemented || !tryoff.resultBuffer) {
      return res.json({
        implemented: false,
        tryoffImageUrl: null,
        message: String(tryoff.message ?? 'Try-off not configured or model unavailable'),
        metadata: { provider: 'tryon-sidecar', latency_ms: Date.now() - started, version: '1.0.0' },
      })
    }

    const filename = `${crypto.randomUUID()}.jpg`
    const diskPath = path.join(processedDir, filename)
    await fs.writeFile(diskPath, tryoff.resultBuffer)
    console.log(
      `[tryoff] saved ${diskPath} (${tryoff.resultBuffer.length} bytes, ${Date.now() - started}ms latency)`,
    )
    return res.json({
      implemented: true,
      tryoffImageUrl: `/storage/processed/${filename}`,
      message: null,
      metadata: {
        provider: 'tryon-sidecar',
        latency_ms: Date.now() - started,
        version: '1.0.0',
        garment_target: body.garmentTarget,
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Try-off failed'
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.message, code: 'VALIDATION' })
    }
    res.status(400).json({ error: message })
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
    const { imageUrl, maskPolygon, subjectFilter, removeBackground, privacyPolicy } = preprocessRequestSchema.parse(req.body)
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
              { id: 'subject_filtering', status: 'skipped', detail: 'already filtered at detect/upload' },
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
              { id: 'subject_filtering', status: 'completed', detail: 'manual mask polygon' },
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
    const personCandidates = Array.isArray(sidecar?.garments)
      ? []
      : []
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
    const metaPre = await sharp(buffer).metadata()
    const preW = metaPre.width ?? 1
    const preH = metaPre.height ?? 1
    const regionsRaw = rawFaces.map(extractBounds).filter(Boolean)
    // Reject likely false positives (chest logos, text prints) before blur.
    const regions = filterValidFacePixelRegions(regionsRaw, preW, preH)
    const facesBeforeBlur = regions.length
    const shouldBlurFaces = (privacyPolicy ?? 'auto') !== 'off' && ((track === 'worn' || privacyPolicy === 'strict') && regions.length > 0)
    const { output, faceDetected, faceBlurApplied } = shouldBlurFaces
      ? await blurFaceRegions(buffer, regions)
      : { output: buffer, faceDetected: rawFaces.length > 0, faceBlurApplied: false }
    const residualFaceScan = await detectResidualFaces(output)
    const needsManualPrivacyReview = residualFaceScan.count > 0
    const garmentBoxes = garmentRegions.map((region) => ({ bbox: normalizedBbox(region.vertices) }))
    const { buffer: subjectFilteredBuffer, warnings: subjectFilterWarnings } =
      await applySubjectFilterToBuffer(output, subjectFilter, personCandidates, garmentBoxes)

    // Crop to garment region (if available) to create clean subject for downstream color + inference.
    const garmentRegion = primaryRegion?.vertices ?? await detectGarmentRegion(subjectFilteredBuffer)
    const cropped = await cropToRegion(subjectFilteredBuffer, garmentRegion)
    const shouldRemoveBackground = typeof removeBackground === 'boolean'
      ? removeBackground
      : track === 'flat_lay'
    const bgAdjusted = shouldRemoveBackground ? await removeBackgroundForGarment(cropped) : cropped
    // Keep inference payloads small for faster Gemini calls and lower token cost.
    const optimized = await downscaleImageForInference(bgAdjusted, 1024)
    const filename = `${crypto.randomUUID()}.jpg`
    const filepath = path.join(processedDir, filename)
    await fs.writeFile(filepath, optimized)
    res.json({
      processedImageUrl: `/storage/processed/${filename}`,
      faceDetected,
      faceBlurApplied,
      facesBeforeBlur,
      facesAfterBlur: residualFaceScan.count,
      residualFaceRegions: residualFaceScan.regions,
      needsManualPrivacyReview,
      personDetected: Math.max(rawFaces.length, Number(sidecar?.person_count ?? 0)) > 0,
      garmentIsolated: Boolean(garmentRegion),
      backgroundRemoved: shouldRemoveBackground,
      scene_track: track,
      scene: {
        face_count: rawFaces.length,
        garment_coverage: Number(garmentCoverage.toFixed(3)),
      },
      validation: validation.metrics,
      warnings: [
        ...(validation.warnings ?? []),
        ...(sceneWarning ? [sceneWarning] : []),
        ...subjectFilterWarnings,
        ...(needsManualPrivacyReview ? ['Residual face regions detected; apply manual blur before finalizing.'] : []),
      ],
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
              id: 'subject_filtering',
              status: subjectFilter ? 'completed' : 'skipped',
              detail: subjectFilter ? `mode=${subjectFilter.mode}` : 'not requested',
            },
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
              status: shouldBlurFaces
                ? (needsManualPrivacyReview ? 'partial' : 'completed')
                : 'skipped',
              detail: shouldBlurFaces
                ? (needsManualPrivacyReview ? 'auto blur applied; residual faces remain' : 'blur faces after scene track')
                : undefined,
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
    const { processedImageUrl, sourceImageStage, forceGemini, skipArbitration } = inferRequestSchema.parse(req.body)
    const started = Date.now()
    const imageResponse = await fetch(
      processedImageUrl.startsWith('http')
        ? processedImageUrl
        : `http://127.0.0.1:${port}${processedImageUrl}`,
    )
    if (!imageResponse.ok) throw new Error('Unable to read processed image')
    const imageBuffer = Buffer.from(await imageResponse.arrayBuffer())

    // Optional hybrid backend from vision-sidecar (Florence + fashion-SigLIP core).
    // Modes:
    // - hybrid_sidecar: require sidecar inference
    // - auto: try sidecar first, then Gemini pipeline
    // - gemini: skip sidecar and use Gemini-only path
    if (!forceGemini && (inferBackend === 'hybrid_sidecar' || inferBackend === 'auto')) {
      const sidecarInfer = await callVisionSidecarInfer(imageBuffer)
      if (sidecarInfer) {
        let out = { ...sidecarInfer }
        const dbg = sidecarInfer?.metadata?.debug ?? {}
        const shouldArb =
          !skipArbitration &&
          inferArbitrateMode === 'gemini_on_disagreement' &&
          genai &&
          dbg.attribute_disagreement === true
        if (shouldArb) {
          try {
            const arbBuf = await downscaleImageForInference(imageBuffer, 768)
            const arb = await arbitrateHybridDisagreement(arbBuf, sidecarInfer)
            const prevU = sidecarInfer.uncertainty ?? {}
            const lowConf = Number(sidecarInfer.confidence_overall ?? 0) < 0.6
            out = {
              ...sidecarInfer,
              item_type: arb.item_type,
              subtype: arb.item_type,
              category: arb.category,
              uncertainty: {
                ...prevU,
                attribute_disagreement: false,
                blocks_embedding: false,
                requires_user_confirmation: lowConf,
                uncertain_fields: lowConf ? ['item_type'] : [],
                arbitration_applied: true,
              },
              metadata: {
                ...(sidecarInfer.metadata ?? {}),
                debug: {
                  ...dbg,
                  gemini_arbitration: { item_type: arb.item_type, category: arb.category },
                },
              },
            }
          } catch {
            // Keep original sidecar payload if arbitration fails
          }
        }
        return res.json({
          ...out,
          metadata: {
            ...(out.metadata ?? {}),
            latency_ms: Date.now() - started,
            pipeline: {
              architecture_id: PIPELINE_ARCHITECTURE_ID,
              stages: [
                { id: 'image_filtering', status: 'skipped', detail: 'handled in /preprocess' },
                { id: 'human_detection', status: 'skipped', detail: 'handled in /detect' },
                { id: 'human_parsing', status: 'completed', detail: 'vision-sidecar hybrid parsing' },
                { id: 'privacy_masking', status: 'skipped', detail: 'handled in /preprocess' },
                { id: 'clothing_extraction', status: 'skipped', detail: 'handled in /preprocess' },
                {
                  id: 'attribute_inference',
                  status: 'completed',
                  detail: 'vision-sidecar Florence + fashion-SigLIP + LAB palette',
                },
                { id: 'post_processing', status: 'completed', detail: 'server schema alignment' },
              ],
            },
          },
        })
      }
      if (inferBackend === 'hybrid_sidecar') {
        return res.status(502).json({
          error: 'Hybrid sidecar inference unavailable. Check VISION_SIDECAR_URL and sidecar model setup.',
        })
      }
    }

    if (!genai) {
      const fallback = await fallbackInferFromVision(imageBuffer)
      return res.json({
        ...fallback,
        warning: 'Gemini unavailable and sidecar infer unavailable; heuristic fallback used.',
      })
    }

    const inferenceBuffer = await downscaleImageForInference(imageBuffer, 1024)
    const base64Image = inferenceBuffer.toString('base64')
    const inferMime = await detectImageMime(inferenceBuffer)

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
      const shouldRunSemanticPass =
        quality.accepted && quality.blur_score >= 0.3 && quality.lighting_score >= 0.3 && quality.occlusion_visible_pct >= 0.35
      let semantic
      if (shouldRunSemanticPass) {
        const semanticModel =
          process.env.INFER_SEMANTIC_USE_PRO === '1' && useProFirst ? inferProModel : inferFlashModel
        semantic = await runGeminiJson(semanticModel, semanticSchema, semanticPrompt, base64Image, {
          mimeType: inferMime,
          temperature: 0.35,
        })
      } else {
        quality.warnings.push('Skipped deep style/occasion pass due to low image quality or visibility.')
        semantic = {
          formality: 5,
          seasons: { spring: 0.6, summer: 0.6, autumn: 0.6, winter: 0.6 },
          occasions: ['casual'],
          style_archetype: 'smart_casual',
          layering_role: 'standalone',
          pairings: [],
          confidence: {
            formality: 0.45,
            occasions: 0.42,
            style_archetype: 0.4,
            seasonality: 0.45,
          },
        }
      }

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
      let colorsBase = mergedColors.length ? mergedColors : modelColors
      // For try-off outputs, white background can dominate. If the top color is neutral
      // and we have a substantial non-neutral candidate, promote the garment-like color.
      if (sourceImageStage === 'tryoff' && colorsBase.length > 1) {
        const first = colorsBase[0]
        const alt = colorsBase.find((c) => !isNeutralHsl(c.hsl) && Number(c.coverage_pct ?? 0) >= 0.1)
        if (first && isNeutralHsl(first.hsl) && alt) {
          colorsBase = [alt, ...colorsBase.filter((c) => c !== alt)]
        }
      }
      const shouldPreferLightNeutrals = quality.framing !== 'worn' && sourceImageStage !== 'tryoff'
      const colors = shouldPreferLightNeutrals
        ? prioritizeLightNeutralsPrimary(colorsBase)
        : colorsBase

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
      const advancedDesignTags = [
        structural.pattern,
        structural.fit && structural.fit !== 'unknown' ? structural.fit : null,
        resolvedGarmentType,
        structural.subtype,
        resolvedCategory,
        structural.material.primary,
        semantic.style_archetype,
        semantic.layering_role ?? 'standalone',
        ...(semantic.occasions ?? []),
        ...season,
        ...(Array.isArray(structural.construction_details) ? structural.construction_details : []),
        ...(colors.slice(0, 3).map((c) => c.name)),
      ]
        .map((tag) => String(tag ?? '').trim().toLowerCase())
        .filter((tag) => tag.length > 1 && tag !== 'unknown')
        .filter((tag, idx, arr) => arr.indexOf(tag) === idx)
        .slice(0, 24)
      const advancedStyleNote = [
        `style=${semantic.style_archetype}`,
        `layering=${semantic.layering_role ?? 'standalone'}`,
        `formality=${semantic.formality}/10`,
        `material=${structural.material.primary}`,
        `pattern=${structural.pattern}`,
        `fit=${structural.fit ?? 'unknown'}`,
        `season=${season.join('/')}`,
        `occasions=${(semantic.occasions ?? []).slice(0, 4).join(',') || 'n/a'}`,
      ].join('; ')

      const parsed = inferSchema.parse({
        schema_version: 2,
        item_type: resolvedGarmentType,
        subtype: structural.subtype,
        category: resolvedCategory,
        color_primary: colors[0]?.name ?? 'Taupe',
        color_secondary: colors[1]?.name,
        color_primary_hsl: colors[0]?.hsl ?? hslFromColorName('Taupe'),
        color_secondary_hsl: colors[1]?.hsl,
        color_palette: colors.map(({ confidence, ...c }) => c),
        dominant_colors: colors.map((c) => c.hex).slice(0, 3),
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
        source_image_stage: sourceImageStage ?? 'blurred_fallback',
        gemini_style_notes: advancedStyleNote,
        gemini_design_tags: advancedDesignTags,
        gemini_brand_like: [],
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
                id: 'attribute_detection',
                status: 'completed',
                detail: 'OpenCV palette + SigLIP/Fusion signals',
              },
              {
                id: 'gemini_enrichment',
                status: 'completed',
                detail: 'design/style/occasion enrichment (advisory)',
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

app.post('/api/items/analyze-advanced-local', async (req, res) => {
  try {
    const { imageUrl, context } = advancedLocalRequestSchema.parse(req.body)
    const started = Date.now()
    const imageBuffer = await resolveImageBuffer(imageUrl)
    const inferenceBuffer = await downscaleImageForInference(imageBuffer, 1024)
    const prompt = `You are a fashion attribute parser. Return strict JSON only.
Schema:
{
  "item_type": "string",
  "category": "${CATEGORY_ENUM.join('|')}",
  "color_primary": "string",
  "material": "string",
  "pattern": "string",
  "confidence_overall": 0-1,
  "design_tags": ["string"],
  "style_notes": "string",
  "style_tags": ["string"],
  "occasions": ["string"]
}
Rules:
- Focus only on the primary garment in frame.
- Prefer precise garment names (e.g. vest dress, wide leg jeans) over generic terms.
- If uncertain, keep confidence low and avoid hallucinations.
- Return valid JSON only.
Existing context: ${JSON.stringify(context ?? {})}`

    const mlx = await runGemmaMlxVision({
      imageBuffer: inferenceBuffer,
      prompt,
      timeoutMs: Number(process.env.GEMMA_MLX_TIMEOUT_MS ?? 40000),
    })
    if (!mlx.ok) {
      return res.status(503).json({
        ok: false,
        error: mlx.error ?? 'Local Gemma MLX unavailable/slow. Try smaller model or warmup.',
        metadata: {
          provider: 'gemma-mlx-fallback',
          model: mlx.model,
          latency_ms: Date.now() - started,
          version: '1.0.0',
        },
      })
    }

    const parsed = advancedLocalGemmaSchema.parse(mlx.result)
    const patch = {
      ...(parsed.item_type ? { item_type: parsed.item_type.trim() } : {}),
      ...(parsed.category ? { category: parsed.category } : {}),
      ...(parsed.color_primary ? { color_primary: parsed.color_primary.trim() } : {}),
      ...(parsed.material ? { material: parsed.material.trim() } : {}),
      ...(parsed.pattern ? { pattern: parsed.pattern.trim() } : {}),
      ...(parsed.style_tags?.length ? { style_tags: parsed.style_tags } : {}),
      ...(parsed.occasions?.length ? { occasions: parsed.occasions } : {}),
    }

    return res.json({
      ok: true,
      advanced: {
        patch,
        confidence_overall: parsed.confidence_overall ?? 0.6,
        design_tags: parsed.design_tags ?? [],
        style_notes: parsed.style_notes ?? '',
        raw: mlx.result,
      },
      metadata: {
        provider: 'gemma-mlx',
        model: mlx.model,
        latency_ms: Date.now() - started,
        version: '1.0.0',
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Local advanced analysis failed'
    return res.status(400).json({ ok: false, error: message })
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
    const cacheKey = buildEmbeddingCacheKey(item, payload)
    const existing = await readEmbeddingsArray()
    const cached = [...existing].reverse().find((entry) => (
      entry?.item_id === item.id &&
      typeof entry?.cache_key === 'string' &&
      entry.cache_key === cacheKey &&
      Array.isArray(entry?.vector) &&
      entry.vector.length > 0
    ))
    if (cached) {
      return res.json({
        vector_id: cached.id,
        dimensions: Array.isArray(cached.vector) ? cached.vector.length : 0,
        metadata: {
          provider: 'cache',
          model: cached.model ?? 'cached',
          latency_ms: Date.now() - started,
          version: '1.2.0',
        },
        cached: true,
      })
    }
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
    existing.push({
      id: vectorId,
      item_id: item.id,
      cache_key: cacheKey,
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

app.post('/api/wardrobe/items/upsert', async (req, res) => {
  try {
    const { item, mannequinImageUrl, attributes } = wardrobeItemUpsertSchema.parse(req.body)
    const started = Date.now()
    if (!item || typeof item !== 'object') {
      return res.status(400).json({ error: 'item is required' })
    }
    const id = String(item.id ?? '').trim() || `item-${crypto.randomUUID()}`
    const normalized = {
      ...item,
      id,
      raw_attributes: typeof item.raw_attributes === 'string'
        ? item.raw_attributes
        : JSON.stringify(item.raw_attributes ?? attributes ?? {}),
      mannequin_image_url: mannequinImageUrl ?? item.mannequin_image_url ?? null,
      updated_at: new Date().toISOString(),
    }
    const embedding = await ensureEmbeddingForItem(normalized)
    const items = await readJsonArrayFile(wardrobeItemsFile)
    const next = [...items.filter((x) => String(x?.id) !== id), {
      ...normalized,
      embedding_vector_id: embedding.id,
      embedding_model: embedding.model ?? null,
      embedding_dim: Array.isArray(embedding.vector) ? embedding.vector.length : 0,
      created_at: items.some((x) => String(x?.id) === id)
        ? items.find((x) => String(x?.id) === id)?.created_at ?? new Date().toISOString()
        : new Date().toISOString(),
    }]
    await writeJsonArrayFile(wardrobeItemsFile, next)
    return res.json({
      ok: true,
      item_id: id,
      embedding: {
        vector_id: embedding.id,
        model: embedding.model ?? null,
        dim: Array.isArray(embedding.vector) ? embedding.vector.length : 0,
      },
      metadata: { latency_ms: Date.now() - started, version: '1.0.0' },
    })
  } catch (error) {
    return res.status(400).json({ error: error instanceof Error ? error.message : 'wardrobe upsert failed' })
  }
})

app.post('/api/wardrobe/compatibility', async (req, res) => {
  try {
    const { itemA, itemB, useLlm } = wardrobeCompatibilitySchema.parse(req.body)
    const started = Date.now()
    const embA = await ensureEmbeddingForItem(itemA)
    const embB = await ensureEmbeddingForItem(itemB)
    const heuristic = computeCompatibilityHeuristic(itemA, itemB, embA.vector, embB.vector)
    let score = heuristic.score
    let explanation = explainHeuristicPairing(itemA, itemB, score)
    let llm = null
    if (useLlm ?? true) {
      llm = await llmCompatibilityAdjustment(itemA, itemB)
      if (llm) {
        score = Math.max(0, Math.min(1, heuristic.score * 0.55 + llm.score * 0.45))
        explanation = llm.explanation
      }
    }
    const record = {
      id: crypto.randomUUID(),
      item1: String(itemA?.id ?? ''),
      item2: String(itemB?.id ?? ''),
      score: Number(score.toFixed(4)),
      breakdown: heuristic.breakdown,
      llm_score: llm ? Number(llm.score.toFixed(4)) : null,
      explanation,
      created_at: new Date().toISOString(),
    }
    const existing = await readJsonArrayFile(wardrobeCompatibilityFile)
    existing.push(record)
    await writeJsonArrayFile(wardrobeCompatibilityFile, existing.slice(-5000))
    return res.json({
      compatibility: record,
      metadata: { latency_ms: Date.now() - started, version: '1.0.0' },
    })
  } catch (error) {
    return res.status(400).json({ error: error instanceof Error ? error.message : 'compatibility failed' })
  }
})

/**
 * Compose a full outfit (top + bottom + shoes + optional outerwear/accessory)
 * from the anchor using top-K alternatives per slot, then rank each composed
 * outfit with a Finish My Fit v3–style aggregation (min*0.6 + avg*0.4) over
 * every pair in the outfit. Adds profile/weather boost and an Ollama
 * explanation when available.
 */
app.post('/api/wardrobe/outfits/build', async (req, res) => {
  try {
    const body = wardrobeOutfitBuildSchema.parse(req.body)
    const started = Date.now()
    const topK = body.topK ?? 3
    const persisted = await readJsonArrayFile(wardrobeItemsFile)
    const mergedMap = new Map()
    for (const it of persisted) mergedMap.set(String(it?.id), it)
    for (const it of body.wardrobeItems ?? []) mergedMap.set(String(it?.id), it)
    const allItems = [...mergedMap.values()].filter((x) => x && typeof x === 'object')
    const anchor =
      body.anchorItem ?? allItems.find((x) => String(x?.id) === String(body.anchorItemId ?? ''))
    if (!anchor) {
      return res
        .status(400)
        .json({ error: 'anchor item not found; provide anchorItem or valid anchorItemId' })
    }

    const anchorCategory = String(anchor?.category ?? 'Tops')
    const required = body.requireCategories ?? defaultRequiredCategoriesForAnchor(anchorCategory, body.weather?.mode)
    const pool = allItems.filter((x) => String(x?.id) !== String(anchor?.id) && !x?.deleted_at)

    const embAnchor = await ensureEmbeddingForItem(anchor)
    if (!Array.isArray(embAnchor?.vector) || embAnchor.vector.length === 0) {
      return res.status(400).json({ error: 'anchor embedding unavailable; re-run item processing' })
    }

    // Per-slot candidates sorted by anchor compatibility (cheap first pass).
    const slotCandidates = {}
    for (const slot of required) {
      if (slot === anchorCategory) continue
      const bySlot = pool.filter((x) => String(x?.category) === slot)
      const scored = []
      for (const cand of bySlot) {
        try {
          const emb = await ensureEmbeddingForItem(cand)
          if (!Array.isArray(emb?.vector) || emb.vector.length !== embAnchor.vector.length) continue
          const heuristic = computeCompatibilityHeuristic(anchor, cand, embAnchor.vector, emb.vector)
          const boosted = applyProfileAndWeatherBoost(heuristic.score, anchor, cand, body.profile, body.weather)
          scored.push({ item: cand, anchorScore: boosted })
        } catch {
          // skip broken candidate
        }
      }
      scored.sort((a, b) => b.anchorScore - a.anchorScore)
      // Beam width per slot: enough variety but bounded to keep combinations sane.
      slotCandidates[slot] = scored.slice(0, 5)
    }

    // Generate outfit compositions via cartesian product of top-K per slot.
    const slotsOrdered = required.filter((s) => s !== anchorCategory)
    const compositions = cartesianSlots(slotsOrdered, slotCandidates).slice(0, 40)

    // Aggregate scores across every pair in each composed outfit.
    const outfitEvals = []
    for (const comp of compositions) {
      const pieces = [anchor, ...comp.map((c) => c.item)]
      const pairs = []
      for (let i = 0; i < pieces.length; i++) {
        for (let j = i + 1; j < pieces.length; j++) {
          try {
            const embI = await ensureEmbeddingForItem(pieces[i])
            const embJ = await ensureEmbeddingForItem(pieces[j])
            if (
              !Array.isArray(embI?.vector) ||
              !Array.isArray(embJ?.vector) ||
              embI.vector.length !== embJ.vector.length
            )
              continue
            const h = computeCompatibilityHeuristic(pieces[i], pieces[j], embI.vector, embJ.vector)
            const b = applyProfileAndWeatherBoost(h.score, pieces[i], pieces[j], body.profile, body.weather)
            pairs.push(b)
          } catch {
            // skip broken pair
          }
        }
      }
      if (!pairs.length) continue
      const min = Math.min(...pairs)
      const avg = pairs.reduce((a, b) => a + b, 0) / pairs.length
      const outfitScore = Math.max(0, Math.min(1, min * 0.6 + avg * 0.4))
      outfitEvals.push({ pieces, outfitScore, min, avg })
    }

    outfitEvals.sort((a, b) => b.outfitScore - a.outfitScore)
    const topOutfits = outfitEvals.slice(0, topK)

    // LLM explanations (single call per outfit; fall back to heuristic).
    const useLlm = body.useLlm ?? true
    const outfits = []
    let usedLlm = false
    for (let i = 0; i < topOutfits.length; i++) {
      const ev = topOutfits[i]
      let explanation = explainOutfitHeuristic(ev.pieces, ev.outfitScore, body.profile, body.weather)
      if (useLlm) {
        const llmText = await llmOutfitExplanation(ev.pieces, body.profile, body.weather)
        if (llmText) {
          explanation = llmText
          usedLlm = true
        }
      }
      outfits.push({
        rank: i + 1,
        score: Number(ev.outfitScore.toFixed(4)),
        min_pair_score: Number(ev.min.toFixed(4)),
        avg_pair_score: Number(ev.avg.toFixed(4)),
        explanation,
        pieces: ev.pieces,
        piece_ids: ev.pieces.map((p) => String(p.id)),
      })
    }

    const source = usedLlm ? 'ollama+heuristic' : 'heuristic-only'
    return res.json({
      anchor_item: anchor,
      outfits,
      summary: outfits.length
        ? `Composed ${outfits.length} ranked outfit${outfits.length > 1 ? 's' : ''} from your wardrobe.`
        : 'Not enough wardrobe items to compose a full outfit yet.',
      metadata: {
        provider: 'vestir-server',
        model: usedLlm ? `blend(${ollamaModel})` : 'heuristic',
        source,
        required_slots: required,
        compositions_considered: compositions.length,
        candidate_pool: pool.length,
        latency_ms: Date.now() - started,
        version: '2.0.0',
      },
    })
  } catch (error) {
    return res.status(400).json({ error: error instanceof Error ? error.message : 'outfit generation failed' })
  }
})

function defaultRequiredCategoriesForAnchor(anchorCategory, weatherMode) {
  const base = new Set(['Tops', 'Bottoms', 'Shoes'])
  base.add(anchorCategory)
  if ((weatherMode ?? 'all') === 'cold') base.add('Outerwear')
  return [...base]
}

function cartesianSlots(slots, slotCandidates) {
  let acc = [[]]
  for (const slot of slots) {
    const options = slotCandidates[slot] ?? []
    if (!options.length) return []
    const next = []
    for (const partial of acc) {
      for (const opt of options) {
        next.push([...partial, { slot, ...opt }])
      }
    }
    acc = next
  }
  return acc
}

function explainOutfitHeuristic(pieces, score, profile, weather) {
  const names = pieces.map((p) => `${p.color_primary ?? ''} ${p.item_type ?? p.category}`.trim()).slice(0, 4)
  const intent = profile?.style_intent ?? 'balanced'
  const mode = weather?.mode ?? 'all'
  const strength =
    score >= 0.75 ? 'Strong' : score >= 0.6 ? 'Solid' : score >= 0.45 ? 'Workable' : 'Edgy'
  return `${strength} ${intent} look for ${mode === 'all' ? 'everyday' : mode + ' weather'}: ${names.join(' + ')}.`
}

async function llmOutfitExplanation(pieces, profile, weather) {
  if (!genai) return null
  const payload = pieces.map((p) => ({
    category: p.category,
    item_type: p.item_type,
    color_primary: p.color_primary,
    material: p.material,
    pattern: p.pattern,
    fit: p.fit,
    formality: p.formality,
  }))
  const prompt = `Return strict JSON only.
Write one concise styling sentence (<=28 words) explaining why this outfit works, mentioning color harmony or formality alignment when relevant.
Profile: ${JSON.stringify(profile ?? {})}
Weather: ${JSON.stringify(weather ?? {})}
Outfit: ${JSON.stringify(payload)}

Schema: {"explanation": "string"}`
  try {
    const out = await genai.models.generateContent({ model: inferFlashModel, contents: prompt })
    const raw = String(out?.text ?? '').trim()
    const parsed = JSON.parse(extractJsonObject(raw))
    const text = String(parsed?.explanation ?? '').trim()
    return text || null
  } catch {
    return null
  }
}

function applyProfileAndWeatherBoost(baseScore, anchor, candidate, profile, weather) {
  let score = baseScore
  const intent = String(profile?.style_intent ?? 'balanced').toLowerCase()
  if (intent === 'formal' && Number(candidate?.formality ?? 5) >= 7) score += 0.06
  if (intent === 'casual' && Number(candidate?.formality ?? 5) <= 5) score += 0.06
  if (intent === 'bold' && (candidate?.pattern || (candidate?.style_tags?.length ?? 0) > 1)) score += 0.05
  const mode = String(weather?.mode ?? 'all').toLowerCase()
  const season = Array.isArray(candidate?.season) ? candidate.season.map((s) => String(s).toLowerCase()) : []
  if (mode === 'warm' && (season.includes('summer') || season.includes('spring'))) score += 0.05
  if (mode === 'cold' && (season.includes('winter') || season.includes('autumn'))) score += 0.05
  return Math.max(0, Math.min(1, score))
}

app.post('/api/wardrobe/suggest-next', async (req, res) => {
  try {
    const body = wardrobeSuggestNextSchema.parse(req.body)
    const started = Date.now()
    const persisted = await readJsonArrayFile(wardrobeItemsFile)
    const mergedMap = new Map()
    for (const it of persisted) mergedMap.set(String(it?.id), it)
    for (const it of body.wardrobeItems ?? []) mergedMap.set(String(it?.id), it)
    const anchor = body.anchorItem
    const allItems = [...mergedMap.values()].filter((x) => x && typeof x === 'object' && String(x?.id) !== String(anchor?.id))
    const embAnchor = await ensureEmbeddingForItem(anchor)
    const scored = []
    for (const cand of allItems) {
      try {
        const emb = await ensureEmbeddingForItem(cand)
        if (!Array.isArray(emb.vector) || emb.vector.length !== embAnchor.vector.length) continue
        const heuristic = computeCompatibilityHeuristic(anchor, cand, embAnchor.vector, emb.vector)
        let finalScore = heuristic.score
        let explanation = explainHeuristicPairing(anchor, cand, heuristic.score)
        let llmScore = null
        if (body.useLlm ?? true) {
          const llm = await llmCompatibilityAdjustment(anchor, cand)
          if (llm) {
            llmScore = llm.score
            finalScore = Math.max(0, Math.min(1, heuristic.score * 0.6 + llm.score * 0.4))
            explanation = llm.explanation
          }
        }
        const boosted = applyProfileAndWeatherBoost(finalScore, anchor, cand, body.profile, body.weather)
        scored.push({
          item_id: String(cand.id),
          score: Number(boosted.toFixed(4)),
          explanation,
          llm_score: llmScore == null ? null : Number(llmScore.toFixed(4)),
        })
      } catch {
        // skip bad candidate
      }
    }
    scored.sort((a, b) => b.score - a.score)
    const suggestions = scored.slice(0, body.topK ?? 6)
    const source = suggestions.some((s) => s.llm_score != null) ? 'ollama+heuristic' : 'heuristic-only'
    return res.json({
      summary: suggestions.length
        ? 'Recommended next items are ranked for profile, aesthetic intent, and weather context.'
        : 'No strong next-item match found yet; add more wardrobe items for richer suggestions.',
      suggestions: suggestions.map(({ item_id, score, explanation }) => ({ item_id, score, explanation })),
      metadata: {
        provider: 'vestir-server',
        model: source === 'ollama+heuristic' ? `blend(${ollamaModel})` : 'heuristic',
        source,
        latency_ms: Date.now() - started,
        version: '1.0.0',
      },
    })
  } catch (error) {
    return res.status(400).json({ error: error instanceof Error ? error.message : 'suggest-next failed' })
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
      body: JSON.stringify({ processedImageUrl: preprocessJson.processedImageUrl, skipArbitration: true }),
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
    let queryVector = []
    let queryEmbeddingProvider = 'deterministic-fallback'
    let queryEmbeddingModel = 'deterministic-fallback'
    if (embeddingSidecarUrl) {
      try {
        const imageRes = await fetch(
          preprocessJson.processedImageUrl.startsWith('http')
            ? preprocessJson.processedImageUrl
            : `http://127.0.0.1:${port}${preprocessJson.processedImageUrl}`,
        )
        const imageBase64 = imageRes.ok ? Buffer.from(await imageRes.arrayBuffer()).toString('base64') : undefined
        const sidecarRes = await fetch(`${embeddingSidecarUrl.replace(/\/$/, '')}/embed`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: payload, image_base64: imageBase64 }),
          signal: AbortSignal.timeout(20000),
        })
        if (sidecarRes.ok) {
          const data = await sidecarRes.json()
          const vec = data.vector ?? data.values ?? data.embedding
          if (Array.isArray(vec) && vec.length) {
            queryVector = vec
            queryEmbeddingProvider = 'embedding-sidecar'
            queryEmbeddingModel = typeof data.model === 'string' ? data.model : 'embedding-sidecar'
          }
        }
      } catch {
        // Fall through to other providers.
      }
    }
    if (!queryVector.length && genai) {
      const gem = await genai.models.embedContent({
        model: embeddingModel,
        contents: payload,
      })
      const vec = gem.embeddings?.[0]?.values ?? []
      if (vec.length) {
        queryVector = vec
        queryEmbeddingProvider = 'google'
        queryEmbeddingModel = embeddingModel
      }
    }
    if (!queryVector.length) {
      queryVector = buildDeterministicFallbackVector(payload, 256)
      queryEmbeddingProvider = 'server'
      queryEmbeddingModel = 'deterministic-fallback'
    }

    const entries = await readEmbeddingsArray()
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
        provider: queryEmbeddingProvider,
        model: queryEmbeddingModel,
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
  // Node 18+ defaults requestTimeout to 5m, which kills long routes (e.g. FLUX try-off).
  try {
    server.requestTimeout = 0
  } catch {
    /* older Node */
  }
  server.timeout = 0
  server.headersTimeout = Math.max(server.headersTimeout ?? 0, 120_000)

  await new Promise((resolve, reject) => {
    server.on('close', resolve)
    server.on('error', reject)
  })
}

startServer().catch((error) => {
  console.error('API server failed:', error)
  process.exit(1)
})
