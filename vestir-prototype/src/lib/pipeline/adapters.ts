import type {
  EmbeddingResult,
  InferenceResult,
  OutfitBuildResult,
  PipelineAdapters,
  PostPipelineSuggestionResult,
  PreprocessResult,
  ReasoningResult,
  StyleProfile,
  WeatherContext,
} from './contracts'
import type { BlurQualityPreset, DetectionResult, FitLabel, Item, NormalizedBBox, NormalizedPoint, SubjectFilterConfig } from '../../types/index'
import { FIT_LABELS } from '../../types/index'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function readJsonOrThrow(response: Response, fallbackMessage: string) {
  const text = await response.text()
  if (!text) {
    throw new Error(`${fallbackMessage}: empty response body`)
  }
  try {
    return JSON.parse(text) as Record<string, unknown>
  } catch {
    throw new Error(`${fallbackMessage}: non-JSON response`)
  }
}

export async function detectItemsFromImage(imageUrl: string, subjectFilter?: SubjectFilterConfig): Promise<DetectionResult> {
  const response = await fetch(`${API_BASE}/api/items/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUrl, subjectFilter }),
  })
  const data = await readJsonOrThrow(response, 'Detection request failed')
  if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : 'Detection failed')
  return data as unknown as DetectionResult
}

export type TryoffExtractionResult = {
  implemented: boolean
  tryoffImageUrl: string | null
  message?: string | null
}

export type ManualBlurResult = {
  blurredImageUrl: string | null
  faceBlurApplied: boolean
  regionsCount: number
}

export type MannequinContext = {
  selectedLabels?: string[]
  attributeHints?: Record<string, string>
  garmentTarget?: 'outfit' | 'ensemble' | 'tshirt' | 'dress' | 'pants' | 'jacket'
}

export type MannequinPipelineMetadata = {
  version?: string
  architecture_id?: string
  stages?: Array<{ id: string; status: string; detail?: string }>
  catalog_attributes?: Record<string, unknown>
  generation_prompt?: string
  diffusion_refined?: boolean
  garment_target?: string
  latency_ms?: number
}

export type MannequinResult = {
  mannequinImageUrl: string | null
  pipeline?: MannequinPipelineMetadata
}

export async function refineMaskWithSam(
  imageUrl: string,
  boxes: NormalizedBBox[],
): Promise<Array<Array<{ x: number; y: number }>> | null> {
  const response = await fetch(`${API_BASE}/api/items/sam/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUrl, boxes }),
  })
  const data = await readJsonOrThrow(response, 'SAM refine request failed')
  if (!response.ok) {
    throw new Error(typeof data.error === 'string' ? data.error : 'SAM refine request failed')
  }
  return Array.isArray(data.polygons) ? (data.polygons as Array<Array<{ x: number; y: number }>>) : null
}

export async function generateMannequinImage(
  imageUrl: string,
  subjectFilter: SubjectFilterConfig,
  context?: MannequinContext,
): Promise<MannequinResult> {
  const response = await fetch(`${API_BASE}/api/items/mannequin`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      imageUrl,
      subjectFilter,
      ...(context ? { context } : {}),
    }),
  })
  const data = await readJsonOrThrow(response, 'Mannequin request failed')
  if (!response.ok) {
    let detail =
      typeof data.error === 'string'
        ? data.error
        : `Mannequin request failed (HTTP ${response.status})`
    if (Array.isArray(data.issues) && data.issues.length) {
      detail += ` — ${JSON.stringify(data.issues)}`
    }
    if (data.code === 'MANNEQUIN_NO_REGION' && data.received) {
      detail += ` — ${JSON.stringify(data.received)}`
    }
    throw new Error(detail)
  }
  const mannequinImageUrl = typeof data.mannequinImageUrl === 'string' ? data.mannequinImageUrl : null
  if (!mannequinImageUrl) {
    throw new Error('Server returned success but no mannequinImageUrl')
  }
  const pipeline =
    data.pipeline && typeof data.pipeline === 'object'
      ? (data.pipeline as MannequinPipelineMetadata)
      : undefined
  return { mannequinImageUrl, pipeline }
}

type ManualBlurRequest = {
  boxes: NormalizedBBox[]
  maskPolygon?: Array<{ x: number; y: number }>
  blurPreset?: BlurQualityPreset
}

/** Virtual try-off: worn photo → garment on white (FLUX + LoRA). Used before /detect when enabled. */
export type TryoffGarmentTarget =
  | 'outfit'
  | 'ensemble'
  | 'tshirt'
  | 'shirt'
  | 'dress'
  | 'pants'
  | 'jacket'

export async function runTryoffExtraction(
  imageUrl: string,
  garmentTarget: TryoffGarmentTarget = 'outfit',
): Promise<TryoffExtractionResult> {
  const mappedTarget = garmentTarget === 'shirt' ? 'tshirt' : garmentTarget
  const response = await fetch(`${API_BASE}/api/items/tryoff`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUrl, garmentTarget: mappedTarget }),
  })
  const data = await readJsonOrThrow(response, 'Try-off request failed')
  if (!response.ok) {
    return {
      implemented: false,
      tryoffImageUrl: null,
      message: typeof data.error === 'string' ? data.error : 'Try-off request failed',
    }
  }
  return {
    implemented: Boolean(data.implemented),
    tryoffImageUrl: typeof data.tryoffImageUrl === 'string' ? data.tryoffImageUrl : null,
    message: typeof data.message === 'string' ? data.message : null,
  }
}

export async function applyManualBlur(
  imageUrl: string,
  request: ManualBlurRequest,
): Promise<ManualBlurResult> {
  const response = await fetch(`${API_BASE}/api/items/blur-manual`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUrl, ...request }),
  })
  const data = await readJsonOrThrow(response, 'Manual blur request failed')
  if (!response.ok) {
    throw new Error(typeof data.error === 'string' ? data.error : 'Manual blur request failed')
  }
  return {
    blurredImageUrl: typeof data.blurredImageUrl === 'string' ? data.blurredImageUrl : null,
    faceBlurApplied: Boolean(data.faceBlurApplied),
    regionsCount: Number(data.regionsCount ?? 0),
  }
}

function canonicalizeColor(value?: string) {
  if (!value) return value
  const c = value.toLowerCase()
  if (c.includes('olive') || c.includes('green')) return 'Olive'
  if (c.includes('navy') || c.includes('blue')) return 'Navy'
  if (c.includes('cream') || c.includes('beige')) return 'Cream'
  if (c.includes('taupe') || c.includes('brown')) return 'Taupe'
  if (c.includes('white')) return 'White'
  if (c.includes('black')) return 'Black'
  if (c.includes('gray') || c.includes('grey')) return 'Gray'
  return value
}

const CATEGORY_CANONICAL: Record<string, Item['category']> = {
  tops: 'Tops',
  top: 'Tops',
  shirts: 'Tops',
  bottoms: 'Bottoms',
  bottom: 'Bottoms',
  pants: 'Bottoms',
  trousers: 'Bottoms',
  outerwear: 'Outerwear',
  shoes: 'Shoes',
  shoe: 'Shoes',
  footwear: 'Shoes',
  accessories: 'Accessories',
  accessory: 'Accessories',
}

function toTitleCase(value: string): string {
  return value
    .trim()
    .split(/\s+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(' ')
}

function normalizeCategory(value: string): Item['category'] {
  const key = value.trim().toLowerCase()
  return CATEGORY_CANONICAL[key] ?? 'Tops'
}

function normalizeSeason(season: InferenceResult['season']) {
  const allowed = new Set(['spring', 'summer', 'autumn', 'winter'])
  const normalized = season
    .map((s) => s.trim().toLowerCase())
    .filter((s): s is 'spring' | 'summer' | 'autumn' | 'winter' => allowed.has(s))
  return normalized.length ? normalized : ['spring', 'summer']
}

/**
 * Map free-form LLM output (e.g. "slim-fit", "tailored cut", "boxy cropped")
 * into a canonical FitLabel. Returns undefined when no confident match exists
 * rather than a lossy toTitleCase of unknown text.
 */
function normalizeFit(fit?: string): FitLabel | undefined {
  if (!fit?.trim()) return undefined
  const key = fit.trim().toLowerCase()
  if (key.includes('oversize')) return 'Oversized'
  if (key.includes('crop')) return 'Cropped'
  if (key.includes('tailor') || key.includes('fitted')) return 'Tailored'
  if (key.includes('slim') || key.includes('skinny') || key.includes('athletic')) return 'Slim'
  if (key.includes('relax') || key.includes('loose') || key.includes('boxy') || key.includes('baggy')) return 'Relaxed'
  if (key.includes('regular') || key.includes('classic') || key.includes('standard') || key.includes('straight')) return 'Regular'
  const titled = toTitleCase(fit) as FitLabel
  return (FIT_LABELS as readonly string[]).includes(titled) ? titled : undefined
}

function normalizeMaterial(material: string) {
  const key = material.trim().toLowerCase()
  if (key.includes('denim')) return 'Denim'
  if (key.includes('cotton')) return 'Cotton'
  if (key.includes('wool') || key.includes('cashmere')) return 'Wool'
  if (key.includes('leather')) return 'Leather'
  if (key.includes('suede')) return 'Suede'
  if (key.includes('linen')) return 'Linen'
  if (key.includes('silk')) return 'Silk'
  return toTitleCase(material || 'Unknown')
}

function normalizeItemType(raw: InferenceResult): string {
  const source = raw.subtype?.trim() || raw.item_type?.trim()
  if (!source) return 'Garment'
  return toTitleCase(source.replace(/[_-]+/g, ' '))
}

function inferCategoryFromItemType(itemType: string): Item['category'] | null {
  const key = itemType.trim().toLowerCase()
  if (!key) return null
  if (/(trouser|trousers|pants|jeans|shorts|skirt|capri|pyjama)/.test(key)) return 'Bottoms'
  if (/(jacket|coat|blazer|outerwear)/.test(key)) return 'Outerwear'
  if (/(shoe|shoes|sneaker|boot|loafer|heel|sandal|footwear)/.test(key)) return 'Shoes'
  if (/(belt|bag|cap|hat|watch|jewelry|accessory|accessories|scarf)/.test(key)) return 'Accessories'
  if (/(shirt|tshirt|t-shirt|top|dress|frock|hoodie|sweater|sweatshirt|vest)/.test(key)) return 'Tops'
  return null
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await readJsonOrThrow(response, `Request failed: ${path}`)
  if (!response.ok) {
    throw new Error(typeof data.error === 'string' ? data.error : `Request failed: ${path}`)
  }
  return data as T
}

export const localPreprocessAdapter = async (imageUrl: string, subjectFilter?: SubjectFilterConfig): Promise<PreprocessResult> => {
  return postJson<PreprocessResult>('/api/items/preprocess', { imageUrl, subjectFilter })
}

/**
 * Cut exactly one garment out of the active image using a single user-drawn polygon.
 * The server masks the polygon into a transparent PNG (no face-blur / no SAM re-run),
 * which we can then surface as an extra DetectedGarment card. This is the building
 * block for "multiple garments on one person" — user draws a polygon per garment.
 */
export const cutPolygonAsGarmentAdapter = async (
  imageUrl: string,
  maskPolygon: NormalizedPoint[],
): Promise<PreprocessResult> => {
  return postJson<PreprocessResult>('/api/items/preprocess', { imageUrl, maskPolygon })
}

export const geminiInferenceAdapter = async (processedImageUrl: string): Promise<InferenceResult> => {
  // Main pipeline: prefer server-configured backend (auto/hybrid sidecar first).
  return postJson<InferenceResult>('/api/items/infer', { processedImageUrl })
}

export const advancedGeminiInferenceAdapter = async (processedImageUrl: string): Promise<InferenceResult> => {
  return postJson<InferenceResult>('/api/items/infer', { processedImageUrl, forceGemini: true })
}

export type AdvancedLocalAnalysisResult = {
  ok: boolean
  advanced?: {
    patch: Partial<Item>
    confidence_overall: number
    design_tags: string[]
    style_notes: string
    raw: Record<string, unknown>
  }
  metadata?: {
    provider: string
    model: string
    latency_ms: number
    version: string
  }
  error?: string
}

export const advancedLocalAnalysisAdapter = async (
  imageUrl: string,
  context?: Partial<Item>,
): Promise<AdvancedLocalAnalysisResult> => {
  return postJson<AdvancedLocalAnalysisResult>('/api/items/analyze-advanced-local', { imageUrl, context })
}

export const normalizeAttributesAdapter = async (raw: InferenceResult): Promise<Partial<Item>> => {
  const normalizedSeason = normalizeSeason(raw.season)
  const normalizedFormality = Math.max(1, Math.min(10, raw.formality || 5))
  const normalizedItemType = normalizeItemType(raw)
  const inferredCategory = inferCategoryFromItemType(normalizedItemType)
  const normalizedCategory = inferredCategory ?? normalizeCategory(raw.category)
  const normalizedFit = normalizeFit(raw.fit)
  const normalizedMaterial = normalizeMaterial(raw.material)
  const styleTags = raw.style_archetype ? [toTitleCase(raw.style_archetype)] : []
  const fashionPretty = (raw.fashion_tags ?? []).map((t) => toTitleCase(t.replace(/_/g, ' ')))
  const mergedStyleTags = [...styleTags]
  for (const ft of fashionPretty) {
    if (!mergedStyleTags.some((s) => s.toLowerCase() === ft.toLowerCase())) {
      mergedStyleTags.push(ft)
    }
  }
  const occasions = (raw.occasions ?? []).map((o) => toTitleCase(o))
  const normalizedPattern = raw.pattern ? toTitleCase(raw.pattern.replace(/_/g, ' ')) : undefined
  const normalizedRaw = {
    ...raw,
    item_type: normalizedItemType,
    category: normalizedCategory,
    fit: normalizedFit,
    material: normalizedMaterial,
    color_primary: canonicalizeColor(raw.color_primary) ?? 'Taupe',
    color_secondary: canonicalizeColor(raw.color_secondary),
    season: normalizedSeason,
    style_tags: mergedStyleTags.slice(0, 32),
    fashion_tags: raw.fashion_tags,
    fashion_descriptor: raw.fashion_descriptor,
    occasions,
    pattern: normalizedPattern,
    label_quality: {
      confidence_overall: raw.confidence_overall,
      material_confidence: raw.material_confidence,
      uncertainty: raw.uncertainty,
      quality: raw.quality,
    },
    canonical: {
      item_type_key: normalizedItemType.toLowerCase().replace(/\s+/g, '_'),
      category_key: normalizedCategory.toLowerCase(),
      primary_color_key: (canonicalizeColor(raw.color_primary) ?? 'Taupe').toLowerCase(),
      material_key: normalizedMaterial.toLowerCase(),
      fit_key: normalizedFit?.toLowerCase() ?? null,
    },
  }
  return {
    item_type: normalizedItemType,
    category: normalizedCategory,
    ...(normalizedFit ? { fit: normalizedFit } : {}),
    ...(normalizedPattern ? { pattern: normalizedPattern } : {}),
    ...(mergedStyleTags.length ? { style_tags: mergedStyleTags.slice(0, 32) } : {}),
    ...(raw.fashion_tags?.length ? { fashion_tags: raw.fashion_tags } : {}),
    ...(occasions.length ? { occasions } : {}),
    color_primary: canonicalizeColor(raw.color_primary) ?? 'Taupe',
    color_secondary: canonicalizeColor(raw.color_secondary),
    color_primary_hsl: raw.color_primary_hsl,
    color_secondary_hsl: raw.color_secondary_hsl,
    material: normalizedMaterial,
    formality: normalizedFormality,
    season: normalizedSeason,
    raw_attributes: JSON.stringify(normalizedRaw, null, 2),
  }
}

export const embeddingAdapter = async (item: Item): Promise<EmbeddingResult> => {
  return postJson<EmbeddingResult>('/api/items/embed', { item })
}

export const reasoningAdapter = async (item: Item): Promise<ReasoningResult> => {
  return postJson<ReasoningResult>('/api/items/reason', { item })
}

export const postPipelineSuggestionAdapter = async (
  anchorItem: Item,
  wardrobeItems: Item[],
  profile?: StyleProfile,
  weather?: WeatherContext,
): Promise<PostPipelineSuggestionResult> => {
  return postJson<PostPipelineSuggestionResult>('/api/wardrobe/suggest-next', {
    anchorItem,
    wardrobeItems,
    profile,
    weather,
    topK: 6,
  })
}

export const outfitBuildAdapter = async (
  anchorItem: Item,
  wardrobeItems: Item[],
  profile?: StyleProfile,
  weather?: WeatherContext,
  topK = 3,
): Promise<OutfitBuildResult> => {
  return postJson<OutfitBuildResult>('/api/wardrobe/outfits/build', {
    anchorItem,
    wardrobeItems,
    profile,
    weather,
    topK,
  })
}

export const defaultPipelineAdapters: PipelineAdapters = {
  preprocessImage: localPreprocessAdapter,
  inferAttributes: geminiInferenceAdapter,
  normalizeAttributes: normalizeAttributesAdapter,
  embedItem: embeddingAdapter,
  generateReasoning: reasoningAdapter,
}
