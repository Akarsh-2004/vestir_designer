import type { EmbeddingResult, InferenceResult, PipelineAdapters, PreprocessResult, ReasoningResult } from './contracts'
import type { DetectionResult, Item, SubjectFilterConfig } from '../../types/index'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

export async function detectItemsFromImage(imageUrl: string, subjectFilter?: SubjectFilterConfig): Promise<DetectionResult> {
  const response = await fetch(`${API_BASE}/api/items/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUrl, subjectFilter }),
  })
  const data = await response.json()
  if (!response.ok) throw new Error(data.error ?? 'Detection failed')
  return data as DetectionResult
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

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error ?? `Request failed: ${path}`)
  }
  return data as T
}

export const localPreprocessAdapter = async (imageUrl: string, subjectFilter?: SubjectFilterConfig): Promise<PreprocessResult> => {
  return postJson<PreprocessResult>('/api/items/preprocess', { imageUrl, subjectFilter })
}

export const geminiInferenceAdapter = async (processedImageUrl: string): Promise<InferenceResult> => {
  return postJson<InferenceResult>('/api/items/infer', { processedImageUrl })
}

export const normalizeAttributesAdapter = async (raw: InferenceResult): Promise<Partial<Item>> => {
  const normalizedSeason = raw.season.length ? raw.season : ['spring', 'summer']
  const normalizedFormality = Math.max(1, Math.min(10, raw.formality || 5))
  return {
    item_type: raw.item_type.trim(),
    category: raw.category,
    ...(raw.fit ? { fit: raw.fit } : {}),
    color_primary: canonicalizeColor(raw.color_primary) ?? 'Taupe',
    color_secondary: canonicalizeColor(raw.color_secondary),
    color_primary_hsl: raw.color_primary_hsl,
    color_secondary_hsl: raw.color_secondary_hsl,
    material: raw.material,
    formality: normalizedFormality,
    season: normalizedSeason,
    raw_attributes: JSON.stringify(raw, null, 2),
  }
}

export const embeddingAdapter = async (item: Item): Promise<EmbeddingResult> => {
  return postJson<EmbeddingResult>('/api/items/embed', { item })
}

export const reasoningAdapter = async (item: Item): Promise<ReasoningResult> => {
  return postJson<ReasoningResult>('/api/items/reason', { item })
}

export const defaultPipelineAdapters: PipelineAdapters = {
  preprocessImage: localPreprocessAdapter,
  inferAttributes: geminiInferenceAdapter,
  normalizeAttributes: normalizeAttributesAdapter,
  embedItem: embeddingAdapter,
  generateReasoning: reasoningAdapter,
}
