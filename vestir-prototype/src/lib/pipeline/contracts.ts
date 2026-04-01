import type {
  Category,
  HSLColor,
  Item,
  PipelineStageInfo,
  ProcessingStage,
  ProcessingStatus,
} from '../../types/index'

export interface PipelineMetadata {
  provider: string
  model: string
  latency_ms: number
  version: string
  /** Documents how this response maps onto the multi-stage pipeline. */
  pipeline?: {
    architecture_id: string
    stages: PipelineStageInfo[]
  }
}

export interface PreprocessResult {
  processedImageUrl: string
  faceDetected: boolean
  faceBlurApplied: boolean
  personDetected?: boolean
  garmentIsolated?: boolean
  scene_track?: 'worn' | 'flat_lay' | 'ambiguous'
  metadata: PipelineMetadata
}

export interface InferenceResult {
  item_type: string
  category: Category
  subtype?: string
  color_primary: string
  color_secondary?: string
  color_primary_hsl: HSLColor
  color_secondary_hsl?: HSLColor
  color_palette?: Array<{
    name: string
    hex: string
    hsl: HSLColor
    coverage_pct: number
    is_neutral?: boolean
  }>
  pattern?: string
  material: string
  material_confidence?: number
  formality: number
  season: string[]
  season_weights?: Record<string, number>
  occasions?: string[]
  style_archetype?: string
  confidence_overall?: number
  uncertainty?: {
    requires_user_confirmation: boolean
    uncertain_fields: string[]
  }
  quality?: {
    blur_score: number
    lighting_score: number
    framing: 'flat_lay' | 'worn' | 'detail'
    occlusion_visible_pct: number
    accepted: boolean
    warnings: string[]
  }
  metadata: PipelineMetadata
}

export interface EmbeddingResult {
  vector_id: string
  dimensions: number
  metadata: PipelineMetadata
}

export interface ReasoningResult {
  summary: string
  pairing_suggestions?: string[]
  avoid_note?: string
  care_context?: string
  metadata: PipelineMetadata
}

export interface StageUpdate {
  itemId: string
  stage: ProcessingStage
  status: ProcessingStatus
  progress: number
  patch?: Partial<Item>
}

export interface PipelineAdapters {
  preprocessImage: (imageUrl: string) => Promise<PreprocessResult>
  inferAttributes: (processedImageUrl: string) => Promise<InferenceResult>
  normalizeAttributes: (raw: InferenceResult) => Promise<Partial<Item>>
  embedItem: (item: Item) => Promise<EmbeddingResult>
  generateReasoning: (item: Item) => Promise<ReasoningResult>
}
