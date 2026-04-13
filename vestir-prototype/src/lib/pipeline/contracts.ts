import type {
  Category,
  HSLColor,
  Item,
  NormalizedBBox,
  PipelineStageInfo,
  ProcessingStage,
  ProcessingStatus,
  SubjectFilterConfig,
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
  facesBeforeBlur?: number
  facesAfterBlur?: number
  residualFaceRegions?: NormalizedBBox[]
  needsManualPrivacyReview?: boolean
  personDetected?: boolean
  garmentIsolated?: boolean
  backgroundRemoved?: boolean
  scene_track?: 'worn' | 'flat_lay' | 'ambiguous'
  metadata: PipelineMetadata
}

export interface InferenceResult {
  schema_version?: number
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
  dominant_colors?: string[]
  pattern?: string
  /** Silhouette / cut when model returns it (pipeline v2.2+). */
  fit?: string
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
    attribute_disagreement?: boolean
    /** When true, pipeline should skip embedding until user confirms type/category. */
    blocks_embedding?: boolean
    arbitration_applied?: boolean
  }
  source_image_stage?: 'tryoff' | 'blurred_fallback'
  gemini_style_notes?: string
  gemini_design_tags?: string[]
  /** Layered SigLIP vocabulary tags (vision-sidecar v3+). */
  fashion_tags?: string[]
  fashion_tags_scored?: Array<{ layer: string; tag: string; score: number }>
  fashion_tags_by_layer?: Record<string, Array<{ tag: string; score: number }>>
  /** Comma-separated stacked descriptor from tag layers. */
  fashion_descriptor?: string
  gemini_brand_like?: Array<{ name: string; confidence: number }>
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
  preprocessImage: (imageUrl: string, subjectFilter?: SubjectFilterConfig) => Promise<PreprocessResult>
  inferAttributes: (processedImageUrl: string) => Promise<InferenceResult>
  normalizeAttributes: (raw: InferenceResult) => Promise<Partial<Item>>
  embedItem: (item: Item) => Promise<EmbeddingResult>
  generateReasoning: (item: Item) => Promise<ReasoningResult>
}
