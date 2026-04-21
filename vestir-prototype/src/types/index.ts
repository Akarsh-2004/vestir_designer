export type Category = 'Tops' | 'Bottoms' | 'Outerwear' | 'Shoes' | 'Accessories'

export const FIT_LABELS = ['Slim', 'Regular', 'Relaxed', 'Oversized', 'Cropped', 'Tailored'] as const
export type FitLabel = (typeof FIT_LABELS)[number]

export interface HSLColor {
  h: number
  s: number
  l: number
}

export interface Item {
  id: string
  wardrobe_id: string
  user_id: string
  image_url: string
  cutout_url?: string
  item_type: string
  category: Category
  color_primary: string
  color_secondary?: string
  color_primary_hsl: HSLColor
  color_secondary_hsl?: HSLColor
  material: string
  /** Canonical fit label when inference returns it. */
  fit?: FitLabel
  pattern?: string
  /** Set when models disagreed; embedding skipped until user confirms (see item detail). */
  attribute_review_pending?: boolean
  style_tags?: string[]
  /** Rich tags from SigLIP + controlled vocabulary (optional). */
  fashion_tags?: string[]
  occasions?: string[]
  formality: number
  season: string[]
  ai_processed: boolean
  processing_stage?: ProcessingStage
  /** Fine-grained step within the multi-stage pipeline (optional UI detail). */
  pipeline_substage?: PipelineStageId
  processing_status?: ProcessingStatus
  processing_progress?: number
  raw_attributes?: string
  reasoning_summary?: string
  processing_error?: string
  deleted_at?: string
  created_at: string
  updated_at: string
}

export interface Wardrobe {
  id: string
  user_id: string
  name: string
  sort_order: number
  item_count: number
  created_at: string
}

export interface Outfit {
  id: string
  user_id: string
  wardrobe_id: string
  name: string
  anchor_item_id: string
  items: Item[]
  created_at: string
}

export interface DetectedGarment {
  id: string
  label: string
  confidence: number
  crop_url: string
  coverage: number
  centrality: number
  salience: number
  is_hero: boolean
  partially_visible: boolean
  warning?: string
  /**
   * Normalized bounding box of the detected garment in the source frame.
   * Used downstream for per-garment SAM refinement (transparent PNG cutouts)
   * without re-running Vision. Optional for backwards compatibility with older
   * detection results that predate this field.
   */
  bbox?: NormalizedBBox
  /** True when crop_url has a transparent background (alpha PNG via SAM/mask). */
  background_removed?: boolean
}

export interface NormalizedPoint {
  x: number
  y: number
}

export interface NormalizedBBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export type SubjectFilterMode = 'keep_selected_person' | 'clothing_only' | 'focus_person_blur_others'

export type BlurQualityPreset = 'soft' | 'pro' | 'strong'

export interface SubjectFilterConfig {
  mode: SubjectFilterMode
  selectedPersonIds?: string[]
  selectedPersonBboxes?: NormalizedBBox[]
  maskPolygon?: NormalizedPoint[]
  aiAssist?: boolean
}

export interface PersonCandidate {
  id: string
  label: string
  confidence: number
  bbox: NormalizedBBox
}

export interface FaceCandidate {
  id: string
  confidence: number
  bbox: NormalizedBBox
  source?: string
}

export interface PipelineStageInfo {
  id: PipelineStageId
  status: 'completed' | 'skipped' | 'partial'
  detail?: string
}

export interface DetectionResult {
  detected: DetectedGarment[]
  person_candidates?: PersonCandidate[]
  person_assignments?: Array<{
    garment_id: string
    person_id: string | null
    confidence: number
    requires_confirmation: boolean
  }>
  face_candidates?: FaceCandidate[]
  scene_track: 'worn' | 'flat_lay' | 'ambiguous'
  source_image_url: string
  source_image_stage?: 'tryoff' | 'blurred_fallback'
  auto_blurred_image_url?: string | null
  manual_blur_required?: boolean
  applied_subject_filter?: SubjectFilterConfig
  /** Server-reported alignment with the multi-stage architecture. */
  pipeline?: {
    architecture_id: string
    stages: PipelineStageInfo[]
    estimated_person_count?: number
    multi_person?: boolean
  }
  warnings?: string[]
}

export type ActiveView = 'items' | 'outfits'

/** Multi-stage vision pipeline (each stage narrows the problem for the next). */
export type PipelineStageId =
  | 'image_filtering'
  | 'subject_filtering'
  | 'human_detection'
  | 'face_detection'
  | 'auto_blur'
  | 'manual_blur'
  | 'tryoff_extraction'
  | 'human_parsing'
  | 'privacy_masking'
  | 'clothing_extraction'
  | 'attribute_detection'
  | 'gemini_enrichment'
  | 'attribute_inference'
  | 'post_processing'

export type ProcessingStage =
  | 'uploaded'
  | 'preprocessing'
  | 'inference'
  | 'normalization'
  | 'embedding'
  | 'reasoning'
  | 'complete'

export type ProcessingStatus = 'queued' | 'running' | 'done' | 'failed'
