export type Category = 'Tops' | 'Bottoms' | 'Outerwear' | 'Shoes' | 'Accessories'

export type FitPreference = 'Relaxed' | 'Regular' | 'Slim'
export type SizeRegion = 'US' | 'UK' | 'EU' | 'IT' | 'FR' | 'JP' | 'AU'
export type BrandSizeCategory =
  | 'Tops'
  | 'Dresses'
  | 'Bottoms'
  | 'Jeans'
  | 'Outerwear'
  | 'Shoes'
  | 'Activewear'
  | 'Lingerie'
  | 'Swimwear'
  | 'Accessories'

export interface BrandSizeEntry {
  id: string
  brand: string
  category: BrandSizeCategory
  size: string
  fit_note?: string
  runs?: 'small' | 'true' | 'large'
}

export interface SizePassport {
  preferred_region: SizeRegion
  top_size: string
  bottom_size: string
  dress_size?: string
  outerwear_size?: string
  jeans_waist_in?: string
  jeans_inseam_in?: string
  shoe_size_us?: string
  shoe_size_uk?: string
  shoe_size_eu?: string
  shoe_size_cm?: string
  shoe_width?: 'Narrow' | 'Standard' | 'Wide' | 'Extra Wide'
  bra_band?: string
  bra_cup?: string
  bra_region?: 'US' | 'UK' | 'EU' | 'AU'
  underwear_size?: string
  ring_size?: string
  hat_size?: string
  glove_size?: string
  belt_size?: string
  sock_size?: string
  height_cm?: string
  weight_kg?: string
  chest_cm?: string
  bust_cm?: string
  underbust_cm?: string
  waist_cm?: string
  hip_cm?: string
  thigh_cm?: string
  inseam_cm?: string
  outseam_cm?: string
  shoulder_cm?: string
  sleeve_cm?: string
  neck_cm?: string
  foot_length_cm?: string
  foot_width_cm?: string
  fit_preference: FitPreference
  preferred_silhouettes: string[]
  brand_sizes: BrandSizeEntry[]
  notes?: string
}

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
  /** Extraction method used by backend for this crop. */
  mask_type?: 'sam' | 'polygon' | 'grabcut' | 'bbox'
  /** 0-1 confidence score for segmentation quality. */
  segmentation_confidence?: number
  /** Soft fallback: item is usable but should be reviewed/refined. */
  requires_manual_review?: boolean
  /** High-level extraction route used by backend. */
  source_stage?: 'original' | 'tryoff' | 'mannequin' | 'preprocess'
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
  source_image_stage?: 'original' | 'tryoff' | 'blurred_fallback' | 'mannequin' | 'preprocess'
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
  /** Server scenario router output used for path-specific UX. */
  scenario_route?: 'flat_lay' | 'single_person' | 'multi_person' | 'ambiguous'
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
