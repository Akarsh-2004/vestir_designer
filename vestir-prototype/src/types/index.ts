export type Category = 'Tops' | 'Bottoms' | 'Outerwear' | 'Shoes' | 'Accessories'

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
  /** Relaxed / slim / … when inference returns it */
  fit?: string
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
}

export interface PipelineStageInfo {
  id: PipelineStageId
  status: 'completed' | 'skipped' | 'partial'
  detail?: string
}

export interface DetectionResult {
  detected: DetectedGarment[]
  scene_track: 'worn' | 'flat_lay' | 'ambiguous'
  source_image_url: string
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
  | 'human_detection'
  | 'human_parsing'
  | 'privacy_masking'
  | 'clothing_extraction'
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
