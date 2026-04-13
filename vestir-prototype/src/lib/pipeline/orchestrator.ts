import type { Item } from '../../types/index'
import type { InferenceResult, PipelineAdapters, ReasoningResult, StageUpdate } from './contracts'

interface OrchestrateOptions {
  adapters: PipelineAdapters
  onStageUpdate: (update: StageUpdate) => void
}

function parseLooseJson(value?: string): Record<string, unknown> {
  if (!value) return {}
  try {
    return JSON.parse(value) as Record<string, unknown>
  } catch {
    return {}
  }
}

function isGenericTypeLabel(value?: string) {
  if (!value) return true
  const v = value.trim().toLowerCase()
  return ['top', 'shirt', 'garment', 'tee'].includes(v)
}

function isSpecificDressLike(value?: string) {
  if (!value) return false
  const v = value.trim().toLowerCase()
  return /dress|frock|gown|kurti/.test(v)
}

function emit(options: OrchestrateOptions, update: StageUpdate) {
  options.onStageUpdate(update)
}

export async function processItemPipeline(item: Item, options: OrchestrateOptions) {
  try {
    // Multi-stage pipeline: coarse UI stages map to substages (image_filtering → … → attribute_inference → post_processing).
    emit(options, {
      itemId: item.id,
      stage: 'preprocessing',
      status: 'running',
      progress: 12,
      patch: {
        processing_stage: 'preprocessing',
        processing_status: 'running',
        processing_progress: 12,
        pipeline_substage: 'image_filtering',
      },
    })
    const preprocessed = await options.adapters.preprocessImage(item.image_url)

    emit(options, {
      itemId: item.id,
      stage: 'preprocessing',
      status: 'running',
      progress: 28,
      patch: {
        processing_stage: 'preprocessing',
        processing_status: 'running',
        processing_progress: 28,
        pipeline_substage: 'face_detection',
        cutout_url: preprocessed.processedImageUrl,
      },
    })

    emit(options, {
      itemId: item.id,
      stage: 'preprocessing',
      status: 'running',
      progress: 36,
      patch: {
        processing_stage: 'preprocessing',
        processing_status: 'running',
        processing_progress: 36,
        pipeline_substage: 'tryoff_extraction',
        cutout_url: preprocessed.processedImageUrl,
      },
    })

    emit(options, {
      itemId: item.id,
      stage: 'inference',
      status: 'running',
      progress: 45,
      patch: {
        processing_stage: 'inference',
        processing_status: 'running',
        processing_progress: 45,
        pipeline_substage: 'attribute_detection',
        cutout_url: preprocessed.processedImageUrl,
      },
    })
    const rawAttributes = await options.adapters.inferAttributes(preprocessed.processedImageUrl)
    const inferPayload = rawAttributes as InferenceResult
    const blocksEmbed = Boolean(inferPayload.uncertainty?.blocks_embedding)

    emit(options, {
      itemId: item.id,
      stage: 'normalization',
      status: 'running',
      progress: 65,
      patch: {
        processing_stage: 'normalization',
        processing_status: 'running',
        processing_progress: 65,
        pipeline_substage: 'gemini_enrichment',
      },
    })
    const normalized = await options.adapters.normalizeAttributes(rawAttributes)
    const guardedNormalized = { ...normalized }
    const detected = parseLooseJson(item.raw_attributes)
    const detectedLabel = typeof detected.detected_label === 'string' ? detected.detected_label.trim() : ''
    const lowConfidence = typeof rawAttributes.confidence_overall === 'number' && rawAttributes.confidence_overall < 0.72
    if (detectedLabel && lowConfidence && (isGenericTypeLabel(normalized.item_type) || !normalized.item_type)) {
      guardedNormalized.item_type = detectedLabel
    }
    if (isSpecificDressLike(item.item_type) && isGenericTypeLabel(normalized.item_type)) {
      guardedNormalized.item_type = item.item_type
    }
    const merged = { ...item, ...guardedNormalized }

    if (blocksEmbed) {
      emit(options, {
        itemId: item.id,
        stage: 'complete',
        status: 'done',
        progress: 100,
        patch: {
          ...guardedNormalized,
          attribute_review_pending: true,
          ai_processed: false,
          processing_stage: 'complete',
          processing_status: 'done',
          processing_progress: 100,
          processing_error: undefined,
          color_secondary: merged.color_secondary,
          raw_attributes: merged.raw_attributes,
          material: merged.material,
          season: merged.season,
          item_type: merged.item_type,
          category: merged.category,
          color_primary: merged.color_primary,
          color_primary_hsl: merged.color_primary_hsl,
          formality: merged.formality,
        },
      })
      const stub: ReasoningResult = {
        summary:
          'Models disagreed on category or type. Fix details on the item page, then run “Finish embedding” so closet search stays accurate.',
        pairing_suggestions: [],
        metadata: { provider: 'vestir-pipeline', model: 'attribute-review-hold', latency_ms: 0, version: '1.0.0' },
      }
      return stub
    }

    emit(options, {
      itemId: item.id,
      stage: 'embedding',
      status: 'running',
      progress: 80,
      patch: {
        processing_stage: 'embedding',
        processing_status: 'running',
        processing_progress: 80,
        pipeline_substage: 'post_processing',
        ...guardedNormalized,
      },
    })
    emit(options, {
      itemId: item.id,
      stage: 'reasoning',
      status: 'running',
      progress: 90,
      patch: {
        processing_stage: 'reasoning',
        processing_status: 'running',
        processing_progress: 90,
        pipeline_substage: 'post_processing',
      },
    })
    const [, reasoning] = await Promise.all([
      options.adapters.embedItem(merged),
      options.adapters.generateReasoning(merged),
    ])

    emit(options, {
      itemId: item.id,
      stage: 'complete',
      status: 'done',
      progress: 100,
      patch: {
        ai_processed: true,
        attribute_review_pending: false,
        processing_stage: 'complete',
        processing_status: 'done',
        processing_progress: 100,
        processing_error: undefined,
        color_secondary: merged.color_secondary,
        raw_attributes: merged.raw_attributes,
        material: merged.material,
        season: merged.season,
        item_type: merged.item_type,
        category: merged.category,
        color_primary: merged.color_primary,
        color_primary_hsl: merged.color_primary_hsl,
        formality: merged.formality,
        reasoning_summary: reasoning.summary,
      },
    })

    return reasoning
  } catch (error) {
    emit(options, {
      itemId: item.id,
      stage: item.processing_stage ?? 'inference',
      status: 'failed',
      progress: item.processing_progress ?? 0,
      patch: {
        processing_status: 'failed',
        processing_error: error instanceof Error ? error.message : 'Pipeline failed',
      },
    })
    throw error
  }
}
