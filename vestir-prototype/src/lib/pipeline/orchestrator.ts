import type { Item } from '../../types/index'
import type { PipelineAdapters, StageUpdate } from './contracts'

interface OrchestrateOptions {
  adapters: PipelineAdapters
  onStageUpdate: (update: StageUpdate) => void
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
      progress: 18,
      patch: {
        processing_stage: 'preprocessing',
        processing_status: 'running',
        processing_progress: 18,
        pipeline_substage: 'image_filtering',
      },
    })
    const preprocessed = await options.adapters.preprocessImage(item.image_url)

    emit(options, {
      itemId: item.id,
      stage: 'preprocessing',
      status: 'running',
      progress: 36,
      patch: {
        processing_stage: 'preprocessing',
        processing_status: 'running',
        processing_progress: 36,
        pipeline_substage: 'clothing_extraction',
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
        pipeline_substage: 'attribute_inference',
        cutout_url: preprocessed.processedImageUrl,
      },
    })
    const rawAttributes = await options.adapters.inferAttributes(preprocessed.processedImageUrl)

    emit(options, {
      itemId: item.id,
      stage: 'normalization',
      status: 'running',
      progress: 65,
      patch: {
        processing_stage: 'normalization',
        processing_status: 'running',
        processing_progress: 65,
        pipeline_substage: 'post_processing',
      },
    })
    const normalized = await options.adapters.normalizeAttributes(rawAttributes)
    const merged = { ...item, ...normalized }

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
        ...normalized,
      },
    })
    await options.adapters.embedItem(merged)

    emit(options, {
      itemId: item.id,
      stage: 'reasoning',
      status: 'running',
      progress: 92,
      patch: {
        processing_stage: 'reasoning',
        processing_status: 'running',
        processing_progress: 92,
        pipeline_substage: 'post_processing',
      },
    })
    const reasoning = await options.adapters.generateReasoning(merged)

    emit(options, {
      itemId: item.id,
      stage: 'complete',
      status: 'done',
      progress: 100,
      patch: {
        ai_processed: true,
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
