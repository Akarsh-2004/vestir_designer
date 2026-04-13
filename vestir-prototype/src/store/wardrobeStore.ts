import { create } from 'zustand'
import { mockItems, mockOutfits, mockWardrobes } from '../data/mock'
import type {
  ActiveView,
  Category,
  DetectedGarment,
  DetectionResult,
  Item,
  NormalizedBBox,
  Outfit,
  SubjectFilterMode,
  Wardrobe,
} from '../types/index'
import {
  applyManualBlur,
  defaultPipelineAdapters,
  detectItemsFromImage,
  embeddingAdapter,
  reasoningAdapter,
  runTryoffExtraction,
  type TryoffGarmentTarget,
} from '../lib/pipeline/adapters'
import { processItemPipeline } from '../lib/pipeline/orchestrator'

interface WardrobeStore {
  wardrobes: Wardrobe[]
  activeWardrobeId: string
  items: Item[]
  outfits: Outfit[]
  activeView: ActiveView
  activeCategory: Category | 'All'
  searchQuery: string
  pendingDetection: DetectionResult | null
  pendingDetectionImageUrl: string | null
  pendingDetectionSelections: Set<string>
  subjectFilterMode: SubjectFilterMode
  subjectFilterPersonSelections: Set<string>
  subjectFilterMaskPolygon: Array<{ x: number; y: number }>
  setActiveWardrobe: (id: string) => void
  setActiveCategory: (cat: Category | 'All') => void
  setActiveView: (view: ActiveView) => void
  setSearchQuery: (query: string) => void
  addWardrobe: (name: string) => void
  deleteWardrobe: (id: string) => void
  updateItem: (id: string, updates: Partial<Item>) => void
  deleteItem: (id: string) => void
  addOutfit: (outfit: Outfit) => void
  detectItemsFromFile: (file: File) => Promise<void>
  toggleDetectionSelection: (id: string) => void
  setSubjectFilterMode: (mode: SubjectFilterMode) => void
  toggleSubjectFilterPersonSelection: (id: string) => void
  setSubjectFilterMaskPolygon: (polygon: Array<{ x: number; y: number }>) => void
  applySubjectFilter: () => Promise<DetectionResult | null>
  applyManualBlurToPending: (boxes: NormalizedBBox[]) => Promise<DetectionResult | null>
  /** FLUX try-off on current pending image (e.g. after blur), then re-run detection on the product shot. */
  applyTryoffExtractionToPending: (
    garmentTarget: TryoffGarmentTarget,
  ) => Promise<
    | { detection: DetectionResult }
    | { error: string; tryoffImageUrl?: string | null }
    | null
  >
  confirmDetectedItems: () => Promise<void>
  dismissDetection: () => void
  addPendingItemsFromFiles: (files: FileList | null) => Promise<void>
  runHybridAiPipeline: () => Promise<void>
  /** After fixing type/category on an item that blocked embedding, run embed + reasoning. */
  completeAttributeReview: (itemId: string) => Promise<void>
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.readAsDataURL(file)
  })
}

function inferTryoffTargetFromFilename(filename: string): 'outfit' | 'shirt' | 'dress' | 'pants' | 'jacket' {
  const f = filename.toLowerCase()
  if (/(tshirt|t-shirt|shirt|tee|top)/.test(f)) return 'shirt'
  if (/(dress|frock|gown)/.test(f)) return 'dress'
  if (/(jeans|pants|trouser|shorts|capri|pyjama)/.test(f)) return 'pants'
  if (/(jacket|coat|blazer|outerwear|hoodie)/.test(f)) return 'jacket'
  return 'outfit'
}

function buildDefaultDetectionSelection(detected: DetectedGarment[]): Set<string> {
  // Default to all detected garments so multi-item photos (top + jeans, etc.)
  // create multiple cards unless the user explicitly unselects some.
  return new Set(detected.map((g) => g.id))
}

export const useWardrobeStore = create<WardrobeStore>((set, get) => ({
  wardrobes: mockWardrobes,
  activeWardrobeId: mockWardrobes[0]?.id ?? '',
  items: mockItems,
  outfits: mockOutfits,
  activeView: 'items',
  activeCategory: 'All',
  searchQuery: '',
  pendingDetection: null,
  pendingDetectionImageUrl: null,
  pendingDetectionSelections: new Set<string>(),
  subjectFilterMode: 'keep_selected_person',
  subjectFilterPersonSelections: new Set<string>(),
  subjectFilterMaskPolygon: [],

  setActiveWardrobe: (id) => set({ activeWardrobeId: id }),
  setActiveCategory: (cat) => set({ activeCategory: cat }),
  setActiveView: (view) => set({ activeView: view }),
  setSearchQuery: (query) => set({ searchQuery: query }),

  addWardrobe: (name) =>
    set((state) => {
      const nextOrder = state.wardrobes.length + 1
      const wardrobe: Wardrobe = {
        id: crypto.randomUUID(),
        user_id: 'user-1',
        name,
        sort_order: nextOrder,
        item_count: 0,
        created_at: new Date().toISOString(),
      }
      return { wardrobes: [...state.wardrobes, wardrobe] }
    }),

  deleteWardrobe: (id) =>
    set((state) => {
      if (state.wardrobes.length <= 1) return state
      const wardrobes = state.wardrobes.filter((w) => w.id !== id)
      const activeWardrobeId = state.activeWardrobeId === id ? wardrobes[0].id : state.activeWardrobeId
      return { wardrobes, activeWardrobeId }
    }),

  updateItem: (id, updates) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id ? { ...item, ...updates, updated_at: new Date().toISOString() } : item,
      ),
    })),

  deleteItem: (id) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id ? { ...item, deleted_at: new Date().toISOString(), updated_at: new Date().toISOString() } : item,
      ),
    })),

  addOutfit: (outfit) => set((state) => ({ outfits: [...state.outfits, outfit] })),

  detectItemsFromFile: async (file) => {
    const imageUrl = await fileToDataUrl(file)
    // Keep uploads responsive by default; enable try-off only when explicitly requested.
    const tryoffFirst = import.meta.env.VITE_TRYOFF_FIRST === '1'
    let detectSourceUrl = imageUrl
    if (tryoffFirst) {
      const target = inferTryoffTargetFromFilename(file.name)
      const tryoff = await runTryoffExtraction(imageUrl, target)
      if (tryoff.implemented && tryoff.tryoffImageUrl) {
        const rel = tryoff.tryoffImageUrl
        detectSourceUrl = rel.startsWith('http') ? rel : `${import.meta.env.VITE_API_BASE_URL ?? ''}${rel}`
      }
    }
    const result = await detectItemsFromImage(detectSourceUrl, {
      mode: 'keep_selected_person',
      aiAssist: true,
    })
    const heroSelections = buildDefaultDetectionSelection(result.detected)
    const defaultPeople = new Set((result.person_candidates ?? []).map((p) => p.id))
    set({
      pendingDetectionImageUrl: detectSourceUrl,
      pendingDetection: result,
      pendingDetectionSelections: heroSelections,
      subjectFilterMode: result.applied_subject_filter?.mode ?? 'keep_selected_person',
      subjectFilterPersonSelections: defaultPeople,
      subjectFilterMaskPolygon: result.applied_subject_filter?.maskPolygon ?? [],
    })
  },

  toggleDetectionSelection: (id) =>
    set((state) => {
      const next = new Set(state.pendingDetectionSelections)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return { pendingDetectionSelections: next }
    }),

  setSubjectFilterMode: (mode) => set({ subjectFilterMode: mode }),

  toggleSubjectFilterPersonSelection: (id) =>
    set((state) => {
      const next = new Set(state.subjectFilterPersonSelections)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return { subjectFilterPersonSelections: next }
    }),

  setSubjectFilterMaskPolygon: (polygon) => set({ subjectFilterMaskPolygon: polygon }),

  applySubjectFilter: async () => {
    const {
      pendingDetectionImageUrl,
      subjectFilterMode,
      subjectFilterPersonSelections,
      subjectFilterMaskPolygon,
    } = get()
    if (!pendingDetectionImageUrl) return null
    const result = await detectItemsFromImage(pendingDetectionImageUrl, {
      mode: subjectFilterMode,
      selectedPersonIds: Array.from(subjectFilterPersonSelections),
      maskPolygon: subjectFilterMaskPolygon.length >= 3 ? subjectFilterMaskPolygon : undefined,
      aiAssist: true,
    })
    const heroSelections = buildDefaultDetectionSelection(result.detected)
    set({
      pendingDetection: result,
      pendingDetectionSelections: heroSelections,
    })
    return result
  },

  applyManualBlurToPending: async (boxes) => {
    const { pendingDetectionImageUrl, pendingDetection, pendingDetectionSelections } = get()
    if (!pendingDetectionImageUrl || !pendingDetection || !boxes.length) return null
    const blur = await applyManualBlur(pendingDetectionImageUrl, boxes)
    if (!blur.blurredImageUrl) return null
    const blurredUrl = blur.blurredImageUrl.startsWith('http')
      ? blur.blurredImageUrl
      : `${import.meta.env.VITE_API_BASE_URL ?? ''}${blur.blurredImageUrl}`
    // Privacy blur should not change garment candidates/labels.
    // Keep the current detection result and only update source image references.
    const result: DetectionResult = {
      ...pendingDetection,
      source_image_url: blurredUrl,
      auto_blurred_image_url: blurredUrl,
      manual_blur_required: false,
      warnings: [
        ...(pendingDetection.warnings ?? []),
        `Manual blur applied to ${blur.regionsCount} region${blur.regionsCount === 1 ? '' : 's'}.`,
      ],
    }
    set({
      pendingDetectionImageUrl: blurredUrl,
      pendingDetection: result,
      pendingDetectionSelections: new Set(pendingDetectionSelections),
    })
    return result
  },

  applyTryoffExtractionToPending: async (garmentTarget) => {
    const { pendingDetectionImageUrl, pendingDetection } = get()
    if (!pendingDetectionImageUrl || !pendingDetection) return null
    const src = pendingDetectionImageUrl.startsWith('http')
      ? pendingDetectionImageUrl
      : `${import.meta.env.VITE_API_BASE_URL ?? ''}${pendingDetectionImageUrl}`
    const tryoff = await runTryoffExtraction(src, garmentTarget)
    if (!tryoff.implemented || !tryoff.tryoffImageUrl) {
      return { error: String(tryoff.message ?? 'Try-off not available or model failed') }
    }
    const tryoffUrl = tryoff.tryoffImageUrl.startsWith('http')
      ? tryoff.tryoffImageUrl
      : `${import.meta.env.VITE_API_BASE_URL ?? ''}${tryoff.tryoffImageUrl}`
    let detected: DetectionResult
    try {
      detected = await detectItemsFromImage(tryoffUrl, {
        mode: 'keep_selected_person',
        aiAssist: true,
      })
    } catch (e1) {
      try {
        detected = await detectItemsFromImage(tryoffUrl, {
          mode: 'keep_selected_person',
          aiAssist: false,
        })
      } catch (e2) {
        const msg = e2 instanceof Error ? e2.message : String(e2)
        return {
          error: `Try-off finished but re-detection failed: ${msg}`,
          tryoffImageUrl: tryoff.tryoffImageUrl,
        }
      }
    }
    const merged: DetectionResult = {
      ...detected,
      source_image_url: tryoffUrl,
      auto_blurred_image_url: null,
      source_image_stage: 'tryoff',
      manual_blur_required: false,
      warnings: [
        ...(detected.warnings ?? []),
        'Virtual try-off (FLUX): garment extracted on white; detections refreshed from product-style image.',
      ],
    }
    const heroSelections = buildDefaultDetectionSelection(merged.detected)
    const defaultPeople = new Set((merged.person_candidates ?? []).map((p) => p.id))
    set({
      pendingDetectionImageUrl: tryoffUrl,
      pendingDetection: merged,
      pendingDetectionSelections: heroSelections,
      subjectFilterMode: merged.applied_subject_filter?.mode ?? 'keep_selected_person',
      subjectFilterPersonSelections: defaultPeople,
      subjectFilterMaskPolygon: merged.applied_subject_filter?.maskPolygon ?? [],
    })
    return { detection: merged }
  },

  confirmDetectedItems: async () => {
    const { pendingDetection, pendingDetectionSelections, activeWardrobeId } = get()
    if (!pendingDetection) return
    const selected = pendingDetection.detected.filter((g) => pendingDetectionSelections.has(g.id))
    if (!selected.length) return
    const now = new Date().toISOString()
    const pending: Item[] = selected.map((garment: DetectedGarment) => ({
      id: crypto.randomUUID(),
      wardrobe_id: activeWardrobeId,
      user_id: 'user-1',
      image_url: garment.crop_url,
      // Do not show detector guess as final type before AI pipeline completes.
      item_type: 'Analyzing...',
      category: 'Tops',
      color_primary: 'Unknown',
      color_primary_hsl: { h: 0, s: 0, l: 0 },
      material: 'Unknown',
      formality: 5,
      season: ['spring', 'summer'],
      ai_processed: false,
      processing_stage: 'uploaded',
      processing_status: 'queued',
      processing_progress: 0,
      raw_attributes: JSON.stringify({
        detected_label: garment.label,
        detected_confidence: garment.confidence,
      }),
      created_at: now,
      updated_at: now,
    }))
    set((state) => ({
      items: [...pending, ...state.items],
      pendingDetection: null,
      pendingDetectionImageUrl: null,
      pendingDetectionSelections: new Set(),
      subjectFilterMaskPolygon: [],
      subjectFilterPersonSelections: new Set(),
    }))
    await Promise.all(
      pending.map((item) =>
        processItemPipeline(item, {
          adapters: defaultPipelineAdapters,
          onStageUpdate: (update) => {
            get().updateItem(update.itemId, {
              ...update.patch,
              processing_stage: update.stage,
              processing_status: update.status,
              processing_progress: update.progress,
            })
          },
        }),
      ),
    )
  },

  dismissDetection: () =>
    set({
      pendingDetection: null,
      pendingDetectionImageUrl: null,
      pendingDetectionSelections: new Set(),
      subjectFilterMaskPolygon: [],
      subjectFilterPersonSelections: new Set(),
    }),

  addPendingItemsFromFiles: async (files) => {
    if (!files || files.length === 0) return
    const previews = await Promise.all(Array.from(files).slice(0, 50).map(fileToDataUrl))
    const now = new Date().toISOString()
    const wardrobeId = get().activeWardrobeId
    const pending: Item[] = previews.map((url) => ({
      id: crypto.randomUUID(),
      wardrobe_id: wardrobeId,
      user_id: 'user-1',
      image_url: url,
      item_type: 'Processing...',
      category: 'Tops',
      color_primary: 'Unknown',
      color_primary_hsl: { h: 0, s: 0, l: 0 },
      material: 'Unknown',
      formality: 3,
      season: ['spring', 'summer'],
      ai_processed: false,
      processing_stage: 'uploaded',
      processing_status: 'queued',
      processing_progress: 0,
      created_at: now,
      updated_at: now,
    }))
    set((state) => ({ items: [...pending, ...state.items] }))
  },

  runHybridAiPipeline: async () => {
    const pending = get().items.filter((i) => !i.ai_processed && i.processing_status !== 'running')
    await Promise.all(
      pending.map((item) =>
        processItemPipeline(item, {
          adapters: defaultPipelineAdapters,
          onStageUpdate: (update) => {
            get().updateItem(update.itemId, {
              ...update.patch,
              processing_stage: update.stage,
              processing_status: update.status,
              processing_progress: update.progress,
            })
          },
        }),
      ),
    )
  },

  completeAttributeReview: async (itemId: string) => {
    const item = get().items.find((i) => i.id === itemId && !i.deleted_at)
    if (!item?.attribute_review_pending) return
    get().updateItem(itemId, {
      processing_status: 'running',
      processing_progress: 92,
      processing_stage: 'embedding',
    })
    try {
      const embedResult = await embeddingAdapter(item)
      const reasoning = await reasoningAdapter(item)
      let baseRaw: Record<string, unknown> = {}
      try {
        baseRaw = JSON.parse(item.raw_attributes ?? '{}') as Record<string, unknown>
      } catch {
        baseRaw = {}
      }
      get().updateItem(itemId, {
        ai_processed: true,
        attribute_review_pending: false,
        processing_stage: 'complete',
        processing_status: 'done',
        processing_progress: 100,
        reasoning_summary: reasoning.summary,
        raw_attributes: JSON.stringify(
          { ...baseRaw, embedding_metadata: embedResult.metadata },
          null,
          2,
        ),
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Embedding failed'
      get().updateItem(itemId, {
        processing_status: 'failed',
        processing_error: msg,
      })
    }
  },
}))
