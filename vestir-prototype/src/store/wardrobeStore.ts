import { create } from 'zustand'
import { mockItems, mockOutfits, mockWardrobes } from '../data/mock'
import type {
  ActiveView,
  BlurQualityPreset,
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
  generateMannequinImage,
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
  blurQualityPreset: BlurQualityPreset
  subjectFilterPersonSelections: Set<string>
  subjectFilterMaskPolygon: Array<{ x: number; y: number }>
  editorHistoryPast: EditorSnapshot[]
  editorHistoryFuture: EditorSnapshot[]
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
  setBlurQualityPreset: (preset: BlurQualityPreset) => void
  toggleSubjectFilterPersonSelection: (id: string) => void
  setSubjectFilterMaskPolygon: (polygon: Array<{ x: number; y: number }>) => void
  undoEditorState: () => void
  redoEditorState: () => void
  applySubjectFilter: () => Promise<DetectionResult | null>
  applyManualBlurToPending: (boxes: NormalizedBBox[]) => Promise<DetectionResult | null>
  /** Returns updated detection, or `null` if nothing pending. Throws on mannequin API failure (so the UI never treats an error object as success during HMR). */
  generateMannequinToPending: () => Promise<DetectionResult | null>
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

type EditorSnapshot = {
  pendingDetection: DetectionResult | null
  pendingDetectionImageUrl: string | null
  pendingDetectionSelections: string[]
  subjectFilterMode: SubjectFilterMode
  blurQualityPreset: BlurQualityPreset
  subjectFilterPersonSelections: string[]
  subjectFilterMaskPolygon: Array<{ x: number; y: number }>
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

/** Bbox around polygon points with minimum span (open polyline / single anchor). */
function axisAlignedBBoxWithMinSpanFromPoints(
  points: Array<{ x: number; y: number }>,
  minSpan = 0.02,
): NormalizedBBox | undefined {
  if (points.length < 1) return undefined
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  let x1 = Math.max(0, Math.min(...xs))
  let y1 = Math.max(0, Math.min(...ys))
  let x2 = Math.min(1, Math.max(...xs))
  let y2 = Math.min(1, Math.max(...ys))
  if (x2 - x1 < minSpan) {
    const cx = (x1 + x2) / 2
    x1 = Math.max(0, cx - minSpan / 2)
    x2 = Math.min(1, cx + minSpan / 2)
  }
  if (y2 - y1 < minSpan) {
    const cy = (y1 + y2) / 2
    y1 = Math.max(0, cy - minSpan / 2)
    y2 = Math.min(1, cy + minSpan / 2)
  }
  if (x2 <= x1) x2 = Math.min(1, x1 + minSpan)
  if (y2 <= y1) y2 = Math.min(1, y1 + minSpan)
  return { x1, y1, x2, y2 }
}

/** Prefer server-processed preview (face blur + subject filter) over raw data URLs. */
function processedPreviewUrlFromDetection(
  result: DetectionResult,
  fallbackDataUrl: string,
): string {
  const apiBase = import.meta.env.VITE_API_BASE_URL ?? ''
  const rel = result.source_image_url || result.auto_blurred_image_url
  if (rel && typeof rel === 'string' && rel.length > 0) {
    return rel.startsWith('http') ? rel : `${apiBase}${rel}`
  }
  return fallbackDataUrl
}

function snapshotFromState(state: WardrobeStore): EditorSnapshot {
  return {
    pendingDetection: state.pendingDetection,
    pendingDetectionImageUrl: state.pendingDetectionImageUrl,
    pendingDetectionSelections: Array.from(state.pendingDetectionSelections),
    subjectFilterMode: state.subjectFilterMode,
    blurQualityPreset: state.blurQualityPreset,
    subjectFilterPersonSelections: Array.from(state.subjectFilterPersonSelections),
    subjectFilterMaskPolygon: state.subjectFilterMaskPolygon.map((p) => ({ ...p })),
  }
}

function restoreSnapshot(snapshot: EditorSnapshot) {
  return {
    pendingDetection: snapshot.pendingDetection,
    pendingDetectionImageUrl: snapshot.pendingDetectionImageUrl,
    pendingDetectionSelections: new Set(snapshot.pendingDetectionSelections),
    subjectFilterMode: snapshot.subjectFilterMode,
    blurQualityPreset: snapshot.blurQualityPreset,
    subjectFilterPersonSelections: new Set(snapshot.subjectFilterPersonSelections),
    subjectFilterMaskPolygon: snapshot.subjectFilterMaskPolygon.map((p) => ({ ...p })),
  }
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
  blurQualityPreset: 'pro',
  subjectFilterPersonSelections: new Set<string>(),
  subjectFilterMaskPolygon: [],
  editorHistoryPast: [],
  editorHistoryFuture: [],

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
    // Temporarily disable try-off-first so SAM/mask quality is evaluated on original uploads.
    const tryoffFirst = false
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
    const previewUrl = processedPreviewUrlFromDetection(result, detectSourceUrl)
    set({
      pendingDetectionImageUrl: previewUrl,
      pendingDetection: result,
      pendingDetectionSelections: heroSelections,
      subjectFilterMode: result.applied_subject_filter?.mode ?? 'keep_selected_person',
      subjectFilterPersonSelections: defaultPeople,
      subjectFilterMaskPolygon: result.applied_subject_filter?.maskPolygon ?? [],
      editorHistoryPast: [],
      editorHistoryFuture: [],
    })
  },

  toggleDetectionSelection: (id) =>
    set((state) => {
      const next = new Set(state.pendingDetectionSelections)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return {
        pendingDetectionSelections: next,
        editorHistoryPast: [...state.editorHistoryPast, snapshotFromState(state)].slice(-30),
        editorHistoryFuture: [],
      }
    }),

  setSubjectFilterMode: (mode) =>
    set((state) => ({
      subjectFilterMode: mode,
      editorHistoryPast: [...state.editorHistoryPast, snapshotFromState(state)].slice(-30),
      editorHistoryFuture: [],
    })),

  setBlurQualityPreset: (preset) =>
    set((state) => ({
      blurQualityPreset: preset,
      editorHistoryPast: [...state.editorHistoryPast, snapshotFromState(state)].slice(-30),
      editorHistoryFuture: [],
    })),

  toggleSubjectFilterPersonSelection: (id) =>
    set((state) => {
      const next = new Set(state.subjectFilterPersonSelections)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return {
        subjectFilterPersonSelections: next,
        editorHistoryPast: [...state.editorHistoryPast, snapshotFromState(state)].slice(-30),
        editorHistoryFuture: [],
      }
    }),

  setSubjectFilterMaskPolygon: (polygon) =>
    set((state) => ({
      subjectFilterMaskPolygon: polygon,
      editorHistoryPast: [...state.editorHistoryPast, snapshotFromState(state)].slice(-30),
      editorHistoryFuture: [],
    })),

  undoEditorState: () =>
    set((state) => {
      const prev = state.editorHistoryPast[state.editorHistoryPast.length - 1]
      if (!prev) return state
      const current = snapshotFromState(state)
      return {
        ...restoreSnapshot(prev),
        editorHistoryPast: state.editorHistoryPast.slice(0, -1),
        editorHistoryFuture: [current, ...state.editorHistoryFuture].slice(0, 30),
      }
    }),

  redoEditorState: () =>
    set((state) => {
      const next = state.editorHistoryFuture[0]
      if (!next) return state
      const current = snapshotFromState(state)
      return {
        ...restoreSnapshot(next),
        editorHistoryPast: [...state.editorHistoryPast, current].slice(-30),
        editorHistoryFuture: state.editorHistoryFuture.slice(1),
      }
    }),

  applySubjectFilter: async () => {
    const {
      pendingDetectionImageUrl,
      subjectFilterMode,
      subjectFilterPersonSelections,
      subjectFilterMaskPolygon,
    } = get()
    if (!pendingDetectionImageUrl) return null
    const before = snapshotFromState(get())
    const fallbackBbox = subjectFilterMaskPolygon.length >= 3
      ? {
          x1: Math.max(0, Math.min(...subjectFilterMaskPolygon.map((p) => p.x))),
          y1: Math.max(0, Math.min(...subjectFilterMaskPolygon.map((p) => p.y))),
          x2: Math.min(1, Math.max(...subjectFilterMaskPolygon.map((p) => p.x))),
          y2: Math.min(1, Math.max(...subjectFilterMaskPolygon.map((p) => p.y))),
        }
      : undefined
    const result = await detectItemsFromImage(pendingDetectionImageUrl, {
      mode: subjectFilterMode,
      selectedPersonIds: Array.from(subjectFilterPersonSelections),
      selectedPersonBboxes:
        subjectFilterMode === 'focus_person_blur_others'
        && subjectFilterPersonSelections.size === 0
        && fallbackBbox
          ? [fallbackBbox]
          : undefined,
      maskPolygon: subjectFilterMaskPolygon.length >= 3 ? subjectFilterMaskPolygon : undefined,
      aiAssist: true,
    })
    const heroSelections = buildDefaultDetectionSelection(result.detected)
    const nextPreviewUrl = result.source_image_url || result.auto_blurred_image_url || pendingDetectionImageUrl
    set({
      pendingDetectionImageUrl: nextPreviewUrl,
      pendingDetection: result,
      pendingDetectionSelections: heroSelections,
      editorHistoryPast: [...get().editorHistoryPast, before].slice(-30),
      editorHistoryFuture: [],
    })
    return result
  },

  applyManualBlurToPending: async (boxes) => {
    const {
      pendingDetectionImageUrl,
      pendingDetection,
      pendingDetectionSelections,
      subjectFilterMaskPolygon,
      blurQualityPreset,
    } = get()
    if (!pendingDetectionImageUrl || !pendingDetection || !boxes.length) return null
    const before = snapshotFromState(get())
    const blur = await applyManualBlur(pendingDetectionImageUrl, {
      boxes,
      maskPolygon: subjectFilterMaskPolygon.length >= 3 ? subjectFilterMaskPolygon : undefined,
      blurPreset: blurQualityPreset,
    })
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
      editorHistoryPast: [...get().editorHistoryPast, before].slice(-30),
      editorHistoryFuture: [],
    })
    return result
  },

  generateMannequinToPending: async () => {
    const {
      pendingDetectionImageUrl,
      pendingDetection,
      subjectFilterMode,
      subjectFilterPersonSelections,
      subjectFilterMaskPolygon,
      pendingDetectionSelections,
    } = get()
    if (!pendingDetectionImageUrl || !pendingDetection) return null
    const appliedMask = pendingDetection.applied_subject_filter?.maskPolygon
    const maskPts =
      subjectFilterMaskPolygon.length > 0
        ? subjectFilterMaskPolygon
        : (Array.isArray(appliedMask) && appliedMask.length > 0 ? appliedMask.map((p) => ({ ...p })) : [])
    const selectedPersonBboxes = (pendingDetection.person_candidates ?? [])
      .filter((p) => subjectFilterPersonSelections.has(p.id))
      .map((p) => p.bbox)
    const bboxFromMask = axisAlignedBBoxWithMinSpanFromPoints(maskPts)
    const fallbackBboxes =
      selectedPersonBboxes.length > 0
        ? selectedPersonBboxes
        : (bboxFromMask ? [bboxFromMask] : undefined)

    // Reject obviously-useless masks BEFORE hitting the server so we never show a
    // misleading "Mannequin generated" toast for a cutout that visually matches the input.
    const maskBbox = bboxFromMask
    const usingPolygonMask = maskPts.length >= 3
    const approxArea = usingPolygonMask && maskBbox
      ? (maskBbox.x2 - maskBbox.x1) * (maskBbox.y2 - maskBbox.y1)
      : fallbackBboxes && fallbackBboxes.length
        ? fallbackBboxes
            .map((b) => Math.max(0, (b.x2 - b.x1) * (b.y2 - b.y1)))
            .reduce((a, b) => Math.max(a, b), 0)
        : 0
    if (!usingPolygonMask && (!fallbackBboxes || fallbackBboxes.length === 0)) {
      throw new Error(
        'No region selected. Draw a rectangle (or pick a person) so SAM can isolate the garment.',
      )
    }
    if (approxArea > 0.97) {
      throw new Error(
        'Selected region covers the entire frame, so the mannequin would look identical to the input. Draw a tighter rectangle around just the garment.',
      )
    }

    const before = snapshotFromState(get())
    // eslint-disable-next-line no-console
    console.info('[mannequin] cutout request', {
      inputImageUrl: pendingDetectionImageUrl,
      maskPoints: maskPts.length,
      bboxes: fallbackBboxes?.length ?? 0,
      approxMaskArea: approxArea,
    })
    const mannequin = await generateMannequinImage(pendingDetectionImageUrl, {
      mode: subjectFilterMode,
      selectedPersonIds: Array.from(subjectFilterPersonSelections),
      selectedPersonBboxes: fallbackBboxes,
      maskPolygon: maskPts.length >= 1 ? maskPts : undefined,
      aiAssist: true,
    })
    if (!mannequin.mannequinImageUrl) {
      throw new Error('Server did not return a mannequin image URL.')
    }
    const rawMannequin = mannequin.mannequinImageUrl.startsWith('http')
      ? mannequin.mannequinImageUrl
      : `${import.meta.env.VITE_API_BASE_URL ?? ''}${mannequin.mannequinImageUrl}`
    // eslint-disable-next-line no-console
    console.info('[mannequin] cutout saved (open this URL to verify the file)', { mannequinImageUrl: rawMannequin })
    // Cache-bust only for the browser <img> preview. Re-detection uses the canonical URL so the API
    // always resolves the same file without query-string edge cases.
    const sep = rawMannequin.includes('?') ? '&' : '?'
    const mannequinUrl = `${rawMannequin}${sep}t=${Date.now()}`
    let detected: DetectionResult | null = null
    const reDetectWarnings: string[] = []
    try {
      detected = await detectItemsFromImage(rawMannequin, {
        mode: 'clothing_only',
        aiAssist: true,
      })
    } catch (e1) {
      // eslint-disable-next-line no-console
      console.warn('[mannequin] re-detect (clothing_only) failed', e1)
      reDetectWarnings.push(
        `Re-detection (clothing_only) failed: ${e1 instanceof Error ? e1.message : String(e1)}. Retrying with keep_selected_person.`,
      )
      try {
        detected = await detectItemsFromImage(rawMannequin, {
          mode: 'keep_selected_person',
          aiAssist: true,
        })
      } catch (e2) {
        // eslint-disable-next-line no-console
        console.warn('[mannequin] re-detect (keep_selected_person) failed', e2)
        reDetectWarnings.push(
          `Re-detection fallback failed: ${e2 instanceof Error ? e2.message : String(e2)}. The mannequin image is still shown above — you can add items manually or retry detection.`,
        )
        detected = null
      }
    }
    const heroSelections = detected
      ? buildDefaultDetectionSelection(detected.detected)
      : new Set(pendingDetectionSelections)
    set({
      pendingDetectionImageUrl: mannequinUrl,
      pendingDetection: {
        ...(detected ?? pendingDetection),
        source_image_url: mannequinUrl,
        auto_blurred_image_url: null,
        source_image_stage: 'blurred_fallback',
        warnings: [
          ...((detected ?? pendingDetection).warnings ?? []),
          ...reDetectWarnings,
          'Mannequin preview generated from selected SAM region.',
        ],
      },
      pendingDetectionSelections: heroSelections,
      editorHistoryPast: [...get().editorHistoryPast, before].slice(-30),
      editorHistoryFuture: [],
    })
    const detection = get().pendingDetection
    if (!detection) throw new Error('State lost after mannequin generation.')
    return detection
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
      editorHistoryPast: [],
      editorHistoryFuture: [],
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
      editorHistoryPast: [],
      editorHistoryFuture: [],
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
      editorHistoryPast: [],
      editorHistoryFuture: [],
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
