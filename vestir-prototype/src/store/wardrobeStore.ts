import { create } from 'zustand'
import { mockItems, mockOutfits, mockWardrobes } from '../data/mock'
import type {
  ActiveView,
  Category,
  DetectedGarment,
  DetectionResult,
  Item,
  Outfit,
  SubjectFilterMode,
  Wardrobe,
} from '../types/index'
import { defaultPipelineAdapters, detectItemsFromImage } from '../lib/pipeline/adapters'
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
  confirmDetectedItems: () => Promise<void>
  dismissDetection: () => void
  addPendingItemsFromFiles: (files: FileList | null) => Promise<void>
  runHybridAiPipeline: () => Promise<void>
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.readAsDataURL(file)
  })
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
    const result = await detectItemsFromImage(imageUrl, {
      mode: 'keep_selected_person',
      aiAssist: true,
    })
    const heroSelections = new Set(result.detected.filter((g) => g.is_hero).map((g) => g.id))
    const defaultPeople = new Set((result.person_candidates ?? []).map((p) => p.id))
    set({
      pendingDetectionImageUrl: imageUrl,
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
    const heroSelections = new Set(result.detected.filter((g) => g.is_hero).map((g) => g.id))
    set({
      pendingDetection: result,
      pendingDetectionSelections: heroSelections,
    })
    return result
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
      item_type: garment.label,
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
}))
