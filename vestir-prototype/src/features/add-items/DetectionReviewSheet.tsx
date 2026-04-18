import { CheckCircle, Circle, Star } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react'
import { toast } from 'sonner'
import { refineMaskWithSam } from '../../lib/pipeline/adapters'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { BlurQualityPreset, DetectedGarment, NormalizedBBox, SubjectFilterMode } from '../../types/index'

type SelectionTool = 'polygon' | 'rectangle'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

/** In dev, load /storage/* through the Vite dev server proxy (same origin as the UI). */
function storagePathForDevProxy(url: string): string | null {
  if (!import.meta.env.DEV) return null
  const m = url.match(/^https?:\/\/(?:127\.0\.0\.1|localhost)(?::\d+)?(\/storage\/[^?#]*)((?:\?|#).*)?$/i)
  if (!m) return null
  return `${m[1]}${m[2] ?? ''}`
}

function absoluteUrl(url: string) {
  if (!url) return url
  if (url.startsWith('data:')) return url
  const resolved = url.startsWith('http') ? url : `${API_BASE}${url}`
  const proxied = storagePathForDevProxy(resolved)
  if (proxied) return proxied
  return resolved
}

interface DetectionCardProps {
  garment: DetectedGarment
  selected: boolean
  onToggle: () => void
}

function polygonArea(points: Array<{ x: number; y: number }>) {
  if (points.length < 3) return 0
  let sum = 0
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i]
    const b = points[(i + 1) % points.length]
    sum += a.x * b.y - b.x * a.y
  }
  return Math.abs(sum) / 2
}

function pointInPolygon(point: { x: number; y: number }, polygon: Array<{ x: number; y: number }>) {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x
    const yi = polygon[i].y
    const xj = polygon[j].x
    const yj = polygon[j].y
    const intersect = ((yi > point.y) !== (yj > point.y))
      && (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || Number.EPSILON) + xi)
    if (intersect) inside = !inside
  }
  return inside
}

function normalizedBboxFromPolygon(poly: Array<{ x: number; y: number }>): NormalizedBBox {
  const xs = poly.map((p) => p.x)
  const ys = poly.map((p) => p.y)
  return {
    x1: Math.max(0, Math.min(...xs)),
    y1: Math.max(0, Math.min(...ys)),
    x2: Math.min(1, Math.max(...xs)),
    y2: Math.min(1, Math.max(...ys)),
  }
}

function DetectionCard({ garment, selected, onToggle }: DetectionCardProps) {
  return (
    <button
      type="button"
      className={`detection-card${selected ? ' detection-card--selected' : ''}`}
      onClick={onToggle}
      aria-pressed={selected}
    >
      <div className="detection-card__img-wrap">
        <img
          src={absoluteUrl(garment.crop_url)}
          alt={garment.label}
          className="detection-card__img"
        />
        {garment.is_hero && (
          <span className="detection-card__hero-badge" title="Most prominent item">
            <Star size={10} fill="currentColor" />
          </span>
        )}
        <span className="detection-card__check">
          {selected
            ? <CheckCircle size={20} fill="var(--primary)" color="white" />
            : <Circle size={20} color="var(--border)" />}
        </span>
      </div>
      <p className="detection-card__label">{garment.label}</p>
      {garment.partially_visible && (
        <p className="detection-card__warn">Partly visible</p>
      )}
    </button>
  )
}

export function DetectionReviewSheet() {
  const {
    pendingDetection,
    pendingDetectionImageUrl,
    subjectFilterMode,
    blurQualityPreset,
    subjectFilterPersonSelections,
    subjectFilterMaskPolygon,
    pendingDetectionSelections,
    setSubjectFilterMode,
    setBlurQualityPreset,
    toggleSubjectFilterPersonSelection,
    setSubjectFilterMaskPolygon,
    applySubjectFilter,
    applyManualBlurToPending,
    generateMannequinToPending,
    toggleDetectionSelection,
    undoEditorState,
    redoEditorState,
    editorHistoryPast,
    editorHistoryFuture,
    confirmDetectedItems,
    dismissDetection,
  } = useWardrobeStore()
  const editorRef = useRef<HTMLDivElement | null>(null)
  const previewLoadWarned = useRef(false)
  const [dragPointIndex, setDragPointIndex] = useState<number | null>(null)
  const [polygonClosed, setPolygonClosed] = useState(false)
  const [hoverPersonId, setHoverPersonId] = useState<string | null>(null)
  const [focusRectStart, setFocusRectStart] = useState<{ x: number; y: number } | null>(null)
  const [focusRectDraft, setFocusRectDraft] = useState<{
    x1: number
    y1: number
    x2: number
    y2: number
  } | null>(null)
  const [selectionTool, setSelectionTool] = useState<SelectionTool>('polygon')
  const [lastSamHintBbox, setLastSamHintBbox] = useState<NormalizedBBox | null>(null)
  /**
   * Tracks long-running async work so the UI can show a blocking overlay
   * (spinner + elapsed timer) instead of the user clicking the scrim in
   * confusion and ending up back on the main page while the request is
   * still in flight.
   */
  const [busyTask, setBusyTask] = useState<{
    label: string
    sub?: string
    startedAt: number
  } | null>(null)
  const [busyElapsedMs, setBusyElapsedMs] = useState(0)
  /** Prevents overlapping mannequin runs — a second click used to leave Sonner + overlay out of sync. */
  const mannequinInFlightRef = useRef(false)

  useEffect(() => {
    if (!busyTask) {
      setBusyElapsedMs(0)
      return undefined
    }
    setBusyElapsedMs(Date.now() - busyTask.startedAt)
    const timer = window.setInterval(() => {
      setBusyElapsedMs(Date.now() - busyTask.startedAt)
    }, 250)
    return () => window.clearInterval(timer)
  }, [busyTask])

  if (!pendingDetection) return null

  const {
    detected,
    scene_track,
    person_candidates = [],
    face_candidates = [],
    source_image_url,
    auto_blurred_image_url,
    source_image_stage,
  } = pendingDetection
  const latestWarning = pendingDetection.warnings?.[pendingDetection.warnings.length - 1]
  const selectedCount = pendingDetectionSelections.size
  const allSelected = detected.every((g) => pendingDetectionSelections.has(g.id))
  const allPeopleSelected = person_candidates.length > 0
    && person_candidates.every((p) => subjectFilterPersonSelections.has(p.id))
  const polygonPointLabel = `${subjectFilterMaskPolygon.length} points`
  const selectedAreaPct = Math.round(polygonArea(subjectFilterMaskPolygon) * 10000) / 100
  const polygonPointsSvg = subjectFilterMaskPolygon
    .map((p) => `${(p.x * 100).toFixed(2)},${(p.y * 100).toFixed(2)}`)
    .join(' ')
  const busyElapsedSeconds = busyTask ? Math.max(0, Math.floor(busyElapsedMs / 1000)) : 0
  const isBusy = Boolean(busyTask)

  // Use store-tracked current preview URL first (latest processed image),
  // then fallback to payload fields.
  const activeImageUrl = pendingDetectionImageUrl || source_image_url || auto_blurred_image_url || ''
  const previewSrc = absoluteUrl(activeImageUrl)

  useEffect(() => {
    previewLoadWarned.current = false
  }, [previewSrc])

  function changeFilterMode(mode: SubjectFilterMode) {
    setSubjectFilterMode(mode)
  }

  function changeBlurPreset(preset: BlurQualityPreset) {
    setBlurQualityPreset(preset)
  }

  function toggleAllPeople() {
    if (allPeopleSelected) {
      person_candidates.forEach((candidate) => {
        if (subjectFilterPersonSelections.has(candidate.id)) toggleSubjectFilterPersonSelection(candidate.id)
      })
    } else {
      person_candidates.forEach((candidate) => {
        if (!subjectFilterPersonSelections.has(candidate.id)) toggleSubjectFilterPersonSelection(candidate.id)
      })
    }
  }

  async function autoSelectPersonWithSam(personId: string, clickPoint?: { x: number; y: number }) {
    const candidate = person_candidates.find((p) => p.id === personId)
    if (!candidate) return
    try {
      const boxes = [candidate.bbox]
      if (clickPoint) {
        const half = 0.13
        boxes.unshift({
          x1: Math.max(candidate.bbox.x1, clickPoint.x - half),
          y1: Math.max(candidate.bbox.y1, clickPoint.y - half),
          x2: Math.min(candidate.bbox.x2, clickPoint.x + half),
          y2: Math.min(candidate.bbox.y2, clickPoint.y + half),
        })
      }
      const polygons = await refineMaskWithSam(activeImageUrl, boxes)
      const valid = (polygons ?? []).filter((poly) => poly.length >= 3)
      if (!valid.length) return
      const picked = clickPoint
        ? valid
          .filter((poly) => pointInPolygon(clickPoint, poly))
          .sort((a, b) => polygonArea(a) - polygonArea(b))[0] ?? valid[0]
        : valid[0]
      if (picked && picked.length >= 3) {
        setSubjectFilterMaskPolygon(picked)
        setPolygonClosed(true)
      }
    } catch {
      // Keep interaction non-blocking; bbox-based person focus still works even if SAM fails.
    }
  }

  function handlePersonChipClick(personId: string) {
    const willSelect = !subjectFilterPersonSelections.has(personId)
    toggleSubjectFilterPersonSelection(personId)
    if (willSelect) {
      void autoSelectPersonWithSam(personId)
      return
    }
    const stillSelectedCount = subjectFilterPersonSelections.size - 1
    if (stillSelectedCount <= 0) {
      setSubjectFilterMaskPolygon([])
      setPolygonClosed(false)
    }
  }

  function pointFromPointerEvent(clientX: number, clientY: number) {
    const container = editorRef.current
    if (!container) return null
    const rect = container.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height)),
    }
  }

  function overlapRatio(
    a: { x1: number; y1: number; x2: number; y2: number },
    b: { x1: number; y1: number; x2: number; y2: number },
  ) {
    const x1 = Math.max(a.x1, b.x1)
    const y1 = Math.max(a.y1, b.y1)
    const x2 = Math.min(a.x2, b.x2)
    const y2 = Math.min(a.y2, b.y2)
    if (x2 <= x1 || y2 <= y1) return 0
    const inter = (x2 - x1) * (y2 - y1)
    const base = Math.max(1e-6, (a.x2 - a.x1) * (a.y2 - a.y1))
    return inter / base
  }

  async function runSamFromFocusRect(rect: { x1: number; y1: number; x2: number; y2: number }) {
    setBusyTask({
      label: 'SAM is isolating your selection…',
      sub: 'First run can take 5–20 s while the model warms up.',
      startedAt: Date.now(),
    })
    let polygons: Array<Array<{ x: number; y: number }>> | null = null
    try {
      polygons = await refineMaskWithSam(activeImageUrl, [rect])
    } catch (error) {
      setBusyTask(null)
      toast.error(error instanceof Error ? error.message : 'SAM refinement failed')
      return
    }
    const best = (polygons ?? [])
      .filter((poly) => poly.length >= 3)
      .sort((a, b) => polygonArea(a) - polygonArea(b))[0]
    if (!best) {
      setBusyTask(null)
      toast.error('SAM could not isolate that person. Try a tighter rectangle.')
      return
    }
    setSubjectFilterMaskPolygon(best)
    setPolygonClosed(true)
    if (person_candidates.length > 0) {
      const pick = person_candidates
        .slice()
        .sort((a, b) => overlapRatio(rect, b.bbox) - overlapRatio(rect, a.bbox))[0]
      if (pick) {
        person_candidates.forEach((candidate) => {
          const selected = subjectFilterPersonSelections.has(candidate.id)
          if (candidate.id === pick.id && !selected) toggleSubjectFilterPersonSelection(candidate.id)
          if (candidate.id !== pick.id && selected) toggleSubjectFilterPersonSelection(candidate.id)
        })
      }
    }
    setLastSamHintBbox({
      x1: Math.min(rect.x1, rect.x2),
      y1: Math.min(rect.y1, rect.y2),
      x2: Math.max(rect.x1, rect.x2),
      y2: Math.max(rect.y1, rect.y2),
    })
    setBusyTask(null)
    toast.success(
      person_candidates.length > 0
        ? 'Person selected from rectangle with SAM'
        : 'Region captured with SAM. Use Refine mask or Generate mannequin next.',
    )
  }

  function handleEditorPointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    if (selectionTool !== 'rectangle') return
    if (e.button !== 0) return
    e.preventDefault()
    try {
      e.currentTarget.setPointerCapture(e.pointerId)
    } catch {
      /* ignore */
    }
    const point = pointFromPointerEvent(e.clientX, e.clientY)
    if (!point) return
    setFocusRectStart(point)
    setFocusRectDraft({ x1: point.x, y1: point.y, x2: point.x, y2: point.y })
  }

  function handleEditorPointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    if (selectionTool !== 'rectangle' || focusRectStart === null) return
    const point = pointFromPointerEvent(e.clientX, e.clientY)
    if (!point) return
    setFocusRectDraft({
      x1: Math.min(focusRectStart.x, point.x),
      y1: Math.min(focusRectStart.y, point.y),
      x2: Math.max(focusRectStart.x, point.x),
      y2: Math.max(focusRectStart.y, point.y),
    })
  }

  function handleEditorPointerUp(e: ReactPointerEvent<HTMLDivElement>) {
    if (selectionTool !== 'rectangle') return
    try {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId)
      }
    } catch {
      /* ignore */
    }
    const rect = focusRectDraft
    setFocusRectStart(null)
    setFocusRectDraft(null)
    if (!rect) return
    if ((rect.x2 - rect.x1) * (rect.y2 - rect.y1) < 0.004) {
      toast.message('Draw a slightly larger rectangle around one person or garment.')
      return
    }
    void runSamFromFocusRect(rect)
  }

  function handleEditorPointerCancel(e: ReactPointerEvent<HTMLDivElement>) {
    if (selectionTool !== 'rectangle') return
    try {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId)
      }
    } catch {
      /* ignore */
    }
    setFocusRectStart(null)
    setFocusRectDraft(null)
  }

  function addPolygonPoint(e: ReactMouseEvent<HTMLDivElement>) {
    if (selectionTool === 'rectangle') return
    if (dragPointIndex !== null || polygonClosed) return
    const container = editorRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    if (subjectFilterMaskPolygon.length >= 3) {
      const first = subjectFilterMaskPolygon[0]
      const distance = Math.hypot(first.x - x, first.y - y)
      if (distance < 0.02) {
        setPolygonClosed(true)
        return
      }
    }
    setSubjectFilterMaskPolygon([...subjectFilterMaskPolygon, { x, y }])
  }

  function updatePoint(clientX: number, clientY: number) {
    if (dragPointIndex === null) return
    const container = editorRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const y = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height))
    const next = [...subjectFilterMaskPolygon]
    next[dragPointIndex] = { x, y }
    setSubjectFilterMaskPolygon(next)
  }

  useEffect(() => {
    if (dragPointIndex === null) return undefined
    const onMove = (event: globalThis.MouseEvent) => {
      updatePoint(event.clientX, event.clientY)
    }
    const onPointerMove = (event: PointerEvent) => {
      updatePoint(event.clientX, event.clientY)
    }
    const onUp = () => {
      setDragPointIndex(null)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('pointerup', onUp)
    }
  }, [dragPointIndex, subjectFilterMaskPolygon])

  function toggleAll() {
    if (allSelected) {
      detected.forEach((g) => {
        if (pendingDetectionSelections.has(g.id)) toggleDetectionSelection(g.id)
      })
    } else {
      detected.forEach((g) => {
        if (!pendingDetectionSelections.has(g.id)) toggleDetectionSelection(g.id)
      })
    }
  }

  function handleConfirm() {
    const label = selectedCount === 1 ? '1 item' : `${selectedCount} items`
    toast.success(`Adding ${label} to your wardrobe…`)
    confirmDetectedItems()
  }

  function clearPolygon() {
    setSubjectFilterMaskPolygon([])
    setPolygonClosed(false)
    setLastSamHintBbox(null)
  }

  function undoPoint() {
    if (!subjectFilterMaskPolygon.length) return
    setSubjectFilterMaskPolygon(subjectFilterMaskPolygon.slice(0, -1))
  }

  function toggleClosePolygon() {
    if (subjectFilterMaskPolygon.length < 3) return
    setPolygonClosed((value) => !value)
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const isUndo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z' && !event.shiftKey
      const isRedo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z' && event.shiftKey
      if (!isUndo && !isRedo) return
      event.preventDefault()
      if (isUndo) undoEditorState()
      if (isRedo) redoEditorState()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [undoEditorState, redoEditorState])

  async function handleApplySubjectFilter() {
    const toastId = toast.loading('Applying subject filter...')
    setBusyTask({ label: 'Applying subject filter…', startedAt: Date.now() })
    try {
      const result = await applySubjectFilter()
      if (!result) {
        toast.error('No pending image to filter.', { id: toastId })
        return
      }
      if (result.warnings?.length) {
        toast.warning(result.warnings[0], { id: toastId })
        return
      }
      toast.success('Updated detection with subject filtering', { id: toastId })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Subject filtering failed', { id: toastId })
    } finally {
      setBusyTask(null)
    }
  }

  function handleScrimPointerDown() {
    if (busyTask) {
      toast.message(
        `${busyTask.label} Please wait — tap outside again once it finishes.`,
      )
      return
    }
    dismissDetection()
  }

  async function handleApplyManualBlur() {
    if (subjectFilterMaskPolygon.length < 3) {
      toast.warning('Draw a polygon around missed face/region first.')
      return
    }
    const xs = subjectFilterMaskPolygon.map((p) => p.x)
    const ys = subjectFilterMaskPolygon.map((p) => p.y)
    const box = {
      x1: Math.max(0, Math.min(...xs)),
      y1: Math.max(0, Math.min(...ys)),
      x2: Math.min(1, Math.max(...xs)),
      y2: Math.min(1, Math.max(...ys)),
    }
    const toastId = toast.loading('Applying manual blur...')
    setBusyTask({ label: 'Applying manual blur…', startedAt: Date.now() })
    try {
      const result = await applyManualBlurToPending([box])
      if (!result) {
        toast.error('Manual blur failed.', { id: toastId })
        return
      }
      toast.success('Manual blur applied with selected quality.', { id: toastId })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Manual blur failed', { id: toastId })
    } finally {
      setBusyTask(null)
    }
  }

  /**
   * Ensures a *tight* SAM polygon exists before we ask the server to cut a mannequin.
   *
   * Without this, clicking "Generate mannequin" with only an auto-selected person chip
   * would send the whole-person bbox (often 80–95% of the frame). The server would then
   * composite that near-full rectangle onto white and return a JPEG that looks identical
   * to the source, making the user think "nothing happened" even though the API returned 200.
   *
   * We try SAM on (in order): the current rectangle hint, selected people, then the first
   * garment bbox if the server exposed `detected_regions` (new in this build).
   */
  async function ensureTightMaskBeforeMannequin(): Promise<boolean> {
    if (polygonClosed && subjectFilterMaskPolygon.length >= 3) return true

    const selectedPeople = person_candidates.filter((p) => subjectFilterPersonSelections.has(p.id))
    const hintBoxes: NormalizedBBox[] = lastSamHintBbox
      ? [lastSamHintBbox]
      : selectedPeople.length > 0
        ? selectedPeople.map((p) => p.bbox)
        : []

    if (hintBoxes.length === 0) {
      toast.error(
        'Draw a rectangle around the person or garment first, then press Generate mannequin.',
      )
      return false
    }

    try {
      const polygons = await refineMaskWithSam(activeImageUrl, hintBoxes)
      const valid = (polygons ?? []).filter((poly) => poly.length >= 3)
      if (!valid.length) {
        toast.error(
          'SAM could not isolate the garment. Draw a tighter rectangle around the clothing and try again.',
        )
        return false
      }
      const picked = [...valid].sort((a, b) => polygonArea(a) - polygonArea(b))[0]
      const bboxArea = (() => {
        const b = normalizedBboxFromPolygon(picked)
        return Math.max(0, (b.x2 - b.x1) * (b.y2 - b.y1))
      })()
      if (bboxArea > 0.97) {
        toast.error(
          'SAM returned a near-full-frame mask. Draw a tighter rectangle around just the garment and retry.',
        )
        return false
      }
      setSubjectFilterMaskPolygon(picked)
      setPolygonClosed(true)
      return true
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'SAM refinement failed')
      return false
    }
  }

  async function handleGenerateMannequin() {
    if (mannequinInFlightRef.current) {
      toast.message('Mannequin generation is already running — wait for it to finish.')
      return
    }
    mannequinInFlightRef.current = true
    const toastId = toast.loading('Preparing mannequin cutout...')
    setBusyTask({
      label: 'Isolating garment with SAM…',
      sub: 'This usually takes 10–30 s on first run (SAM warms up). Please keep this sheet open.',
      startedAt: Date.now(),
    })
    try {
      const masked = await ensureTightMaskBeforeMannequin()
      if (!masked) {
        toast.dismiss(toastId)
        return
      }
      setBusyTask({
        label: 'Building mannequin on white background…',
        sub: 'Compositing cutout, then re-running clothing detection. 3–20 s typical.',
        startedAt: Date.now(),
      })
      toast.loading('Generating mannequin on white background...', { id: toastId })
      const result = await generateMannequinToPending()
      if (!result) {
        toast.error('No pending image to generate from.', { id: toastId })
        return
      }
      // Clear overlay + dismiss loading toast *before* success so Sonner never shows
      // a loading spinner and a success message on the same toast id in one frame.
      setBusyTask(null)
      toast.dismiss(toastId)
      toast.success('Mannequin generated. Background is white and detections refreshed.')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Mannequin generation failed', { id: toastId })
    } finally {
      setBusyTask(null)
      mannequinInFlightRef.current = false
    }
  }

  async function handleSamRefineMask() {
    const selectedPeople = person_candidates.filter((p) => subjectFilterPersonSelections.has(p.id))
    let boxes: NormalizedBBox[] = []
    if (selectedPeople.length > 0) {
      boxes = selectedPeople.map((p) => p.bbox)
    } else if (polygonClosed && subjectFilterMaskPolygon.length >= 3) {
      boxes = [normalizedBboxFromPolygon(subjectFilterMaskPolygon)]
    } else if (lastSamHintBbox) {
      boxes = [lastSamHintBbox]
    } else {
      toast.warning(
        'Draw a rectangle (Rectangle tool), close a polygon, or select people — then SAM can refine the mask.',
      )
      return
    }
    const toastId = toast.loading('Refining mask with SAM...')
    setBusyTask({
      label: 'Refining mask with SAM…',
      sub: 'Segment Anything is processing your selection. 2–10 s typical.',
      startedAt: Date.now(),
    })
    try {
      const polygons = await refineMaskWithSam(activeImageUrl, boxes)
      const valid = (polygons ?? []).filter((poly) => poly.length >= 3)
      if (!valid.length) {
        toast.error('SAM could not generate a clean polygon. Try adjusting selection.', { id: toastId })
        return
      }
      const picked =
        boxes.length === 1
          ? [...valid].sort((a, b) => polygonArea(a) - polygonArea(b))[0]
          : valid[0]
      if (!picked || picked.length < 3) {
        toast.error('SAM could not generate a clean polygon. Try adjusting selection.', { id: toastId })
        return
      }
      setSubjectFilterMaskPolygon(picked)
      setPolygonClosed(true)
      if (boxes.length === 1) {
        setLastSamHintBbox(boxes[0])
      }
      toast.success('Mask refined with SAM. You can still edit points manually.', { id: toastId })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'SAM refinement failed', { id: toastId })
    } finally {
      setBusyTask(null)
    }
  }

  return (
    <div className="sheet-scrim" onClick={handleScrimPointerDown}>
      <div
        className="sheet detection-sheet"
        onClick={(e) => e.stopPropagation()}
        style={{ paddingBottom: '24px' }}
      >
        <div className="sheet-handle" />
        <div className="detection-sheet__header">
          <div>
            <h3 style={{ margin: 0, fontSize: '15px' }}>
              We found {detected.length} item{detected.length !== 1 ? 's' : ''}
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '12px', opacity: 0.6 }}>
              {scene_track === 'worn' ? 'Outfit photo' : scene_track === 'flat_lay' ? 'Flat lay' : 'Photo'} · tap the ones that look right
            </p>
            <p style={{ margin: '2px 0 0', fontSize: '12px', opacity: 0.6 }}>
              Faces detected: {face_candidates.length}
            </p>
            <p style={{ margin: '6px 0 0', fontSize: '11px', opacity: 0.55, maxWidth: 420 }}>
              After AI runs, you may be prompted on the item page to confirm category (e.g. dress vs top) before
              closet embedding — that keeps extension match and search accurate.
            </p>
            {source_image_stage === 'tryoff' && (
              <p style={{ margin: '6px 0 0', fontSize: '12px', color: 'var(--primary)' }}>
                Showing try-off product shot (garment on white).
              </p>
            )}
          </div>
          <button
            type="button"
            className="detection-sheet__toggle-all"
            onClick={toggleAll}
          >
            {allSelected ? 'Clear selection' : 'Select all'}
          </button>
        </div>

        <div className="subject-filter">
          <h4 className="subject-filter__title">Subject filtering</h4>
          <p className="subject-filter__hint">Choose who stays, or keep clothing regions only.</p>
          <div className="subject-filter__mode-row">
            <button
              type="button"
              className={`subject-filter__mode ${subjectFilterMode === 'keep_selected_person' ? 'is-active' : ''}`}
              onClick={() => changeFilterMode('keep_selected_person')}
            >
              Keep selected people
            </button>
            <button
              type="button"
              className={`subject-filter__mode ${subjectFilterMode === 'clothing_only' ? 'is-active' : ''}`}
              onClick={() => changeFilterMode('clothing_only')}
            >
              Clothing only
            </button>
            <button
              type="button"
              className={`subject-filter__mode ${subjectFilterMode === 'focus_person_blur_others' ? 'is-active' : ''}`}
              onClick={() => changeFilterMode('focus_person_blur_others')}
            >
              Focus person (blur others)
            </button>
          </div>
          <div className="subject-filter__mode-row">
            <span className="subject-filter__hint" style={{ margin: 0 }}>Blur quality</span>
            {(['soft', 'pro', 'strong'] as BlurQualityPreset[]).map((preset) => (
              <button
                key={preset}
                type="button"
                className={`subject-filter__mode ${blurQualityPreset === preset ? 'is-active' : ''}`}
                onClick={() => changeBlurPreset(preset)}
              >
                {preset}
              </button>
            ))}
          </div>
          {person_candidates.length > 0 && (
            <>
              <div className="subject-filter__person-head">
                <span>Select people</span>
                <button type="button" className="subject-filter__link" onClick={toggleAllPeople}>
                  {allPeopleSelected ? 'Clear' : 'Select all'}
                </button>
              </div>
              <div className="subject-filter__chips">
                {person_candidates.map((candidate) => (
                  <button
                    key={candidate.id}
                    type="button"
                    className={`subject-filter__chip ${subjectFilterPersonSelections.has(candidate.id) ? 'is-active' : ''}`}
                    onClick={() => handlePersonChipClick(candidate.id)}
                    onMouseEnter={() => setHoverPersonId(candidate.id)}
                    onMouseLeave={() => setHoverPersonId(null)}
                  >
                    {candidate.label}
                  </button>
                ))}
              </div>
            </>
          )}
          <div className="subject-filter__polygon">
            <p className="subject-filter__hint" style={{ marginBottom: 6 }}>
              Faces are blurred automatically when detected (preview uses the processed image).
              Use Rectangle (SAM) to outline one person or garment, then Generate mannequin; detection refreshes from that cutout.
            </p>
            <div className="subject-filter__mode-row" style={{ flexWrap: 'wrap', gap: 6 }}>
              <span className="subject-filter__hint" style={{ margin: 0 }}>Mask tool</span>
              <button
                type="button"
                className={`subject-filter__mode ${selectionTool === 'polygon' ? 'is-active' : ''}`}
                onClick={() => setSelectionTool('polygon')}
              >
                Polygon
              </button>
              <button
                type="button"
                className={`subject-filter__mode ${selectionTool === 'rectangle' ? 'is-active' : ''}`}
                onClick={() => setSelectionTool('rectangle')}
              >
                Rectangle (SAM)
              </button>
            </div>
            <p className="subject-filter__hint" style={{ marginBottom: 6 }}>
              {selectionTool === 'polygon'
                ? 'Polygon: click to add points, drag points to adjust, click first point to close.'
                : 'Rectangle: drag on the photo to run SAM on that region (works for any subject filter mode).'}
            </p>
            <div className="subject-filter__metrics">
              <span>{polygonPointLabel}</span>
              <span>{polygonClosed ? 'Closed' : 'Open'} polygon</span>
              <span>{selectedAreaPct}% selected</span>
            </div>
            {latestWarning && (
              <p className="subject-filter__hint" style={{ marginBottom: 6 }}>
                {latestWarning}
              </p>
            )}
            <div
              ref={editorRef}
              className="subject-filter__editor"
              style={{ cursor: isBusy ? 'wait' : selectionTool === 'rectangle' ? 'crosshair' : undefined }}
              onClick={isBusy ? undefined : addPolygonPoint}
              onPointerDown={isBusy ? undefined : handleEditorPointerDown}
              onPointerMove={isBusy ? undefined : handleEditorPointerMove}
              onPointerUp={isBusy ? undefined : handleEditorPointerUp}
              onPointerCancel={isBusy ? undefined : handleEditorPointerCancel}
              title={
                isBusy
                  ? busyTask?.label ?? 'Working…'
                  : selectionTool === 'rectangle'
                    ? 'Drag a rectangle over one person or garment; SAM builds the mask'
                    : 'Click to add polygon points'
              }
            >
              <img
                key={previewSrc}
                src={previewSrc}
                alt="Source for subject filtering and mannequin preview"
                className="subject-filter__image"
                draggable={false}
                onError={() => {
                  if (previewLoadWarned.current) return
                  previewLoadWarned.current = true
                  toast.error(
                    'Preview image failed to load. In dev, leave VITE_API_BASE_URL empty so /storage uses the Vite proxy, or ensure the API is running on the URL in .env.',
                  )
                }}
              />
              {busyTask && (
                <div
                  className="subject-filter__busy"
                  role="status"
                  aria-live="polite"
                  onClick={(e) => e.stopPropagation()}
                  onPointerDown={(e) => e.stopPropagation()}
                >
                  <div className="subject-filter__busy-spinner" aria-hidden="true" />
                  <div className="subject-filter__busy-label">
                    {busyTask.label} {busyElapsedSeconds}s
                  </div>
                  {busyTask.sub && (
                    <div className="subject-filter__busy-sub">{busyTask.sub}</div>
                  )}
                </div>
              )}
              <svg className="subject-filter__overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
                {person_candidates.map((candidate) => {
                  const selected = subjectFilterPersonSelections.has(candidate.id)
                  const hovered = hoverPersonId === candidate.id
                  const bbox = candidate.bbox
                  return (
                    <rect
                      key={candidate.id}
                      x={bbox.x1 * 100}
                      y={bbox.y1 * 100}
                      width={(bbox.x2 - bbox.x1) * 100}
                      height={(bbox.y2 - bbox.y1) * 100}
                      className={`subject-filter__person-box ${selected ? 'is-selected' : ''} ${hovered ? 'is-hovered' : ''}`}
                    />
                  )
                })}
                {subjectFilterMaskPolygon.length >= 2 && (
                  polygonClosed
                    ? (
                      <polygon
                        points={polygonPointsSvg}
                        className="subject-filter__poly-fill"
                      />
                    )
                    : (
                      <polyline
                        points={polygonPointsSvg}
                        className="subject-filter__poly-line"
                      />
                    )
                )}
                {selectionTool === 'rectangle' && focusRectDraft && (
                  <rect
                    x={focusRectDraft.x1 * 100}
                    y={focusRectDraft.y1 * 100}
                    width={(focusRectDraft.x2 - focusRectDraft.x1) * 100}
                    height={(focusRectDraft.y2 - focusRectDraft.y1) * 100}
                    className="subject-filter__person-box is-selected"
                  />
                )}
                {subjectFilterMaskPolygon.map((point, idx) => (
                  <g
                    key={`${point.x}-${point.y}-${idx}`}
                    onMouseDown={(e) => {
                      e.stopPropagation()
                      setDragPointIndex(idx)
                    }}
                    onPointerDown={(e) => {
                      e.stopPropagation()
                      setDragPointIndex(idx)
                    }}
                    style={{ cursor: 'move' }}
                  >
                    <circle
                      cx={point.x * 100}
                      cy={point.y * 100}
                      r={idx === 0 ? 1.7 : 1.2}
                      className="subject-filter__point"
                    />
                  </g>
                ))}
              </svg>
            </div>
            <div className="subject-filter__actions">
              <button type="button" className="detection-sheet__toggle-all" onClick={undoPoint}>
                Undo point
              </button>
              <button
                type="button"
                className="detection-sheet__toggle-all"
                onClick={undoEditorState}
                disabled={editorHistoryPast.length === 0}
              >
                Undo
              </button>
              <button
                type="button"
                className="detection-sheet__toggle-all"
                onClick={redoEditorState}
                disabled={editorHistoryFuture.length === 0}
              >
                Redo
              </button>
              <button
                type="button"
                className="detection-sheet__toggle-all"
                onClick={toggleClosePolygon}
                disabled={subjectFilterMaskPolygon.length < 3}
              >
                {polygonClosed ? 'Reopen' : 'Close'} polygon
              </button>
              <button type="button" className="detection-sheet__toggle-all" onClick={clearPolygon}>
                Clear polygon
              </button>
              <button
                type="button"
                className="btn secondary"
                onClick={handleApplySubjectFilter}
                disabled={isBusy}
              >
                Apply filter
              </button>
              <button
                type="button"
                className="btn secondary"
                onClick={handleSamRefineMask}
                disabled={isBusy}
              >
                Refine mask (SAM)
              </button>
              <button
                type="button"
                className="btn secondary"
                onClick={handleApplyManualBlur}
                disabled={isBusy}
              >
                Apply manual blur
              </button>
              <button
                type="button"
                className="btn secondary"
                onClick={handleGenerateMannequin}
                disabled={isBusy}
              >
                {isBusy && busyTask?.label.toLowerCase().includes('mannequin')
                  ? `Generating mannequin… ${busyElapsedSeconds}s`
                  : isBusy
                    ? 'Working…'
                    : 'Generate mannequin (white bg)'}
              </button>
            </div>
            {/* Try-off temporarily disabled to evaluate SAM-based mannequin extraction quality. */}
          </div>
        </div>

        <div className="detection-grid">
          {detected.map((garment) => (
            <DetectionCard
              key={garment.id}
              garment={garment}
              selected={pendingDetectionSelections.has(garment.id)}
              onToggle={() => toggleDetectionSelection(garment.id)}
            />
          ))}
        </div>

        <div className="detection-sheet__actions">
          <button
            type="button"
            className="btn"
            disabled={selectedCount === 0 || isBusy}
            onClick={handleConfirm}
            style={{ flex: 1 }}
          >
            Add {selectedCount === 1 ? '1 item' : `${selectedCount} items`}
          </button>
          <button
            type="button"
            className="btn secondary"
            onClick={() => {
              if (isBusy) {
                toast.message('Hold on — ' + (busyTask?.label ?? 'a task is still running…'))
                return
              }
              dismissDetection()
            }}
            style={{ flex: 0, padding: '10px 14px' }}
            aria-disabled={isBusy}
          >
            Not now
          </button>
        </div>
      </div>
    </div>
  )
}
