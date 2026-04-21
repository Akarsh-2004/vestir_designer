import { CheckCircle, Circle, Star } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react'
import { toast } from 'sonner'
import { refineMaskWithSam } from '../../lib/pipeline/adapters'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { BlurQualityPreset, DetectedGarment, NormalizedBBox, SubjectFilterMode } from '../../types/index'

type SelectionTool = 'polygon' | 'rectangle' | 'tap'

/**
 * SAM-style overlay palette — each detected garment gets one color from this
 * rotation (pink/green/blue/yellow/orange/teal) so the user can see at a
 * glance which mask belongs to which item, like the LabelErr/Meta SAM2
 * visualization. Chosen to be distinguishable from the brand accent AND from
 * each other even on low-contrast photos.
 */
const MASK_PALETTE = [
  { fill: 'rgba(236, 72, 153, 0.32)', stroke: '#ec4899' }, // pink
  { fill: 'rgba(34, 197, 94, 0.32)', stroke: '#22c55e' },  // green
  { fill: 'rgba(59, 130, 246, 0.32)', stroke: '#3b82f6' }, // blue
  { fill: 'rgba(234, 179, 8, 0.36)', stroke: '#eab308' },  // yellow
  { fill: 'rgba(249, 115, 22, 0.32)', stroke: '#f97316' }, // orange
  { fill: 'rgba(20, 184, 166, 0.32)', stroke: '#14b8a6' }, // teal
  { fill: 'rgba(168, 85, 247, 0.32)', stroke: '#a855f7' }, // violet
] as const

function paletteFor(index: number) {
  return MASK_PALETTE[((index % MASK_PALETTE.length) + MASK_PALETTE.length) % MASK_PALETTE.length]
}

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

interface PersonAssignmentInfo {
  personId: string | null
  personLabel: string | null
  confidence: number
  requiresConfirmation: boolean
  isOverride: boolean
}

interface DetectionCardProps {
  garment: DetectedGarment
  selected: boolean
  onToggle: () => void
  assignment?: PersonAssignmentInfo
  people?: Array<{ id: string; label: string }>
  onReassign?: (personId: string | null) => void
  /** Optional color dot shown in the top-left of the thumbnail that matches
   *  the SAM overlay color on the photo above. Helps the user link cards to
   *  their corresponding mask when there are multiple garments. */
  overlayColor?: string
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

function DetectionCard({ garment, selected, onToggle, assignment, people, onReassign, overlayColor }: DetectionCardProps) {
  const hasPeople = Array.isArray(people) && people.length > 0
  const assignedPerson = assignment?.personId
    ? people?.find((p) => p.id === assignment.personId)
    : null
  const assignmentTone = assignment?.requiresConfirmation && !assignment.isOverride
    ? 'detection-card__assignment--warn'
    : assignment?.isOverride
      ? 'detection-card__assignment--confirmed'
      : ''
  return (
    <div className={`detection-card${selected ? ' detection-card--selected' : ''}`}>
      <button
        type="button"
        className="detection-card__body"
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
          {overlayColor && (
            <span
              title="Color of this item's overlay on the photo above"
              style={{
                position: 'absolute',
                top: 6,
                left: 6,
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: overlayColor,
                border: '1.5px solid white',
                boxShadow: '0 0 0 1px rgba(0,0,0,0.2)',
              }}
            />
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
      {hasPeople && assignment ? (
        <div className={`detection-card__assignment ${assignmentTone}`}>
          <small>
            {assignedPerson ? assignedPerson.label : 'Unassigned'}
            {' · '}
            {Math.round(assignment.confidence * 100)}%
            {assignment.requiresConfirmation && !assignment.isOverride ? ' · needs check' : ''}
            {assignment.isOverride ? ' · confirmed' : ''}
          </small>
          <div className="detection-card__person-row">
            {people!.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`detection-card__person-chip${
                  assignment.personId === p.id ? ' is-active' : ''
                }`}
                onClick={(e) => {
                  e.stopPropagation()
                  onReassign?.(p.id)
                }}
              >
                {p.label}
              </button>
            ))}
            <button
              type="button"
              className={`detection-card__person-chip${
                assignment.personId == null ? ' is-active' : ''
              }`}
              onClick={(e) => {
                e.stopPropagation()
                onReassign?.(null)
              }}
              title="No person (flat lay or uncertain)"
            >
              None
            </button>
          </div>
        </div>
      ) : null}
    </div>
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
    cutGarmentFromActivePolygon,
    refineDetectedGarmentBackgrounds,
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
   * Per-garment SAM polygons, captured when the user runs "Remove backgrounds".
   * Rendered as SAM-style colored fills (pink/green/blue/yellow…) on the
   * canvas so the user can visually confirm what got segmented per item —
   * mirrors the LabelErr / Meta SAM2 demo visualization the user asked for.
   */
  const [maskOverlays, setMaskOverlays] = useState<Record<string, Array<{ x: number; y: number }>>>({})
  /** Lets the user hide overlays if they clutter the canvas. Default on. */
  const [showGarmentOverlays, setShowGarmentOverlays] = useState(true)
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
  const [mannequinHints, setMannequinHints] = useState({
    category: '',
    color: '',
    collar: '',
    sleeves: '',
    placket: '',
    fabric: '',
    details: '',
  })
  /** Prevents overlapping mannequin runs — a second click used to leave Sonner + overlay out of sync. */
  const mannequinInFlightRef = useRef(false)
  /** User-corrected garment→person assignments. Clears when the detection payload changes. */
  const [assignmentOverrides, setAssignmentOverrides] = useState<Record<string, string | null>>({})

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
    person_assignments = [],
    face_candidates = [],
    source_image_url,
    auto_blurred_image_url,
    source_image_stage,
  } = pendingDetection
  const personLookup = new Map(person_candidates.map((p, idx) => [p.id, { ...p, label: `Person ${String.fromCharCode(65 + idx)}` }]))
  const peopleForUi = [...personLookup.values()].map((p) => ({ id: p.id, label: p.label }))
  const assignmentMap = new Map(person_assignments.map((a) => [a.garment_id, a]))
  function getAssignment(garmentId: string): PersonAssignmentInfo | undefined {
    if (person_candidates.length === 0) return undefined
    const baseline = assignmentMap.get(garmentId)
    const overridePersonId = Object.prototype.hasOwnProperty.call(assignmentOverrides, garmentId)
      ? assignmentOverrides[garmentId]
      : undefined
    const personId = overridePersonId !== undefined ? overridePersonId : (baseline?.person_id ?? null)
    const isOverride = overridePersonId !== undefined
    return {
      personId,
      personLabel: personId ? (personLookup.get(personId)?.label ?? null) : null,
      confidence: isOverride ? 1 : Number(baseline?.confidence ?? 0),
      requiresConfirmation: Boolean(baseline?.requires_confirmation),
      isOverride,
    }
  }
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
  const selectedLabelHints = detected
    .filter((g) => pendingDetectionSelections.has(g.id))
    .map((g) => g.label)

  // Use store-tracked current preview URL first (latest processed image),
  // then fallback to payload fields.
  const activeImageUrl = pendingDetectionImageUrl || source_image_url || auto_blurred_image_url || ''
  const previewSrc = absoluteUrl(activeImageUrl)

  useEffect(() => {
    previewLoadWarned.current = false
  }, [previewSrc])

  useEffect(() => {
    const firstSelected = selectedLabelHints[0] ?? ''
    setMannequinHints((prev) => {
      if (prev.category.trim().length > 0) return prev
      return { ...prev, category: firstSelected }
    })
  }, [selectedLabelHints])

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

  async function runSamTapSelection(point: { x: number; y: number }) {
    const half = 0.12
    const tapBox: NormalizedBBox = {
      x1: Math.max(0, point.x - half),
      y1: Math.max(0, point.y - half),
      x2: Math.min(1, point.x + half),
      y2: Math.min(1, point.y + half),
    }
    await runSamFromFocusRect(tapBox)
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
      const center = { x: (rect.x1 + rect.x2) / 2, y: (rect.y1 + rect.y2) / 2 }
      void runSamTapSelection(center)
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

    // Tap tool: each click runs SAM on a small box around the tap point and
    // replaces any existing polygon with the freshly refined mask. This is the
    // most mobile-friendly entry point — no drag, no multi-click, just tap.
    if (selectionTool === 'tap') {
      if (isBusy) return
      void runSamTapSelection({ x, y })
      return
    }

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

  /**
   * One-click mask for the currently selected person chip. No polygon, no
   * rectangle drawing — just SAM on the person's bbox. Works well as a safety
   * fallback when the user finds tap/polygon/rectangle tools too fiddly.
   */
  async function handleIsolateSelectedPeople() {
    const selectedPeople = person_candidates.filter((p) => subjectFilterPersonSelections.has(p.id))
    if (selectedPeople.length === 0) {
      toast.warning('Pick a person chip first (or use Tap (SAM) to point at someone).')
      return
    }
    const toastId = toast.loading('Isolating selected person with SAM...')
    setBusyTask({
      label: 'Isolating person with SAM…',
      sub: 'Running Segment Anything on the selected chip. 2–10 s typical.',
      startedAt: Date.now(),
    })
    try {
      const polygons = await refineMaskWithSam(activeImageUrl, selectedPeople.map((p) => p.bbox))
      const valid = (polygons ?? []).filter((poly) => poly.length >= 3)
      if (!valid.length) {
        toast.error('SAM could not isolate the person. Try Tap (SAM) on them.', { id: toastId })
        return
      }
      // If multiple chips were selected, merge polygons by keeping all points.
      // For a single person, just use the tightest polygon.
      const picked =
        valid.length === 1 ? valid[0] : valid.reduce((acc, cur) => (polygonArea(cur) < polygonArea(acc) ? cur : acc), valid[0])
      setSubjectFilterMaskPolygon(picked)
      setPolygonClosed(true)
      toast.success('Person isolated. Now press "Apply filter" or "Cut this as a garment".', {
        id: toastId,
      })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'SAM isolation failed', { id: toastId })
    } finally {
      setBusyTask(null)
    }
  }

  /**
   * Runs SAM + alpha-mask on every *selected* detected garment so each card
   * renders as an isolated item with no background. Clean visual result for
   * the wardrobe, even on cluttered group photos.
   */
  async function handleRefineBackgrounds() {
    const toastId = toast.loading('Cleaning up backgrounds with SAM...')
    setBusyTask({
      label: 'Removing backgrounds with SAM…',
      sub: 'Producing transparent PNGs per garment. ~2–5 s per item.',
      startedAt: Date.now(),
    })
    try {
      const result = await refineDetectedGarmentBackgrounds({
        onlySelected: true,
        onProgress: (done, total, label) => {
          setBusyTask({
            label: `Cleaning garment ${Math.min(done + 1, total)} / ${total}`,
            sub: label,
            startedAt: Date.now(),
          })
        },
      })
      if (result.refined === 0 && result.failed === 0) {
        toast.message(
          'Nothing to clean — select items first, or they were already background-removed.',
          { id: toastId },
        )
        return
      }
      // Store the per-garment polygons so the canvas can display them as
      // SAM-style colored overlays (each garment gets its own color).
      if (Object.keys(result.polygons).length > 0) {
        setMaskOverlays((prev) => ({ ...prev, ...result.polygons }))
      }
      const parts = [`${result.refined} cleaned`]
      if (result.skipped > 0) parts.push(`${result.skipped} skipped`)
      if (result.failed > 0) parts.push(`${result.failed} failed`)
      toast.success(parts.join(' · '), { id: toastId })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Background cleanup failed', { id: toastId })
    } finally {
      setBusyTask(null)
    }
  }

  /**
   * Cut the current polygon out as *one* garment card and clear the polygon so
   * the user can immediately draw another. This is the escape hatch when one
   * person wears multiple garments and the auto-detector / SAM person-polygon
   * would otherwise collapse them into a single item.
   */
  async function handleCutPolygonAsGarment() {
    if (!polygonClosed || subjectFilterMaskPolygon.length < 3) {
      toast.warning(
        'Close a polygon around one garment first (e.g. just the top). Then press "Cut this as a garment".',
      )
      return
    }
    const toastId = toast.loading('Cutting garment from polygon...')
    setBusyTask({
      label: 'Cutting garment from polygon…',
      sub: 'Masking the selected region; detection card will appear when done.',
      startedAt: Date.now(),
    })
    try {
      const garment = await cutGarmentFromActivePolygon()
      if (!garment) {
        toast.error('Could not cut garment — is the polygon valid?', { id: toastId })
        return
      }
      setPolygonClosed(false)
      toast.success('Garment card added. Draw another polygon for the next garment.', {
        id: toastId,
      })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Polygon cutout failed', { id: toastId })
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
      const attributeHints = Object.fromEntries(
        Object.entries(mannequinHints)
          .map(([k, v]) => [k, v.trim()])
          .filter(([, v]) => v.length > 0),
      )
      const result = await generateMannequinToPending({
        attributeHints,
      })
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
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <button
              type="button"
              className="detection-sheet__toggle-all"
              onClick={toggleAll}
            >
              {allSelected ? 'Clear selection' : 'Select all'}
            </button>
            <button
              type="button"
              className="btn secondary"
              onClick={handleRefineBackgrounds}
              disabled={isBusy || detected.length === 0}
              title="Run SAM on each selected garment to produce clean alpha PNGs (no background). ~2–5 s per item."
              style={{ fontSize: 12, padding: '6px 10px' }}
            >
              Remove backgrounds
            </button>
            <button
              type="button"
              className="btn secondary"
              onClick={() => setShowGarmentOverlays((v) => !v)}
              disabled={detected.length === 0}
              title="Toggle the colored mask overlays on the photo above. Each garment gets its own color so you can tell them apart."
              style={{ fontSize: 12, padding: '6px 10px' }}
            >
              {showGarmentOverlays ? 'Hide masks' : 'Show masks'}
            </button>
          </div>
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
              <div style={{ marginTop: 6 }}>
                <button
                  type="button"
                  className="btn secondary"
                  onClick={handleIsolateSelectedPeople}
                  disabled={isBusy || subjectFilterPersonSelections.size === 0}
                  title="One-click mask: runs SAM on the selected person chip(s) so you can skip manual polygon/rectangle drawing."
                  style={{ fontSize: 12, padding: '6px 10px' }}
                >
                  Isolate selected person (SAM)
                </button>
              </div>
            </>
          )}
          <div className="subject-filter__polygon">
            <p className="subject-filter__hint" style={{ marginBottom: 6 }}>
              Faces are blurred automatically when detected (preview uses the processed image).
              Use Rectangle (SAM) to outline one person or garment, then Generate mannequin; detection refreshes from that cutout.
              <br />
              <strong>Multiple garments on one person?</strong> Draw a polygon around just one piece (e.g. the top), close it, and press
              <em> "Cut this as a garment"</em>. Repeat for the pants, shoes, etc. — each gets its own card.
            </p>
            <div className="subject-filter__mode-row" style={{ flexWrap: 'wrap', gap: 6 }}>
              <span className="subject-filter__hint" style={{ margin: 0 }}>Mask tool</span>
              <button
                type="button"
                className={`subject-filter__mode ${selectionTool === 'tap' ? 'is-active' : ''}`}
                onClick={() => setSelectionTool('tap')}
                title="Tap anywhere on the photo — SAM runs on that point and produces a mask."
              >
                Tap (SAM)
              </button>
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
              {selectionTool === 'tap'
                ? 'Tap: click once on a person or garment — SAM will generate the mask automatically.'
                : selectionTool === 'polygon'
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
              style={{
                cursor: isBusy
                  ? 'wait'
                  : selectionTool === 'rectangle'
                    ? 'crosshair'
                    : selectionTool === 'tap'
                      ? 'pointer'
                      : undefined,
              }}
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
                {/* SAM-style per-garment overlays: colored bboxes always, plus
                    filled polygons once "Remove backgrounds" has run SAM on each
                    item. Mirrors the LabelErr/Meta SAM2 visualization. */}
                {showGarmentOverlays && detected.map((g, idx) => {
                  const color = paletteFor(idx)
                  const poly = maskOverlays[g.id]
                  if (poly && poly.length >= 3) {
                    const pts = poly.map((p) => `${p.x * 100},${p.y * 100}`).join(' ')
                    return (
                      <g key={`overlay-${g.id}`} pointerEvents="none">
                        <polygon points={pts} fill={color.fill} stroke={color.stroke} strokeWidth={0.4} />
                      </g>
                    )
                  }
                  if (!g.bbox) return null
                  const b = g.bbox
                  return (
                    <g key={`overlay-${g.id}`} pointerEvents="none">
                      <rect
                        x={b.x1 * 100}
                        y={b.y1 * 100}
                        width={(b.x2 - b.x1) * 100}
                        height={(b.y2 - b.y1) * 100}
                        fill={color.fill}
                        stroke={color.stroke}
                        strokeWidth={0.4}
                        rx={0.6}
                      />
                    </g>
                  )
                })}
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
                className="btn"
                onClick={handleCutPolygonAsGarment}
                disabled={isBusy || !polygonClosed || subjectFilterMaskPolygon.length < 3}
                title="Use when one person wears multiple garments — draw a polygon around one piece (e.g. the top) and press this to add it as a separate card. Repeat for the pants."
              >
                Cut this as a garment
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
            <div
              className="subject-filter__mode-row"
              style={{ flexWrap: 'wrap', alignItems: 'flex-end', gap: 8, marginTop: 10 }}
            >
              <span className="subject-filter__hint" style={{ width: '100%', margin: 0 }}>
                Mannequin attribute hints (optional, improves prompt + extraction quality)
              </span>
              {(
                [
                  ['category', 'Category (e.g. kurta)'],
                  ['color', 'Color (e.g. navy blue)'],
                  ['collar', 'Collar'],
                  ['sleeves', 'Sleeves'],
                  ['placket', 'Placket'],
                  ['fabric', 'Fabric'],
                  ['details', 'Details (comma separated)'],
                ] as const
              ).map(([key, label]) => (
                <label
                  key={key}
                  style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 170, flex: '1 1 170px' }}
                >
                  <span className="subject-filter__hint" style={{ margin: 0 }}>{label}</span>
                  <input
                    type="text"
                    value={mannequinHints[key]}
                    onChange={(e) => setMannequinHints((prev) => ({ ...prev, [key]: e.target.value }))}
                    disabled={isBusy}
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                      padding: '7px 9px',
                      background: 'var(--panel)',
                      color: 'var(--text)',
                    }}
                  />
                </label>
              ))}
            </div>
            {/* Try-off temporarily disabled to evaluate SAM-based mannequin extraction quality. */}
          </div>
        </div>

        <div className="detection-grid">
          {detected.map((garment, idx) => (
            <DetectionCard
              key={garment.id}
              garment={garment}
              selected={pendingDetectionSelections.has(garment.id)}
              onToggle={() => toggleDetectionSelection(garment.id)}
              assignment={getAssignment(garment.id)}
              people={peopleForUi}
              onReassign={(personId) =>
                setAssignmentOverrides((prev) => ({ ...prev, [garment.id]: personId }))
              }
              overlayColor={showGarmentOverlays ? paletteFor(idx).stroke : undefined}
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
