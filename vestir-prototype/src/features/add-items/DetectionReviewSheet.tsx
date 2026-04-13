import { CheckCircle, Circle, Star } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { toast } from 'sonner'
import type { TryoffGarmentTarget } from '../../lib/pipeline/adapters'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { DetectedGarment, SubjectFilterMode } from '../../types/index'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

function absoluteUrl(url: string) {
  if (url.startsWith('http')) return url
  return `${API_BASE}${url}`
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
    subjectFilterMode,
    subjectFilterPersonSelections,
    subjectFilterMaskPolygon,
    pendingDetectionSelections,
    setSubjectFilterMode,
    toggleSubjectFilterPersonSelection,
    setSubjectFilterMaskPolygon,
    applySubjectFilter,
    applyManualBlurToPending,
    applyTryoffExtractionToPending,
    toggleDetectionSelection,
    confirmDetectedItems,
    dismissDetection,
  } = useWardrobeStore()
  const editorRef = useRef<HTMLDivElement | null>(null)
  const [dragPointIndex, setDragPointIndex] = useState<number | null>(null)
  const [polygonClosed, setPolygonClosed] = useState(false)
  const [tryoffGarmentTarget, setTryoffGarmentTarget] = useState<TryoffGarmentTarget>('outfit')
  const [extractClothingBusy, setExtractClothingBusy] = useState(false)

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
  const selectedCount = pendingDetectionSelections.size
  const allSelected = detected.every((g) => pendingDetectionSelections.has(g.id))
  const allPeopleSelected = person_candidates.length > 0
    && person_candidates.every((p) => subjectFilterPersonSelections.has(p.id))
  const polygonPointLabel = `${subjectFilterMaskPolygon.length} points`
  const selectedAreaPct = Math.round(polygonArea(subjectFilterMaskPolygon) * 10000) / 100
  const polygonPointsSvg = subjectFilterMaskPolygon
    .map((p) => `${(p.x * 100).toFixed(2)},${(p.y * 100).toFixed(2)}`)
    .join(' ')

  const activeImageUrl = auto_blurred_image_url || source_image_url

  function changeFilterMode(mode: SubjectFilterMode) {
    setSubjectFilterMode(mode)
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

  function addPolygonPoint(e: ReactMouseEvent<HTMLDivElement>) {
    if (dragPointIndex !== null || polygonClosed) return
    const container = editorRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    if (subjectFilterMaskPolygon.length >= 3) {
      const first = subjectFilterMaskPolygon[0]
      const distance = Math.hypot(first.x - x, first.y - y)
      if (distance < 0.035) {
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
    const onUp = () => {
      setDragPointIndex(null)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
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
  }

  function undoPoint() {
    if (!subjectFilterMaskPolygon.length) return
    setSubjectFilterMaskPolygon(subjectFilterMaskPolygon.slice(0, -1))
  }

  function toggleClosePolygon() {
    if (subjectFilterMaskPolygon.length < 3) return
    setPolygonClosed((value) => !value)
  }

  async function handleApplySubjectFilter() {
    const toastId = toast.loading('Applying subject filter...')
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
    }
  }

  async function handleExtractClothingTryoff() {
    const toastId = toast.loading('Extracting clothing (virtual try-off)… this can take a few minutes.')
    setExtractClothingBusy(true)
    try {
      const out = await applyTryoffExtractionToPending(tryoffGarmentTarget)
      if (out === null) {
        toast.error('No image to process.', { id: toastId, duration: 12_000 })
        return
      }
      if ('error' in out) {
        const api = import.meta.env.VITE_API_BASE_URL ?? ''
        const path = out.tryoffImageUrl ?? ''
        const viewUrl = path.startsWith('http') ? path : `${api}${path}`
        const hint = path
          ? ` Try-off image was saved — open: ${viewUrl}`
          : ''
        toast.error(`${out.error}.${hint}`, { id: toastId, duration: 25_000 })
        return
      }
      toast.success('Garment extracted on white. Detections updated from the product shot.', {
        id: toastId,
        duration: 8000,
      })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Try-off failed', { id: toastId, duration: 15_000 })
    } finally {
      setExtractClothingBusy(false)
    }
  }

  function handleScrimPointerDown() {
    if (extractClothingBusy) {
      toast.message('Try-off is still running — wait for it to finish, or use Not now after it completes.')
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
    try {
      const result = await applyManualBlurToPending([box])
      if (!result) {
        toast.error('Manual blur failed.', { id: toastId })
        return
      }
      toast.success('Manual blur applied and detections refreshed.', { id: toastId })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Manual blur failed', { id: toastId })
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
                    onClick={() => toggleSubjectFilterPersonSelection(candidate.id)}
                  >
                    {candidate.label}
                  </button>
                ))}
              </div>
            </>
          )}
          <div className="subject-filter__polygon">
            <p className="subject-filter__hint" style={{ marginBottom: 6 }}>
              Manual override: click to add points, drag points to adjust, click first point to close.
            </p>
            <div className="subject-filter__metrics">
              <span>{polygonPointLabel}</span>
              <span>{polygonClosed ? 'Closed' : 'Open'} polygon</span>
              <span>{selectedAreaPct}% selected</span>
            </div>
            <div
              ref={editorRef}
              className="subject-filter__editor"
              onClick={addPolygonPoint}
            >
              <img
                src={absoluteUrl(activeImageUrl)}
                alt="Source for subject filtering"
                className="subject-filter__image"
                draggable={false}
              />
              <svg className="subject-filter__overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
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
                {subjectFilterMaskPolygon.map((point, idx) => (
                  <g
                    key={`${point.x}-${point.y}-${idx}`}
                    onMouseDown={(e) => {
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
                onClick={toggleClosePolygon}
                disabled={subjectFilterMaskPolygon.length < 3}
              >
                {polygonClosed ? 'Reopen' : 'Close'} polygon
              </button>
              <button type="button" className="detection-sheet__toggle-all" onClick={clearPolygon}>
                Clear polygon
              </button>
              <button type="button" className="btn secondary" onClick={handleApplySubjectFilter}>
                Apply filter
              </button>
              <button type="button" className="btn secondary" onClick={handleApplyManualBlur}>
                Apply manual blur
              </button>
            </div>
            <div className="subject-filter__tryoff" style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
              <p className="subject-filter__hint" style={{ marginBottom: 8 }}>
                After faces are blurred, extract clothing as a studio product shot (FLUX + fal try-off LoRA). Pick what to emphasize, then run try-off on the image above.
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
                <label htmlFor="tryoff-garment-target" className="subject-filter__hint" style={{ margin: 0 }}>
                  Garment focus
                </label>
                <select
                  id="tryoff-garment-target"
                  className="subject-filter__select"
                  value={tryoffGarmentTarget}
                  onChange={(e) => setTryoffGarmentTarget(e.target.value as TryoffGarmentTarget)}
                >
                  <option value="outfit">Full outfit (standard)</option>
                  <option value="ensemble">Full outfit (stacked ensemble, premium prompt)</option>
                  <option value="tshirt">T-shirt / top</option>
                  <option value="dress">Dress</option>
                  <option value="pants">Pants / jeans</option>
                  <option value="jacket">Jacket / coat</option>
                </select>
                <button
                  type="button"
                  className="btn"
                  onClick={handleExtractClothingTryoff}
                  disabled={extractClothingBusy}
                >
                  {extractClothingBusy ? 'Extracting…' : 'Extract clothing'}
                </button>
              </div>
            </div>
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
            disabled={selectedCount === 0}
            onClick={handleConfirm}
            style={{ flex: 1 }}
          >
            Add {selectedCount === 1 ? '1 item' : `${selectedCount} items`}
          </button>
          <button
            type="button"
            className="btn secondary"
            onClick={dismissDetection}
            disabled={extractClothingBusy}
            style={{ flex: 0, padding: '10px 14px' }}
          >
            Not now
          </button>
        </div>
      </div>
    </div>
  )
}
