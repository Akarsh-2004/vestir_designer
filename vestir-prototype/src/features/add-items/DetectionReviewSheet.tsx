import { CheckCircle, Circle, Star } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { toast } from 'sonner'
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
    toggleDetectionSelection,
    confirmDetectedItems,
    dismissDetection,
  } = useWardrobeStore()
  const editorRef = useRef<HTMLDivElement | null>(null)
  const [dragPointIndex, setDragPointIndex] = useState<number | null>(null)
  const [polygonClosed, setPolygonClosed] = useState(false)

  if (!pendingDetection) return null

  const { detected, scene_track, person_candidates = [], source_image_url } = pendingDetection
  const selectedCount = pendingDetectionSelections.size
  const allSelected = detected.every((g) => pendingDetectionSelections.has(g.id))
  const allPeopleSelected = person_candidates.length > 0
    && person_candidates.every((p) => subjectFilterPersonSelections.has(p.id))
  const polygonPointLabel = `${subjectFilterMaskPolygon.length} points`
  const selectedAreaPct = Math.round(polygonArea(subjectFilterMaskPolygon) * 10000) / 100
  const polygonPointsSvg = subjectFilterMaskPolygon
    .map((p) => `${(p.x * 100).toFixed(2)},${(p.y * 100).toFixed(2)}`)
    .join(' ')

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

  return (
    <div className="sheet-scrim" onClick={dismissDetection}>
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
                src={absoluteUrl(source_image_url)}
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
            style={{ flex: 0, padding: '10px 14px' }}
          >
            Not now
          </button>
        </div>
      </div>
    </div>
  )
}
