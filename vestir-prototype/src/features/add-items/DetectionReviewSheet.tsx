import { CheckCircle, Circle, Star } from 'lucide-react'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { DetectedGarment } from '../../types/index'

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
        <p className="detection-card__warn">Partial</p>
      )}
    </button>
  )
}

export function DetectionReviewSheet() {
  const {
    pendingDetection,
    pendingDetectionSelections,
    toggleDetectionSelection,
    confirmDetectedItems,
    dismissDetection,
  } = useWardrobeStore()

  if (!pendingDetection) return null

  const { detected, scene_track } = pendingDetection
  const selectedCount = pendingDetectionSelections.size
  const allSelected = detected.every((g) => pendingDetectionSelections.has(g.id))

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
              {detected.length} item{detected.length !== 1 ? 's' : ''} found
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '12px', opacity: 0.6 }}>
              {scene_track === 'worn' ? 'Outfit photo' : scene_track === 'flat_lay' ? 'Flat lay' : 'Photo'} · tap to select
            </p>
          </div>
          <button
            type="button"
            className="detection-sheet__toggle-all"
            onClick={toggleAll}
          >
            {allSelected ? 'Deselect all' : 'Select all'}
          </button>
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
            onClick={confirmDetectedItems}
            style={{ flex: 1 }}
          >
            Add {selectedCount > 0 ? `${selectedCount} item${selectedCount !== 1 ? 's' : ''}` : 'selected'}
          </button>
          <button
            type="button"
            className="btn secondary"
            onClick={dismissDetection}
            style={{ flex: 0, padding: '10px 14px' }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
