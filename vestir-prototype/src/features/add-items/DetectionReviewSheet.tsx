import { CheckCircle, Circle, Star } from 'lucide-react'
import { toast } from 'sonner'
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
        <p className="detection-card__warn">Partly visible</p>
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

  function handleConfirm() {
    const label = selectedCount === 1 ? '1 item' : `${selectedCount} items`
    toast.success(`Adding ${label} to your wardrobe…`)
    confirmDetectedItems()
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
