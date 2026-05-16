import { useEffect, useId } from 'react'
import { Check } from 'lucide-react'
import { useBodyScrollLock } from '../../hooks/useBodyScrollLock'
import { useWardrobeStore } from '../../store/wardrobeStore'

interface Props {
  open: boolean
  onClose: () => void
}

export function WardrobeSwitcher({ open, onClose }: Props) {
  const { wardrobes, activeWardrobeId, setActiveWardrobe } = useWardrobeStore()
  const titleId = useId()
  useBodyScrollLock(open)
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="sheet-scrim" onClick={onClose} role="presentation">
      <div
        className="sheet"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <div className="sheet-handle" aria-hidden />
        <h2 className="sheet-title" id={titleId} style={{ marginBottom: 16 }}>
          Switch wardrobe
        </h2>
        {wardrobes.map((wardrobe) => {
          const active = wardrobe.id === activeWardrobeId
          return (
            <button
              key={wardrobe.id}
              type="button"
              className="sheet-option"
              style={active ? { borderColor: 'var(--primary)' } : undefined}
              onClick={() => {
                setActiveWardrobe(wardrobe.id)
                onClose()
              }}
            >
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: 14 }}>{wardrobe.name}</div>
                <div style={{ fontSize: 12, opacity: 0.6 }}>{wardrobe.item_count} items</div>
              </div>
              {active && <Check size={16} color="var(--primary)" />}
            </button>
          )
        })}
      </div>
    </div>
  )
}
