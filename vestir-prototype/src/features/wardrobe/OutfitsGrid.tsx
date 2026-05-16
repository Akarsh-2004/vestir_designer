import { Layers, Shirt } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { Item, Outfit } from '../../types/index'

interface OutfitsGridProps {
  onBrowseItems: () => void
}

export function OutfitsGrid({ onBrowseItems }: OutfitsGridProps) {
  const outfits = useWardrobeStore((s) => s.outfits)
  const navigate = useNavigate()

  if (!outfits.length) {
    return (
      <div className="empty-state">
        <div className="empty-state__icon">
          <Layers size={28} />
        </div>
        <h3>No outfits yet</h3>
        <p>Pick any item and tap "Build a look" to create your first outfit.</p>
        <button className="btn" type="button" onClick={onBrowseItems} style={{ marginTop: 4 }}>
          Browse items
        </button>
      </div>
    )
  }

  function openOutfit(outfit: Outfit) {
    const anchor = outfit.anchor_item_id
      ? outfit.items.find((it) => it.id === outfit.anchor_item_id)
      : outfit.items[0]
    if (!anchor) return
    const preselectedIds = outfit.items.filter((it) => it.id !== anchor.id).map((it) => it.id)
    navigate(`/finish-my-fit/${anchor.id}`, { state: { preselectedIds } })
  }

  return (
    <div className="outfits-grid">
      {outfits.map((outfit) => {
        const pieces = outfit.items.slice(0, 4)
        const empties = Array.from({ length: Math.max(0, 4 - pieces.length) })

        return (
          <button
            type="button"
            className="outfit-card outfit-card--button"
            key={outfit.id}
            onClick={() => openOutfit(outfit)}
            aria-label={`Open ${outfit.name}`}
          >
            <div className="outfit-card__collage">
              {pieces.map((item: Item) => (
                item.image_url ? (
                  <img
                    key={item.id}
                    src={item.image_url}
                    alt={item.item_type}
                    className="outfit-card__collage-img"
                  />
                ) : (
                  <div key={item.id} className="outfit-card__collage-empty outfit-card__collage-empty--icon">
                    <Shirt size={16} />
                  </div>
                )
              ))}
              {empties.map((_, i) => (
                <div key={`empty-${i}`} className="outfit-card__collage-empty" />
              ))}
            </div>
            <div className="outfit-card__meta">
              <p className="outfit-card__name">{outfit.name}</p>
              <p className="outfit-card__count">{outfit.items.length} piece{outfit.items.length !== 1 ? 's' : ''}</p>
            </div>
          </button>
        )
      })}
    </div>
  )
}
