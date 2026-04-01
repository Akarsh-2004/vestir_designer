import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { calculateCompatibility } from '../../lib/colorHarmony'
import { useWardrobeStore } from '../../store/wardrobeStore'
import { ItemPhoto } from '../../components/ItemPhoto'
import type { Category, Item } from '../../types/index'

const layerOrder: Category[] = ['Outerwear', 'Tops', 'Bottoms', 'Shoes', 'Accessories']

export function FinishMyFitScreen() {
  const navigate = useNavigate()
  const { anchorId } = useParams()
  const { items, addOutfit } = useWardrobeStore()
  const [selectedIds, setSelectedIds] = useState<string[]>([])

  const anchor = items.find((i) => i.id === anchorId && !i.deleted_at)
  const candidates = useMemo(() => {
    if (!anchor) return []
    return items
      .filter((i) => i.id !== anchor.id && !i.deleted_at && i.ai_processed)
      .map((item) => ({ item, score: calculateCompatibility(anchor, item) }))
      .sort((a, b) => b.score - a.score)
  }, [anchor, items])

  if (!anchor) return <div className="card">Anchor item not found.</div>

  return (
    <section>
      <div className="card anchor-card">
        <ItemPhoto itemId={anchor.id} imageUrl={anchor.image_url} alt={anchor.item_type} size="md" />
        <div>
          <strong>{anchor.item_type}</strong>
          <p className="muted">
            {anchor.color_primary} · {anchor.material}
          </p>
        </div>
      </div>

      {layerOrder
        .filter((layer) => layer !== anchor.category)
        .map((layer) => {
          const layerItems = candidates.filter((c) => c.item.category === layer)
          if (!layerItems.length) return null
          return (
            <div key={layer} className="layer-block">
              <h3>{layer}</h3>
              <div className="carousel">
                {layerItems.map(({ item, score }) => {
                  const selected = selectedIds.includes(item.id)
                  return (
                    <button
                      key={item.id}
                      className={`fit-card ${selected ? 'fit-selected' : ''}`}
                      onClick={() =>
                        setSelectedIds((state) =>
                          state.includes(item.id) ? state.filter((id) => id !== item.id) : [...state, item.id],
                        )
                      }
                    >
                      <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} size="full" />
                      <small>{item.item_type}</small>
                      <small>{score} pts</small>
                    </button>
                  )
                })}
              </div>
            </div>
          )
        })}

      {selectedIds.length >= 1 ? (
        <div className="sticky-save">
          <span>Save Outfit ({selectedIds.length + 1} items)</span>
          <button
            className="btn"
            onClick={() => {
              const selectedItems: Item[] = items.filter((i) => selectedIds.includes(i.id))
              addOutfit({
                id: crypto.randomUUID(),
                user_id: anchor.user_id,
                wardrobe_id: anchor.wardrobe_id,
                name: `${anchor.color_primary} ${anchor.item_type}`,
                anchor_item_id: anchor.id,
                items: [anchor, ...selectedItems],
                created_at: new Date().toISOString(),
              })
              navigate('/')
            }}
          >
            Save
          </button>
        </div>
      ) : null}
    </section>
  )
}
