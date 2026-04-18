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
  const [lockedByLayer, setLockedByLayer] = useState<Partial<Record<Category, string>>>({})
  const [occasionFilter, setOccasionFilter] = useState('all')
  const [weatherFilter, setWeatherFilter] = useState('all')
  const [styleIntent, setStyleIntent] = useState('balanced')

  const anchor = items.find((i) => i.id === anchorId && !i.deleted_at)
  const candidates = useMemo(() => {
    if (!anchor) return []
    return items
      .filter((i) => i.id !== anchor.id && !i.deleted_at && i.ai_processed)
      .filter((i) => occasionFilter === 'all' || (i.occasions ?? []).map((o) => o.toLowerCase()).includes(occasionFilter))
      .filter((i) => {
        if (weatherFilter === 'all') return true
        if (weatherFilter === 'warm') return i.season.includes('summer') || i.season.includes('spring')
        return i.season.includes('winter') || i.season.includes('autumn')
      })
      .map((item) => ({ item, score: calculateCompatibility(anchor, item) }))
      .map(({ item, score }) => {
        const formalityDiff = Math.abs(anchor.formality - item.formality)
        const styleBoost = styleIntent === 'formal' && item.formality >= 7
          ? 8
          : styleIntent === 'casual' && item.formality <= 5
            ? 8
            : styleIntent === 'bold' && (item.pattern || (item.style_tags?.length ?? 0) > 1)
              ? 6
              : 0
        const confidence = Math.max(0, Math.min(100, 62 + score - formalityDiff * 3 + styleBoost))
        return { item, score, confidence }
      })
      .sort((a, b) => b.score - a.score)
  }, [anchor, items, occasionFilter, weatherFilter, styleIntent])

  if (!anchor) return <div className="card">Anchor item not found.</div>

  return (
    <section>
      <div className="card">
        <strong>Create My Fit Controls</strong>
        <div className="actions" style={{ marginTop: 10 }}>
          <select className="subject-filter__select" value={occasionFilter} onChange={(e) => setOccasionFilter(e.target.value)}>
            <option value="all">Any occasion</option>
            <option value="work">Work</option>
            <option value="casual">Casual</option>
            <option value="party">Party</option>
            <option value="travel">Travel</option>
          </select>
          <select className="subject-filter__select" value={weatherFilter} onChange={(e) => setWeatherFilter(e.target.value)}>
            <option value="all">Any weather</option>
            <option value="warm">Warm</option>
            <option value="cold">Cold</option>
          </select>
          <select className="subject-filter__select" value={styleIntent} onChange={(e) => setStyleIntent(e.target.value)}>
            <option value="balanced">Balanced style</option>
            <option value="formal">More formal</option>
            <option value="casual">More casual</option>
            <option value="bold">More statement</option>
          </select>
        </div>
      </div>
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
                {layerItems.map(({ item, score, confidence }) => {
                  const selected = selectedIds.includes(item.id) || lockedByLayer[layer] === item.id
                  const locked = lockedByLayer[layer] === item.id
                  return (
                    <button
                      key={item.id}
                      className={`fit-card ${selected ? 'fit-selected' : ''}`}
                      onClick={() =>
                        setSelectedIds((state) => {
                          const base = state.includes(item.id) ? state.filter((id) => id !== item.id) : [...state, item.id]
                          return base
                        })
                      }
                    >
                      <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} size="full" />
                      <small>{item.item_type}</small>
                      <small>{score} pts</small>
                      <small>{confidence}% fit confidence</small>
                      <small>{item.color_primary} · {item.material}</small>
                      <small>{(item.occasions ?? []).slice(0, 2).join(', ') || 'versatile'}</small>
                      <span
                        className="subject-filter__hint"
                        style={{ marginTop: 4, color: locked ? '#d24d57' : 'inherit' }}
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                          setLockedByLayer((state) => ({
                            ...state,
                            [layer]: state[layer] === item.id ? undefined : item.id,
                          }))
                        }}
                      >
                        {locked ? 'Unlock this layer' : 'Lock this layer'}
                      </span>
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
              const lockedItems: Item[] = items.filter((i) => Object.values(lockedByLayer).includes(i.id))
              const dedup = [...selectedItems, ...lockedItems].filter(
                (item, index, arr) => arr.findIndex((v) => v.id === item.id) === index,
              )
              addOutfit({
                id: crypto.randomUUID(),
                user_id: anchor.user_id,
                wardrobe_id: anchor.wardrobe_id,
                name: `${anchor.color_primary} ${anchor.item_type}`,
                anchor_item_id: anchor.id,
                items: [anchor, ...dedup],
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
