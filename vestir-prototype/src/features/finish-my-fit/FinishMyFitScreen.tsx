import { useMemo, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { evaluateCandidate, type CandidateReasons } from '../../lib/finishMyFitEngine'
import { useWardrobeStore } from '../../store/wardrobeStore'
import { ItemPhoto } from '../../components/ItemPhoto'
import type { Category, Item } from '../../types/index'

function reasonLabel(kind: keyof CandidateReasons, value: string): { label: string; tone: 'good' | 'warn' | 'bad' } {
  const good = 'good'
  const warn = 'warn'
  const bad = 'bad'
  if (kind === 'color') {
    if (value === 'harmonious') return { label: 'Color in harmony', tone: good }
    if (value === 'neutral') return { label: 'Neutral pairing', tone: good }
    return { label: 'Color clash', tone: bad }
  }
  if (kind === 'formality') {
    if (value === 'match') return { label: 'Formality match', tone: good }
    if (value === 'adjacent') return { label: 'Adjacent formality', tone: good }
    if (value === 'gap') return { label: 'Formality gap', tone: warn }
    return { label: 'Hard formality gap', tone: bad }
  }
  if (value === 'shared') return { label: 'Same season', tone: good }
  if (value === 'adjacent') return { label: 'Close season', tone: good }
  return { label: 'Opposite season', tone: warn }
}

const layerOrder: Category[] = ['Outerwear', 'Tops', 'Bottoms', 'Shoes', 'Accessories']

export function FinishMyFitScreen() {
  const navigate = useNavigate()
  const location = useLocation()
  const { anchorId } = useParams()
  const { items, addOutfit } = useWardrobeStore()
  const seeded = Array.isArray((location.state as { preselectedIds?: string[] } | null)?.preselectedIds)
    ? ((location.state as { preselectedIds?: string[] }).preselectedIds ?? [])
    : []
  const [selectedIds, setSelectedIds] = useState<string[]>(seeded)
  const [lockedByLayer, setLockedByLayer] = useState<Partial<Record<Category, string>>>({})
  const [weatherFilter, setWeatherFilter] = useState<'all' | 'warm' | 'cold'>('all')

  const anchor = items.find((i) => i.id === anchorId && !i.deleted_at)
  const selectedItems = useMemo(
    () => items.filter((i) => selectedIds.includes(i.id)),
    [items, selectedIds],
  )
  const selectedPlusAnchor = useMemo(
    () => (anchor ? [anchor, ...selectedItems] : []),
    [anchor, selectedItems],
  )
  const clashCount = useMemo(
    () => (anchor ? selectedItems.filter((i) => evaluateCandidate([anchor], i).isClash).length : 0),
    [anchor, selectedItems],
  )

  function vibrate(type: 'light' | 'warning') {
    if (!('vibrate' in navigator)) return
    navigator.vibrate(type === 'warning' ? [30, 50, 30] : [15])
  }

  const candidates = useMemo(() => {
    if (!anchor) return []
    return items
      .filter((i) => i.id !== anchor.id && !i.deleted_at && i.ai_processed)
      .filter((i) => {
        if (weatherFilter === 'all') return true
        if (weatherFilter === 'warm') return i.season.includes('summer') || i.season.includes('spring')
        return i.season.includes('winter') || i.season.includes('autumn')
      })
      .map((item) => {
        const selectedExcludingCurrent = selectedPlusAnchor.filter((x) => x.id !== item.id)
        const evalResult = evaluateCandidate(selectedExcludingCurrent, item)
        return { item, evalResult }
      })
      .sort((a, b) => {
        if (a.evalResult.sortGroup !== b.evalResult.sortGroup) return a.evalResult.sortGroup === 'valid' ? -1 : 1
        return b.evalResult.score - a.evalResult.score
      })
  }, [anchor, items, weatherFilter, selectedPlusAnchor])

  if (!anchor) return <div className="card">Anchor item not found.</div>

  return (
    <section>
      <div className="card">
        <strong>Create My Fit Controls</strong>
        <div className="actions" style={{ marginTop: 10, flexWrap: 'wrap', gap: 8 }}>
          <select className="subject-filter__select" value={weatherFilter} onChange={(e) => setWeatherFilter(e.target.value as 'all' | 'warm' | 'cold')}>
            <option value="all">Any weather</option>
            <option value="warm">Warm</option>
            <option value="cold">Cold</option>
          </select>
          {/* Anchor switcher: lets the user jump to another item as the anchor
              without leaving the builder. Critical for testing color harmony
              and formality against many different starting points. */}
          <select
            className="subject-filter__select"
            value={anchor.id}
            onChange={(e) => {
              const nextId = e.target.value
              if (nextId && nextId !== anchor.id) {
                setSelectedIds([])
                setLockedByLayer({})
                navigate(`/finish-my-fit/${nextId}`, { replace: true })
              }
            }}
            title="Switch the anchor to another wardrobe item"
          >
            {items
              .filter((i) => !i.deleted_at && i.ai_processed)
              .sort((a, b) => a.category.localeCompare(b.category) || a.item_type.localeCompare(b.item_type))
              .map((i) => (
                <option key={i.id} value={i.id}>
                  {i.category} · {i.item_type} ({i.color_primary})
                </option>
              ))}
          </select>
          <span className="subject-filter__hint" style={{ margin: 0 }}>
            Outfit health: {clashCount === 0 ? 'Clean fit' : `${clashCount} clashes`}
          </span>
          <span className="subject-filter__hint" style={{ margin: 0, opacity: 0.65 }}>
            {items.filter((i) => !i.deleted_at && i.ai_processed).length} items in wardrobe · {candidates.length} candidates shown
          </span>
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
                {layerItems.map(({ item, evalResult }) => {
                  const selected = selectedIds.includes(item.id) || lockedByLayer[layer] === item.id
                  const locked = lockedByLayer[layer] === item.id
                  return (
                    <button
                      key={item.id}
                      className={`fit-card ${selected ? 'fit-selected' : ''}`}
                      style={{
                        opacity: evalResult.isClash ? 0.3 : evalResult.level === 'fair' ? 0.65 : 1,
                        transform: evalResult.isClash ? 'scale(0.88)' : undefined,
                        borderColor: evalResult.isClash ? '#d24d57' : '#4aa66a',
                        filter: evalResult.isClash ? 'saturate(0.15) blur(0.7px)' : undefined,
                      }}
                      onClick={() =>
                        setSelectedIds((state) => {
                          const base = state.includes(item.id) ? state.filter((id) => id !== item.id) : [...state, item.id]
                          vibrate(evalResult.hapticType)
                          return base
                        })
                      }
                    >
                      <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} size="full" />
                      <small>{item.item_type}</small>
                      <small style={{ textTransform: 'capitalize' }}>{evalResult.isClash ? 'Clash' : evalResult.level}</small>
                      <small>{item.color_primary} · {item.material}</small>
                      <small>{(item.occasions ?? []).slice(0, 2).join(', ') || 'versatile'}</small>
                      <div className="fmf-reasons" onClick={(e) => e.stopPropagation()}>
                        {(['color', 'formality', 'season'] as const).map((dim) => {
                          const label = reasonLabel(dim, evalResult.reasons[dim])
                          return (
                            <span key={dim} className={`fmf-reason fmf-reason--${label.tone}`}>
                              {label.label}
                            </span>
                          )
                        })}
                      </div>
                      {evalResult.isClash && evalResult.worstPairWith ? (
                        <small style={{ color: '#d24d57' }}>
                          Tension with {evalResult.worstPairWith}
                        </small>
                      ) : null}
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
          <span>{clashCount === 0 ? 'Save Outfit' : 'Save anyway'} ({selectedIds.length + 1} items)</span>
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
