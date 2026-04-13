import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { EmptyState } from '../../components/EmptyState'
import { ItemPhoto } from '../../components/ItemPhoto'
import { useWardrobeStore } from '../../store/wardrobeStore'

interface ItemsGridProps {
  onAddItems: () => void
}

export function ItemsGrid({ onAddItems }: ItemsGridProps) {
  const allItems = useWardrobeStore((s) => s.items)
  const activeWardrobeId = useWardrobeStore((s) => s.activeWardrobeId)
  const activeCategory = useWardrobeStore((s) => s.activeCategory)
  const searchQuery = useWardrobeStore((s) => s.searchQuery)

  const items = useMemo(() => {
    return allItems.filter((item) => {
      if (item.deleted_at) return false
      if (item.wardrobe_id !== activeWardrobeId) return false
      if (activeCategory !== 'All' && item.category !== activeCategory) return false
      if (!searchQuery.trim()) return true
      const q = searchQuery.toLowerCase()
      return (
        item.item_type.toLowerCase().includes(q) ||
        item.color_primary.toLowerCase().includes(q) ||
        item.material.toLowerCase().includes(q) ||
        item.category.toLowerCase().includes(q)
      )
    })
  }, [allItems, activeWardrobeId, activeCategory, searchQuery])

  const hasProcessing = items.some((i) => !i.ai_processed)
  if (!items.length) {
    return (
      <EmptyState
        title="Your wardrobe is getting started"
        description="Add a photo and we'll sort your pieces into place."
        action={
          <button className="btn" type="button" onClick={onAddItems}>
            Add a photo
          </button>
        }
      />
    )
  }

  return (
    <div className="grid">
      {items.map((item) => (
        <Link key={item.id} to={`/item/${item.id}`} className="item-card">
          <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} size="full" />
          <div className="item-meta">
            <strong>{item.ai_processed ? item.item_type : 'Analyzing...'}</strong>
            <small>{item.ai_processed ? item.color_primary : 'Please wait'}</small>
            {!item.ai_processed ? (
              <small>Analyzing…</small>
            ) : null}
          </div>
        </Link>
      ))}
      {hasProcessing ? <div className="muted">Working in the background—feel free to browse.</div> : null}
    </div>
  )
}
