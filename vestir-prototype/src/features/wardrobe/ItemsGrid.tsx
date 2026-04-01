import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { EmptyState } from '../../components/EmptyState'
import { ItemPhoto } from '../../components/ItemPhoto'
import { SkeletonCard } from '../../components/SkeletonCard'
import { useWardrobeStore } from '../../store/wardrobeStore'

export function ItemsGrid() {
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
  if (!items.length) return <EmptyState title="No items yet" description="Add photos to build your wardrobe." />

  return (
    <div className="grid">
      {items.map((item) => (
        <Link key={item.id} to={`/item/${item.id}`} className="item-card">
          {!item.ai_processed ? (
            <SkeletonCard />
          ) : (
            <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} size="full" />
          )}
          <div className="item-meta">
            <strong>{item.item_type}</strong>
            <small>{item.color_primary}</small>
            {!item.ai_processed ? (
              <small>Processing… {item.processing_progress ?? 0}%</small>
            ) : null}
          </div>
        </Link>
      ))}
      {hasProcessing ? <div className="muted">AI enrichment running in background...</div> : null}
    </div>
  )
}
