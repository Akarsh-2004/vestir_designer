import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Heart, Shirt } from 'lucide-react'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { Item } from '../../types/index'

interface ItemsGridProps {
  onAddItems: () => void
}

function hslToBg(hsl?: { h: number; s: number; l: number }): string {
  if (!hsl) return '#f3eee7'
  // Soft desaturated pastel tint for card background
  return `hsl(${hsl.h}, ${Math.max(12, Math.round(hsl.s * 30))}%, ${Math.min(94, Math.max(88, Math.round(hsl.l * 100 * 1.3)))}%)`
}

function ItemCard({ item }: { item: Item }) {
  const [liked, setLiked] = useState(false)
  const isProcessing = !item.ai_processed
  const progress = item.processing_progress ?? 0
  const bgColor = hslToBg(item.color_primary_hsl)

  return (
    <div className="item-card" style={{ position: 'relative' }}>
      <Link to={`/item/${item.id}`} className="item-card__link" aria-label={item.item_type}>
        <div className="item-card__photo" style={{ background: bgColor }}>
          {item.image_url ? (
            <img src={item.image_url} alt={item.item_type} className="item-card__img" />
          ) : (
            <div className="item-card__placeholder">
              <Shirt size={32} strokeWidth={1.2} style={{ color: 'var(--primary)', opacity: 0.6 }} />
            </div>
          )}

          {/* Processing overlay */}
          {isProcessing && (
            <div className="item-card__processing-overlay">
              <div className="item-card__spinner" />
              <span className="item-card__processing-label">
                {progress > 0 ? `${progress}%` : 'Analysing…'}
              </span>
            </div>
          )}

        </div>

        <div className="item-card__meta">
          <p className="item-card__type">
            {isProcessing ? 'Analysing…' : item.item_type}
          </p>
          <p className="item-card__color">
            {isProcessing ? '—' : item.color_primary}
          </p>
        </div>
      </Link>

      {/* Heart button — sits outside the link */}
      <button
        type="button"
        className={`item-card__heart${liked ? ' item-card__heart--active' : ''}`}
        aria-label={liked ? 'Unlike' : 'Like'}
        onClick={(e) => { e.preventDefault(); setLiked((v) => !v) }}
      >
        <Heart size={14} fill={liked ? 'currentColor' : 'none'} />
      </button>
    </div>
  )
}

export function ItemsGrid({ onAddItems }: ItemsGridProps) {
  const allItems = useWardrobeStore((s) => s.items)
  const activeWardrobeId = useWardrobeStore((s) => s.activeWardrobeId)
  const activeCategory = useWardrobeStore((s) => s.activeCategory)
  const searchQuery = useWardrobeStore((s) => s.searchQuery)
  const hasAnyItems = useMemo(
    () => allItems.some((item) => !item.deleted_at && item.wardrobe_id === activeWardrobeId),
    [allItems, activeWardrobeId],
  )

  const items = useMemo((): Item[] => {
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

  if (!items.length) {
    return (
      <div className="empty-state">
        <div className="empty-state__icon">
          <Shirt size={28} />
        </div>
        <h3>{hasAnyItems ? 'No matches found' : 'Your wardrobe is empty'}</h3>
        <p>
          {hasAnyItems
            ? 'Try a different search term or category to see more pieces.'
            : "Add photos of your clothes and we'll organise them for you."}
        </p>
        {!hasAnyItems && (
          <button className="btn" type="button" onClick={onAddItems} style={{ marginTop: 4 }}>
            Add items
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="items-grid">
      {items.map((item) => (
        <ItemCard key={item.id} item={item} />
      ))}
    </div>
  )
}
