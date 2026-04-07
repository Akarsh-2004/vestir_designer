import { EmptyState } from '../../components/EmptyState'
import { ItemPhoto } from '../../components/ItemPhoto'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { Item } from '../../types/index'

interface OutfitsGridProps {
  onBrowseItems: () => void
}

export function OutfitsGrid({ onBrowseItems }: OutfitsGridProps) {
  const outfits = useWardrobeStore((s) => s.outfits)
  if (!outfits.length) {
    return (
      <EmptyState
        title="No outfits yet"
        description="Start by picking an item. We'll help you pull a look together."
        action={
          <button className="btn" type="button" onClick={onBrowseItems}>
            Browse items
          </button>
        }
      />
    )
  }

  return (
    <div className="grid">
      {outfits.map((outfit) => (
        <div className="item-card" key={outfit.id}>
          <div className="outfit-collage">
            {outfit.items.slice(0, 4).map((item: Item) => (
              <ItemPhoto key={item.id} itemId={item.id} imageUrl={item.image_url} alt={item.item_type} size="sm" />
            ))}
          </div>
          <div className="item-meta">
            <strong>{outfit.name}</strong>
          </div>
        </div>
      ))}
    </div>
  )
}
