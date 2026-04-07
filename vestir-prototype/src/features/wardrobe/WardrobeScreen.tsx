import { useState } from 'react'
import { Plus, Search, X } from 'lucide-react'
import { AddItemsSheet } from '../add-items/AddItemsSheet'
import { CategoryFilter } from './CategoryFilter'
import { ItemsGrid } from './ItemsGrid'
import { OutfitsGrid } from './OutfitsGrid'
import { WardrobePills } from './WardrobePills'
import { useWardrobeStore } from '../../store/wardrobeStore'

export function WardrobeScreen() {
  const { activeView, setActiveView, searchQuery, setSearchQuery } = useWardrobeStore()
  const [sheetOpen, setSheetOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)

  return (
    <section>
      <header className="home-header">
        <div className="brand">My Wardrobe</div>
        <button className="icon-btn" type="button" onClick={() => setSearchOpen((v) => !v)} aria-expanded={searchOpen}>
          <Search size={16} />
        </button>
      </header>

      {searchOpen ? (
        <div className="card">
          <div className="actions" style={{ marginBottom: 0 }}>
            <input
              className="search-input"
              placeholder="Search your pieces..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoFocus
              style={{ flex: 1, width: 'auto' }}
            />
            <button className="icon-btn" type="button" onClick={() => setSearchOpen(false)} aria-label="Close search">
              <X size={16} />
            </button>
          </div>
        </div>
      ) : null}

      <WardrobePills />
      <div className="view-toggle">
        <button
          type="button"
          className={activeView === 'items' ? 'active-toggle' : ''}
          onClick={() => setActiveView('items')}
          aria-pressed={activeView === 'items'}
        >
          Items
        </button>
        <button
          type="button"
          className={activeView === 'outfits' ? 'active-toggle' : ''}
          onClick={() => setActiveView('outfits')}
          aria-pressed={activeView === 'outfits'}
        >
          Outfits
        </button>
      </div>

      <CategoryFilter />
      {activeView === 'items' ? <ItemsGrid onAddItems={() => setSheetOpen(true)} /> : <OutfitsGrid onBrowseItems={() => setActiveView('items')} />}

      {activeView === 'items' ? (
        <button className="fab" type="button" onClick={() => setSheetOpen(true)} aria-label="Add items">
          <Plus size={22} />
        </button>
      ) : null}
      <AddItemsSheet open={sheetOpen} onClose={() => setSheetOpen(false)} />
    </section>
  )
}
