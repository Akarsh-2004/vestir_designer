import { useState } from 'react'
import { Plus, Search, Settings } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { AddItemsSheet } from '../add-items/AddItemsSheet'
import { CategoryFilter } from './CategoryFilter'
import { ItemsGrid } from './ItemsGrid'
import { OutfitsGrid } from './OutfitsGrid'
import { WardrobePills } from './WardrobePills'
import { useWardrobeStore } from '../../store/wardrobeStore'

export function WardrobeScreen() {
  const navigate = useNavigate()
  const { activeView, setActiveView, searchQuery, setSearchQuery } = useWardrobeStore()
  const [sheetOpen, setSheetOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)

  return (
    <section>
      <header className="home-header">
        <button className="icon-btn" onClick={() => navigate('/settings')}>
          <Settings size={16} />
        </button>
        <div className="wordmark">VESTIR</div>
        <button className="icon-btn" onClick={() => setSearchOpen((v) => !v)}>
          <Search size={16} />
        </button>
      </header>

      {searchOpen ? (
        <div className="card">
          <input
            className="search-input"
            placeholder="Search items..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      ) : null}

      <WardrobePills />
      <div className="view-toggle">
        <button className={activeView === 'items' ? 'active-toggle' : ''} onClick={() => setActiveView('items')}>
          Items
        </button>
        <button className={activeView === 'outfits' ? 'active-toggle' : ''} onClick={() => setActiveView('outfits')}>
          Outfits
        </button>
      </div>

      <CategoryFilter />
      {activeView === 'items' ? <ItemsGrid /> : <OutfitsGrid />}

      {activeView === 'items' ? (
        <button className="fab" onClick={() => setSheetOpen(true)}>
          <Plus size={22} />
        </button>
      ) : null}
      <AddItemsSheet open={sheetOpen} onClose={() => setSheetOpen(false)} />
    </section>
  )
}
