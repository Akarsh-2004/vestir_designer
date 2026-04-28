import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronDown, Plus, Search, Settings, X } from 'lucide-react'
import { AddItemsSheet } from '../add-items/AddItemsSheet'
import { CategoryFilter } from './CategoryFilter'
import { ItemsGrid } from './ItemsGrid'
import { MorningBriefing } from './MorningBriefing'
import { OutfitsGrid } from './OutfitsGrid'
import { WardrobePills } from './WardrobePills'
import { WardrobeSwitcher } from './WardrobeSwitcher'
import { useWardrobeStore } from '../../store/wardrobeStore'

export function WardrobeScreen() {
  const { activeView, setActiveView, searchQuery, setSearchQuery, wardrobes, activeWardrobeId, items, outfits } =
    useWardrobeStore()
  const [sheetOpen, setSheetOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [switcherOpen, setSwitcherOpen] = useState(false)

  const activeWardrobe = wardrobes.find((w) => w.id === activeWardrobeId)
  const itemCount = items.filter((item) => !item.deleted_at && item.wardrobe_id === activeWardrobeId).length
  const outfitCount = outfits.filter((o) => {
    const wardrobeIds = (o.items ?? []).map((it) => it.wardrobe_id).filter(Boolean)
    return wardrobeIds.length === 0 || wardrobeIds.includes(activeWardrobeId)
  }).length

  return (
    <section className="wardrobe-screen">
      <header className="wardrobe-header">
        <Link to="/settings" className="wardrobe-header-icon" aria-label="Settings">
          <Settings size={18} strokeWidth={1.6} />
        </Link>

        <button
          type="button"
          className="wardrobe-header-title"
          onClick={() => setSwitcherOpen(true)}
          aria-label="Switch wardrobe"
        >
          <span className="wardrobe-name">{activeWardrobe?.name ?? 'My Wardrobe'}</span>
          <ChevronDown size={14} strokeWidth={1.5} style={{ opacity: 0.55 }} />
        </button>

        {searchOpen ? (
          <button
            type="button"
            className="wardrobe-header-icon"
            onClick={() => { setSearchOpen(false); setSearchQuery('') }}
            aria-label="Close search"
          >
            <X size={18} strokeWidth={1.6} />
          </button>
        ) : (
          <button
            type="button"
            className="wardrobe-header-icon"
            onClick={() => setSearchOpen(true)}
            aria-label="Search"
          >
            <Search size={18} strokeWidth={1.6} />
          </button>
        )}
      </header>

      {searchOpen && (
        <div className="wardrobe-search-bar">
          <input
            className="search-input"
            placeholder="Search items, colours, materials…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            autoFocus
          />
        </div>
      )}

      <div className="wardrobe-controls">
        <WardrobePills />

        <div className="view-toggle">
          <button
            type="button"
            className={activeView === 'items' ? 'active-toggle' : ''}
            onClick={() => setActiveView('items')}
            aria-pressed={activeView === 'items'}
          >
            Items
            <span>{itemCount}</span>
          </button>
          <button
            type="button"
            className={activeView === 'outfits' ? 'active-toggle' : ''}
            onClick={() => setActiveView('outfits')}
            aria-pressed={activeView === 'outfits'}
          >
            Outfits
            <span>{outfitCount}</span>
          </button>
        </div>

        {activeView === 'items' && <CategoryFilter />}
      </div>

      {activeView === 'items' ? (
        <>
          <div className="wardrobe-briefing-wrap">
            <MorningBriefing />
          </div>
          <ItemsGrid onAddItems={() => setSheetOpen(true)} />
        </>
      ) : (
        <OutfitsGrid onBrowseItems={() => setActiveView('items')} />
      )}

      {activeView === 'items' && (
        <button className="fab" type="button" onClick={() => setSheetOpen(true)} aria-label="Add items">
          <Plus size={20} strokeWidth={1.8} />
        </button>
      )}

      <AddItemsSheet open={sheetOpen} onClose={() => setSheetOpen(false)} />
      <WardrobeSwitcher open={switcherOpen} onClose={() => setSwitcherOpen(false)} />
    </section>
  )
}
