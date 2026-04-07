import { useState } from 'react'
import { GripVertical, Plus, Trash2 } from 'lucide-react'
import { useWardrobeStore } from '../../store/wardrobeStore'

export function SettingsScreen() {
  const { wardrobes, addWardrobe, deleteWardrobe } = useWardrobeStore()
  const [name, setName] = useState('')

  return (
    <section className="card">
      <h2>Settings</h2>
      <h3>Wardrobes</h3>
      <div className="settings-list">
        {wardrobes.map((wardrobe) => (
          <div className="settings-row" key={wardrobe.id}>
            <span>
              <GripVertical size={14} />
            </span>
            <span>{wardrobe.name}</span>
            <button className="icon-btn" disabled={wardrobes.length <= 1} onClick={() => deleteWardrobe(wardrobe.id)}>
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
      <div className="actions">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New wardrobe name" className="search-input" />
        <button
          className="btn"
          onClick={() => {
            if (!name.trim()) return
            addWardrobe(name.trim())
            setName('')
          }}
        >
          <Plus size={14} />
          Add Wardrobe
        </button>
      </div>
    </section>
  )
}
