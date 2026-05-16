import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronLeft, ChevronRight, GripVertical, Plus, Ruler, Trash2, User, Palette, Layers } from 'lucide-react'
import { useWardrobeStore } from '../../store/wardrobeStore'
import { computePassportCompletion } from '../../data/sizingReference'

function SettingsSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="settings-section">
      <p className="settings-section-title">{title}</p>
      <div className="settings-section-body">{children}</div>
    </div>
  )
}

function SettingsRow({
  icon,
  label,
  value,
  to,
  onClick,
  danger,
}: {
  icon?: React.ReactNode
  label: string
  value?: string
  to?: string
  onClick?: () => void
  danger?: boolean
}) {
  const inner = (
    <div className={`settings-row-inner${danger ? ' settings-row-inner--danger' : ''}`}>
      {icon && <span className="settings-row-icon">{icon}</span>}
      <span className="settings-row-label">{label}</span>
      <span className="settings-row-right">
        {value && <span className="settings-row-value">{value}</span>}
        <ChevronRight size={14} opacity={0.4} />
      </span>
    </div>
  )

  if (to) {
    return <Link to={to} className="settings-row-link">{inner}</Link>
  }
  if (onClick) {
    return (
      <button type="button" className="settings-row-link" onClick={onClick}>
        {inner}
      </button>
    )
  }
  return <div className="settings-row-link">{inner}</div>
}

export function SettingsScreen() {
  const { wardrobes, addWardrobe, deleteWardrobe, sizePassport } = useWardrobeStore()
  const [newName, setNewName] = useState('')
  const [showAddWardrobe, setShowAddWardrobe] = useState(false)

  const passportCompletion = computePassportCompletion({
    top_size: sizePassport.top_size,
    bottom_size: sizePassport.bottom_size,
    shoe_size_us: sizePassport.shoe_size_us,
    bra_band: sizePassport.bra_band,
    height_cm: sizePassport.height_cm,
    brand_sizes: sizePassport.brand_sizes,
  })

  return (
    <div className="screen">
      {/* Header */}
      <header className="screen-header">
        <Link to="/" className="back-btn" aria-label="Back to wardrobe">
          <ChevronLeft size={20} />
        </Link>
        <span className="screen-title" style={{ flex: 1 }}>Settings</span>
      </header>

      <div className="settings-content">
        {/* Profile */}
        <SettingsSection title="PROFILE">
          <SettingsRow icon={<User size={15} />} label="Name" value="Add name" />
          <SettingsRow icon={<User size={15} />} label="Email" value="Add email" />
        </SettingsSection>

        {/* Fit & Sizing */}
        <SettingsSection title="FIT & SIZING">
          <SettingsRow
            icon={<Ruler size={15} />}
            label="Size Passport"
            value={`${passportCompletion}% complete`}
            to="/settings/size-passport"
          />
        </SettingsSection>

        {/* Preferences */}
        <SettingsSection title="PREFERENCES">
          <SettingsRow icon={<Palette size={15} />} label="Units" value="cm" />
          <SettingsRow icon={<Layers size={15} />} label="Weather location" value="Auto-detect" />
        </SettingsSection>

        {/* Wardrobes */}
        <SettingsSection title="WARDROBES">
          <div className="wardrobe-list">
            {wardrobes.map((wardrobe) => (
              <div key={wardrobe.id} className="wardrobe-manage-row">
                <GripVertical size={14} style={{ opacity: 0.35, flexShrink: 0 }} />
                <span className="wardrobe-manage-name">{wardrobe.name}</span>
                <span className="wardrobe-manage-count">{wardrobe.item_count} items</span>
                <button
                  type="button"
                  className="icon-btn"
                  disabled={wardrobes.length <= 1}
                  onClick={() => deleteWardrobe(wardrobe.id)}
                  aria-label={`Delete ${wardrobe.name}`}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>

          {showAddWardrobe ? (
            <div className="add-wardrobe-form">
              <input
                className="passport-input"
                placeholder="Wardrobe name…"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && newName.trim()) {
                    addWardrobe(newName.trim())
                    setNewName('')
                    setShowAddWardrobe(false)
                  }
                  if (e.key === 'Escape') setShowAddWardrobe(false)
                }}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  type="button"
                  className="btn"
                  style={{ flex: 1, justifyContent: 'center' }}
                  disabled={!newName.trim()}
                  onClick={() => {
                    addWardrobe(newName.trim())
                    setNewName('')
                    setShowAddWardrobe(false)
                  }}
                >
                  <Plus size={14} /> Add
                </button>
                <button
                  type="button"
                  className="btn secondary"
                  onClick={() => setShowAddWardrobe(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              className="add-wardrobe-btn"
              onClick={() => setShowAddWardrobe(true)}
            >
              <Plus size={14} /> New wardrobe
            </button>
          )}
        </SettingsSection>

        {/* About */}
        <SettingsSection title="ABOUT">
          <SettingsRow label="Version" value="1.0.0" />
          <SettingsRow label="Privacy policy" />
          <SettingsRow label="Terms of service" />
        </SettingsSection>
      </div>
    </div>
  )
}
