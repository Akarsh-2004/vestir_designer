import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronLeft, Plus, Trash2 } from 'lucide-react'
import { useWardrobeStore } from '../../store/wardrobeStore'
import type { BrandSizeCategory, SizeRegion } from '../../types/index'
import {
  REGIONS,
  LETTER_SIZES,
  APPAREL_SIZES_BY_REGION,
  JEAN_WAIST_INCHES,
  JEAN_INSEAM_INCHES,
  SHOE_SIZES_US_WOMEN,
  SHOE_SIZES_UK,
  SHOE_SIZES_EU,
  SHOE_SIZES_CM,
  SHOE_WIDTHS,
  BRA_BANDS_US,
  BRA_BANDS_EU,
  BRA_CUPS_US_UK,
  BRA_CUPS_EU,
  RING_SIZES_US,
  HAT_SIZES,
  GLOVE_SIZES,
  BELT_SIZES_INCHES,
  SOCK_SIZES,
  SILHOUETTE_OPTIONS,
  BRAND_CATALOGUE,
  BRAND_FIT_TIPS,
  getSizeOptionsFor,
  computePassportCompletion,
} from '../../data/sizingReference'

type Tab = 'standard' | 'brands' | 'body' | 'fit'

const BRAND_SIZE_CATEGORIES: BrandSizeCategory[] = [
  'Tops', 'Dresses', 'Bottoms', 'Jeans', 'Outerwear',
  'Shoes', 'Activewear', 'Lingerie', 'Swimwear', 'Accessories',
]

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="passport-section">
      <h3 className="passport-section-title">{title}</h3>
      {children}
    </div>
  )
}

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="passport-field-row">
      <label className="passport-field-label">{label}</label>
      {children}
    </div>
  )
}

function Select({
  value,
  onChange,
  options,
  placeholder = 'Select…',
}: {
  value: string
  onChange: (v: string) => void
  options: string[]
  placeholder?: string
}) {
  return (
    <select
      className="passport-select"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">{placeholder}</option>
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  )
}

export function SizePassportScreen() {
  const { sizePassport, updateSizePassport, addBrandSize, removeBrandSize } = useWardrobeStore()
  const [activeTab, setActiveTab] = useState<Tab>('standard')
  const [newBrandCat, setNewBrandCat] = useState<BrandSizeCategory>('Tops')
  const [newBrandName, setNewBrandName] = useState('')
  const [newBrandSize, setNewBrandSize] = useState('')
  const [newBrandRuns, setNewBrandRuns] = useState<'small' | 'true' | 'large' | ''>('')
  const [newBrandNote, setNewBrandNote] = useState('')

  const region = (sizePassport.preferred_region ?? 'US') as SizeRegion
  const completion = computePassportCompletion({
    top_size: sizePassport.top_size,
    bottom_size: sizePassport.bottom_size,
    shoe_size_us: sizePassport.shoe_size_us,
    bra_band: sizePassport.bra_band,
    height_cm: sizePassport.height_cm,
    brand_sizes: sizePassport.brand_sizes,
  })

  const braRegion = sizePassport.bra_region ?? 'US'
  const braBands = braRegion === 'EU' ? BRA_BANDS_EU : BRA_BANDS_US
  const braCups = braRegion === 'EU' ? BRA_CUPS_EU : BRA_CUPS_US_UK

  const apparel = [...LETTER_SIZES, ...APPAREL_SIZES_BY_REGION[region]]

  function handleAddBrandSize() {
    if (!newBrandName || !newBrandSize) return
    addBrandSize({
      brand: newBrandName,
      category: newBrandCat,
      size: newBrandSize,
      runs: newBrandRuns || undefined,
      fit_note: newBrandNote || undefined,
    })
    setNewBrandName('')
    setNewBrandSize('')
    setNewBrandRuns('')
    setNewBrandNote('')
  }

  const tabs: { id: Tab; label: string }[] = [
    { id: 'standard', label: 'Standard' },
    { id: 'brands', label: 'Brands' },
    { id: 'body', label: 'Body' },
    { id: 'fit', label: 'Fit' },
  ]

  return (
    <div className="screen">
      {/* Header */}
      <header className="screen-header">
        <Link to="/settings" className="back-btn" aria-label="Back">
          <ChevronLeft size={20} />
        </Link>
        <div className="screen-header-center">
          <span className="screen-title">Size Passport</span>
          <span className="screen-subtitle">Every size, every brand — in one place.</span>
        </div>
        <div style={{ width: 36 }} />
      </header>

      {/* Region + Progress */}
      <div className="passport-meta">
        <div className="passport-region-row">
          <label className="passport-field-label">Default region</label>
          <Select
            value={region}
            onChange={(v) => updateSizePassport({ preferred_region: v as SizeRegion })}
            options={REGIONS}
          />
        </div>
        <div className="passport-progress-wrap">
          <div className="passport-progress-bar">
            <div className="passport-progress-fill" style={{ width: `${completion}%` }} />
          </div>
          <span className="passport-progress-label">{completion}% complete</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="passport-tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`passport-tab${activeTab === t.id ? ' passport-tab--active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="passport-content">

        {/* ── Standard ─────────────────────────────── */}
        {activeTab === 'standard' && (
          <div className="passport-tab-body">
            <SectionCard title="Apparel">
              <FieldRow label="Top size">
                <Select value={sizePassport.top_size} onChange={(v) => updateSizePassport({ top_size: v })} options={apparel} />
              </FieldRow>
              <FieldRow label="Bottom size">
                <Select value={sizePassport.bottom_size} onChange={(v) => updateSizePassport({ bottom_size: v })} options={apparel} />
              </FieldRow>
              <FieldRow label="Dress size">
                <Select value={sizePassport.dress_size ?? ''} onChange={(v) => updateSizePassport({ dress_size: v })} options={apparel} />
              </FieldRow>
              <FieldRow label="Outerwear size">
                <Select value={sizePassport.outerwear_size ?? ''} onChange={(v) => updateSizePassport({ outerwear_size: v })} options={apparel} />
              </FieldRow>
            </SectionCard>

            <SectionCard title="Denim">
              <FieldRow label="Waist (in)">
                <Select value={sizePassport.jeans_waist_in ?? ''} onChange={(v) => updateSizePassport({ jeans_waist_in: v })} options={JEAN_WAIST_INCHES} />
              </FieldRow>
              <FieldRow label="Inseam (in)">
                <Select value={sizePassport.jeans_inseam_in ?? ''} onChange={(v) => updateSizePassport({ jeans_inseam_in: v })} options={JEAN_INSEAM_INCHES} />
              </FieldRow>
            </SectionCard>

            <SectionCard title="Shoes">
              <FieldRow label="US size">
                <Select value={sizePassport.shoe_size_us ?? ''} onChange={(v) => updateSizePassport({ shoe_size_us: v })} options={SHOE_SIZES_US_WOMEN} />
              </FieldRow>
              <FieldRow label="UK size">
                <Select value={sizePassport.shoe_size_uk ?? ''} onChange={(v) => updateSizePassport({ shoe_size_uk: v })} options={SHOE_SIZES_UK} />
              </FieldRow>
              <FieldRow label="EU size">
                <Select value={sizePassport.shoe_size_eu ?? ''} onChange={(v) => updateSizePassport({ shoe_size_eu: v })} options={SHOE_SIZES_EU} />
              </FieldRow>
              <FieldRow label="CM size">
                <Select value={sizePassport.shoe_size_cm ?? ''} onChange={(v) => updateSizePassport({ shoe_size_cm: v })} options={SHOE_SIZES_CM} />
              </FieldRow>
              <FieldRow label="Width">
                <Select value={sizePassport.shoe_width ?? ''} onChange={(v) => updateSizePassport({ shoe_width: v as 'Narrow' | 'Standard' | 'Wide' | 'Extra Wide' })} options={SHOE_WIDTHS} />
              </FieldRow>
            </SectionCard>

            <SectionCard title="Lingerie">
              <FieldRow label="Bra system">
                <Select
                  value={braRegion}
                  onChange={(v) => updateSizePassport({ bra_region: v as 'US' | 'UK' | 'EU' | 'AU' })}
                  options={['US', 'UK', 'EU', 'AU']}
                />
              </FieldRow>
              <FieldRow label="Band size">
                <Select value={sizePassport.bra_band ?? ''} onChange={(v) => updateSizePassport({ bra_band: v })} options={braBands} />
              </FieldRow>
              <FieldRow label="Cup size">
                <Select value={sizePassport.bra_cup ?? ''} onChange={(v) => updateSizePassport({ bra_cup: v })} options={braCups} />
              </FieldRow>
              <FieldRow label="Underwear size">
                <Select value={sizePassport.underwear_size ?? ''} onChange={(v) => updateSizePassport({ underwear_size: v })} options={LETTER_SIZES} />
              </FieldRow>
            </SectionCard>

            <SectionCard title="Accessories">
              <FieldRow label="Ring size (US)">
                <Select value={sizePassport.ring_size ?? ''} onChange={(v) => updateSizePassport({ ring_size: v })} options={RING_SIZES_US} />
              </FieldRow>
              <FieldRow label="Hat size">
                <Select value={sizePassport.hat_size ?? ''} onChange={(v) => updateSizePassport({ hat_size: v })} options={HAT_SIZES} />
              </FieldRow>
              <FieldRow label="Glove size">
                <Select value={sizePassport.glove_size ?? ''} onChange={(v) => updateSizePassport({ glove_size: v })} options={GLOVE_SIZES} />
              </FieldRow>
              <FieldRow label="Belt size">
                <Select value={sizePassport.belt_size ?? ''} onChange={(v) => updateSizePassport({ belt_size: v })} options={BELT_SIZES_INCHES} />
              </FieldRow>
              <FieldRow label="Sock size">
                <Select value={sizePassport.sock_size ?? ''} onChange={(v) => updateSizePassport({ sock_size: v })} options={SOCK_SIZES} />
              </FieldRow>
            </SectionCard>
          </div>
        )}

        {/* ── Brands ───────────────────────────────── */}
        {activeTab === 'brands' && (
          <div className="passport-tab-body">
            {/* Existing brand entries */}
            {sizePassport.brand_sizes.length > 0 && (
              <SectionCard title="Saved brands">
                {sizePassport.brand_sizes.map((entry) => (
                  <div key={entry.id} className="brand-entry">
                    <div className="brand-entry-info">
                      <span className="brand-entry-name">{entry.brand}</span>
                      <span className="brand-entry-meta">{entry.category} · {entry.size}{entry.runs ? ` · runs ${entry.runs}` : ''}</span>
                      {entry.fit_note && <span className="brand-entry-note">{entry.fit_note}</span>}
                    </div>
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() => removeBrandSize(entry.id)}
                      aria-label="Remove"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </SectionCard>
            )}

            {/* Add new brand entry */}
            <SectionCard title="Add brand size">
              <FieldRow label="Category">
                <Select
                  value={newBrandCat}
                  onChange={(v) => {
                    setNewBrandCat(v as BrandSizeCategory)
                    setNewBrandName('')
                    setNewBrandSize('')
                  }}
                  options={BRAND_SIZE_CATEGORIES}
                />
              </FieldRow>
              <FieldRow label="Brand">
                <Select
                  value={newBrandName}
                  onChange={setNewBrandName}
                  options={BRAND_CATALOGUE[newBrandCat] ?? []}
                  placeholder="Select brand…"
                />
              </FieldRow>
              <FieldRow label="Your size">
                <Select
                  value={newBrandSize}
                  onChange={setNewBrandSize}
                  options={getSizeOptionsFor(newBrandCat, region)}
                  placeholder="Select size…"
                />
              </FieldRow>
              <FieldRow label="Runs">
                <Select
                  value={newBrandRuns}
                  onChange={(v) => setNewBrandRuns(v as 'small' | 'true' | 'large' | '')}
                  options={['small', 'true', 'large']}
                  placeholder="Runs…"
                />
              </FieldRow>
              <FieldRow label="Fit note">
                <input
                  className="passport-input"
                  placeholder="e.g. fits large in shoulders"
                  value={newBrandNote}
                  onChange={(e) => setNewBrandNote(e.target.value)}
                />
              </FieldRow>

              {/* Brand fit tip */}
              {newBrandName && BRAND_FIT_TIPS[newBrandName] && (
                <div className="brand-tip">
                  <span className="brand-tip-label">Vestir tip</span>
                  <p className="brand-tip-text">{BRAND_FIT_TIPS[newBrandName]}</p>
                </div>
              )}

              <button
                type="button"
                className="btn"
                style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}
                disabled={!newBrandName || !newBrandSize}
                onClick={handleAddBrandSize}
              >
                <Plus size={14} />
                Add brand size
              </button>
            </SectionCard>

            {/* Inline brand-edit on existing entries */}
            {sizePassport.brand_sizes.length > 0 && (
              <p className="passport-hint">Tap a brand entry to edit inline — coming soon.</p>
            )}
          </div>
        )}

        {/* ── Body ─────────────────────────────────── */}
        {activeTab === 'body' && (
          <div className="passport-tab-body">
            <SectionCard title="General">
              <div className="body-grid">
                <BodyField label="Height (cm)" value={sizePassport.height_cm ?? ''} onChange={(v) => updateSizePassport({ height_cm: v })} />
                <BodyField label="Weight (kg)" value={sizePassport.weight_kg ?? ''} onChange={(v) => updateSizePassport({ weight_kg: v })} />
              </div>
            </SectionCard>
            <SectionCard title="Upper body">
              <div className="body-grid">
                <BodyField label="Bust (cm)" value={sizePassport.bust_cm ?? ''} onChange={(v) => updateSizePassport({ bust_cm: v })} />
                <BodyField label="Underbust (cm)" value={sizePassport.underbust_cm ?? ''} onChange={(v) => updateSizePassport({ underbust_cm: v })} />
                <BodyField label="Chest (cm)" value={sizePassport.chest_cm ?? ''} onChange={(v) => updateSizePassport({ chest_cm: v })} />
                <BodyField label="Shoulder (cm)" value={sizePassport.shoulder_cm ?? ''} onChange={(v) => updateSizePassport({ shoulder_cm: v })} />
                <BodyField label="Sleeve (cm)" value={sizePassport.sleeve_cm ?? ''} onChange={(v) => updateSizePassport({ sleeve_cm: v })} />
                <BodyField label="Neck (cm)" value={sizePassport.neck_cm ?? ''} onChange={(v) => updateSizePassport({ neck_cm: v })} />
              </div>
            </SectionCard>
            <SectionCard title="Lower body">
              <div className="body-grid">
                <BodyField label="Waist (cm)" value={sizePassport.waist_cm ?? ''} onChange={(v) => updateSizePassport({ waist_cm: v })} />
                <BodyField label="Hip (cm)" value={sizePassport.hip_cm ?? ''} onChange={(v) => updateSizePassport({ hip_cm: v })} />
                <BodyField label="Thigh (cm)" value={sizePassport.thigh_cm ?? ''} onChange={(v) => updateSizePassport({ thigh_cm: v })} />
                <BodyField label="Inseam (cm)" value={sizePassport.inseam_cm ?? ''} onChange={(v) => updateSizePassport({ inseam_cm: v })} />
                <BodyField label="Outseam (cm)" value={sizePassport.outseam_cm ?? ''} onChange={(v) => updateSizePassport({ outseam_cm: v })} />
              </div>
            </SectionCard>
            <SectionCard title="Feet">
              <div className="body-grid">
                <BodyField label="Foot length (cm)" value={sizePassport.foot_length_cm ?? ''} onChange={(v) => updateSizePassport({ foot_length_cm: v })} />
                <BodyField label="Foot width (cm)" value={sizePassport.foot_width_cm ?? ''} onChange={(v) => updateSizePassport({ foot_width_cm: v })} />
              </div>
            </SectionCard>
          </div>
        )}

        {/* ── Fit ──────────────────────────────────── */}
        {activeTab === 'fit' && (
          <div className="passport-tab-body">
            <SectionCard title="Preferred fit">
              <div className="fit-preference-row">
                {(['Relaxed', 'Regular', 'Slim'] as const).map((pref) => (
                  <button
                    key={pref}
                    type="button"
                    className={`fit-pref-btn${sizePassport.fit_preference === pref ? ' fit-pref-btn--active' : ''}`}
                    onClick={() => updateSizePassport({ fit_preference: pref })}
                  >
                    {pref}
                  </button>
                ))}
              </div>
            </SectionCard>

            <SectionCard title="Preferred silhouettes">
              <div className="silhouette-chips">
                {SILHOUETTE_OPTIONS.map((s) => {
                  const active = sizePassport.preferred_silhouettes.includes(s)
                  return (
                    <button
                      key={s}
                      type="button"
                      className={`silhouette-chip${active ? ' silhouette-chip--active' : ''}`}
                      onClick={() => {
                        const current = sizePassport.preferred_silhouettes
                        updateSizePassport({
                          preferred_silhouettes: active
                            ? current.filter((x) => x !== s)
                            : [...current, s],
                        })
                      }}
                    >
                      {s}
                    </button>
                  )
                })}
              </div>
            </SectionCard>

            <SectionCard title="Notes">
              <textarea
                className="passport-textarea"
                placeholder="Any sizing quirks, preferences, or notes for your future self…"
                value={sizePassport.notes ?? ''}
                onChange={(e) => updateSizePassport({ notes: e.target.value })}
                rows={4}
              />
            </SectionCard>
          </div>
        )}
      </div>
    </div>
  )
}

function BodyField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="body-field">
      <label className="passport-field-label">{label}</label>
      <input
        type="number"
        className="passport-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="—"
        inputMode="decimal"
      />
    </div>
  )
}
