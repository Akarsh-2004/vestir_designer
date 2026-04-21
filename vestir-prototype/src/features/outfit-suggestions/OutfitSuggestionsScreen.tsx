import { ArrowLeft } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { ItemPhoto } from '../../components/ItemPhoto'
import { outfitBuildAdapter } from '../../lib/pipeline/adapters'
import type { OutfitBuildResult } from '../../lib/pipeline/contracts'
import { useWardrobeStore } from '../../store/wardrobeStore'

type WeatherMode = 'all' | 'warm' | 'cold'
type StyleIntent = 'balanced' | 'formal' | 'casual' | 'bold'

export function OutfitSuggestionsScreen() {
  const navigate = useNavigate()
  const { anchorId } = useParams()
  const items = useWardrobeStore((s) => s.items)
  const storedProfile = useWardrobeStore((s) => s.styleProfile)
  const storedWeather = useWardrobeStore((s) => s.weatherContext)

  const anchor = useMemo(
    () => items.find((i) => i.id === anchorId && !i.deleted_at),
    [items, anchorId],
  )
  const wardrobeItems = useMemo(
    () => items.filter((i) => i.ai_processed && !i.deleted_at),
    [items],
  )

  const [intent, setIntent] = useState<StyleIntent>(
    (storedProfile.style_intent as StyleIntent | undefined) ?? 'balanced',
  )
  const [weatherMode, setWeatherMode] = useState<WeatherMode>(
    (storedWeather.mode as WeatherMode | undefined) ?? 'all',
  )
  const [result, setResult] = useState<OutfitBuildResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!anchor) return
    let cancelled = false
    const run = async () => {
      setLoading(true)
      setError(null)
      try {
        const r = await outfitBuildAdapter(
          anchor,
          wardrobeItems,
          { ...storedProfile, style_intent: intent },
          { ...storedWeather, mode: weatherMode === 'all' ? 'all' : weatherMode },
          4,
        )
        if (!cancelled) setResult(r)
      } catch (err: unknown) {
        if (cancelled) return
        const msg = err instanceof Error ? err.message : 'Could not generate outfits'
        setError(msg)
        toast.error(msg)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [anchor, wardrobeItems, intent, weatherMode, storedProfile, storedWeather])

  if (!anchor) {
    return (
      <div className="card">
        <button className="btn secondary" type="button" onClick={() => navigate(-1)}>
          <ArrowLeft size={16} /> Back
        </button>
        <p>Anchor item not found.</p>
      </div>
    )
  }

  return (
    <section>
      <div className="card">
        <button className="btn secondary" type="button" onClick={() => navigate(-1)}>
          <ArrowLeft size={16} /> Back
        </button>
        <strong>Outfit suggestions</strong>
        <p className="muted">Ranked full looks built from your wardrobe around this anchor.</p>
        <div className="actions" style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <select
            className="subject-filter__select"
            value={intent}
            onChange={(e) => setIntent(e.target.value as StyleIntent)}
          >
            <option value="balanced">Balanced</option>
            <option value="casual">Casual</option>
            <option value="formal">Formal</option>
            <option value="bold">Bold</option>
          </select>
          <select
            className="subject-filter__select"
            value={weatherMode}
            onChange={(e) => setWeatherMode(e.target.value as WeatherMode)}
          >
            <option value="all">Any weather</option>
            <option value="warm">Warm</option>
            <option value="cold">Cold</option>
          </select>
        </div>
      </div>

      <div className="card anchor-card">
        <ItemPhoto itemId={anchor.id} imageUrl={anchor.image_url} alt={anchor.item_type} size="md" />
        <div>
          <strong>{anchor.item_type}</strong>
          <p className="muted">{anchor.color_primary} · {anchor.material}{anchor.fit ? ` · ${anchor.fit}` : ''}</p>
        </div>
      </div>

      {loading ? (
        <div className="card">Composing outfits…</div>
      ) : error ? (
        <div className="card">Couldn't build outfits: {error}</div>
      ) : result && result.outfits.length > 0 ? (
        <>
          <p className="muted" style={{ marginTop: 8 }}>
            {result.summary}{' '}
            {result.metadata.source === 'ollama+heuristic' ? '(LLM-blended)' : '(heuristic-only)'}
          </p>
          <div className="outfit-grid">
            {result.outfits.map((o) => (
              <div key={o.rank} className="card outfit-card">
                <div className="outfit-header">
                  <strong>Outfit #{o.rank}</strong>
                  <span className="muted">{Math.round(o.score * 100)}% match</span>
                </div>
                <div className="outfit-pieces">
                  {o.pieces.map((p) => (
                    <Link key={p.id} to={`/item/${p.id}`} className="outfit-piece">
                      <ItemPhoto itemId={p.id} imageUrl={p.image_url} alt={p.item_type} size="sm" />
                      <small>{p.item_type}</small>
                      <small className="muted">{p.color_primary}</small>
                    </Link>
                  ))}
                </div>
                <p className="muted">{o.explanation}</p>
                <Link
                  className="btn secondary"
                  to={`/finish-my-fit/${anchor.id}`}
                  state={{ preselectedIds: o.piece_ids.filter((pid) => pid !== anchor.id) }}
                >
                  Open in Finish My Fit
                </Link>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="card">
          No complete outfits yet — add a few more wardrobe items (top, bottom, shoes) to unlock full-look suggestions.
        </div>
      )}
    </section>
  )
}
