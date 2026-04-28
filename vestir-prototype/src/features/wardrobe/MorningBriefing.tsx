import { useMemo } from 'react'
import { Sun, Cloud, CloudRain } from 'lucide-react'
import { useWardrobeStore } from '../../store/wardrobeStore'

function getGreeting(): string {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

// Very lightweight weather sim based on current hour (replace with real API later)
function getMockWeather(): { icon: React.ReactNode; temp: string; desc: string } | null {
  const h = new Date().getHours()
  if (h >= 22 || h < 6) return null // night — skip weather card
  const conditions = [
    { icon: <Sun size={20} className="briefing-weather-icon briefing-weather-icon--sun" />, temp: '22°C', desc: 'Sunny · Perfect for light layers' },
    { icon: <Cloud size={20} className="briefing-weather-icon briefing-weather-icon--cloud" />, temp: '17°C', desc: 'Cloudy · Layer up a little' },
    { icon: <CloudRain size={20} className="briefing-weather-icon briefing-weather-icon--cloud" />, temp: '14°C', desc: 'Rainy · Grab a jacket' },
  ]
  return conditions[h % 3]
}

export function MorningBriefing() {
  const items   = useWardrobeStore((s) => s.items)
  const outfits = useWardrobeStore((s) => s.outfits)
  const activeWardrobeId = useWardrobeStore((s) => s.activeWardrobeId)

  const stats = useMemo(() => {
    const wardrobeItems = items.filter((i) => !i.deleted_at && i.wardrobe_id === activeWardrobeId)
    const wardrobeOutfits = outfits.filter((o) => {
      const ws = (o.items ?? []).map((it) => it.wardrobe_id).filter(Boolean)
      return ws.length === 0 || ws.includes(activeWardrobeId)
    })
    const ready = wardrobeItems.filter((i) => i.ai_processed)
    const processing = wardrobeItems.filter((i) => !i.ai_processed)
    return {
      total: wardrobeItems.length,
      outfits: wardrobeOutfits.length,
      ready: ready.length,
      processing: processing.length,
    }
  }, [items, outfits, activeWardrobeId])

  const weather = getMockWeather()

  return (
    <div className="briefing-card">
      <div className="briefing-copy">
        <p className="briefing-greeting-text">{getGreeting()}</p>
        <p className="briefing-overview">
          {stats.total} items · {stats.outfits} outfits · {stats.ready} styled
        </p>
      </div>

      {weather && (
        <div className="briefing-weather">
          {weather.icon}
          <span className="briefing-weather-temp">{weather.temp}</span>
          <span className="briefing-weather-desc">{weather.desc}</span>
        </div>
      )}

      {stats.processing > 0 && (
        <div className="briefing-processing">
          <div className="briefing-processing-dot" />
          Analysing {stats.processing} item{stats.processing > 1 ? 's' : ''}
        </div>
      )}
    </div>
  )
}
