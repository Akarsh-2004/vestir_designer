import { useWardrobeStore } from '../../store/wardrobeStore'

export function WardrobePills() {
  const { wardrobes, activeWardrobeId, setActiveWardrobe } = useWardrobeStore()
  return (
    <div className="pills-row">
      {wardrobes.map((wardrobe) => (
        <button
          key={wardrobe.id}
          className={`pill-btn ${activeWardrobeId === wardrobe.id ? 'pill-active' : ''}`}
          onClick={() => setActiveWardrobe(wardrobe.id)}
        >
          {wardrobe.name}
        </button>
      ))}
    </div>
  )
}
