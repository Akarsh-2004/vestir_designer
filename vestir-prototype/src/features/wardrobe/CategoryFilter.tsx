import type { Category } from '../../types/index'
import { useWardrobeStore } from '../../store/wardrobeStore'

const CATEGORIES: Array<{ value: Category | 'All'; label: string }> = [
  { value: 'All', label: 'All' },
  { value: 'Tops', label: 'Tops' },
  { value: 'Bottoms', label: 'Bottoms' },
  { value: 'Outerwear', label: 'Outerwear' },
  { value: 'Shoes', label: 'Shoes' },
  { value: 'Accessories', label: 'Accessories' },
]

export function CategoryFilter() {
  const { activeCategory, setActiveCategory } = useWardrobeStore()
  return (
    <div className="category-row">
      {CATEGORIES.map(({ value, label }) => (
        <button
          key={value}
          type="button"
          className={`category-btn${activeCategory === value ? ' category-active' : ''}`}
          onClick={() => setActiveCategory(value)}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
