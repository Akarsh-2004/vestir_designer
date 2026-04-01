import type { Category } from '../../types/index'
import { useWardrobeStore } from '../../store/wardrobeStore'

const categories: Array<Category | 'All'> = ['All', 'Tops', 'Bottoms', 'Outerwear', 'Shoes', 'Accessories']

export function CategoryFilter() {
  const { activeCategory, setActiveCategory } = useWardrobeStore()
  return (
    <div className="category-row">
      {categories.map((category) => (
        <button
          key={category}
          className={`category-btn ${activeCategory === category ? 'category-active' : ''}`}
          onClick={() => setActiveCategory(category)}
        >
          {category}
        </button>
      ))}
    </div>
  )
}
