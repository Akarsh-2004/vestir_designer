import { ITEM_IMAGES } from '../data/images'

interface ItemPhotoProps {
  itemId: string
  imageUrl?: string
  alt: string
  size?: 'sm' | 'md' | 'lg' | 'full'
  className?: string
}

const sizeClasses = {
  sm: 'item-photo-sm',
  md: 'item-photo-md',
  lg: 'item-photo-lg',
  full: 'item-photo-full',
}

export function ItemPhoto({ itemId, imageUrl, alt, size = 'full', className }: ItemPhotoProps) {
  const src = imageUrl || ITEM_IMAGES[itemId]
  if (!src) return <div className={`photo-fallback ${sizeClasses[size]} ${className ?? ''}`}>No photo</div>
  return <img src={src} alt={alt} className={`${sizeClasses[size]} ${className ?? ''}`} />
}
