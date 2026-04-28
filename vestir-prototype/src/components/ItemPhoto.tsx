import { ITEM_IMAGES } from '../data/images'
import { useMemo, useState } from 'react'

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

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

function storagePathForDevProxy(url: string): string | null {
  if (!import.meta.env.DEV) return null
  const m = url.match(/^https?:\/\/(?:127\.0\.0\.1|localhost)(?::\d+)?(\/storage\/[^?#]*)((?:\?|#).*)?$/i)
  if (!m) return null
  return `${m[1]}${m[2] ?? ''}`
}

function toDisplayUrl(url?: string) {
  if (!url) return ''
  if (url.startsWith('data:')) return url
  const resolved = url.startsWith('http') ? url : `${API_BASE}${url}`
  const proxied = storagePathForDevProxy(resolved)
  return proxied ?? resolved
}

export function ItemPhoto({ itemId, imageUrl, alt, size = 'full', className }: ItemPhotoProps) {
  const [failed, setFailed] = useState(false)
  const primarySrc = useMemo(() => toDisplayUrl(imageUrl), [imageUrl])
  const fallbackSrc = ITEM_IMAGES[itemId]
  const src = failed ? fallbackSrc : (primarySrc || fallbackSrc)

  if (!src) {
    return (
      <div className={`photo-fallback ${sizeClasses[size]} ${className ?? ''}`} role="img" aria-label={alt}>
        Photo not available
      </div>
    )
  }
  return (
    <img
      src={src}
      alt={alt}
      className={`${sizeClasses[size]} ${className ?? ''}`}
      onError={() => setFailed(true)}
    />
  )
}
