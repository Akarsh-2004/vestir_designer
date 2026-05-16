import type { ReactNode } from 'react'

interface EmptyStateProps {
  title: string
  description?: string
  action?: ReactNode
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="empty-state" role="status" aria-live="polite">
      <h3>{title}</h3>
      {description ? <p>{description}</p> : null}
      {action ?? null}
    </div>
  )
}
