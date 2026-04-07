import { NavLink as RouterLink } from 'react-router-dom'
import type { ReactNode } from 'react'

interface NavLinkProps {
  to: string
  children: ReactNode
}

export function NavLink({ to, children }: NavLinkProps) {
  return (
    <RouterLink className={({ isActive }) => (isActive ? 'active-link' : 'inactive-link')} to={to}>
      {children}
    </RouterLink>
  )
}
