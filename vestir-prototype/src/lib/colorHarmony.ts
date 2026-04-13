import type { Item } from '../types/index'

function hueDiff(a: number, b: number) {
  const diff = Math.abs(a - b)
  return Math.min(diff, 360 - diff)
}

const BUSY_PATTERNS = new Set(['stripe', 'plaid', 'graphic', 'floral', 'check', 'mixed', 'texture'])

function patternKey(item: Item): string {
  const p = (item.pattern ?? '').trim().toLowerCase()
  return p || 'solid'
}

function patternModifier(anchor: Item, candidate: Item): number {
  const pa = patternKey(anchor)
  const pb = patternKey(candidate)
  const busyA = BUSY_PATTERNS.has(pa)
  const busyB = BUSY_PATTERNS.has(pb)
  if (busyA && busyB) return -12
  if (busyA !== busyB) return 4
  return 0
}

function colorScore(anchor: Item, candidate: Item) {
  const satA = anchor.color_primary_hsl.s
  const satB = candidate.color_primary_hsl.s
  const diff = hueDiff(anchor.color_primary_hsl.h, candidate.color_primary_hsl.h)

  let base: number
  if (satA < 0.15 || satB < 0.15) base = 38
  else if (diff >= 165 && diff <= 195) base = 40
  else if (diff <= 40) base = 35
  else if (diff < 15) base = 32
  else if (diff >= 110 && diff <= 130) base = 30
  else if (diff >= 60 && diff <= 160 && satA > 0.35 && satB > 0.35) base = 5
  else base = 20
  return base + patternModifier(anchor, candidate)
}

function formalityScore(anchor: Item, candidate: Item) {
  const diff = Math.abs(anchor.formality - candidate.formality)
  if (diff === 0) return 30
  if (diff === 1) return 25
  if (diff === 2) return 15
  return 0
}

const adjacentSeasons: Record<string, string[]> = {
  spring: ['summer', 'winter'],
  summer: ['spring', 'autumn'],
  autumn: ['summer', 'winter'],
  winter: ['autumn', 'spring'],
}

function seasonScore(anchor: Item, candidate: Item) {
  const direct = anchor.season.some((s) => candidate.season.includes(s))
  if (direct) return 20
  const adjacent = anchor.season.some((s) => (adjacentSeasons[s] ?? []).some((a) => candidate.season.includes(a)))
  return adjacent ? 15 : 5
}

export function calculateCompatibility(anchor: Item, candidate: Item) {
  return (
    colorScore(anchor, candidate) + formalityScore(anchor, candidate) + seasonScore(anchor, candidate)
  )
}
