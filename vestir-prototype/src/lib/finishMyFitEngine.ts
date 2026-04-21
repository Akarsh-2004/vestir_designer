import type { Category, Item } from '../types/index'
import {
  OKLCH_NEUTRAL_CHROMA,
  OKLCH_SATURATED_CHROMA,
  oklchFromHsl,
  oklchHarmonyZone,
  type Oklch,
} from './color/oklch'

export type HarmonyLevel = 'great' | 'good' | 'fair' | 'clash'
export type ColorReason = 'harmonious' | 'neutral' | 'clash'
export type FormalityReason = 'match' | 'adjacent' | 'gap' | 'hard-gap'
export type SeasonReason = 'shared' | 'adjacent' | 'opposite'

export interface CandidateReasons {
  color: ColorReason
  formality: FormalityReason
  season: SeasonReason
}

export interface CandidateEvaluation {
  score: number
  level: HarmonyLevel
  isClash: boolean
  sortGroup: 'valid' | 'clash'
  hapticType: 'light' | 'warning'
  /**
   * Qualitative per-dimension reasons (per Finish My Fit v3 spec). Surfaced
   * in the UI as small badges so the user sees *why* an item ranks the way
   * it does without exposing raw scores.
   */
  reasons: CandidateReasons
  /**
   * The hardest constraint among all paired items. Used by the UI to show a
   * single worst-paired label on the card (e.g. "formality gap with Tops").
   */
  worstPairWith?: string
}

function itemOklch(item: Item): Oklch {
  const hsl = item.color_primary_hsl ?? { h: 0, s: 0, l: 0.5 }
  return oklchFromHsl(hsl.h, hsl.s, hsl.l)
}

function isNeutralOklch(oklch: Oklch) {
  return oklch.C < OKLCH_NEUTRAL_CHROMA
}

/**
 * Perceptual color classification using OKLCH. Mirrors the v3 spec's color
 * dimensions but with perceptually uniform hue zones. Earth-tone and neutral
 * rules now use OKLCH chroma (not HSL saturation), so muted colors like
 * terracotta + olive no longer fire false clashes.
 */
function colorScore(a: Item, b: Item) {
  const oa = itemOklch(a)
  const ob = itemOklch(b)
  if (isNeutralOklch(oa) && isNeutralOklch(ob)) return { score: 35, classification: 'harmonious' as const }
  if (isNeutralOklch(oa) || isNeutralOklch(ob)) return { score: 28, classification: 'harmonious' as const }

  const zone = oklchHarmonyZone(oa, ob)
  if (zone === 'monochromatic') return { score: 35, classification: 'harmonious' as const }
  if (zone === 'analogous') return { score: 32, classification: 'harmonious' as const }
  if (zone === 'complementary') return { score: 38, classification: 'harmonious' as const }
  if (zone === 'triadic') {
    // Saturated-triadic override uses OKLCH chroma (perceptual saturation).
    if (oa.C > OKLCH_SATURATED_CHROMA && ob.C > OKLCH_SATURATED_CHROMA) {
      return { score: 8, classification: 'clash' as const }
    }
    return { score: 22, classification: 'harmonious' as const }
  }

  // Discord zone: earth-tone rule — if average chroma is low, downgrade to
  // harmonious (prevents false clashes like terracotta + olive, sage + tan).
  const avgC = (oa.C + ob.C) / 2
  if (avgC < 0.09) return { score: 30, classification: 'harmonious' as const }
  return { score: 8, classification: 'clash' as const }
}

function formalityClashThreshold(a: Category, b: Category) {
  const key = `${a.toLowerCase()}_${b.toLowerCase()}`
  const rev = `${b.toLowerCase()}_${a.toLowerCase()}`
  const map: Record<string, number> = {
    shoes_bottoms: 2,
    tops_outerwear: 3,
  }
  return map[key] ?? map[rev] ?? 3
}

function formalityScore(a: Item, b: Item) {
  const gap = Math.abs(a.formality - b.formality)
  if (gap === 0) return 30
  if (gap === 1) return 25
  if (gap === 2) return 15
  return 0
}

function seasonScore(a: Item, b: Item) {
  const as = new Set((a.season ?? []).map((s) => s.toLowerCase()))
  const bs = new Set((b.season ?? []).map((s) => s.toLowerCase()))
  const shared = [...as].some((s) => bs.has(s))
  if (shared) return 20
  const warmA = as.has('spring') || as.has('summer')
  const warmB = bs.has('spring') || bs.has('summer')
  return warmA === warmB ? 15 : 5
}

function colorReason(a: Item, b: Item, classification: 'harmonious' | 'clash'): ColorReason {
  if (classification === 'clash') return 'clash'
  if (isNeutralOklch(itemOklch(a)) || isNeutralOklch(itemOklch(b))) return 'neutral'
  return 'harmonious'
}

function formalityReason(gap: number, threshold: number): FormalityReason {
  if (gap === 0) return 'match'
  if (gap === 1) return 'adjacent'
  if (gap >= threshold) return 'hard-gap'
  return 'gap'
}

function seasonReason(a: Item, b: Item): SeasonReason {
  const as = new Set((a.season ?? []).map((s) => s.toLowerCase()))
  const bs = new Set((b.season ?? []).map((s) => s.toLowerCase()))
  const shared = [...as].some((s) => bs.has(s))
  if (shared) return 'shared'
  const warmA = as.has('spring') || as.has('summer')
  const warmB = bs.has('spring') || bs.has('summer')
  return warmA === warmB ? 'adjacent' : 'opposite'
}

function pairScore(a: Item, b: Item) {
  const c = colorScore(a, b)
  const fGap = Math.abs(a.formality - b.formality)
  const threshold = formalityClashThreshold(a.category, b.category)
  const f = formalityScore(a, b)
  const s = seasonScore(a, b)
  let total = c.score + f + s
  if (fGap >= 4) total = Math.min(total, 20)
  // Vivid-clash override uses OKLCH chroma (perceptually uniform saturation).
  const oa = itemOklch(a)
  const ob = itemOklch(b)
  if (c.classification === 'clash' && oa.C > OKLCH_SATURATED_CHROMA && ob.C > OKLCH_SATURATED_CHROMA) {
    total = Math.min(total, 20)
  }
  if (fGap >= 3 && c.score <= 12) total = Math.min(total, 28)
  const isClash = c.classification === 'clash' || fGap >= threshold
  return {
    total,
    isClash,
    reasons: {
      color: colorReason(a, b, c.classification),
      formality: formalityReason(fGap, threshold),
      season: seasonReason(a, b),
    } satisfies CandidateReasons,
    pairedWithLabel: a.item_type ?? a.category,
  }
}

/**
 * Rank severity of each dimension so we can pick the single worst reason to
 * surface on a card. Higher = worse.
 */
const reasonSeverity: Record<string, number> = {
  'clash': 100,
  'hard-gap': 90,
  'gap': 60,
  'opposite': 40,
  'adjacent': 20,
  'neutral': 10,
  'harmonious': 0,
  'match': 0,
  'shared': 0,
}

function worstReason(reasons: CandidateReasons): keyof CandidateReasons {
  const entries: Array<[keyof CandidateReasons, string]> = [
    ['color', reasons.color],
    ['formality', reasons.formality],
    ['season', reasons.season],
  ]
  entries.sort((a, b) => (reasonSeverity[b[1]] ?? 0) - (reasonSeverity[a[1]] ?? 0))
  return entries[0][0]
}

export function evaluateCandidate(selectedItems: Item[], candidate: Item): CandidateEvaluation {
  if (selectedItems.length === 0) {
    // No anchors → treat candidate as neutral / fully available.
    return {
      score: 60,
      level: 'good',
      isClash: false,
      sortGroup: 'valid',
      hapticType: 'light',
      reasons: { color: 'harmonious', formality: 'match', season: 'shared' },
    }
  }
  const pairs = selectedItems.map((it) => pairScore(it, candidate))
  const min = Math.min(...pairs.map((p) => p.total))
  const avg = pairs.reduce((sum, p) => sum + p.total, 0) / pairs.length
  const score = Math.max(0, Math.min(88, min * 0.6 + avg * 0.4))
  const isClash = pairs.some((p) => p.isClash) || score < 30
  const level: HarmonyLevel = isClash ? 'clash' : score >= 65 ? 'great' : score >= 45 ? 'good' : 'fair'

  // Aggregate per-dimension reasons: worst wins (clash > hard-gap > gap > opposite > ...).
  const aggregate: CandidateReasons = pairs.reduce<CandidateReasons>(
    (acc, p) => ({
      color: (reasonSeverity[p.reasons.color] ?? 0) > (reasonSeverity[acc.color] ?? 0) ? p.reasons.color : acc.color,
      formality: (reasonSeverity[p.reasons.formality] ?? 0) > (reasonSeverity[acc.formality] ?? 0) ? p.reasons.formality : acc.formality,
      season: (reasonSeverity[p.reasons.season] ?? 0) > (reasonSeverity[acc.season] ?? 0) ? p.reasons.season : acc.season,
    }),
    { color: 'harmonious', formality: 'match', season: 'shared' },
  )
  const worstKey = worstReason(aggregate)
  const worstPair = pairs
    .slice()
    .sort((a, b) => (reasonSeverity[b.reasons[worstKey]] ?? 0) - (reasonSeverity[a.reasons[worstKey]] ?? 0))[0]

  return {
    score: Math.round(score),
    level,
    isClash,
    sortGroup: isClash ? 'clash' : 'valid',
    hapticType: isClash ? 'warning' : 'light',
    reasons: aggregate,
    worstPairWith: worstPair?.pairedWithLabel,
  }
}
