#!/usr/bin/env node
/**
 * Spec assertions for the work delivered in the current cycle:
 *   1. Finish My Fit v3 compatibility engine (color + formality + season rules,
 *      clash overrides, multi-item aggregation, UI mapping).
 *   2. Post-pipeline Ollama suggestion contract shape + heuristic fallback.
 *   3. Face-region filter that protects garment logos/text from being blurred.
 *
 * This file intentionally avoids external test runners so it can run anywhere
 * with `node scripts/test-today-cycle.mjs`. It mirrors the TypeScript engine
 * logic in `src/lib/finishMyFitEngine.ts`. Keep rules in sync with the spec.
 */

let failures = 0
function assert(cond, label) {
  if (!cond) {
    failures += 1
    console.error(`  FAIL  ${label}`)
  } else {
    console.log(`  ok    ${label}`)
  }
}

/* ------------------------------------------------------------------ */
/* Finish My Fit engine (mirror of src/lib/finishMyFitEngine.ts)       */
/* ------------------------------------------------------------------ */

function itemOklch(item) {
  const hsl = item.color_primary_hsl ?? { h: 0, s: 0, l: 0.5 }
  // Note: oklchFromHsl is defined further below (in the OKLCH test block).
  // Node hoists function declarations within the same module, so this works
  // even though colorScore is invoked in tests defined before the OKLCH block.
  return oklchFromHsl(hsl.h, hsl.s, hsl.l)
}

function isNeutral(item) {
  return itemOklch(item).C < 0.045
}

function hueDistance(a, b) {
  const diff = Math.abs(a - b) % 360
  return diff > 180 ? 360 - diff : diff
}

function colorScore(a, b) {
  const oa = itemOklch(a)
  const ob = itemOklch(b)
  if (oa.C < 0.045 && ob.C < 0.045) return { score: 35, classification: 'harmonious' }
  if (oa.C < 0.045 || ob.C < 0.045) return { score: 28, classification: 'harmonious' }
  const d = hueDistance(oa.h, ob.h)
  if (d <= 12) return { score: 35, classification: 'harmonious' }
  if (d <= 35) return { score: 32, classification: 'harmonious' }
  if (d >= 165 && d <= 195) return { score: 38, classification: 'harmonious' }
  if (d >= 115 && d <= 135) {
    if (oa.C > 0.16 && ob.C > 0.16) return { score: 8, classification: 'clash' }
    return { score: 22, classification: 'harmonious' }
  }
  const avgC = (oa.C + ob.C) / 2
  if (avgC < 0.09) return { score: 30, classification: 'harmonious' }
  return { score: 8, classification: 'clash' }
}

function formalityClashThreshold(a, b) {
  const key = `${a.toLowerCase()}_${b.toLowerCase()}`
  const rev = `${b.toLowerCase()}_${a.toLowerCase()}`
  const map = { shoes_bottoms: 2, tops_outerwear: 3 }
  return map[key] ?? map[rev] ?? 3
}

function formalityScore(a, b) {
  const gap = Math.abs(a.formality - b.formality)
  if (gap === 0) return 30
  if (gap === 1) return 25
  if (gap === 2) return 15
  return 0
}

function seasonScore(a, b) {
  const as = new Set((a.season ?? []).map((s) => s.toLowerCase()))
  const bs = new Set((b.season ?? []).map((s) => s.toLowerCase()))
  const shared = [...as].some((s) => bs.has(s))
  if (shared) return 20
  const warmA = as.has('spring') || as.has('summer')
  const warmB = bs.has('spring') || bs.has('summer')
  return warmA === warmB ? 15 : 5
}

function pairScore(a, b) {
  const c = colorScore(a, b)
  const fGap = Math.abs(a.formality - b.formality)
  const f = formalityScore(a, b)
  const s = seasonScore(a, b)
  let total = c.score + f + s
  if (fGap >= 4) total = Math.min(total, 20)
  const oa = itemOklch(a)
  const ob = itemOklch(b)
  if (c.classification === 'clash' && oa.C > 0.16 && ob.C > 0.16) {
    total = Math.min(total, 20)
  }
  if (fGap >= 3 && c.score <= 12) total = Math.min(total, 28)
  const isClash = c.classification === 'clash' || fGap >= formalityClashThreshold(a.category, b.category)
  return { total, isClash }
}

function evaluateCandidate(selected, candidate) {
  const pairs = selected.map((it) => pairScore(it, candidate))
  const min = Math.min(...pairs.map((p) => p.total))
  const avg = pairs.reduce((sum, p) => sum + p.total, 0) / pairs.length
  const score = Math.max(0, Math.min(88, min * 0.6 + avg * 0.4))
  const isClash = pairs.some((p) => p.isClash) || score < 30
  const level = isClash ? 'clash' : score >= 65 ? 'great' : score >= 45 ? 'good' : 'fair'
  return {
    score: Math.round(score),
    level,
    isClash,
    sortGroup: isClash ? 'clash' : 'valid',
    hapticType: isClash ? 'warning' : 'light',
  }
}

function make(overrides) {
  return {
    id: 'x',
    category: 'Tops',
    formality: 5,
    season: ['summer'],
    color_primary_hsl: { h: 0, s: 0, l: 0.5 },
    ...overrides,
  }
}

/* Spec tests -------------------------------------------------------- */

console.log('Finish My Fit v3 engine')

// Neutral + neutral
{
  const r = evaluateCandidate(
    [make({ category: 'Tops', formality: 5, color_primary_hsl: { h: 0, s: 0.05, l: 0.2 } })],
    make({ category: 'Bottoms', formality: 5, color_primary_hsl: { h: 0, s: 0.05, l: 0.8 } }),
  )
  assert(!r.isClash && r.level !== 'clash', 'neutrals never clash on color alone')
  assert(r.hapticType === 'light', 'clean pair uses light haptic')
}

// Complementary colors
{
  const r = evaluateCandidate(
    [make({ color_primary_hsl: { h: 10, s: 0.6, l: 0.5 } })],
    make({ color_primary_hsl: { h: 190, s: 0.55, l: 0.5 } }),
  )
  assert(!r.isClash && r.level === 'great', 'complementary hues score as great')
}

// Saturated triadic clash
{
  const r = evaluateCandidate(
    [make({ color_primary_hsl: { h: 0, s: 0.8, l: 0.5 } })],
    make({ color_primary_hsl: { h: 120, s: 0.8, l: 0.5 } }),
  )
  assert(r.isClash, 'saturated triadic is flagged as clash')
  assert(r.hapticType === 'warning', 'clash uses warning haptic')
  assert(r.sortGroup === 'clash', 'clashes sort last')
}

// Large formality gap
{
  const r = evaluateCandidate(
    [make({ category: 'Tops', formality: 0 })],
    make({ category: 'Bottoms', formality: 4 }),
  )
  assert(r.isClash, 'formality gap >= 4 is a clash')
  assert(r.score <= 20, 'gap >= 4 caps score at <=20')
}

// Category-sensitive shoes/bottoms (gap 2 should clash per v3)
{
  const r = evaluateCandidate(
    [make({ category: 'Bottoms', formality: 2 })],
    make({ category: 'Shoes', formality: 4 }),
  )
  assert(r.isClash, 'shoes/bottoms formality gap >= 2 clashes')
}

// Multi-item aggregation: one bad pair drags result down
{
  const anchor = make({ category: 'Tops', formality: 3, color_primary_hsl: { h: 0, s: 0.05, l: 0.2 } })
  const good = make({ id: 'g', category: 'Bottoms', formality: 3, color_primary_hsl: { h: 0, s: 0.05, l: 0.7 } })
  const bad = make({ id: 'b', category: 'Shoes', formality: 3, color_primary_hsl: { h: 120, s: 0.9, l: 0.5 } })
  const candidate = make({ id: 'c', category: 'Outerwear', formality: 3, color_primary_hsl: { h: 0, s: 0.85, l: 0.5 } })
  const r = evaluateCandidate([anchor, good, bad], candidate)
  assert(r.isClash, 'one bad pairing propagates to candidate')
}

// Save-behavior UI mapping
{
  const r = evaluateCandidate(
    [make({ color_primary_hsl: { h: 0, s: 0.05, l: 0.2 } })],
    make({ color_primary_hsl: { h: 0, s: 0.05, l: 0.8 } }),
  )
  assert(typeof r.score === 'number' && r.score >= 0 && r.score <= 88, 'score in [0,88]')
  assert(['great', 'good', 'fair', 'clash'].includes(r.level), 'level matches spec enum')
}

/* ------------------------------------------------------------------ */
/* Post-pipeline suggestion contract                                   */
/* ------------------------------------------------------------------ */

console.log('\nPost-pipeline suggestion contract')

function validateSuggestPayload(body) {
  if (!body || typeof body !== 'object') return 'body must be object'
  if (!Array.isArray(body.suggestions)) return 'suggestions must be array'
  for (const s of body.suggestions) {
    if (typeof s.item_id !== 'string') return 'suggestions[].item_id must be string'
    if (typeof s.score !== 'number' || s.score < 0 || s.score > 1)
      return 'suggestions[].score must be in [0,1]'
    if (typeof s.explanation !== 'string') return 'suggestions[].explanation must be string'
  }
  if (typeof body.summary !== 'string') return 'summary must be string'
  const src = body?.metadata?.source
  if (src && !['ollama+heuristic', 'heuristic-only', 'cache-fallback'].includes(src))
    return 'metadata.source must be one of allowed values'
  return null
}

{
  const ok = {
    summary: 'x',
    suggestions: [{ item_id: 'a', score: 0.82, explanation: 'neutral pair' }],
    metadata: { provider: 'vestir', model: 'heuristic', source: 'heuristic-only', version: '1.0.0' },
  }
  assert(validateSuggestPayload(ok) === null, 'valid heuristic-only payload passes')
}

{
  const bad = {
    summary: 'x',
    suggestions: [{ item_id: 'a', score: 2, explanation: 'invalid' }],
    metadata: { source: 'heuristic-only' },
  }
  assert(validateSuggestPayload(bad) !== null, 'score > 1 rejected')
}

{
  const bad = {
    summary: 'x',
    suggestions: [],
    metadata: { source: 'nonsense' },
  }
  assert(validateSuggestPayload(bad) !== null, 'unknown metadata.source rejected')
}

/* ------------------------------------------------------------------ */
/* Face-region filter protects garment logos/text                      */
/* ------------------------------------------------------------------ */

console.log('\nFace-region false-positive filter')

// Mirrors isLikelyTrueFaceRegion in server/index.mjs.
function isLikelyFace(region, w, h) {
  if (!region || !w || !h) return false
  const areaFrac = (region.width * region.height) / (w * h)
  if (areaFrac < 0.005) return false
  if (areaFrac > 0.9) return false
  const aspect = region.width / region.height
  if (aspect < 0.45 || aspect > 1.9) return false
  return true
}

{
  // Large near-square face (top-center): accepted.
  assert(isLikelyFace({ width: 220, height: 240 }, 1000, 1000), 'real face dimensions accepted')
}

{
  // Long narrow logo strip (e.g., "CHANEL" across chest): rejected by aspect ratio.
  assert(!isLikelyFace({ width: 300, height: 30 }, 1000, 1000), 'long narrow logo rejected')
}

{
  // Tiny stray detection (<0.5% of frame): rejected as too small.
  assert(!isLikelyFace({ width: 20, height: 15 }, 1000, 1000), 'tiny detection rejected')
}

{
  // Whole-frame false positive: rejected.
  assert(!isLikelyFace({ width: 990, height: 990 }, 1000, 1000), 'whole-frame detection rejected')
}

/* ------------------------------------------------------------------ */
/* Fit normalization (mirrors normalizeFit in src/lib/pipeline/adapters.ts) */
/* ------------------------------------------------------------------ */

console.log('\nFit normalization')

const FIT_LABELS = ['Slim', 'Regular', 'Relaxed', 'Oversized', 'Cropped', 'Tailored']
function normalizeFit(fit) {
  if (!fit || !String(fit).trim()) return undefined
  const key = String(fit).trim().toLowerCase()
  if (key.includes('oversize')) return 'Oversized'
  if (key.includes('crop')) return 'Cropped'
  if (key.includes('tailor') || key.includes('fitted')) return 'Tailored'
  if (key.includes('slim') || key.includes('skinny') || key.includes('athletic')) return 'Slim'
  if (key.includes('relax') || key.includes('loose') || key.includes('boxy') || key.includes('baggy')) return 'Relaxed'
  if (key.includes('regular') || key.includes('classic') || key.includes('standard') || key.includes('straight')) return 'Regular'
  return FIT_LABELS.includes(fit) ? fit : undefined
}

assert(normalizeFit('slim-fit') === 'Slim', '"slim-fit" → Slim')
assert(normalizeFit('oversized boxy') === 'Oversized', 'oversized wins over boxy')
assert(normalizeFit('boxy silhouette') === 'Relaxed', 'boxy → Relaxed')
assert(normalizeFit('cropped relaxed') === 'Cropped', 'cropped wins over relaxed')
assert(normalizeFit('tailored cut') === 'Tailored', 'tailored → Tailored')
assert(normalizeFit('straight leg') === 'Regular', 'straight leg → Regular')
assert(normalizeFit(undefined) === undefined, 'undefined stays undefined')
assert(normalizeFit('unknown weird phrase') === undefined, 'unknown phrase rejected')
assert(normalizeFit('Regular') === 'Regular', 'already-canonical preserved')

/* ------------------------------------------------------------------ */
/* Outfit aggregation + slot expansion (mirrors /api/wardrobe/outfits/build) */
/* ------------------------------------------------------------------ */

console.log('\nOutfit build aggregation')

function aggregateOutfitScore(pairs) {
  if (!pairs.length) return 0
  const min = Math.min(...pairs)
  const avg = pairs.reduce((a, b) => a + b, 0) / pairs.length
  return Math.max(0, Math.min(1, min * 0.6 + avg * 0.4))
}

{
  const allHigh = aggregateOutfitScore([0.9, 0.85, 0.88, 0.92])
  assert(allHigh > 0.85, 'uniformly strong pairs yield a strong outfit score')
}
{
  const oneWeak = aggregateOutfitScore([0.9, 0.85, 0.2, 0.88])
  const allWeak = aggregateOutfitScore([0.3, 0.3, 0.3, 0.3])
  assert(oneWeak < 0.6, 'one weak pair drags the outfit score down')
  assert(oneWeak > allWeak, 'outfit with one weak pair still > uniformly weak outfit')
}
{
  const a = aggregateOutfitScore([0.6, 0.6])
  const b = aggregateOutfitScore([0.6, 0.6, 0.6])
  // avg is stable, min is stable → adding concordant pairs should not hurt
  assert(Math.abs(a - b) < 1e-9, 'adding concordant pairs keeps score stable')
}

function defaultRequiredCategoriesForAnchor(anchorCategory, weatherMode) {
  const base = new Set(['Tops', 'Bottoms', 'Shoes'])
  base.add(anchorCategory)
  if ((weatherMode ?? 'all') === 'cold') base.add('Outerwear')
  return [...base]
}

{
  const warm = defaultRequiredCategoriesForAnchor('Tops', 'warm')
  assert(!warm.includes('Outerwear'), 'warm weather excludes outerwear')
  const cold = defaultRequiredCategoriesForAnchor('Tops', 'cold')
  assert(cold.includes('Outerwear'), 'cold weather adds outerwear')
  const anchorShoes = defaultRequiredCategoriesForAnchor('Shoes', 'all')
  assert(anchorShoes.includes('Shoes'), 'anchor category always in required slots')
}

function applyProfileAndWeatherBoost(base, candidate, profile, weather) {
  let score = base
  const intent = String(profile?.style_intent ?? 'balanced').toLowerCase()
  if (intent === 'formal' && Number(candidate?.formality ?? 5) >= 7) score += 0.06
  if (intent === 'casual' && Number(candidate?.formality ?? 5) <= 5) score += 0.06
  const mode = String(weather?.mode ?? 'all').toLowerCase()
  const season = Array.isArray(candidate?.season) ? candidate.season.map((s) => String(s).toLowerCase()) : []
  if (mode === 'warm' && (season.includes('summer') || season.includes('spring'))) score += 0.05
  if (mode === 'cold' && (season.includes('winter') || season.includes('autumn'))) score += 0.05
  return Math.max(0, Math.min(1, score))
}

{
  const base = 0.7
  const formalItem = { formality: 8, season: ['autumn'] }
  const boosted = applyProfileAndWeatherBoost(base, formalItem, { style_intent: 'formal' }, { mode: 'cold' })
  assert(boosted > base, 'formal intent + cold weather boosts matching item')
  const capped = applyProfileAndWeatherBoost(0.98, formalItem, { style_intent: 'formal' }, { mode: 'cold' })
  assert(capped === 1, 'boost clamps to 1.0 ceiling')
}

/* ------------------------------------------------------------------ */
/* OKLCH color-space (mirrors src/lib/color/oklch.ts)                 */
/* ------------------------------------------------------------------ */

console.log('\nOKLCH color math')

function hslToSrgb(h, s, l) {
  const hh = ((h % 360) + 360) % 360
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hh / 60) % 2) - 1))
  const m = l - c / 2
  let r = 0, g = 0, b = 0
  if (hh < 60) { r = c; g = x }
  else if (hh < 120) { r = x; g = c }
  else if (hh < 180) { g = c; b = x }
  else if (hh < 240) { g = x; b = c }
  else if (hh < 300) { r = x; b = c }
  else { r = c; b = x }
  return { r: r + m, g: g + m, b: b + m }
}
function srgbToLinear(c) {
  if (c <= 0.04045) return c / 12.92
  return Math.pow((c + 0.055) / 1.055, 2.4)
}
function linearSrgbToOklab(r, g, b) {
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
  const l_ = Math.cbrt(l), m_ = Math.cbrt(m), s_ = Math.cbrt(s)
  return {
    L: 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    a: 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  }
}
function oklabToOklch(lab) {
  const C = Math.hypot(lab.a, lab.b)
  let h = (Math.atan2(lab.b, lab.a) * 180) / Math.PI
  if (h < 0) h += 360
  return { L: lab.L, C, h }
}
function oklchFromHsl(h, s, l) {
  const rgb = hslToSrgb(h, s, l)
  return oklabToOklch(linearSrgbToOklab(srgbToLinear(rgb.r), srgbToLinear(rgb.g), srgbToLinear(rgb.b)))
}
function oklchHueDistance(h1, h2) {
  const diff = Math.abs(h1 - h2) % 360
  return diff > 180 ? 360 - diff : diff
}
const OKLCH_NEUTRAL_CHROMA = 0.045
const OKLCH_SATURATED_CHROMA = 0.16

{
  // Pure black should collapse to a neutral (C near 0).
  const black = oklchFromHsl(0, 0, 0.05)
  assert(black.C < OKLCH_NEUTRAL_CHROMA, 'black is OKLCH neutral')

  // Pure gray too.
  const gray = oklchFromHsl(0, 0.02, 0.5)
  assert(gray.C < OKLCH_NEUTRAL_CHROMA, 'gray is OKLCH neutral')

  // Saturated red should be "saturated" in OKLCH.
  const red = oklchFromHsl(0, 1, 0.5)
  assert(red.C > OKLCH_SATURATED_CHROMA, 'saturated red passes OKLCH saturation threshold')

  // Hue distance wraps correctly.
  assert(oklchHueDistance(10, 350) === 20, 'hue distance wraps across 0/360 boundary')
  assert(oklchHueDistance(0, 180) === 180, 'opposite hues are 180° apart')

  // Navy vs olive: these register as harmonious in fashion (muted + analogous).
  const navy = oklchFromHsl(220, 0.45, 0.32)
  const olive = oklchFromHsl(95, 0.34, 0.44)
  const avgC = (navy.C + olive.C) / 2
  assert(avgC < 0.15, 'navy + olive have muted OKLCH chroma on average')
  // Not both saturated — so the triadic override should NOT fire if it hit.
  assert(!(navy.C > OKLCH_SATURATED_CHROMA && olive.C > OKLCH_SATURATED_CHROMA), 'navy + olive not both over saturation cap')
}

/* ------------------------------------------------------------------ */
/* FMF dimension reasons (mirror of worstReason / aggregate in engine) */
/* ------------------------------------------------------------------ */

console.log('\nFinish My Fit reasons (color / formality / season)')

const reasonSeverity = {
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

function colorReason(a, b, classification) {
  if (classification === 'clash') return 'clash'
  if (isNeutral(a) || isNeutral(b)) return 'neutral'
  return 'harmonious'
}

function formalityReason(gap, threshold) {
  if (gap === 0) return 'match'
  if (gap === 1) return 'adjacent'
  if (gap >= threshold) return 'hard-gap'
  return 'gap'
}

function seasonReason(a, b) {
  const as = new Set((a.season ?? []).map((s) => String(s).toLowerCase()))
  const bs = new Set((b.season ?? []).map((s) => String(s).toLowerCase()))
  const shared = [...as].some((s) => bs.has(s))
  if (shared) return 'shared'
  const warmA = as.has('spring') || as.has('summer')
  const warmB = bs.has('spring') || bs.has('summer')
  return warmA === warmB ? 'adjacent' : 'opposite'
}

{
  const navyShirt = {
    category: 'Tops',
    color_primary_hsl: { h: 220, s: 0.45, l: 0.32 },
    formality: 2,
    season: ['autumn', 'winter'],
  }
  const oliveChinos = {
    category: 'Bottoms',
    color_primary_hsl: { h: 95, s: 0.34, l: 0.44 },
    formality: 2,
    season: ['autumn', 'spring'],
  }
  const ballGown = {
    category: 'Outerwear',
    color_primary_hsl: { h: 10, s: 0.9, l: 0.45 },
    formality: 4,
    season: ['winter'],
  }
  const neonTop = {
    category: 'Tops',
    color_primary_hsl: { h: 125, s: 0.95, l: 0.5 },
    formality: 0,
    season: ['summer'],
  }
  const cobaltTop = {
    category: 'Tops',
    color_primary_hsl: { h: 240, s: 0.9, l: 0.45 },
    formality: 0,
    season: ['summer'],
  }

  const navyOliveColor = colorScore(navyShirt, oliveChinos)
  assert(
    colorReason(navyShirt, oliveChinos, navyOliveColor.classification) === 'harmonious',
    'navy + olive should be harmonious (earth tone / adjacent)',
  )

  assert(
    formalityReason(Math.abs(navyShirt.formality - ballGown.formality), 3) === 'gap',
    'formality gap 2 → 4 (gap=2 < threshold=3) surfaces as gap (soft)',
  )
  assert(
    formalityReason(3, 3) === 'hard-gap',
    'gap at or above threshold surfaces as hard-gap',
  )
  assert(
    formalityReason(4, 3) === 'hard-gap',
    'extreme formality gap surfaces as hard-gap',
  )

  const triadic = colorScore(neonTop, cobaltTop)
  assert(triadic.classification === 'clash', 'saturated triadic fires color clash')
  assert(colorReason(neonTop, cobaltTop, triadic.classification) === 'clash', 'color clash reason surfaces')

  assert(seasonReason(navyShirt, oliveChinos) === 'shared', 'navy + olive → shared season (autumn)')
  assert(seasonReason(neonTop, navyShirt) === 'opposite', 'summer + winter → opposite season')

  // Aggregate: worst reason across pairs must pick the worst severity.
  const worseOfTwo = [
    { color: 'clash', formality: 'match', season: 'shared' },
    { color: 'harmonious', formality: 'hard-gap', season: 'shared' },
  ].reduce(
    (acc, r) => ({
      color: (reasonSeverity[r.color] ?? 0) > (reasonSeverity[acc.color] ?? 0) ? r.color : acc.color,
      formality: (reasonSeverity[r.formality] ?? 0) > (reasonSeverity[acc.formality] ?? 0) ? r.formality : acc.formality,
      season: (reasonSeverity[r.season] ?? 0) > (reasonSeverity[acc.season] ?? 0) ? r.season : acc.season,
    }),
    { color: 'harmonious', formality: 'match', season: 'shared' },
  )
  assert(worseOfTwo.color === 'clash', 'aggregate preserves color clash across pairs')
  assert(worseOfTwo.formality === 'hard-gap', 'aggregate preserves hard-gap across pairs')
  assert(worseOfTwo.season === 'shared', 'aggregate keeps good season when no pair is worse')
}

/* ------------------------------------------------------------------ */

console.log('')
if (failures > 0) {
  console.error(`${failures} assertion(s) failed.`)
  process.exit(1)
}
console.log('All assertions passed.')
