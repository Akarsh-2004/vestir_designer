/**
 * OKLCH color utilities for the Finish My Fit compatibility engine.
 *
 * Why OKLCH instead of HSL?
 *   HSL hue distance is perceptually uneven — a 30° rotation at yellow feels
 *   very different from a 30° rotation at blue. OKLab / OKLCH is designed for
 *   perceptual uniformity (Björn Ottosson, 2020), so equal hue distances map
 *   to equal perceived color differences. This matters for outfit matching:
 *   navy + olive (technically a large HSL hue gap) actually reads as
 *   harmonious, while cobalt + neon green (a smaller HSL gap) reads as a
 *   clash. OKLCH gets those calls right.
 *
 * Pipeline: HSL -> sRGB (0..1) -> linear sRGB -> OKLab -> OKLCH.
 * Hue is in degrees (0..360), chroma is 0..~0.37, lightness is 0..1.
 */

export interface Oklch {
  L: number
  C: number
  h: number
}

/** Standard HSL -> sRGB conversion (all inputs 0..1, 0..1, 0..1). */
function hslToSrgb(h: number, s: number, l: number): { r: number; g: number; b: number } {
  const hh = ((h % 360) + 360) % 360
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hh / 60) % 2) - 1))
  const m = l - c / 2
  let r = 0
  let g = 0
  let b = 0
  if (hh < 60) {
    r = c
    g = x
  } else if (hh < 120) {
    r = x
    g = c
  } else if (hh < 180) {
    g = c
    b = x
  } else if (hh < 240) {
    g = x
    b = c
  } else if (hh < 300) {
    r = x
    b = c
  } else {
    r = c
    b = x
  }
  return { r: r + m, g: g + m, b: b + m }
}

/** sRGB companding: 0..1 gamma-encoded -> 0..1 linear-light. */
function srgbChannelToLinear(c: number): number {
  if (c <= 0.04045) return c / 12.92
  return Math.pow((c + 0.055) / 1.055, 2.4)
}

/** Linear sRGB -> OKLab (per Ottosson). */
function linearSrgbToOklab(r: number, g: number, b: number): { L: number; a: number; b: number } {
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

  const l_ = Math.cbrt(l)
  const m_ = Math.cbrt(m)
  const s_ = Math.cbrt(s)

  return {
    L: 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    a: 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  }
}

/** OKLab -> OKLCH (polar form). */
function oklabToOklch(lab: { L: number; a: number; b: number }): Oklch {
  const C = Math.hypot(lab.a, lab.b)
  let h = (Math.atan2(lab.b, lab.a) * 180) / Math.PI
  if (h < 0) h += 360
  return { L: lab.L, C, h }
}

/**
 * Convert HSL (h in 0..360, s/l in 0..1) to OKLCH. When stored color metadata
 * only has HSL, this is the bridge into perceptual space.
 */
export function oklchFromHsl(h: number, s: number, l: number): Oklch {
  const rgb = hslToSrgb(h, s, l)
  const rLin = srgbChannelToLinear(rgb.r)
  const gLin = srgbChannelToLinear(rgb.g)
  const bLin = srgbChannelToLinear(rgb.b)
  const lab = linearSrgbToOklab(rLin, gLin, bLin)
  return oklabToOklch(lab)
}

/** Cylindrical hue distance on the OKLCH hue wheel (degrees, 0..180). */
export function oklchHueDistance(h1: number, h2: number): number {
  const diff = Math.abs(h1 - h2) % 360
  return diff > 180 ? 360 - diff : diff
}

/**
 * Chroma threshold for treating a color as "neutral" in OKLCH. Ottosson's
 * OKLCH chroma maxes around ~0.37 for saturated primaries; values under
 * ~0.045 look achromatic (black/white/gray/charcoal). Fashion-domain tuning:
 * navy (~C=0.10), olive (~C=0.08), denim wash (~C=0.06) should all still be
 * treated as colored, not neutral, so we pick 0.045 rather than 0.06.
 */
export const OKLCH_NEUTRAL_CHROMA = 0.045

/**
 * Chroma threshold above which both items are considered "saturated" for the
 * triadic-clash override (cobalt + neon green kind of combos). Equivalent to
 * HSL s > 0.54 but in OKLCH terms. Calibrated against Ottosson's primaries.
 */
export const OKLCH_SATURATED_CHROMA = 0.16

/**
 * Perceptual color-zone classification on OKLCH. Zones intentionally mirror
 * the HSL zones in v3 spec but with perceptually uniform boundaries.
 *
 * Returned values:
 *   - monochromatic: hueDistance <= 12
 *   - analogous:     12 < hueDistance <= 35
 *   - complementary: 165 <= hueDistance <= 195
 *   - triadic:       115 <= hueDistance <= 135
 *   - discord:       anything else in 35..165 OR 195..325 that isn't covered
 */
export type OklchHarmonyZone = 'monochromatic' | 'analogous' | 'complementary' | 'triadic' | 'discord'

export function oklchHarmonyZone(a: Oklch, b: Oklch): OklchHarmonyZone {
  const d = oklchHueDistance(a.h, b.h)
  if (d <= 12) return 'monochromatic'
  if (d <= 35) return 'analogous'
  if (d >= 165 && d <= 195) return 'complementary'
  if (d >= 115 && d <= 135) return 'triadic'
  return 'discord'
}
