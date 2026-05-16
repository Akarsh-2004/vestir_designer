import type { Category, Item } from '../types/index'

/**
 * Authoritative text-based category inference.
 *
 * Returns a Category when the input clearly belongs to one, otherwise null.
 * This is the single source of truth used by:
 *  - `normalizeAttributesAdapter` after model inference
 *  - `wardrobeStore.updateItem` to prevent silent mismatches
 *  - `ItemDetailScreen` to surface a "Fix category" suggestion
 *  - `FinishMyFit` engine when reasoning about layers
 *
 * The order matters: more specific patterns are checked before more general
 * ones (e.g. "tank top" must match Tops before "top" might match anything).
 */
export function inferCategoryFromText(value?: string | null): Category | null {
  const key = value?.toString().trim().toLowerCase() ?? ''
  if (!key) return null

  if (
    /(trouser|trousers|pant|pants|jean|jeans|denim[-_ ]?(?:pant|pants)|chino|chinos|short|shorts|skirt|skort|capri|capris|legging|leggings|jogger|joggers|cargo|sweatpant|sweatpants|track[-_ ]?pant|trackpants|pyjama|pajama)/.test(
      key,
    )
  ) {
    return 'Bottoms'
  }
  if (/(jacket|coat|blazer|parka|puffer|trench|overcoat|outerwear|cardigan|windbreaker|anorak)/.test(key)) {
    return 'Outerwear'
  }
  if (
    /(shoe|shoes|sneaker|sneakers|boot|boots|loafer|loafers|heel|heels|sandal|sandals|footwear|clog|clogs|flip[-_ ]?flop|trainer|trainers|mule|mules|oxfords?|brogues?|moccasin|moccasins|espadrille|espadrilles)/.test(
      key,
    )
  ) {
    return 'Shoes'
  }
  if (
    /(belt|bag|tote|backpack|cap|hat|beanie|watch|jewelry|jewellery|necklace|earring|bracelet|ring|scarf|tie|sunglasses|glove|gloves|wallet|accessory|accessories)/.test(
      key,
    )
  ) {
    return 'Accessories'
  }
  if (
    /(shirt|tshirt|t[-_ ]?shirt|tee|polo|top|tank|cami|camisole|blouse|hoodie|sweater|sweatshirt|jumper|pullover|vest|cardigan|kurta|kurti|saree|sari|dress|frock|gown|tunic|crop[-_ ]?top|bodysuit)/.test(
      key,
    )
  ) {
    // Dresses live under Tops in this app's wardrobe taxonomy.
    return 'Tops'
  }
  return null
}

/**
 * Try to infer a category from any of the rich textual signals on an Item.
 * The first confident hit wins. Falls back to the explicitly stored category
 * (which may itself be a placeholder) if no text signal matches.
 */
export function inferCategoryFromItem(item: Pick<Item, 'item_type' | 'category' | 'style_tags' | 'fashion_tags' | 'pattern' | 'raw_attributes'>): Category | null {
  const probes: Array<string | undefined> = [
    item.item_type,
    ...(item.style_tags ?? []),
    ...(item.fashion_tags ?? []),
    item.pattern,
  ]

  for (const probe of probes) {
    const hit = inferCategoryFromText(probe)
    if (hit) return hit
  }

  // raw_attributes can be a JSON blob with subtype/garment_type/etc.
  if (item.raw_attributes) {
    try {
      const parsed = JSON.parse(item.raw_attributes) as Record<string, unknown>
      const fields: Array<unknown> = [
        parsed.subtype,
        parsed.garment_type,
        parsed.item_type,
        parsed.category,
        parsed.fashion_descriptor,
      ]
      const tags = parsed.fashion_tags
      if (Array.isArray(tags)) {
        fields.push(...tags)
      }
      for (const field of fields) {
        if (typeof field === 'string') {
          const hit = inferCategoryFromText(field)
          if (hit) return hit
        }
      }
    } catch {
      // ignore unparseable raw_attributes
    }
  }
  return null
}

/**
 * Returns a corrected category for an item if there is a strong text signal
 * that disagrees with the currently stored category. Returns null when the
 * stored category already matches or no text signal is available.
 *
 * This is intentionally conservative: when text is ambiguous we leave the
 * stored category alone so manual user edits aren't clobbered.
 */
export function suggestCategoryFix(item: Pick<Item, 'item_type' | 'category' | 'style_tags' | 'fashion_tags' | 'pattern' | 'raw_attributes'>): Category | null {
  const inferred = inferCategoryFromItem(item)
  if (!inferred) return null
  if (inferred === item.category) return null
  return inferred
}
