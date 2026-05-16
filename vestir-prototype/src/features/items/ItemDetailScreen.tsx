import { ChevronLeft, Pencil, Trash2, X, Sparkles, Zap } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { ItemPhoto } from '../../components/ItemPhoto'
import {
  advancedGeminiInferenceAdapter,
  localPreprocessAdapter,
  normalizeAttributesAdapter,
  reasoningAdapter,
} from '../../lib/pipeline/adapters'
import { useWardrobeStore } from '../../store/wardrobeStore'
import { FIT_LABELS, type FitLabel } from '../../types/index'
import { suggestCategoryFix } from '../../lib/categoryInference'

function parseRawAttributes(raw?: string) {
  if (!raw) return null
  try {
    return JSON.parse(raw) as {
      category?: string
      metadata?: { provider?: string; model?: string }
      confidence_overall?: number
      advanced_error?: string
      pattern?: string
      style_tags?: string[]
      occasions?: string[]
      gemini_design_tags?: string[]
      gemini_style_notes?: string
      quality?: { warnings?: string[] }
      uncertainty?: { requires_user_confirmation?: boolean; uncertain_fields?: string[] }
    }
  } catch {
    return null
  }
}

function parseLooseJson(raw?: string) {
  if (!raw) return {}
  try {
    return JSON.parse(raw) as Record<string, unknown>
  } catch {
    return {}
  }
}

function hslToStyle(hsl?: { h: number; s: number; l: number }) {
  if (!hsl) return undefined
  return `hsl(${hsl.h}, ${Math.round(hsl.s * 100)}%, ${Math.round(hsl.l * 100)}%)`
}

function hasMaterialChange(
  before: {
    item_type?: string
    category?: string
    color_primary?: string
    material?: string
    pattern?: string
    fit?: string
    season?: string[]
    occasions?: string[]
    style_tags?: string[]
  },
  after: {
    item_type?: string
    category?: string
    color_primary?: string
    material?: string
    pattern?: string
    fit?: string
    season?: string[]
    occasions?: string[]
    style_tags?: string[]
  },
) {
  const changed = (
    before.item_type !== after.item_type ||
    before.category !== after.category ||
    before.color_primary !== after.color_primary ||
    before.material !== after.material ||
    before.pattern !== after.pattern ||
    before.fit !== after.fit ||
    JSON.stringify(before.season ?? []) !== JSON.stringify(after.season ?? []) ||
    JSON.stringify(before.occasions ?? []) !== JSON.stringify(after.occasions ?? []) ||
    JSON.stringify(before.style_tags ?? []) !== JSON.stringify(after.style_tags ?? [])
  )
  return changed
}

function shouldProtectIdentity(rawInferred: Record<string, unknown>) {
  const confidence = typeof rawInferred.confidence_overall === 'number' ? rawInferred.confidence_overall : undefined
  const quality = (rawInferred.quality ?? {}) as { warnings?: unknown }
  const warnings = Array.isArray(quality.warnings)
    ? quality.warnings.filter((w): w is string => typeof w === 'string').map((w) => w.toLowerCase())
    : []
  const uncertainty = (rawInferred.uncertainty ?? {}) as { requires_user_confirmation?: unknown; uncertain_fields?: unknown }
  const requiresUserConfirmation = Boolean(uncertainty.requires_user_confirmation)
  const uncertainFields = Array.isArray(uncertainty.uncertain_fields)
    ? uncertainty.uncertain_fields.filter((f): f is string => typeof f === 'string').map((f) => f.toLowerCase())
    : []
  const hasBlurWarning = warnings.some((w) => w.includes('blur'))
  const identityUncertain = uncertainFields.some((f) => f.includes('item_type') || f.includes('category'))
  const lowConfidence = typeof confidence === 'number' && confidence < 0.72
  return hasBlurWarning || requiresUserConfirmation || identityUncertain || lowConfidence
}

export function ItemDetailScreen() {
  const navigate = useNavigate()
  const { id } = useParams()
  const item = useWardrobeStore((s) => s.items.find((it) => it.id === id && !it.deleted_at))
  const deleteItem = useWardrobeStore((s) => s.deleteItem)
  const updateItem = useWardrobeStore((s) => s.updateItem)
  const completeAttributeReview = useWardrobeStore((s) => s.completeAttributeReview)
  const itemSuggestions = useWardrobeStore((s) => s.itemSuggestions[item?.id ?? ''])
  const refreshPostPipelineSuggestions = useWardrobeStore((s) => s.refreshPostPipelineSuggestions)
  const [advancedRunning, setAdvancedRunning] = useState(false)
  const [embedBusy, setEmbedBusy] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editedType, setEditedType] = useState('')
  const [editedCategory, setEditedCategory] = useState<'Tops' | 'Bottoms' | 'Outerwear' | 'Shoes' | 'Accessories'>('Tops')
  const [editedColor, setEditedColor] = useState('')
  const [editedMaterial, setEditedMaterial] = useState('')
  const [editedPattern, setEditedPattern] = useState('')
  const [editedFit, setEditedFit] = useState<'' | FitLabel>('')

  const parsedAttrs = useMemo(() => parseRawAttributes(item?.raw_attributes), [item?.raw_attributes])
  const suggestedCategory = useMemo(() => (item ? suggestCategoryFix(item) : null), [item])
  const aiNotes = useMemo(() => {
    const q = parsedAttrs?.quality?.warnings ?? []
    const u = parsedAttrs?.uncertainty
    const intro =
      u?.requires_user_confirmation && (u.uncertain_fields?.length ?? 0) > 0
        ? 'Some fields may need a quick check.'
        : null
    return [...(intro ? [intro] : []), ...q]
  }, [parsedAttrs])
  const styleTags = item?.style_tags?.length ? item.style_tags : (parsedAttrs?.style_tags ?? [])
  const occasions = item?.occasions?.length ? item.occasions : (parsedAttrs?.occasions ?? [])
  const pattern = item?.pattern ?? parsedAttrs?.pattern
  const designTags = parsedAttrs?.gemini_design_tags ?? []
  const styleNotes = parsedAttrs?.gemini_style_notes
  const provider = parsedAttrs?.metadata?.provider
  const model = parsedAttrs?.metadata?.model
  const confidenceOverall = parsedAttrs?.confidence_overall
  const advancedError = parsedAttrs?.advanced_error
  const providerLower = (provider ?? '').toLowerCase()
  const hasAdvancedOutput = designTags.length > 0 || Boolean(styleNotes) || providerLower.includes('gemma-mlx')
  const hideFallbackSummary = providerLower.includes('fallback')

  useEffect(() => {
    if (!item) return
    setEditedType(item.item_type ?? '')
    setEditedCategory(item.category)
    setEditedColor(item.color_primary ?? '')
    setEditedMaterial(item.material ?? '')
    setEditedPattern(item.pattern ?? '')
    setEditedFit(item.fit ?? '')
  }, [item?.id])

  if (!item) {
    return (
      <div className="screen" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--muted)' }}>Item not found.</p>
        <Link to="/" className="btn" style={{ marginTop: 12 }}>Back to wardrobe</Link>
      </div>
    )
  }

  const showAttrs = item.processing_status === 'done'
  const needsEmbedding = Boolean(item.attribute_review_pending)
  const processingDone = item.ai_processed && item.processing_stage === 'complete'
  const progress = item.processing_progress ?? 0

  const statusLabel = processingDone
    ? 'Ready to style'
    : needsEmbedding
      ? 'Review needed'
      : item.processing_stage
        ? item.processing_stage.replace(/_/g, ' ')
        : 'Queued'

  const primaryColor = hslToStyle(item.color_primary_hsl) ?? item.color_primary
  const secondaryColor = item.color_secondary_hsl
    ? hslToStyle(item.color_secondary_hsl)
    : undefined

  async function runAdvancedAnalysis() {
    if (!item || advancedRunning) return
    setAdvancedRunning(true)
    const toastId = toast.loading('Running advanced analysis…')
    try {
      const sourceUrl = item.cutout_url ?? item.image_url
      const preprocessed = await localPreprocessAdapter(sourceUrl)
      const inferred = await advancedGeminiInferenceAdapter(preprocessed.processedImageUrl)
      const normalized = await normalizeAttributesAdapter(inferred)
      const inferredRaw = inferred as unknown as Record<string, unknown>
      const protectIdentity = shouldProtectIdentity(inferredRaw)
      const safeNormalized = protectIdentity
        ? {
            ...normalized,
            item_type: item.item_type,
            category: item.category,
          }
        : normalized
      const merged = { ...item, ...safeNormalized }
      const reasoning = await reasoningAdapter(merged)
      const currentRaw = parseLooseJson(item.raw_attributes)
      const nextRaw = {
        ...currentRaw,
        ...inferredRaw,
        metadata: (inferred as unknown as { metadata?: Record<string, unknown> })?.metadata ?? currentRaw.metadata,
        advanced_error: undefined,
      }
      const nextPatch = {
        ...safeNormalized,
        reasoning_summary: reasoning.summary,
        raw_attributes: JSON.stringify(nextRaw, null, 2),
        processing_error: undefined,
      }
      const changed = hasMaterialChange(
        {
          item_type: item.item_type,
          category: item.category,
          color_primary: item.color_primary,
          material: item.material,
          pattern: item.pattern,
          fit: item.fit,
          season: item.season,
          occasions: item.occasions,
          style_tags: item.style_tags,
        },
        {
          item_type: nextPatch.item_type as string | undefined,
          category: nextPatch.category as string | undefined,
          color_primary: nextPatch.color_primary as string | undefined,
          material: nextPatch.material as string | undefined,
          pattern: nextPatch.pattern as string | undefined,
          fit: nextPatch.fit as string | undefined,
          season: nextPatch.season as string[] | undefined,
          occasions: nextPatch.occasions as string[] | undefined,
          style_tags: nextPatch.style_tags as string[] | undefined,
        },
      )
      updateItem(item.id, nextPatch)
      if (changed) {
        toast.success(
          protectIdentity
            ? 'Analysis updated (kept original type/category due to low confidence).'
            : 'Analysis updated with new attributes.',
          { id: toastId },
        )
      } else {
        toast.message('Analysis completed, but no stronger attribute changes were found.')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Advanced analysis failed'
      const currentRaw = parseLooseJson(item.raw_attributes)
      updateItem(item.id, { raw_attributes: JSON.stringify({ ...currentRaw, advanced_error: message }, null, 2) })
      toast.error(message, { id: toastId })
    } finally {
      setAdvancedRunning(false)
    }
  }

  async function runFinishEmbedding() {
    if (!item || embedBusy) return
    setEmbedBusy(true)
    const toastId = toast.loading('Indexing…')
    try {
      await completeAttributeReview(item.id)
      toast.success('Done.', { id: toastId })
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed', { id: toastId })
    } finally {
      setEmbedBusy(false)
    }
  }

  return (
    <div className="item-detail">
      {/* Header */}
      <header className="item-detail__header">
        <button className="back-btn" type="button" onClick={() => navigate(-1)} aria-label="Back">
          <ChevronLeft size={20} />
        </button>
        <div style={{ flex: 1 }} />
        <button
          className="icon-btn"
          type="button"
          onClick={() => setIsEditing((v) => !v)}
          aria-label={isEditing ? 'Close editor' : 'Edit item'}
        >
          {isEditing ? <X size={16} /> : <Pencil size={16} />}
        </button>
      </header>

      {/* Hero photo */}
      <div className="item-detail__hero">
        <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} />
      </div>

      <div className="item-detail__body">

        {/* Processing status */}
        {!processingDone && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--cocoa)' }}>{statusLabel}</span>
              {progress > 0 && <span style={{ fontSize: 11, color: 'var(--muted)' }}>{progress}%</span>}
            </div>
            {progress > 0 && (
              <div className="processing-bar">
                <div className="processing-bar__fill" style={{ width: `${progress}%` }} />
              </div>
            )}
          </div>
        )}

        {/* Review banner */}
        {needsEmbedding && (
          <div className="detail-notes" style={{ borderColor: 'var(--warning)' }}>
            <strong>Check category &amp; type</strong>
            <p>Vision models disagreed on this crop. Fix any mistakes below, then tap <em>Finish embedding</em>.</p>
          </div>
        )}

        {/* Title + subtitle */}
        <div>
          <h1 className="item-detail__title">
            {showAttrs ? item.item_type : 'Analysing…'}
          </h1>
          <p className="item-detail__subtitle">
            {showAttrs
              ? [item.color_primary, item.material].filter(Boolean).join(' · ')
              : 'Working in the background…'}
          </p>
        </div>

        {/* Category mismatch suggestion: when item_type clearly belongs to
            another category (e.g. "trousers" stored as Tops), surface a
            one-tap fix so users don't have to open the edit form. */}
        {showAttrs && suggestedCategory && (
          <div className="category-fix">
            <div className="category-fix__copy">
              <strong>Looks like {suggestedCategory.toLowerCase()}</strong>
              <span>
                Stored as <em>{item.category}</em> · we think this is a {suggestedCategory.toLowerCase()} piece based on
                "{item.item_type}".
              </span>
            </div>
            <button
              type="button"
              className="btn secondary category-fix__btn"
              onClick={() => {
                updateItem(item.id, {
                  category: suggestedCategory,
                  style_tags: Array.from(new Set([...(item.style_tags ?? []), 'UserCorrected'])),
                })
                toast.success(`Moved to ${suggestedCategory}.`)
              }}
            >
              Move to {suggestedCategory}
            </button>
          </div>
        )}

        {/* Attribute pills */}
        {showAttrs && (
          <div className="item-detail__pills">
            <span className="pill pill--primary">{item.category}</span>
            {item.fit && <span className="pill">{item.fit}</span>}
            {pattern && <span className="pill">{pattern}</span>}
            {item.season?.map((s) => (
              <span key={s} className="pill pill--season">{s}</span>
            ))}
            {styleTags.slice(0, 3).map((t) => (
              <span key={t} className="pill">{t}</span>
            ))}
          </div>
        )}

        {/* Color swatches */}
        {showAttrs && (primaryColor || secondaryColor) && (
          <div>
            <p className="item-detail__section-title">Colours</p>
            <div className="color-swatches">
              {primaryColor && (
                <div className="color-swatch" style={{ background: primaryColor }} title={item.color_primary} />
              )}
              {secondaryColor && (
                <div className="color-swatch" style={{ background: secondaryColor }} title={item.color_secondary} />
              )}
              <span style={{ fontSize: 13, color: 'var(--muted)', marginLeft: 4 }}>
                {[item.color_primary, item.color_secondary].filter(Boolean).join(' + ')}
              </span>
            </div>
          </div>
        )}

        {/* Styling note */}
        {item.reasoning_summary && !hideFallbackSummary && (
          <div className="detail-notes">
            <strong>Styling note</strong>
            <p>{item.reasoning_summary}</p>
          </div>
        )}

        {/* AI notes */}
        {aiNotes.length > 0 && (
          <div className="detail-notes">
            <strong>Notes</strong>
            <ul className="detail-notes-list">
              {aiNotes.map((line, i) => <li key={i}>{line}</li>)}
            </ul>
          </div>
        )}

        {/* Edit form */}
        {isEditing && (
          <div className="edit-form">
            <p style={{ margin: '0 0 4px', fontWeight: 700, fontSize: 14 }}>Edit details</p>
            <input value={editedType} onChange={(e) => setEditedType(e.target.value)} placeholder="Type (e.g. Blazer)" />
            <select value={editedCategory} onChange={(e) => setEditedCategory(e.target.value as typeof editedCategory)}>
              <option value="Tops">Tops</option>
              <option value="Bottoms">Bottoms</option>
              <option value="Outerwear">Outerwear</option>
              <option value="Shoes">Shoes</option>
              <option value="Accessories">Accessories</option>
            </select>
            <input value={editedColor} onChange={(e) => setEditedColor(e.target.value)} placeholder="Primary colour" />
            <input value={editedMaterial} onChange={(e) => setEditedMaterial(e.target.value)} placeholder="Material" />
            <input value={editedPattern} onChange={(e) => setEditedPattern(e.target.value)} placeholder="Pattern" />
            <select value={editedFit} onChange={(e) => setEditedFit(e.target.value as '' | FitLabel)}>
              <option value="">Fit (unspecified)</option>
              {FIT_LABELS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
            <div className="actions">
              <button
                className="btn"
                type="button"
                style={{ flex: 1 }}
                onClick={() => {
                  updateItem(item.id, {
                    item_type: editedType.trim() || item.item_type,
                    category: editedCategory,
                    color_primary: editedColor.trim() || item.color_primary,
                    material: editedMaterial.trim() || item.material,
                    pattern: editedPattern.trim() || undefined,
                    fit: editedFit === '' ? undefined : editedFit,
                    style_tags: Array.from(new Set([...(item.style_tags ?? []), 'UserCorrected'])),
                  })
                  toast.success('Saved.')
                  setIsEditing(false)
                }}
              >
                Save
              </button>
              <button className="btn secondary" type="button" onClick={() => setIsEditing(false)}>Cancel</button>
            </div>
          </div>
        )}

        {/* CTAs */}
        <div className="item-detail__cta-stack">
          {needsEmbedding && (
            <button className="btn btn-full" type="button" disabled={embedBusy} onClick={() => void runFinishEmbedding()}>
              {embedBusy ? 'Embedding…' : 'Finish embedding'}
            </button>
          )}

          <Link className="btn btn-full" to={`/finish-my-fit/${item.id}`} style={{ textAlign: 'center' }}>
            Build a look from this
          </Link>

          {item.ai_processed && (
            <Link className="btn secondary btn-full" to={`/outfit-suggestions/${item.id}`} style={{ textAlign: 'center' }}>
              Full outfit ideas
            </Link>
          )}

          {item.ai_processed && (
            <button
              className="btn secondary btn-full"
              type="button"
              onClick={() => void refreshPostPipelineSuggestions(item.id)}
            >
              Refresh suggestions
            </button>
          )}

          <button
            className="btn secondary btn-full"
            type="button"
            disabled={advancedRunning || !showAttrs}
            onClick={() => void runAdvancedAnalysis()}
          >
            <Zap size={14} />
            {advancedRunning ? 'Analysing…' : 'Advanced analysis'}
          </button>
        </div>

        {/* Suggestions */}
        {itemSuggestions && itemSuggestions.suggestions.length > 0 && (
          <div>
            <p className="item-detail__section-title">Suggested pairings</p>
            <div className="detail-notes">
              <p style={{ margin: '0 0 8px' }}>{itemSuggestions.summary}</p>
              <ul className="detail-notes-list">
                {itemSuggestions.suggestions.slice(0, 3).map((s) => (
                  <li key={s.item_id}>{Math.round(s.score * 100)}% · {s.explanation}</li>
                ))}
              </ul>
              <Link
                className="btn secondary"
                to={`/finish-my-fit/${item.id}`}
                state={{ preselectedIds: itemSuggestions.suggestions.map((s) => s.item_id) }}
                style={{ marginTop: 10, display: 'inline-flex' }}
              >
                <Sparkles size={14} /> Open in Finish My Fit
              </Link>
            </div>
          </div>
        )}

        {/* Advanced AI debug output */}
        {showAttrs && hasAdvancedOutput && (
          <div className="detail-notes" style={{ fontSize: 12 }}>
            <strong>AI analysis</strong>
            <ul className="detail-notes-list">
              {provider && <li>Provider: {provider}</li>}
              {model && <li>Model: {model}</li>}
              {typeof confidenceOverall === 'number' && <li>Confidence: {confidenceOverall.toFixed(2)}</li>}
              {designTags.length > 0 && <li>Tags: {designTags.join(', ')}</li>}
              {styleNotes && <li>Note: {styleNotes}</li>}
            </ul>
          </div>
        )}

        {advancedError && (
          <div className="detail-notes" style={{ borderColor: 'var(--destructive)' }}>
            <strong>Analysis error</strong>
            <p>{advancedError}</p>
          </div>
        )}

        {/* Occasions */}
        {occasions.length > 0 && showAttrs && (
          <div className="item-detail__pills">
            {occasions.map((o) => <span key={o} className="pill">{o}</span>)}
          </div>
        )}

        {/* Delete */}
        <button
          className="danger-link"
          type="button"
          onClick={() => {
            if (!window.confirm('Remove this item from your wardrobe?')) return
            deleteItem(item.id)
            toast.success('Item removed.')
            navigate('/')
          }}
        >
          <Trash2 size={14} />
          Remove item
        </button>
      </div>
    </div>
  )
}
