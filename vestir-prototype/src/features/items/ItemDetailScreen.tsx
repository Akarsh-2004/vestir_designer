import { ArrowLeft, Pencil, Trash2 } from 'lucide-react'
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
  const hasAdvancedOutput =
    designTags.length > 0 ||
    Boolean(styleNotes) ||
    providerLower.includes('gemma-mlx')
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

  if (!item) return <div className="card">This item isn’t here anymore.</div>

  const showAttrs = item.processing_status === 'done'
  const needsEmbedding = Boolean(item.attribute_review_pending)
  const processingDone = item.ai_processed && item.processing_stage === 'complete'
  const statusLine = processingDone
    ? 'Ready to style'
    : needsEmbedding
      ? 'Needs quick review — confirm type/category, then finish embedding'
      : [item.processing_stage ?? 'queued', item.pipeline_substage ? ` · ${item.pipeline_substage}` : '']
          .join('')
          .trim() || 'Working'

  async function runAdvancedAnalysis() {
    if (!item || advancedRunning) return
    setAdvancedRunning(true)
    const toastId = toast.loading('Running advanced analysis...')
    try {
      const sourceUrl = item.cutout_url ?? item.image_url
      const preprocessed = await localPreprocessAdapter(sourceUrl)
      const inferred = await advancedGeminiInferenceAdapter(preprocessed.processedImageUrl)
      const normalized = await normalizeAttributesAdapter(inferred)
      const merged = { ...item, ...normalized }
      const reasoning = await reasoningAdapter(merged)
      const currentRaw = parseLooseJson(item.raw_attributes)
      const nextRaw = {
        ...currentRaw,
        ...(inferred as unknown as Record<string, unknown>),
        metadata: (inferred as unknown as { metadata?: Record<string, unknown> })?.metadata ?? currentRaw.metadata,
        advanced_error: undefined,
      }
      updateItem(item.id, {
        ...normalized,
        reasoning_summary: reasoning.summary,
        raw_attributes: JSON.stringify(nextRaw, null, 2),
        processing_error: undefined,
      })
      const providerName = (
        (inferred as unknown as { metadata?: { provider?: string } })?.metadata?.provider ?? ''
      ).toLowerCase()
      if (providerName.includes('fallback')) {
        toast.warning('Advanced run completed using fallback mode.', { id: toastId })
      } else {
        toast.success('Advanced analysis updated (Gemini).', { id: toastId })
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Advanced analysis failed'
      const currentRaw = parseLooseJson(item.raw_attributes)
      updateItem(item.id, {
        raw_attributes: JSON.stringify({ ...currentRaw, advanced_error: message }, null, 2),
      })
      toast.error(message, { id: toastId })
    } finally {
      setAdvancedRunning(false)
    }
  }

  async function runFinishEmbedding() {
    if (!item || embedBusy) return
    setEmbedBusy(true)
    const toastId = toast.loading('Indexing item for search…')
    try {
      await completeAttributeReview(item.id)
      toast.success('Embedding complete. Closet search is updated.', { id: toastId })
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Embedding failed', { id: toastId })
    } finally {
      setEmbedBusy(false)
    }
  }

  return (
    <section className="card">
      {needsEmbedding ? (
        <div className="detail-notes" style={{ marginBottom: 12, borderColor: 'var(--warning, #c9a227)' }}>
          <strong>Check category &amp; type</strong>
          <p className="muted" style={{ margin: '6px 0 0' }}>
            Vision models disagreed on this crop. Fix any mistakes with <em>Edit prediction</em>, then run{' '}
            <strong>Finish embedding</strong> so similarity search stays accurate.
          </p>
        </div>
      ) : null}
      <div className="detail-head">
        <button className="icon-btn" type="button" onClick={() => navigate(-1)} aria-label="Back">
          <ArrowLeft size={16} />
        </button>
        <button
          className="icon-btn"
          type="button"
          onClick={() => setIsEditing((value) => !value)}
          title={isEditing ? 'Close editor' : 'Edit prediction'}
        >
          <Pencil size={16} />
        </button>
      </div>
      <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} />
      {isEditing ? (
        <div className="detail-notes" style={{ marginTop: 10 }}>
          <strong>Correct prediction</strong>
          <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
            <input value={editedType} onChange={(e) => setEditedType(e.target.value)} placeholder="Type (e.g. Vest dress)" />
            <select value={editedCategory} onChange={(e) => setEditedCategory(e.target.value as typeof editedCategory)}>
              <option value="Tops">Tops</option>
              <option value="Bottoms">Bottoms</option>
              <option value="Outerwear">Outerwear</option>
              <option value="Shoes">Shoes</option>
              <option value="Accessories">Accessories</option>
            </select>
            <input value={editedColor} onChange={(e) => setEditedColor(e.target.value)} placeholder="Primary color" />
            <input value={editedMaterial} onChange={(e) => setEditedMaterial(e.target.value)} placeholder="Material" />
            <input value={editedPattern} onChange={(e) => setEditedPattern(e.target.value)} placeholder="Pattern" />
            <select value={editedFit} onChange={(e) => setEditedFit(e.target.value as '' | FitLabel)}>
              <option value="">Fit (unspecified)</option>
              {FIT_LABELS.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="btn secondary"
                type="button"
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
                  toast.success('Saved correction.')
                  setIsEditing(false)
                }}
              >
                Save correction
              </button>
              <button className="btn secondary" type="button" onClick={() => setIsEditing(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
      <div className="detail-meta">
        <p>
          <strong>Type:</strong> {showAttrs ? item.item_type : 'Analyzing...'}
        </p>
        <p>
          <strong>Category:</strong> {showAttrs ? item.category : 'Analyzing...'}
        </p>
        <p>
          <strong>Primary Colour:</strong> {showAttrs ? item.color_primary : 'Analyzing...'}
        </p>
        {pattern ? (
          <p>
            <strong>Pattern:</strong> {pattern}
          </p>
        ) : null}
        <p>
          <strong>Material:</strong> {showAttrs ? item.material : 'Analyzing...'}
        </p>
        {item.fit ? (
          <p>
            <strong>Fit:</strong> {item.fit}
          </p>
        ) : null}
        {showAttrs && styleTags.length ? (
          <p>
            <strong>Style tags:</strong> {styleTags.join(', ')}
          </p>
        ) : null}
        {showAttrs && occasions.length ? (
          <p>
            <strong>Occasions:</strong> {occasions.join(', ')}
          </p>
        ) : null}
        {designTags.length ? (
          <p>
            <strong>Advanced tags:</strong> {designTags.join(', ')}
          </p>
        ) : null}
        {styleNotes ? (
          <p>
            <strong>Advanced note:</strong> {styleNotes}
          </p>
        ) : null}
        <p>
          <strong>Status:</strong> {statusLine}
        </p>
        {aiNotes.length > 0 ? (
          <div className="detail-notes">
            <strong>Notes</strong>
            <ul className="detail-notes-list">
              {aiNotes.map((line, i) => (
                <li key={`${i}-${line}`}>{line}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {item.reasoning_summary && !hideFallbackSummary ? (
          <p>
            <strong>Styling note:</strong> {item.reasoning_summary}
          </p>
        ) : null}
        {!showAttrs ? (
          <div className="detail-notes">
            <strong>AI analysis</strong>
            <ul className="detail-notes-list">
              <li>Running model analysis...</li>
            </ul>
          </div>
        ) : hasAdvancedOutput ? (
          <div className="detail-notes">
            <strong>AI analysis</strong>
            <ul className="detail-notes-list">
              <li>Provider: {provider ?? 'gemma-mlx'}</li>
              <li>Model: {model ?? 'not reported'}</li>
              {typeof confidenceOverall === 'number' ? <li>Confidence: {confidenceOverall.toFixed(2)}</li> : null}
              <li>
                Tags: {designTags.length ? designTags.join(', ') : 'none'}
              </li>
            </ul>
          </div>
        ) : advancedError ? (
          <div className="detail-notes">
            <strong>AI analysis</strong>
            <ul className="detail-notes-list">
              <li>{advancedError}</li>
              <li>Tip: first local Gemma run may take a while to load model weights.</li>
            </ul>
          </div>
        ) : null}
      </div>
      {needsEmbedding ? (
        <button className="btn" type="button" disabled={embedBusy} onClick={() => void runFinishEmbedding()}>
          {embedBusy ? 'Embedding…' : 'Finish embedding'}
        </button>
      ) : null}
      <Link className="btn" to={`/finish-my-fit/${item.id}`}>
        Build a look from this
      </Link>
      {item.ai_processed ? (
        <Link className="btn" to={`/outfit-suggestions/${item.id}`}>
          Full outfit ideas
        </Link>
      ) : null}
      {item.ai_processed ? (
        <button
          className="btn secondary"
          type="button"
          onClick={() => void refreshPostPipelineSuggestions(item.id)}
        >
          Refresh AI suggestions
        </button>
      ) : null}
      {itemSuggestions ? (
        <div className="detail-notes">
          <strong>Suggested next items</strong>
          <p className="muted" style={{ margin: '4px 0 8px' }}>{itemSuggestions.summary}</p>
          <ul className="detail-notes-list">
            {itemSuggestions.suggestions.slice(0, 3).map((s) => (
              <li key={s.item_id}>
                {Math.round(s.score * 100)}% match · {s.explanation}
              </li>
            ))}
            {itemSuggestions.suggestions.length === 0 ? (
              <li>{itemSuggestions.warning ?? 'No suggestions available yet.'}</li>
            ) : null}
          </ul>
          {itemSuggestions.suggestions.length > 0 ? (
            <Link
              className="btn secondary"
              to={`/finish-my-fit/${item.id}`}
              state={{ preselectedIds: itemSuggestions.suggestions.map((s) => s.item_id) }}
            >
              Open Finish My Fit with suggestions
            </Link>
          ) : null}
        </div>
      ) : null}
      <button
        className="btn secondary"
        type="button"
        disabled={advancedRunning || !showAttrs}
        onClick={runAdvancedAnalysis}
        title={!showAttrs ? 'Wait for base prediction to finish first' : undefined}
      >
        {advancedRunning ? 'Running advanced analysis...' : 'Advanced analysis'}
      </button>
      <button
        className="danger-link"
        type="button"
        onClick={() => {
          const ok = window.confirm('Delete this item from your wardrobe? You can always add it again later.')
          if (!ok) return
          deleteItem(item.id)
          toast.success('Item removed.')
          navigate('/')
        }}
      >
        <Trash2 size={14} />
        Delete item
      </button>
    </section>
  )
}
