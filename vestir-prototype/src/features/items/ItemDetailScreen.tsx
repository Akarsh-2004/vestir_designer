import { ArrowLeft, Pencil, Trash2 } from 'lucide-react'
import { useMemo } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { ItemPhoto } from '../../components/ItemPhoto'
import { useWardrobeStore } from '../../store/wardrobeStore'

function parseRawAttributes(raw?: string) {
  if (!raw) return null
  try {
    return JSON.parse(raw) as {
      quality?: { warnings?: string[] }
      uncertainty?: { requires_user_confirmation?: boolean; uncertain_fields?: string[] }
    }
  } catch {
    return null
  }
}

export function ItemDetailScreen() {
  const navigate = useNavigate()
  const { id } = useParams()
  const item = useWardrobeStore((s) => s.items.find((it) => it.id === id && !it.deleted_at))
  const deleteItem = useWardrobeStore((s) => s.deleteItem)

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

  if (!item) return <div className="card">This item isn’t here anymore.</div>

  const processingDone = item.ai_processed && item.processing_stage === 'complete'
  const statusLine = processingDone
    ? 'Ready to style'
    : [item.processing_stage ?? 'queued', item.pipeline_substage ? ` · ${item.pipeline_substage}` : '']
        .join('')
        .trim() || 'Working'

  return (
    <section className="card">
      <div className="detail-head">
        <button className="icon-btn" type="button" onClick={() => navigate(-1)} aria-label="Back">
          <ArrowLeft size={16} />
        </button>
        <button className="icon-btn" type="button" disabled title="Editing coming soon" aria-disabled="true">
          <Pencil size={16} />
        </button>
      </div>
      <ItemPhoto itemId={item.id} imageUrl={item.image_url} alt={item.item_type} />
      <div className="detail-meta">
        <p>
          <strong>Type:</strong> {item.item_type}
        </p>
        <p>
          <strong>Primary Colour:</strong> {item.color_primary}
        </p>
        <p>
          <strong>Material:</strong> {item.material}
        </p>
        <p>
          <strong>Status:</strong> {statusLine}
          {!processingDone ? ` (${item.processing_progress ?? 0}%)` : null}
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
        {item.reasoning_summary ? (
          <p>
            <strong>Styling note:</strong> {item.reasoning_summary}
          </p>
        ) : null}
      </div>
      <Link className="btn" to={`/finish-my-fit/${item.id}`}>
        Build a look from this
      </Link>
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
