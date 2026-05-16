import { useEffect, useId, useState } from 'react'
import { Camera, Image, Loader } from 'lucide-react'
import { toast } from 'sonner'
import { useBodyScrollLock } from '../../hooks/useBodyScrollLock'
import { useWardrobeStore } from '../../store/wardrobeStore'

interface AddItemsSheetProps {
  open: boolean
  onClose: () => void
}

export function AddItemsSheet({ open, onClose }: AddItemsSheetProps) {
  const { detectItemsFromFile, addPendingItemsFromFiles, runHybridAiPipeline } = useWardrobeStore()
  const [detecting, setDetecting] = useState(false)
  const titleId = useId()
  useBodyScrollLock(open)
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  if (!open) return null

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.currentTarget.files
    if (!files || files.length === 0) return
    if (files.length === 1) {
      // Single file — use smart multi-item detect flow.
      try {
        toast.info('Scanning your photo…')
        setDetecting(true)
        onClose()
        await detectItemsFromFile(files[0])
      } catch (error) {
        // Detection unavailable — fall back to direct pipeline.
        try {
          const reason = error instanceof Error ? error.message : 'detect pipeline unavailable'
          toast.info(`Detect/tryoff fallback: ${reason}`)
          await addPendingItemsFromFiles(files)
          await runHybridAiPipeline()
        } catch (pipelineError) {
          toast.error(pipelineError instanceof Error ? pipelineError.message : 'Something went wrong while processing your photo.')
        }
      } finally {
        setDetecting(false)
      }
    } else {
      // Multiple files — skip detect UX, process each directly.
      try {
        toast.info(`Adding ${files.length} photos…`)
        onClose()
        await addPendingItemsFromFiles(files)
        await runHybridAiPipeline()
      } catch (error) {
        toast.error(error instanceof Error ? error.message : 'Something went wrong. Please try again.')
      }
    }
  }

  return (
    <div className="sheet-scrim" onClick={onClose} role="presentation">
      <div
        className="sheet"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <div className="sheet-handle" aria-hidden />
        <h3 className="sheet-title" id={titleId}>
          Add items
        </h3>
        <p className="muted sheet-subtitle">
          One photo to detect every garment, or batch import a whole album.
        </p>
        <label className="sheet-option" style={{ opacity: detecting ? 0.6 : 1, pointerEvents: detecting ? 'none' : undefined }}>
          <span className="sheet-icon">
            <Camera size={20} />
          </span>
          <span>
            <strong>Take photo</strong>
            <small>Snap a garment with your camera</small>
          </span>
          <input
            hidden
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileChange}
          />
        </label>
        <label className="sheet-option" style={{ opacity: detecting ? 0.6 : 1, pointerEvents: detecting ? 'none' : undefined }}>
          <span className="sheet-icon">
            {detecting ? <Loader size={20} className="spin" /> : <Image size={20} />}
          </span>
          <span>
            <strong>Photo library</strong>
            <small>{detecting ? 'Detecting items…' : 'Single photo detects every garment'}</small>
          </span>
          <input
            hidden
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileChange}
          />
        </label>

        <button className="btn secondary" type="button" onClick={onClose} style={{ width: '100%', justifyContent: 'center' }}>
          Not now
        </button>
      </div>
    </div>
  )
}
