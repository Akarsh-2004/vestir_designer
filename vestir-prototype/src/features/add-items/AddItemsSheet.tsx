import { useState } from 'react'
import { Camera, Image, Loader } from 'lucide-react'
import { toast } from 'sonner'
import { useWardrobeStore } from '../../store/wardrobeStore'

interface AddItemsSheetProps {
  open: boolean
  onClose: () => void
}

export function AddItemsSheet({ open, onClose }: AddItemsSheetProps) {
  const { detectItemsFromFile, addPendingItemsFromFiles, runHybridAiPipeline } = useWardrobeStore()
  const [detecting, setDetecting] = useState(false)
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
          toast.info("No worries—I'll process the photo directly.")
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
    <div className="sheet-scrim" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <div className="sheet-handle" />
        <h3>Add Items</h3>
        <button className="sheet-option" type="button" disabled>
          <span className="sheet-icon">
            <Camera size={20} />
          </span>
          <span>
            <strong>Camera</strong>
            <small>Coming soon</small>
          </span>
        </button>
        <label className="sheet-option" style={{ opacity: detecting ? 0.6 : 1, pointerEvents: detecting ? 'none' : undefined }}>
          <span className="sheet-icon">
            {detecting ? <Loader size={20} className="spin" /> : <Image size={20} />}
          </span>
          <span>
            <strong>Photo Library</strong>
            <small>{detecting ? 'Detecting items…' : 'Single photo detects all garments in one shot'}</small>
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
