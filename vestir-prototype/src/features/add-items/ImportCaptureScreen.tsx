import { useEffect, useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { useWardrobeStore } from '../../store/wardrobeStore'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

export function ImportCaptureScreen() {
  const { token } = useParams()
  const navigate = useNavigate()
  const detectItemsFromImageUrl = useWardrobeStore((s) => s.detectItemsFromImageUrl)
  const startedRef = useRef(false)

  useEffect(() => {
    if (!token || startedRef.current) return
    startedRef.current = true
    const run = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/extension/capture/${token}`)
        const data = await response.json()
        if (!response.ok) throw new Error(data.error ?? 'Capture lookup failed')
        toast.info('Importing item from extension...')
        await detectItemsFromImageUrl(String(data.imageUrl))
        navigate('/', { replace: true })
      } catch (error) {
        toast.error(error instanceof Error ? error.message : 'Capture import failed')
        navigate('/', { replace: true })
      }
    }
    void run()
  }, [token, detectItemsFromImageUrl, navigate])

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Importing capture</h3>
      <p className="muted" style={{ marginBottom: 0 }}>
        We are pulling your product from the extension and starting detection.
      </p>
    </div>
  )
}
