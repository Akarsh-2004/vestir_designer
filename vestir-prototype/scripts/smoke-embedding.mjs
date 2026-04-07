/**
 * Smoke test for the embedding sidecar (no Gemini required).
 * Start sidecar: cd embedding-sidecar && ./start.sh
 * Run: EMBEDDING_SIDECAR_URL=http://127.0.0.1:8010 node scripts/smoke-embedding.mjs
 */
const base = (process.env.EMBEDDING_SIDECAR_URL ?? 'http://127.0.0.1:8010').replace(/\/$/, '')

const health = await fetch(`${base}/health`)
const hj = await health.json()
console.log('health', hj)
if (!health.ok || !hj.ok) {
  console.error('Sidecar not healthy — start embedding-sidecar and ensure weights downloaded.')
  process.exit(1)
}

const embedRes = await fetch(`${base}/embed`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'navy cotton oxford shirt, casual' }),
})
const ej = await embedRes.json()
console.log('embed status', embedRes.status, 'dim', ej.dim ?? ej.vector?.length)
if (!embedRes.ok || !Array.isArray(ej.vector) || !ej.vector.length) {
  console.error(ej)
  process.exit(1)
}
console.log('ok', ej.model)
