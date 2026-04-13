import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

function extractJsonObject(text) {
  const trimmed = String(text ?? '').trim()
  if (!trimmed) throw new Error('Gemma MLX returned empty response')
  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i)
  if (fencedMatch?.[1]) return fencedMatch[1].trim()
  const firstBrace = trimmed.indexOf('{')
  const lastBrace = trimmed.lastIndexOf('}')
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error(`Gemma MLX did not return JSON. Raw: ${trimmed.slice(0, 280)}`)
  }
  return trimmed.slice(firstBrace, lastBrace + 1)
}

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const workerPath = path.join(__dirname, 'gemma_mlx_worker.py')

let worker = null
let workerReady = null
let requestSeq = 0
const inflight = new Map()

function ensureWorker() {
  if (worker && workerReady) return workerReady
  const model = process.env.GEMMA_MLX_MODEL ?? 'mlx-community/gemma-3-4b-it-4bit'
  worker = spawn(process.env.GEMMA_MLX_COMMAND ?? 'python3', [workerPath, model], { stdio: ['pipe', 'pipe', 'pipe'] })

  workerReady = new Promise((resolve, reject) => {
    let outBuffer = ''
    const onStdout = (chunk) => {
      outBuffer += String(chunk)
      const lines = outBuffer.split('\n')
      outBuffer = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        let msg
        try {
          msg = JSON.parse(trimmed)
        } catch {
          continue
        }
        if (msg.type === 'ready') {
          if (msg.ok) resolve(msg)
          else reject(new Error(msg.error ?? 'Gemma worker failed to initialize'))
          continue
        }
        if (msg.type === 'result' && msg.id) {
          const pending = inflight.get(msg.id)
          if (!pending) continue
          inflight.delete(msg.id)
          if (!msg.ok) pending.reject(new Error(msg.error ?? 'Gemma worker request failed'))
          else pending.resolve(msg.text ?? '')
        }
      }
    }

    worker.stdout.on('data', onStdout)
    worker.stderr.on('data', () => {
      // Keep stderr streamed for debugging; request errors are returned via result frames.
    })
    worker.on('error', (error) => {
      reject(error)
    })
    worker.on('close', () => {
      for (const [, pending] of inflight) pending.reject(new Error('Gemma worker exited unexpectedly'))
      inflight.clear()
      worker = null
      workerReady = null
    })
  })
  return workerReady
}

function runWorkerRequest({ prompt, imageBase64, maxTokens, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const id = `req_${Date.now()}_${requestSeq += 1}`
    const timer = setTimeout(() => {
      inflight.delete(id)
      reject(new Error(`Gemma MLX timed out after ${timeoutMs}ms`))
    }, timeoutMs)
    inflight.set(id, {
      resolve: (text) => {
        clearTimeout(timer)
        resolve(text)
      },
      reject: (error) => {
        clearTimeout(timer)
        reject(error)
      },
    })
    worker.stdin.write(`${JSON.stringify({ id, prompt, image_base64: imageBase64, max_tokens: maxTokens })}\n`)
  })
}

export async function runGemmaMlxVision({ imageBuffer, prompt, timeoutMs = 40000 }) {
  try {
    const model = process.env.GEMMA_MLX_MODEL ?? 'mlx-community/gemma-3-4b-it-4bit'
    const maxTokens = Math.max(80, Number.parseInt(process.env.GEMMA_MLX_MAX_TOKENS ?? '220', 10) || 220)
    await ensureWorker()
    const stdout = await runWorkerRequest({
      prompt,
      imageBase64: imageBuffer.toString('base64'),
      maxTokens,
      timeoutMs,
    })
    const payload = extractJsonObject(stdout)
    return {
      ok: true,
      model,
      result: JSON.parse(payload),
      raw: stdout,
    }
  } catch (error) {
    return {
      ok: false,
      model: process.env.GEMMA_MLX_MODEL ?? 'mlx-community/gemma-3-4b-it-4bit',
      error: error instanceof Error ? error.message : 'Gemma MLX invocation failed',
    }
  }
}

