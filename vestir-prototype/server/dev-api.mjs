import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import killPort from 'kill-port'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const apiPort = Number(process.env.API_PORT ?? 8787)

try {
  await killPort(apiPort, 'tcp')
  console.log(`Freed port ${apiPort} before API start`)
} catch {
  // No process was listening; safe to continue.
}

const child = spawn(process.execPath, [path.join(__dirname, 'index.mjs')], {
  stdio: 'inherit',
  env: process.env,
})

child.on('exit', (code) => {
  process.exit(code ?? 0)
})
