import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // Vite writes dependency prebundle artifacts under `cacheDir`.
  // We keep it out of `node_modules/.vite` because parts may have been created as `root`,
  // causing EACCES errors for the normal dev user.
  cacheDir: './.vite-cache',
  plugins: [react()],
  server: {
    watch: {
      ignored: ['**/.venv/**'],
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
        // FLUX try-off can run 30–120+ minutes on CPU; avoid dev proxy cutting the connection.
        timeout: 7_200_000,
        proxyTimeout: 7_200_000,
      },
      '/storage': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
        timeout: 7_200_000,
        proxyTimeout: 7_200_000,
      },
    },
  },
})
