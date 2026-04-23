import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/predict': 'http://sentiment-ui:8000',
      '/chat':    'http://sentiment-ui:8000',
    }
  },
  preview: {
    host: '0.0.0.0',
    port: 3000,
  }
})