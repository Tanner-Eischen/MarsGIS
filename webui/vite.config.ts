import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  worker: {
    format: 'es',
  },
  resolve: {
    // Prefer source TypeScript files when both .ts/.tsx and generated .js exist.
    extensions: [".ts", ".tsx", ".mjs", ".js", ".jsx", ".json"],
  },
  server: {
    port: 4000,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
