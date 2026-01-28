import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Vite configuration following best practices from documentation
// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // Path aliases for cleaner imports
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  
  server: {
    port: 3000,
    open: true,
    // Enable HMR overlay for development errors
    hmr: {
      overlay: true
    }
  },
  
  build: {
    // Optimize for production
    target: 'esnext',
    // Warn about large chunks
    chunkSizeWarningLimit: 1000,
    // Enable source maps for debugging
    sourcemap: false,
    // Use default Oxc minifier (fastest)
    minify: 'oxc'
  }
})
