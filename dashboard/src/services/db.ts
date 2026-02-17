/**
 * Neon Serverless PostgreSQL — direct browser-to-DB connection.
 *
 * Uses Neon's HTTP-based driver (@neondatabase/serverless) which works
 * from the browser via fetch(). No WebSocket/TCP needed.
 *
 * The DATABASE_URL is baked in at build time via VITE_DATABASE_URL,
 * or falls back to the hardcoded Neon connection string.
 */
import { neon } from '@neondatabase/serverless'

const DATABASE_URL =
  import.meta.env.VITE_DATABASE_URL ||
  'postgresql://neondb_owner:npg_RwvSkJe24uIY@ep-icy-breeze-ag5us00v-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require'

/** Neon HTTP query function — works in the browser */
const sql = neon(DATABASE_URL)

// Diagnostic: test DB connection on module load (check browser console)
sql`SELECT 1 as ok`
  .then((r: any) => console.info('[db.ts] ✅ Neon DB connected:', r))
  .catch((e: any) => console.error('[db.ts] ❌ Neon DB failed:', e?.message || e, '\nDATABASE_URL starts with:', DATABASE_URL?.slice(0, 30)))

export default sql
