/**
 * Neon Serverless PostgreSQL — direct browser-to-DB via raw fetch().
 *
 * Replaces the @neondatabase/serverless driver with a minimal fetch-based
 * implementation that calls the same Neon HTTP SQL API directly.
 * This avoids any driver/bundler incompatibility in the browser.
 *
 * Exports a tagged-template function `sql` that works the same way:
 *   const rows = await sql`SELECT * FROM users WHERE id = ${id}`
 */

const DATABASE_URL =
  import.meta.env.VITE_DATABASE_URL ||
  'postgresql://neondb_owner:npg_RwvSkJe24uIY@ep-icy-breeze-ag5us00v-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require'

/** Extract the hostname from the connection string for the HTTP endpoint */
function getNeonEndpoint(connStr: string): string {
  const match = connStr.match(/@([^/]+)/)
  if (!match) throw new Error('Invalid DATABASE_URL: cannot extract hostname')
  return `https://${match[1]}/sql`
}

const NEON_ENDPOINT = getNeonEndpoint(DATABASE_URL)

/**
 * Send a parameterized query to Neon's HTTP SQL API.
 * Returns the rows as an array of objects (column names as keys).
 */
async function neonQuery(query: string, params: unknown[] = []): Promise<any[]> {
  const res = await fetch(NEON_ENDPOINT, {
    method: 'POST',
    headers: { 'Neon-Connection-String': DATABASE_URL },
    body: JSON.stringify({ query, params }),
  })

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`Neon HTTP ${res.status}: ${text}`)
  }

  const json = await res.json()
  return json.rows ?? []
}

/**
 * Tagged-template SQL function — drop-in replacement for neon().
 *
 * Usage:  const rows = await sql`SELECT * FROM users WHERE id = ${id}`
 *
 * Template interpolations become $1, $2, … parameters for safe queries.
 */
function sql(strings: TemplateStringsArray, ...values: unknown[]): Promise<any[]> {
  // Build the query with $1, $2, $3... placeholders
  let query = strings[0]
  for (let i = 0; i < values.length; i++) {
    query += `$${i + 1}` + strings[i + 1]
  }
  return neonQuery(query, values)
}

// Diagnostic: test DB connection on module load (check browser console)
sql`SELECT 1 as ok`
  .then((r: any) => console.info('[db.ts] Neon DB connected via raw fetch:', r))
  .catch((e: any) => console.error('[db.ts] Neon DB failed:', e?.message || e))

export default sql
