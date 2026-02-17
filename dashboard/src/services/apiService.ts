/**
 * Smart API Service — tries the server API first, falls back to direct DB.
 *
 * Every public function attempts a fetch() to the API server with a short
 * timeout. If the server is unreachable (network error, timeout, or HTTP 5xx),
 * it transparently falls back to querying the Neon database directly from the
 * browser via the @neondatabase/serverless HTTP driver.
 *
 * This makes the dashboard fully functional even when the Python server
 * on the other Mac is offline.
 */
import API_CONFIG from '../config'
import {
  dbLogin,
  dbGetKPI,
  dbGetData,
  dbGetClient360,
  dbGetPlaybooks,
  dbGetAdvisors,
  dbGetAdvisorWorkload,
  dbGetEvents,
  dbGetEventDetail,
  dbGetConcepts,
  dbGetReportSummary,
} from './dbService'

// ─── Configuration ────────────────────────────────────────────────
/** Max time for API before abort (background cleanup — race handles UX) */
const API_TIMEOUT = 2000
/** Hard global timeout — page never hangs longer than this */
const GLOBAL_TIMEOUT = 8000

/** In-memory cache — survives tab switches (component unmount/remount) */
const cache = new Map<string, { data: any; ts: number; promise?: Promise<any> }>()
const CACHE_TTL = 120_000 // 2 minutes — data stays fresh across tab switches
const CLIENT_CACHE_TTL = 300_000 // 5 min for individual client profiles

function getCached<T>(key: string, ttl = CACHE_TTL): T | null {
  const entry = cache.get(key)
  if (!entry) return null
  if (Date.now() - entry.ts > ttl) { cache.delete(key); return null }
  return entry.data as T
}

function setCache(key: string, data: any) {
  cache.set(key, { data, ts: Date.now() })
}

/** Invalidate cache entries matching a prefix (used after write ops) */
export function invalidateCache(prefix?: string) {
  if (!prefix) { cache.clear(); return }
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) cache.delete(key)
  }
}

/** Track whether the API server is known to be down or slow */
let serverDown = false
let serverDownSince = 0
const SERVER_RETRY_INTERVAL = 60_000 // retry server every 60s after failure

function isServerKnownDown(): boolean {
  if (!serverDown) return false
  if (Date.now() - serverDownSince > SERVER_RETRY_INTERVAL) {
    serverDown = false // time to retry
    return false
  }
  return true
}

function markServerDown() {
  serverDown = true
  serverDownSince = Date.now()
}

function markServerUp() {
  serverDown = false
}

/** Source indicator for debugging */
export type DataSource = 'api' | 'db'
let lastSource: DataSource = 'api'
export function getLastDataSource(): DataSource { return lastSource }

// ─── Core fetch with timeout ──────────────────────────────────────
async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), API_TIMEOUT)

  try {
    const res = await fetch(url, { ...init, signal: controller.signal })
    clearTimeout(timeout)
    return res
  } catch (err) {
    clearTimeout(timeout)
    throw err
  }
}

/**
 * Race pattern: fire API and DB in parallel, use whichever resolves first.
 *
 * This means the dashboard is always as fast as the *fastest* source.
 * If the LAN server is quick (< 200ms), it wins. If Neon DB is faster,
 * it wins. The slow source's in-flight request just finishes in the
 * background without affecting UX.
 */
async function withFallback<T>(
  apiFn: () => Promise<T>,
  dbFn: () => Promise<T>,
): Promise<{ data: T; source: DataSource }> {
  // If server is known down/slow, skip straight to DB (no wasted request)
  if (isServerKnownDown()) {
    console.info('[apiService] Server known down, using DB directly')
    try {
      const data = await dbFn()
      lastSource = 'db'
      return { data, source: 'db' }
    } catch (dbErr) {
      console.error('[apiService] DB-only path failed:', dbErr)
      throw dbErr
    }
  }

  // Race: start both simultaneously, first success wins
  // Hard timeout ensures the page NEVER hangs forever
  return new Promise<{ data: T; source: DataSource }>((resolve, reject) => {
    let done = false
    let failures = 0
    const t0 = Date.now()

    // Hard safety timeout
    const hardTimeout = setTimeout(() => {
      if (!done) {
        done = true
        console.error(`[apiService] HARD TIMEOUT after ${GLOBAL_TIMEOUT}ms — both sources unresponsive`)
        reject(new Error(`Data fetch timed out after ${GLOBAL_TIMEOUT / 1000}s`))
      }
    }, GLOBAL_TIMEOUT)

    const finish = (data: T, source: DataSource) => {
      if (!done) {
        done = true
        clearTimeout(hardTimeout)
        lastSource = source
        console.info(`[apiService] ${source} responded in ${Date.now() - t0}ms`)
        resolve({ data, source })
      }
    }

    const fail = (source: string, err: any) => {
      console.warn(`[apiService] ${source} failed (${Date.now() - t0}ms):`, err?.message || err)
      if (source === 'api') markServerDown()
      failures++
      if (failures === 2 && !done) {
        done = true
        clearTimeout(hardTimeout)
        reject(new Error('Both API and DB queries failed'))
      }
    }

    apiFn()
      .then(data => { markServerUp(); finish(data, 'api') })
      .catch(err => fail('api', err))

    dbFn()
      .then(data => finish(data, 'db'))
      .catch(err => fail('db', err))
  })
}

/**
 * Cached + deduped wrapper around withFallback.
 * - Returns cached data instantly on tab switches (no network hit)
 * - Deduplicates concurrent requests for the same key
 * - After TTL expires, fetches fresh data
 */
async function cachedFetch<T>(
  key: string,
  apiFn: () => Promise<T>,
  dbFn: () => Promise<T>,
  ttl = CACHE_TTL,
): Promise<{ data: T; source: DataSource }> {
  // 1. Return cached data instantly
  const cached = getCached<{ data: T; source: DataSource }>(key, ttl)
  if (cached) return cached

  // 2. Deduplicate in-flight requests
  const inflight = cache.get(key)
  if (inflight?.promise) return inflight.promise

  // 3. Fire the real fetch, store the promise for dedup
  const promise = withFallback(apiFn, dbFn).then(result => {
    setCache(key, result)
    return result
  })
  cache.set(key, { data: null, ts: Date.now(), promise })

  try {
    return await promise
  } catch (err) {
    cache.delete(key) // don't cache errors
    throw err
  }
}

// ─── Public API ───────────────────────────────────────────────────

/** Login — tries API server, falls back to direct DB user lookup */
export async function login(username: string, password: string) {
  const { data } = await withFallback(
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Login failed' }))
        throw new Error(body.detail || `Login failed (${res.status})`)
      }
      return res.json()
    },
    () => dbLogin(username, password),
  )
  return data
}

/** KPI dashboard data */
export async function getKPI() {
  return cachedFetch(
    'kpi',
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/kpi`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    dbGetKPI,
  )
}

/** Main data (clients, segments, scatter3d, concepts, heatmap) */
export async function getData() {
  return cachedFetch(
    'data',
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/data`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    dbGetData,
  )
}

/** Client 360° detail */
export async function getClient360(clientId: string) {
  return cachedFetch(
    `client360:${clientId}`,
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/clients/${clientId}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    () => dbGetClient360(clientId),
    CLIENT_CACHE_TTL,
  )
}

/** Playbooks list */
export async function getPlaybooks(category?: string) {
  return cachedFetch(
    `playbooks:${category || 'all'}`,
    async () => {
      const params = category ? `?category=${category}` : ''
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/playbooks${params}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    () => dbGetPlaybooks(category),
  )
}

/** Advisors list */
export async function getAdvisors() {
  return cachedFetch(
    'advisors',
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/advisors`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    dbGetAdvisors,
  )
}

/** Advisor workload */
export async function getAdvisorWorkload() {
  return cachedFetch(
    'advisor-workload',
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/advisors/workload`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    dbGetAdvisorWorkload,
  )
}

/** Calendar events */
export async function getEvents(month?: number, year?: number, status?: string) {
  return cachedFetch(
    `events:${month}-${year}-${status || ''}`,
    async () => {
      const params = new URLSearchParams()
      if (month !== undefined) params.set('month', String(month))
      if (year !== undefined) params.set('year', String(year))
      if (status) params.set('status', status)
      const qs = params.toString() ? `?${params.toString()}` : ''
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/events${qs}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    () => dbGetEvents(month, year, status),
  )
}

/** Single event detail */
export async function getEventDetail(eventId: number) {
  return cachedFetch(
    `event-detail:${eventId}`,
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/events/${eventId}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    () => dbGetEventDetail(eventId),
  )
}

/** Concepts for calendar event creation */
export async function getConcepts() {
  return cachedFetch(
    'concepts',
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/concepts`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    dbGetConcepts,
  )
}

/** Report summary */
export async function getReportSummary() {
  return cachedFetch(
    'report-summary',
    async () => {
      const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/report/summary`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    dbGetReportSummary,
  )
}

// ─── Write operations (API-only, no DB fallback) ──────────────────
// These require the server to be online. They fail gracefully with
// a clear error message instead of spinning forever.

export async function createEvent(eventData: any) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(eventData),
  })
  if (!res.ok) throw new Error(`Failed to create event (${res.status})`)
  invalidateCache('events')
  invalidateCache('event-detail')
  invalidateCache('kpi')
  return res.json()
}

export async function updateEvent(eventId: number, eventData: any) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/events/${eventId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(eventData),
  })
  if (!res.ok) throw new Error(`Failed to update event (${res.status})`)
  return res.json()
}

export async function updateEventStatus(eventId: number, status: string) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/events/${eventId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  })
  if (!res.ok) throw new Error(`Failed to update event status (${res.status})`)
  invalidateCache('events')
  invalidateCache(`event-detail:${eventId}`)
  return res.json()
}

export async function updateTarget(eventId: number, clientId: string, actionStatus: string) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/events/${eventId}/targets/${clientId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_status: actionStatus }),
  })
  if (!res.ok) throw new Error(`Failed to update target (${res.status})`)
  invalidateCache(`event-detail:${eventId}`)
  invalidateCache('events')
  return res.json()
}

export async function deleteEvent(eventId: number) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/calendar/events/${eventId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`Failed to delete event (${res.status})`)
  invalidateCache('events')
  invalidateCache(`event-detail:${eventId}`)
  invalidateCache('kpi')
  return res.json()
}

export async function activatePlaybook(playbookId: number, data: any) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/playbooks/${playbookId}/activate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error(`Failed to activate playbook (${res.status})`)
  invalidateCache('events')
  invalidateCache('kpi')
  return res.json()
}

export async function createPlaybook(data: any) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/playbooks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error(`Failed to create playbook (${res.status})`)
  invalidateCache('playbooks')
  return res.json()
}

export async function autoAssignAdvisors(data: any) {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/advisors/auto-assign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error(`Failed to auto-assign advisors (${res.status})`)
  invalidateCache('advisors')
  invalidateCache('advisor-workload')
  return res.json()
}

export async function exportCSV(type: string): Promise<Blob> {
  const res = await apiFetch(`${API_CONFIG.BASE_URL}/api/export/${type}`)
  if (!res.ok) throw new Error(`Failed to export ${type} (${res.status})`)
  return res.blob()
}
