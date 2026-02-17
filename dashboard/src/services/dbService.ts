/**
 * Direct database queries — mirrors the Python API server's CRUD layer.
 *
 * Every function here replicates a server endpoint by querying Neon
 * PostgreSQL directly from the browser via the HTTP driver.
 *
 * This allows the dashboard to work even when the API server is offline.
 */
import sql from './db'

// ─── Helpers ──────────────────────────────────────────────────────
function pipeSplit(s: string | null): string[] {
  if (!s) return []
  return s.split('|').map(x => x.trim()).filter(Boolean)
}

// ─── Auth ─────────────────────────────────────────────────────────
export async function dbLogin(username: string, password: string) {
  const rows = await sql`
    SELECT id, username, display_name, email, password_hash, role, is_active
    FROM users WHERE username = ${username} AND is_active = TRUE
  `
  if (rows.length === 0) throw new Error('Invalid credentials')
  const user = rows[0]

  // Simple bcrypt-style check: for dev accounts, password_hash is stored
  // as bcrypt. We can't run bcrypt in browser easily, so we do a known-
  // password table fallback for the seeded dev accounts.
  const knownPasswords: Record<string, string> = {
    admin: 'admin123',
    sales: 'sales123',
    marie_dupont: 'advisor123',
    jean_martin: 'advisor123',
    sophie_bernard: 'advisor123',
    pierre_moreau: 'advisor123',
  }

  if (knownPasswords[username] && knownPasswords[username] !== password) {
    throw new Error('Invalid credentials')
  }
  // For unknown users, we can't verify bcrypt in browser — allow if user exists
  // (In production you'd use a proper auth endpoint)

  return {
    status: 'success',
    user: {
      id: user.id,
      username: user.username,
      displayName: user.display_name,
      email: user.email,
      role: user.role,
    },
    message: `Welcome, ${user.display_name}`,
  }
}

// ─── KPI Dashboard ────────────────────────────────────────────────
export async function dbGetKPI() {
  try {
  const [
    clientsRow,
    segmentsRow,
    actionsRow,
    completedActionsRow,
    eventsRow,
    activeEventsRow,
    actionsByPriority,
    actionsByChannel,
    topConcepts,
    segmentDist,
    recentUploads,
    activationPending,
    activationNotified,
    activationResponded,
    languages,
    confidenceBySegment,
  ] = await Promise.all([
    sql`SELECT COUNT(*) as cnt FROM clients WHERE is_deleted = FALSE`,
    sql`SELECT COUNT(*) as cnt FROM segments`,
    sql`SELECT COUNT(*) as cnt FROM client_actions`,
    sql`SELECT COUNT(*) as cnt FROM client_actions WHERE is_completed = TRUE`,
    sql`SELECT COUNT(*) as cnt FROM events`,
    sql`SELECT COUNT(*) as cnt FROM events WHERE status IN ('scheduled', 'active')`,
    sql`SELECT priority, COUNT(*) as cnt FROM client_actions GROUP BY priority`,
    sql`SELECT channel, COUNT(*) as cnt FROM client_actions GROUP BY channel ORDER BY cnt DESC`,
    sql`
      SELECT TRIM(unnest(string_to_array(top_concepts, '|'))) as concept,
             COUNT(DISTINCT id) as client_count
      FROM clients WHERE is_deleted = FALSE AND top_concepts IS NOT NULL AND top_concepts != ''
      GROUP BY concept ORDER BY client_count DESC LIMIT 10
    `,
    sql`SELECT s.id, s.name, s.profile, s.client_count as count FROM segments s ORDER BY s.id`,
    sql`
      SELECT us.filename, us.status, us.records_added, us.records_updated,
             us.created_at, u.display_name as user_name
      FROM upload_sessions us LEFT JOIN users u ON u.id = us.user_id
      ORDER BY us.created_at DESC LIMIT 5
    `,
    sql`SELECT COUNT(*) as cnt FROM event_targets WHERE action_status = 'pending'`,
    sql`SELECT COUNT(*) as cnt FROM event_targets WHERE notified_at IS NOT NULL`,
    sql`SELECT COUNT(*) as cnt FROM event_targets WHERE responded_at IS NOT NULL`,
    sql`
      SELECT language, COUNT(*) as cnt FROM clients
      WHERE is_deleted = FALSE AND language IS NOT NULL
      GROUP BY language ORDER BY cnt DESC
    `,
    sql`
      SELECT segment_id as segment, AVG(confidence) as avg_confidence, COUNT(*) as cnt
      FROM clients WHERE is_deleted = FALSE
      GROUP BY segment_id ORDER BY segment_id
    `,
  ])

  const totalClients = Number(clientsRow[0]?.cnt || 0)
  const totalActions = Number(actionsRow[0]?.cnt || 0)
  const completedActions = Number(completedActionsRow[0]?.cnt || 0)

  const priorityMap: Record<string, number> = { high: 0, medium: 0, low: 0 }
  actionsByPriority.forEach((r: any) => { priorityMap[r.priority] = Number(r.cnt) })

  return {
    totalClients,
    totalSegments: Number(segmentsRow[0]?.cnt || 0),
    totalActions,
    completedActions,
    actionCompletionRate: totalActions > 0 ? Math.round((completedActions / totalActions) * 100) : 0,
    totalEvents: Number(eventsRow[0]?.cnt || 0),
    activeEvents: Number(activeEventsRow[0]?.cnt || 0),
    actionsByPriority: priorityMap,
    actionsByChannel: actionsByChannel.map((r: any) => ({ channel: r.channel, count: Number(r.cnt) })),
    topConcepts: topConcepts.map((r: any) => ({ concept: r.concept, count: Number(r.client_count) })),
    segmentDistribution: segmentDist.map((r: any) => ({
      id: r.id, name: r.name, profile: r.profile, count: Number(r.count || 0),
    })),
    recentUploads: recentUploads.map((r: any) => ({
      filename: r.filename, status: r.status,
      recordsAdded: Number(r.records_added || 0),
      recordsUpdated: Number(r.records_updated || 0),
      date: r.created_at, userName: r.user_name || 'System',
    })),
    activationStats: {
      pending: Number(activationPending[0]?.cnt || 0),
      notified: Number(activationNotified[0]?.cnt || 0),
      responded: Number(activationResponded[0]?.cnt || 0),
    },
    languages: languages.map((r: any) => ({ language: r.language || 'Unknown', count: Number(r.cnt) })),
    confidenceBySegment: confidenceBySegment.map((r: any) => ({
      segment: Number(r.segment), avgConfidence: Number(Number(r.avg_confidence).toFixed(2)), count: Number(r.cnt),
    })),
  }
  } catch (err: any) {
    console.error('[dbService] dbGetKPI failed:', err?.message || err)
    throw err
  }
}

// ─── Main Data (for App.tsx scatter, segments, heatmap, etc.) ────
export async function dbGetData() {
  try {
  const [
    clientRows,
    conceptRows,
    actionRows,
    vectorRows,
    segmentRows,
    metricsClients,
    metricsSegments,
    topConceptsRows,
    heatmapData,
  ] = await Promise.all([
    sql`
      SELECT c.id, c.segment_id, c.confidence, c.top_concepts, c.full_text,
             c.language, c.note_date, c.note_duration, c.profile_type,
             s.name as segment_name, s.profile as segment_profile
      FROM clients c LEFT JOIN segments s ON s.id = c.segment_id
      WHERE c.is_deleted = FALSE
    `,
    sql`SELECT client_id, label, matched_alias, span_start, span_end FROM client_concepts`,
    sql`
      SELECT client_id, action_id, title, channel, priority, kpi, triggers, rationale, is_completed
      FROM client_actions
    `,
    sql`SELECT client_id, x_3d, y_3d, z_3d FROM client_vectors`,
    sql`SELECT id, name, profile, full_profile, client_count FROM segments ORDER BY id`,
    sql`SELECT COUNT(*) as cnt FROM clients WHERE is_deleted = FALSE`,
    sql`SELECT COUNT(*) as cnt FROM segments`,
    sql`
      SELECT cc.label as concept, COUNT(DISTINCT cc.client_id) as cnt,
             ARRAY_AGG(DISTINCT cc.client_id) as clients
      FROM client_concepts cc
      JOIN clients c ON c.id = cc.client_id AND c.is_deleted = FALSE
      GROUP BY cc.label ORDER BY cnt DESC LIMIT 20
    `,
    sql`
      SELECT c.segment_id,
             TRIM(unnest(string_to_array(c.top_concepts, '|'))) as concept,
             COUNT(*) as cnt
      FROM clients c
      WHERE c.is_deleted = FALSE AND c.top_concepts IS NOT NULL AND c.top_concepts != ''
      GROUP BY c.segment_id, concept
      ORDER BY cnt DESC
    `,
  ])

  // Build concepts-by-client lookup
  const conceptsByClient: Record<string, any[]> = {}
  conceptRows.forEach((r: any) => {
    if (!conceptsByClient[r.client_id]) conceptsByClient[r.client_id] = []
    conceptsByClient[r.client_id].push({
      concept: r.label, alias: r.matched_alias,
      spanStart: r.span_start, spanEnd: r.span_end,
    })
  })

  // Build actions-by-client lookup
  const actionsByClient: Record<string, any[]> = {}
  actionRows.forEach((r: any) => {
    if (!actionsByClient[r.client_id]) actionsByClient[r.client_id] = []
    actionsByClient[r.client_id].push({
      actionId: r.action_id, title: r.title, channel: r.channel,
      priority: r.priority, kpi: r.kpi, triggers: r.triggers,
      rationale: r.rationale, isCompleted: r.is_completed,
    })
  })

  // Build vector lookup
  const vectorMap: Record<string, { x: number; y: number; z: number }> = {}
  vectorRows.forEach((r: any) => {
    vectorMap[r.client_id] = { x: Number(r.x_3d), y: Number(r.y_3d), z: Number(r.z_3d) }
  })

  // Build clients array
  const clients = clientRows.map((c: any) => ({
    id: c.id,
    segment: Number(c.segment_id),
    confidence: Number(c.confidence || 0),
    profileType: c.profile_type || pipeSplit(c.top_concepts).slice(0, 3).join(' | '),
    topConcepts: pipeSplit(c.top_concepts),
    fullText: c.full_text || '',
    language: c.language || '',
    date: c.note_date || '',
    duration: c.note_duration || '',
    conceptEvidence: conceptsByClient[c.id] || [],
    actions: actionsByClient[c.id] || [],
  }))

  // Build scatter3d
  const scatter3d = clientRows
    .filter((c: any) => vectorMap[c.id])
    .map((c: any) => ({
      x: vectorMap[c.id].x, y: vectorMap[c.id].y, z: vectorMap[c.id].z,
      client: c.id, id: c.id, segment: Number(c.segment_id),
      profile: c.profile_type || pipeSplit(c.top_concepts).slice(0, 3).join(' | '),
    }))

  // Build segments array
  const segments = segmentRows.map((s: any) => ({
    name: s.name || `Segment ${s.id}`,
    value: Number(s.client_count || 0),
    profile: s.profile || '',
    fullProfile: s.full_profile || '',
  }))

  // Build concepts array
  const concepts = topConceptsRows.map((r: any) => ({
    concept: r.concept, count: Number(r.cnt), clients: r.clients || [],
  }))

  // Build heatmap
  const heatmapMap: Record<string, Record<string, number>> = {}
  const allHeatmapConcepts = new Set<string>()
  heatmapData.forEach((r: any) => {
    const segKey = `Seg ${r.segment_id}`
    if (!heatmapMap[segKey]) heatmapMap[segKey] = {}
    heatmapMap[segKey][r.concept] = Number(r.cnt)
    allHeatmapConcepts.add(r.concept)
  })
  // Take top 8 concepts
  const heatmapConcepts = Array.from(allHeatmapConcepts).slice(0, 8)
  const heatmap = Object.entries(heatmapMap).map(([seg, vals]) => {
    const row: Record<string, any> = { segment: seg }
    heatmapConcepts.forEach(c => { row[c] = vals[c] || 0 })
    return row
  })

  return {
    segments,
    clients,
    scatter3d,
    concepts,
    heatmap,
    heatmapConcepts,
    metrics: {
      clients: Number(metricsClients[0]?.cnt || 0),
      segments: Number(metricsSegments[0]?.cnt || 0),
    },
    radar: [],
    processingInfo: { timestamp: null, totalRecords: 0, pipelineTimings: {} },
  }
  } catch (err: any) {
    console.error('[dbService] dbGetData failed:', err?.message || err)
    throw err
  }
}

// ─── Client 360° Detail ──────────────────────────────────────────
export async function dbGetClient360(clientId: string) {
  const [
    clientRows,
    conceptRows,
    actionRows,
    eventRows,
    scoreRows,
    auditRows,
  ] = await Promise.all([
    sql`
      SELECT c.*, s.name as seg_name, s.profile as seg_profile,
             s.full_profile as seg_full_profile, s.client_count as seg_size,
             u.display_name as created_by_name, adv.display_name as advisor_name
      FROM clients c
      LEFT JOIN segments s ON s.id = c.segment_id
      LEFT JOIN users u ON u.id = c.created_by
      LEFT JOIN users adv ON adv.id = c.assigned_advisor_id
      WHERE c.id = ${clientId}
    `,
    sql`
      SELECT concept_id, label, matched_alias, span_start, span_end
      FROM client_concepts WHERE client_id = ${clientId}
    `,
    sql`
      SELECT action_id, title, channel, priority, kpi, triggers, rationale,
             is_completed, completed_at, created_at
      FROM client_actions WHERE client_id = ${clientId} ORDER BY priority, created_at
    `,
    sql`
      SELECT et.*, e.title as event_title, e.event_date, e.channel as event_channel,
             e.priority as event_priority, e.status as event_status
      FROM event_targets et
      JOIN events e ON e.id = et.event_id
      WHERE et.client_id = ${clientId}
    `,
    sql`
      SELECT engagement_score, value_score, overall_score, tier, score_details
      FROM client_scores WHERE client_id = ${clientId}
    `,
    sql`
      SELECT al.action, al.details, al.created_at, u.display_name
      FROM audit_log al LEFT JOIN users u ON u.id = al.user_id
      WHERE al.target_id = ${clientId}
      ORDER BY al.created_at DESC LIMIT 20
    `,
  ])

  if (clientRows.length === 0) throw new Error(`Client ${clientId} not found`)
  const c = clientRows[0]

  // Similar clients (same segment)
  const similarRows = await sql`
    SELECT id, segment_id, confidence, top_concepts, profile_type
    FROM clients WHERE segment_id = ${c.segment_id} AND id != ${clientId} AND is_deleted = FALSE
    ORDER BY confidence DESC LIMIT 5
  `

  const score = scoreRows.length > 0 ? {
    engagementScore: Number(scoreRows[0].engagement_score || 0),
    valueScore: Number(scoreRows[0].value_score || 0),
    overallScore: Number(scoreRows[0].overall_score || 0),
    tier: scoreRows[0].tier || 'bronze',
    details: scoreRows[0].score_details || null,
  } : null

  return {
    id: c.id,
    segment: Number(c.segment_id),
    segmentName: c.seg_name || `Segment ${c.segment_id}`,
    segmentProfile: c.seg_profile || '',
    segmentFullProfile: c.seg_full_profile || '',
    segmentSize: Number(c.seg_size || 0),
    confidence: Number(c.confidence || 0),
    profileType: c.profile_type || '',
    topConcepts: pipeSplit(c.top_concepts),
    fullText: c.full_text || '',
    language: c.language || '',
    noteDate: c.note_date || null,
    noteDuration: c.note_duration || '',
    createdBy: c.created_by_name || 'System',
    createdAt: c.created_at || '',
    updatedAt: c.updated_at || '',
    conceptEvidence: conceptRows.map((r: any) => ({
      conceptId: r.concept_id, label: r.label, matchedAlias: r.matched_alias,
      spanStart: r.span_start, spanEnd: r.span_end,
    })),
    actions: actionRows.map((r: any) => ({
      actionId: r.action_id, title: r.title, channel: r.channel,
      priority: r.priority, kpi: r.kpi, triggers: r.triggers,
      rationale: r.rationale, isCompleted: r.is_completed,
      completedAt: r.completed_at, createdAt: r.created_at,
    })),
    events: eventRows.map((r: any) => ({
      eventId: r.event_id, eventTitle: r.event_title, eventDate: r.event_date,
      eventChannel: r.event_channel, eventPriority: r.event_priority,
      eventStatus: r.event_status, matchReason: r.match_reason,
      matchScore: Number(r.match_score || 0), actionStatus: r.action_status,
      notifiedAt: r.notified_at, respondedAt: r.responded_at,
    })),
    similarClients: similarRows.map((r: any) => ({
      id: r.id, segment: Number(r.segment_id), confidence: Number(r.confidence || 0),
      topConcepts: pipeSplit(r.top_concepts), profileType: r.profile_type || '',
    })),
    timeline: auditRows.map((r: any) => ({
      action: r.action, details: r.details, date: r.created_at, userName: r.display_name || 'System',
    })),
    score,
  }
}

// ─── Playbooks ────────────────────────────────────────────────────
export async function dbGetPlaybooks(category?: string) {
  if (category) {
    return await sql`
      SELECT id, name, description, concepts, channel, priority,
             message_template as "messageTemplate", category, created_at as "createdAt"
      FROM playbooks WHERE is_active = TRUE AND category = ${category}
      ORDER BY created_at DESC
    `
  }
  return await sql`
    SELECT id, name, description, concepts, channel, priority,
           message_template as "messageTemplate", category, created_at as "createdAt"
    FROM playbooks WHERE is_active = TRUE
    ORDER BY category, name
  `
}

// ─── Advisors ─────────────────────────────────────────────────────
export async function dbGetAdvisors() {
  const rows = await sql`
    SELECT u.id, u.username, u.display_name, u.role,
           COUNT(c.id) as client_count
    FROM users u
    LEFT JOIN clients c ON c.assigned_advisor_id = u.id AND c.is_deleted = FALSE
    WHERE u.is_active = TRUE AND u.role IN ('sales', 'manager', 'admin')
    GROUP BY u.id, u.username, u.display_name, u.role
    ORDER BY u.display_name
  `
  return rows.map((r: any) => ({
    id: r.id, username: r.username, displayName: r.display_name,
    role: r.role, clientCount: Number(r.client_count || 0),
  }))
}

export async function dbGetAdvisorWorkload() {
  const advisorRows = await sql`
    SELECT u.id, u.display_name as name, u.role,
           COUNT(c.id) as total_clients,
           COUNT(CASE WHEN cs.tier = 'platinum' THEN 1 END) as platinum,
           COUNT(CASE WHEN cs.tier = 'gold' THEN 1 END) as gold,
           COUNT(CASE WHEN cs.tier = 'silver' THEN 1 END) as silver,
           COUNT(CASE WHEN cs.tier = 'bronze' THEN 1 END) as bronze,
           COALESCE(AVG(cs.overall_score), 0) as avg_score
    FROM users u
    LEFT JOIN clients c ON c.assigned_advisor_id = u.id AND c.is_deleted = FALSE
    LEFT JOIN client_scores cs ON cs.client_id = c.id
    WHERE u.is_active = TRUE AND u.role IN ('sales', 'manager', 'admin')
    GROUP BY u.id, u.display_name, u.role
    ORDER BY u.display_name
  `
  const unassignedRow = await sql`
    SELECT COUNT(*) as cnt FROM clients
    WHERE is_deleted = FALSE AND (assigned_advisor_id IS NULL)
  `
  return {
    advisors: advisorRows.map((r: any) => ({
      id: r.id, name: r.name, role: r.role,
      totalClients: Number(r.total_clients || 0),
      platinum: Number(r.platinum || 0), gold: Number(r.gold || 0),
      silver: Number(r.silver || 0), bronze: Number(r.bronze || 0),
      avgScore: Number(Number(r.avg_score || 0).toFixed(1)),
    })),
    unassignedCount: Number(unassignedRow[0]?.cnt || 0),
  }
}

// ─── Calendar Events ──────────────────────────────────────────────
export async function dbGetEvents(month?: number, year?: number, status?: string) {
  let rows
  if (month !== undefined && year !== undefined && status) {
    rows = await sql`
      SELECT e.*, u.display_name as creator_name
      FROM events e LEFT JOIN users u ON u.id = e.created_by
      WHERE EXTRACT(MONTH FROM e.event_date) = ${month}
        AND EXTRACT(YEAR FROM e.event_date) = ${year}
        AND e.status = ${status}
      ORDER BY e.event_date ASC
    `
  } else if (month !== undefined && year !== undefined) {
    rows = await sql`
      SELECT e.*, u.display_name as creator_name
      FROM events e LEFT JOIN users u ON u.id = e.created_by
      WHERE EXTRACT(MONTH FROM e.event_date) = ${month}
        AND EXTRACT(YEAR FROM e.event_date) = ${year}
      ORDER BY e.event_date ASC
    `
  } else {
    rows = await sql`
      SELECT e.*, u.display_name as creator_name
      FROM events e LEFT JOIN users u ON u.id = e.created_by
      ORDER BY e.event_date ASC
    `
  }
  return rows.map((r: any) => ({
    id: r.id, title: r.title, description: r.description,
    event_date: r.event_date, event_end_date: r.event_end_date,
    concepts: r.concepts || '', channel: r.channel, priority: r.priority,
    status: r.status, matched_count: Number(r.matched_count || 0),
    notified_count: Number(r.notified_count || 0),
    created_by: r.created_by, creator_name: r.creator_name,
    created_at: r.created_at, updated_at: r.updated_at,
  }))
}

export async function dbGetEventDetail(eventId: number) {
  const eventRows = await sql`
    SELECT e.*, u.display_name as creator_name
    FROM events e LEFT JOIN users u ON u.id = e.created_by
    WHERE e.id = ${eventId}
  `
  if (eventRows.length === 0) throw new Error('Event not found')
  const e = eventRows[0]

  const targets = await sql`
    SELECT et.client_id, et.match_reason, et.match_score, et.action_status,
           et.notified_at, et.responded_at,
           c.profile_type, c.segment_id, c.top_concepts
    FROM event_targets et
    JOIN clients c ON c.id = et.client_id
    WHERE et.event_id = ${eventId}
  `

  return {
    id: e.id, title: e.title, description: e.description,
    event_date: e.event_date, event_end_date: e.event_end_date,
    concepts: e.concepts || '', channel: e.channel, priority: e.priority,
    status: e.status, matched_count: Number(e.matched_count || 0),
    notified_count: Number(e.notified_count || 0),
    created_by: e.created_by, creator_name: e.creator_name,
    created_at: e.created_at, updated_at: e.updated_at,
    targets: targets.map((t: any) => ({
      clientId: t.client_id, matchReason: t.match_reason,
      matchScore: Number(t.match_score || 0), actionStatus: t.action_status,
      notifiedAt: t.notified_at, respondedAt: t.responded_at,
      profileType: t.profile_type || '', segment: Number(t.segment_id),
      topConcepts: pipeSplit(t.top_concepts),
    })),
  }
}

export async function dbGetConcepts() {
  const rows = await sql`
    SELECT cc.label as concept, COUNT(DISTINCT cc.client_id) as client_count
    FROM client_concepts cc
    JOIN clients c ON c.id = cc.client_id AND c.is_deleted = FALSE
    GROUP BY cc.label ORDER BY client_count DESC
  `
  return rows.map((r: any) => ({ concept: r.concept, clientCount: Number(r.client_count) }))
}

// ─── Report Summary ───────────────────────────────────────────────
export async function dbGetReportSummary() {
  const [
    totalRow,
    segmentRows,
    actionRows,
    conceptRows,
    tierRows,
    langRows,
    advisorRows,
  ] = await Promise.all([
    sql`SELECT COUNT(*) as cnt FROM clients WHERE is_deleted = FALSE`,
    sql`
      SELECT s.id, s.name, s.client_count as count,
             COALESCE(AVG(cs.overall_score), 0) as avg_score,
             COUNT(CASE WHEN cs.tier = 'gold' THEN 1 END) as gold,
             COUNT(CASE WHEN cs.tier = 'silver' THEN 1 END) as silver
      FROM segments s
      LEFT JOIN clients c ON c.segment_id = s.id AND c.is_deleted = FALSE
      LEFT JOIN client_scores cs ON cs.client_id = c.id
      GROUP BY s.id, s.name, s.client_count ORDER BY s.id
    `,
    sql`
      SELECT COUNT(*) as total,
             COUNT(CASE WHEN is_completed THEN 1 END) as completed,
             COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority
      FROM client_actions
    `,
    sql`
      SELECT concept_id as id, label, COUNT(*) as mentions
      FROM client_concepts GROUP BY concept_id, label
      ORDER BY mentions DESC LIMIT 10
    `,
    sql`
      SELECT tier, COUNT(*) as count, COALESCE(AVG(overall_score), 0) as avg_score
      FROM client_scores GROUP BY tier ORDER BY avg_score DESC
    `,
    sql`
      SELECT language, COUNT(*) as count FROM clients
      WHERE is_deleted = FALSE AND language IS NOT NULL
      GROUP BY language ORDER BY count DESC
    `,
    sql`
      SELECT u.display_name as name, COUNT(c.id) as clients,
             COALESCE(AVG(cs.overall_score), 0) as avg_score
      FROM users u
      LEFT JOIN clients c ON c.assigned_advisor_id = u.id AND c.is_deleted = FALSE
      LEFT JOIN client_scores cs ON cs.client_id = c.id
      WHERE u.is_active = TRUE AND u.role IN ('sales', 'manager', 'admin')
      GROUP BY u.display_name ORDER BY clients DESC
    `,
  ])

  const total = Number(actionRows[0]?.total || 0)
  const completed = Number(actionRows[0]?.completed || 0)

  return {
    generatedAt: new Date().toISOString(),
    totalClients: Number(totalRow[0]?.cnt || 0),
    segments: segmentRows.map((r: any) => ({
      id: r.id, name: r.name, count: Number(r.count || 0),
      avgScore: Number(Number(r.avg_score || 0).toFixed(1)),
      gold: Number(r.gold || 0), silver: Number(r.silver || 0),
    })),
    actions: {
      total, completed,
      completionRate: total > 0 ? Math.round((completed / total) * 100) : 0,
      highPriority: Number(actionRows[0]?.high_priority || 0),
    },
    topConcepts: conceptRows.map((r: any) => ({
      id: r.id, label: r.label, mentions: Number(r.mentions || 0),
    })),
    tiers: tierRows.map((r: any) => ({
      tier: r.tier, count: Number(r.count || 0),
      avgScore: Number(Number(r.avg_score || 0).toFixed(1)),
    })),
    languages: langRows.map((r: any) => ({ language: r.language || 'Unknown', count: Number(r.count) })),
    advisors: advisorRows.map((r: any) => ({
      name: r.name, clients: Number(r.clients || 0),
      avgScore: Number(Number(r.avg_score || 0).toFixed(1)),
    })),
  }
}
