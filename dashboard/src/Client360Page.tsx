import { useState, useEffect, useMemo } from 'react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, Tooltip, PieChart, Pie, Cell
} from 'recharts'
import { getClient360 } from './services/apiService'

// ─── Types ────────────────────────────────────────────
interface ConceptEvidence {
  conceptId: string
  label: string
  matchedAlias: string
  spanStart: number
  spanEnd: number
}

interface ClientAction {
  actionId: string
  title: string
  channel: string
  priority: string
  kpi: string
  triggers: string
  rationale: string
  isCompleted: boolean
  completedAt: string | null
  createdAt: string
}

interface ClientEvent {
  eventId: number
  eventTitle: string
  eventDate: string
  eventChannel: string
  eventPriority: string
  eventStatus: string
  matchReason: string
  matchScore: number
  actionStatus: string
  notifiedAt: string | null
  respondedAt: string | null
}

interface SimilarClient {
  id: string
  segment: number
  confidence: number
  topConcepts: string[]
  profileType: string
}

interface TimelineEntry {
  action: string
  details: any
  date: string
  userName: string
}

interface ClientScore {
  engagementScore: number
  valueScore: number
  overallScore: number
  tier: string
  details: {
    conceptScore: number
    actionScore: number
    completionRate: number
    eventScore: number
    recencyScore: number
    confidencePct: number
    priorityScore: number
    richnessScore: number
  } | null
}

interface Client360Data {
  id: string
  segment: number
  segmentName: string
  segmentProfile: string
  segmentFullProfile: string
  segmentSize: number
  confidence: number
  profileType: string
  topConcepts: string[]
  fullText: string
  language: string
  noteDate: string | null
  noteDuration: string
  createdBy: string
  createdAt: string
  updatedAt: string
  conceptEvidence: ConceptEvidence[]
  actions: ClientAction[]
  events: ClientEvent[]
  similarClients: SimilarClient[]
  timeline: TimelineEntry[]
  score: ClientScore | null
}

// ─── Colour palette ───────────────────────────────────
const SEGMENT_COLORS: Record<number, string> = {
  0: '#6366f1', 1: '#8b5cf6', 2: '#ec4899', 3: '#f43f5e',
  4: '#3b82f6', 5: '#06b6d4', 6: '#10b981', 7: '#84cc16',
}

const PRIORITY_COLORS: Record<string, string> = {
  high: '#ef4444', medium: '#f59e0b', low: '#22c55e',
}

const STATUS_COLORS: Record<string, string> = {
  pending: '#94a3b8', notified: '#3b82f6', responded: '#22c55e', skipped: '#9ca3af',
}

// ─── Icons ────────────────────────────────────────────
const Icons = {
  back: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>,
  user: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
  tag: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>,
  actions: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>,
  calendar: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>,
  globe: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>,
  check: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>,
  clock: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>,
  star: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>,
  link: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>,
  doc: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>,
}

// ─── Component ────────────────────────────────────────
export default function Client360Page({
  clientId,
  onBack,
}: {
  clientId: string
  onBack: () => void
}) {
  const [data, setData] = useState<Client360Data | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'actions' | 'events' | 'notes'>('overview')

  useEffect(() => {
    const fetchClient = async () => {
      setLoading(true)
      setError(null)
      try {
        const { data: clientData } = await getClient360(clientId)
        setData(clientData)
      } catch (e: any) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    fetchClient()
  }, [clientId])

  const color = SEGMENT_COLORS[data?.segment ?? 0] || '#6366f1'

  // Compute concept radar data
  const radarData = useMemo(() => {
    if (!data?.topConcepts?.length) return []
    return data.topConcepts.slice(0, 8).map((c, i) => ({
      concept: c.length > 14 ? c.slice(0, 12) + '…' : c,
      value: Math.max(20, 100 - i * 12),
    }))
  }, [data])

  // Action stats
  const actionStats = useMemo(() => {
    if (!data?.actions?.length) return { total: 0, completed: 0, high: 0, pending: 0 }
    return {
      total: data.actions.length,
      completed: data.actions.filter(a => a.isCompleted).length,
      high: data.actions.filter(a => a.priority === 'high').length,
      pending: data.actions.filter(a => !a.isCompleted).length,
    }
  }, [data])

  if (loading) {
    return (
      <div className="page c360-page">
        <div className="c360-loading">
          <div className="loading-spinner"><div></div><div></div><div></div></div>
          <p>Loading client profile…</p>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="page c360-page">
        <button className="c360-back-btn" onClick={onBack}>{Icons.back} Back to Clients</button>
        <div className="c360-error">
          <h3>Client Not Found</h3>
          <p>{error || 'No data returned.'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="page c360-page">
      {/* Back Button */}
      <button className="c360-back-btn" onClick={onBack}>
        {Icons.back} Back to Clients
      </button>

      {/* ─── Header Card ──────────────────────────────── */}
      <div className="c360-header" style={{ '--c360-color': color } as any}>
        <div className="c360-header-left">
          <div className="c360-avatar" style={{ backgroundColor: color }}>
            {data.id.slice(-3)}
          </div>
          <div className="c360-identity">
            <h1>{data.id}</h1>
            <div className="c360-meta-row">
              <span className="c360-segment-badge" style={{ backgroundColor: color + '20', color }}>
                {data.segmentName} — {data.segmentProfile}
              </span>
              <span className="c360-meta-item">{Icons.globe} {data.language.toUpperCase()}</span>
              {data.noteDate && (
                <span className="c360-meta-item">{Icons.calendar} {new Date(data.noteDate).toLocaleDateString()}</span>
              )}
              {data.createdBy && (
                <span className="c360-meta-item">{Icons.user} {data.createdBy}</span>
              )}
            </div>
          </div>
        </div>
        <div className="c360-header-right">
          {data.score && (
            <div className="c360-tier-section">
              <span className={`c360-tier-badge tier-${data.score.tier}`}>
                {Icons.star} {data.score.tier.toUpperCase()}
              </span>
              <span className="c360-overall-score">Score: {data.score.overallScore}</span>
            </div>
          )}
          <div className="c360-score-ring" style={{ '--confidence': data.confidence, '--ring-color': color } as any}>
            <span>{Math.round(data.confidence * 100)}%</span>
          </div>
          <div className="c360-score-label">Confidence</div>
        </div>
      </div>

      {/* ─── Quick Stats ──────────────────────────────── */}
      <div className="c360-stats">
        <div className="c360-stat">
          <div className="c360-stat-value">{data.topConcepts.length}</div>
          <div className="c360-stat-label">Concepts</div>
        </div>
        <div className="c360-stat">
          <div className="c360-stat-value">{actionStats.total}</div>
          <div className="c360-stat-label">Actions</div>
        </div>
        <div className="c360-stat">
          <div className="c360-stat-value">{actionStats.completed}</div>
          <div className="c360-stat-label">Completed</div>
        </div>
        <div className="c360-stat">
          <div className="c360-stat-value">{data.events.length}</div>
          <div className="c360-stat-label">Activations</div>
        </div>
        <div className="c360-stat">
          <div className="c360-stat-value">{data.similarClients.length}</div>
          <div className="c360-stat-label">Similar</div>
        </div>
      </div>

      {/* ─── Tab Navigation ───────────────────────────── */}
      <div className="c360-tabs">
        {(['overview', 'actions', 'events', 'notes'] as const).map(tab => (
          <button
            key={tab}
            className={`c360-tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'overview' && Icons.user}
            {tab === 'actions' && Icons.actions}
            {tab === 'events' && Icons.calendar}
            {tab === 'notes' && Icons.doc}
            <span>{tab.charAt(0).toUpperCase() + tab.slice(1)}</span>
          </button>
        ))}
      </div>

      {/* ─── Tab Content ──────────────────────────────── */}
      <div className="c360-content">
        {activeTab === 'overview' && (
          <div className="c360-overview-grid">
            {/* Concept Cloud */}
            <div className="card c360-card">
              <h3 className="card-title">{Icons.tag} Concept Profile</h3>
              <div className="c360-concept-cloud">
                {data.topConcepts.map((concept, i) => (
                  <span
                    key={i}
                    className="c360-concept-badge"
                    style={{
                      backgroundColor: color + (i < 3 ? '25' : '12'),
                      color,
                      borderColor: color + '40',
                      fontSize: i < 3 ? '0.9rem' : '0.8rem',
                      fontWeight: i < 3 ? 600 : 400,
                    }}
                  >
                    {concept}
                  </span>
                ))}
              </div>
              {radarData.length > 2 && (
                <div style={{ width: '100%', height: 260, marginTop: '1rem' }}>
                  <ResponsiveContainer>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="var(--border-light)" />
                      <PolarAngleAxis dataKey="concept" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                      <Radar
                        dataKey="value"
                        stroke={color}
                        fill={color}
                        fillOpacity={0.2}
                        strokeWidth={2}
                      />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Segment Info */}
            <div className="card c360-card">
              <h3 className="card-title">{Icons.star} Segment Details</h3>
              <div className="c360-segment-info">
                <div className="c360-segment-hero" style={{ backgroundColor: color + '10' }}>
                  <div className="c360-segment-number" style={{ color }}>{data.segment}</div>
                  <div>
                    <div className="c360-segment-name">{data.segmentName}</div>
                    <div className="c360-segment-count">{data.segmentSize} clients in this segment</div>
                  </div>
                </div>
                <p className="c360-segment-desc">{data.segmentFullProfile || data.segmentProfile}</p>
              </div>

              {/* Similar Clients */}
              {data.similarClients.length > 0 && (
                <>
                  <h4 style={{ marginTop: '1.5rem', marginBottom: '0.75rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    {Icons.link} Similar Clients
                  </h4>
                  <div className="c360-similar-list">
                    {data.similarClients.map(sc => (
                      <div key={sc.id} className="c360-similar-chip">
                        <div
                          className="c360-similar-avatar"
                          style={{ backgroundColor: SEGMENT_COLORS[sc.segment] || '#6366f1' }}
                        >
                          {sc.id.slice(-2)}
                        </div>
                        <div className="c360-similar-info">
                          <span className="c360-similar-id">{sc.id}</span>
                          <span className="c360-similar-conf">{Math.round(sc.confidence * 100)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* Action Summary Donut */}
            {actionStats.total > 0 && (
              <div className="card c360-card">
                <h3 className="card-title">{Icons.actions} Action Summary</h3>
                <div style={{ width: '100%', height: 200 }}>
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Completed', value: actionStats.completed, fill: '#22c55e' },
                          { name: 'Pending', value: actionStats.pending, fill: '#94a3b8' },
                        ]}
                        cx="50%" cy="50%"
                        innerRadius={50} outerRadius={75}
                        paddingAngle={4}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}`}
                      >
                        <Cell fill="#22c55e" />
                        <Cell fill="#94a3b8" />
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="c360-action-prio-row">
                  {['high', 'medium', 'low'].map(p => {
                    const count = data.actions.filter(a => a.priority === p).length
                    return count > 0 ? (
                      <span key={p} className="c360-prio-badge" style={{ backgroundColor: PRIORITY_COLORS[p] + '18', color: PRIORITY_COLORS[p] }}>
                        {p}: {count}
                      </span>
                    ) : null
                  })}
                </div>
              </div>
            )}

            {/* Score Breakdown */}
            {data.score && data.score.details && (
              <div className="card c360-card" style={{ gridColumn: '1 / -1' }}>
                <h3 className="card-title">{Icons.star} Client Score Breakdown</h3>
                <div className="c360-score-grid">
                  <div className="c360-score-summary">
                    <div className={`c360-tier-large tier-${data.score.tier}`}>
                      {data.score.tier.toUpperCase()}
                    </div>
                    <div className="c360-score-bars">
                      <div className="c360-score-bar-item">
                        <span>Engagement</span>
                        <div className="c360-score-bar-bg">
                          <div className="c360-score-bar" style={{ width: `${data.score.engagementScore}%`, background: '#6366f1' }} />
                        </div>
                        <span className="c360-score-num">{data.score.engagementScore}</span>
                      </div>
                      <div className="c360-score-bar-item">
                        <span>Value</span>
                        <div className="c360-score-bar-bg">
                          <div className="c360-score-bar" style={{ width: `${data.score.valueScore}%`, background: '#ec4899' }} />
                        </div>
                        <span className="c360-score-num">{data.score.valueScore}</span>
                      </div>
                      <div className="c360-score-bar-item">
                        <span>Overall</span>
                        <div className="c360-score-bar-bg">
                          <div className="c360-score-bar" style={{ width: `${data.score.overallScore}%`, background: '#10b981' }} />
                        </div>
                        <span className="c360-score-num">{data.score.overallScore}</span>
                      </div>
                    </div>
                  </div>
                  <div className="c360-score-details-grid">
                    {[
                      { label: 'Concepts', value: data.score.details.conceptScore, color: '#6366f1' },
                      { label: 'Actions', value: data.score.details.actionScore, color: '#8b5cf6' },
                      { label: 'Completion', value: data.score.details.completionRate, color: '#22c55e' },
                      { label: 'Events', value: data.score.details.eventScore, color: '#3b82f6' },
                      { label: 'Recency', value: data.score.details.recencyScore, color: '#f59e0b' },
                      { label: 'Confidence', value: data.score.details.confidencePct, color: '#ec4899' },
                      { label: 'Priority', value: data.score.details.priorityScore, color: '#ef4444' },
                      { label: 'Richness', value: data.score.details.richnessScore, color: '#06b6d4' },
                    ].map(item => (
                      <div key={item.label} className="c360-score-detail-item">
                        <div className="c360-score-detail-bar-bg">
                          <div
                            className="c360-score-detail-bar"
                            style={{ height: `${item.value}%`, backgroundColor: item.color }}
                          />
                        </div>
                        <span className="c360-score-detail-val">{Math.round(item.value)}</span>
                        <span className="c360-score-detail-label">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'actions' && (
          <div className="c360-actions-list">
            {data.actions.length === 0 ? (
              <div className="c360-empty">{Icons.actions}<p>No actions for this client.</p></div>
            ) : (
              data.actions.map((action, i) => (
                <div key={i} className={`c360-action-card ${action.isCompleted ? 'completed' : ''}`}>
                  <div className="c360-action-header">
                    <span className={`c360-prio-dot ${action.priority}`} />
                    <span className="c360-action-title">{action.title}</span>
                    <span className="c360-action-channel">{action.channel}</span>
                    {action.isCompleted && <span className="c360-completed-badge">{Icons.check} Done</span>}
                  </div>
                  {action.rationale && <p className="c360-action-rationale">{action.rationale}</p>}
                  <div className="c360-action-meta">
                    {action.kpi && <span>KPI: {action.kpi}</span>}
                    {action.triggers && (
                      <div className="c360-trigger-tags">
                        {action.triggers.split('|').map((t, j) => (
                          <span key={j} className="c360-trigger-tag">{t.trim()}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'events' && (
          <div className="c360-events-list">
            {data.events.length === 0 ? (
              <div className="c360-empty">{Icons.calendar}<p>No activation events match this client.</p></div>
            ) : (
              data.events.map((ev, i) => (
                <div key={i} className="c360-event-card">
                  <div className="c360-event-header">
                    <span className={`c360-event-status ${ev.eventStatus}`}>{ev.eventStatus}</span>
                    <h4>{ev.eventTitle}</h4>
                    <span className="c360-event-date">{new Date(ev.eventDate).toLocaleDateString()}</span>
                  </div>
                  <div className="c360-event-body">
                    <div className="c360-event-detail">
                      <span className="label">Channel</span>
                      <span>{ev.eventChannel}</span>
                    </div>
                    <div className="c360-event-detail">
                      <span className="label">Match Score</span>
                      <span>{Math.round(ev.matchScore * 100)}%</span>
                    </div>
                    <div className="c360-event-detail">
                      <span className="label">Match Reason</span>
                      <span>{ev.matchReason}</span>
                    </div>
                    <div className="c360-event-detail">
                      <span className="label">Status</span>
                      <span
                        className="c360-target-status"
                        style={{ color: STATUS_COLORS[ev.actionStatus] || '#94a3b8' }}
                      >
                        {ev.actionStatus}
                      </span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'notes' && (
          <div className="c360-notes-section">
            <div className="card c360-card">
              <h3 className="card-title">{Icons.doc} Original Note / Transcript</h3>
              {data.fullText ? (
                <div className="c360-note-content">
                  <div className="c360-note-meta">
                    {data.language && <span>{Icons.globe} {data.language.toUpperCase()}</span>}
                    {data.noteDate && <span>{Icons.calendar} {new Date(data.noteDate).toLocaleDateString()}</span>}
                    {data.noteDuration && <span>{Icons.clock} {data.noteDuration}</span>}
                  </div>
                  <blockquote className="c360-note-text">{data.fullText}</blockquote>
                </div>
              ) : (
                <div className="c360-empty"><p>No transcript available for this client.</p></div>
              )}
            </div>

            {/* Concept Evidence */}
            {data.conceptEvidence.length > 0 && (
              <div className="card c360-card" style={{ marginTop: '1rem' }}>
                <h3 className="card-title">{Icons.tag} Extracted Concepts (Evidence)</h3>
                <table className="c360-evidence-table">
                  <thead>
                    <tr>
                      <th>Concept</th>
                      <th>Label</th>
                      <th>Matched Alias</th>
                      <th>Span</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.conceptEvidence.map((ce, i) => (
                      <tr key={i}>
                        <td><code>{ce.conceptId}</code></td>
                        <td>{ce.label}</td>
                        <td>{ce.matchedAlias || '—'}</td>
                        <td>{ce.spanStart}–{ce.spanEnd}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Timeline */}
            {data.timeline.length > 0 && (
              <div className="card c360-card" style={{ marginTop: '1rem' }}>
                <h3 className="card-title">{Icons.clock} Activity Timeline</h3>
                <div className="c360-timeline">
                  {data.timeline.map((entry, i) => (
                    <div key={i} className="c360-timeline-item">
                      <div className="c360-timeline-dot" />
                      <div className="c360-timeline-content">
                        <span className="c360-timeline-action">{entry.action}</span>
                        {entry.userName && <span className="c360-timeline-user">by {entry.userName}</span>}
                        <span className="c360-timeline-date">{new Date(entry.date).toLocaleString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
