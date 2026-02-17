import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  PieChart, Pie, Cell,
  ResponsiveContainer, Tooltip, Legend
} from 'recharts'
import { getKPI } from './services/apiService'

// ─── Types ────────────────────────────────────────────
interface KPIData {
  totalClients: number
  totalSegments: number
  totalActions: number
  completedActions: number
  actionCompletionRate: number
  totalEvents: number
  activeEvents: number
  actionsByPriority: Record<string, number>
  actionsByChannel: { channel: string; count: number }[]
  topConcepts: { concept: string; count: number }[]
  segmentDistribution: { id: number; name: string; profile: string; count: number }[]
  recentUploads: { filename: string; status: string; recordsAdded: number; recordsUpdated: number; date: string; userName: string }[]
  activationStats: Record<string, number>
  languages: { language: string; count: number }[]
  confidenceBySegment: { segment: number; avgConfidence: number; count: number }[]
}

// ─── Color palette ────────────────────────────────────
const SEGMENT_COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#3b82f6', '#06b6d4', '#10b981', '#84cc16']
const CONCEPT_COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#3b82f6', '#06b6d4', '#10b981', '#84cc16', '#eab308', '#f97316']

// ─── Icons ────────────────────────────────────────────
const KPIIcons = {
  clients: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75"/></svg>,
  segments: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0110 10M12 2v10l7.07 7.07"/></svg>,
  actions: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>,
  calendar: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>,
  check: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>,
  trending: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>,
  upload: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>,
  globe: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>,
}

// ─── Component ────────────────────────────────────────
export default function KPIDashboard() {
  const [data, setData] = useState<KPIData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchKPI = async () => {
      setLoading(true)
      try {
        const { data: kpiData } = await getKPI()
        setData(kpiData)
        setError(null)
      } catch (e: any) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    fetchKPI()
  }, [])

  if (loading) {
    return (
      <div className="page kpi-page">
        <div className="c360-loading">
          <div className="loading-spinner"><div></div><div></div><div></div></div>
          <p>Loading dashboard…</p>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="page kpi-page">
        <div className="c360-error">
          <h3>Could not load KPI data</h3>
          <p>{error || 'No data returned.'}</p>
        </div>
      </div>
    )
  }

  const totalActivations = Object.values(data.activationStats).reduce((s, v) => s + v, 0)

  return (
    <div className="page kpi-page">
      <div className="page-header">
        <div>
          <h1>Dashboard</h1>
          <p>Executive overview of your client intelligence platform.</p>
        </div>
      </div>

      {/* ─── Hero KPI Cards ─────────────────────────── */}
      <div className="kpi-hero-grid">
        <div className="kpi-hero-card kpi-purple">
          <div className="kpi-hero-icon">{KPIIcons.clients}</div>
          <div className="kpi-hero-body">
            <div className="kpi-hero-value">{data.totalClients.toLocaleString()}</div>
            <div className="kpi-hero-label">Total Clients</div>
          </div>
        </div>
        <div className="kpi-hero-card kpi-blue">
          <div className="kpi-hero-icon">{KPIIcons.segments}</div>
          <div className="kpi-hero-body">
            <div className="kpi-hero-value">{data.totalSegments}</div>
            <div className="kpi-hero-label">Segments</div>
          </div>
        </div>
        <div className="kpi-hero-card kpi-amber">
          <div className="kpi-hero-icon">{KPIIcons.actions}</div>
          <div className="kpi-hero-body">
            <div className="kpi-hero-value">{data.totalActions.toLocaleString()}</div>
            <div className="kpi-hero-label">Actions</div>
            <div className="kpi-hero-sub">{data.actionCompletionRate}% completed</div>
          </div>
        </div>
        <div className="kpi-hero-card kpi-green">
          <div className="kpi-hero-icon">{KPIIcons.calendar}</div>
          <div className="kpi-hero-body">
            <div className="kpi-hero-value">{data.totalEvents}</div>
            <div className="kpi-hero-label">Activations</div>
            <div className="kpi-hero-sub">{data.activeEvents} active</div>
          </div>
        </div>
      </div>

      {/* ─── Charts Row 1 ───────────────────────────── */}
      <div className="kpi-charts-row">
        {/* Segment Distribution */}
        <div className="card kpi-chart-card">
          <h3 className="card-title">{KPIIcons.segments} Segment Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={data.segmentDistribution.map(s => ({ ...s, name: s.profile || s.name }))}
                cx="50%" cy="50%"
                innerRadius={55} outerRadius={95}
                paddingAngle={3}
                dataKey="count"
                label={({ name, percent }) => `${name?.slice(0, 15)} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {data.segmentDistribution.map((_, i) => (
                  <Cell key={i} fill={SEGMENT_COLORS[i % SEGMENT_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => [`${value} clients`, 'Clients']} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Top Concepts */}
        <div className="card kpi-chart-card">
          <h3 className="card-title">{KPIIcons.trending} Top Concepts</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.topConcepts} layout="vertical" margin={{ left: 10, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="concept"
                width={130}
                tick={{ fontSize: 11 }}
                tickFormatter={(v: string) => v.length > 18 ? v.slice(0, 16) + '…' : v}
              />
              <Tooltip />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={18}>
                {data.topConcepts.map((_, i) => (
                  <Cell key={i} fill={CONCEPT_COLORS[i % CONCEPT_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ─── Charts Row 2 ───────────────────────────── */}
      <div className="kpi-charts-row">
        {/* Actions by Channel */}
        <div className="card kpi-chart-card">
          <h3 className="card-title">{KPIIcons.actions} Actions by Channel</h3>
          {data.actionsByChannel.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data.actionsByChannel} margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
                <XAxis
                  dataKey="channel"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v: string) => v.length > 12 ? v.slice(0, 10) + '…' : v}
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={32} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="kpi-empty">No action data yet</div>
          )}
        </div>

        {/* Activation Funnel */}
        <div className="card kpi-chart-card">
          <h3 className="card-title">{KPIIcons.calendar} Activation Funnel</h3>
          {totalActivations > 0 ? (
            <div className="kpi-funnel">
              {[
                { label: 'Matched', key: 'pending', color: '#94a3b8' },
                { label: 'Notified', key: 'notified', color: '#3b82f6' },
                { label: 'Responded', key: 'responded', color: '#22c55e' },
                { label: 'Skipped', key: 'skipped', color: '#9ca3af' },
              ].map(stage => {
                const val = data.activationStats[stage.key] || 0
                const pct = totalActivations ? Math.round(val / totalActivations * 100) : 0
                return (
                  <div key={stage.key} className="kpi-funnel-stage">
                    <div className="kpi-funnel-bar-bg">
                      <div
                        className="kpi-funnel-bar"
                        style={{ width: `${pct}%`, backgroundColor: stage.color }}
                      />
                    </div>
                    <div className="kpi-funnel-label">
                      <span>{stage.label}</span>
                      <span className="kpi-funnel-value">{val} ({pct}%)</span>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="kpi-empty">No activation data yet. Create events in the Calendar to see results.</div>
          )}
        </div>
      </div>

      {/* ─── Bottom Row ─────────────────────────────── */}
      <div className="kpi-charts-row kpi-bottom-row">
        {/* Languages */}
        <div className="card kpi-chart-card kpi-compact">
          <h3 className="card-title">{KPIIcons.globe} Client Languages</h3>
          <div className="kpi-language-list">
            {data.languages.map((lang, i) => (
              <div key={i} className="kpi-lang-item">
                <span className="kpi-lang-name">{lang.language}</span>
                <div className="kpi-lang-bar-bg">
                  <div
                    className="kpi-lang-bar"
                    style={{
                      width: `${(lang.count / data.totalClients) * 100}%`,
                      backgroundColor: SEGMENT_COLORS[i % SEGMENT_COLORS.length],
                    }}
                  />
                </div>
                <span className="kpi-lang-count">{lang.count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence by Segment */}
        <div className="card kpi-chart-card kpi-compact">
          <h3 className="card-title">{KPIIcons.trending} Avg. Confidence by Segment</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data.confidenceBySegment} margin={{ left: 5, right: 15 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
              <XAxis dataKey="segment" tick={{ fontSize: 11 }} tickFormatter={(v: number) => `Seg ${v}`} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Avg Confidence']} />
              <Bar dataKey="avgConfidence" radius={[4, 4, 0, 0]} barSize={28}>
                {data.confidenceBySegment.map((_, i) => (
                  <Cell key={i} fill={SEGMENT_COLORS[i % SEGMENT_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Uploads */}
        <div className="card kpi-chart-card kpi-compact">
          <h3 className="card-title">{KPIIcons.upload} Recent Uploads</h3>
          {data.recentUploads.length > 0 ? (
            <div className="kpi-upload-list">
              {data.recentUploads.map((u, i) => (
                <div key={i} className="kpi-upload-item">
                  <div className="kpi-upload-info">
                    <span className="kpi-upload-name">{u.filename || 'Voice memo'}</span>
                    <span className="kpi-upload-date">{new Date(u.date).toLocaleDateString()}</span>
                  </div>
                  <div className="kpi-upload-meta">
                    <span className={`kpi-upload-status ${u.status}`}>{u.status}</span>
                    <span className="kpi-upload-records">+{u.recordsAdded}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="kpi-empty">No uploads yet</div>
          )}
        </div>
      </div>
    </div>
  )
}
