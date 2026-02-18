import { useState, useEffect, useCallback } from 'react'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { getReportSummary, exportCSV } from './services/apiService'

// ─── Types ───────────────────────────────────────────────────────
interface SummaryReport {
  generatedAt: string
  totalClients: number
  segments: Array<{
    id: number; name: string; count: number
    avgScore: number; gold: number; silver: number
  }>
  actions: {
    total: number; completed: number
    completionRate: number; highPriority: number
  }
  topConcepts: Array<{ id: string; label: string; mentions: number }>
  tiers: Array<{ tier: string; count: number; avgScore: number }>
  languages: Array<{ language: string; count: number }>
  advisors: Array<{ name: string; clients: number; avgScore: number }>
}

const TIER_COLORS: Record<string, string> = {
  platinum: '#818cf8', gold: '#f59e0b', silver: '#94a3b8', bronze: '#d97706'
}
const PIE_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

// ─── Component ───────────────────────────────────────────────────
export default function ExportPage() {
  const [report, setReport] = useState<SummaryReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState<string | null>(null)

  const fetchReport = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await getReportSummary()
      setReport(data)
    } catch (err) {
      console.error('Failed to fetch report:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchReport() }, [fetchReport])

  // ─── Download Handler ──────────────────────────────────────
  const handleDownload = async (type: string) => {
    setDownloading(type)
    try {
      const blob = await exportCSV(type)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${type}_export.csv`
      document.body.appendChild(a)
      a.click()
      URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error(`Failed to download ${type}:`, err)
    } finally {
      setDownloading(null)
    }
  }

  if (loading) {
    return (
      <div className="exp-loading">
        <div className="exp-spinner" />
        <span>Generating report…</span>
      </div>
    )
  }

  if (!report) {
    return <div className="exp-empty">Failed to load report data.</div>
  }

  return (
    <div className="exp-page">
      {/* Header */}
      <div className="exp-header">
        <div className="exp-header-left">
          <h2 className="exp-title">Reports & Export</h2>
          <p className="exp-subtitle">Executive summary and data exports</p>
        </div>
      </div>

      {/* Export Buttons */}
      <div className="exp-download-row">
        {[
          { type: 'clients', label: 'Client Profiles', desc: 'All clients with scores, segments, advisors' },
          { type: 'scores', label: 'Client Scores', desc: 'Engagement, value, overall scores & tiers' },
          { type: 'actions', label: 'Action Plans', desc: 'Recommended actions by client' },
        ].map(item => (
          <button
            key={item.type}
            className="exp-download-card"
            onClick={() => handleDownload(item.type)}
            disabled={downloading === item.type}
          >
            <span className="exp-dl-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            </span>
            <div className="exp-dl-text">
              <span className="exp-dl-label">{item.label}</span>
              <span className="exp-dl-desc">{item.desc}</span>
            </div>
            <span className="exp-dl-action">
              {downloading === item.type ? 'Exporting…' : 'Download CSV'}
            </span>
          </button>
        ))}
      </div>

      {/* Executive Summary */}
      <div className="exp-section">
        <h3 className="exp-section-title">Executive Summary</h3>

        {/* Top KPIs */}
        <div className="exp-kpi-row">
          <div className="exp-kpi">
            <span className="exp-kpi-value">{report.totalClients.toLocaleString()}</span>
            <span className="exp-kpi-label">Total Clients</span>
          </div>
          <div className="exp-kpi">
            <span className="exp-kpi-value">{report.actions.total.toLocaleString()}</span>
            <span className="exp-kpi-label">Total Actions</span>
          </div>
          <div className="exp-kpi">
            <span className="exp-kpi-value">{report.actions.completionRate}%</span>
            <span className="exp-kpi-label">Completion Rate</span>
          </div>
          <div className="exp-kpi">
            <span className="exp-kpi-value">{report.segments.length}</span>
            <span className="exp-kpi-label">Segments</span>
          </div>
        </div>

        {/* Charts Row 1 */}
        <div className="exp-charts-row">
          {/* Tier Distribution */}
          <div className="exp-chart-card">
            <h4 className="exp-chart-label">Client Tier Distribution</h4>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={report.tiers}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  paddingAngle={4}
                  dataKey="count"
                  nameKey="tier"
                  label={({ tier, percent }) => `${tier} ${(percent * 100).toFixed(0)}%`}
                >
                  {report.tiers.map((t) => (
                    <Cell key={t.tier} fill={TIER_COLORS[t.tier] || '#6366f1'} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Top Concepts */}
          <div className="exp-chart-card">
            <h4 className="exp-chart-label">Top Concepts</h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart
                data={report.topConcepts.slice(0, 8)}
                margin={{ top: 5, right: 5, left: 0, bottom: 5 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                <XAxis type="number" fontSize={11} />
                <YAxis type="category" dataKey="label" fontSize={11} width={120} />
                <Tooltip contentStyle={{ background: 'white', borderRadius: 10, border: '1px solid #e2e8f0' }} />
                <Bar dataKey="mentions" fill="#6366f1" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts Row 2 */}
        <div className="exp-charts-row">
          {/* Language Breakdown */}
          <div className="exp-chart-card">
            <h4 className="exp-chart-label">Language Distribution</h4>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={report.languages.slice(0, 8)}
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="count"
                  nameKey="language"
                  label={({ language, percent }) => `${language} ${(percent * 100).toFixed(0)}%`}
                >
                  {report.languages.slice(0, 8).map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Advisor Workload */}
          <div className="exp-chart-card">
            <h4 className="exp-chart-label">Advisor Workload</h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart
                data={report.advisors.filter(a => a.clients > 0)}
                margin={{ top: 5, right: 5, left: 0, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                <XAxis dataKey="name" fontSize={11} tickFormatter={n => n.split(' ')[0]} />
                <YAxis fontSize={11} />
                <Tooltip contentStyle={{ background: 'white', borderRadius: 10, border: '1px solid #e2e8f0' }} />
                <Legend />
                <Bar dataKey="clients" fill="#6366f1" name="Clients" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Segment Table */}
        <div className="exp-table-card">
          <h4 className="exp-chart-label">Segment Performance</h4>
          <div className="exp-table-wrapper">
            <table className="exp-table">
              <thead>
                <tr>
                  <th>Segment</th>
                  <th>Clients</th>
                  <th>Avg Score</th>
                  <th>Gold</th>
                  <th>Silver</th>
                </tr>
              </thead>
              <tbody>
                {report.segments.map(seg => (
                  <tr key={seg.id}>
                    <td className="exp-td-name">{seg.name || `Segment ${seg.id}`}</td>
                    <td>{seg.count}</td>
                    <td>
                      <span className="exp-score-pill" style={{
                        background: seg.avgScore >= 60 ? 'rgba(16,185,129,0.1)' :
                          seg.avgScore >= 40 ? 'rgba(245,158,11,0.1)' : 'rgba(239,68,68,0.1)',
                        color: seg.avgScore >= 60 ? '#059669' :
                          seg.avgScore >= 40 ? '#d97706' : '#dc2626',
                      }}>
                        {seg.avgScore}
                      </span>
                    </td>
                    <td>{seg.gold}</td>
                    <td>{seg.silver}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
