import { useState, useEffect, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import { getAdvisorWorkload, getAdvisors as fetchAdvisorsService, autoAssignAdvisors } from './services/apiService'

// ─── Types ───────────────────────────────────────────────────────
interface Advisor {
  id: number
  username: string
  displayName: string
  role: string
  clientCount: number
}

interface AdvisorWorkload {
  id: number
  name: string
  role: string
  totalClients: number
  platinum: number
  gold: number
  silver: number
  bronze: number
  avgScore: number
}

interface WorkloadData {
  advisors: AdvisorWorkload[]
  unassignedCount: number
}

interface AdvisorsPageProps {
  userId?: number
}

// ─── Helpers ─────────────────────────────────────────────────────
const TIER_COLORS: Record<string, string> = {
  platinum: '#818cf8',
  gold: '#f59e0b',
  silver: '#94a3b8',
  bronze: '#d97706',
}

const roleIcons: Record<string, string> = {
  sales: '💼', manager: '👔', admin: '🛡️', viewer: '👁️',
}

const PIE_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

// ─── Component ───────────────────────────────────────────────────
export default function AdvisorsPage({ userId }: AdvisorsPageProps) {
  const [workload, setWorkload] = useState<WorkloadData | null>(null)
  const [advisors, setAdvisors] = useState<Advisor[]>([])
  const [loading, setLoading] = useState(true)
  const [assigning, setAssigning] = useState(false)
  const [assignResult, setAssignResult] = useState<any>(null)

  // ─── Fetch ──────────────────────────────────────────────────
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [wResult, aResult] = await Promise.all([
        getAdvisorWorkload(),
        fetchAdvisorsService(),
      ])
      setWorkload(wResult.data)
      setAdvisors(aResult.data)
    } catch (err) {
      console.error('Failed to fetch advisor data:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])

  // ─── Auto-Assign ───────────────────────────────────────────
  const handleAutoAssign = async (strategy: string) => {
    setAssigning(true)
    setAssignResult(null)
    try {
      const result = await autoAssignAdvisors({ strategy })
      setAssignResult(result)
      fetchData()
    } catch (err) {
      console.error('Failed to auto-assign:', err)
    } finally {
      setAssigning(false)
    }
  }

  // ─── Derived Data ──────────────────────────────────────────
  const activeAdvisors = workload?.advisors.filter(a => a.totalClients > 0) || []
  const totalManaged = activeAdvisors.reduce((s, a) => s + a.totalClients, 0)
  const avgPerAdvisor = activeAdvisors.length > 0
    ? Math.round(totalManaged / activeAdvisors.length)
    : 0

  // Bar chart data for workload
  const barData = workload?.advisors
    .filter(a => a.totalClients > 0)
    .map(a => ({
      name: a.name.split(' ')[0],
      gold: a.gold,
      silver: a.silver,
      bronze: a.bronze,
      total: a.totalClients,
    })) || []

  // Pie chart data for distribution
  const pieData = workload?.advisors
    .filter(a => a.totalClients > 0)
    .map(a => ({
      name: a.name,
      value: a.totalClients,
    })) || []

  // ─── Render ─────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="adv-loading">
        <div className="adv-spinner" />
        <span>Loading advisor data…</span>
      </div>
    )
  }

  return (
    <div className="adv-page">
      {/* Header */}
      <div className="adv-header">
        <div className="adv-header-left">
          <h2 className="adv-title">👔 Advisor Assignment</h2>
          <p className="adv-subtitle">
            Manage client-advisor relationships and balance workloads.
          </p>
        </div>
        <div className="adv-header-actions">
          {(workload?.unassignedCount ?? 0) > 0 && (
            <>
              <button
                className="adv-btn adv-btn-primary"
                onClick={() => handleAutoAssign('round_robin')}
                disabled={assigning}
              >
                {assigning ? 'Assigning…' : '⚡ Round-Robin Assign'}
              </button>
              <button
                className="adv-btn adv-btn-secondary"
                onClick={() => handleAutoAssign('segment')}
                disabled={assigning}
              >
                📊 By Segment
              </button>
            </>
          )}
        </div>
      </div>

      {/* Assign result toast */}
      {assignResult && (
        <div className="adv-toast">
          ✅ Assigned <strong>{assignResult.assigned}</strong> clients
          using <strong>{assignResult.strategy}</strong> strategy
          <button onClick={() => setAssignResult(null)}>✕</button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="adv-kpi-row">
        <div className="adv-kpi-card">
          <span className="adv-kpi-value">{advisors.length}</span>
          <span className="adv-kpi-label">Total Advisors</span>
        </div>
        <div className="adv-kpi-card">
          <span className="adv-kpi-value">{totalManaged.toLocaleString()}</span>
          <span className="adv-kpi-label">Clients Managed</span>
        </div>
        <div className="adv-kpi-card">
          <span className="adv-kpi-value">{(workload?.unassignedCount ?? 0).toLocaleString()}</span>
          <span className="adv-kpi-label">Unassigned</span>
        </div>
        <div className="adv-kpi-card">
          <span className="adv-kpi-value">{avgPerAdvisor.toLocaleString()}</span>
          <span className="adv-kpi-label">Avg per Advisor</span>
        </div>
      </div>

      {/* Charts Row */}
      <div className="adv-charts-row">
        {/* Workload Bar Chart */}
        <div className="adv-chart-card">
          <h3 className="adv-chart-title">Workload by Tier</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis dataKey="name" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip
                contentStyle={{ background: 'white', borderRadius: 10, border: '1px solid #e2e8f0' }}
              />
              <Legend />
              <Bar dataKey="gold" stackId="a" fill={TIER_COLORS.gold} name="Gold" radius={[0, 0, 0, 0]} />
              <Bar dataKey="silver" stackId="a" fill={TIER_COLORS.silver} name="Silver" />
              <Bar dataKey="bronze" stackId="a" fill={TIER_COLORS.bronze} name="Bronze" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Distribution Pie Chart */}
        <div className="adv-chart-card">
          <h3 className="adv-chart-title">Client Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={95}
                paddingAngle={3}
                dataKey="value"
                label={({ name, percent }) => `${name.split(' ')[0]} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Advisor Cards */}
      <h3 className="adv-section-title">Advisor Portfolio</h3>
      <div className="adv-grid">
        {workload?.advisors
          .sort((a, b) => b.totalClients - a.totalClients)
          .map(advisor => (
            <div key={advisor.id} className="adv-card">
              <div className="adv-card-header">
                <div className="adv-avatar">
                  {advisor.name.split(' ').map(n => n[0]).join('')}
                </div>
                <div className="adv-card-identity">
                  <h4 className="adv-card-name">{advisor.name}</h4>
                  <span className="adv-card-role">
                    {roleIcons[advisor.role] || '👤'} {advisor.role}
                  </span>
                </div>
                <div className="adv-card-total">
                  <span className="adv-total-num">{advisor.totalClients.toLocaleString()}</span>
                  <span className="adv-total-label">clients</span>
                </div>
              </div>

              {advisor.totalClients > 0 && (
                <div className="adv-card-body">
                  <div className="adv-tier-bars">
                    {(['gold', 'silver', 'bronze'] as const).map(tier => {
                      const count = advisor[tier]
                      const pct = advisor.totalClients > 0
                        ? (count / advisor.totalClients) * 100
                        : 0
                      return (
                        <div key={tier} className="adv-tier-row">
                          <span className="adv-tier-label">{tier}</span>
                          <div className="adv-tier-bar-bg">
                            <div
                              className="adv-tier-bar-fill"
                              style={{
                                width: `${pct}%`,
                                background: TIER_COLORS[tier],
                              }}
                            />
                          </div>
                          <span className="adv-tier-count">{count}</span>
                        </div>
                      )
                    })}
                  </div>
                  <div className="adv-card-score">
                    <span className="adv-score-label">Avg Score</span>
                    <span className="adv-score-value">{advisor.avgScore}</span>
                  </div>
                </div>
              )}

              {advisor.totalClients === 0 && (
                <div className="adv-card-empty">
                  No clients assigned
                </div>
              )}
            </div>
          ))}
      </div>
    </div>
  )
}
