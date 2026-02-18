import { useState, useMemo, useEffect, Key } from 'react'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, Tooltip, Legend
} from 'recharts'
import Plot from 'react-plotly.js'
import Graph from "react-graph-vis";
import HeatMapGrid from "react-heatmap-grid";
import API_CONFIG from './config'
import { getData, invalidateCache, completeAction } from './services/apiService'
import FileUpload from './FileUpload'
import VoiceRecorder from './VoiceRecorder'
import { useAuth } from './auth/AuthContext'
import LoginScreen from './auth/LoginScreen'
import CalendarPage from './CalendarPage'
import Client360Page from './Client360Page'
import KPIDashboard from './KPIDashboard'
import PlaybooksPage from './PlaybooksPage'
import AdvisorsPage from './AdvisorsPage'
import ExportPage from './ExportPage'
import './App.css'

// Type Definitions
interface ProcessingInfo {
  timestamp?: string;
  totalRecords?: number;
  totalConcepts?: number;
  totalSegments?: number;
  dashboardGenTime?: number;
  pipelineTimings?: Record<string, number>;
}

interface DashboardData {
  segments?: Segment[];
  radar?: RadarData[];
  clients?: Client[];
  scatter3d?: Scatter3DPoint[];
  concepts?: Concept[];
  heatmap?: HeatmapData[];
  metrics?: { clients: number; segments: number };
  processingInfo?: ProcessingInfo;
}

interface Segment {
  name: string;
  value: number;
  profile: string;
  fullProfile: string;
}

interface RadarData {
  subject: string;
  [key: string]: string | number;
}

interface ClientAction {
  actionId: string;
  title: string;
  channel: string;
  priority: string;
  kpi: string;
  triggers: string;
  rationale: string;
}

interface Client {
  id: string;
  segment: number;
  topConcepts?: string[];
  fullText?: string;
  language?: string;
  date?: string;
  confidence?: number;
  actions?: ClientAction[];
}

interface Scatter3DPoint {
  x: number;
  y: number;
  z: number;
  client: string;
  segment: number;
  text: string;
  cluster?: number;
  profile?: string;
  id?: string;
  confidence?: number;
  topConcepts?: string[];
  originalNote?: string;
}

interface Concept {
  concept: string;
  count: number;
  clients: string[];
}

interface HeatmapData {
  segment: number;
  concept: string;
  value: number;
}

// Color palette
const COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#3b82f6', '#06b6d4', '#10b981', '#84cc16']
const SEGMENT_COLORS = {
  0: '#6366f1', 1: '#8b5cf6', 2: '#ec4899', 3: '#f43f5e',
  4: '#3b82f6', 5: '#06b6d4', 6: '#10b981', 7: '#84cc16'
}

// Icons as SVG components
const Icons = {
  actions: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>,
  segments: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0110 10M12 2v10l7.07 7.07"/></svg>,
  clients: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75"/></svg>,
  chart: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>,
  search: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>,
  chevron: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18l6-6-6-6"/></svg>,
  close: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>,
  user: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
  tag: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>,
  calendar: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>,
  globe: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>,
  cube: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>,
  share: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>,
  barchart: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>,
  grid: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>,
  settings: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>,
  timer: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>,
}

// Navigation Component - Modern Top Navigation
const Navigation = ({ activePage, setActivePage, data, onClearClient }: { 
  activePage: string; 
  setActivePage: (page: string) => void; 
  data: DashboardData | null;
  onClearClient?: () => void;
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const { user, logout } = useAuth()
  
  const pages = [
    { id: 'dashboard', label: 'Dashboard', icon: Icons.barchart },
    { id: 'clients', label: 'Clients', icon: Icons.clients },
    { id: 'segments', label: 'Segments', icon: Icons.segments },
    { id: 'actions', label: 'Actions', icon: Icons.actions },
    { id: 'data', label: 'Analytics', icon: Icons.chart },
    { id: 'calendar', label: 'Calendar', icon: Icons.calendar },
    { id: 'playbooks', label: 'Playbooks', icon: Icons.grid },
    { id: 'advisors', label: 'Advisors', icon: Icons.user },
    { id: 'reports', label: 'Reports', icon: Icons.share },
    { id: 'upload', label: 'Import', icon: Icons.tag },
    ...((['admin', 'data-scientist', 'data-analyst'] as string[]).includes(user?.role || '') ? [{ id: 'admin', label: 'Admin', icon: Icons.settings }] : []),
  ]

  const handleNavClick = (pageId: string) => {
    setActivePage(pageId)
    setMobileMenuOpen(false)
    onClearClient?.()
  }

  const initials = user?.display_name
    ?.split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || user?.username?.slice(0, 2).toUpperCase() || '??'

  return (
    <header className="topnav">
      <div className="topnav-container">
        <div className="topnav-brand">
          <span className="topnav-logo">LVMH</span>
          <span className="topnav-divider"></span>
          <span className="topnav-subtitle">Client Intelligence</span>
        </div>
        
        <button 
          className="mobile-menu-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? Icons.close : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="6" x2="21" y2="6"/>
              <line x1="3" y1="12" x2="21" y2="12"/>
              <line x1="3" y1="18" x2="21" y2="18"/>
            </svg>
          )}
        </button>

        <nav className={`topnav-links ${mobileMenuOpen ? 'open' : ''}`}>
          {pages.map(page => (
            <button
              key={page.id}
              className={`topnav-link ${activePage === page.id ? 'active' : ''}`}
              onClick={() => handleNavClick(page.id)}
            >
              <span className="topnav-icon">{page.icon}</span>
              <span>{page.label}</span>
            </button>
          ))}
        </nav>

        <div className="topnav-metrics">
          <div className="topnav-metric">
            <span className="topnav-metric-value">{data?.metrics?.clients || 0}</span>
            <span className="topnav-metric-label">Clients</span>
          </div>
          <div className="topnav-metric">
            <span className="topnav-metric-value">{data?.metrics?.segments || 0}</span>
            <span className="topnav-metric-label">Segments</span>
          </div>
          {data?.processingInfo?.pipelineTimings?.total && (
            <div className="topnav-metric accent">
              <span className="topnav-metric-value">{data.processingInfo.pipelineTimings.total}s</span>
              <span className="topnav-metric-label">Pipeline</span>
            </div>
          )}
        </div>

        {/* User Menu */}
        {user && (
          <div className="user-menu">
            <button
              className="user-menu-trigger"
              onClick={() => setUserMenuOpen(!userMenuOpen)}
            >
              <div className={`user-avatar-small ${user.role}`}>{initials}</div>
              <span className="user-name">{user.display_name}</span>
              <span className={`user-role-badge ${user.role}`}>{user.role}</span>
            </button>

            {userMenuOpen && (
              <>
                <div
                  style={{ position: 'fixed', inset: 0, zIndex: 99 }}
                  onClick={() => setUserMenuOpen(false)}
                />
                <div className="user-dropdown">
                  <div className="user-dropdown-header">
                    <div className="user-full-name">{user.display_name}</div>
                    <div className="user-email">{user.email || user.username}</div>
                  </div>
                  <button
                    className="user-dropdown-item danger"
                    onClick={() => { setUserMenuOpen(false); logout(); }}
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>
                    </svg>
                    Sign Out
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </header>
  )
}

// Actions Page - Redesigned for clarity
const ActionsPage = ({ data, onClientClick, onRefresh }: { data: DashboardData | null; onClientClick?: (id: string) => void; onRefresh?: () => void }) => {
  const [filter, setFilter] = useState('all')
  const [selectedAction, setSelectedAction] = useState<any>(null)
  const [completing, setCompleting] = useState<string | null>(null)
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set())

  // Channel → icon mapping
  const channelIcons: Record<string, React.ReactNode> = {
    'event': Icons.calendar,
    'crm': Icons.clients,
    'email': Icons.actions,
    'client service': Icons.user,
  }

  const actions = useMemo(() => {
    if (!data?.clients) return []
    
    const actionList: any[] = []
    
    data.clients.forEach(client => {
      (client.actions || []).forEach((act, idx) => {
        const prio = (act.priority || 'low').toLowerCase()
        const channelKey = (act.channel || '').toLowerCase()
        actionList.push({
          id: `${client.id}-${act.actionId}-${idx}`,
          actionId: act.actionId,
          client,
          title: act.title,
          channel: act.channel,
          priority: prio,
          kpi: act.kpi,
          triggers: act.triggers,
          rationale: act.rationale,
          icon: channelIcons[channelKey] || Icons.actions,
        })
      })
    })
    
    // Sort: high → medium → low, then by confidence
    const order: Record<string, number> = { high: 0, medium: 1, low: 2 }
    return actionList.sort((a, b) => {
      const pDiff = (order[a.priority] ?? 3) - (order[b.priority] ?? 3)
      if (pDiff !== 0) return pDiff
      return (b.client.confidence || 0) - (a.client.confidence || 0)
    })
  }, [data])

  const filteredActions = filter === 'all' ? actions : actions.filter(a => a.priority === filter)
  const highCount = useMemo(() => actions.filter(a => a.priority === 'high').length, [actions])
  const mediumCount = useMemo(() => actions.filter(a => a.priority === 'medium').length, [actions])
  const lowCount = useMemo(() => actions.filter(a => a.priority === 'low').length, [actions])

  const uniqueClients = useMemo(() => new Set(actions.map(a => a.client.id)).size, [actions])

  const handleActionClick = (action: any) => {
    setSelectedAction(action)
  }

  const handleComplete = async (action: any) => {
    const key = `${action.client.id}-${action.actionId}`
    setCompleting(key)
    try {
      await completeAction(action.client.id, action.actionId)
      setCompletedIds(prev => new Set(prev).add(key))
      setSelectedAction(null)
      onRefresh?.()
    } catch (err) {
      console.error('Failed to complete action:', err)
    } finally {
      setCompleting(null)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Recommended Actions</h1>
          <p>{actions.length} actions for {uniqueClients} clients — prioritize your interactions.</p>
        </div>
        <div className="filter-group">
          <button className={`filter-btn ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>
            All ({actions.length})
          </button>
          <button className={`filter-btn high ${filter === 'high' ? 'active' : ''}`} onClick={() => setFilter('high')}>
            Urgent ({highCount})
          </button>
          <button className={`filter-btn medium ${filter === 'medium' ? 'active' : ''}`} onClick={() => setFilter('medium')}>
            Medium ({mediumCount})
          </button>
          <button className={`filter-btn ${filter === 'low' ? 'active' : ''}`} onClick={() => setFilter('low')}>
            Low ({lowCount})
          </button>
        </div>
      </div>

      <div className="stats-row">
        <div className="stat-card accent-red">
          <div className="stat-icon">{Icons.actions}</div>
          <div className="stat-content">
            <div className="stat-value">{highCount}</div>
            <div className="stat-label">Urgent Actions</div>
          </div>
        </div>
        <div className="stat-card accent-blue">
          <div className="stat-icon">{Icons.calendar}</div>
          <div className="stat-content">
            <div className="stat-value">{actions.filter(a => (a.channel || '').toLowerCase() === 'event').length}</div>
            <div className="stat-label">Events</div>
          </div>
        </div>
        <div className="stat-card accent-green">
          <div className="stat-icon">{Icons.tag}</div>
          <div className="stat-content">
            <div className="stat-value">{actions.filter(a => (a.channel || '').toLowerCase() === 'crm').length}</div>
            <div className="stat-label">Actions CRM</div>
          </div>
        </div>
        <div className="stat-card accent-purple">
          <div className="stat-icon">{Icons.user}</div>
          <div className="stat-content">
            <div className="stat-value">{uniqueClients}</div>
            <div className="stat-label">Active Clients</div>
          </div>
        </div>
      </div>

      <div className="actions-layout">
        <div className="card action-list-card">
          <div className="action-list">
            {filteredActions.map(action => (
              <div 
                key={action.id} 
                className={`action-item ${selectedAction?.id === action.id ? 'selected' : ''}`}
                onClick={() => handleActionClick(action)}
              >
                <div className="action-priority">
                  <span className={`priority-dot ${action.priority}`}></span>
                </div>
                <div className="action-icon">{action.icon}</div>
                <div className="action-content">
                  <div className="action-header">
                    <span className="action-client">{action.client.id}</span>
                    <span className="action-segment" style={{ backgroundColor: SEGMENT_COLORS[action.client.segment as keyof typeof SEGMENT_COLORS] }}>
                      Seg {action.client.segment}
                    </span>
                    <span className="action-channel-tag">{action.channel}</span>
                  </div>
                  <div className="action-label">{action.title}</div>
                </div>
                <div className="action-confidence">
                  <span>{((action.client.confidence || 0) * 100).toFixed(0)}%</span>
                </div>
                <div className="action-chevron">{Icons.chevron}</div>
              </div>
            ))}
          </div>
        </div>
        
        {selectedAction ? (
          <div className="card action-detail-card">
            <div className="action-detail-header">
              <h3>Action Detail</h3>
              <button className="close-btn" onClick={() => setSelectedAction(null)}>{Icons.close}</button>
            </div>
            <div className="action-detail-content">
              <div className="client-summary">
                <div className="client-avatar" style={{ backgroundColor: SEGMENT_COLORS[selectedAction.client.segment as keyof typeof SEGMENT_COLORS] }}>
                  {selectedAction.client.id.slice(-2)}
                </div>
                <div className="client-info">
                  <div className="client-id">{selectedAction.client.id}</div>
                  <div className="client-segment">Segment {selectedAction.client.segment}</div>
                </div>
              </div>
              
              <div className="detail-item">
                <span className="detail-label">Action</span>
                <span className="detail-value">{selectedAction.title}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Channel</span>
                <span className="detail-value">{selectedAction.channel}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Priority</span>
                <span className={`detail-value priority-tag ${selectedAction.priority}`}>{selectedAction.priority}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">KPI</span>
                <span className="detail-value">{selectedAction.kpi}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Confidence</span>
                <span className="detail-value">{((selectedAction.client.confidence || 0) * 100).toFixed(0)}%</span>
              </div>

              <div className="detail-item detail-rationale">
                <span className="detail-label">Recommendation</span>
                <span className="detail-value">{selectedAction.rationale}</span>
              </div>

              <div className="detail-item">
                <span className="detail-label">Triggers</span>
                <div className="trigger-tags">
                  {(selectedAction.triggers || '').split('|').map((t: string, i: number) => (
                    <span key={i} className="concept-badge" style={{ backgroundColor: '#f3f4f6', color: '#374151' }}>{t.trim()}</span>
                  ))}
                </div>
              </div>

              <div className="client-concepts">
                <h4>Client's Top Concepts</h4>
                {selectedAction.client.topConcepts?.map((c: string, i: number) => (
                  <span key={i} className="concept-badge" style={{ backgroundColor: SEGMENT_COLORS[selectedAction.client.segment as keyof typeof SEGMENT_COLORS] + '20', color: SEGMENT_COLORS[selectedAction.client.segment as keyof typeof SEGMENT_COLORS] }}>{c}</span>
                ))}
              </div>

              {selectedAction.client.originalNote && (
                <div className="client-note">
                  <h4>Original Note</h4>
                  <p>{selectedAction.client.originalNote}</p>
                </div>
              )}

              <div className="action-detail-buttons">
                {!completedIds.has(`${selectedAction.client.id}-${selectedAction.actionId}`) ? (
                  <button
                    className="action-complete-btn"
                    onClick={() => handleComplete(selectedAction)}
                    disabled={completing === `${selectedAction.client.id}-${selectedAction.actionId}`}
                  >
                    {completing === `${selectedAction.client.id}-${selectedAction.actionId}` ? 'Completing…' : 'Mark as Done'}
                  </button>
                ) : (
                  <span className="action-done-badge">Completed</span>
                )}
                {onClientClick && (
                  <button
                    className="action-view-client-btn"
                    onClick={() => onClientClick(selectedAction.client.id)}
                  >
                    View Client 360 →
                  </button>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="card action-detail-placeholder">
            <div className="placeholder-content">
              {Icons.actions}
              <h4>Select an Action</h4>
              <p>Click an action from the list to see client details and the recommendation.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Segments Page - Redesigned for clarity
const SegmentsPage = ({ data, onClientClick }: { data: DashboardData | null; onClientClick?: (id: string) => void }) => {
  const [selectedSegment, setSelectedSegment] = useState<number | null>(0) // Select first segment by default
  const segmentData = data?.segments || []
  const radarData = data?.radar || []
  const clients = data?.clients || []

  const selectedSegmentData = useMemo(() => {
    if (selectedSegment === null) return null
    return segmentData[selectedSegment]
  }, [selectedSegment, segmentData])

  const clientsInSegment = useMemo(() => {
    if (selectedSegment === null) return []
    return clients.filter(c => c.segment === selectedSegment).slice(0, 10)
  }, [selectedSegment, clients])

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Segment Analysis</h1>
          <p>Explore {segmentData.length} client profiles identified by the AI.</p>
        </div>
      </div>

      <div className="segments-layout">
        {/* Left Column: Segment List */}
        <div className="card segments-list-card">
          <h3 className="card-title">Client Profiles</h3>
          <div className="segments-list">
            {segmentData.map((segment, idx) => (
              <div 
                key={segment.name}
                className={`segment-list-item ${selectedSegment === idx ? 'selected' : ''}`}
                onClick={() => setSelectedSegment(idx)}
                style={{ '--segment-color': COLORS[idx % COLORS.length] }}
              >
                <div className="segment-item-header">
                  <div className="segment-item-number">{idx}</div>
                  <div className="segment-item-profile">{segment.profile}</div>
                </div>
                <div className="segment-item-footer">
                  <div className="segment-item-count">{segment.value} clients</div>
                  <div className="segment-item-bar">
                    <div 
                      className="segment-item-bar-fill" 
                      style={{ width: `${(segment.value / Math.max(...segmentData.map(s => s.value))) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Details and Charts */}
        <div className="segments-detail-column">
          {selectedSegmentData ? (
            <div className="card segment-detail-card">
              <div className="segment-detail-header">
                <h3 className="card-title">Segment {selectedSegment}</h3>
                <div className="segment-detail-pills">
                  <span className="pill">{selectedSegmentData.value} clients</span>
                  <span className="pill">{selectedSegmentData.fullProfile.split(' | ').length} key concepts</span>
                </div>
              </div>
              <p className="segment-full-profile">{selectedSegmentData.fullProfile}</p>
              
              <h4>Representative Clients</h4>
              <div className="representative-clients">
                {clientsInSegment.map(client => (
                  <div key={client.id} className="client-chip" style={{ cursor: onClientClick ? 'pointer' : 'default' }} onClick={() => onClientClick?.(client.id)}>
                    <div className="client-chip-avatar" style={{ backgroundColor: COLORS[client.segment % COLORS.length] }}>
                      {client.id.slice(-2)}
                    </div>
                    <span>{client.id}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="card placeholder-card">
              <p>Select a segment to see details.</p>
            </div>
          )}

          <div className="charts-row">
            <div className="card">
              <h3 className="card-title">Segment Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie 
                    data={segmentData} 
                    cx="50%" 
                    cy="50%" 
                    innerRadius={60} 
                    outerRadius={100} 
                    paddingAngle={3} 
                    dataKey="value"
                    label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {segmentData.map((entry, index) => (
                      <Cell 
                        key={entry.name} 
                        fill={COLORS[index % COLORS.length]} 
                        opacity={selectedSegment === null || selectedSegment === index ? 1 : 0.3}
                      />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)', boxShadow: 'var(--shadow-lg)' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3 className="card-title">Radar Profile</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                  <PolarGrid stroke="var(--border-light)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} tickCount={4} />
                  {segmentData.map((seg, i) => (
                    <Radar 
                      key={i} 
                      name={`Segment ${i}`} 
                      dataKey={`seg${i}`} 
                      stroke={COLORS[i]} 
                      fill={COLORS[i]} 
                      fillOpacity={selectedSegment === i ? 0.3 : (selectedSegment === null ? 0.15 : 0.05)}
                      strokeWidth={selectedSegment === i ? 2.5 : 1.5}
                      strokeOpacity={selectedSegment === null || selectedSegment === i ? 1 : 0.4}
                    />
                  ))}
                  <Tooltip contentStyle={{ borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)', boxShadow: 'var(--shadow-lg)' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Clients Page - layout cartes segments comme la maquette
const segmentColors: Record<number, string> = {
  0: "#1d4ed8",
  1: "#16a34a",
  2: "#f97316",
  3: "#dc2626",
  4: "#7c3aed",
  5: "#0891b2",
  6: "#eab308",
  7: "#db2777",
}

const ClientsPage = ({ data, onClientClick }: { data: DashboardData | null; onClientClick?: (id: string) => void }) => {
  const [searchQuery, setSearchQuery] = useState("")
  const [filterSegment, setFilterSegment] = useState<string>("all")
  const [sortOrder, setSortOrder] = useState<string>("confidence_desc")
  const [currentPage, setCurrentPage] = useState(1)
  const PAGE_SIZE = 24

  const clients = data?.clients || []
  const segments = data?.segments || []

  const filteredAndSortedClients = useMemo(() => {
    let filtered = clients
    
    if (filterSegment !== "all") {
      filtered = filtered.filter(c => c.segment === parseInt(filterSegment))
    }

    if (searchQuery) {
      filtered = filtered.filter((client) =>
        client.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        client.topConcepts?.some((c) =>
          c.toLowerCase().includes(searchQuery.toLowerCase())
        ) ||
        client.fullText?.toLowerCase().includes(searchQuery.toLowerCase())
      )
    }

    return filtered.sort((a, b) => {
      const aConf = (a.confidence || 0)
      const bConf = (b.confidence || 0)
      switch (sortOrder) {
        case 'confidence_asc': return aConf - bConf
        case 'id_asc': return a.id.localeCompare(b.id)
        case 'id_desc': return b.id.localeCompare(a.id)
        default: return bConf - aConf
      }
    })
  }, [clients, searchQuery, filterSegment, sortOrder])

  // Reset to page 1 when filters change
  useEffect(() => { setCurrentPage(1) }, [searchQuery, filterSegment, sortOrder])

  const totalPages = Math.ceil(filteredAndSortedClients.length / PAGE_SIZE)
  const paginatedClients = filteredAndSortedClients.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE,
  )

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Client Explorer</h1>
          <p>Showing {filteredAndSortedClients.length} of {clients.length} clients</p>
        </div>
        <div className="search-group">
          <div className="search-input-wrapper">
            <span className="search-icon">{Icons.search}</span>
            <input
              type="text"
              placeholder="Search by ID, concept, or note..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
          </div>
          <select
            className="segment-select"
            value={filterSegment}
            onChange={(e) => setFilterSegment(e.target.value)}
          >
            <option value="all">All Segments</option>
            {segments.map((s, idx) => (
              <option key={s.name} value={idx.toString()}>Segment {idx} — {s.profile} ({s.value})</option>
            ))}
          </select>
          <select
            className="segment-select"
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
          >
            <option value="confidence_desc">Confidence ↓</option>
            <option value="confidence_asc">Confidence ↑</option>
            <option value="id_asc">Client ID (A→Z)</option>
            <option value="id_desc">Client ID (Z→A)</option>
          </select>
        </div>
      </header>

      <div className="clients-grid">
        {paginatedClients.map((client) => {
          const similarity = Math.round((client as any).confidence ? (client as any).confidence * 100 : 0)
          const color = SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS] ?? "#18181b"

          return (
            <div
              key={client.id}
              className="client-card"
              style={{ '--segment-color': color, cursor: 'pointer' } as any}
              onClick={() => onClientClick?.(client.id)}
            >
              <div className="client-header">
                <div
                  className="client-avatar"
                  style={{ backgroundColor: color }}
                >
                  {client.id.slice(-2)}
                </div>

                <div className="client-info">
                  <div className="client-id">{client.id}</div>
                  <div className="client-segment">Segment {client.segment} - {segments[client.segment]?.profile}</div>
                </div>

                <div className="client-confidence">
                  <div className="confidence-ring" style={{ "--confidence": similarity / 100, '--ring-color': color } as any}>
                    <span>{similarity}%</span>
                  </div>
                  <span className="confidence-label">match</span>
                </div>
              </div>

              <div className="client-concepts">
                {client.topConcepts?.slice(0, 5).map((concept, idx) => (
                  <span key={idx} className="concept-badge" style={{ backgroundColor: color + '15', color: color, borderLeft: `2px solid ${color}` }}>
                    {concept}
                  </span>
                ))}
              </div>

              {client.fullText && (
                <div className="client-text">
                  <p className="client-note-text">{client.fullText.substring(0, 160)}{client.fullText.length > 160 ? '…' : ''}</p>
                </div>
              )}

              <div className="client-footer">
                {client.language && (
                  <span className="client-meta">
                    {Icons.globe}
                    {client.language.toUpperCase()}
                  </span>
                )}
                {client.date && (
                  <span className="client-meta">
                    {Icons.calendar}
                    {new Date(client.date).toLocaleDateString()}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination-controls">
          <button
            className="pagination-btn"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(p => p - 1)}
          >
            ← Prev
          </button>
          <div className="pagination-pages">
            {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
              let pageNum: number
              if (totalPages <= 7) {
                pageNum = i + 1
              } else if (currentPage <= 4) {
                pageNum = i + 1
              } else if (currentPage >= totalPages - 3) {
                pageNum = totalPages - 6 + i
              } else {
                pageNum = currentPage - 3 + i
              }
              return (
                <button
                  key={pageNum}
                  className={`pagination-page ${currentPage === pageNum ? 'active' : ''}`}
                  onClick={() => setCurrentPage(pageNum)}
                >
                  {pageNum}
                </button>
              )
            })}
          </div>
          <button
            className="pagination-btn"
            disabled={currentPage === totalPages}
            onClick={() => setCurrentPage(p => p + 1)}
          >
            Next →
          </button>
          <span className="pagination-info">
            {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, filteredAndSortedClients.length)} of {filteredAndSortedClients.length}
          </span>
        </div>
      )}
    </div>
  )
}

// Data Page - Redesigned for clarity and better UX
const DataPage = ({ data, onClientClick }: { data: DashboardData | null; onClientClick?: (id: string) => void }) => {
  const [view, setView] = useState('3d')
  const [selectedPoint, setSelectedPoint] = useState<any>(null)
  const [selectedConcept, setSelectedConcept] = useState<string | null>(null)
  const [selectedHeatmapCell, setSelectedHeatmapCell] = useState<any>(null)
  const [highlightSegment, setHighlightSegment] = useState<number | null>(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [selectedKGClient, setSelectedKGClient] = useState<string | null>(data?.clients[0]?.id || null)
  const [kgDepth, setKgDepth] = useState(2)
  
  const scatter3d = data?.scatter3d || []
  const concepts = data?.concepts || []
  const heatmap = data?.heatmap || []
  const clients = data?.clients || []

  // Build Knowledge Graph data for selected client
  const knowledgeGraphData = useMemo(() => {
    if (!selectedKGClient) return null
    
    const client = clients.find((c: Client) => c.id === selectedKGClient)
    if (!client) return null
    
    const nodes: any[] = []
    const edges: any[] = []
    
    // Central client node
    nodes.push({
      id: client.id,
      label: client.id,
      size: 30,
      color: SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS],
      font: { color: '#fff', strokeWidth: 3, strokeColor: '#222' },
    })
    
    // Add concept nodes
    const clientConcepts = client.topConcepts || []
    clientConcepts.forEach((concept) => {
      const conceptId = `concept-${concept}`
      if (!nodes.find(n => n.id === conceptId)) {
        nodes.push({
          id: conceptId,
          label: concept,
          size: 20,
          color: '#64748b',
          font: { color: '#fff', strokeWidth: 3, strokeColor: '#222' },
        })
      }
      edges.push({ from: client.id, to: conceptId })
      
      // If depth >= 2, find other clients with same concept
      if (kgDepth >= 2) {
        const relatedClients = clients.filter((c: Client) => 
          c.id !== client.id && 
          c.topConcepts?.includes(concept)
        ).slice(0, 3) // Limit to 3 per concept
        
        relatedClients.forEach((rc: Client) => {
          if (!nodes.find(n => n.id === rc.id)) {
            nodes.push({
              id: rc.id,
              label: rc.id,
              size: 18,
              color: SEGMENT_COLORS[rc.segment as keyof typeof SEGMENT_COLORS],
              font: { color: '#fff', strokeWidth: 3, strokeColor: '#222' },
            })
          }
          if (!edges.find((e: any) => e.from === conceptId && e.to === rc.id)) {
            edges.push({ from: conceptId, to: rc.id })
          }
        })
      }
    })
    
    // Add segment node
    const segmentId = `segment-${client.segment}`
    if (!nodes.find(n => n.id === segmentId)) {
      nodes.push({
        id: segmentId,
        label: `Segment ${client.segment}`,
        size: 25,
        color: SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS],
        shape: 'diamond',
        font: { color: '#fff', strokeWidth: 3, strokeColor: '#222' },
      })
    }
    edges.push({ from: client.id, to: segmentId })
    
    return { nodes, edges, client }
  }, [selectedKGClient, clients, kgDepth])

  // Get clients for selected concept
  const conceptClients = useMemo(() => {
    if (!selectedConcept) return []
    return clients.filter(c => 
      c.topConcepts?.some((tc: any) => typeof tc === 'string' && tc.toLowerCase().includes(selectedConcept.toLowerCase()))
    ).slice(0, 10)
  }, [selectedConcept, clients])

  // Get clients for selected heatmap cell
  const heatmapClients = useMemo(() => {
    if (!selectedHeatmapCell) return []
    const { segment, concept } = selectedHeatmapCell
    const segNum = typeof segment === 'string' ? parseInt(segment.replace('Seg ', '')) : segment
    return clients.filter(c => 
      c.segment === segNum && 
      c.topConcepts?.some((tc: any) => String(tc).toLowerCase().includes(String(concept).toLowerCase()))
    ).slice(0, 8)
  }, [selectedHeatmapCell, clients])

  // Handle 3D plot click
  const handle3DClick = (eventData: any) => {
    if (eventData.points && eventData.points[0]) {
      const point = eventData.points[0]
      const clientId = point.text
      const client = clients.find(c => c.id === clientId)
      if (client) {
        setSelectedPoint(client)
      }
    }
  }

  // Handle concept bar click — Recharts <Bar onClick> passes (barData, index)
  const handleConceptClick = (data: any) => {
    if (data?.concept) {
      setSelectedConcept(data.concept)
    } else if (data?.activePayload?.[0]?.payload?.concept) {
      setSelectedConcept(data.activePayload[0].payload.concept)
    }
  }

  // Handle heatmap cell click
  const handleHeatmapClick = (segment: any, concept: any, value: any) => {
    if (value > 0) {
      setSelectedHeatmapCell({ segment, concept, value })
    }
  }

  const renderView = () => {
    switch(view) {
      case '3d':
        return <ThreeDView 
                  scatter3d={scatter3d} 
                  handle3DClick={handle3DClick} 
                  selectedPoint={selectedPoint} 
                  setSelectedPoint={setSelectedPoint}
                  highlightSegment={highlightSegment}
                  setHighlightSegment={setHighlightSegment}
                  zoomLevel={zoomLevel}
                  setZoomLevel={setZoomLevel}
                  onClientClick={onClientClick}
                />
      case 'knowledge':
        return <KnowledgeGraphView 
                  knowledgeGraphData={knowledgeGraphData}
                  clients={clients}
                  selectedKGClient={selectedKGClient}
                  setSelectedKGClient={setSelectedKGClient}
                  kgDepth={kgDepth}
                  setKgDepth={setKgDepth}
                  onClientClick={onClientClick}
                />
      case 'concepts':
        return <ConceptsView 
                  concepts={concepts}
                  handleConceptClick={handleConceptClick}
                  selectedConcept={selectedConcept}
                  setSelectedConcept={setSelectedConcept}
                  conceptClients={conceptClients}
                  onClientClick={onClientClick}
                />
      case 'heatmap':
        return <HeatmapView 
                  heatmap={heatmap}
                  handleHeatmapClick={handleHeatmapClick}
                  selectedHeatmapCell={selectedHeatmapCell}
                  setSelectedHeatmapCell={setSelectedHeatmapCell}
                  heatmapClients={heatmapClients}
                  onClientClick={onClientClick}
                />
      default:
        return null
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Data Analytics</h1>
          <p>Explore complex relationships between clients, concepts, and segments.</p>
        </div>
        <div className="filter-group">
          <button className={`filter-btn ${view === '3d' ? 'active' : ''}`} onClick={() => setView('3d')}>
            {Icons.cube} 3D Space
          </button>
          <button className={`filter-btn ${view === 'knowledge' ? 'active' : ''}`} onClick={() => setView('knowledge')}>
            {Icons.share} Knowledge Graph
          </button>
          <button className={`filter-btn ${view === 'concepts' ? 'active' : ''}`} onClick={() => setView('concepts')}>
            {Icons.barchart} Top Concepts
          </button>
          <button className={`filter-btn ${view === 'heatmap' ? 'active' : ''}`} onClick={() => setView('heatmap')}>
            {Icons.grid} Heatmap
          </button>
        </div>
      </div>
      {renderView()}
    </div>
  )
}

// Individual View Components for DataPage

const ThreeDView = ({ scatter3d, handle3DClick, selectedPoint, setSelectedPoint, highlightSegment, setHighlightSegment, zoomLevel, setZoomLevel, onClientClick }: any) => (
  <div className="data-view-layout">
    <div className="card main-chart-card">
      <div className="chart-controls">
        <div className="control-group">
          <label>Highlight Segment</label>
          <select value={highlightSegment ?? 'all'} onChange={e => setHighlightSegment(e.target.value === 'all' ? null : Number(e.target.value))}>
            <option value="all">None</option>
            {[...new Set(scatter3d.map((p:any) => p.segment))].map((s: Key | null | undefined) => <option key={s} value={s as any}>Segment {s}</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Zoom</label>
          <input 
            type="range" 
            min="0.5" 
            max="2.5" 
            step="0.1" 
            value={zoomLevel} 
            onChange={e => setZoomLevel(parseFloat(e.target.value))}
          />
        </div>
      </div>
      <div className="chart-container">
        <Plot
          data={[{
            x: scatter3d.map((p: any) => p.x),
            y: scatter3d.map((p: any) => p.y),
            z: scatter3d.map((p: any) => p.z),
            text: scatter3d.map((p: any) => p.id),
            mode: 'markers',
            type: 'scatter3d',
            marker: {
              size: 5,
              color: scatter3d.map((p: any) => SEGMENT_COLORS[p.segment as keyof typeof SEGMENT_COLORS]),
              opacity: scatter3d.map((p: any) => highlightSegment === null || p.segment === highlightSegment ? 0.8 : 0.1),
            },
          }]}
          layout={{
            autosize: true,
            margin: { l: 0, r: 0, b: 0, t: 0 },
            scene: {
              camera: {
                eye: { x: 1.25 * zoomLevel, y: 1.25 * zoomLevel, z: 1.25 * zoomLevel }
              }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
          }}
          config={{ responsive: true, displayModeBar: false }}
          onClick={handle3DClick}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
    {selectedPoint && (
      <div className="card info-sidebar">
        <div className="sidebar-header">
          <h3>Selected Client</h3>
          <button className="close-btn" onClick={() => setSelectedPoint(null)}>{Icons.close}</button>
        </div>
        <ClientDetailCard client={selectedPoint} onClientClick={onClientClick} />
      </div>
    )}
  </div>
)

const KnowledgeGraphView = ({ knowledgeGraphData, clients, selectedKGClient, setSelectedKGClient, kgDepth, setKgDepth, onClientClick }: any) => (
  <div className="data-view-layout">
    <div className="card main-chart-card">
      <div className="chart-controls">
        <div className="control-group">
          <label>Central Client</label>
          <select value={selectedKGClient ?? ''} onChange={e => setSelectedKGClient(e.target.value)}>
            {clients.slice(0, 100).map((c:any) => <option key={c.id} value={c.id}>{c.id}</option>)}
          </select>
        </div>
        <div className="control-group">
          <label>Depth</label>
          <select value={kgDepth} onChange={e => setKgDepth(Number(e.target.value))}>
            <option value={1}>1 (Concepts)</option>
            <option value={2}>2 (Related Clients)</option>
          </select>
        </div>
      </div>
      <div className="chart-container">
        {knowledgeGraphData && (
          <Graph
            graph={knowledgeGraphData}
            options={{
              height: '600px',
              nodes: {
                shape: 'dot',
                font: {
                  color: '#fff',
                  strokeWidth: 3,
                  strokeColor: '#222'
                }
              },
              edges: {
                color: '#e2e8f0',
                arrows: {
                  to: { enabled: false }
                }
              },
              physics: {
                enabled: true,
                barnesHut: {
                  gravitationalConstant: -3000,
                  springConstant: 0.02,
                  springLength: 150
                }
              }
            }}
          />
        )}
      </div>
    </div>
    {knowledgeGraphData?.client && (
      <div className="card info-sidebar">
        <div className="sidebar-header">
          <h3>Central Client</h3>
        </div>
        <ClientDetailCard client={knowledgeGraphData.client} onClientClick={onClientClick} />
      </div>
    )}
  </div>
)

const ConceptsView = ({ concepts, handleConceptClick, selectedConcept, setSelectedConcept, conceptClients, onClientClick }: any) => {
  if (!concepts || concepts.length === 0) {
    return (
      <div className="data-view-layout">
        <div className="card main-chart-card">
          <h3 className="card-title">Top 20 Client Concepts</h3>
          <p style={{ padding: '2rem', opacity: 0.6 }}>No concept data available.</p>
        </div>
      </div>
    )
  }
  return (
  <div className="data-view-layout">
    <div className="card main-chart-card">
      <h3 className="card-title">Top 20 Client Concepts</h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={550}>
          <BarChart layout="vertical" data={concepts} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="concept" width={150} tick={{ fontSize: 12 }} />
            <Tooltip cursor={{ fill: 'var(--bg-tertiary)' }} contentStyle={{ borderRadius: 'var(--radius-md)' }} />
            <Bar dataKey="count" fill="var(--accent-primary)" barSize={20} onClick={handleConceptClick}>
              {concepts.map((entry: any, index: number) => (
                <Cell key={`cell-${index}`} fill={entry.concept === selectedConcept ? 'var(--accent-primary-dark)' : 'var(--accent-primary)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
    {selectedConcept && (
      <div className="card info-sidebar">
        <div className="sidebar-header">
          <h3>Clients for "{selectedConcept}"</h3>
          <button className="close-btn" onClick={() => setSelectedConcept(null)}>{Icons.close}</button>
        </div>
        <div className="sidebar-client-list">
          {conceptClients.map((c: any) => <ClientChip key={c.id} client={c} onClientClick={onClientClick} />)}
        </div>
      </div>
    )}
  </div>
  )
}

const HeatmapView = ({ heatmap, handleHeatmapClick, selectedHeatmapCell, setSelectedHeatmapCell, heatmapClients, onClientClick }: any) => {
  // Safety: if heatmap is not in the expected { data, xLabels, yLabels, max } shape, show a message
  if (!heatmap?.data || !heatmap?.xLabels || !heatmap?.yLabels) {
    return (
      <div className="data-view-layout">
        <div className="card main-chart-card">
          <h3 className="card-title">Heatmap: Segment vs Concept</h3>
          <p style={{ padding: '2rem', opacity: 0.6 }}>No heatmap data available.</p>
        </div>
      </div>
    )
  }
  return (
  <div className="data-view-layout">
    <div className="card main-chart-card" style={{ overflowX: 'auto' }}>
      <h3 className="card-title">Heatmap: Segment vs Concept</h3>
      <div style={{ minWidth: '800px', padding: '1rem' }}>
          <HeatMapGrid
            data={heatmap.data}
            xLabels={heatmap.xLabels}
            yLabels={heatmap.yLabels}
            cellRender={(value: number, _xLabel: any, _yLabel: any) => (
              <div style={{ fontSize: '11px' }}>{value > 0 ? value : ''}</div>
            )}
            xLabelWidth={120}
            yLabelWidth={80}
            cellStyle={(_bg: any, value: any, _min: any, _max: any) => ({
              background: `rgba(99, 102, 241, ${Math.min(value / (heatmap.max || 1), 1)})`,
              fontSize: "11px",
              color: value > (heatmap.max || 1) * 0.5 ? "#fff" : "#444",
            })}
            onClick={(xi: number, yi: number) => handleHeatmapClick(heatmap.yLabels[yi], heatmap.xLabels[xi], heatmap.data[yi]?.[xi] ?? 0)}
          />
      </div>
    </div>
    {selectedHeatmapCell && (
      <div className="card info-sidebar">
        <div className="sidebar-header">
          <h3>Clients at Intersection</h3>
          <button className="close-btn" onClick={() => setSelectedHeatmapCell(null)}>{Icons.close}</button>
        </div>
        <div className="sidebar-info">
          <p><strong>Segment:</strong> {selectedHeatmapCell.segment}</p>
          <p><strong>Concept:</strong> {selectedHeatmapCell.concept}</p>
          <p><strong>Count:</strong> {selectedHeatmapCell.value}</p>
        </div>
        <div className="sidebar-client-list">
          {heatmapClients.map((c: any) => <ClientChip key={c.id} client={c} onClientClick={onClientClick} />)}
        </div>
      </div>
    )}
  </div>
  )
}

const ClientDetailCard = ({ client, onClientClick }: { client: any; onClientClick?: (id: string) => void }) => (
  <div className="client-detail-content">
    <div className="client-summary">
      <div className="client-avatar" style={{ backgroundColor: SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS] }}>
        {client.id.slice(-2)}
      </div>
      <div className="client-info">
        <div className="client-id">{client.id}</div>
        <div className="client-segment">Segment {client.segment}</div>
      </div>
    </div>
    <div className="detail-item">
      <span className="detail-label">Confidence</span>
      <span className="detail-value">{((client.confidence || 0) * 100).toFixed(0)}%</span>
    </div>
    <div className="client-concepts">
      <h4>Top Concepts</h4>
      {client.topConcepts?.map((c: string, i: number) => (
        <span key={i} className="concept-badge" style={{ backgroundColor: SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS] + '20', color: SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS] }}>{c}</span>
      ))}
    </div>
    {client.fullText && (
      <div className="client-note">
        <h4>Original Note</h4>
        <p>{client.fullText}</p>
      </div>
    )}
    {onClientClick && (
      <button className="sidebar-view-360-btn" onClick={() => onClientClick(client.id)}>
        View Client 360 →
      </button>
    )}
  </div>
)

const ClientChip = ({ client, onClientClick }: { client: any; onClientClick?: (id: string) => void }) => (
  <div className="client-chip" style={{ cursor: onClientClick ? 'pointer' : 'default' }} onClick={() => onClientClick?.(client.id)}>
    <div className="client-chip-avatar" style={{ backgroundColor: SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS] }}>
      {client.id.slice(-2)}
    </div>
    <span>{client.id}</span>
  </div>
)
// Admin Page (admin-only) — DB status, upload history, user management
const AdminPage = ({ data }: { data: DashboardData | null }) => {
  const { user } = useAuth()
  const [dbStatus, setDbStatus] = useState<any>(null)
  const [uploadHistory, setUploadHistory] = useState<any[]>([])
  const [dbLoading, setDbLoading] = useState(true)

  useEffect(() => {
    const fetchAdminData = async () => {
      setDbLoading(true)
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 4000)
        const [statusRes, historyRes] = await Promise.all([
          fetch(`${API_CONFIG.BASE_URL}/api/db/status`, { signal: controller.signal }).then(r => r.json()).catch(() => null),
          fetch(`${API_CONFIG.BASE_URL}/api/upload-history`, { signal: controller.signal }).then(r => r.json()).catch(() => []),
        ])
        clearTimeout(timeout)
        setDbStatus(statusRes)
        setUploadHistory(Array.isArray(historyRes) ? historyRes : [])
      } catch (e) {
        console.error('Failed to fetch admin data', e)
      } finally {
        setDbLoading(false)
      }
    }
    fetchAdminData()
  }, [])

  const handleSyncDb = async () => {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 10000)
      const res = await fetch(`${API_CONFIG.BASE_URL}/api/db/sync`, { method: 'POST', signal: controller.signal })
      clearTimeout(timeout)
      const result = await res.json()
      alert(result.message || 'Sync complete')
    } catch (e: any) {
      alert('Sync failed: ' + (e.name === 'AbortError' ? 'Server not reachable (timeout)' : e.message))
    }
  }

  if (!(['admin', 'data-scientist', 'data-analyst'] as string[]).includes(user?.role || '')) {
    return (
      <div className="page">
        <div className="page-header"><div><h1>Access Denied</h1><p>Administrator or data team access required.</p></div></div>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Administration</h1>
          <p>Database status, upload history, and system management.</p>
        </div>
      </div>

      <div className="admin-grid">
        {/* DB Status */}
        <div className="admin-card">
          <h3>{Icons.cube} Database</h3>
          {dbLoading ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Loading…</p>
          ) : dbStatus ? (
            <>
              <div className="admin-stat-row">
                <span className="admin-stat-label">Status</span>
                <span className="admin-stat-value">
                  <span className={`db-status-dot ${dbStatus.connected ? 'connected' : 'disconnected'}`} />
                  {dbStatus.connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              {dbStatus.stats && Object.entries(dbStatus.stats).map(([key, val]) => (
                <div className="admin-stat-row" key={key}>
                  <span className="admin-stat-label" style={{ textTransform: 'capitalize' }}>{key}</span>
                  <span className="admin-stat-value">{String(val)}</span>
                </div>
              ))}
              <button className="admin-btn primary" onClick={handleSyncDb}>Sync Files → DB</button>
            </>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
              <span className="db-status-dot disconnected" />
              Database not configured. Set DATABASE_URL in .env
            </p>
          )}
        </div>

        {/* Pipeline Info */}
        <div className="admin-card">
          <h3>{Icons.timer} Pipeline</h3>
          <div className="admin-stat-row">
            <span className="admin-stat-label">Total Clients</span>
            <span className="admin-stat-value">{data?.metrics?.clients || 0}</span>
          </div>
          <div className="admin-stat-row">
            <span className="admin-stat-label">Total Segments</span>
            <span className="admin-stat-value">{data?.metrics?.segments || 0}</span>
          </div>
          {data?.processingInfo?.pipelineTimings && (
            <div className="admin-stat-row">
              <span className="admin-stat-label">Last Pipeline Time</span>
              <span className="admin-stat-value">{data.processingInfo.pipelineTimings.total}s</span>
            </div>
          )}
          {data?.processingInfo?.timestamp && (
            <div className="admin-stat-row">
              <span className="admin-stat-label">Last Run</span>
              <span className="admin-stat-value">{new Date(data.processingInfo.timestamp).toLocaleString()}</span>
            </div>
          )}
        </div>

        {/* Upload History */}
        <div className="admin-card" style={{ gridColumn: '1 / -1' }}>
          <h3>{Icons.tag} Upload History</h3>
          {uploadHistory.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>No uploads recorded yet.</p>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table className="upload-history-table">
                <thead>
                  <tr>
                    <th>File</th>
                    <th>User</th>
                    <th>Status</th>
                    <th>Added</th>
                    <th>Updated</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {uploadHistory.map((h: any, i: number) => (
                    <tr key={i}>
                      <td>{h.filename}</td>
                      <td>{h.username || h.user_id || '—'}</td>
                      <td><span className={`status-badge ${h.status}`}>{h.status}</span></td>
                      <td>{h.records_added ?? '—'}</td>
                      <td>{h.records_updated ?? '—'}</td>
                      <td>{h.created_at ? new Date(h.created_at).toLocaleString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Main App Component
function App() {
  const { user, isLoading: authLoading } = useAuth()
  const [activePage, setActivePage] = useState('dashboard')
  const [data, setData] = useState<DashboardData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedClientId, setSelectedClientId] = useState<string | null>(null)
  const [dbDebug, setDbDebug] = useState<string>('Testing DB...')

  // DB diagnostic — runs once on mount, shows result in header
  useEffect(() => {
    import('./services/db').then(mod => {
      const sql = mod.default
      sql`SELECT COUNT(*) as cnt FROM clients WHERE is_deleted = FALSE`
        .then((r: any) => setDbDebug(`DB OK: ${r[0]?.cnt} clients`))
        .catch((e: any) => setDbDebug(`DB error: ${e?.message || e}`))
    }).catch((e: any) => setDbDebug(`Import error: ${e?.message}`))
  }, [])

  /** Clean raw data from any source */
  const cleanData = (result: any): DashboardData => {
    const cleanedClients = (result.clients || []).map((c: any) => ({
      ...c,
      confidence: typeof c.confidence === 'number' ? c.confidence : 0,
      segment: typeof c.segment === 'number' ? c.segment : 0,
    }))
    const cleanedRadar = (result.radar || []).map((item: any) => ({
      subject: item.subject || item.dimension, ...item
    }))

    // Transform heatmap from flat rows to { data, xLabels, yLabels, max } for HeatMapGrid
    let cleanedHeatmap = result.heatmap || []
    if (Array.isArray(cleanedHeatmap) && cleanedHeatmap.length > 0 && cleanedHeatmap[0]?.segment !== undefined) {
      const heatmapConcepts: string[] = result.heatmapConcepts ||
        Object.keys(cleanedHeatmap[0]).filter((k: string) => k !== 'segment')
      const yLabels = cleanedHeatmap.map((row: any) => String(row.segment))
      const xLabels = heatmapConcepts
      let maxVal = 0
      const data = cleanedHeatmap.map((row: any) =>
        xLabels.map((concept: string) => {
          const val = Number(row[concept] || 0)
          if (val > maxVal) maxVal = val
          return val
        })
      )
      cleanedHeatmap = { data, xLabels, yLabels, max: maxVal || 1 }
    }
    return { ...result, clients: cleanedClients, radar: cleanedRadar, heatmap: cleanedHeatmap }
  }

  const fetchData = async () => {
    try {
      setLoading(true)
      console.time('[Dashboard] fetchData')
      const { data: result, source } = await getData()
      console.timeEnd('[Dashboard] fetchData')
      console.info(`[Dashboard] Data loaded from ${source}`, {
        clients: result.clients?.length,
        segments: result.segments?.length,
        concepts: result.concepts?.length,
      })

      if (!result.clients || !Array.isArray(result.clients)) {
        throw new Error("Invalid or missing 'clients' data")
      }

      setData(cleanData(result))
      setError(null)
    } catch (e: any) {
      console.error('[Dashboard] Live data fetch failed:', e)
      // Fall back to bundled local data.json only if live fetch fails
      try {
        const localData = (await import('./data.json')).default as any
        const localRadar = localData.radar?.map((item: any) => ({
          subject: item.subject || item.dimension, ...item
        })) || []
        setData(cleanData({ ...localData, radar: localRadar }))
        setError(`Live data unavailable (${e.message}). Showing local fallback.`)
      } catch {
        setError(`Failed to load data: ${e.message}`)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (user) {
      fetchData()
    }
  }, [user])

  // Safety valve: if loading takes >10s, force-stop and show local data
  useEffect(() => {
    if (!loading) return
    const safety = setTimeout(() => {
      console.error('[Dashboard] Safety timeout — forcing loading=false after 10s')
      setLoading(false)
      if (!data) {
        import('./data.json').then(mod => {
          const localData = mod.default as any
          const localRadar = localData.radar?.map((item: any) => ({
            subject: item.subject || item.dimension, ...item
          })) || []
          setData(cleanData({ ...localData, radar: localRadar }))
          setError('Loading timed out. Showing local fallback data.')
        }).catch(() => {
          setError('Loading timed out and no fallback data available.')
        })
      }
    }, 10_000)
    return () => clearTimeout(safety)
  }, [loading])

  // Show nothing while checking stored session
  if (authLoading) {
    return <div className="login-backdrop"><div className="loading-spinner"><div></div><div></div><div></div></div></div>
  }

  // If not logged in, show login screen
  if (!user) {
    return <LoginScreen />
  }

  const renderPage = () => {
    if (loading) {
      return <div className="loading-spinner"><div></div><div></div><div></div></div>
    }
    if (error && !data) {
      return <div className="error-banner">{error}</div>
    }
    
    // If a client is selected, always show Client 360
    if (selectedClientId) {
      return <Client360Page
        clientId={selectedClientId}
        onBack={() => setSelectedClientId(null)}
      />
    }

    let pageToRender;
    switch (activePage) {
      case 'dashboard':
        pageToRender = <KPIDashboard onNavigate={setActivePage} />;
        break;
      case 'actions':
        pageToRender = <ActionsPage data={data} onClientClick={(id: string) => setSelectedClientId(id)} onRefresh={fetchData} />;
        break;
      case 'segments':
        pageToRender = <SegmentsPage data={data} onClientClick={(id: string) => setSelectedClientId(id)} />;
        break;
      case 'clients':
        pageToRender = <ClientsPage data={data} onClientClick={(id: string) => setSelectedClientId(id)} />;
        break;
      case 'data':
        pageToRender = <DataPage data={data} onClientClick={(id: string) => setSelectedClientId(id)} />;
        break;
      case 'admin':
        pageToRender = <AdminPage data={data} />;
        break;
      case 'calendar':
        pageToRender = <CalendarPage userId={user.id} />;
        break;
      case 'playbooks':
        pageToRender = <PlaybooksPage userId={user.id} onNavigateToCalendar={() => setActivePage('calendar')} />;
        break;
      case 'advisors':
        pageToRender = <AdvisorsPage userId={user.id} />;
        break;
      case 'reports':
        pageToRender = <ExportPage />;
        break;
      case 'upload':
        pageToRender = (
          <div className="page">
            <div className="page-header">
              <div><h1>Import Data</h1><p>Upload a CSV file with client notes or record a voice memo.</p></div>
            </div>
            <div className="upload-grid">
              <div className="upload-section">
                <FileUpload onUploadSuccess={() => { invalidateCache(); fetchData() }} userId={user.id} />
              </div>
              <div className="upload-section">
                <VoiceRecorder onRecordingComplete={() => {
                  console.log('Voice memo recorded and uploaded')
                  invalidateCache()
                  fetchData()
                }} userId={user.id} />
              </div>
            </div>
          </div>
        );
        break;
      default:
        pageToRender = <KPIDashboard onNavigate={setActivePage} />;
    }

    return pageToRender;
  }

  return (
    <div className="app">
      <Navigation activePage={activePage} setActivePage={setActivePage} data={data} onClearClient={() => setSelectedClientId(null)} />
      {error && <div className="error-banner">{error}</div>}
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  )
}

export default App
