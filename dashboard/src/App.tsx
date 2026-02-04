import { useState, useMemo, useEffect } from 'react'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, Tooltip, Legend
} from 'recharts'
import Plot from 'react-plotly.js'
import API_CONFIG from './config'
import FileUpload from './FileUpload'
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

interface Client {
  id: string;
  segment: number;
  topConcepts?: string[];
  fullText?: string;
  language?: string;
  date?: string;
  confidence?: number;
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
}

// Navigation Component
const Navigation = ({ activePage, setActivePage, data }: { 
  activePage: string; 
  setActivePage: (page: string) => void; 
  data: DashboardData | null;
}) => {
  const pages = [
    { id: 'upload', label: 'Upload', icon: Icons.actions },
    { id: 'actions', label: 'Actions', icon: Icons.actions },
    { id: 'segments', label: 'Segments', icon: Icons.segments },
    { id: 'clients', label: 'Clients', icon: Icons.clients },
    { id: 'data', label: 'Data', icon: Icons.chart },
  ]

  return (
    <nav className="nav">
      <div className="nav-brand">
        <span className="nav-logo">LVMH</span>
        <span className="nav-subtitle">Client Intelligence</span>
      </div>
      <div className="nav-links">
        {pages.map(page => (
          <button
            key={page.id}
            className={`nav-link ${activePage === page.id ? 'active' : ''}`}
            onClick={() => setActivePage(page.id)}
          >
            <span className="nav-icon">{page.icon}</span>
            <span>{page.label}</span>
          </button>
        ))}
      </div>
      <div className="nav-metrics">
        <div className="nav-metric">
          <span className="nav-metric-value">{data?.metrics?.clients || 0}</span>
          <span className="nav-metric-label">Clients</span>
        </div>
        <div className="nav-metric">
          <span className="nav-metric-value">{data?.metrics?.segments || 0}</span>
          <span className="nav-metric-label">Segments</span>
        </div>
        {data?.processingInfo?.pipelineTimings?.total && (
          <div className="nav-metric processing-time">
            <span className="nav-metric-value">{data.processingInfo.pipelineTimings.total}s</span>
            <span className="nav-metric-label">⏱️ Pipeline</span>
          </div>
        )}
      </div>
    </nav>
  )
}

// Actions Page
const ActionsPage = ({ data }: { data: DashboardData | null }) => {
  const [filter, setFilter] = useState('all')
  
  const actions = useMemo(() => {
    if (!data?.clients) return []
    
    const actionList: any[] = []
    const actionTypes = {
      'Visite Privée': { priority: 'high', type: 'appointment', label: 'Planifier visite privée' },
      'Prêt à Acheter': { priority: 'high', type: 'sale', label: 'Finaliser vente' },
      'Virement': { priority: 'high', type: 'payment', label: 'Confirmer paiement' },
      'VIP': { priority: 'medium', type: 'vip', label: 'Attention VIP' },
      'Joaillerie': { priority: 'medium', type: 'product', label: 'Présenter joaillerie' },
      'Couture': { priority: 'medium', type: 'product', label: 'Présenter couture' },
      'Corporate': { priority: 'medium', type: 'corporate', label: 'Solutions corporate' },
    }
    
    data.clients.forEach(client => {
      const concepts = client.topConcepts || []
      concepts.forEach(concept => {
        const actionInfo = Object.entries(actionTypes).find(([key]) => 
          concept.toLowerCase().includes(key.toLowerCase())
        )
        if (actionInfo) {
          actionList.push({
            id: `${client.id}-${actionInfo[0]}`,
            clientId: client.id,
            segment: client.segment,
            ...actionInfo[1],
            concept: actionInfo[0],
            confidence: client.confidence
          })
        }
      })
    })
    
    return actionList.slice(0, 50)
  }, [])

  const filteredActions = filter === 'all' ? actions : actions.filter(a => a.priority === filter)
  const highCount = actions.filter(a => a.priority === 'high').length
  const mediumCount = actions.filter(a => a.priority === 'medium').length

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Actions Recommandées</h1>
          <p>Priorisez vos interactions client</p>
        </div>
        <div className="filter-group">
          <button className={`filter-btn ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>
            Toutes ({actions.length})
          </button>
          <button className={`filter-btn high ${filter === 'high' ? 'active' : ''}`} onClick={() => setFilter('high')}>
            Urgentes ({highCount})
          </button>
          <button className={`filter-btn medium ${filter === 'medium' ? 'active' : ''}`} onClick={() => setFilter('medium')}>
            Moyennes ({mediumCount})
          </button>
        </div>
      </div>

      <div className="stats-row">
        <div className="stat-card accent-red">
          <div className="stat-icon">{Icons.actions}</div>
          <div className="stat-content">
            <div className="stat-value">{highCount}</div>
            <div className="stat-label">Urgentes</div>
          </div>
        </div>
        <div className="stat-card accent-blue">
          <div className="stat-icon">{Icons.calendar}</div>
          <div className="stat-content">
            <div className="stat-value">{actions.filter(a => a.type === 'appointment').length}</div>
            <div className="stat-label">Visites</div>
          </div>
        </div>
        <div className="stat-card accent-green">
          <div className="stat-icon">{Icons.tag}</div>
          <div className="stat-content">
            <div className="stat-value">{actions.filter(a => a.type === 'sale').length}</div>
            <div className="stat-label">Ventes</div>
          </div>
        </div>
        <div className="stat-card accent-purple">
          <div className="stat-icon">{Icons.user}</div>
          <div className="stat-content">
            <div className="stat-value">{actions.filter(a => a.type === 'vip').length}</div>
            <div className="stat-label">VIP</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="action-list">
          {filteredActions.map(action => (
            <div key={action.id} className={`action-item priority-${action.priority}`}>
              <div className="action-priority">
                <span className={`priority-dot ${action.priority}`}></span>
              </div>
              <div className="action-content">
                <div className="action-header">
                  <span className="action-client">{action.clientId}</span>
                  <span className="action-segment" style={{ backgroundColor: SEGMENT_COLORS[action.segment as keyof typeof SEGMENT_COLORS] }}>
                    Seg {action.segment}
                  </span>
                </div>
                <div className="action-label">{action.label}</div>
                <div className="action-meta">
                  <span className="action-confidence">{(action.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
              <button className="action-btn">{Icons.chevron}</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Segments Page
const SegmentsPage = ({ data }: { data: DashboardData | null }) => {
  const [selectedSegment, setSelectedSegment] = useState<number | null>(null)
  const segmentData = data?.segments || []
  const radarData = data?.radar || []

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Analyse des Segments</h1>
          <p>{segmentData.length} segments par clustering sémantique</p>
        </div>
      </div>

      <div className="segments-grid">
        {segmentData.map((segment, idx) => (
          <div 
            key={segment.name}
            className={`segment-card ${selectedSegment === idx ? 'selected' : ''}`}
            onClick={() => setSelectedSegment(selectedSegment === idx ? null : idx)}
            style={{ '--segment-color': COLORS[idx % COLORS.length] }}
          >
            <div className="segment-header">
              <div className="segment-number">{idx}</div>
              <div className="segment-count">{segment.value} clients</div>
            </div>
            <div className="segment-profile">{segment.profile}</div>
            <div className="segment-bar">
              <div 
                className="segment-bar-fill" 
                style={{ width: `${(segment.value / Math.max(...segmentData.map(s => s.value))) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="charts-row">
        <div className="card">
          <h3 className="card-title">Distribution des Segments</h3>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie 
                data={segmentData} 
                cx="50%" 
                cy="50%" 
                innerRadius={70} 
                outerRadius={120} 
                paddingAngle={3} 
                dataKey="value"
                label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {segmentData.map((entry, index) => (
                  <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value, _name, props) => [`${value} clients`, props.payload.profile]} 
                contentStyle={{ borderRadius: '8px', border: '1px solid #e6ebf1', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} 
              />
              <Legend 
                layout="vertical" 
                verticalAlign="middle" 
                align="right"
                wrapperStyle={{ paddingLeft: '20px', fontSize: '13px' }}
                formatter={(value, entry: any) => <span style={{ color: '#525f7f' }}>{`Segment ${entry.payload?.name?.replace('Segment ', '')}`}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="card-title">Profil Radar par Segment</h3>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="#e6ebf1" strokeDasharray="3 3" />
              <PolarAngleAxis 
                dataKey="dimension" 
                tick={{ fontSize: 11, fill: '#525f7f' }} 
                tickLine={false}
              />
              <PolarRadiusAxis 
                angle={30} 
                domain={[0, 100]} 
                tick={{ fontSize: 9, fill: '#8898aa' }}
                tickCount={4}
              />
              {[0, 1, 2, 3].map(i => (
                <Radar 
                  key={i} 
                  name={`Segment ${i}`} 
                  dataKey={`seg${i}`} 
                  stroke={COLORS[i]} 
                  fill={COLORS[i]} 
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              ))}
              <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e6ebf1', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} />
              <Legend 
                layout="horizontal" 
                verticalAlign="bottom" 
                align="center"
                wrapperStyle={{ paddingTop: '16px', fontSize: '12px' }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {selectedSegment !== null && (
        <div className="card segment-detail">
          <h3 className="card-title">
            Segment {selectedSegment}
            <button className="close-btn" onClick={() => setSelectedSegment(null)}>{Icons.close}</button>
          </h3>
          <div className="segment-detail-content">
            <p><strong>Clients:</strong> {segmentData[selectedSegment]?.value}</p>
            <p><strong>Profil:</strong> {segmentData[selectedSegment]?.fullProfile}</p>
            <div className="segment-concepts">
              {segmentData[selectedSegment]?.fullProfile?.split(' | ').map((concept, i) => (
                <span key={i} className="concept-chip" style={{ backgroundColor: COLORS[selectedSegment % COLORS.length] + '20', color: COLORS[selectedSegment % COLORS.length] }}>
                  {concept}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
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

const ClientsPage = ({ data }: { data: DashboardData | null }) => {
  const [searchQuery, setSearchQuery] = useState("")
  const [filterType, setFilterType] = useState<"all" | "segment">("all")

  const clients = data?.clients || []

  const filteredClients = useMemo(() => {
    return clients.filter((client) =>
      client.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      client.topConcepts?.some((c) =>
        c.toLowerCase().includes(searchQuery.toLowerCase())
      )
    )
  }, [clients, searchQuery])

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Base Clients</h1>
          <p>{clients.length} clients analysés</p>
        </div>
        <div className="search-group">
          <div className="search-input-wrapper">
            <span className="search-icon">{Icons.search}</span>
            <input
              type="text"
              placeholder="Rechercher..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
          </div>
          <select
            className="segment-select"
            value={filterType}
            onChange={(e) =>
              setFilterType(e.target.value as "all" | "segment")
            }
          >
            <option value="all">Tous</option>
            <option value="segment">Par segment</option>
          </select>
        </div>
      </header>

      <div className="clients-grid">
        {filteredClients.map((client) => {
          const similarity = Math.round((client as any).confidence ? (client as any).confidence * 100 : 0)
          const color = segmentColors[client.segment] ?? "#18181b"

          return (
            <div
              key={client.id}
              className="client-card"
            >
              <div className="client-header">
                <div
                  className="client-avatar"
                  style={{ backgroundColor: color }}
                >
                  {client.segment}
                </div>

                <div className="client-info">
                  <div className="client-id">{client.id}</div>
                  <div className="client-segment">Segment {client.segment}</div>
                </div>

                <div className="client-confidence">
                  <div className="confidence-ring" style={{ "--confidence": similarity / 100 } as any}>
                    <span>{similarity}%</span>
                  </div>
                </div>
              </div>

              <div className="client-concepts">
                {client.topConcepts?.slice(0, 5).map((concept, idx) => (
                  <span key={idx} className="concept-badge" style={{ backgroundColor: color + '20', color: color }}>
                    {concept}
                  </span>
                ))}
              </div>

              {client.fullText && (
                <div className="client-text">
                  <p>{client.fullText.substring(0, 150)}...</p>
                </div>
              )}

              <div className="client-footer">
                {client.language && (
                  <span className="client-meta">
                    <span className="meta-icon">🌐</span>
                    {client.language}
                  </span>
                )}
                {client.date && (
                  <span className="client-meta">
                    <span className="meta-icon">📅</span>
                    {client.date}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Data Page - Interactive
const DataPage = ({ data }: { data: DashboardData | null }) => {
  const [view, setView] = useState('3d')
  const [selectedPoint, setSelectedPoint] = useState<any>(null)
  const [selectedConcept, setSelectedConcept] = useState<string | null>(null)
  const [selectedHeatmapCell, setSelectedHeatmapCell] = useState<any>(null)
  const [highlightSegment, setHighlightSegment] = useState<number | null>(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [selectedKGClient, setSelectedKGClient] = useState<string | null>(null)
  const [kgDepth, setKgDepth] = useState(2)
  
  const scatter3d = data?.scatter3d || []
  const concepts = data?.concepts || []
  const heatmap = data?.heatmap || []
  const clients = data?.clients || []

  // Build Knowledge Graph data for selected client
  const knowledgeGraphData = useMemo(() => {
    if (!selectedKGClient) return null
    
    const client = clients.find(c => c.id === selectedKGClient)
    if (!client) return null
    
    const nodes: any[] = []
    const links = []
    
    // Central client node
    nodes.push({
      id: client.id,
      type: 'client',
      label: client.id,
      segment: client.segment,
      size: 30
    })
    
    // Add concept nodes
    const clientConcepts = client.topConcepts || []
    clientConcepts.forEach((concept) => {
      const conceptId = `concept-${concept}`
      nodes.push({
        id: conceptId,
        type: 'concept',
        label: concept,
        size: 20
      })
      links.push({
        source: client.id,
        target: conceptId,
        type: 'has_concept'
      })
      
      // If depth >= 2, find other clients with same concept
      if (kgDepth >= 2) {
        const relatedClients = clients.filter(c => 
          c.id !== client.id && 
          c.topConcepts?.includes(concept)
        ).slice(0, 3) // Limit to 3 per concept
        
        relatedClients.forEach(rc => {
          if (!nodes.find(n => n.id === rc.id)) {
            nodes.push({
              id: rc.id,
              type: 'related-client',
              label: rc.id,
              segment: rc.segment,
              size: 18
            })
          }
          links.push({
            source: conceptId,
            target: rc.id,
            type: 'shared_by'
          })
        })
      }
    })
    
    // Add segment node
    const segmentId = `segment-${client.segment}`
    nodes.push({
      id: segmentId,
      type: 'segment',
      label: `Segment ${client.segment}`,
      segment: client.segment,
      size: 25
    })
    links.push({
      source: client.id,
      target: segmentId,
      type: 'belongs_to'
    })
    
    return { nodes, links, client }
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

  // Handle concept bar click
  const handleConceptClick = (data: any) => {
    if (data && data.activePayload && data.activePayload[0]) {
      setSelectedConcept(data.activePayload[0].payload.concept)
    }
  }

  // Handle heatmap cell click
  const handleHeatmapClick = (segment: any, concept: any, value: any) => {
    if (value > 0) {
      setSelectedHeatmapCell({ segment, concept, value })
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Visualisation Interactive</h1>
          <p>Cliquez sur les éléments pour explorer les données</p>
        </div>
        <div className="view-toggle">
          <button className={`toggle-btn ${view === '3d' ? 'active' : ''}`} onClick={() => { setView('3d'); setSelectedPoint(null); }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="toggle-icon">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            3D Space
          </button>
          <button className={`toggle-btn ${view === 'knowledge' ? 'active' : ''}`} onClick={() => { setView('knowledge'); setSelectedKGClient(null); }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="toggle-icon">
              <circle cx="12" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/>
              <line x1="12" y1="8" x2="5" y2="16"/><line x1="12" y1="8" x2="19" y2="16"/>
            </svg>
            Knowledge Graph
          </button>
          <button className={`toggle-btn ${view === 'concepts' ? 'active' : ''}`} onClick={() => { setView('concepts'); setSelectedConcept(null); }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="toggle-icon">
              <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
            Concepts
          </button>
          <button className={`toggle-btn ${view === 'heatmap' ? 'active' : ''}`} onClick={() => { setView('heatmap'); setSelectedHeatmapCell(null); }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="toggle-icon">
              <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
            </svg>
            Heatmap
          </button>
        </div>
      </div>

      {view === '3d' && (
        <div className="data-layout">
          <div className="card viz-card">
            <div className="card-header">
              <h3 className="card-title">Espace Vectoriel 3D</h3>
              <div className="card-controls">
                <span className="hint">💡 Cliquez sur un point pour voir le détail</span>
                <div className="segment-filters">
                  {[0, 1, 2, 3, 4, 5, 6, 7].filter(i => scatter3d.some(p => (p.cluster ?? p.segment) === i)).map(i => (
                    <button
                      key={i}
                      className={`segment-filter-btn ${highlightSegment === i ? 'active' : ''}`}
                      style={{ '--seg-color': COLORS[i] }}
                      onClick={() => setHighlightSegment(highlightSegment === i ? null : i)}
                    >
                      {i}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <Plot
              data={Object.entries(
                scatter3d.reduce((acc: Record<string, { x: number[], y: number[], z: number[], text: string[], ids: string[] }>, point) => {
                  const key = `Segment ${point.cluster ?? point.segment}`
                  if (!acc[key]) acc[key] = { x: [], y: [], z: [], text: [], ids: [] }
                  acc[key].x.push(point.x)
                  acc[key].y.push(point.y)
                  acc[key].z.push(point.z)
                  acc[key].text.push(`${point.client}<br>Profile: ${point.profile ?? 'Unknown'}`)
                  acc[key].ids.push(point.client)
                  return acc
                }, {})
              ).map(([name, points], idx) => ({
                type: 'scatter3d' as const,
                mode: 'markers' as const,
                name,
                x: points.x,
                y: points.y,
                z: points.z,
                text: points.text,
                customdata: points.ids,
                hovertemplate: '<b>%{text}</b><extra></extra>',
                marker: {
                  size: highlightSegment !== null ? (idx === highlightSegment ? 8 : 3) : 5,
                  color: COLORS[idx % COLORS.length],
                  opacity: highlightSegment !== null ? (idx === highlightSegment ? 1 : 0.2) : 0.8,
                  line: { width: highlightSegment === idx ? 1 : 0, color: '#fff' }
                }
              }))}
              layout={{
                autosize: true,
                height: 500,
                margin: { l: 0, r: 0, b: 0, t: 0 },
                scene: {
                  xaxis: { title: '', showticklabels: false, showgrid: true, gridcolor: '#e2e8f0', zerolinecolor: '#e2e8f0' },
                  yaxis: { title: '', showticklabels: false, showgrid: true, gridcolor: '#e2e8f0', zerolinecolor: '#e2e8f0' },
                  zaxis: { title: '', showticklabels: false, showgrid: true, gridcolor: '#e2e8f0', zerolinecolor: '#e2e8f0' },
                  bgcolor: '#fafafa',
                  camera: { eye: { x: 1.5 * zoomLevel, y: 1.5 * zoomLevel, z: 1.2 * zoomLevel } }
                },
                legend: { orientation: 'h', y: -0.02, font: { size: 11 } },
                paper_bgcolor: 'transparent',
                hoverlabel: { bgcolor: '#1e293b', font: { color: '#fff', size: 12 }, bordercolor: '#1e293b' }
              }}
              config={{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'] }}
              style={{ width: '100%', height: '500px' }}
              onClick={handle3DClick}
            />
            <div className="zoom-controls">
              <button onClick={() => setZoomLevel(z => Math.max(0.5, z - 0.2))}>−</button>
              <span>Zoom</span>
              <button onClick={() => setZoomLevel(z => Math.min(2, z + 0.2))}>+</button>
            </div>
          </div>

          {selectedPoint && (
            <div className="detail-panel">
              <div className="detail-header">
                <h3>Client Sélectionné</h3>
                <button className="close-btn" onClick={() => setSelectedPoint(null)}>{Icons.close}</button>
              </div>
              <div className="detail-content">
                <div className="detail-avatar" style={{ backgroundColor: SEGMENT_COLORS[selectedPoint.segment as keyof typeof SEGMENT_COLORS] }}>
                  {(selectedPoint.id || selectedPoint.client).slice(-3)}
                </div>
                <h4>{selectedPoint.id || selectedPoint.client}</h4>
                <span className="detail-segment" style={{ backgroundColor: SEGMENT_COLORS[selectedPoint.segment as keyof typeof SEGMENT_COLORS] }}>
                  Segment {selectedPoint.segment}
                </span>
                <div className="detail-confidence">
                  <div className="confidence-bar" style={{ '--conf': selectedPoint.confidence ?? 0 } as React.CSSProperties}/>
                  <span>{((selectedPoint.confidence ?? 0) * 100).toFixed(0)}% confiance</span>
                </div>
                <div className="detail-concepts">
                  <h5>Concepts</h5>
                  {selectedPoint.topConcepts?.map((c: string, i: number) => (
                    <span key={i} className="detail-chip">{c}</span>
                  ))}
                </div>
                {selectedPoint.originalNote && (
                  <div className="detail-note">
                    <h5>Note</h5>
                    <p>{selectedPoint.originalNote.substring(0, 200)}...</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {view === 'knowledge' && (
        <div className="data-layout kg-layout">
          {/* Client Selector */}
          <div className="kg-sidebar">
            <div className="kg-sidebar-header">
              <h3>Sélectionner un Client</h3>
              <p>Choisissez un client pour voir son réseau</p>
            </div>
            <div className="kg-client-list">
              {clients.slice(0, 30).map(client => (
                <button
                  key={client.id}
                  className={`kg-client-btn ${selectedKGClient === client.id ? 'active' : ''}`}
                  onClick={() => setSelectedKGClient(client.id)}
                >
                  <div className="kg-client-avatar" style={{ backgroundColor: SEGMENT_COLORS[client.segment as keyof typeof SEGMENT_COLORS] }}>
                    {client.id.slice(-2)}
                  </div>
                  <div className="kg-client-info">
                    <span className="kg-client-id">{client.id}</span>
                    <span className="kg-client-meta">Seg {client.segment} • {client.topConcepts?.length || 0} concepts</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Knowledge Graph Visualization */}
          <div className="card viz-card kg-main">
            {!selectedKGClient ? (
              <div className="kg-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="kg-placeholder-icon">
                  <circle cx="12" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/>
                  <line x1="12" y1="8" x2="5" y2="16"/><line x1="12" y1="8" x2="19" y2="16"/>
                </svg>
                <h3>Graphe de Connaissances</h3>
                <p>Sélectionnez un client dans la liste pour visualiser ses connexions</p>
              </div>
            ) : (
              <>
                <div className="card-header">
                  <h3 className="card-title">Réseau de {selectedKGClient}</h3>
                  <div className="kg-controls">
                    <label>Profondeur:</label>
                    <select value={kgDepth} onChange={e => setKgDepth(Number(e.target.value))}>
                      <option value={1}>1 (Concepts seulement)</option>
                      <option value={2}>2 (+ Clients liés)</option>
                    </select>
                  </div>
                </div>
                
                {/* Force-directed graph using Plotly */}
                {knowledgeGraphData && (
                  <div className="kg-graph">
                    <Plot
                      data={[
                        // Links
                        ...knowledgeGraphData.links.map(link => {
                          const source = knowledgeGraphData.nodes.find(n => n.id === link.source)
                          const target = knowledgeGraphData.nodes.find(n => n.id === link.target)
                          if (!source || !target) return null
                          
                          // Calculate positions in a circle/radial layout
                          const nodeIndex = knowledgeGraphData.nodes.findIndex(n => n.id === source.id)
                          const targetIndex = knowledgeGraphData.nodes.findIndex(n => n.id === target.id)
                          
                          return {
                            type: 'scatter',
                            mode: 'lines',
                            x: [nodeIndex, targetIndex],
                            y: [Math.sin(nodeIndex * 0.5) * 2, Math.sin(targetIndex * 0.5) * 2],
                            line: { color: '#cbd5e1', width: 1 },
                            hoverinfo: 'none',
                            showlegend: false
                          }
                        }).filter(Boolean),
                        // Nodes by type
                        {
                          type: 'scatter',
                          mode: 'markers+text',
                          name: 'Client Principal',
                          x: [0],
                          y: [0],
                          text: [knowledgeGraphData.client.id],
                          textposition: 'bottom center',
                          marker: {
                            size: 40,
                            color: SEGMENT_COLORS[knowledgeGraphData.client.segment],
                            line: { width: 3, color: '#fff' }
                          },
                          hovertemplate: '<b>%{text}</b><br>Client Principal<extra></extra>'
                        },
                        {
                          type: 'scatter',
                          mode: 'markers+text',
                          name: 'Concepts',
                          x: knowledgeGraphData.nodes.filter(n => n.type === 'concept').map((_, i) => Math.cos(i * Math.PI / 3) * 3),
                          y: knowledgeGraphData.nodes.filter(n => n.type === 'concept').map((_, i) => Math.sin(i * Math.PI / 3) * 3),
                          text: knowledgeGraphData.nodes.filter(n => n.type === 'concept').map(n => n.label),
                          textposition: 'top center',
                          marker: {
                            size: 25,
                            color: '#6366f1',
                            symbol: 'diamond'
                          },
                          hovertemplate: '<b>%{text}</b><br>Concept<extra></extra>'
                        },
                        {
                          type: 'scatter',
                          mode: 'markers+text',
                          name: 'Clients Liés',
                          x: knowledgeGraphData.nodes.filter(n => n.type === 'related-client').map((_, i) => Math.cos((i + 0.5) * Math.PI / 4) * 5),
                          y: knowledgeGraphData.nodes.filter(n => n.type === 'related-client').map((_, i) => Math.sin((i + 0.5) * Math.PI / 4) * 5),
                          text: knowledgeGraphData.nodes.filter(n => n.type === 'related-client').map(n => n.label),
                          textposition: 'bottom center',
                          marker: {
                            size: 20,
                            color: knowledgeGraphData.nodes.filter(n => n.type === 'related-client').map(n => SEGMENT_COLORS[n.segment] || '#94a3b8'),
                            line: { width: 1, color: '#fff' }
                          },
                          hovertemplate: '<b>%{text}</b><br>Client Lié<extra></extra>'
                        },
                        {
                          type: 'scatter',
                          mode: 'markers+text',
                          name: 'Segment',
                          x: [0],
                          y: [-4],
                          text: [`Segment ${knowledgeGraphData.client.segment}`],
                          textposition: 'bottom center',
                          marker: {
                            size: 30,
                            color: SEGMENT_COLORS[knowledgeGraphData.client.segment],
                            symbol: 'square',
                            opacity: 0.7
                          },
                          hovertemplate: '<b>%{text}</b><br>Cluster<extra></extra>'
                        }
                      ]}
                      layout={{
                        autosize: true,
                        height: 500,
                        margin: { l: 20, r: 20, b: 20, t: 20 },
                        xaxis: { visible: false, range: [-7, 7] },
                        yaxis: { visible: false, range: [-6, 6], scaleanchor: 'x' },
                        showlegend: true,
                        legend: { orientation: 'h', y: -0.05, font: { size: 11 } },
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        hoverlabel: { bgcolor: '#1e293b', font: { color: '#fff' } }
                      }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%', height: '500px' }}
                    />
                  </div>
                )}

                {/* Client Details */}
                <div className="kg-client-details">
                  <div className="kg-detail-card">
                    <h4>Informations Client</h4>
                    <div className="kg-detail-grid">
                      <div className="kg-detail-item">
                        <span className="kg-detail-label">Segment</span>
                        <span className="kg-detail-value" style={{ color: SEGMENT_COLORS[knowledgeGraphData?.client?.segment as keyof typeof SEGMENT_COLORS] }}>
                          {knowledgeGraphData?.client?.segment}
                        </span>
                      </div>
                      <div className="kg-detail-item">
                        <span className="kg-detail-label">Confiance</span>
                        <span className="kg-detail-value">{((knowledgeGraphData?.client?.confidence || 0) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="kg-detail-item">
                        <span className="kg-detail-label">Concepts</span>
                        <span className="kg-detail-value">{knowledgeGraphData?.client?.topConcepts?.length || 0}</span>
                      </div>
                      <div className="kg-detail-item">
                        <span className="kg-detail-label">Connexions</span>
                        <span className="kg-detail-value">{knowledgeGraphData?.links?.length || 0}</span>
                      </div>
                    </div>
                  </div>
                  <div className="kg-detail-card">
                    <h4>Concepts</h4>
                    <div className="kg-concepts-list">
                      {knowledgeGraphData?.client?.topConcepts?.map((c: any, i: number) => (
                        <span key={i} className="kg-concept-tag">{c}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {view === 'concepts' && (
        <div className="data-layout">
          <div className="card viz-card">
            <div className="card-header">
              <h3 className="card-title">Top Concepts</h3>
              <span className="hint">💡 Cliquez sur une barre pour voir les clients</span>
            </div>
            <ResponsiveContainer width="100%" height={450}>
              <BarChart data={concepts} layout="vertical" margin={{ left: 100, right: 20 }} onClick={handleConceptClick}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={true} vertical={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={{ stroke: '#e2e8f0' }} />
                <YAxis 
                  dataKey="concept" 
                  type="category" 
                  tick={{ fontSize: 12, fill: '#334155', fontWeight: 500 }} 
                  width={95} 
                  axisLine={{ stroke: '#e2e8f0' }}
                />
                <Tooltip 
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.15)', padding: '12px' }}
                  formatter={(value) => [`${value} occurrences`, 'Fréquence']}
                  cursor={{ fill: 'rgba(99, 102, 241, 0.1)' }}
                />
                <Bar dataKey="count" radius={[0, 6, 6, 0]} cursor="pointer">
                  {concepts.map((entry, index) => (
                    <Cell 
                      key={entry.concept} 
                      fill={selectedConcept === entry.concept ? '#1e293b' : COLORS[index % COLORS.length]}
                      style={{ transition: 'all 0.2s' }}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {selectedConcept && (
            <div className="detail-panel">
              <div className="detail-header">
                <h3>Concept: {selectedConcept}</h3>
                <button className="close-btn" onClick={() => setSelectedConcept(null)}>{Icons.close}</button>
              </div>
              <div className="detail-content">
                <div className="detail-stat">
                  <span className="stat-number">{conceptClients.length}+</span>
                  <span className="stat-text">clients avec ce concept</span>
                </div>
                <h5>Clients associés</h5>
                <div className="detail-client-list">
                  {conceptClients.map(client => (
                    <div key={client.id} className="detail-client-item">
                      <div className="client-mini-avatar" style={{ backgroundColor: SEGMENT_COLORS[client.segment] }}>
                        {client.id.slice(-2)}
                      </div>
                      <div className="client-mini-info">
                        <span className="client-mini-id">{client.id}</span>
                        <span className="client-mini-segment">Seg {client.segment}</span>
                      </div>
                      <span className="client-mini-conf">{(client.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {view === 'heatmap' && heatmap.length > 0 && (
        <div className="data-layout">
          <div className="card viz-card">
            <div className="card-header">
              <h3 className="card-title">Segments × Concepts</h3>
              <span className="hint">💡 Cliquez sur une cellule pour voir les détails</span>
            </div>
            <div className="heatmap-container">
              {(() => {
                // Extract concept keys from heatmap data (all keys except 'segment')
                const heatmapConcepts = heatmap.length > 0 
                  ? Object.keys(heatmap[0]).filter(k => k !== 'segment')
                  : []
                return (
                  <table className="heatmap-table">
                    <thead>
                      <tr>
                        <th className="heatmap-corner"></th>
                        {heatmapConcepts.map(c => (
                          <th key={c} className="heatmap-header">{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {heatmap.map((row: any, i) => (
                        <tr key={row.segment}>
                          <td className="heatmap-label">
                            <span className="segment-dot" style={{ backgroundColor: COLORS[i] }}></span>
                            {row.segment}
                          </td>
                          {heatmapConcepts.map(c => {
                            const val = row[c] || 0
                            const max = Math.max(...heatmap.map((r: any) => r[c] || 0))
                            const intensity = max > 0 ? val / max : 0
                            const isSelected = selectedHeatmapCell?.segment === row.segment && selectedHeatmapCell?.concept === c
                            return (
                              <td 
                                key={c} 
                                className={`heatmap-cell ${val > 0 ? 'clickable' : ''} ${isSelected ? 'selected' : ''}`}
                                style={{ 
                                  backgroundColor: `rgba(99, 102, 241, ${intensity * 0.85})`,
                                  color: intensity > 0.5 ? '#fff' : '#334155'
                                }}
                                onClick={() => handleHeatmapClick(row.segment, c, val)}
                              >
                                {val > 0 && val}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )
              })()}
            </div>
            <div className="heatmap-legend">
              <span>Faible</span>
              <div className="legend-gradient"></div>
              <span>Élevé</span>
            </div>
          </div>

          {selectedHeatmapCell && (
            <div className="detail-panel">
              <div className="detail-header">
                <h3>{selectedHeatmapCell.segment} × {selectedHeatmapCell.concept}</h3>
                <button className="close-btn" onClick={() => setSelectedHeatmapCell(null)}>{Icons.close}</button>
              </div>
              <div className="detail-content">
                <div className="detail-stat">
                  <span className="stat-number">{selectedHeatmapCell.value}</span>
                  <span className="stat-text">correspondances</span>
                </div>
                <h5>Clients dans cette intersection</h5>
                <div className="detail-client-list">
                  {heatmapClients.map(client => (
                    <div key={client.id} className="detail-client-item">
                      <div className="client-mini-avatar" style={{ backgroundColor: SEGMENT_COLORS[client.segment] }}>
                        {client.id.slice(-2)}
                      </div>
                      <div className="client-mini-info">
                        <span className="client-mini-id">{client.id}</span>
                        <span className="client-mini-concepts">{client.topConcepts?.slice(0, 2).join(', ')}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-icon">{Icons.user}</div>
          <div className="stat-content">
            <div className="stat-value">{data?.metrics?.clients || clients.length}</div>
            <div className="stat-label">Clients</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">{Icons.segments}</div>
          <div className="stat-content">
            <div className="stat-value">{data?.metrics?.segments || 0}</div>
            <div className="stat-label">Segments</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">{Icons.tag}</div>
          <div className="stat-content">
            <div className="stat-value">{concepts.length}</div>
            <div className="stat-label">Concepts</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">{Icons.chart}</div>
          <div className="stat-content">
            <div className="stat-value">{scatter3d.length}</div>
            <div className="stat-label">Points 3D</div>
          </div>
        </div>
        {data?.processingInfo?.pipelineTimings?.total && (
          <div className="stat-card processing-timer">
            <div className="stat-icon">⏱️</div>
            <div className="stat-content">
              <div className="stat-value">{data.processingInfo.pipelineTimings.total}s</div>
              <div className="stat-label">Temps de traitement</div>
            </div>
          </div>
        )}
      </div>

      {/* Processing Details */}
      {data?.processingInfo?.pipelineTimings && Object.keys(data.processingInfo.pipelineTimings).length > 1 && (
        <div className="processing-details">
          <h4>⏱️ Détails du traitement</h4>
          <div className="timing-grid">
            {Object.entries(data.processingInfo.pipelineTimings)
              .filter(([key]) => key !== 'total')
              .map(([stage, time]) => (
                <div key={stage} className="timing-item">
                  <span className="timing-stage">{stage}</span>
                  <div className="timing-bar-container">
                    <div 
                      className="timing-bar" 
                      style={{ 
                        width: `${Math.min(100, (Number(time) / Number(data.processingInfo?.pipelineTimings?.total || 1)) * 100)}%` 
                      }}
                    />
                  </div>
                  <span className="timing-value">{Number(time).toFixed(1)}s</span>
                </div>
              ))}
          </div>
          {data.processingInfo.timestamp && (
            <div className="processing-timestamp">
              Dernière mise à jour: {new Date(data.processingInfo.timestamp).toLocaleString('fr-FR')}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Main App
export default function App() {
  const [activePage, setActivePage] = useState('upload')
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch data from API on mount
  const fetchData = async () => {
    try {
      setLoading(true)
      console.log('Fetching from:', `${API_CONFIG.BASE_URL}/api/data`)
      
      // Add timeout to fetch
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10 second timeout
      
      const response = await fetch(`${API_CONFIG.BASE_URL}/api/data`, {
        signal: controller.signal
      })
      clearTimeout(timeoutId)
      
      console.log('Response status:', response.status)
      
      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.statusText}`)
      }
      const jsonData = await response.json()
      console.log('Data received:', Object.keys(jsonData))
      setData(jsonData)
      setError(null)
    } catch (err: any) {
      console.error('Error fetching data:', err)
      setError(err.name === 'AbortError' ? 'Connection timeout - server not responding' : err.message)
      // Try to load local fallback data if API fails
      try {
        const localData = await import('./data.json')
        setData(localData.default)
        setError('Using local data (API unavailable)')
      } catch {
        setError('Failed to load data from API and no local fallback available')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const handleUploadSuccess = (result) => {
    // Refresh data after successful upload and processing
    if (result.status === 'completed') {
      fetchData()
    }
  }

  if (loading) {
    return (
      <div className="app" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <h2>Loading dashboard...</h2>
          <p>Connecting to {API_CONFIG.BASE_URL}</p>
        </div>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="app" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div style={{ textAlign: 'center', color: '#f43f5e' }}>
          <h2>Error loading data</h2>
          <p>{error}</p>
          <p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
            Make sure the server is running at {API_CONFIG.BASE_URL}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      {error && (
        <div style={{ 
          background: '#fef3c7', 
          color: '#92400e', 
          padding: '0.75rem', 
          textAlign: 'center',
          borderBottom: '1px solid #fbbf24'
        }}>
          {error}
        </div>
      )}
      <Navigation activePage={activePage} setActivePage={setActivePage} data={data} />
      <main className="main">
        {activePage === 'upload' && (
          <div className="page">
            <div className="page-header">
              <div>
                <h1>Upload New Data</h1>
                <p>Upload a CSV file to process and analyze</p>
              </div>
            </div>
            <FileUpload onUploadSuccess={handleUploadSuccess} />
          </div>
        )}
        {activePage === 'actions' && <ActionsPage data={data} />}
        {activePage === 'segments' && <SegmentsPage data={data} />}
        {activePage === 'clients' && <ClientsPage data={data} />}
        {activePage === 'data' && <DataPage data={data} />}
      </main>
    </div>
  )
}
