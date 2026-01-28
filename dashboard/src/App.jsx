import { useMemo, useState, useCallback } from 'react'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, Tooltip, Legend
} from 'recharts'
import Plot from 'react-plotly.js'
import data from './data.json'

// Debounce hook for search input
const useDebounce = (callback, delay) => {
  const timeoutRef = useMemo(() => ({ current: null }), [])
  return useCallback((...args) => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(() => callback(...args), delay)
  }, [callback, delay])
}

// Color palette for segments - consistent across all visualizations
const COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']

// Language code to name mapping
const LANGUAGE_NAMES = {
  'FR': 'Français', 'EN': 'English', 'IT': 'Italiano', 'ES': 'Español',
  'DE': 'Deutsch', 'PT': 'Português', 'NL': 'Nederlands', 'RU': 'Русский',
  'AR': 'العربية', 'ZH': '中文', 'JA': '日本語', 'KO': '한국어'
}

// Client Detail Modal Component
const ClientDetailModal = ({ client, onClose }) => {
  if (!client) return null

  // Highlight matched text in the original note
  const renderHighlightedNote = () => {
    if (!client.originalNote || !client.conceptEvidence?.length) {
      return <p className="original-note">{client.originalNote || 'Note non disponible'}</p>
    }

    // Sort evidence by span start
    const sortedEvidence = [...client.conceptEvidence].sort((a, b) => a.spanStart - b.spanStart)
    
    // Build highlighted segments
    const segments = []
    let lastEnd = 0
    
    sortedEvidence.forEach((ev, idx) => {
      // Add text before this match
      if (ev.spanStart > lastEnd) {
        segments.push({
          type: 'text',
          content: client.originalNote.slice(lastEnd, ev.spanStart)
        })
      }
      // Add highlighted match
      if (ev.spanStart < client.originalNote.length) {
        segments.push({
          type: 'highlight',
          content: client.originalNote.slice(ev.spanStart, ev.spanEnd),
          concept: ev.concept,
          alias: ev.matchedAlias
        })
        lastEnd = Math.max(lastEnd, ev.spanEnd)
      }
    })
    
    // Add remaining text
    if (lastEnd < client.originalNote.length) {
      segments.push({
        type: 'text',
        content: client.originalNote.slice(lastEnd)
      })
    }

    return (
      <p className="original-note">
        {segments.map((seg, idx) => 
          seg.type === 'highlight' ? (
            <span 
              key={idx} 
              className="highlight-match"
              title={`Concept: ${seg.concept}\nMatched: "${seg.alias}"`}
            >
              {seg.content}
            </span>
          ) : (
            <span key={idx}>{seg.content}</span>
          )
        )}
      </p>
    )
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>
        
        <div className="modal-header">
          <h2>🔍 Analyse Client: {client.id}</h2>
          <span 
            className="segment-badge-large"
            style={{ backgroundColor: COLORS[client.segment % COLORS.length] }}
          >
            Segment {client.segment}
          </span>
        </div>

        {/* Algorithm Explanation Section */}
        <div className="modal-section explanation-section">
          <h3>🧠 Pourquoi ce segment ?</h3>
          <div className="explanation-box">
            <p>
              <strong>L'algorithme a classé ce client dans le Segment {client.segment}</strong> basé sur l'analyse suivante:
            </p>
            <ol className="explanation-steps">
              <li>
                <strong>Extraction de texte:</strong> La note originale a été analysée en <em>{LANGUAGE_NAMES[client.noteLanguage] || client.noteLanguage}</em>
                {['KO', 'ZH', 'JA', 'AR'].includes(client.noteLanguage) && (
                  <span className="lang-note"> (langue non-latine — détection de concepts limitée)</span>
                )}
              </li>
              <li>
                <strong>Détection de concepts:</strong> {client.conceptEvidence?.length || 0} concept(s) identifiés par correspondance textuelle
                {client.conceptEvidence?.length === 0 && (
                  <span className="lang-note"> — la classification reste valide via l'embedding sémantique</span>
                )}
              </li>
              <li>
                <strong>Vectorisation multilingue:</strong> Le modèle SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) 
                comprend {Object.keys(LANGUAGE_NAMES).length}+ langues et convertit le texte en vecteur 384D <em>indépendamment de la langue</em>
              </li>
              <li>
                <strong>Clustering K-Means:</strong> Le vecteur a été regroupé avec {client.segmentInfo?.clientCount || '?'} autres clients 
                ayant des profils sémantiquement similaires
              </li>
              <li>
                <strong>Score de confiance:</strong> <span className={`confidence-score ${client.confidence > 0.6 ? 'high' : client.confidence > 0.4 ? 'medium' : 'low'}`}>
                  {(client.confidence * 100).toFixed(0)}%
                </span> — proximité au centroïde du cluster
              </li>
            </ol>
          </div>
        </div>

        {/* Original Note */}
        <div className="modal-section">
          <h3>📝 Note Originale</h3>
          <div className="note-metadata">
            <span>🌐 {LANGUAGE_NAMES[client.noteLanguage] || client.noteLanguage}</span>
            <span>📅 {client.noteDate}</span>
          </div>
          <div className="note-box">
            {renderHighlightedNote()}
          </div>
          <p className="note-hint">
            {client.conceptEvidence?.length > 0 
              ? '💡 Les parties surlignées correspondent aux concepts détectés par correspondance textuelle'
              : `💡 Aucun surlignage — le texte en ${LANGUAGE_NAMES[client.noteLanguage] || client.noteLanguage} a été classé par son embedding sémantique`
            }
          </p>
        </div>

        {/* Detected Concepts */}
        <div className="modal-section">
          <h3>🏷️ Concepts Détectés ({client.conceptEvidence?.length || 0})</h3>
          {client.conceptEvidence?.length > 0 ? (
            <div className="concepts-grid">
              {client.conceptEvidence.map((ev, idx) => (
                <div key={idx} className="concept-card">
                  <div className="concept-name">{ev.concept}</div>
                  <div className="concept-match">
                    Détecté: "<em>{ev.matchedAlias}</em>"
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-concepts-box">
              <div className="no-concepts-icon">🌐</div>
              <div className="no-concepts-title">Classification par Embedding Multilingue</div>
              <p className="no-concepts-text">
                Aucun concept n'a été détecté par correspondance textuelle directe dans cette note en <strong>{LANGUAGE_NAMES[client.noteLanguage] || client.noteLanguage}</strong>.
              </p>
              <p className="no-concepts-text">
                <strong>Cependant</strong>, le modèle SentenceTransformer comprend le sens sémantique du texte 
                et l'a converti en un vecteur qui capture sa signification. Ce vecteur est proche 
                des autres clients du Segment {client.segment}.
              </p>
              <div className="semantic-match-info">
                <span className="semantic-label">Profil sémantique assigné:</span>
                <div className="semantic-tags">
                  {client.topTags?.slice(0, 5).map((tag, idx) => (
                    <span key={idx} className="concept-tag inferred">{tag}</span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Segment Comparison */}
        <div className="modal-section">
          <h3>📊 Profil du Segment {client.segment}</h3>
          <div className="segment-comparison">
            <div className="segment-profile-box">
              <div className="profile-label">Concepts dominants du segment:</div>
              <div className="profile-tags">
                {client.segmentInfo?.topConcepts?.map((concept, idx) => (
                  <span 
                    key={idx} 
                    className={`concept-tag ${client.topTags?.includes(concept) ? 'matched' : ''}`}
                  >
                    {concept}
                    {client.topTags?.includes(concept) && <span className="match-indicator">✓</span>}
                  </span>
                ))}
              </div>
            </div>
            <div className="client-match-summary">
              <strong>Correspondance:</strong> Le client partage{' '}
              <span className="match-count">
                {client.topTags?.filter(t => client.segmentInfo?.topConcepts?.includes(t)).length || 0}
              </span>
              {' '}concept(s) avec le profil typique du segment
            </div>
          </div>
        </div>

        {/* 3D Space Position */}
        <div className="modal-section">
          <h3>🌐 Position dans l'Espace Vectoriel 3D</h3>
          <div className="coords-display">
            <div className="coord-box">
              <span className="coord-label">X (UMAP 1)</span>
              <span className="coord-value">{client.coords3d?.x?.toFixed(4) || '0.0000'}</span>
            </div>
            <div className="coord-box">
              <span className="coord-label">Y (UMAP 2)</span>
              <span className="coord-value">{client.coords3d?.y?.toFixed(4) || '0.0000'}</span>
            </div>
            <div className="coord-box">
              <span className="coord-label">Z (UMAP 3)</span>
              <span className="coord-value">{client.coords3d?.z?.toFixed(4) || '0.0000'}</span>
            </div>
          </div>
          <div className="mini-3d-container">
            <Plot
              data={[
                // All points from the same segment (faded)
                {
                  type: 'scatter3d',
                  mode: 'markers',
                  name: `Autres Segment ${client.segment}`,
                  x: data.scatter3d.filter(p => p.cluster === client.segment && p.client !== client.id).map(p => p.x),
                  y: data.scatter3d.filter(p => p.cluster === client.segment && p.client !== client.id).map(p => p.y),
                  z: data.scatter3d.filter(p => p.cluster === client.segment && p.client !== client.id).map(p => p.z),
                  marker: {
                    size: 4,
                    color: COLORS[client.segment % COLORS.length],
                    opacity: 0.3
                  },
                  hoverinfo: 'skip'
                },
                // Current client (highlighted)
                {
                  type: 'scatter3d',
                  mode: 'markers+text',
                  name: client.id,
                  x: [client.coords3d?.x || 0],
                  y: [client.coords3d?.y || 0],
                  z: [client.coords3d?.z || 0],
                  text: [client.id],
                  textposition: 'top center',
                  textfont: { size: 12, color: '#333' },
                  marker: {
                    size: 12,
                    color: COLORS[client.segment % COLORS.length],
                    opacity: 1,
                    symbol: 'diamond',
                    line: { width: 2, color: '#fff' }
                  },
                  hovertemplate: `<b>${client.id}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>`
                }
              ]}
              layout={{
                autosize: true,
                height: 350,
                margin: { l: 0, r: 0, b: 0, t: 30 },
                scene: {
                  xaxis: { title: 'UMAP 1', showticklabels: false, gridcolor: '#eee' },
                  yaxis: { title: 'UMAP 2', showticklabels: false, gridcolor: '#eee' },
                  zaxis: { title: 'UMAP 3', showticklabels: false, gridcolor: '#eee' },
                  camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
                },
                showlegend: true,
                legend: { orientation: 'h', y: -0.05 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%', height: '350px' }}
            />
          </div>
          <p className="coords-explanation">
            💡 Ce graphique montre la position du client (◆) par rapport aux autres clients du même segment.
            Les clients proches dans cet espace partagent des caractéristiques sémantiques similaires.
          </p>
        </div>

        {/* All Tags */}
        <div className="modal-section">
          <h3>🔖 Tous les Tags du Client</h3>
          <div className="all-tags">
            {client.topTags?.map((tag, idx) => (
              <span key={idx} className="concept-tag primary">{tag}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Metric Card Component
const MetricCard = ({ icon, value, label }) => (
  <div className="metric-card">
    <div className="metric-icon">{icon}</div>
    <div className="metric-value">{value}</div>
    <div className="metric-label">{label}</div>
  </div>
)

// Segment Pie Chart - following Recharts best practices
const SegmentPieChart = ({ segments }) => {
  // Custom label renderer for better positioning
  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
    const RADIAN = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)

    return percent > 0.05 ? (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        fontSize={11}
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    ) : null
  }

  return (
    <ResponsiveContainer width="100%" height={350}>
      <PieChart>
        <Pie
          data={segments}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={renderCustomLabel}
          outerRadius={120}
          innerRadius={40}
          fill="#8884d8"
          dataKey="value"
          paddingAngle={2}
        >
          {segments.map((entry, index) => (
            <Cell key={`cell-${entry.name}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          formatter={(value, name, props) => {
            const profile = props.payload.fullProfile || props.payload.profile
            return [`${value} clients`, profile]
          }}
          contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.15)', maxWidth: '300px' }}
        />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  )
}

// Concepts Bar Chart - with CartesianGrid per Recharts best practices
const ConceptsBarChart = ({ concepts }) => (
  <ResponsiveContainer width="100%" height={350}>
    <BarChart
      data={concepts}
      layout="vertical"
      margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
    >
      <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
      <XAxis type="number" tick={{ fontSize: 11 }} />
      <YAxis dataKey="concept" type="category" width={80} tick={{ fontSize: 11 }} />
      <Tooltip
        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}
        formatter={(value) => [`${value} occurrences`, 'Count']}
      />
      <Bar dataKey="count" fill="#667eea" radius={[0, 5, 5, 0]}>
        {concepts.map((entry) => (
          <Cell key={`cell-${entry.concept}`} fill={COLORS[concepts.indexOf(entry) % COLORS.length]} />
        ))}
      </Bar>
    </BarChart>
  </ResponsiveContainer>
)

// Radar Chart - dynamically render ALL segments
const SegmentRadarChart = ({ radarData, numSegments = 8 }) => (
  <ResponsiveContainer width="100%" height={400}>
    <RadarChart data={radarData}>
      <PolarGrid />
      <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11 }} />
      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
      {Array.from({ length: numSegments }, (_, i) => (
        <Radar
          key={`seg${i}`}
          name={`Segment ${i}`}
          dataKey={`seg${i}`}
          stroke={COLORS[i % COLORS.length]}
          fill={COLORS[i % COLORS.length]}
          fillOpacity={0.2}
        />
      ))}
      <Legend />
      <Tooltip />
    </RadarChart>
  </ResponsiveContainer>
)

// Heatmap Table
const ConceptHeatmap = ({ heatmapData, concepts }) => {
  const maxVal = Math.max(...heatmapData.flatMap(row => concepts.map(c => row[c] || 0)))
  const getColor = (val) => {
    const intensity = val / maxVal
    return `rgba(102, 126, 234, ${0.1 + intensity * 0.8})`
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="segment-table">
        <thead>
          <tr>
            <th>Segment</th>
            {concepts.map(c => <th key={c} style={{ fontSize: '0.85em' }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {heatmapData.map((row, i) => (
            <tr key={i}>
              <td><strong>{row.segment}</strong></td>
              {concepts.map(c => (
                <td key={c} style={{
                  background: getColor(row[c] || 0),
                  textAlign: 'center',
                  fontWeight: row[c] > maxVal * 0.5 ? 'bold' : 'normal',
                  color: row[c] > maxVal * 0.7 ? 'white' : '#333'
                }}>
                  {row[c] || 0}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Segment Details Table - shows full cluster profile
const SegmentDetailsTable = ({ details }) => (
  <table className="segment-table">
    <thead>
      <tr>
        <th>Segment</th>
        <th>Clients</th>
        <th>Profil Complet</th>
        <th>Top Concepts</th>
      </tr>
    </thead>
    <tbody>
      {details.map(seg => (
        <tr key={seg.id}>
          <td><strong>Segment {seg.id}</strong></td>
          <td>{seg.clients}</td>
          <td style={{ maxWidth: '300px', whiteSpace: 'normal' }}>
            {seg.profile}
          </td>
          <td>
            {seg.topConcepts.slice(0, 5).map(c => (
              <span key={c} className="concept-tag">{c}</span>
            ))}
          </td>
        </tr>
      ))}
    </tbody>
  </table>
)

// Client Search Component
const ClientSearch = ({ clients }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchType, setSearchType] = useState('all') // 'all', 'id', 'tag'
  const [selectedSegment, setSelectedSegment] = useState('all')
  const [filteredResults, setFilteredResults] = useState([])
  const [displayCount, setDisplayCount] = useState(50)
  const [isSearching, setIsSearching] = useState(false)
  const [selectedClient, setSelectedClient] = useState(null) // For modal

  // Get unique segments for filter
  const segments = useMemo(() => {
    const unique = [...new Set(clients.map(c => c.segment))].sort((a, b) => a - b)
    return unique
  }, [clients])

  // Search function with debounce
  const performSearch = useCallback((query, type, segment) => {
    setIsSearching(true)
    setDisplayCount(50) // Reset display count on new search
    const q = query.toLowerCase().trim()
    
    let results = clients

    // Filter by segment first
    if (segment !== 'all') {
      results = results.filter(c => c.segment === parseInt(segment))
    }

    // Then filter by search query
    if (q) {
      results = results.filter(client => {
        if (type === 'id' || type === 'all') {
          if (client.id.toLowerCase().includes(q)) return true
        }
        if (type === 'tag' || type === 'all') {
          // Search in all tags
          const allTagsMatch = client.allTags.some(tag => 
            tag.toLowerCase().includes(q)
          )
          // Search in top tags
          const topTagsMatch = client.topTags.some(tag => 
            tag.toLowerCase().includes(q)
          )
          if (allTagsMatch || topTagsMatch) return true
        }
        return false
      })
    }

    // Store all filtered results (no limit)
    setFilteredResults(results)
    setIsSearching(false)
  }, [clients])

  // Debounced search
  const debouncedSearch = useDebounce(performSearch, 300)

  // Handle search input change
  const handleSearchChange = (e) => {
    const value = e.target.value
    setSearchQuery(value)
    debouncedSearch(value, searchType, selectedSegment)
  }

  // Handle filter changes
  const handleTypeChange = (e) => {
    const type = e.target.value
    setSearchType(type)
    performSearch(searchQuery, type, selectedSegment)
  }

  const handleSegmentChange = (e) => {
    const segment = e.target.value
    setSelectedSegment(segment)
    performSearch(searchQuery, searchType, segment)
  }

  // Load more function
  const loadMore = () => {
    setDisplayCount(prev => Math.min(prev + 50, filteredResults.length))
  }

  // Show all function
  const showAll = () => {
    setDisplayCount(filteredResults.length)
  }

  // Initial load - show all clients
  useMemo(() => {
    if (filteredResults.length === 0 && !searchQuery && selectedSegment === 'all') {
      setFilteredResults(clients)
    }
  }, [clients, filteredResults.length, searchQuery, selectedSegment])

  // Get displayed results (limited by displayCount)
  const displayedResults = filteredResults.slice(0, displayCount)
  const hasMore = displayCount < filteredResults.length

  return (
    <div className="client-search">
      <div className="search-controls">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Rechercher par ID client ou tag..."
            value={searchQuery}
            onChange={handleSearchChange}
            className="search-input"
          />
          {searchQuery && (
            <button 
              className="clear-btn"
              onClick={() => {
                setSearchQuery('')
                setFilteredResults(clients)
                setDisplayCount(50)
              }}
            >
              ✕
            </button>
          )}
        </div>
        
        <div className="search-filters">
          <select value={searchType} onChange={handleTypeChange} className="filter-select">
            <option value="all">Tous</option>
            <option value="id">ID Client</option>
            <option value="tag">Tags</option>
          </select>
          
          <select value={selectedSegment} onChange={handleSegmentChange} className="filter-select">
            <option value="all">Tous les segments</option>
            {segments.map(s => (
              <option key={s} value={s}>Segment {s}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="search-results-info">
        {isSearching ? (
          <span>Recherche en cours...</span>
        ) : (
          <span>
            <strong>{filteredResults.length}</strong> client(s) trouvé(s)
            {hasMore && ` — Affichage de ${displayCount}`}
          </span>
        )}
      </div>

      <div className="search-results">
        <table className="segment-table client-table">
          <thead>
            <tr>
              <th>ID Client</th>
              <th>Segment</th>
              <th>Tags Principaux</th>
              <th>Confiance</th>
            </tr>
          </thead>
          <tbody>
            {displayedResults.map(client => (
              <tr 
                key={client.id} 
                className={`segment-${client.segment} clickable-row`}
                onClick={() => setSelectedClient(client)}
                title="Cliquer pour voir les détails"
              >
                <td>
                  <strong className="client-id">{client.id}</strong>
                </td>
                <td>
                  <span className="segment-badge" style={{ 
                    backgroundColor: COLORS[client.segment % COLORS.length],
                    color: 'white',
                    padding: '2px 8px',
                    borderRadius: '12px',
                    fontSize: '0.85em'
                  }}>
                    Seg {client.segment}
                  </span>
                </td>
                <td className="tags-cell">
                  {client.topTags.slice(0, 4).map((tag, i) => (
                    <span 
                      key={i} 
                      className="concept-tag"
                      onClick={() => {
                        setSearchQuery(tag)
                        setSearchType('tag')
                        performSearch(tag, 'tag', selectedSegment)
                      }}
                      style={{ cursor: 'pointer' }}
                      title="Cliquer pour rechercher ce tag"
                    >
                      {tag}
                    </span>
                  ))}
                  {client.topTags.length > 4 && (
                    <span className="more-tags">+{client.topTags.length - 4}</span>
                  )}
                </td>
                <td>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ 
                        width: `${client.confidence * 100}%`,
                        backgroundColor: client.confidence > 0.6 ? '#43e97b' : client.confidence > 0.4 ? '#f5576c' : '#667eea'
                      }}
                    />
                    <span className="confidence-value">{(client.confidence * 100).toFixed(0)}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        
        {displayedResults.length === 0 && searchQuery && (
          <div className="no-results">
            <span>😕</span>
            <p>Aucun client trouvé pour "{searchQuery}"</p>
            <p className="hint">Essayez de modifier votre recherche ou les filtres</p>
          </div>
        )}
        
        {/* Load More / Show All buttons */}
        {hasMore && (
          <div className="load-more-container">
            <button className="load-more-btn" onClick={loadMore}>
              Charger 50 de plus ({filteredResults.length - displayCount} restants)
            </button>
            <button className="show-all-btn" onClick={showAll}>
              Afficher tous ({filteredResults.length})
            </button>
          </div>
        )}
      </div>

      {/* Client Detail Modal */}
      {selectedClient && (
        <ClientDetailModal 
          client={selectedClient} 
          onClose={() => setSelectedClient(null)} 
        />
      )}
    </div>
  )
}

// 3D Scatter Plot - using useMemo for performance optimization per React best practices
const Scatter3D = ({ scatter3d }) => {
  // Memoize cluster grouping to avoid recalculation on every render
  const traces = useMemo(() => {
    const clusters = {}
    scatter3d.forEach(point => {
      if (!clusters[point.cluster]) {
        clusters[point.cluster] = { x: [], y: [], z: [], text: [] }
      }
      clusters[point.cluster].x.push(point.x)
      clusters[point.cluster].y.push(point.y)
      clusters[point.cluster].z.push(point.z)
      clusters[point.cluster].text.push(`${point.client}<br>${point.profile}`)
    })

    return Object.keys(clusters).sort().map((clusterId, idx) => ({
      type: 'scatter3d',
      mode: 'markers',
      name: `Segment ${clusterId}`,
      x: clusters[clusterId].x,
      y: clusters[clusterId].y,
      z: clusters[clusterId].z,
      text: clusters[clusterId].text,
      hovertemplate: '%{text}<extra></extra>',
      marker: {
        size: 4,
        color: COLORS[idx % COLORS.length],
        opacity: 0.8,
        line: { width: 0 }
      }
    }))
  }, [scatter3d])

  // Memoize layout config
  const layout = useMemo(() => ({
    autosize: true,
    height: 500,
    margin: { l: 0, r: 0, b: 0, t: 30 },
    scene: {
      xaxis: { title: 'UMAP 1', showticklabels: false, showgrid: true, gridcolor: '#eee' },
      yaxis: { title: 'UMAP 2', showticklabels: false, showgrid: true, gridcolor: '#eee' },
      zaxis: { title: 'UMAP 3', showticklabels: false, showgrid: true, gridcolor: '#eee' },
      camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
    },
    legend: { orientation: 'h', y: -0.1 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)'
  }), [])

  // Memoize config
  const config = useMemo(() => ({
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  }), [])

  return (
    <Plot
      data={traces}
      layout={layout}
      config={config}
      style={{ width: '100%', height: '500px' }}
      useResizeHandler={true}
    />
  )
}

// Main App
function App() {
  const { metrics, segments, concepts, radar, heatmap, heatmapConcepts, segmentDetails, scatter3d, clients } = data

  return (
    <div className="dashboard">
      <div className="header">
        <h1>🏷️ LVMH Voice-to-Tag</h1>
        <p>Analyse des Profils Clients — Dashboard Interactif</p>
      </div>

      <div className="metrics-grid">
        <MetricCard icon="👥" value={metrics.clients.toLocaleString()} label="Clients" />
        <MetricCard icon="📊" value={metrics.segments} label="Segments" />
        <MetricCard icon="🏷️" value={metrics.concepts} label="Concepts" />
        <MetricCard icon="📈" value={metrics.avgConceptsPerClient} label="Concepts/Client" />
        <MetricCard icon="✅" value={`${metrics.coverage}%`} label="Couverture" />
      </div>

      {/* Client Search Section */}
      <div className="charts-grid">
        <div className="chart-card full-width">
          <div className="chart-title">🔍 Recherche de Clients</div>
          <ClientSearch clients={clients || []} />
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-card full-width">
          <div className="chart-title">🌐 Espace Vectoriel 3D des Clients</div>
          <Scatter3D scatter3d={scatter3d} />
        </div>

        <div className="chart-card">
          <div className="chart-title">📊 Distribution des Segments</div>
          <SegmentPieChart segments={segments} />
        </div>

        <div className="chart-card">
          <div className="chart-title">🏷️ Top 12 Concepts</div>
          <ConceptsBarChart concepts={concepts} />
        </div>

        <div className="chart-card">
          <div className="chart-title">🎯 Profil des Segments (Radar)</div>
          <SegmentRadarChart radarData={radar} numSegments={metrics.segments} />
        </div>

        <div className="chart-card">
          <div className="chart-title">🔥 Heatmap: Concepts par Segment</div>
          <ConceptHeatmap heatmapData={heatmap} concepts={heatmapConcepts} />
        </div>

        <div className="chart-card full-width">
          <div className="chart-title">📋 Détails des Segments</div>
          <SegmentDetailsTable details={segmentDetails} />
        </div>
      </div>

      <div className="footer">
        <p>💡 Survolez les graphiques pour plus de détails | Faites glisser la vue 3D pour l'explorer</p>
        <p>LVMH Voice-to-Tag Pipeline — Généré automatiquement</p>
      </div>
    </div>
  )
}

export default App
