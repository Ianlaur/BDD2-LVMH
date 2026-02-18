import { useState, useEffect, useCallback } from 'react'
import { getPlaybooks as fetchPlaybooksService, activatePlaybook, createPlaybook, updatePlaybook, deletePlaybook, getConcepts as fetchConceptsService, getMatchCount } from './services/apiService'

// ─── Types ───────────────────────────────────────────────────────
interface Playbook {
  id: number
  name: string
  description: string
  concepts: string
  channel: string
  priority: string
  messageTemplate: string
  category: string
  createdAt: string
}

interface ConceptOption {
  concept: string
  clientCount: number
}

interface PlaybooksPageProps {
  userId?: number
  onNavigateToCalendar?: () => void
}

// ─── Helpers ─────────────────────────────────────────────────────
const channelLabels: Record<string, string> = {
  email: 'Email', sms: 'SMS', whatsapp: 'WhatsApp', phone: 'Phone', in_store: 'In-Store', multi: 'Multi'
}

const priorityColors: Record<string, string> = {
  high: '#ef4444', medium: '#f59e0b', low: '#22c55e'
}

const categoryLabels: Record<string, string> = {
  launch: 'Product Launch', reengagement: 'Re-engagement',
  birthday: 'Birthday & Celebration', seasonal: 'Seasonal Campaign', custom: 'Custom'
}

// ─── Component ───────────────────────────────────────────────────
export default function PlaybooksPage({ userId, onNavigateToCalendar }: PlaybooksPageProps) {
  const [playbooks, setPlaybooks] = useState<Playbook[]>([])
  const [loading, setLoading] = useState(true)
  const [filterCategory, setFilterCategory] = useState<string>('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showActivateModal, setShowActivateModal] = useState(false)
  const [selectedPlaybook, setSelectedPlaybook] = useState<Playbook | null>(null)
  const [activating, setActivating] = useState(false)
  const [activateResult, setActivateResult] = useState<any>(null)

  // Activate form
  const [activateForm, setActivateForm] = useState({
    eventName: '',
    eventDate: '',
    matchLimit: 200,
  })

  // Create form
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    concepts: '',
    channel: 'email',
    priority: 'medium',
    messageTemplate: '',
    category: 'custom',
  })
  const [creating, setCreating] = useState(false)
  const [editingPlaybook, setEditingPlaybook] = useState<Playbook | null>(null)
  const [deleting, setDeleting] = useState<number | null>(null)
  const [conceptOptions, setConceptOptions] = useState<ConceptOption[]>([])
  const [conceptSearch, setConceptSearch] = useState('')
  const [previewCount, setPreviewCount] = useState<number | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)

  // ─── Data Fetching ──────────────────────────────────────────
  const fetchPlaybooks = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await fetchPlaybooksService(filterCategory || undefined)
      setPlaybooks(data)
    } catch (err) {
      console.error('Failed to fetch playbooks:', err)
    } finally {
      setLoading(false)
    }
  }, [filterCategory])

  useEffect(() => { fetchPlaybooks() }, [fetchPlaybooks])

  // Fetch available concepts for picker
  useEffect(() => {
    fetchConceptsService().then(({ data }) => setConceptOptions(data)).catch(() => {})
  }, [])

  // Concept picker helpers
  const selectedConcepts = createForm.concepts
    ? createForm.concepts.split('|').map(c => c.trim()).filter(Boolean)
    : []

  const toggleConcept = (concept: string) => {
    const current = selectedConcepts
    const next = current.includes(concept)
      ? current.filter(c => c !== concept)
      : [...current, concept]
    setCreateForm(prev => ({ ...prev, concepts: next.join('|') }))
  }

  const filteredConceptOptions = conceptOptions.filter(c =>
    c.concept.toLowerCase().includes(conceptSearch.toLowerCase())
  )

  // Preview match count when concepts change
  useEffect(() => {
    if (selectedConcepts.length === 0) {
      setPreviewCount(null)
      return
    }
    let cancelled = false
    setPreviewLoading(true)
    const timer = setTimeout(async () => {
      try {
        const result = await getMatchCount(selectedConcepts)
        if (!cancelled) setPreviewCount(result.count ?? 0)
      } catch {
        if (!cancelled) setPreviewCount(null)
      } finally {
        if (!cancelled) setPreviewLoading(false)
      }
    }, 300)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [createForm.concepts])

  // ─── Activate Playbook ─────────────────────────────────────
  const openActivateModal = (pb: Playbook) => {
    setSelectedPlaybook(pb)
    setActivateForm({
      eventName: pb.name,
      eventDate: getFutureDate(14),
      matchLimit: 200,
    })
    setActivateResult(null)
    setShowActivateModal(true)
  }

  const handleActivate = async () => {
    if (!selectedPlaybook) return
    setActivating(true)
    try {
      const result = await activatePlaybook(selectedPlaybook.id, activateForm)
      setActivateResult(result)
    } catch (err) {
      console.error('Failed to activate playbook:', err)
    } finally {
      setActivating(false)
    }
  }

  // ─── Create Playbook ───────────────────────────────────────
  const handleCreate = async () => {
    if (!createForm.name.trim()) return
    setCreating(true)
    try {
      await createPlaybook({
        ...createForm,
        createdBy: userId || null,
      })
      setShowCreateModal(false)
      setCreateForm({
        name: '', description: '', concepts: '', channel: 'email',
        priority: 'medium', messageTemplate: '', category: 'custom',
      })
      fetchPlaybooks()
    } catch (err) {
      console.error('Failed to create playbook:', err)
    } finally {
      setCreating(false)
    }
  }

  // ─── Edit Playbook ─────────────────────────────────────────
  const openEditModal = (pb: Playbook) => {
    setEditingPlaybook(pb)
    setCreateForm({
      name: pb.name,
      description: pb.description,
      concepts: pb.concepts,
      channel: pb.channel,
      priority: pb.priority,
      messageTemplate: pb.messageTemplate || '',
      category: pb.category,
    })
    setShowCreateModal(true)
  }

  const handleSaveEdit = async () => {
    if (!editingPlaybook || !createForm.name.trim()) return
    setCreating(true)
    try {
      await updatePlaybook(editingPlaybook.id, {
        ...createForm,
        createdBy: userId || null,
      })
      setShowCreateModal(false)
      setEditingPlaybook(null)
      setCreateForm({
        name: '', description: '', concepts: '', channel: 'email',
        priority: 'medium', messageTemplate: '', category: 'custom',
      })
      fetchPlaybooks()
    } catch (err) {
      console.error('Failed to update playbook:', err)
    } finally {
      setCreating(false)
    }
  }

  // ─── Delete Playbook ───────────────────────────────────────
  const handleDelete = async (pb: Playbook) => {
    if (!confirm(`Delete playbook "${pb.name}"? This cannot be undone.`)) return
    setDeleting(pb.id)
    try {
      await deletePlaybook(pb.id)
      fetchPlaybooks()
    } catch (err) {
      console.error('Failed to delete playbook:', err)
    } finally {
      setDeleting(null)
    }
  }

  const closeCreateModal = () => {
    setShowCreateModal(false)
    setEditingPlaybook(null)
    setCreateForm({
      name: '', description: '', concepts: '', channel: 'email',
      priority: 'medium', messageTemplate: '', category: 'custom',
    })
  }

  // ─── Helpers ────────────────────────────────────────────────
  function getFutureDate(days: number) {
    const d = new Date()
    d.setDate(d.getDate() + days)
    return d.toISOString().split('T')[0]
  }

  // Group playbooks by category
  const categories = Array.from(new Set(playbooks.map(p => p.category)))
  const grouped = categories.reduce((acc, cat) => {
    acc[cat] = playbooks.filter(p => p.category === cat)
    return acc
  }, {} as Record<string, Playbook[]>)

  // ─── Render ─────────────────────────────────────────────────
  return (
    <div className="pb-page">
      {/* Header */}
      <div className="pb-header">
        <div className="pb-header-left">
          <h2 className="pb-title">Playbook Templates</h2>
          <p className="pb-subtitle">
            Pre-configured activation strategies. Launch a campaign in one click.
          </p>
        </div>
        <div className="pb-header-actions">
          <button
            className="pb-btn pb-btn-secondary"
            onClick={() => { setEditingPlaybook(null); setCreateForm({ name: '', description: '', concepts: '', channel: 'email', priority: 'medium', messageTemplate: '', category: 'custom' }); setShowCreateModal(true) }}
          >
            + New Playbook
          </button>
        </div>
      </div>

      {/* Category Filters */}
      <div className="pb-filters">
        <button
          className={`pb-filter-chip ${!filterCategory ? 'pb-filter-active' : ''}`}
          onClick={() => setFilterCategory('')}
        >
          All
        </button>
        {['launch', 'reengagement', 'birthday', 'seasonal', 'custom'].map(cat => (
          <button
            key={cat}
            className={`pb-filter-chip ${filterCategory === cat ? 'pb-filter-active' : ''}`}
            onClick={() => setFilterCategory(filterCategory === cat ? '' : cat)}
          >
            {categoryLabels[cat]}
          </button>
        ))}
      </div>

      {/* Playbook Grid */}
      {loading ? (
        <div className="pb-loading">
          <div className="pb-spinner" />
          <span>Loading playbooks…</span>
        </div>
      ) : playbooks.length === 0 ? (
        <div className="pb-empty">
          <p>No playbooks found. Create your first playbook template!</p>
        </div>
      ) : (
        Object.entries(grouped).map(([cat, pbs]) => (
          <div key={cat} className="pb-category-section">
            <h3 className="pb-category-title">
              {categoryLabels[cat] || cat}
              <span className="pb-category-count">{pbs.length}</span>
            </h3>
            <div className="pb-grid">
              {pbs.map(pb => (
                <div key={pb.id} className="pb-card">
                  <div className="pb-card-header">
                    <div className="pb-card-title-row">
                      <h4 className="pb-card-name">{pb.name}</h4>
                      <span
                        className="pb-priority-dot"
                        style={{ background: priorityColors[pb.priority] }}
                        title={`${pb.priority} priority`}
                      />
                    </div>
                    <p className="pb-card-desc">{pb.description}</p>
                  </div>

                  <div className="pb-card-body">
                    <div className="pb-card-meta">
                      <span className="pb-meta-item" title="Channel">
                        {channelLabels[pb.channel] || pb.channel}
                      </span>
                      <span className="pb-meta-item" title="Priority">
                        <span className="pb-priority-label" style={{
                          color: priorityColors[pb.priority],
                          fontWeight: 600,
                        }}>
                          {pb.priority}
                        </span>
                      </span>
                    </div>

                    <div className="pb-card-concepts">
                      {pb.concepts.split('|').filter(Boolean).map((c, i) => (
                        <span key={i} className="pb-concept-tag">{c.trim()}</span>
                      ))}
                    </div>

                    {pb.messageTemplate && (
                      <div className="pb-card-message">
                        <span className="pb-message-label">Template</span>
                        <p className="pb-message-text">"{pb.messageTemplate}"</p>
                      </div>
                    )}
                  </div>

                  <div className="pb-card-footer">
                    <button
                      className="pb-btn pb-btn-ghost"
                      onClick={() => openEditModal(pb)}
                    >
                      Edit
                    </button>
                    <button
                      className="pb-btn pb-btn-activate"
                      onClick={() => openActivateModal(pb)}
                    >
                      Activate
                    </button>
                    <button
                      className="pb-btn pb-btn-danger-sm"
                      onClick={() => handleDelete(pb)}
                      disabled={deleting === pb.id}
                    >
                      {deleting === pb.id ? '…' : 'Delete'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))
      )}

      {/* ─── Activate Modal ───────────────────────────────── */}
      {showActivateModal && selectedPlaybook && (
        <div className="pb-modal-overlay" onClick={() => { setShowActivateModal(false); setActivateResult(null) }}>
          <div className="pb-modal" onClick={e => e.stopPropagation()}>
            <div className="pb-modal-header">
              <h3>Activate Playbook</h3>
              <button className="pb-modal-close" onClick={() => { setShowActivateModal(false); setActivateResult(null) }}>✕</button>
            </div>

            {activateResult ? (
              <div className="pb-modal-body">
                <div className="pb-activate-success">
                  <div className="pb-success-icon">✓</div>
                  <h4>Event Created!</h4>
                  <div className="pb-success-stats">
                    <div className="pb-success-stat">
                      <span className="pb-stat-value">{activateResult.matched_count ?? 0}</span>
                      <span className="pb-stat-label">Clients Matched</span>
                    </div>
                    <div className="pb-success-stat">
                      <span className="pb-stat-value">{activateResult.notified_count ?? 0}</span>
                      <span className="pb-stat-label">Ready to Notify</span>
                    </div>
                  </div>
                  <p className="pb-success-note">
                    From playbook: <strong>{activateResult.fromPlaybook}</strong>
                  </p>
                  <div className="pb-success-actions">
                    <button
                      className="pb-btn pb-btn-primary"
                      onClick={() => {
                        setShowActivateModal(false)
                        setActivateResult(null)
                        onNavigateToCalendar?.()
                      }}
                    >
                      View in Calendar →
                    </button>
                    <button
                      className="pb-btn pb-btn-secondary"
                      onClick={() => { setShowActivateModal(false); setActivateResult(null) }}
                    >
                      Close
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="pb-modal-body">
                <div className="pb-playbook-preview">
                  <h4>{selectedPlaybook.name}</h4>
                  <p>{selectedPlaybook.description}</p>
                  <div className="pb-preview-meta">
                    <span>{channelLabels[selectedPlaybook.channel] || selectedPlaybook.channel}</span>
                    <span style={{ color: priorityColors[selectedPlaybook.priority] }}>
                      ● {selectedPlaybook.priority} priority
                    </span>
                  </div>
                  <div className="pb-preview-concepts">
                    {selectedPlaybook.concepts.split('|').filter(Boolean).map((c, i) => (
                      <span key={i} className="pb-concept-tag">{c.trim()}</span>
                    ))}
                  </div>
                </div>

                <div className="pb-form-group">
                  <label>Event Name</label>
                  <input
                    type="text"
                    value={activateForm.eventName}
                    onChange={e => setActivateForm(prev => ({ ...prev, eventName: e.target.value }))}
                    placeholder="Enter event name"
                  />
                </div>

                <div className="pb-form-group">
                  <label>Event Date</label>
                  <input
                    type="date"
                    value={activateForm.eventDate}
                    onChange={e => setActivateForm(prev => ({ ...prev, eventDate: e.target.value }))}
                  />
                </div>

                <div className="pb-form-group">
                  <label>Max Clients to Match</label>
                  <input
                    type="number"
                    value={activateForm.matchLimit}
                    min={10}
                    max={5000}
                    onChange={e => setActivateForm(prev => ({ ...prev, matchLimit: parseInt(e.target.value) || 200 }))}
                  />
                </div>

                <div className="pb-modal-actions">
                  <button
                    className="pb-btn pb-btn-primary"
                    onClick={handleActivate}
                    disabled={activating || !activateForm.eventName.trim()}
                  >
                    {activating ? 'Creating Event…' : 'Launch Activation'}
                  </button>
                  <button
                    className="pb-btn pb-btn-secondary"
                    onClick={() => setShowActivateModal(false)}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ─── Create Playbook Modal ────────────────────────── */}
      {showCreateModal && (
        <div className="pb-modal-overlay" onClick={closeCreateModal}>
          <div className="pb-modal pb-modal-wide" onClick={e => e.stopPropagation()}>
            <div className="pb-modal-header">
              <h3>{editingPlaybook ? 'Edit Playbook' : '+ New Playbook Template'}</h3>
              <button className="pb-modal-close" onClick={closeCreateModal}>✕</button>
            </div>

            <div className="pb-modal-body">
              <div className="pb-form-row">
                <div className="pb-form-group pb-form-flex">
                  <label>Name *</label>
                  <input
                    type="text"
                    value={createForm.name}
                    onChange={e => setCreateForm(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="e.g. Holiday Gift Guide"
                  />
                </div>
                <div className="pb-form-group">
                  <label>Category</label>
                  <select
                    value={createForm.category}
                    onChange={e => setCreateForm(prev => ({ ...prev, category: e.target.value }))}
                  >
                    {Object.entries(categoryLabels).map(([val, label]) => (
                      <option key={val} value={val}>{label}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="pb-form-group">
                <label>Description</label>
                <textarea
                  value={createForm.description}
                  onChange={e => setCreateForm(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe the purpose and target audience of this playbook"
                  rows={2}
                />
              </div>

              <div className="pb-form-group">
                <label>Target Concepts</label>
                <input
                  type="text"
                  placeholder="Search concepts..."
                  className="pb-concept-search"
                  value={conceptSearch}
                  onChange={e => setConceptSearch(e.target.value)}
                />
                <div className="pb-concept-list">
                  {filteredConceptOptions.slice(0, 30).map(c => (
                    <button
                      key={c.concept}
                      type="button"
                      className={`pb-concept-chip ${selectedConcepts.includes(c.concept) ? 'pb-concept-selected' : ''}`}
                      onClick={() => toggleConcept(c.concept)}
                    >
                      {c.concept}
                      <span className="pb-concept-count">{c.clientCount}</span>
                    </button>
                  ))}
                  {filteredConceptOptions.length === 0 && conceptOptions.length > 0 && (
                    <p className="pb-concept-empty">No concepts match "{conceptSearch}"</p>
                  )}
                  {conceptOptions.length === 0 && (
                    <p className="pb-concept-empty">No concepts available. Run the pipeline first.</p>
                  )}
                </div>
                {selectedConcepts.length > 0 && (
                  <div className="pb-selected-concepts">
                    <span className="pb-selected-label">Selected:</span>
                    {selectedConcepts.map(c => (
                      <span key={c} className="pb-concept-chip pb-concept-selected" onClick={() => toggleConcept(c)}>
                        {c} ✕
                      </span>
                    ))}
                  </div>
                )}
                {selectedConcepts.length > 0 && (
                  <div className="pb-match-preview">
                    {previewLoading ? (
                      <span>Estimating clients…</span>
                    ) : previewCount !== null ? (
                      <span>~<strong>{previewCount}</strong> clients match these concepts</span>
                    ) : null}
                  </div>
                )}
              </div>

              <div className="pb-form-row">
                <div className="pb-form-group">
                  <label>Channel</label>
                  <select
                    value={createForm.channel}
                    onChange={e => setCreateForm(prev => ({ ...prev, channel: e.target.value }))}
                  >
                    {['email', 'sms', 'whatsapp', 'phone', 'in_store', 'multi'].map(ch => (
                      <option key={ch} value={ch}>{channelLabels[ch]}</option>
                    ))}
                  </select>
                </div>
                <div className="pb-form-group">
                  <label>Priority</label>
                  <select
                    value={createForm.priority}
                    onChange={e => setCreateForm(prev => ({ ...prev, priority: e.target.value }))}
                  >
                    {['high', 'medium', 'low'].map(p => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="pb-form-group">
                <label>Message Template</label>
                <textarea
                  value={createForm.messageTemplate}
                  onChange={e => setCreateForm(prev => ({ ...prev, messageTemplate: e.target.value }))}
                  placeholder="Draft a personalized message template for this activation"
                  rows={3}
                />
              </div>

              <div className="pb-modal-actions">
                <button
                  className="pb-btn pb-btn-primary"
                  onClick={editingPlaybook ? handleSaveEdit : handleCreate}
                  disabled={creating || !createForm.name.trim()}
                >
                  {creating ? (editingPlaybook ? 'Saving…' : 'Creating…') : (editingPlaybook ? 'Save Changes' : 'Create Playbook')}
                </button>
                <button
                  className="pb-btn pb-btn-secondary"
                  onClick={closeCreateModal}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
