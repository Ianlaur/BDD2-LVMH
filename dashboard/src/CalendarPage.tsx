import { useState, useEffect, useCallback, useRef } from 'react'
import {
  getEvents as fetchEventsService,
  getConcepts as fetchConceptsService,
  getEventDetail as fetchEventDetailService,
  createEvent,
  updateEvent,
  updateEventStatus,
  updateTarget,
  deleteEvent as deleteEventService,
  getMatchCount,
} from './services/apiService'

// ─── Types ───────────────────────────────────────────────────────
interface CalendarEvent {
  id: number
  title: string
  description: string
  event_date: string
  event_end_date?: string
  concepts: string
  channel: string
  priority: string
  status: string
  matched_count: number
  notified_count: number
  created_by?: number
  creator_name?: string
  created_at: string
  updated_at: string
  targets?: EventTarget[]
}

interface EventTarget {
  clientId: string
  matchReason: string
  matchScore: number
  actionStatus: string
  notifiedAt?: string
  respondedAt?: string
  profileType: string
  segment: number
  topConcepts: string[]
}

interface ConceptOption {
  concept: string
  clientCount: number
}

interface CalendarPageProps {
  userId?: number
}

// ─── Helpers ─────────────────────────────────────────────────────
const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

const CHANNELS = ['email', 'sms', 'whatsapp', 'phone', 'in_store', 'multi']
const PRIORITIES = ['high', 'medium', 'low']
const STATUSES = ['draft', 'scheduled', 'active', 'completed', 'cancelled']

const channelIcons: Record<string, string> = {
  email: 'EM', sms: 'SM', whatsapp: 'WA', phone: 'PH', in_store: 'IS', multi: 'ML'
}

const priorityColors: Record<string, string> = {
  high: '#ef4444', medium: '#f59e0b', low: '#22c55e'
}

const statusColors: Record<string, string> = {
  draft: '#94a3b8', scheduled: '#3b82f6', active: '#22c55e',
  completed: '#6366f1', cancelled: '#ef4444'
}

function getDaysInMonth(year: number, month: number): number {
  return new Date(year, month + 1, 0).getDate()
}

function getFirstDayOfMonth(year: number, month: number): number {
  return new Date(year, month, 1).getDay()
}

const MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const WEEKDAYS = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

// ─── Custom Date Picker ──────────────────────────────────────────
function DatePicker({ value, onChange, placeholder, required }: {
  value: string
  onChange: (val: string) => void
  placeholder?: string
  required?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [viewMonth, setViewMonth] = useState(() => {
    if (value) { const d = new Date(value + 'T00:00:00'); return d.getMonth() }
    return new Date().getMonth()
  })
  const [viewYear, setViewYear] = useState(() => {
    if (value) { const d = new Date(value + 'T00:00:00'); return d.getFullYear() }
    return new Date().getFullYear()
  })
  const ref = useRef<HTMLDivElement>(null)

  // Close on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    if (open) document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const days = getDaysInMonth(viewYear, viewMonth)
  // Monday-first: shift Sunday (0) to end
  const firstDayRaw = new Date(viewYear, viewMonth, 1).getDay()
  const firstDay = firstDayRaw === 0 ? 6 : firstDayRaw - 1

  const today = new Date()
  const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`

  const selectDay = (day: number) => {
    const dateStr = `${viewYear}-${String(viewMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    onChange(dateStr)
    setOpen(false)
  }

  const prevMonth = () => {
    if (viewMonth === 0) { setViewMonth(11); setViewYear(y => y - 1) }
    else setViewMonth(m => m - 1)
  }
  const nextMonth = () => {
    if (viewMonth === 11) { setViewMonth(0); setViewYear(y => y + 1) }
    else setViewMonth(m => m + 1)
  }

  const displayValue = value
    ? new Date(value + 'T00:00:00').toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' })
    : ''

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation()
    onChange('')
  }

  return (
    <div className="dp-wrapper" ref={ref}>
      <button
        type="button"
        className={`dp-trigger ${open ? 'dp-trigger-open' : ''} ${!value ? 'dp-trigger-empty' : ''}`}
        onClick={() => setOpen(!open)}
      >
        <svg className="dp-icon" viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
          <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zM4 8h12v8H4V8z" clipRule="evenodd" />
        </svg>
        <span className="dp-value">{displayValue || placeholder || 'Pick a date'}</span>
        {value && !required && (
          <span className="dp-clear" onClick={clear}>✕</span>
        )}
      </button>

      {open && (
        <div className="dp-dropdown">
          <div className="dp-nav">
            <button type="button" className="dp-nav-btn" onClick={prevMonth}>‹</button>
            <span className="dp-nav-label">{MONTHS_SHORT[viewMonth]} {viewYear}</span>
            <button type="button" className="dp-nav-btn" onClick={nextMonth}>›</button>
          </div>
          <div className="dp-grid">
            {WEEKDAYS.map(wd => (
              <span key={wd} className="dp-weekday">{wd}</span>
            ))}
            {Array.from({ length: firstDay }, (_, i) => (
              <span key={`e-${i}`} className="dp-day dp-day-empty" />
            ))}
            {Array.from({ length: days }, (_, i) => {
              const day = i + 1
              const dateStr = `${viewYear}-${String(viewMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
              const isSelected = dateStr === value
              const isToday = dateStr === todayStr
              return (
                <button
                  key={day}
                  type="button"
                  className={`dp-day ${isSelected ? 'dp-day-selected' : ''} ${isToday ? 'dp-day-today' : ''}`}
                  onClick={() => selectDay(day)}
                >
                  {day}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Component ───────────────────────────────────────────────────
export default function CalendarPage({ userId }: CalendarPageProps) {
  const now = new Date()
  const [currentMonth, setCurrentMonth] = useState(now.getMonth())
  const [currentYear, setCurrentYear] = useState(now.getFullYear())
  const [events, setEvents] = useState<CalendarEvent[]>([])
  const [conceptOptions, setConceptOptions] = useState<ConceptOption[]>([])
  const [loading, setLoading] = useState(false)

  // Modal states
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedDate, setSelectedDate] = useState<string>('')
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null)
  const [showEventDetail, setShowEventDetail] = useState(false)

  // Form state
  const [form, setForm] = useState({
    title: '',
    description: '',
    event_date: '',
    event_end_date: '',
    concepts: [] as string[],
    channel: 'email',
    priority: 'medium',
  })
  const [conceptSearch, setConceptSearch] = useState('')
  const [creating, setCreating] = useState(false)
  const [editingEvent, setEditingEvent] = useState<CalendarEvent | null>(null)
  const [saving, setSaving] = useState(false)
  const [previewCount, setPreviewCount] = useState<number | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)

  // ─── Data Fetching ──────────────────────────────────────────
  const fetchEvents = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await fetchEventsService(currentMonth + 1, currentYear)
      setEvents(data)
    } catch (err) {
      console.error('Failed to fetch events:', err)
    } finally {
      setLoading(false)
    }
  }, [currentMonth, currentYear])

  const fetchConcepts = useCallback(async () => {
    try {
      const { data } = await fetchConceptsService()
      setConceptOptions(data)
    } catch (err) {
      console.error('Failed to fetch concepts:', err)
    }
  }, [])

  useEffect(() => { fetchEvents() }, [fetchEvents])
  useEffect(() => { fetchConcepts() }, [fetchConcepts])

  // Preview match count when concepts change
  useEffect(() => {
    if (form.concepts.length === 0) {
      setPreviewCount(null)
      return
    }
    let cancelled = false
    setPreviewLoading(true)
    const timer = setTimeout(async () => {
      try {
        const result = await getMatchCount(form.concepts)
        if (!cancelled) setPreviewCount(result.count ?? 0)
      } catch {
        if (!cancelled) setPreviewCount(null)
      } finally {
        if (!cancelled) setPreviewLoading(false)
      }
    }, 300)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [form.concepts])

  // ─── Event Creation ─────────────────────────────────────────
  const handleCreateEvent = async () => {
    if (!form.title.trim() || !form.event_date) return
    setCreating(true)
    try {
      const created = await createEvent({
        ...form,
        concepts: form.concepts.join('|'),
        created_by: userId || null,
      })
      setShowCreateModal(false)
      resetForm()
      fetchEvents()
      // Auto open detail if clients matched
      if (created.matched_count > 0) {
        handleOpenEvent(created.id)
      }
    } catch (err) {
      console.error('Failed to create event:', err)
    } finally {
      setCreating(false)
    }
  }

  const resetForm = () => {
    setForm({
      title: '', description: '', event_date: '', event_end_date: '',
      concepts: [], channel: 'email', priority: 'medium',
    })
    setConceptSearch('')
  }

  // ─── Edit Event ─────────────────────────────────────────────
  const openEditModal = (event: CalendarEvent) => {
    setEditingEvent(event)
    setForm({
      title: event.title,
      description: event.description || '',
      event_date: event.event_date,
      event_end_date: event.event_end_date || '',
      concepts: event.concepts ? event.concepts.split('|').filter(Boolean).map(c => c.trim()) : [],
      channel: event.channel,
      priority: event.priority,
    })
    setConceptSearch('')
    setShowEventDetail(false)
    setShowCreateModal(true)
  }

  const handleSaveEdit = async () => {
    if (!editingEvent || !form.title.trim() || !form.event_date) return
    setSaving(true)
    try {
      await updateEvent(editingEvent.id, {
        ...form,
        concepts: form.concepts.join('|'),
      })
      setShowCreateModal(false)
      setEditingEvent(null)
      resetForm()
      fetchEvents()
    } catch (err) {
      console.error('Failed to update event:', err)
    } finally {
      setSaving(false)
    }
  }

  const closeCreateModal = () => {
    setShowCreateModal(false)
    setEditingEvent(null)
    resetForm()
  }

  // ─── Event Detail ───────────────────────────────────────────
  const handleOpenEvent = async (eventId: number) => {
    try {
      const { data } = await fetchEventDetailService(eventId)
      setSelectedEvent(data)
      setShowEventDetail(true)
    } catch (err) {
      console.error('Failed to fetch event:', err)
    }
  }

  const handleUpdateStatus = async (eventId: number, status: string) => {
    try {
      await updateEventStatus(eventId, status)
      fetchEvents()
      if (selectedEvent && selectedEvent.id === eventId) {
        handleOpenEvent(eventId)
      }
    } catch (err) {
      console.error('Failed to update event:', err)
    }
  }

  const handleUpdateTarget = async (eventId: number, clientId: string, actionStatus: string) => {
    try {
      await updateTarget(eventId, clientId, actionStatus)
      handleOpenEvent(eventId) // Refresh detail
    } catch (err) {
      console.error('Failed to update target:', err)
    }
  }

  const handleDeleteEvent = async (eventId: number) => {
    if (!confirm('Delete this event and all associated targets?')) return
    try {
      await deleteEventService(eventId)
      setShowEventDetail(false)
      setSelectedEvent(null)
      fetchEvents()
    } catch (err) {
      console.error('Failed to delete event:', err)
    }
  }

  // ─── Concept Picker Logic ──────────────────────────────────
  const toggleConcept = (concept: string) => {
    setForm(prev => ({
      ...prev,
      concepts: prev.concepts.includes(concept)
        ? prev.concepts.filter(c => c !== concept)
        : [...prev.concepts, concept]
    }))
  }

  const filteredConcepts = conceptOptions.filter(c =>
    c.concept.toLowerCase().includes(conceptSearch.toLowerCase())
  )

  // ─── Calendar Grid ─────────────────────────────────────────
  const daysInMonth = getDaysInMonth(currentYear, currentMonth)
  const firstDay = getFirstDayOfMonth(currentYear, currentMonth)
  const today = new Date()
  const isToday = (day: number) =>
    day === today.getDate() && currentMonth === today.getMonth() && currentYear === today.getFullYear()

  const getEventsForDay = (day: number): CalendarEvent[] => {
    const dateStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    return events.filter(e => {
      const start = e.event_date
      const end = e.event_end_date || e.event_date
      return dateStr >= start && dateStr <= end
    })
  }

  const handleDayClick = (day: number) => {
    const dateStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    setSelectedDate(dateStr)
    setForm(prev => ({ ...prev, event_date: dateStr }))
    setShowCreateModal(true)
  }

  const prevMonth = () => {
    if (currentMonth === 0) { setCurrentMonth(11); setCurrentYear(y => y - 1) }
    else setCurrentMonth(m => m - 1)
  }

  const nextMonth = () => {
    if (currentMonth === 11) { setCurrentMonth(0); setCurrentYear(y => y + 1) }
    else setCurrentMonth(m => m + 1)
  }

  // ─── Render ─────────────────────────────────────────────────
  return (
    <div className="calendar-page">
      {/* Header */}
      <div className="calendar-header">
        <div className="calendar-header-left">
          <h2>Activation Calendar</h2>
          <span className="calendar-subtitle">Plan events and target matching clients</span>
        </div>
        <div className="calendar-header-right">
          <button className="cal-btn cal-btn-primary" onClick={() => { resetForm(); setEditingEvent(null); setShowCreateModal(true) }}>
            + New Event
          </button>
        </div>
      </div>

      {/* Month navigation */}
      <div className="calendar-nav">
        <button className="cal-btn cal-btn-ghost" onClick={prevMonth}>‹</button>
        <h3>{MONTHS[currentMonth]} {currentYear}</h3>
        <button className="cal-btn cal-btn-ghost" onClick={nextMonth}>›</button>
        <button className="cal-btn cal-btn-sm" onClick={() => { setCurrentMonth(now.getMonth()); setCurrentYear(now.getFullYear()) }}>
          Today
        </button>
      </div>

      {/* Calendar grid */}
      <div className="calendar-grid">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(d => (
          <div key={d} className="calendar-day-header">{d}</div>
        ))}

        {/* Empty cells before first day */}
        {Array.from({ length: firstDay }, (_, i) => (
          <div key={`empty-${i}`} className="calendar-cell calendar-cell-empty" />
        ))}

        {/* Day cells */}
        {Array.from({ length: daysInMonth }, (_, i) => {
          const day = i + 1
          const dayEvents = getEventsForDay(day)
          return (
            <div
              key={day}
              className={`calendar-cell ${isToday(day) ? 'calendar-cell-today' : ''} ${dayEvents.length > 0 ? 'calendar-cell-has-events' : ''}`}
              onClick={() => handleDayClick(day)}
            >
              <span className="calendar-day-number">{day}</span>
              <div className="calendar-cell-events">
                {dayEvents.slice(0, 3).map(ev => (
                  <div
                    key={ev.id}
                    className="calendar-event-pill"
                    style={{ borderLeft: `3px solid ${priorityColors[ev.priority] || '#94a3b8'}` }}
                    onClick={(e) => { e.stopPropagation(); handleOpenEvent(ev.id) }}
                  >
                    <span className="event-pill-icon">{channelIcons[ev.channel] || '—'}</span>
                    <span className="event-pill-title">{ev.title}</span>
                    {ev.matched_count > 0 && (
                      <span className="event-pill-badge">{ev.matched_count}</span>
                    )}
                  </div>
                ))}
                {dayEvents.length > 3 && (
                  <span className="calendar-cell-more">+{dayEvents.length - 3} more</span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Upcoming events list */}
      {events.length > 0 && (
        <div className="calendar-upcoming">
          <h3>Events this month</h3>
          <div className="calendar-event-list">
            {events.map(ev => (
              <div key={ev.id} className="calendar-event-card" onClick={() => handleOpenEvent(ev.id)}>
                <div className="event-card-left">
                  <div className="event-card-date">
                    {new Date(ev.event_date + 'T00:00:00').toLocaleDateString('en-US', { day: 'numeric', month: 'short' })}
                  </div>
                  <span className="event-card-status" style={{ background: statusColors[ev.status] }}>
                    {ev.status}
                  </span>
                </div>
                <div className="event-card-body">
                  <h4>{ev.title}</h4>
                  <p className="event-card-desc">{ev.description}</p>
                  <div className="event-card-tags">
                    {ev.concepts.split('|').filter(Boolean).map(c => (
                      <span key={c} className="event-concept-tag">{c.trim()}</span>
                    ))}
                  </div>
                </div>
                <div className="event-card-right">
                  <span className="event-channel-badge">{channelIcons[ev.channel]} {ev.channel}</span>
                  <div className="event-match-stats">
                    <span className="event-stat"><strong>{ev.matched_count}</strong> matched</span>
                    <span className="event-stat"><strong>{ev.notified_count}</strong> notified</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {loading && <div className="calendar-loading">Loading events...</div>}

      {/* ─── CREATE / EDIT EVENT MODAL ─────────────────────────── */}
      {showCreateModal && (
        <div className="cal-modal-overlay" onClick={closeCreateModal}>
          <div className="cal-modal" onClick={e => e.stopPropagation()}>
            <div className="cal-modal-header">
              <h3>{editingEvent ? 'Edit Event' : 'New Activation Event'}</h3>
              <button className="cal-modal-close" onClick={closeCreateModal}>✕</button>
            </div>

            <div className="cal-modal-body">
              {/* Title */}
              <div className="cal-field">
                <label>Event Title *</label>
                <input
                  type="text"
                  placeholder="e.g. Spring Collaboration Launch"
                  value={form.title}
                  onChange={e => setForm(prev => ({ ...prev, title: e.target.value }))}
                />
              </div>

              {/* Description */}
              <div className="cal-field">
                <label>Description</label>
                <textarea
                  placeholder="Describe the activation..."
                  rows={3}
                  value={form.description}
                  onChange={e => setForm(prev => ({ ...prev, description: e.target.value }))}
                />
              </div>

              {/* Dates */}
              <div className="cal-field-row">
                <div className="cal-field">
                  <label>Start Date *</label>
                  <DatePicker
                    value={form.event_date}
                    onChange={val => setForm(prev => ({ ...prev, event_date: val }))}
                    placeholder="Pick start date"
                    required
                  />
                </div>
                <div className="cal-field">
                  <label>End Date</label>
                  <DatePicker
                    value={form.event_end_date}
                    onChange={val => setForm(prev => ({ ...prev, event_end_date: val }))}
                    placeholder="Optional end date"
                  />
                </div>
              </div>

              {/* Channel & Priority */}
              <div className="cal-field-row">
                <div className="cal-field">
                  <label>Channel</label>
                  <select
                    value={form.channel}
                    onChange={e => setForm(prev => ({ ...prev, channel: e.target.value }))}
                  >
                    {CHANNELS.map(ch => (
                      <option key={ch} value={ch}>{channelIcons[ch]} {ch.replace('_', ' ')}</option>
                    ))}
                  </select>
                </div>
                <div className="cal-field">
                  <label>Priority</label>
                  <select
                    value={form.priority}
                    onChange={e => setForm(prev => ({ ...prev, priority: e.target.value }))}
                  >
                    {PRIORITIES.map(p => (
                      <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Concept Picker */}
              <div className="cal-field">
                <label>
                  Target Concepts
                  <span className="cal-field-hint">Select concepts to match with interested clients</span>
                </label>
                <input
                  type="text"
                  placeholder="Search concepts..."
                  className="cal-concept-search"
                  value={conceptSearch}
                  onChange={e => setConceptSearch(e.target.value)}
                />
                <div className="cal-concept-list">
                  {filteredConcepts.slice(0, 30).map(c => (
                    <button
                      key={c.concept}
                      className={`cal-concept-chip ${form.concepts.includes(c.concept) ? 'selected' : ''}`}
                      onClick={() => toggleConcept(c.concept)}
                    >
                      {c.concept}
                      <span className="cal-concept-count">{c.clientCount}</span>
                    </button>
                  ))}
                  {filteredConcepts.length === 0 && conceptOptions.length === 0 && (
                    <p className="cal-concept-empty">No concepts available. Run the pipeline first to generate client concepts.</p>
                  )}
                  {filteredConcepts.length === 0 && conceptOptions.length > 0 && (
                    <p className="cal-concept-empty">No concepts match "{conceptSearch}"</p>
                  )}
                </div>
                {form.concepts.length > 0 && (
                  <div className="cal-selected-concepts">
                    <span className="cal-selected-label">Selected:</span>
                    {form.concepts.map(c => (
                      <span key={c} className="cal-concept-chip selected" onClick={() => toggleConcept(c)}>
                        {c} ✕
                      </span>
                    ))}
                  </div>
                )}
                {form.concepts.length > 0 && (
                  <div className="cal-match-preview">
                    {previewLoading ? (
                      <span className="cal-match-preview-text">Estimating clients…</span>
                    ) : previewCount !== null ? (
                      <span className="cal-match-preview-text">
                        ~<strong>{previewCount}</strong> clients match these concepts
                      </span>
                    ) : null}
                  </div>
                )}
              </div>
            </div>

            <div className="cal-modal-footer">
              <button className="cal-btn cal-btn-ghost" onClick={closeCreateModal}>Cancel</button>
              {editingEvent ? (
                <button
                  className="cal-btn cal-btn-primary"
                  disabled={!form.title.trim() || !form.event_date || saving}
                  onClick={handleSaveEdit}
                >
                  {saving ? 'Saving…' : 'Save Changes'}
                </button>
              ) : (
                <button
                  className="cal-btn cal-btn-primary"
                  disabled={!form.title.trim() || !form.event_date || creating}
                  onClick={handleCreateEvent}
                >
                  {creating ? 'Creating & Matching...' : 'Create Event & Match Clients'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ─── EVENT DETAIL MODAL ───────────────────────────────── */}
      {showEventDetail && selectedEvent && (
        <div className="cal-modal-overlay" onClick={() => setShowEventDetail(false)}>
          <div className="cal-modal cal-modal-lg" onClick={e => e.stopPropagation()}>
            <div className="cal-modal-header">
              <div>
                <h3>{selectedEvent.title}</h3>
                <span className="event-detail-date">
                  {new Date(selectedEvent.event_date + 'T00:00:00').toLocaleDateString('en-US', {
                    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
                  })}
                  {selectedEvent.event_end_date && selectedEvent.event_end_date !== selectedEvent.event_date && (
                    <> → {new Date(selectedEvent.event_end_date + 'T00:00:00').toLocaleDateString('en-US', {
                      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
                    })}</>
                  )}
                </span>
              </div>
              <button className="cal-modal-close" onClick={() => setShowEventDetail(false)}>✕</button>
            </div>

            <div className="cal-modal-body">
              {/* Event Meta */}
              <div className="event-detail-meta">
                <span className="event-card-status" style={{ background: statusColors[selectedEvent.status] }}>
                  {selectedEvent.status}
                </span>
                <span className="event-channel-badge">{channelIcons[selectedEvent.channel]} {selectedEvent.channel}</span>
                <span className="event-priority-badge" style={{ color: priorityColors[selectedEvent.priority] }}>
                  ● {selectedEvent.priority} priority
                </span>
                {selectedEvent.creator_name && (
                  <span className="event-creator">Created by {selectedEvent.creator_name}</span>
                )}
              </div>

              {selectedEvent.description && (
                <p className="event-detail-desc">{selectedEvent.description}</p>
              )}

              {/* Concepts */}
              <div className="event-detail-concepts">
                <h4>Target Concepts</h4>
                <div className="event-card-tags">
                  {selectedEvent.concepts.split('|').filter(Boolean).map(c => (
                    <span key={c} className="event-concept-tag">{c.trim()}</span>
                  ))}
                </div>
              </div>

              {/* Status Actions */}
              <div className="event-detail-actions">
                <h4>Event Status</h4>
                <div className="event-status-btns">
                  {STATUSES.map(s => (
                    <button
                      key={s}
                      className={`cal-btn cal-btn-sm ${selectedEvent.status === s ? 'cal-btn-active' : 'cal-btn-ghost'}`}
                      style={selectedEvent.status === s ? { background: statusColors[s], color: '#fff' } : {}}
                      onClick={() => handleUpdateStatus(selectedEvent.id, s)}
                    >
                      {s}
                    </button>
                  ))}
                  <button
                    className="cal-btn cal-btn-sm cal-btn-primary"
                    onClick={() => openEditModal(selectedEvent)}
                  >
                    Edit
                  </button>
                  <button
                    className="cal-btn cal-btn-sm cal-btn-danger"
                    onClick={() => handleDeleteEvent(selectedEvent.id)}
                  >
                    Delete
                  </button>
                </div>
              </div>

              {/* Matched Clients */}
              <div className="event-detail-targets">
                <h4>
                  Matched Clients
                  <span className="event-match-summary">
                    {selectedEvent.matched_count} matched · {selectedEvent.notified_count} notified
                  </span>
                </h4>

                {selectedEvent.targets && selectedEvent.targets.length > 0 ? (
                  <div className="event-target-list">
                    <div className="event-target-header">
                      <span>Client</span>
                      <span>Match Reason</span>
                      <span>Score</span>
                      <span>Status</span>
                      <span>Action</span>
                    </div>
                    {selectedEvent.targets.map(t => (
                      <div key={t.clientId} className="event-target-row">
                        <div className="target-client">
                          <strong>{t.clientId}</strong>
                          <span className="target-profile">{t.profileType}</span>
                        </div>
                        <div className="target-reason">{t.matchReason}</div>
                        <div className="target-score">
                          <div className="score-bar">
                            <div className="score-fill" style={{ width: `${Math.round(t.matchScore * 100)}%` }} />
                          </div>
                          <span>{Math.round(t.matchScore * 100)}%</span>
                        </div>
                        <div>
                          <span
                            className="target-status-badge"
                            style={{
                              background: t.actionStatus === 'notified' ? '#3b82f6'
                                : t.actionStatus === 'responded' ? '#22c55e'
                                : t.actionStatus === 'skipped' ? '#ef4444'
                                : '#94a3b8'
                            }}
                          >
                            {t.actionStatus}
                          </span>
                        </div>
                        <div className="target-actions">
                          {t.actionStatus === 'pending' && (
                            <>
                              <button
                                className="cal-btn cal-btn-xs cal-btn-primary"
                                onClick={() => handleUpdateTarget(selectedEvent.id, t.clientId, 'notified')}
                              >
                                {channelIcons[selectedEvent.channel]} Notify
                              </button>
                              <button
                                className="cal-btn cal-btn-xs cal-btn-ghost"
                                onClick={() => handleUpdateTarget(selectedEvent.id, t.clientId, 'skipped')}
                              >
                                Skip
                              </button>
                            </>
                          )}
                          {t.actionStatus === 'notified' && (
                            <button
                              className="cal-btn cal-btn-xs"
                              style={{ background: '#22c55e', color: '#fff' }}
                              onClick={() => handleUpdateTarget(selectedEvent.id, t.clientId, 'responded')}
                            >
                              ✓ Responded
                            </button>
                          )}
                          {(t.actionStatus === 'skipped' || t.actionStatus === 'responded') && (
                            <button
                              className="cal-btn cal-btn-xs cal-btn-ghost"
                              onClick={() => handleUpdateTarget(selectedEvent.id, t.clientId, 'pending')}
                            >
                              Reset
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="event-no-targets">
                    <p>No clients matched for this event.</p>
                    <p className="event-no-targets-hint">
                      Try adding more concepts or running the pipeline to enrich client profiles.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
