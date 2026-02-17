"""
Database schema definition and migration.

Creates all tables if they don't exist. Safe to call multiple times.
Uses IF NOT EXISTS so it's idempotent.
"""
import logging
from server.db.connection import get_sync_connection, sync_cursor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- ============================================================
-- Users & Authentication
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(100) UNIQUE NOT NULL,
    display_name    VARCHAR(200) NOT NULL,
    email           VARCHAR(255) UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(50) NOT NULL DEFAULT 'sales',
    -- role: 'admin', 'sales', 'manager', 'viewer', 'data-scientist', 'data-analyst'
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Upload Sessions (tracks each CSV or voice upload)
-- ============================================================
CREATE TABLE IF NOT EXISTS upload_sessions (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    filename        VARCHAR(500),
    upload_type     VARCHAR(50) NOT NULL DEFAULT 'csv',
    -- upload_type: 'csv', 'voice_memo'
    status          VARCHAR(50) NOT NULL DEFAULT 'uploaded',
    -- status: 'uploaded', 'processing', 'completed', 'failed'
    records_added   INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    processing_time REAL,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Clients (the core entity)
-- ============================================================
CREATE TABLE IF NOT EXISTS clients (
    id              VARCHAR(100) PRIMARY KEY,
    -- id is the client_id from the CSV (e.g., "CA001")
    segment_id      INTEGER,
    confidence      REAL DEFAULT 0.0,
    profile_type    TEXT,
    top_concepts    TEXT,
    -- pipe-separated concept list, e.g. "leather|gift|VIP"
    full_text       TEXT,
    -- original note / transcription
    language        VARCHAR(10) DEFAULT 'FR',
    note_date       DATE,
    note_duration   VARCHAR(50),
    created_by      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    -- which user/salesperson created this client
    assigned_advisor_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    -- assigned client advisor
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted      BOOLEAN NOT NULL DEFAULT FALSE
    -- soft delete for GDPR (can be hard-deleted separately)
);

CREATE INDEX IF NOT EXISTS idx_clients_segment ON clients(segment_id);
CREATE INDEX IF NOT EXISTS idx_clients_created_by ON clients(created_by);
CREATE INDEX IF NOT EXISTS idx_clients_is_deleted ON clients(is_deleted);

-- ============================================================
-- Client Concepts (many-to-many: client ↔ concept)
-- ============================================================
CREATE TABLE IF NOT EXISTS client_concepts (
    id              SERIAL PRIMARY KEY,
    client_id       VARCHAR(100) REFERENCES clients(id) ON DELETE CASCADE,
    concept_id      VARCHAR(200) NOT NULL,
    label           VARCHAR(300),
    matched_alias   VARCHAR(300),
    span_start      INTEGER DEFAULT 0,
    span_end        INTEGER DEFAULT 0,
    UNIQUE(client_id, concept_id, span_start)
);

CREATE INDEX IF NOT EXISTS idx_client_concepts_client ON client_concepts(client_id);
CREATE INDEX IF NOT EXISTS idx_client_concepts_concept ON client_concepts(concept_id);

-- ============================================================
-- Segments (cluster definitions)
-- ============================================================
CREATE TABLE IF NOT EXISTS segments (
    id              INTEGER PRIMARY KEY,
    -- cluster_id from the pipeline
    name            VARCHAR(200),
    profile         TEXT,
    full_profile    TEXT,
    client_count    INTEGER DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Client Actions (recommended actions per client)
-- ============================================================
CREATE TABLE IF NOT EXISTS client_actions (
    id              SERIAL PRIMARY KEY,
    client_id       VARCHAR(100) REFERENCES clients(id) ON DELETE CASCADE,
    action_id       VARCHAR(200),
    title           TEXT,
    channel         VARCHAR(100),
    priority        VARCHAR(50) DEFAULT 'low',
    kpi             TEXT,
    triggers        TEXT,
    rationale       TEXT,
    is_completed    BOOLEAN NOT NULL DEFAULT FALSE,
    completed_by    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_client_actions_client ON client_actions(client_id);
CREATE INDEX IF NOT EXISTS idx_client_actions_priority ON client_actions(priority);

-- ============================================================
-- Client Vectors (embeddings for 3D scatter / similarity)
-- ============================================================
CREATE TABLE IF NOT EXISTS client_vectors (
    client_id       VARCHAR(100) PRIMARY KEY REFERENCES clients(id) ON DELETE CASCADE,
    embedding       BYTEA,
    -- stored as numpy bytes for efficiency
    x_3d            REAL,
    y_3d            REAL,
    z_3d            REAL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Lexicon (concept vocabulary)
-- ============================================================
CREATE TABLE IF NOT EXISTS lexicon (
    concept_id      VARCHAR(200) PRIMARY KEY,
    label           VARCHAR(300) NOT NULL,
    aliases         TEXT,
    -- JSON array of aliases
    category        VARCHAR(200),
    frequency       INTEGER DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Audit Log (tracks who did what, for GDPR compliance)
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action          VARCHAR(100) NOT NULL,
    -- 'upload_csv', 'upload_voice', 'delete_client', 'gdpr_erase', 'login', etc.
    target_type     VARCHAR(100),
    -- 'client', 'upload', 'pipeline', etc.
    target_id       VARCHAR(200),
    details         JSONB,
    ip_address      VARCHAR(50),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);

-- ============================================================
-- Pipeline Runs (track processing history)
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              SERIAL PRIMARY KEY,
    upload_session_id INTEGER REFERENCES upload_sessions(id) ON DELETE SET NULL,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    status          VARCHAR(50) NOT NULL DEFAULT 'running',
    -- 'running', 'completed', 'failed'
    total_time      REAL,
    stage_timings   JSONB,
    records_processed INTEGER DEFAULT 0,
    new_clients     INTEGER DEFAULT 0,
    updated_clients INTEGER DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- ============================================================
-- Events / Activations Calendar
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(500) NOT NULL,
    description     TEXT,
    event_date      DATE NOT NULL,
    event_end_date  DATE,
    -- concepts this event relates to (pipe-separated, e.g. "new_products|collaboration|leather")
    concepts        TEXT NOT NULL DEFAULT '',
    -- outreach channel: 'email', 'sms', 'whatsapp', 'phone', 'in_store', 'multi'
    channel         VARCHAR(100) NOT NULL DEFAULT 'email',
    -- priority: 'high', 'medium', 'low'
    priority        VARCHAR(50) NOT NULL DEFAULT 'medium',
    -- status: 'draft', 'scheduled', 'active', 'completed', 'cancelled'
    status          VARCHAR(50) NOT NULL DEFAULT 'draft',
    -- how many clients were matched
    matched_count   INTEGER DEFAULT 0,
    notified_count  INTEGER DEFAULT 0,
    created_by      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);

-- ============================================================
-- Event Targets (matched clients for each event)
-- ============================================================
CREATE TABLE IF NOT EXISTS event_targets (
    id              SERIAL PRIMARY KEY,
    event_id        INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    client_id       VARCHAR(100) NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    -- match_reason: which concept(s) caused the match
    match_reason    TEXT,
    -- match_score: how relevant (number of matching concepts)
    match_score     REAL DEFAULT 1.0,
    -- action: 'pending', 'notified', 'responded', 'skipped'
    action_status   VARCHAR(50) NOT NULL DEFAULT 'pending',
    -- outcome: 'none', 'visited', 'purchased', 'no_response'
    outcome         VARCHAR(50) NOT NULL DEFAULT 'none',
    outcome_value   REAL DEFAULT 0.0,
    -- revenue attributed to this activation
    outcome_notes   TEXT,
    notified_at     TIMESTAMPTZ,
    responded_at    TIMESTAMPTZ,
    outcome_at      TIMESTAMPTZ,
    UNIQUE(event_id, client_id)
);

CREATE INDEX IF NOT EXISTS idx_event_targets_event ON event_targets(event_id);
CREATE INDEX IF NOT EXISTS idx_event_targets_client ON event_targets(client_id);
CREATE INDEX IF NOT EXISTS idx_event_targets_status ON event_targets(action_status);

-- ============================================================
-- Client Scores (computed engagement & value scores)
-- ============================================================
CREATE TABLE IF NOT EXISTS client_scores (
    client_id       VARCHAR(100) PRIMARY KEY REFERENCES clients(id) ON DELETE CASCADE,
    engagement_score REAL DEFAULT 0.0,
    -- 0–100: based on concepts, actions, events, recency
    value_score      REAL DEFAULT 0.0,
    -- 0–100: based on confidence, segment, action priority
    overall_score    REAL DEFAULT 0.0,
    -- weighted combination of engagement + value
    tier             VARCHAR(20) DEFAULT 'bronze',
    -- 'platinum', 'gold', 'silver', 'bronze'
    score_details    JSONB,
    -- breakdown of how scores were computed
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Playbook Templates (pre-built activation templates)
-- ============================================================
CREATE TABLE IF NOT EXISTS playbooks (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(300) NOT NULL,
    description     TEXT,
    -- concepts to match (pipe-separated)
    concepts        TEXT NOT NULL DEFAULT '',
    channel         VARCHAR(100) NOT NULL DEFAULT 'email',
    priority        VARCHAR(50) NOT NULL DEFAULT 'medium',
    -- template message
    message_template TEXT,
    -- category: 'launch', 'birthday', 'reengagement', 'seasonal', 'vip', 'custom'
    category        VARCHAR(100) NOT NULL DEFAULT 'custom',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_by      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# ---------------------------------------------------------------------------
# Default admin user (created on first init)
# ---------------------------------------------------------------------------
SEED_ADMIN_SQL = """
INSERT INTO users (username, display_name, email, password_hash, role)
VALUES ('admin', 'Administrator', 'admin@lvmh.local', '$2b$12$placeholder_hash', 'admin')
ON CONFLICT (username) DO NOTHING;
"""


def init_database():
    """Create all tables if they don't exist. Safe to call multiple times."""
    logger.info("Initializing database schema...")
    with sync_cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute(SEED_ADMIN_SQL)
    logger.info("Database schema initialized successfully")


def reset_database():
    """Drop and recreate all tables. DESTRUCTIVE — use only in development."""
    logger.warning("RESETTING DATABASE — all data will be lost!")
    tables = [
        "event_targets", "events",
        "audit_log", "pipeline_runs", "client_actions",
        "client_concepts", "client_vectors", "clients",
        "upload_sessions", "segments", "lexicon", "users"
    ]
    with sync_cursor() as cur:
        for table in tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    init_database()
    logger.info("Database reset complete")


if __name__ == "__main__":
    init_database()
    print("✅ Database schema initialized")
