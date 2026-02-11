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
    -- role: 'admin', 'sales', 'manager', 'viewer'
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
