"""
Neon PostgreSQL connection management.

Uses asyncpg for async FastAPI integration with Neon's serverless Postgres.
Falls back to psycopg2 for sync pipeline operations.

Connection string is read from DATABASE_URL env var or .env file.
"""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env from project root if python-dotenv is available
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from dotenv import load_dotenv
    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.info(f"Loaded .env from {_env_path}")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Connection URL
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get("DATABASE_URL", "").strip("'\"")

def _require_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add it to .env or export it: "
            "export DATABASE_URL='postgresql://user:pass@host/dbname?sslmode=require'"
        )
    return DATABASE_URL


def _asyncpg_safe_url(url: str) -> str:
    """Strip query params that asyncpg doesn't understand (e.g. channel_binding)."""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    # asyncpg handles ssl via its own kwarg; channel_binding is unsupported
    params.pop("channel_binding", None)
    clean_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=clean_query))


# ---------------------------------------------------------------------------
# Async pool (for FastAPI endpoints)  — uses asyncpg
# ---------------------------------------------------------------------------
_async_pool = None

async def get_async_pool():
    """Get or create the async connection pool."""
    global _async_pool
    if _async_pool is None:
        import asyncpg
        url = _asyncpg_safe_url(_require_url())
        _async_pool = await asyncpg.create_pool(
            url,
            min_size=2,
            max_size=10,
            command_timeout=60,
            ssl="require",
        )
        logger.info("Async connection pool created")
    return _async_pool


async def close_async_pool():
    """Close the async connection pool (call on shutdown)."""
    global _async_pool
    if _async_pool:
        await _async_pool.close()
        _async_pool = None
        logger.info("Async connection pool closed")


@asynccontextmanager
async def get_connection():
    """Async context manager that yields an asyncpg connection."""
    pool = await get_async_pool()
    async with pool.acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Sync connection (for pipeline stages that run in threads/background)
# ---------------------------------------------------------------------------
_sync_conn = None

def get_sync_connection():
    """Get a synchronous psycopg2 connection (for pipeline operations)."""
    global _sync_conn
    if _sync_conn is None or _sync_conn.closed:
        import psycopg2
        url = _require_url()
        _sync_conn = psycopg2.connect(url, sslmode="require")
        _sync_conn.autocommit = False
        logger.info("Sync DB connection established")
    return _sync_conn


def close_sync_connection():
    """Close the synchronous connection."""
    global _sync_conn
    if _sync_conn and not _sync_conn.closed:
        _sync_conn.close()
        _sync_conn = None


@contextmanager
def sync_cursor():
    """Context manager that yields a psycopg2 cursor and commits on success."""
    conn = get_sync_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
