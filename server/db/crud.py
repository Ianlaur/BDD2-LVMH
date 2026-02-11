"""
Database CRUD operations for the LVMH pipeline.

Provides both sync (for pipeline stages) and async (for FastAPI) methods.
All writes go through here so we have a single source of truth.

Uses psycopg2.extras.execute_values for fast bulk inserts over Neon.
"""
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ===================================================================
# SYNC operations (used by pipeline stages running in background)
# ===================================================================

def sync_upsert_clients(profiles_df, concepts_df, notes_df=None, actions_df=None,
                        user_id: Optional[int] = None):
    """
    Upsert client profiles into the database after a pipeline run.
    Uses bulk operations for speed over remote Neon connections.
    """
    from server.db.connection import sync_cursor
    from psycopg2.extras import execute_values
    import pandas as pd

    # Build lookup for transcript data
    transcript_lookup = {}
    if notes_df is not None:
        for _, row in notes_df.iterrows():
            nid = str(row.get("note_id", row.get("client_id", "")))
            transcript_lookup[nid] = {
                "text": str(row.get("text", "")),
                "date": str(row.get("date", "")),
                "language": str(row.get("language", "FR")),
                "duration": str(row.get("duration", "")),
            }

    # ---- Prepare client rows ----
    client_values = []
    processed_client_ids = []

    for _, row in profiles_df.iterrows():
        client_id = str(row["client_id"])
        processed_client_ids.append(client_id)
        cluster_id = int(row.get("cluster_id", 0))
        confidence = float(row.get("confidence", 0.5))
        profile_type = str(row.get("profile_type", ""))
        top_concepts = str(row.get("top_concepts", ""))

        t_info = transcript_lookup.get(client_id, {})
        full_text = t_info.get("text", "")
        language = t_info.get("language", "FR")
        note_date_str = t_info.get("date", "")
        note_duration = t_info.get("duration", "")

        note_date = None
        if note_date_str:
            try:
                note_date = pd.to_datetime(note_date_str).date()
            except Exception:
                note_date = None

        client_values.append((
            client_id, cluster_id, confidence, profile_type, top_concepts,
            full_text, language, note_date, note_duration, user_id
        ))

    # ---- Prepare concept rows ----
    concept_values = []
    if concepts_df is not None and not concepts_df.empty:
        processed_set = set(processed_client_ids)
        for _, row in concepts_df.iterrows():
            cid = str(row.get("client_id", ""))
            if cid not in processed_set:
                continue
            concept_values.append((
                cid,
                str(row.get("concept_id", "")),
                str(row.get("label", "")),
                str(row.get("matched_alias", "")),
                int(row.get("span_start", 0)) if pd.notna(row.get("span_start")) else 0,
                int(row.get("span_end", 0)) if pd.notna(row.get("span_end")) else 0,
            ))

    # ---- Prepare action rows ----
    action_values = []
    if actions_df is not None and not actions_df.empty:
        processed_set = set(processed_client_ids)
        for _, row in actions_df.iterrows():
            cid = str(row.get("client_id", ""))
            if cid not in processed_set:
                continue
            action_values.append((
                cid,
                str(row.get("action_id", "")),
                str(row.get("title", "")),
                str(row.get("channel", "")),
                str(row.get("priority", "low")).lower(),
                str(row.get("kpi", "")),
                str(row.get("triggers", "")),
                str(row.get("rationale", "")),
            ))

    # ---- Execute everything in one transaction ----
    new_count = 0
    updated_count = 0

    with sync_cursor() as cur:
        # 1. Bulk upsert clients
        if client_values:
            results = execute_values(cur, """
                INSERT INTO clients (id, segment_id, confidence, profile_type, top_concepts,
                                     full_text, language, note_date, note_duration, created_by, updated_at)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    segment_id = EXCLUDED.segment_id,
                    confidence = EXCLUDED.confidence,
                    profile_type = EXCLUDED.profile_type,
                    top_concepts = EXCLUDED.top_concepts,
                    full_text = COALESCE(EXCLUDED.full_text, clients.full_text),
                    language = COALESCE(EXCLUDED.language, clients.language),
                    note_date = COALESCE(EXCLUDED.note_date, clients.note_date),
                    note_duration = COALESCE(EXCLUDED.note_duration, clients.note_duration),
                    updated_at = NOW(),
                    is_deleted = FALSE
                RETURNING (xmax = 0) AS is_insert
            """, client_values, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                fetch=True, page_size=500)

            for (is_insert,) in results:
                if is_insert:
                    new_count += 1
                else:
                    updated_count += 1

            logger.info(f"Clients upserted: {new_count} new, {updated_count} updated")

        # 2. Replace concepts for processed clients (bulk delete + bulk insert)
        if processed_client_ids:
            cur.execute(
                "DELETE FROM client_concepts WHERE client_id = ANY(%s)",
                (processed_client_ids,)
            )

        if concept_values:
            execute_values(cur, """
                INSERT INTO client_concepts (client_id, concept_id, label, matched_alias, span_start, span_end)
                VALUES %s
                ON CONFLICT (client_id, concept_id, span_start) DO NOTHING
            """, concept_values, page_size=1000)
            logger.info(f"Concepts inserted: {len(concept_values)} rows")

        # 3. Replace actions for processed clients
        if processed_client_ids:
            cur.execute(
                "DELETE FROM client_actions WHERE client_id = ANY(%s) AND is_completed = FALSE",
                (processed_client_ids,)
            )

        if action_values:
            execute_values(cur, """
                INSERT INTO client_actions (client_id, action_id, title, channel, priority, kpi, triggers, rationale)
                VALUES %s
            """, action_values, page_size=1000)
            logger.info(f"Actions inserted: {len(action_values)} rows")

    logger.info(f"DB upsert complete: {new_count} new, {updated_count} updated, {len(processed_client_ids)} total")
    return {"new_clients": new_count, "updated_clients": updated_count, "total": len(processed_client_ids)}


def sync_upsert_segments(profiles_df):
    """Upsert segment definitions from pipeline output."""
    from server.db.connection import sync_cursor
    import pandas as pd

    segment_counts = profiles_df["cluster_id"].value_counts().sort_index()

    with sync_cursor() as cur:
        for cluster_id, count in segment_counts.items():
            cluster_rows = profiles_df[profiles_df["cluster_id"] == cluster_id]
            profile_type = str(cluster_rows["profile_type"].iloc[0])
            top_concepts = str(cluster_rows["top_concepts"].iloc[0])

            if pd.notna(top_concepts) and top_concepts:
                concepts_list = top_concepts.split("|")[:3]
                profile_short = " | ".join(concepts_list)
            else:
                profile_short = profile_type.split(" | ")[0] if profile_type else f"Segment {cluster_id}"

            cur.execute("""
                INSERT INTO segments (id, name, profile, full_profile, client_count, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    profile = EXCLUDED.profile,
                    full_profile = EXCLUDED.full_profile,
                    client_count = EXCLUDED.client_count,
                    updated_at = NOW()
            """, (
                int(cluster_id),
                f"Segment {cluster_id}",
                profile_short,
                profile_type,
                int(count),
            ))

    logger.info(f"Upserted {len(segment_counts)} segments")


def sync_upsert_vectors(vectors_df):
    """Store client embeddings and 3D coordinates (bulk)."""
    from server.db.connection import sync_cursor
    from psycopg2.extras import execute_values
    import numpy as np

    values = []
    for _, row in vectors_df.iterrows():
        client_id = str(row["client_id"])
        embedding = row.get("embedding")
        embedding_bytes = embedding.tobytes() if isinstance(embedding, np.ndarray) else None
        values.append((client_id, embedding_bytes))

    if not values:
        return

    with sync_cursor() as cur:
        execute_values(cur, """
            INSERT INTO client_vectors (client_id, embedding, updated_at)
            VALUES %s
            ON CONFLICT (client_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """, values, template="(%s, %s, NOW())", page_size=500)

    logger.info(f"Upserted {len(values)} client vectors")


def sync_update_3d_coords(scatter3d_list: list):
    """Update 3D coordinates from UMAP projection (bulk)."""
    from server.db.connection import sync_cursor
    from psycopg2.extras import execute_values

    if not scatter3d_list:
        return

    values = [
        (float(p["x"]), float(p["y"]), float(p["z"]), str(p["client"]))
        for p in scatter3d_list
    ]

    with sync_cursor() as cur:
        # Use a temp table approach for bulk update
        execute_values(cur, """
            UPDATE client_vectors AS cv SET
                x_3d = v.x, y_3d = v.y, z_3d = v.z, updated_at = NOW()
            FROM (VALUES %s) AS v(x, y, z, cid)
            WHERE cv.client_id = v.cid
        """, values, template="(%s::real, %s::real, %s::real, %s)")

    logger.info(f"Updated 3D coordinates for {len(scatter3d_list)} clients")


def sync_upsert_lexicon(lexicon_df):
    """Upsert the concept lexicon (bulk)."""
    from server.db.connection import sync_cursor
    from psycopg2.extras import execute_values

    values = []
    for _, row in lexicon_df.iterrows():
        values.append((
            str(row.get("concept_id", "")),
            str(row.get("label", "")),
            str(row.get("aliases", "")) if "aliases" in row else None,
            str(row.get("category", "")) if "category" in row else None,
        ))

    if not values:
        return

    with sync_cursor() as cur:
        execute_values(cur, """
            INSERT INTO lexicon (concept_id, label, aliases, category, updated_at)
            VALUES %s
            ON CONFLICT (concept_id) DO UPDATE SET
                label = EXCLUDED.label,
                aliases = COALESCE(EXCLUDED.aliases, lexicon.aliases),
                category = COALESCE(EXCLUDED.category, lexicon.category),
                updated_at = NOW()
        """, values, template="(%s, %s, %s, %s, NOW())", page_size=500)

    logger.info(f"Upserted {len(values)} lexicon entries")


def sync_log_pipeline_run(upload_session_id=None, user_id=None, status="running"):
    """Create a pipeline run record, returns the run ID."""
    from server.db.connection import sync_cursor

    with sync_cursor() as cur:
        cur.execute("""
            INSERT INTO pipeline_runs (upload_session_id, user_id, status, started_at)
            VALUES (%s, %s, %s, NOW())
            RETURNING id
        """, (upload_session_id, user_id, status))
        return cur.fetchone()[0]


def sync_complete_pipeline_run(run_id: int, status: str, total_time: float,
                               stage_timings: dict, records_processed: int,
                               new_clients: int, updated_clients: int,
                               error_message: str = None):
    """Update a pipeline run record on completion."""
    from server.db.connection import sync_cursor

    with sync_cursor() as cur:
        cur.execute("""
            UPDATE pipeline_runs
            SET status = %s, total_time = %s, stage_timings = %s,
                records_processed = %s, new_clients = %s, updated_clients = %s,
                error_message = %s, completed_at = NOW()
            WHERE id = %s
        """, (status, total_time, json.dumps(stage_timings),
              records_processed, new_clients, updated_clients,
              error_message, run_id))


def sync_log_audit(user_id: int, action: str, target_type: str = None,
                   target_id: str = None, details: dict = None, ip: str = None):
    """Insert an audit log entry."""
    from server.db.connection import sync_cursor

    with sync_cursor() as cur:
        cur.execute("""
            INSERT INTO audit_log (user_id, action, target_type, target_id, details, ip_address)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, action, target_type, target_id,
              json.dumps(details) if details else None, ip))


def sync_soft_delete_client(client_id: str):
    """Soft-delete a client (mark as deleted, keep data for GDPR audit trail)."""
    from server.db.connection import sync_cursor

    with sync_cursor() as cur:
        cur.execute("""
            UPDATE clients SET is_deleted = TRUE, updated_at = NOW()
            WHERE id = %s
            RETURNING id
        """, (client_id,))
        result = cur.fetchone()
        if result:
            logger.info(f"Soft-deleted client {client_id}")
            return True
        return False


def sync_hard_delete_client(client_id: str):
    """Hard-delete a client and all associated data (for GDPR erasure)."""
    from server.db.connection import sync_cursor

    with sync_cursor() as cur:
        # CASCADE handles client_concepts, client_actions, client_vectors
        cur.execute("DELETE FROM clients WHERE id = %s RETURNING id", (client_id,))
        result = cur.fetchone()
        if result:
            logger.info(f"Hard-deleted client {client_id} (GDPR erasure)")
            return True
        return False


def sync_get_existing_client_ids() -> set:
    """Get all existing (non-deleted) client IDs. Used for incremental processing."""
    from server.db.connection import sync_cursor

    with sync_cursor() as cur:
        cur.execute("SELECT id FROM clients WHERE is_deleted = FALSE")
        return {row[0] for row in cur.fetchall()}


def sync_get_client_hashes() -> dict:
    """
    Get a hash of each client's note text for change detection.
    Returns {client_id: hash_of_full_text}
    """
    from server.db.connection import sync_cursor
    import hashlib

    with sync_cursor() as cur:
        cur.execute("SELECT id, full_text FROM clients WHERE is_deleted = FALSE AND full_text IS NOT NULL")
        result = {}
        for row in cur.fetchall():
            cid, text = row
            result[cid] = hashlib.md5(text.encode()).hexdigest() if text else ""
        return result


# ===================================================================
# ASYNC operations (used by FastAPI endpoints)
# ===================================================================

async def async_get_all_clients(include_deleted: bool = False):
    """Get all clients with their concepts and actions. Uses batch queries (not N+1)."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        if include_deleted:
            where = ""
            where_c = ""
        else:
            where = "WHERE c.is_deleted = FALSE"
            where_c = "AND c.is_deleted = FALSE"

        # 1. Fetch all clients in one query
        rows = await conn.fetch(f"""
            SELECT c.id, c.segment_id, c.confidence, c.profile_type, c.top_concepts,
                   c.full_text, c.language, c.note_date, c.note_duration,
                   c.created_by, c.created_at, c.is_deleted
            FROM clients c
            {where}
            ORDER BY c.id
        """)

        if not rows:
            return []

        # 2. Fetch ALL concepts in one query, group in Python
        concept_rows = await conn.fetch(f"""
            SELECT cc.client_id, cc.concept_id, cc.label, cc.matched_alias, cc.span_start, cc.span_end
            FROM client_concepts cc
            JOIN clients c ON c.id = cc.client_id
            WHERE 1=1 {where_c}
        """)

        concepts_by_client: dict[str, list] = {}
        for cr in concept_rows:
            cid = cr["client_id"]
            if cid not in concepts_by_client:
                concepts_by_client[cid] = []
            concepts_by_client[cid].append({
                "concept": cr["label"] or cr["concept_id"],
                "alias": cr["matched_alias"] or "",
                "spanStart": cr["span_start"] or 0,
                "spanEnd": cr["span_end"] or 0,
            })

        # 3. Fetch ALL actions in one query, group in Python
        action_rows = await conn.fetch(f"""
            SELECT ca.client_id, ca.action_id, ca.title, ca.channel, ca.priority,
                   ca.kpi, ca.triggers, ca.rationale, ca.is_completed
            FROM client_actions ca
            JOIN clients c ON c.id = ca.client_id
            WHERE 1=1 {where_c}
            ORDER BY
                CASE ca.priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                ca.created_at
        """)

        actions_by_client: dict[str, list] = {}
        for ar in action_rows:
            cid = ar["client_id"]
            if cid not in actions_by_client:
                actions_by_client[cid] = []
            actions_by_client[cid].append({
                "actionId": ar["action_id"] or "",
                "title": ar["title"] or "",
                "channel": ar["channel"] or "",
                "priority": ar["priority"] or "low",
                "kpi": ar["kpi"] or "",
                "triggers": ar["triggers"] or "",
                "rationale": ar["rationale"] or "",
                "isCompleted": ar["is_completed"],
            })

        # 4. Assemble client objects
        clients = []
        for row in rows:
            client_id = row["id"]
            top_concepts = row["top_concepts"].split("|") if row["top_concepts"] else []

            clients.append({
                "id": client_id,
                "segment": row["segment_id"] or 0,
                "confidence": row["confidence"] or 0.0,
                "profileType": row["profile_type"] or "",
                "topConcepts": top_concepts,
                "fullText": row["full_text"] or "",
                "language": row["language"] or "FR",
                "date": str(row["note_date"]) if row["note_date"] else "",
                "duration": row["note_duration"] or "",
                "createdBy": row["created_by"],
                "conceptEvidence": concepts_by_client.get(client_id, []),
                "actions": actions_by_client.get(client_id, []),
            })

        return clients


async def async_get_segments():
    """Get all segments from DB."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT id, name, profile, full_profile, client_count
            FROM segments ORDER BY id
        """)
        return [
            {
                "name": row["name"],
                "value": row["client_count"],
                "profile": row["profile"],
                "fullProfile": row["full_profile"] or "",
            }
            for row in rows
        ]


async def async_get_dashboard_data():
    """
    Build dashboard data from DB instead of reading data.json.
    This is the DB-backed replacement for the /api/data endpoint.
    """
    from server.db.connection import get_connection

    clients = await async_get_all_clients()
    segments = await async_get_segments()

    async with get_connection() as conn:
        # Get 3D scatter from vectors table
        scatter_rows = await conn.fetch("""
            SELECT cv.client_id, cv.x_3d, cv.y_3d, cv.z_3d,
                   c.segment_id, c.profile_type
            FROM client_vectors cv
            JOIN clients c ON c.id = cv.client_id
            WHERE c.is_deleted = FALSE AND cv.x_3d IS NOT NULL
        """)

        scatter3d = [
            {
                "x": row["x_3d"],
                "y": row["y_3d"],
                "z": row["z_3d"],
                "client": row["client_id"],
                "id": row["client_id"],
                "segment": row["segment_id"] or 0,
                "profile": row["profile_type"] or "",
            }
            for row in scatter_rows
        ]

        # Get top concepts for bar chart
        concept_rows = await conn.fetch("""
            SELECT label, COUNT(*) as cnt,
                   ARRAY_AGG(DISTINCT client_id) as client_ids
            FROM client_concepts cc
            JOIN clients c ON c.id = cc.client_id
            WHERE c.is_deleted = FALSE AND cc.label IS NOT NULL
            GROUP BY label
            ORDER BY cnt DESC
            LIMIT 20
        """)

        concepts = [
            {"concept": row["label"], "count": row["cnt"], "clients": list(row["client_ids"])}
            for row in concept_rows
        ]

        # Heatmap: segment × top-8 concepts
        top_8_labels = [c["concept"] for c in concepts[:8]]
        heatmap_rows = await conn.fetch("""
            SELECT c.segment_id, cc.label, COUNT(*) as cnt
            FROM client_concepts cc
            JOIN clients c ON c.id = cc.client_id
            WHERE c.is_deleted = FALSE AND cc.label = ANY($1::text[])
            GROUP BY c.segment_id, cc.label
        """, top_8_labels)

        # Build heatmap structure
        heatmap_dict = {}
        for row in heatmap_rows:
            seg = row["segment_id"]
            if seg not in heatmap_dict:
                heatmap_dict[seg] = {}
            heatmap_dict[seg][row["label"]] = row["cnt"]

        heatmap = []
        for seg_id in sorted(heatmap_dict.keys()):
            entry = {"segment": f"Seg {seg_id}"}
            for label in top_8_labels:
                entry[label] = heatmap_dict.get(seg_id, {}).get(label, 0)
            heatmap.append(entry)

        # Metrics
        client_count = await conn.fetchval(
            "SELECT COUNT(*) FROM clients WHERE is_deleted = FALSE"
        )
        segment_count = await conn.fetchval(
            "SELECT COUNT(*) FROM segments"
        )

        # Last pipeline run info
        last_run = await conn.fetchrow("""
            SELECT total_time, stage_timings, records_processed, completed_at
            FROM pipeline_runs WHERE status = 'completed'
            ORDER BY completed_at DESC LIMIT 1
        """)

    metrics = {
        "clients": client_count or 0,
        "segments": segment_count or 0,
    }

    processing_info = {
        "timestamp": str(last_run["completed_at"]) if last_run else None,
        "totalRecords": last_run["records_processed"] if last_run else 0,
        "pipelineTimings": json.loads(last_run["stage_timings"]) if last_run and last_run["stage_timings"] else {},
    }
    if last_run and last_run["total_time"]:
        processing_info["pipelineTimings"]["total"] = last_run["total_time"]

    return {
        "segments": segments,
        "clients": clients,
        "scatter3d": scatter3d,
        "concepts": concepts,
        "heatmap": heatmap,
        "heatmapConcepts": top_8_labels,
        "metrics": metrics,
        "radar": [],  # radar computed on-the-fly is complex — keep file-based for now
        "processingInfo": processing_info,
    }


async def async_get_users():
    """Get all users (admin only)."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT id, username, display_name, email, role, is_active, created_at
            FROM users ORDER BY id
        """)
        return [dict(row) for row in rows]


async def async_create_user(username: str, display_name: str, email: str,
                            password_hash: str, role: str = "sales"):
    """Create a new user."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        row = await conn.fetchrow("""
            INSERT INTO users (username, display_name, email, password_hash, role)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, username, display_name, email, role, created_at
        """, username, display_name, email, password_hash, role)
        return dict(row) if row else None


async def async_authenticate_user(username: str, password: str):
    """Authenticate a user by username and password. Returns user dict or None."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        row = await conn.fetchrow("""
            SELECT id, username, display_name, email, password_hash, role, is_active
            FROM users WHERE username = $1 AND is_active = TRUE
        """, username)

        if not row:
            return None

        import bcrypt
        stored_hash = row["password_hash"].encode()
        if bcrypt.checkpw(password.encode(), stored_hash):
            return {
                "id": row["id"],
                "username": row["username"],
                "displayName": row["display_name"],
                "email": row["email"],
                "role": row["role"],
            }
        return None


async def async_get_upload_history(user_id: int = None, limit: int = 20):
    """Get recent upload sessions."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        if user_id:
            rows = await conn.fetch("""
                SELECT us.*, u.display_name as user_name
                FROM upload_sessions us
                LEFT JOIN users u ON u.id = us.user_id
                WHERE us.user_id = $1
                ORDER BY us.created_at DESC LIMIT $2
            """, user_id, limit)
        else:
            rows = await conn.fetch("""
                SELECT us.*, u.display_name as user_name
                FROM upload_sessions us
                LEFT JOIN users u ON u.id = us.user_id
                ORDER BY us.created_at DESC LIMIT $1
            """, limit)
        return [dict(row) for row in rows]
