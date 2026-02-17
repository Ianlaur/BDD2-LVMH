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


# ===================================================================
# EVENT / ACTIVATION CALENDAR operations
# ===================================================================

async def async_create_event(title: str, description: str, event_date: str,
                             event_end_date: str = None, concepts: str = "",
                             channel: str = "email", priority: str = "medium",
                             created_by: int = None):
    """
    Create a new activation event, then auto-match clients whose
    top_concepts overlap with the event's concept tags.
    Returns the created event dict with matched_count.
    """
    from server.db.connection import get_connection
    from datetime import date as date_type

    async with get_connection() as conn:
        row = await conn.fetchrow("""
            INSERT INTO events (title, description, event_date, event_end_date,
                                concepts, channel, priority, status, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'draft', $8)
            RETURNING *
        """, title, description or "",
            event_date,
            event_end_date,
            concepts, channel, priority, created_by)

        event = dict(row)

        # Auto-match clients
        matched = await _match_event_clients(conn, event["id"], concepts)
        event["matched_count"] = matched

        return event


async def _match_event_clients(conn, event_id: int, concepts_str: str):
    """
    Core matching engine: find clients whose concepts overlap with event concepts.
    Matches against clients.top_concepts (pipe-separated concept strings).
    Returns the number of matched clients.
    """
    if not concepts_str or not concepts_str.strip():
        return 0

    concept_list = [c.strip().lower() for c in concepts_str.split("|") if c.strip()]
    if not concept_list:
        return 0

    # Primary strategy: match via clients.top_concepts (where the real data lives)
    tc_rows = await conn.fetch("""
        SELECT c.id, c.top_concepts
        FROM clients c
        WHERE c.is_deleted = FALSE
          AND c.top_concepts IS NOT NULL AND c.top_concepts != ''
    """)

    # Build score map: client_id -> { score, reasons[] }
    match_map: dict[str, dict] = {}

    for row in tc_rows:
        cid = row["id"]
        client_concepts_lower = [t.strip().lower() for t in row["top_concepts"].split("|")]
        overlap = set(concept_list) & set(client_concepts_lower)
        if overlap:
            match_map[cid] = {
                "score": float(len(overlap)),
                "reasons": list(overlap),
            }

    if not match_map:
        return 0

    # Normalise scores to 0–1 range
    max_score = max(m["score"] for m in match_map.values()) or 1.0

    # Insert matches into event_targets
    values = [
        (event_id, cid, " | ".join(m["reasons"]), round(m["score"] / max_score, 3))
        for cid, m in match_map.items()
    ]

    await conn.executemany("""
        INSERT INTO event_targets (event_id, client_id, match_reason, match_score)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (event_id, client_id) DO UPDATE
        SET match_reason = EXCLUDED.match_reason, match_score = EXCLUDED.match_score
    """, values)

    # Update event.matched_count
    await conn.execute("""
        UPDATE events SET matched_count = $1, updated_at = NOW() WHERE id = $2
    """, len(values), event_id)

    return len(values)


async def async_get_events(month: int = None, year: int = None, status: str = None):
    """List events, optionally filtered by month/year and status."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        conditions = []
        params = []
        idx = 1

        if month and year:
            conditions.append(f"EXTRACT(MONTH FROM event_date) = ${idx}")
            params.append(month)
            idx += 1
            conditions.append(f"EXTRACT(YEAR FROM event_date) = ${idx}")
            params.append(year)
            idx += 1
        elif year:
            conditions.append(f"EXTRACT(YEAR FROM event_date) = ${idx}")
            params.append(year)
            idx += 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        rows = await conn.fetch(f"""
            SELECT e.*, u.display_name as creator_name
            FROM events e
            LEFT JOIN users u ON u.id = e.created_by
            {where}
            ORDER BY e.event_date ASC
        """, *params)

        events = []
        for row in rows:
            ev = dict(row)
            ev["event_date"] = str(ev["event_date"])
            if ev.get("event_end_date"):
                ev["event_end_date"] = str(ev["event_end_date"])
            ev["created_at"] = str(ev["created_at"])
            ev["updated_at"] = str(ev["updated_at"])
            events.append(ev)
        return events


async def async_get_event_detail(event_id: int):
    """Get a single event with its matched client targets."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        event_row = await conn.fetchrow("""
            SELECT e.*, u.display_name as creator_name
            FROM events e
            LEFT JOIN users u ON u.id = e.created_by
            WHERE e.id = $1
        """, event_id)

        if not event_row:
            return None

        event = dict(event_row)
        event["event_date"] = str(event["event_date"])
        if event.get("event_end_date"):
            event["event_end_date"] = str(event["event_end_date"])
        event["created_at"] = str(event["created_at"])
        event["updated_at"] = str(event["updated_at"])

        # Fetch matched clients with their profile info
        target_rows = await conn.fetch("""
            SELECT et.client_id, et.match_reason, et.match_score,
                   et.action_status, et.notified_at, et.responded_at,
                   c.profile_type, c.segment_id, c.top_concepts
            FROM event_targets et
            JOIN clients c ON c.id = et.client_id
            WHERE et.event_id = $1
            ORDER BY et.match_score DESC
        """, event_id)

        event["targets"] = [
            {
                "clientId": r["client_id"],
                "matchReason": r["match_reason"],
                "matchScore": r["match_score"],
                "actionStatus": r["action_status"],
                "notifiedAt": str(r["notified_at"]) if r["notified_at"] else None,
                "respondedAt": str(r["responded_at"]) if r["responded_at"] else None,
                "profileType": r["profile_type"] or "",
                "segment": r["segment_id"] or 0,
                "topConcepts": r["top_concepts"].split("|") if r["top_concepts"] else [],
            }
            for r in target_rows
        ]

        return event


async def async_update_event(event_id: int, **fields):
    """Update event fields (title, description, status, channel, etc.)."""
    from server.db.connection import get_connection

    allowed = {"title", "description", "event_date", "event_end_date",
               "concepts", "channel", "priority", "status"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}

    if not updates:
        return None

    async with get_connection() as conn:
        set_clauses = []
        params = []
        idx = 1
        for key, val in updates.items():
            set_clauses.append(f"{key} = ${idx}")
            params.append(val)
            idx += 1

        params.append(event_id)
        row = await conn.fetchrow(f"""
            UPDATE events SET {', '.join(set_clauses)}, updated_at = NOW()
            WHERE id = ${idx}
            RETURNING *
        """, *params)

        if not row:
            return None

        event = dict(row)

        # If concepts changed, re-run matching
        if "concepts" in updates:
            # Clear old matches
            await conn.execute("DELETE FROM event_targets WHERE event_id = $1", event_id)
            matched = await _match_event_clients(conn, event_id, updates["concepts"])
            event["matched_count"] = matched

        event["event_date"] = str(event["event_date"])
        if event.get("event_end_date"):
            event["event_end_date"] = str(event["event_end_date"])
        event["created_at"] = str(event["created_at"])
        event["updated_at"] = str(event["updated_at"])
        return event


async def async_update_target_status(event_id: int, client_id: str, action_status: str):
    """Update a target client's action status (pending → notified → responded | skipped)."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        extra = ""
        if action_status == "notified":
            extra = ", notified_at = NOW()"
        elif action_status == "responded":
            extra = ", responded_at = NOW()"

        row = await conn.fetchrow(f"""
            UPDATE event_targets
            SET action_status = $1 {extra}
            WHERE event_id = $2 AND client_id = $3
            RETURNING *
        """, action_status, event_id, client_id)

        if row:
            # Update notified_count on the event
            notified = await conn.fetchval("""
                SELECT COUNT(*) FROM event_targets
                WHERE event_id = $1 AND action_status IN ('notified', 'responded')
            """, event_id)
            await conn.execute("""
                UPDATE events SET notified_count = $1, updated_at = NOW() WHERE id = $2
            """, notified, event_id)

        return dict(row) if row else None


async def async_delete_event(event_id: int):
    """Delete an event and its targets (CASCADE)."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        result = await conn.execute("DELETE FROM events WHERE id = $1", event_id)
        return "DELETE 1" in result


async def async_update_target_outcome(event_id: int, client_id: str,
                                      outcome: str, outcome_value: float = 0.0,
                                      outcome_notes: str = ""):
    """
    Record the business outcome for an event target:
    outcome: 'visited', 'purchased', 'no_response', 'none'
    outcome_value: revenue amount (for purchases)
    """
    from server.db.connection import get_connection

    async with get_connection() as conn:
        row = await conn.fetchrow("""
            UPDATE event_targets
            SET outcome = $1, outcome_value = $2, outcome_notes = $3, outcome_at = NOW()
            WHERE event_id = $4 AND client_id = $5
            RETURNING *
        """, outcome, outcome_value, outcome_notes or "", event_id, client_id)
        return dict(row) if row else None


async def async_get_activation_metrics():
    """
    Get conversion metrics across all events:
    - total targets, notification rate, response rate, conversion rate
    - revenue attributed, by channel, by concept
    """
    from server.db.connection import get_connection

    async with get_connection() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM event_targets")
        notified = await conn.fetchval(
            "SELECT COUNT(*) FROM event_targets WHERE action_status IN ('notified','responded')")
        responded = await conn.fetchval(
            "SELECT COUNT(*) FROM event_targets WHERE action_status = 'responded'")
        purchased = await conn.fetchval(
            "SELECT COUNT(*) FROM event_targets WHERE outcome = 'purchased'")
        visited = await conn.fetchval(
            "SELECT COUNT(*) FROM event_targets WHERE outcome = 'visited'")
        total_revenue = await conn.fetchval(
            "SELECT COALESCE(SUM(outcome_value), 0) FROM event_targets WHERE outcome = 'purchased'")

        # By channel
        channel_rows = await conn.fetch("""
            SELECT e.channel,
                COUNT(*) as targets,
                COUNT(*) FILTER (WHERE et.outcome = 'purchased') as purchases,
                COALESCE(SUM(et.outcome_value) FILTER (WHERE et.outcome = 'purchased'), 0) as revenue
            FROM event_targets et
            JOIN events e ON e.id = et.event_id
            GROUP BY e.channel ORDER BY purchases DESC
        """)

        # By event
        event_rows = await conn.fetch("""
            SELECT e.id, e.title, e.event_date, e.channel,
                COUNT(*) as total_targets,
                COUNT(*) FILTER (WHERE et.action_status IN ('notified','responded')) as notified,
                COUNT(*) FILTER (WHERE et.outcome = 'purchased') as purchased,
                COUNT(*) FILTER (WHERE et.outcome = 'visited') as visited,
                COALESCE(SUM(et.outcome_value) FILTER (WHERE et.outcome = 'purchased'), 0) as revenue
            FROM events e
            JOIN event_targets et ON et.event_id = e.id
            GROUP BY e.id, e.title, e.event_date, e.channel
            ORDER BY e.event_date DESC
        """)

        return {
            "total": total or 0,
            "notified": notified or 0,
            "responded": responded or 0,
            "purchased": purchased or 0,
            "visited": visited or 0,
            "totalRevenue": float(total_revenue or 0),
            "notificationRate": round(notified / total * 100, 1) if total else 0,
            "responseRate": round(responded / total * 100, 1) if total else 0,
            "conversionRate": round(purchased / total * 100, 1) if total else 0,
            "byChannel": [
                {"channel": r["channel"], "targets": r["targets"],
                 "purchases": r["purchases"], "revenue": float(r["revenue"])}
                for r in channel_rows
            ],
            "byEvent": [
                {"id": r["id"], "title": r["title"],
                 "date": str(r["event_date"]), "channel": r["channel"],
                 "totalTargets": r["total_targets"], "notified": r["notified"],
                 "purchased": r["purchased"], "visited": r["visited"],
                 "revenue": float(r["revenue"])}
                for r in event_rows
            ],
        }


async def async_get_concept_list():
    """
    Get all distinct concept labels from the DB — used for the event creation
    concept picker in the dashboard.
    Extracts individual concepts from clients.top_concepts (pipe-separated).
    """
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT concept, COUNT(*) as client_count
            FROM (
                SELECT TRIM(unnest(string_to_array(top_concepts, '|'))) as concept
                FROM clients
                WHERE is_deleted = FALSE
                  AND top_concepts IS NOT NULL
                  AND top_concepts != ''
            ) sub
            WHERE concept != ''
            GROUP BY concept
            ORDER BY client_count DESC
        """)
        return [{"concept": r["concept"], "clientCount": r["client_count"]} for r in rows]


# ===================================================================
# PLAYBOOK TEMPLATES
# ===================================================================

SEED_PLAYBOOKS = [
    {
        "name": "New Product Launch",
        "description": "Target clients interested in leather goods and limited editions. Ideal for seasonal collections.",
        "concepts": "leather|limited|edizione limitata",
        "channel": "email",
        "priority": "high",
        "message_template": "Dear client, discover our exclusive new collection curated just for you.",
        "category": "launch",
    },
    {
        "name": "VIP Re-Engagement",
        "description": "Re-activate high-value VIP clients who haven't been contacted recently.",
        "concepts": "vip|client vip|budget préfère",
        "channel": "phone",
        "priority": "high",
        "message_template": "We'd love to welcome you back for a private viewing of our latest pieces.",
        "category": "reengagement",
    },
    {
        "name": "Leather Goods Showcase",
        "description": "Invite leather enthusiasts to discover the latest handbags, wallets, and accessories.",
        "concepts": "leather|interessata a portefeuille|looking for",
        "channel": "in_store",
        "priority": "medium",
        "message_template": "You're invited to an exclusive preview of our new leather collection.",
        "category": "launch",
    },
    {
        "name": "Birthday & Anniversary",
        "description": "Send personalized birthday wishes with a curated selection of gift ideas.",
        "concepts": "gift seeking|anniversaire|mariage",
        "channel": "sms",
        "priority": "medium",
        "message_template": "Wishing you a wonderful celebration! Discover our curated gift selection.",
        "category": "birthday",
    },
    {
        "name": "Seasonal Holiday Campaign",
        "description": "Holiday-themed activation for gift-givers. High volume, multi-channel approach.",
        "concepts": "gift seeking|budget préfère|rendez vous avec",
        "channel": "multi",
        "priority": "high",
        "message_template": "This holiday season, find the perfect gift. Book your private consultation.",
        "category": "seasonal",
    },
    {
        "name": "Post-Purchase Follow-Up",
        "description": "Follow up with recent purchasers to build loyalty and gather feedback.",
        "concepts": "leather|budget préfère|classic/timeless",
        "channel": "email",
        "priority": "low",
        "message_template": "Thank you for your recent purchase. We'd love to hear about your experience.",
        "category": "reengagement",
    },
]


async def async_seed_playbooks():
    """Seed default playbook templates if none exist."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM playbooks")
        if count > 0:
            return {"seeded": 0, "message": "Playbooks already exist"}

        for pb in SEED_PLAYBOOKS:
            await conn.execute("""
                INSERT INTO playbooks (name, description, concepts, channel, priority,
                                      message_template, category)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, pb["name"], pb["description"], pb["concepts"],
                pb["channel"], pb["priority"], pb["message_template"], pb["category"])

        return {"seeded": len(SEED_PLAYBOOKS)}


async def async_get_playbooks(category: str = None):
    """Get all playbook templates."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        if category:
            rows = await conn.fetch("""
                SELECT * FROM playbooks WHERE is_active = TRUE AND category = $1
                ORDER BY created_at DESC
            """, category)
        else:
            rows = await conn.fetch("""
                SELECT * FROM playbooks WHERE is_active = TRUE
                ORDER BY category, name
            """)

        return [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "concepts": r["concepts"],
                "channel": r["channel"],
                "priority": r["priority"],
                "messageTemplate": r["message_template"] or "",
                "category": r["category"],
                "createdAt": str(r["created_at"]),
            }
            for r in rows
        ]


async def async_create_playbook(name: str, description: str, concepts: str,
                                channel: str, priority: str, message_template: str,
                                category: str, created_by: int = None):
    """Create a custom playbook template."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        row = await conn.fetchrow("""
            INSERT INTO playbooks (name, description, concepts, channel, priority,
                                   message_template, category, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """, name, description, concepts, channel, priority,
            message_template, category, created_by)
        return dict(row) if row else None


# ===================================================================
# ADVISOR ASSIGNMENT SYSTEM
# ===================================================================

async def async_get_advisors():
    """Get all users who can be advisors (sales, manager, admin roles)."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT u.id, u.username, u.display_name, u.role,
                   COUNT(c.id) as client_count
            FROM users u
            LEFT JOIN clients c ON c.assigned_advisor_id = u.id AND c.is_deleted = FALSE
            WHERE u.is_active = TRUE
              AND u.role IN ('sales', 'manager', 'admin')
            GROUP BY u.id, u.username, u.display_name, u.role
            ORDER BY u.display_name
        """)
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "displayName": r["display_name"],
                "role": r["role"],
                "clientCount": r["client_count"],
            }
            for r in rows
        ]


async def async_assign_advisor(client_id: str, advisor_id: int):
    """Assign an advisor to a client."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        row = await conn.fetchrow("""
            UPDATE clients
            SET assigned_advisor_id = $2, updated_at = NOW()
            WHERE id = $1 AND is_deleted = FALSE
            RETURNING id, assigned_advisor_id
        """, client_id, advisor_id)
        if not row:
            return None
        # Fetch advisor name
        advisor = await conn.fetchrow("SELECT display_name FROM users WHERE id = $1", advisor_id)
        return {
            "clientId": row["id"],
            "advisorId": row["assigned_advisor_id"],
            "advisorName": advisor["display_name"] if advisor else None,
        }


async def async_bulk_assign_advisor(client_ids: list[str], advisor_id: int):
    """Assign an advisor to multiple clients at once."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        result = await conn.execute("""
            UPDATE clients
            SET assigned_advisor_id = $1, updated_at = NOW()
            WHERE id = ANY($2) AND is_deleted = FALSE
        """, advisor_id, client_ids)
        count = int(result.split()[-1])
        advisor = await conn.fetchrow("SELECT display_name FROM users WHERE id = $1", advisor_id)
        return {
            "updated": count,
            "advisorId": advisor_id,
            "advisorName": advisor["display_name"] if advisor else None,
        }


async def async_unassign_advisor(client_id: str):
    """Remove advisor assignment from a client."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        await conn.execute("""
            UPDATE clients SET assigned_advisor_id = NULL, updated_at = NOW()
            WHERE id = $1
        """, client_id)
        return {"clientId": client_id, "advisorId": None}


async def async_auto_assign_advisors(strategy: str = "round_robin"):
    """
    Auto-assign unassigned clients to available advisors.
    Strategies: 'round_robin' (even distribution), 'segment' (same advisor per segment).
    """
    from server.db.connection import get_connection

    async with get_connection() as conn:
        # Get unassigned clients
        unassigned = await conn.fetch("""
            SELECT id, segment_id FROM clients
            WHERE is_deleted = FALSE AND assigned_advisor_id IS NULL
            ORDER BY segment_id, id
        """)
        if not unassigned:
            return {"assigned": 0, "message": "No unassigned clients"}

        # Get available advisors
        advisors = await conn.fetch("""
            SELECT id, display_name FROM users
            WHERE is_active = TRUE AND role IN ('sales', 'manager')
            ORDER BY id
        """)
        if not advisors:
            return {"assigned": 0, "message": "No available advisors"}

        advisor_ids = [a["id"] for a in advisors]

        if strategy == "segment":
            # Assign all clients in same segment to same advisor
            segments = {}
            for row in unassigned:
                seg = row["segment_id"] or 0
                if seg not in segments:
                    segments[seg] = []
                segments[seg].append(row["id"])

            assigned = 0
            for i, (seg, client_ids) in enumerate(segments.items()):
                advisor_id = advisor_ids[i % len(advisor_ids)]
                result = await conn.execute("""
                    UPDATE clients SET assigned_advisor_id = $1, updated_at = NOW()
                    WHERE id = ANY($2)
                """, advisor_id, client_ids)
                assigned += int(result.split()[-1])
        else:
            # Round robin
            assigned = 0
            for i, row in enumerate(unassigned):
                advisor_id = advisor_ids[i % len(advisor_ids)]
                await conn.execute("""
                    UPDATE clients SET assigned_advisor_id = $1, updated_at = NOW()
                    WHERE id = $2
                """, advisor_id, row["id"])
                assigned += 1

        return {
            "assigned": assigned,
            "strategy": strategy,
            "advisors": [{"id": a["id"], "name": a["display_name"]} for a in advisors],
        }


async def async_get_advisor_workload():
    """Get workload stats per advisor: total clients, by segment, by tier."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT
                u.id as advisor_id,
                u.display_name as advisor_name,
                u.role,
                COUNT(c.id) as total_clients,
                COUNT(CASE WHEN cs.tier = 'platinum' THEN 1 END) as platinum,
                COUNT(CASE WHEN cs.tier = 'gold' THEN 1 END) as gold,
                COUNT(CASE WHEN cs.tier = 'silver' THEN 1 END) as silver,
                COUNT(CASE WHEN cs.tier = 'bronze' THEN 1 END) as bronze,
                ROUND(AVG(cs.overall_score)::numeric, 1) as avg_score
            FROM users u
            LEFT JOIN clients c ON c.assigned_advisor_id = u.id AND c.is_deleted = FALSE
            LEFT JOIN client_scores cs ON cs.client_id = c.id
            WHERE u.is_active = TRUE
              AND u.role IN ('sales', 'manager', 'admin')
            GROUP BY u.id, u.display_name, u.role
            ORDER BY total_clients DESC
        """)

        unassigned_count = await conn.fetchval("""
            SELECT COUNT(*) FROM clients
            WHERE is_deleted = FALSE AND assigned_advisor_id IS NULL
        """)

        return {
            "advisors": [
                {
                    "id": r["advisor_id"],
                    "name": r["advisor_name"],
                    "role": r["role"],
                    "totalClients": r["total_clients"],
                    "platinum": r["platinum"],
                    "gold": r["gold"],
                    "silver": r["silver"],
                    "bronze": r["bronze"],
                    "avgScore": float(r["avg_score"]) if r["avg_score"] else 0,
                }
                for r in rows
            ],
            "unassignedCount": unassigned_count,
        }


# ===================================================================
# EXPORT & REPORTING
# ===================================================================

async def async_export_clients_csv():
    """Export all clients with scores, segments, advisors as CSV rows."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT c.id, c.segment_id, c.confidence, c.profile_type,
                   c.top_concepts, c.language, c.note_date,
                   s.name as segment_name,
                   cs.engagement_score, cs.value_score, cs.overall_score, cs.tier,
                   adv.display_name as advisor_name,
                   (SELECT COUNT(*) FROM client_actions ca WHERE ca.client_id = c.id) as action_count,
                   (SELECT COUNT(*) FROM client_concepts cc WHERE cc.client_id = c.id) as concept_count
            FROM clients c
            LEFT JOIN segments s ON s.id = c.segment_id
            LEFT JOIN client_scores cs ON cs.client_id = c.id
            LEFT JOIN users adv ON adv.id = c.assigned_advisor_id
            WHERE c.is_deleted = FALSE
            ORDER BY c.id
        """)
        return [dict(r) for r in rows]


async def async_export_actions_csv():
    """Export all actions with client and segment info."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT ca.client_id, ca.action_id, ca.title, ca.channel,
                   ca.priority, ca.kpi, ca.triggers, ca.is_completed,
                   c.segment_id, s.name as segment_name, c.profile_type
            FROM client_actions ca
            JOIN clients c ON c.id = ca.client_id AND c.is_deleted = FALSE
            LEFT JOIN segments s ON s.id = c.segment_id
            ORDER BY ca.client_id, ca.priority DESC
        """)
        return [dict(r) for r in rows]


async def async_export_scores_csv():
    """Export all client scores with tier breakdown."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT cs.client_id, cs.engagement_score, cs.value_score,
                   cs.overall_score, cs.tier, cs.computed_at,
                   c.segment_id, s.name as segment_name, c.profile_type,
                   c.language, c.confidence
            FROM client_scores cs
            JOIN clients c ON c.id = cs.client_id AND c.is_deleted = FALSE
            LEFT JOIN segments s ON s.id = c.segment_id
            ORDER BY cs.overall_score DESC
        """)
        return [dict(r) for r in rows]


async def async_generate_summary_report():
    """Generate a comprehensive summary report for executives."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        # Client stats
        total_clients = await conn.fetchval("SELECT COUNT(*) FROM clients WHERE is_deleted = FALSE")
        segment_stats = await conn.fetch("""
            SELECT s.id, s.name, s.client_count,
                   ROUND(AVG(cs.overall_score)::numeric, 1) as avg_score,
                   COUNT(CASE WHEN cs.tier = 'gold' THEN 1 END) as gold_count,
                   COUNT(CASE WHEN cs.tier = 'silver' THEN 1 END) as silver_count
            FROM segments s
            LEFT JOIN clients c ON c.segment_id = s.id AND c.is_deleted = FALSE
            LEFT JOIN client_scores cs ON cs.client_id = c.id
            GROUP BY s.id, s.name, s.client_count
            ORDER BY s.client_count DESC
        """)

        # Action stats
        action_stats = await conn.fetchrow("""
            SELECT COUNT(*) as total,
                   COUNT(CASE WHEN is_completed THEN 1 END) as completed,
                   COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority
            FROM client_actions
        """)

        # Top concepts
        top_concepts = await conn.fetch("""
            SELECT concept_id, label, COUNT(*) as mentions
            FROM client_concepts
            GROUP BY concept_id, label
            ORDER BY mentions DESC
            LIMIT 10
        """)

        # Tier distribution
        tier_stats = await conn.fetch("""
            SELECT tier, COUNT(*) as count,
                   ROUND(AVG(overall_score)::numeric, 1) as avg_score
            FROM client_scores
            GROUP BY tier
            ORDER BY avg_score DESC
        """)

        # Language breakdown
        lang_stats = await conn.fetch("""
            SELECT language, COUNT(*) as count
            FROM clients WHERE is_deleted = FALSE AND language IS NOT NULL
            GROUP BY language ORDER BY count DESC
        """)

        # Advisor workload
        advisor_stats = await conn.fetch("""
            SELECT u.display_name as advisor,
                   COUNT(c.id) as clients,
                   ROUND(AVG(cs.overall_score)::numeric, 1) as avg_score
            FROM users u
            LEFT JOIN clients c ON c.assigned_advisor_id = u.id AND c.is_deleted = FALSE
            LEFT JOIN client_scores cs ON cs.client_id = c.id
            WHERE u.role IN ('sales', 'manager')
            GROUP BY u.display_name
            ORDER BY clients DESC
        """)

        return {
            "generatedAt": "now",
            "totalClients": total_clients,
            "segments": [
                {
                    "id": r["id"], "name": r["name"], "count": r["client_count"],
                    "avgScore": float(r["avg_score"]) if r["avg_score"] else 0,
                    "gold": r["gold_count"], "silver": r["silver_count"],
                }
                for r in segment_stats
            ],
            "actions": {
                "total": action_stats["total"],
                "completed": action_stats["completed"],
                "completionRate": round(action_stats["completed"] / max(action_stats["total"], 1) * 100, 1),
                "highPriority": action_stats["high_priority"],
            },
            "topConcepts": [
                {"id": r["concept_id"], "label": r["label"], "mentions": r["mentions"]}
                for r in top_concepts
            ],
            "tiers": [
                {"tier": r["tier"], "count": r["count"], "avgScore": float(r["avg_score"]) if r["avg_score"] else 0}
                for r in tier_stats
            ],
            "languages": [
                {"language": r["language"], "count": r["count"]}
                for r in lang_stats
            ],
            "advisors": [
                {"name": r["advisor"], "clients": r["clients"], "avgScore": float(r["avg_score"]) if r["avg_score"] else 0}
                for r in advisor_stats
            ],
        }


async def async_get_client_360(client_id: str):
    """
    Get a full 360° view for a single client:
    - profile data, segment info, concepts, actions, event history, timeline
    """
    from server.db.connection import get_connection

    async with get_connection() as conn:
        # 1. Core client data
        client_row = await conn.fetchrow("""
            SELECT c.*, s.name as segment_name, s.profile as segment_profile,
                   s.full_profile as segment_full_profile, s.client_count as segment_size,
                   u.display_name as created_by_name,
                   adv.display_name as advisor_name
            FROM clients c
            LEFT JOIN segments s ON s.id = c.segment_id
            LEFT JOIN users u ON u.id = c.created_by
            LEFT JOIN users adv ON adv.id = c.assigned_advisor_id
            WHERE c.id = $1 AND c.is_deleted = FALSE
        """, client_id)

        if not client_row:
            return None

        client = dict(client_row)

        # 2. Concepts from client_concepts table
        concept_rows = await conn.fetch("""
            SELECT concept_id, label, matched_alias, span_start, span_end
            FROM client_concepts
            WHERE client_id = $1
            ORDER BY concept_id
        """, client_id)

        # 3. Actions
        action_rows = await conn.fetch("""
            SELECT action_id, title, channel, priority, kpi, triggers,
                   rationale, is_completed, completed_by, completed_at, created_at
            FROM client_actions
            WHERE client_id = $1
            ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                     created_at DESC
        """, client_id)

        # 4. Event participations
        event_rows = await conn.fetch("""
            SELECT et.match_reason, et.match_score, et.action_status,
                   et.notified_at, et.responded_at,
                   e.id as event_id, e.title as event_title, e.event_date,
                   e.channel as event_channel, e.priority as event_priority,
                   e.status as event_status
            FROM event_targets et
            JOIN events e ON e.id = et.event_id
            WHERE et.client_id = $1
            ORDER BY e.event_date DESC
        """, client_id)

        # 5. Similar clients (same segment, top 5 by confidence)
        similar_rows = await conn.fetch("""
            SELECT id, segment_id, confidence, top_concepts, profile_type
            FROM clients
            WHERE segment_id = $1 AND id != $2 AND is_deleted = FALSE
            ORDER BY confidence DESC
            LIMIT 5
        """, client_row["segment_id"], client_id)

        # 6. Build timeline from audit_log
        audit_rows = await conn.fetch("""
            SELECT al.action, al.details, al.created_at, u.display_name as user_name
            FROM audit_log al
            LEFT JOIN users u ON u.id = al.user_id
            WHERE al.target_id = $1
            ORDER BY al.created_at DESC
            LIMIT 20
        """, client_id)

        # 7. Client score (if computed)
        score_row = await conn.fetchrow("""
            SELECT engagement_score, value_score, overall_score, tier, score_details
            FROM client_scores WHERE client_id = $1
        """, client_id)

        # Build response
        top_concepts = client["top_concepts"].split("|") if client["top_concepts"] else []

        return {
            "id": client["id"],
            "segment": client["segment_id"] or 0,
            "segmentName": client.get("segment_name") or f"Segment {client['segment_id']}",
            "segmentProfile": client.get("segment_profile") or "",
            "segmentFullProfile": client.get("segment_full_profile") or "",
            "segmentSize": client.get("segment_size") or 0,
            "confidence": client["confidence"] or 0.0,
            "profileType": client["profile_type"] or "",
            "topConcepts": top_concepts,
            "fullText": client["full_text"] or "",
            "language": client["language"] or "FR",
            "noteDate": str(client["note_date"]) if client["note_date"] else None,
            "noteDuration": client["note_duration"] or "",
            "createdBy": client.get("created_by_name") or "",
            "createdAt": str(client["created_at"]),
            "updatedAt": str(client["updated_at"]),
            "conceptEvidence": [
                {
                    "conceptId": r["concept_id"],
                    "label": r["label"] or r["concept_id"],
                    "matchedAlias": r["matched_alias"] or "",
                    "spanStart": r["span_start"] or 0,
                    "spanEnd": r["span_end"] or 0,
                }
                for r in concept_rows
            ],
            "actions": [
                {
                    "actionId": r["action_id"] or "",
                    "title": r["title"] or "",
                    "channel": r["channel"] or "",
                    "priority": r["priority"] or "low",
                    "kpi": r["kpi"] or "",
                    "triggers": r["triggers"] or "",
                    "rationale": r["rationale"] or "",
                    "isCompleted": r["is_completed"],
                    "completedAt": str(r["completed_at"]) if r["completed_at"] else None,
                    "createdAt": str(r["created_at"]),
                }
                for r in action_rows
            ],
            "events": [
                {
                    "eventId": r["event_id"],
                    "eventTitle": r["event_title"],
                    "eventDate": str(r["event_date"]),
                    "eventChannel": r["event_channel"],
                    "eventPriority": r["event_priority"],
                    "eventStatus": r["event_status"],
                    "matchReason": r["match_reason"],
                    "matchScore": r["match_score"],
                    "actionStatus": r["action_status"],
                    "notifiedAt": str(r["notified_at"]) if r["notified_at"] else None,
                    "respondedAt": str(r["responded_at"]) if r["responded_at"] else None,
                }
                for r in event_rows
            ],
            "similarClients": [
                {
                    "id": r["id"],
                    "segment": r["segment_id"] or 0,
                    "confidence": r["confidence"] or 0.0,
                    "topConcepts": r["top_concepts"].split("|") if r["top_concepts"] else [],
                    "profileType": r["profile_type"] or "",
                }
                for r in similar_rows
            ],
            "timeline": [
                {
                    "action": r["action"],
                    "details": r["details"],
                    "date": str(r["created_at"]),
                    "userName": r["user_name"] or "",
                }
                for r in audit_rows
            ],
            "score": {
                "engagementScore": score_row["engagement_score"],
                "valueScore": score_row["value_score"],
                "overallScore": score_row["overall_score"],
                "tier": score_row["tier"],
                "details": json.loads(score_row["score_details"]) if isinstance(score_row["score_details"], str) else score_row["score_details"],
            } if score_row else None,
        }


async def async_compute_client_scores():
    """
    Compute engagement + value scores for all clients and store in client_scores.
    
    Engagement Score (0–100):
      - concept_count: more concepts = higher (max 20 for full score)
      - action_count: having actions shows engagement (max 15 for full)
      - action_completion: % of completed actions
      - event_participation: matched to activation events
      - recency: how recently their note was recorded
    
    Value Score (0–100):
      - confidence: ML confidence of segment assignment
      - action_priority: higher priority actions = higher value
      - concept_richness: diverse concepts = richer profile
    
    Overall = 0.6 * engagement + 0.4 * value
    Tier: platinum (>=80), gold (>=60), silver (>=40), bronze (<40)
    """
    from server.db.connection import get_connection
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    async with get_connection() as conn:
        # Get all clients with their stats
        rows = await conn.fetch("""
            SELECT 
                c.id,
                c.confidence,
                c.top_concepts,
                c.note_date,
                c.segment_id,
                COALESCE(act.total_actions, 0) as total_actions,
                COALESCE(act.completed_actions, 0) as completed_actions,
                COALESCE(act.high_actions, 0) as high_actions,
                COALESCE(evt.event_count, 0) as event_count,
                COALESCE(evt.responded_count, 0) as responded_count
            FROM clients c
            LEFT JOIN (
                SELECT client_id,
                    COUNT(*) as total_actions,
                    COUNT(*) FILTER (WHERE is_completed) as completed_actions,
                    COUNT(*) FILTER (WHERE priority = 'high') as high_actions
                FROM client_actions GROUP BY client_id
            ) act ON act.client_id = c.id
            LEFT JOIN (
                SELECT client_id,
                    COUNT(*) as event_count,
                    COUNT(*) FILTER (WHERE action_status = 'responded') as responded_count
                FROM event_targets GROUP BY client_id
            ) evt ON evt.client_id = c.id
            WHERE c.is_deleted = FALSE
        """)

        values = []
        for row in rows:
            # --- Engagement score components ---
            concepts = [c.strip() for c in (row["top_concepts"] or "").split("|") if c.strip()]
            concept_score = min(len(concepts) / 8.0 * 100, 100)  # 8 concepts = full
            action_score = min(row["total_actions"] / 10.0 * 100, 100)  # 10 actions = full
            completion_rate = (row["completed_actions"] / row["total_actions"] * 100
                              if row["total_actions"] > 0 else 0)
            event_score = min(row["event_count"] / 3.0 * 100, 100)  # 3 events = full

            # Recency: days since note_date (cap at 365)
            recency_score = 0
            if row["note_date"]:
                days_ago = (now.date() - row["note_date"]).days
                recency_score = min(100, max(0, 100 - (days_ago / 365 * 100)))

            engagement = (
                concept_score * 0.20 +
                action_score * 0.20 +
                completion_rate * 0.25 +
                event_score * 0.15 +
                recency_score * 0.20
            )

            # --- Value score components ---
            confidence_pct = (row["confidence"] or 0) * 100
            priority_score = min(row["high_actions"] / 5.0 * 100, 100)  # 5 high = full
            richness = min(len(concepts) / 6.0 * 100, 100)

            value = (
                confidence_pct * 0.40 +
                priority_score * 0.30 +
                richness * 0.30
            )

            overall = engagement * 0.6 + value * 0.4

            if overall >= 80:
                tier = "platinum"
            elif overall >= 60:
                tier = "gold"
            elif overall >= 40:
                tier = "silver"
            else:
                tier = "bronze"

            details = json.dumps({
                "conceptScore": round(concept_score, 1),
                "actionScore": round(action_score, 1),
                "completionRate": round(completion_rate, 1),
                "eventScore": round(event_score, 1),
                "recencyScore": round(recency_score, 1),
                "confidencePct": round(confidence_pct, 1),
                "priorityScore": round(priority_score, 1),
                "richnessScore": round(richness, 1),
            })

            values.append((
                row["id"], round(engagement, 1), round(value, 1),
                round(overall, 1), tier, details
            ))

        # Bulk upsert
        if values:
            await conn.executemany("""
                INSERT INTO client_scores (client_id, engagement_score, value_score,
                    overall_score, tier, score_details, computed_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
                ON CONFLICT (client_id) DO UPDATE SET
                    engagement_score = EXCLUDED.engagement_score,
                    value_score = EXCLUDED.value_score,
                    overall_score = EXCLUDED.overall_score,
                    tier = EXCLUDED.tier,
                    score_details = EXCLUDED.score_details,
                    computed_at = NOW()
            """, values)

        logger.info(f"Computed scores for {len(values)} clients")

        # Return summary
        tier_counts = {}
        for v in values:
            tier_counts[v[4]] = tier_counts.get(v[4], 0) + 1

        return {
            "scored": len(values),
            "tiers": tier_counts,
            "avgEngagement": round(sum(v[1] for v in values) / len(values), 1) if values else 0,
            "avgValue": round(sum(v[2] for v in values) / len(values), 1) if values else 0,
            "avgOverall": round(sum(v[3] for v in values) / len(values), 1) if values else 0,
        }


async def async_get_client_scores(tier: str = None, sort_by: str = "overall_score",
                                   limit: int = 50, offset: int = 0):
    """Get client scores, optionally filtered by tier."""
    from server.db.connection import get_connection

    async with get_connection() as conn:
        conditions = ["c.is_deleted = FALSE"]
        params = []
        idx = 1

        if tier:
            conditions.append(f"cs.tier = ${idx}")
            params.append(tier)
            idx += 1

        where = "WHERE " + " AND ".join(conditions)

        allowed_sorts = {"overall_score", "engagement_score", "value_score"}
        sort_col = sort_by if sort_by in allowed_sorts else "overall_score"

        params.extend([limit, offset])
        rows = await conn.fetch(f"""
            SELECT cs.*, c.segment_id, c.top_concepts, c.confidence,
                   s.profile as segment_profile
            FROM client_scores cs
            JOIN clients c ON c.id = cs.client_id
            LEFT JOIN segments s ON s.id = c.segment_id
            {where}
            ORDER BY cs.{sort_col} DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """, *params)

        total = await conn.fetchval(f"""
            SELECT COUNT(*) FROM client_scores cs
            JOIN clients c ON c.id = cs.client_id
            {where}
        """, *(params[:-2] if tier else []))

        return {
            "total": total or 0,
            "clients": [
                {
                    "clientId": r["client_id"],
                    "segment": r["segment_id"] or 0,
                    "segmentProfile": r["segment_profile"] or "",
                    "confidence": r["confidence"] or 0.0,
                    "topConcepts": (r["top_concepts"] or "").split("|")[:5],
                    "engagementScore": r["engagement_score"],
                    "valueScore": r["value_score"],
                    "overallScore": r["overall_score"],
                    "tier": r["tier"],
                    "scoreDetails": r["score_details"],
                    "computedAt": str(r["computed_at"]),
                }
                for r in rows
            ],
        }


async def async_get_kpi_data():
    """
    Get executive KPI dashboard data:
    - total clients, segments, actions, events
    - action completion rates
    - top concepts distribution
    - recent activity
    - activation stats
    """
    from server.db.connection import get_connection

    async with get_connection() as conn:
        # Core counts
        total_clients = await conn.fetchval(
            "SELECT COUNT(*) FROM clients WHERE is_deleted = FALSE")
        total_segments = await conn.fetchval(
            "SELECT COUNT(*) FROM segments")
        total_actions = await conn.fetchval(
            "SELECT COUNT(*) FROM client_actions")
        completed_actions = await conn.fetchval(
            "SELECT COUNT(*) FROM client_actions WHERE is_completed = TRUE")
        total_events = await conn.fetchval(
            "SELECT COUNT(*) FROM events")
        active_events = await conn.fetchval(
            "SELECT COUNT(*) FROM events WHERE status IN ('scheduled', 'active')")

        # Actions by priority
        priority_rows = await conn.fetch("""
            SELECT priority, COUNT(*) as cnt
            FROM client_actions GROUP BY priority
        """)
        actions_by_priority = {r["priority"]: r["cnt"] for r in priority_rows}

        # Actions by channel
        channel_rows = await conn.fetch("""
            SELECT channel, COUNT(*) as cnt
            FROM client_actions GROUP BY channel ORDER BY cnt DESC
        """)
        actions_by_channel = [{"channel": r["channel"], "count": r["cnt"]} for r in channel_rows]

        # Top 10 concepts
        concept_rows = await conn.fetch("""
            SELECT concept, COUNT(*) as client_count
            FROM (
                SELECT TRIM(unnest(string_to_array(top_concepts, '|'))) as concept
                FROM clients WHERE is_deleted = FALSE
                  AND top_concepts IS NOT NULL AND top_concepts != ''
            ) sub
            WHERE concept != ''
            GROUP BY concept ORDER BY client_count DESC LIMIT 10
        """)
        top_concepts = [{"concept": r["concept"], "count": r["client_count"]} for r in concept_rows]

        # Segment distribution
        segment_rows = await conn.fetch("""
            SELECT s.id, s.name, s.profile, s.client_count
            FROM segments s ORDER BY s.id
        """)
        segment_dist = [
            {"id": r["id"], "name": r["name"], "profile": r["profile"], "count": r["client_count"]}
            for r in segment_rows
        ]

        # Recent uploads
        upload_rows = await conn.fetch("""
            SELECT us.filename, us.status, us.records_added, us.records_updated,
                   us.created_at, u.display_name as user_name
            FROM upload_sessions us
            LEFT JOIN users u ON u.id = us.user_id
            ORDER BY us.created_at DESC LIMIT 5
        """)
        recent_uploads = [
            {
                "filename": r["filename"],
                "status": r["status"],
                "recordsAdded": r["records_added"] or 0,
                "recordsUpdated": r["records_updated"] or 0,
                "date": str(r["created_at"]),
                "userName": r["user_name"] or "",
            }
            for r in upload_rows
        ]

        # Event activation stats
        activation_rows = await conn.fetch("""
            SELECT action_status, COUNT(*) as cnt
            FROM event_targets GROUP BY action_status
        """)
        activation_stats = {r["action_status"]: r["cnt"] for r in activation_rows}

        # Clients per language
        lang_rows = await conn.fetch("""
            SELECT COALESCE(language, 'Unknown') as lang, COUNT(*) as cnt
            FROM clients WHERE is_deleted = FALSE
            GROUP BY language ORDER BY cnt DESC
        """)
        languages = [{"language": r["lang"], "count": r["cnt"]} for r in lang_rows]

        # Average confidence by segment
        conf_rows = await conn.fetch("""
            SELECT segment_id, ROUND(AVG(confidence)::numeric, 3) as avg_conf,
                   COUNT(*) as cnt
            FROM clients WHERE is_deleted = FALSE
            GROUP BY segment_id ORDER BY segment_id
        """)
        confidence_by_segment = [
            {"segment": r["segment_id"], "avgConfidence": float(r["avg_conf"] or 0), "count": r["cnt"]}
            for r in conf_rows
        ]

        return {
            "totalClients": total_clients or 0,
            "totalSegments": total_segments or 0,
            "totalActions": total_actions or 0,
            "completedActions": completed_actions or 0,
            "actionCompletionRate": round((completed_actions / total_actions * 100), 1) if total_actions else 0,
            "totalEvents": total_events or 0,
            "activeEvents": active_events or 0,
            "actionsByPriority": actions_by_priority,
            "actionsByChannel": actions_by_channel,
            "topConcepts": top_concepts,
            "segmentDistribution": segment_dist,
            "recentUploads": recent_uploads,
            "activationStats": activation_stats,
            "languages": languages,
            "confidenceBySegment": confidence_by_segment,
        }


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
