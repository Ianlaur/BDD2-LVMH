"""
Sync pipeline results to the Neon database.

Called at the end of the pipeline (after Stage 10) to persist all
results to the database. Also used for incremental processing —
detects which clients are new/changed and only processes those.
"""
import logging
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional

from server.shared.config import DATA_OUTPUTS, DATA_PROCESSED, TAXONOMY_DIR
from server.shared.utils import log_stage

logger = logging.getLogger(__name__)


def get_new_and_changed_clients(csv_path: str) -> tuple:
    """
    Compare incoming CSV with existing DB data to find new/changed clients.
    
    Returns:
        (new_ids, changed_ids, unchanged_ids) — sets of client IDs
    """
    from server.db.crud import sync_get_client_hashes

    # Read the incoming CSV
    df = pd.read_csv(csv_path)

    # Detect ID column
    id_col = None
    for col in ["ID", "id", "client_id", "Client_ID"]:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        logger.warning("Could not detect ID column — treating all as new")
        return set(df.iloc[:, 0].astype(str)), set(), set()

    # Detect text column
    text_col = None
    for col in ["Transcription", "transcription", "text", "Text", "notes", "Notes"]:
        if col in df.columns:
            text_col = col
            break

    # Get existing hashes from DB
    existing_hashes = sync_get_client_hashes()

    new_ids = set()
    changed_ids = set()
    unchanged_ids = set()

    for _, row in df.iterrows():
        cid = str(row[id_col])
        if text_col:
            text = str(row.get(text_col, ""))
            new_hash = hashlib.md5(text.encode()).hexdigest()
        else:
            new_hash = ""

        if cid not in existing_hashes:
            new_ids.add(cid)
        elif new_hash != existing_hashes.get(cid, ""):
            changed_ids.add(cid)
        else:
            unchanged_ids.add(cid)

    logger.info(
        f"Incremental analysis: {len(new_ids)} new, "
        f"{len(changed_ids)} changed, {len(unchanged_ids)} unchanged"
    )
    return new_ids, changed_ids, unchanged_ids


def sync_results_to_db(user_id: Optional[int] = None,
                       upload_session_id: Optional[int] = None,
                       pipeline_run_id: Optional[int] = None):
    """
    Read pipeline output files and upsert everything into the database.
    
    This is called after the pipeline completes (after Stage 10).
    It reads the same files that generate_dashboard_data.py reads,
    then writes them to the DB.
    """
    from server.db.crud import (
        sync_upsert_clients,
        sync_upsert_segments,
        sync_upsert_vectors,
        sync_update_3d_coords,
        sync_upsert_lexicon,
    )

    log_stage("db_sync", "Syncing pipeline results to database...")

    # --- Load pipeline outputs ---
    profiles_path = DATA_OUTPUTS / "client_profiles.csv"
    concepts_path = DATA_OUTPUTS / "note_concepts.csv"
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    actions_path = DATA_OUTPUTS / "recommended_actions.csv"
    vectors_path = DATA_OUTPUTS / "note_vectors.parquet"
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.csv"

    if not profiles_path.exists():
        log_stage("db_sync", "No client_profiles.csv found — skipping DB sync")
        return {"error": "No pipeline outputs found"}

    profiles_df = pd.read_csv(profiles_path)

    concepts_df = None
    if concepts_path.exists():
        concepts_df = pd.read_csv(concepts_path)
        # Join client_id if missing
        if "client_id" not in concepts_df.columns and notes_path.exists():
            notes_df_ids = pd.read_parquet(notes_path, columns=["note_id", "client_id"])
            if "note_id" in concepts_df.columns:
                concepts_df["note_id"] = concepts_df["note_id"].astype(str)
                notes_df_ids["note_id"] = notes_df_ids["note_id"].astype(str)
                concepts_df = concepts_df.merge(
                    notes_df_ids[["note_id", "client_id"]],
                    on="note_id", how="left"
                )

    notes_df = None
    if notes_path.exists():
        notes_df = pd.read_parquet(notes_path)

    actions_df = None
    if actions_path.exists():
        actions_df = pd.read_csv(actions_path)
        actions_df["client_id"] = actions_df["client_id"].astype(str)

    # --- 1. Upsert clients ---
    result = sync_upsert_clients(
        profiles_df, concepts_df, notes_df, actions_df,
        user_id=user_id
    )

    # --- 2. Upsert segments ---
    sync_upsert_segments(profiles_df)

    # --- 3. Upsert vectors ---
    if vectors_path.exists():
        try:
            vectors_df = pd.read_parquet(vectors_path)
            sync_upsert_vectors(vectors_df)
        except Exception as e:
            log_stage("db_sync", f"Warning: could not sync vectors: {e}")

    # --- 4. Upsert lexicon ---
    if lexicon_path.exists():
        try:
            lexicon_df = pd.read_csv(lexicon_path)
            sync_upsert_lexicon(lexicon_df)
        except Exception as e:
            log_stage("db_sync", f"Warning: could not sync lexicon: {e}")

    # --- 5. Update 3D scatter coords from data.json if it exists ---
    from server.shared.config import BASE_DIR
    import json
    data_json_path = BASE_DIR / "dashboard" / "src" / "data.json"
    if data_json_path.exists():
        try:
            with open(data_json_path, "r") as f:
                data_json = json.load(f)
            scatter3d = data_json.get("scatter3d", [])
            if scatter3d:
                sync_update_3d_coords(scatter3d)
        except Exception as e:
            log_stage("db_sync", f"Warning: could not sync 3D coords: {e}")

    log_stage("db_sync", f"DB sync complete: {result}")
    return result
