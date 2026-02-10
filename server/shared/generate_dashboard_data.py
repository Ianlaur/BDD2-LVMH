"""
Generate dashboard data as JSON for the React dashboard.
Single source of truth - outputs to dashboard/src/data.json

Two-phase approach:
  Phase 1 (fast):  segments, concepts, heatmap, metrics, radar, clients → data.json
  Phase 2 (heavy): UMAP 3D projection → updates data.json with scatter3d
"""
import json
import threading
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from server.shared.config import DATA_OUTPUTS, DATA_PROCESSED, TAXONOMY_DIR, BASE_DIR
from server.shared.utils import log_stage

# Module-level path
_OUTPUT_PATH = BASE_DIR / "dashboard" / "src" / "data.json"


# ── helpers ──────────────────────────────────────────────────────

def _write_json(data: dict, path: Path = _OUTPUT_PATH):
    """Atomically write data.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _compute_scatter3d(vectors_df: pd.DataFrame,
                       profiles_df: pd.DataFrame) -> list:
    """Run UMAP and return scatter3d list (the expensive part)."""
    from umap import UMAP

    np.random.seed(42)
    umap_3d = UMAP(n_components=3, random_state=42,
                   n_neighbors=min(15, len(vectors_df) - 1),
                   min_dist=0.1, n_jobs=1)
    coords_3d = umap_3d.fit_transform(
        np.vstack(vectors_df['embedding'].values)
    )

    profile_lookup = {}
    cluster_lookup = {}
    for _, row in profiles_df.iterrows():
        cid = row['client_id']
        cluster_lookup[cid] = int(row['cluster_id'])
        tc = row.get('top_concepts', '')
        if pd.notna(tc) and tc:
            profile_lookup[cid] = tc.replace('|', ' | ')
        else:
            profile_lookup[cid] = row.get('profile_type', 'Unknown')

    scatter3d = []
    for i, row in vectors_df.iterrows():
        client_id = row['client_id']
        scatter3d.append({
            'x': float(coords_3d[i, 0]),
            'y': float(coords_3d[i, 1]),
            'z': float(coords_3d[i, 2]),
            'cluster': cluster_lookup.get(client_id, -1),
            'client': client_id,
            'profile': profile_lookup.get(client_id, 'Unknown'),
        })
    return scatter3d


# ── main ─────────────────────────────────────────────────────────

def generate_dashboard_data(pipeline_timings: dict = None):
    """Generate JSON data file for the React dashboard.

    Phase 1 writes core analytics instantly.
    Phase 2 appends 3D scatter in a background thread.

    Args:
        pipeline_timings: Optional dict with stage timings from the pipeline
    """
    start_time = time.time()
    log_stage("dashboard", "Generating dashboard data...")

    # ── load data ────────────────────────────────────────────────
    profiles_df = pd.read_csv(DATA_OUTPUTS / 'client_profiles.csv')
    concepts_df = pd.read_csv(DATA_OUTPUTS / 'note_concepts.csv')

    # Join client_id from notes_clean if missing
    if 'client_id' not in concepts_df.columns:
        notes_clean = pd.read_parquet(
            DATA_PROCESSED / "notes_clean.parquet",
            columns=["note_id", "client_id"]
        )
        if "note_id" in concepts_df.columns:
            concepts_df["note_id"] = concepts_df["note_id"].astype(str)
            notes_clean["note_id"] = notes_clean["note_id"].astype(str)
            concepts_df = concepts_df.merge(
                notes_clean[["note_id", "client_id"]],
                on="note_id", how="left"
            )
            log_stage("dashboard", "Joined client_id from notes_clean into concepts")

    # Lexicon
    lexicon_path = TAXONOMY_DIR / 'lexicon_v1.csv'
    if lexicon_path.exists():
        lexicon_df = pd.read_csv(lexicon_path)
    else:
        lexicon_json = TAXONOMY_DIR / 'lexicon_v1.json'
        if lexicon_json.exists():
            with open(lexicon_json) as f:
                lexicon_data = json.load(f)
            lexicon_df = pd.DataFrame(lexicon_data.get('concepts', []))
        else:
            log_stage("dashboard", "Warning: No lexicon found")
            lexicon_df = pd.DataFrame({'concept_id': [], 'label': []})

    # ── Phase 1: core analytics (instant) ────────────────────────

    # Segment distribution
    segment_counts = profiles_df['cluster_id'].value_counts().sort_index()
    segments = []
    for i, count in segment_counts.items():
        profile_full = profiles_df[profiles_df['cluster_id'] == i]['profile_type'].iloc[0]
        top_concepts = profiles_df[profiles_df['cluster_id'] == i]['top_concepts'].iloc[0]
        if pd.notna(top_concepts) and top_concepts:
            concepts_list = top_concepts.split('|')[:3]
            profile_display = ' | '.join(concepts_list)
        else:
            profile_display = profile_full.split(' | ')[0] if profile_full else f'Segment {i}'
        segments.append({
            'name': f'Segment {i}',
            'value': int(count),
            'profile': profile_display,
            'fullProfile': profile_full
        })

    # Top concepts
    concept_labels = dict(zip(lexicon_df['concept_id'], lexicon_df['label']))
    concepts_df['label'] = concepts_df['concept_id'].map(concept_labels)
    top_concepts = concepts_df['label'].value_counts().head(12)
    concepts_bar = [{'concept': c, 'count': int(v)} for c, v in top_concepts.items()]

    # Heatmap
    client_cluster = dict(zip(profiles_df['client_id'], profiles_df['cluster_id']))
    concepts_df['cluster_id'] = concepts_df['client_id'].map(client_cluster)
    top_8 = top_concepts.head(8).index.tolist()
    heatmap = []
    for cid in sorted(profiles_df['cluster_id'].unique()):
        cc = concepts_df[concepts_df['cluster_id'] == cid]['label'].value_counts()
        row = {'segment': f'Seg {cid}'}
        for c in top_8:
            row[c] = int(cc.get(c, 0))
        heatmap.append(row)

    # Metrics
    metrics = {
        'clients': len(profiles_df),
        'segments': int(profiles_df['cluster_id'].nunique()),
        'concepts': len(lexicon_df),
        'avgConceptsPerClient': round(concepts_df.groupby('client_id').size().mean(), 1),
        'coverage': round((concepts_df['client_id'].nunique() / len(profiles_df)) * 100, 1)
    }

    # Radar chart data
    dims = ['Cadeaux', 'Voyage', 'Mode', 'Famille', 'Budget', 'VIP', 'Contraintes', 'Loisirs']
    dim_kw = {
        'Cadeaux': ['cadeau', 'gift', 'anniversaire', 'présent', 'regalo'],
        'Voyage': ['voyage', 'travel', 'valise', 'malle', 'trunk', 'bagage'],
        'Mode': ['cuir', 'leather', 'style', 'sac', 'haute couture', 'mode', 'fashion'],
        'Famille': ['famille', 'family', 'enfants', 'mari', 'épouse', 'enfant'],
        'Budget': ['budget', 'prix', 'flexible', 'investissement', 'achat'],
        'VIP': ['vip', 'excellent', 'fidèle', 'client vip', 'privilégié'],
        'Contraintes': ['allergie', 'végétarien', 'vegan', 'régime', 'restriction'],
        'Loisirs': ['golf', 'tennis', 'art', 'vin', 'yoga', 'champagne', 'collection']
    }
    radar = []
    for d in dims:
        row = {'dimension': d}
        for cid in range(metrics['segments']):
            cc = concepts_df[concepts_df['cluster_id'] == cid]
            nc = profiles_df[profiles_df['cluster_id'] == cid]['client_id'].nunique()
            matches = cc[cc['label'].str.lower().isin([k.lower() for k in dim_kw[d]])]
            score = min((len(matches) / max(nc, 1)) * 200, 100)
            row[f'seg{cid}'] = round(score, 1)
        radar.append(row)

    # Segment details
    details = []
    for cid in sorted(profiles_df['cluster_id'].unique()):
        cdf = profiles_df[profiles_df['cluster_id'] == cid]
        cc = concepts_df[concepts_df['cluster_id'] == cid]['label'].value_counts().head(3).index.tolist()
        details.append({
            'id': int(cid),
            'clients': len(cdf),
            'profile': cdf['profile_type'].iloc[0],
            'topConcepts': cc
        })

    # Client list with transcript data
    # Use notes_clean.parquet as the single source of truth for transcripts.
    # This is the file produced by the pipeline (ingest or adaptive),
    # so it always matches the data that was actually processed — regardless
    # of whether the pipeline was triggered from CLI or the dashboard upload.
    clients = []
    transcript_data = {}
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if notes_path.exists():
        try:
            notes_df = pd.read_parquet(notes_path)
            for _, row in notes_df.iterrows():
                nid = str(row.get("note_id", ""))
                transcript_data[nid] = {
                    "transcription": str(row.get("text", "")),
                    "date": str(row.get("date", "")),
                    "language": str(row.get("language", "FR")),
                    "duration": str(row.get("duration", "")),
                }
        except Exception as e:
            log_stage("dashboard", f"  Warning: Could not load transcripts from parquet: {e}")

    for _, row in profiles_df.iterrows():
        client_id = row['client_id']
        client_concepts = concepts_df[concepts_df['client_id'] == client_id]

        evidence = []
        for _, c in client_concepts.iterrows():
            evidence.append({
                'concept': c.get('label', c['concept_id']),
                'alias': c.get('matched_alias', ''),
                'spanStart': int(c.get('span_start', 0)) if pd.notna(c.get('span_start')) else 0,
                'spanEnd': int(c.get('span_end', 0)) if pd.notna(c.get('span_end')) else 0
            })

        t_info = transcript_data.get(client_id, {})
        clients.append({
            'id': client_id,
            'segment': int(row['cluster_id']),
            'confidence': float(row.get('confidence', 0.5)),
            'profileType': row.get('profile_type', ''),
            'topConcepts': row.get('top_concepts', '').split('|') if pd.notna(row.get('top_concepts')) else [],
            'conceptEvidence': evidence,
            'originalNote': t_info.get('transcription', ''),
            'noteDate': t_info.get('date', ''),
            'noteLanguage': t_info.get('language', 'FR'),
            'noteDuration': t_info.get('duration', '')
        })

    phase1_time = time.time() - start_time

    # ── Write Phase 1 immediately (no 3D yet) ───────────────────
    data = {
        'segments': segments,
        'concepts': concepts_bar,
        'heatmap': heatmap,
        'heatmapConcepts': top_8,
        'metrics': metrics,
        'radar': radar,
        'segmentDetails': details,
        'scatter3d': [],          # placeholder — filled by Phase 2
        'clients': clients,
        'processingInfo': {
            'timestamp': datetime.now().isoformat(),
            'totalRecords': len(clients),
            'totalConcepts': len(concepts_bar),
            'totalSegments': len(segments),
            'dashboardGenTime': round(phase1_time, 2),
            'pipelineTimings': pipeline_timings or {},
            'scatter3dReady': False,
        }
    }

    _write_json(data)
    log_stage("dashboard", f"  Phase 1 done in {phase1_time:.2f}s — core data written")

    # ── Phase 2: 3D projection (background) ─────────────────────
    vectors_path = DATA_OUTPUTS / 'note_vectors.parquet'
    if not vectors_path.exists():
        log_stage("dashboard", "  No vectors found — skipping 3D projection")
        return

    vectors_df = pd.read_parquet(vectors_path)

    def _phase2():
        try:
            t0 = time.time()
            scatter3d = _compute_scatter3d(vectors_df, profiles_df)
            data['scatter3d'] = scatter3d
            data['processingInfo']['scatter3dReady'] = True
            data['processingInfo']['scatter3dTime'] = round(time.time() - t0, 2)
            _write_json(data)
            log_stage("dashboard", f"  Phase 2 done in {time.time() - t0:.1f}s — 3D scatter ({len(scatter3d)} pts) written")
        except Exception as e:
            log_stage("dashboard", f"  Phase 2 warning: 3D projection failed: {e}")

    thread = threading.Thread(target=_phase2, name="dashboard-3d", daemon=True)
    thread.start()

    # Wait for Phase 2 to finish (keeps pipeline output tidy)
    thread.join()

    total_time = time.time() - start_time
    log_stage("dashboard", f"  {len(clients)} clients, {len(segments)} segments")
    log_stage("dashboard", f"  Total: {total_time:.2f}s  (phase1={phase1_time:.2f}s, umap={total_time - phase1_time:.2f}s)")


if __name__ == "__main__":
    generate_dashboard_data()
