"""
Generate dashboard data as JSON for the React dashboard.
Single source of truth - outputs to dashboard/src/data.json
"""
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from server.shared.config import DATA_OUTPUTS, TAXONOMY_DIR, BASE_DIR, DATA_INPUT
from server.shared.utils import log_stage


def generate_dashboard_data(pipeline_timings: dict = None):
    """Generate JSON data file for the React dashboard.
    
    Args:
        pipeline_timings: Optional dict with stage timings from the pipeline
    """
    start_time = time.time()
    log_stage("dashboard", "Generating dashboard data...")
    
    output_path = BASE_DIR / "dashboard" / "src" / "data.json"
    
    # Load data files
    profiles_df = pd.read_csv(DATA_OUTPUTS / 'client_profiles.csv')
    concepts_df = pd.read_csv(DATA_OUTPUTS / 'note_concepts.csv')
    lexicon_path = TAXONOMY_DIR / 'lexicon_v1.csv'
    
    if lexicon_path.exists():
        lexicon_df = pd.read_csv(lexicon_path)
    else:
        # Try JSON format
        lexicon_json = TAXONOMY_DIR / 'lexicon_v1.json'
        if lexicon_json.exists():
            with open(lexicon_json) as f:
                lexicon_data = json.load(f)
            lexicon_df = pd.DataFrame(lexicon_data.get('concepts', []))
        else:
            log_stage("dashboard", "Warning: No lexicon found")
            lexicon_df = pd.DataFrame({'concept_id': [], 'label': []})
    
    # Load vectors for 3D visualization
    vectors_path = DATA_OUTPUTS / 'note_vectors.parquet'
    if vectors_path.exists():
        vectors_df = pd.read_parquet(vectors_path)
        
        # Generate 3D coordinates using UMAP
        from umap import UMAP
        np.random.seed(42)
        umap_3d = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        coords_3d = umap_3d.fit_transform(np.vstack(vectors_df['embedding'].values))
        
        # Build 3D scatter data
        scatter3d = []
        for i, row in vectors_df.iterrows():
            client_id = row['client_id']
            cluster_info = profiles_df[profiles_df['client_id'] == client_id]
            if len(cluster_info) > 0:
                cluster_id = int(cluster_info['cluster_id'].iloc[0])
                top_concepts = cluster_info['top_concepts'].iloc[0]
                if pd.notna(top_concepts) and top_concepts:
                    profile = top_concepts.replace('|', ' | ')
                else:
                    profile = cluster_info['profile_type'].iloc[0]
            else:
                cluster_id = -1
                profile = 'Unknown'
            
            scatter3d.append({
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                'cluster': cluster_id,
                'client': client_id,
                'profile': profile
            })
        log_stage("dashboard", f"  3D coordinates: {len(scatter3d)} clients")
    else:
        scatter3d = []
        log_stage("dashboard", "  Warning: No vectors found for 3D viz")
    
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
    clients = []
    
    # Try to load original transcripts
    transcript_data = {}
    csv_files = list(DATA_INPUT.glob('*.csv'))
    if csv_files:
        try:
            raw_df = pd.read_csv(csv_files[0])
            if 'ID' in raw_df.columns and 'Transcription' in raw_df.columns:
                for _, row in raw_df.iterrows():
                    transcript_data[row['ID']] = {
                        'transcription': row.get('Transcription', ''),
                        'date': row.get('Date', ''),
                        'language': row.get('Language', 'FR'),
                        'duration': row.get('Duration', '')
                    }
        except Exception as e:
            log_stage("dashboard", f"  Warning: Could not load transcripts: {e}")
    
    for _, row in profiles_df.iterrows():
        client_id = row['client_id']
        client_concepts = concepts_df[concepts_df['client_id'] == client_id]
        
        # Get concept evidence with positions
        evidence = []
        for _, c in client_concepts.iterrows():
            evidence.append({
                'concept': c.get('label', c['concept_id']),
                'alias': c.get('matched_alias', ''),
                'spanStart': int(c.get('span_start', 0)) if pd.notna(c.get('span_start')) else 0,
                'spanEnd': int(c.get('span_end', 0)) if pd.notna(c.get('span_end')) else 0
            })
        
        # Get transcript info
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
    
    # Assemble final data
    processing_time = time.time() - start_time
    
    data = {
        'segments': segments,
        'concepts': concepts_bar,
        'heatmap': heatmap,
        'heatmapConcepts': top_8,
        'metrics': metrics,
        'radar': radar,
        'segmentDetails': details,
        'scatter3d': scatter3d,
        'clients': clients,
        'processingInfo': {
            'timestamp': datetime.now().isoformat(),
            'totalRecords': len(clients),
            'totalConcepts': len(concepts_bar),
            'totalSegments': len(segments),
            'dashboardGenTime': round(processing_time, 2),
            'pipelineTimings': pipeline_timings or {}
        }
    }
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    log_stage("dashboard", f"  Output: {output_path}")
    log_stage("dashboard", f"  {len(clients)} clients, {len(segments)} segments, {len(scatter3d)} 3D points")
    log_stage("dashboard", f"  Generated in {processing_time:.2f}s")
    

if __name__ == "__main__":
    generate_dashboard_data()
