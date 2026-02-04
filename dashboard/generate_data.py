"""Generate dashboard data as JSON for React app."""
import json
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/ian/Desktop/BDD2-LVMH')

from src.shared.config import DATA_OUTPUTS, TAXONOMY_DIR

profiles_df = pd.read_csv(DATA_OUTPUTS / 'client_profiles.csv')
concepts_df = pd.read_csv(DATA_OUTPUTS / 'note_concepts.csv')
lexicon_df = pd.read_csv(TAXONOMY_DIR / 'lexicon_v1.csv')

# Load vectors for 3D visualization
vectors_df = pd.read_parquet(DATA_OUTPUTS / 'note_vectors.parquet')

# Generate 3D coordinates using UMAP
from umap import UMAP
np.random.seed(42)
umap_3d = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
coords_3d = umap_3d.fit_transform(np.vstack(vectors_df['embedding'].values))

# Build 3D scatter data - show full profile info
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

print(f"Generated 3D coordinates for {len(scatter3d)} clients")

# Segment distribution - show full profile with all top concepts
segment_counts = profiles_df['cluster_id'].value_counts().sort_index()
segments = []
for i, count in segment_counts.items():
    profile_full = profiles_df[profiles_df['cluster_id'] == i]['profile_type'].iloc[0]
    # Get top concepts for better labeling
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

# Radar - include ALL segments (not just 4)
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
    # Include ALL segments (0-7)
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

data = {
    'segments': segments,
    'concepts': concepts_bar,
    'heatmap': heatmap,
    'heatmapConcepts': top_8,
    'metrics': metrics,
    'radar': radar,
    'segmentDetails': details,
    'scatter3d': scatter3d
}

with open('/Users/ian/Desktop/BDD2-LVMH/dashboard/src/data.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Data exported to dashboard/src/data.json")
