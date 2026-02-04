"""
3D projection stage: Create interactive visualization of embedding space.

This module:
- Loads note vectors
- Reduces to 3D using UMAP (or PCA fallback)
- Creates interactive Plotly 3D scatter plot
- Includes concept anchors if available

Output: demo/embedding_space_3d.html
"""
import sys
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from server.shared.config import (
    DATA_OUTPUTS, TAXONOMY_DIR, DASHBOARD_DIR,
    UMAP_RANDOM_STATE, NUMPY_SEED
)
from server.shared.utils import log_stage, set_all_seeds


def reduce_to_3d_umap(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 3D using UMAP."""
    try:
        import umap
        
        log_stage("projection", "Reducing to 3D with UMAP...")
        
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=UMAP_RANDOM_STATE,
            n_jobs=1  # Deterministic
        )
        
        coords_3d = reducer.fit_transform(embeddings)
        return coords_3d
        
    except ImportError:
        log_stage("projection", "UMAP not available, falling back to PCA")
        return reduce_to_3d_pca(embeddings)


def reduce_to_3d_pca(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 3D using PCA."""
    from sklearn.decomposition import PCA
    
    log_stage("projection", "Reducing to 3D with PCA...")
    
    n_components = min(3, embeddings.shape[1], embeddings.shape[0])
    
    pca = PCA(
        n_components=n_components,
        random_state=NUMPY_SEED
    )
    
    coords_3d = pca.fit_transform(embeddings)
    
    # Pad with zeros if needed (shouldn't happen normally)
    if coords_3d.shape[1] < 3:
        padding = np.zeros((coords_3d.shape[0], 3 - coords_3d.shape[1]))
        coords_3d = np.hstack([coords_3d, padding])
    
    explained_var = sum(pca.explained_variance_ratio_)
    log_stage("projection", f"PCA explained variance (3D): {explained_var:.2%}")
    
    return coords_3d


def create_3d_visualization(
    coords: np.ndarray,
    note_ids: List[str],
    client_ids: List[str],
    labels: Optional[List[str]] = None,
    cluster_ids: Optional[List[int]] = None,
    concept_coords: Optional[np.ndarray] = None,
    concept_labels: Optional[List[str]] = None
) -> str:
    """
    Create interactive Plotly 3D scatter plot.
    Returns HTML string.
    """
    import plotly.graph_objects as go
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for segments
    segment_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue  
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
    ]
    
    # Group by cluster and create separate traces for legend
    if cluster_ids is not None and labels is not None:
        # Get unique clusters and their profile names
        cluster_profiles = {}
        for i, (cid, label) in enumerate(zip(cluster_ids, labels)):
            if cid not in cluster_profiles:
                cluster_profiles[cid] = label
        
        # Create a trace for each cluster segment
        for cluster_id in sorted(cluster_profiles.keys()):
            # Get indices for this cluster
            indices = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
            
            # Build hover text with full details
            hover_texts = []
            for i in indices:
                hover_texts.append(
                    f"<b>👤 {client_ids[i]}</b><br><br>"
                    f"<b>Segment:</b> {labels[i]}<br><br>"
                    f"<b>Note ID:</b> {note_ids[i]}<br>"
                    f"<extra></extra>"
                )
            
            # Get profile name for legend (shortened)
            profile_parts = cluster_profiles[cluster_id].split(' | ')
            legend_name = ' | '.join(profile_parts[:2]) if len(profile_parts) > 2 else cluster_profiles[cluster_id]
            
            fig.add_trace(go.Scatter3d(
                x=coords[indices, 0],
                y=coords[indices, 1],
                z=coords[indices, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=segment_colors[cluster_id % len(segment_colors)],
                    opacity=0.85,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hovertemplate='%{text}',
                name=f"Segment {cluster_id}: {legend_name}",
                legendgroup=f"cluster_{cluster_id}"
            ))
    else:
        # Fallback: single trace with basic info
        hover_texts = [
            f"<b>👤 {cid}</b><br>Note: {nid}"
            for nid, cid in zip(note_ids, client_ids)
        ]
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=6, color='steelblue', opacity=0.7),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name='Clients'
        ))
    
    # Add concept anchors if available
    if concept_coords is not None and concept_labels is not None:
        fig.add_trace(go.Scatter3d(
            x=concept_coords[:, 0],
            y=concept_coords[:, 1],
            z=concept_coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                opacity=0.9
            ),
            text=concept_labels,
            textposition='top center',
            textfont=dict(size=8),
            hoverinfo='text',
            name='Concepts'
        ))
    
    # Update layout with descriptive axis labels
    # UMAP preserves local structure: nearby points = similar client profiles
    fig.update_layout(
        title=dict(
            text='<b>LVMH Voice-to-Tag</b><br><span style="font-size:14px">Carte de Similarité des Profils Clients</span>',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='← Classique | Moderne →',
            yaxis_title='← Quotidien | Événements →', 
            zaxis_title='← Budget | Premium →',
            xaxis=dict(
                backgroundcolor='rgba(230, 230, 250, 0.3)',
                gridcolor='rgba(0, 0, 0, 0.1)',
                showbackground=True,
                showticklabels=False
            ),
            yaxis=dict(
                backgroundcolor='rgba(250, 235, 215, 0.3)',
                gridcolor='rgba(0, 0, 0, 0.1)',
                showbackground=True,
                showticklabels=False
            ),
            zaxis=dict(
                backgroundcolor='rgba(240, 255, 240, 0.3)',
                gridcolor='rgba(0, 0, 0, 0.1)',
                showbackground=True,
                showticklabels=False
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        legend=dict(
            title=dict(text='<b>Segments Clients</b>', font=dict(size=12)),
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='lightgray',
            borderwidth=1,
            font=dict(size=10),
            itemsizing='constant'
        ),
        margin=dict(l=0, r=0, b=30, t=80),
        width=1100,
        height=750,
        hoverlabel=dict(
            bgcolor='white',
            font_size=13,
            font_family='Arial'
        ),
        annotations=[
            dict(
                text="💡 Survolez pour voir les détails | Cliquez sur la légende pour filtrer les segments",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.03,
                font=dict(size=11, color="gray")
            )
        ]
    )
    
    return fig.to_html(include_plotlyjs='cdn', full_html=True)


def project_3d() -> Optional[str]:
    """
    Main 3D projection function.
    
    Returns:
        Path to output HTML file, or None if skipped
        
    Side effects:
        Writes demo/embedding_space_3d.html
    """
    set_all_seeds()
    
    log_stage("projection", "Starting 3D projection...")
    
    # Load note vectors
    vectors_path = DATA_OUTPUTS / "note_vectors.parquet"
    if not vectors_path.exists():
        log_stage("projection", f"Vectors not found: {vectors_path}. Skipping projection.")
        return None
    
    vectors_df = pd.read_parquet(vectors_path)
    log_stage("projection", f"Loaded {len(vectors_df)} note vectors")
    
    if len(vectors_df) < 3:
        log_stage("projection", "Too few vectors for 3D projection. Skipping.")
        return None
    
    # Extract embeddings
    embeddings = np.array(vectors_df["embedding"].tolist())
    note_ids = vectors_df["note_id"].tolist()
    client_ids = vectors_df["client_id"].tolist()
    
    # Try to load profile labels and cluster IDs
    labels = None
    cluster_ids = None
    profiles_path = DATA_OUTPUTS / "client_profiles.csv"
    if profiles_path.exists():
        profiles_df = pd.read_csv(profiles_path)
        client_labels = dict(zip(profiles_df["client_id"].astype(str), profiles_df["profile_type"]))
        client_clusters = dict(zip(profiles_df["client_id"].astype(str), profiles_df["cluster_id"]))
        labels = [client_labels.get(cid, "Unknown") for cid in client_ids]
        cluster_ids = [client_clusters.get(cid, 0) for cid in client_ids]
    
    # Reduce to 3D
    try:
        coords_3d = reduce_to_3d_umap(embeddings)
    except Exception as e:
        log_stage("projection", f"UMAP failed: {e}, trying PCA")
        coords_3d = reduce_to_3d_pca(embeddings)
    
    # Optionally add concept anchors
    # For now, skip concept anchors (could embed lexicon labels and project them)
    concept_coords = None
    concept_labels = None
    
    # Create visualization
    log_stage("projection", "Creating interactive visualization...")
    html_content = create_3d_visualization(
        coords_3d,
        note_ids,
        client_ids,
        labels,
        cluster_ids,
        concept_coords,
        concept_labels
    )
    
    # Skip writing HTML file - React dashboard handles visualization
    log_stage("projection", "3D projection complete (data in dashboard/src/data.json)")
    
    return None


def main():
    """CLI entry point."""
    try:
        project_3d()
    except Exception as e:
        log_stage("projection", f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
