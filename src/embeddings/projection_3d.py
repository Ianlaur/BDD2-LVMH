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

from src.config import (
    DATA_OUTPUTS, TAXONOMY_DIR, DEMO_DIR,
    UMAP_RANDOM_STATE, NUMPY_SEED
)
from src.utils import log_stage, set_all_seeds


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
    concept_coords: Optional[np.ndarray] = None,
    concept_labels: Optional[List[str]] = None
) -> str:
    """
    Create interactive Plotly 3D scatter plot.
    Returns HTML string.
    """
    import plotly.graph_objects as go
    
    # Prepare hover text
    if labels:
        hover_texts = [
            f"Note: {nid}<br>Client: {cid}<br>Profile: {lbl}"
            for nid, cid, lbl in zip(note_ids, client_ids, labels)
        ]
    else:
        hover_texts = [
            f"Note: {nid}<br>Client: {cid}"
            for nid, cid in zip(note_ids, client_ids)
        ]
    
    # Create figure
    fig = go.Figure()
    
    # Add note/client points
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=list(range(len(coords))),
            colorscale='Viridis',
            opacity=0.7
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Notes/Clients'
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
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='LVMH Client Intelligence: Embedding Space (3D)',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1000,
        height=700
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
    
    # Try to load profile labels
    labels = None
    profiles_path = DATA_OUTPUTS / "client_profiles.csv"
    if profiles_path.exists():
        profiles_df = pd.read_csv(profiles_path)
        client_labels = dict(zip(profiles_df["client_id"].astype(str), profiles_df["profile_type"]))
        labels = [client_labels.get(cid, "Unknown") for cid in client_ids]
    
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
        concept_coords,
        concept_labels
    )
    
    # Write output
    output_path = DEMO_DIR / "embedding_space_3d.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    log_stage("projection", f"Wrote visualization to {output_path}")
    log_stage("projection", "3D projection complete!")
    
    return str(output_path)


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
