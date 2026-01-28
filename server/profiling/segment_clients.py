"""
Client segmentation stage: Cluster clients and assign profile types.

This module:
- Loads note/client vectors
- Aggregates to client level if multiple notes
- Clusters using KMeans with fixed random_state
- Computes top concepts per cluster from note_concepts.csv
- Assigns human-readable profile_type labels
- Computes confidence scores

Output: data/outputs/client_profiles.csv
"""
import sys
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from server.shared.config import (
    DATA_PROCESSED, DATA_OUTPUTS, TAXONOMY_DIR,
    CLUSTERS_K, SKLEARN_RANDOM_STATE
)
from server.shared.utils import log_stage, set_all_seeds


def compute_k(n_samples: int, env_k: int = CLUSTERS_K) -> int:
    """
    Determine number of clusters.
    If CLUSTERS_K env var is set and > 0, use it.
    Otherwise, use heuristic: min(8, max(3, sqrt(n/2)))
    """
    if env_k > 0:
        return env_k
    
    heuristic_k = int(math.sqrt(n_samples / 2))
    return min(8, max(3, heuristic_k))


def aggregate_client_vectors(vectors_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate note vectors to client level.
    For clients with multiple notes, compute mean embedding.
    
    Returns:
        Tuple of (embeddings array, client_ids list)
    """
    # Group by client_id
    client_groups = vectors_df.groupby("client_id")
    
    client_ids = []
    client_embeddings = []
    
    for client_id, group in sorted(client_groups):
        embeddings = np.array(group["embedding"].tolist())
        # Mean aggregation
        mean_embedding = embeddings.mean(axis=0)
        # Re-normalize
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        
        client_ids.append(client_id)
        client_embeddings.append(mean_embedding)
    
    return np.array(client_embeddings), client_ids


def get_top_concepts_for_cluster(
    cluster_client_ids: List[str],
    note_concepts_df: pd.DataFrame,
    lexicon_df: pd.DataFrame,
    top_n: int = 3
) -> List[str]:
    """
    Get top concepts by frequency for clients in a cluster.
    Returns list of concept labels.
    """
    if note_concepts_df is None or len(note_concepts_df) == 0:
        return []
    
    # Filter to clients in this cluster
    cluster_concepts = note_concepts_df[
        note_concepts_df["client_id"].isin(cluster_client_ids)
    ]
    
    if len(cluster_concepts) == 0:
        return []
    
    # Count concept occurrences
    concept_counts = Counter(cluster_concepts["concept_id"])
    top_concept_ids = [cid for cid, _ in concept_counts.most_common(top_n)]
    
    # Map to labels
    concept_id_to_label = dict(zip(lexicon_df["concept_id"], lexicon_df["label"]))
    top_labels = [
        concept_id_to_label.get(cid, cid)
        for cid in top_concept_ids
    ]
    
    return top_labels


def format_profile_type(concepts: List[str]) -> str:
    """Format top concepts into a profile_type string."""
    if not concepts:
        return "General"
    
    # Clean up labels: capitalize, remove underscores
    cleaned = []
    for c in concepts[:3]:  # Max 3 concepts
        c = str(c).replace("_", " ").title()
        cleaned.append(c)
    
    return " | ".join(cleaned)


def compute_confidence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Compute confidence score for each sample.
    Uses inverse of normalized distance to centroid.
    """
    n_samples = len(embeddings)
    confidences = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster_id = labels[i]
        centroid = centroids[cluster_id]
        
        # Distance to own centroid
        dist_to_centroid = np.linalg.norm(embeddings[i] - centroid)
        
        # Confidence: 1 / (1 + distance)
        confidences[i] = 1.0 / (1.0 + dist_to_centroid)
    
    return confidences


def segment_clients() -> pd.DataFrame:
    """
    Main client segmentation function.
    
    Returns:
        DataFrame with client profiles
        
    Side effects:
        Writes data/outputs/client_profiles.csv
    """
    set_all_seeds()
    
    log_stage("profiles", "Starting client segmentation...")
    
    # Load note vectors
    vectors_path = DATA_OUTPUTS / "note_vectors.parquet"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors not found: {vectors_path}. Run vector building first.")
    
    vectors_df = pd.read_parquet(vectors_path)
    log_stage("profiles", f"Loaded {len(vectors_df)} note vectors")
    
    # Aggregate to client level
    client_embeddings, client_ids = aggregate_client_vectors(vectors_df)
    log_stage("profiles", f"Aggregated to {len(client_ids)} client vectors")
    
    # Determine k
    k = compute_k(len(client_ids))
    log_stage("profiles", f"Using k={k} clusters")
    
    # Ensure k doesn't exceed n_samples
    k = min(k, len(client_ids))
    
    if k < 2:
        log_stage("profiles", "Too few clients for clustering, assigning all to cluster 0")
        labels = np.zeros(len(client_ids), dtype=int)
        centroids = client_embeddings.mean(axis=0, keepdims=True)
    else:
        # Cluster
        log_stage("profiles", "Running KMeans clustering...")
        kmeans = KMeans(
            n_clusters=k,
            random_state=SKLEARN_RANDOM_STATE,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(client_embeddings)
        centroids = kmeans.cluster_centers_
        
        # Silhouette score
        if k >= 2 and len(set(labels)) > 1:
            silhouette = silhouette_score(client_embeddings, labels)
            log_stage("profiles", f"Silhouette score: {silhouette:.4f}")
    
    # Load note concepts and lexicon for profile labeling
    note_concepts_df = None
    lexicon_df = None
    
    concepts_path = DATA_OUTPUTS / "note_concepts.csv"
    if concepts_path.exists():
        note_concepts_df = pd.read_csv(concepts_path)
        note_concepts_df["client_id"] = note_concepts_df["client_id"].astype(str)
    
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.csv"
    if lexicon_path.exists():
        lexicon_df = pd.read_csv(lexicon_path)
    
    # Build profile for each cluster
    cluster_profiles: Dict[int, str] = {}
    cluster_top_concepts: Dict[int, List[str]] = {}
    
    for cluster_id in sorted(set(labels)):
        cluster_client_ids = [
            cid for cid, lbl in zip(client_ids, labels) if lbl == cluster_id
        ]
        
        if note_concepts_df is not None and lexicon_df is not None:
            top_concepts = get_top_concepts_for_cluster(
                cluster_client_ids, note_concepts_df, lexicon_df
            )
        else:
            top_concepts = []
        
        cluster_top_concepts[cluster_id] = top_concepts
        cluster_profiles[cluster_id] = format_profile_type(top_concepts)
    
    # Compute confidence scores
    confidences = compute_confidence(client_embeddings, labels, centroids)
    
    # Build output DataFrame
    profiles_data = []
    for i, (client_id, cluster_id) in enumerate(zip(client_ids, labels)):
        profiles_data.append({
            "client_id": client_id,
            "cluster_id": int(cluster_id),
            "profile_type": cluster_profiles[cluster_id],
            "top_concepts": "|".join(cluster_top_concepts[cluster_id]),
            "confidence": round(float(confidences[i]), 4)
        })
    
    profiles_df = pd.DataFrame(profiles_data)
    
    # Sort by client_id for determinism
    profiles_df = profiles_df.sort_values("client_id").reset_index(drop=True)
    
    # Write output
    output_path = DATA_OUTPUTS / "client_profiles.csv"
    profiles_df.to_csv(output_path, index=False)
    log_stage("profiles", f"Wrote {len(profiles_df)} client profiles to {output_path}")
    
    # Summary
    log_stage("profiles", "Cluster distribution:")
    for cluster_id in sorted(set(labels)):
        count = sum(1 for l in labels if l == cluster_id)
        profile = cluster_profiles[cluster_id]
        log_stage("profiles", f"  Cluster {cluster_id}: {count} clients - {profile}")
    
    log_stage("profiles", "Client segmentation complete!")
    
    return profiles_df


def main():
    """CLI entry point."""
    try:
        segment_clients()
    except Exception as e:
        log_stage("profiles", f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
