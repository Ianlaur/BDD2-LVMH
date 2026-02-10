"""
Client Clustering & Similarity Analysis

Uses embeddings to find similar clients even if they use different words.
Example: "loves modern art" ≈ "prefers contemporary pieces"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from server.shared.model_cache import get_sentence_transformer
import json
from datetime import datetime


class ClientClusterer:
    """
    Find similar clients and create segments based on extracted concepts.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the clustering system.
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.client_embeddings = {}
        self.clusters = {}
        
    def _ensure_model_loaded(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = get_sentence_transformer()
            
    def create_client_profile(self, concepts: List[str]) -> str:
        """
        Create a text profile from extracted concepts.
        
        Args:
            concepts: List of concept strings
            
        Returns:
            Text profile for embedding
        """
        if not concepts:
            return "No preferences recorded"
            
        # Join concepts into a readable profile
        return " ".join(concepts)
        
    def compute_embeddings(
        self, 
        concepts_df: pd.DataFrame,
        client_id_col: str = "client_id",
        concept_col: str = "concept"
    ) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for all clients based on their concepts.
        
        Args:
            concepts_df: DataFrame with client_id and concept columns
            client_id_col: Name of client ID column
            concept_col: Name of concept column (or use matched_alias if available)
            
        Returns:
            Dictionary mapping client_id to embedding vector
        """
        self._ensure_model_loaded()
        
        print("\n🔄 Computing client embeddings...")
        
        # Use matched_alias if concept column not available
        if concept_col not in concepts_df.columns and 'matched_alias' in concepts_df.columns:
            concept_col = 'matched_alias'
            print(f"   Using '{concept_col}' column for concepts")
        
        # Group concepts by client
        client_concepts = concepts_df.groupby(client_id_col)[concept_col].apply(list).to_dict()
        
        # Create profiles and compute embeddings
        embeddings = {}
        for client_id, concepts in client_concepts.items():
            profile = self.create_client_profile(concepts)
            embedding = self.model.encode(profile, show_progress_bar=False)
            embeddings[client_id] = embedding
            
        self.client_embeddings = embeddings
        print(f"✅ Computed embeddings for {len(embeddings)} clients")
        
        return embeddings
        
    def find_similar_clients(
        self, 
        client_id: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar clients to a given client.
        
        Args:
            client_id: Client to find similarities for
            top_k: Number of similar clients to return
            
        Returns:
            List of (client_id, similarity_score) tuples
        """
        if client_id not in self.client_embeddings:
            return []
            
        target_embedding = self.client_embeddings[client_id].reshape(1, -1)
        
        # Compute similarities with all other clients
        similarities = []
        for other_id, other_embedding in self.client_embeddings.items():
            if other_id == client_id:
                continue
                
            other_emb = other_embedding.reshape(1, -1)
            sim = cosine_similarity(target_embedding, other_emb)[0][0]
            similarities.append((other_id, float(sim)))
            
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
        
    def cluster_clients(
        self, 
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Dict[str, int]:
        """
        Cluster clients into segments based on their concept profiles.
        
        Args:
            n_clusters: Number of clusters to create
            method: "kmeans" or "dbscan"
            
        Returns:
            Dictionary mapping client_id to cluster_id
        """
        if not self.client_embeddings:
            raise ValueError("Must compute embeddings first")
            
        print(f"\n🔄 Clustering {len(self.client_embeddings)} clients...")
        
        # Prepare data
        client_ids = list(self.client_embeddings.keys())
        X = np.array([self.client_embeddings[cid] for cid in client_ids])
        
        # Cluster
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(X)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.3, min_samples=2)
            labels = clusterer.fit_predict(X)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Map client IDs to cluster labels
        clusters = {client_ids[i]: int(labels[i]) for i in range(len(client_ids))}
        self.clusters = clusters
        
        # Print cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        print(f"✅ Created {len(cluster_counts)} clusters:")
        for cluster_id, count in cluster_counts.items():
            print(f"   Cluster {cluster_id}: {count} clients")
            
        return clusters
        
    def describe_cluster(
        self, 
        cluster_id: int,
        concepts_df: pd.DataFrame,
        client_id_col: str = "client_id",
        concept_col: str = "concept",
        top_k: int = 10
    ) -> Dict:
        """
        Describe a cluster by its most common concepts.
        
        Args:
            cluster_id: Cluster to describe
            concepts_df: DataFrame with concepts
            client_id_col: Name of client ID column
            concept_col: Name of concept column (or matched_alias)
            top_k: Number of top concepts to show
            
        Returns:
            Dictionary with cluster description
        """
        if not self.clusters:
            raise ValueError("Must cluster clients first")
            
        # Use matched_alias if concept column not available
        if concept_col not in concepts_df.columns and 'matched_alias' in concepts_df.columns:
            concept_col = 'matched_alias'
            
        # Get clients in this cluster
        cluster_clients = [cid for cid, cid_cluster in self.clusters.items() 
                          if cid_cluster == cluster_id]
        
        if not cluster_clients:
            return {"cluster_id": cluster_id, "size": 0, "top_concepts": []}
            
        # Get concepts for these clients
        cluster_concepts = concepts_df[
            concepts_df[client_id_col].isin(cluster_clients)
        ][concept_col].tolist()
        
        # Count concept frequencies
        concept_counts = pd.Series(cluster_concepts).value_counts()
        
        return {
            "cluster_id": cluster_id,
            "size": len(cluster_clients),
            "client_ids": cluster_clients[:10],  # Sample
            "top_concepts": concept_counts.head(top_k).to_dict()
        }
        
    def export_results(self, output_path: Path):
        """
        Export clustering results to JSON.
        
        Args:
            output_path: Path to save results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "num_clients": len(self.client_embeddings),
            "num_clusters": len(set(self.clusters.values())) if self.clusters else 0,
            "clusters": self.clusters
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Exported clustering results to {output_path}")


def main():
    """Test the clustering system."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load sample data
    concepts_file = Path("outputs/concepts.csv")
    if not concepts_file.exists():
        print("❌ No concepts file found. Run extraction first.")
        return
        
    concepts_df = pd.read_csv(concepts_file)
    print(f"📊 Loaded {len(concepts_df)} concept matches")
    
    # Initialize clusterer
    clusterer = ClientClusterer()
    
    # Compute embeddings
    embeddings = clusterer.compute_embeddings(concepts_df)
    
    # Cluster clients
    clusters = clusterer.cluster_clients(n_clusters=5)
    
    # Describe each cluster
    print("\n" + "="*80)
    print("CLUSTER DESCRIPTIONS")
    print("="*80)
    
    for cluster_id in sorted(set(clusters.values())):
        desc = clusterer.describe_cluster(cluster_id, concepts_df)
        print(f"\n📊 Cluster {cluster_id} ({desc['size']} clients):")
        print(f"   Top concepts:")
        for concept, count in list(desc['top_concepts'].items())[:5]:
            print(f"      - {concept}: {count}")
            
    # Find similar clients (example)
    sample_client = list(embeddings.keys())[0]
    similar = clusterer.find_similar_clients(sample_client, top_k=3)
    print(f"\n🔍 Clients similar to {sample_client}:")
    for client_id, score in similar:
        print(f"   - {client_id}: {score:.3f} similarity")
        
    # Export results
    clusterer.export_results(Path("outputs/clustering_results.json"))


if __name__ == "__main__":
    main()
