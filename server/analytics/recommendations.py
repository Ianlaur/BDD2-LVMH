"""
Recommendation Engine - Product/Service Recommendations

Matches clients to products based on concept profiles.
Example: Client with [vintage, watches, heritage] → Recommend Patek Philippe
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from server.shared.model_cache import get_sentence_transformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime


class RecommendationEngine:
    """
    Recommend products/services to clients based on extracted concepts.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize recommendation engine.
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.client_embeddings = {}
        self.product_embeddings = {}
        self.product_catalog = {}
        
    def _ensure_model_loaded(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = get_sentence_transformer()
            
    def load_product_catalog(
        self, 
        catalog_df: pd.DataFrame,
        product_id_col: str = "product_id",
        name_col: str = "name",
        description_col: str = "description",
        category_col: str = "category",
        price_col: Optional[str] = "price"
    ):
        """
        Load product catalog and compute embeddings.
        
        Args:
            catalog_df: DataFrame with product information
            product_id_col: Column with product ID
            name_col: Column with product name
            description_col: Column with product description
            category_col: Column with product category
            price_col: Optional column with price
        """
        self._ensure_model_loaded()
        
        print(f"\n🔄 Loading product catalog ({len(catalog_df)} products)...")
        
        for _, row in catalog_df.iterrows():
            product_id = row[product_id_col]
            
            # Create product profile for embedding
            profile = f"{row[name_col]}. {row[description_col]}. Category: {row[category_col]}"
            
            # Store product info
            self.product_catalog[product_id] = {
                'name': row[name_col],
                'description': row[description_col],
                'category': row[category_col],
                'price': row.get(price_col) if price_col else None
            }
            
            # Compute embedding
            embedding = self.model.encode(profile, show_progress_bar=False)
            self.product_embeddings[product_id] = embedding
            
        print(f"✅ Loaded {len(self.product_catalog)} products")
        
    def create_client_embeddings(
        self,
        concepts_df: pd.DataFrame,
        client_id_col: str = "client_id",
        concept_col: str = "concept"
    ):
        """
        Create embeddings for clients based on their concepts.
        
        Args:
            concepts_df: DataFrame with client concepts
            client_id_col: Name of client ID column
            concept_col: Name of concept column (or matched_alias)
        """
        self._ensure_model_loaded()
        
        print(f"\n🔄 Creating client embeddings...")
        
        # Use matched_alias if concept column not available
        if concept_col not in concepts_df.columns and 'matched_alias' in concepts_df.columns:
            concept_col = 'matched_alias'
            print(f"   Using '{concept_col}' column for concepts")
        
        # Group concepts by client
        client_concepts = concepts_df.groupby(client_id_col)[concept_col].apply(list).to_dict()
        
        for client_id, concepts in client_concepts.items():
            # Create profile from concepts
            profile = " ".join(concepts)
            
            # Compute embedding
            embedding = self.model.encode(profile, show_progress_bar=False)
            self.client_embeddings[client_id] = embedding
            
        print(f"✅ Created embeddings for {len(self.client_embeddings)} clients")
        
    def recommend_for_client(
        self,
        client_id: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """
        Get product recommendations for a client.
        
        Args:
            client_id: Client to recommend for
            top_k: Number of recommendations
            category_filter: Optional category to filter by
            price_range: Optional (min, max) price range
            
        Returns:
            List of recommendation dictionaries
        """
        if client_id not in self.client_embeddings:
            return []
            
        client_emb = self.client_embeddings[client_id].reshape(1, -1)
        
        # Compute similarities with all products
        recommendations = []
        for product_id, product_emb in self.product_embeddings.items():
            product_info = self.product_catalog[product_id]
            
            # Apply filters
            if category_filter and product_info['category'] != category_filter:
                continue
                
            if price_range and product_info['price']:
                if not (price_range[0] <= product_info['price'] <= price_range[1]):
                    continue
                    
            # Compute similarity
            prod_emb = product_emb.reshape(1, -1)
            similarity = cosine_similarity(client_emb, prod_emb)[0][0]
            
            recommendations.append({
                'product_id': product_id,
                'name': product_info['name'],
                'description': product_info['description'],
                'category': product_info['category'],
                'price': product_info['price'],
                'similarity_score': float(similarity)
            })
            
        # Sort by similarity
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return recommendations[:top_k]
        
    def recommend_for_all_clients(
        self,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Generate recommendations for all clients.
        
        Args:
            top_k: Number of recommendations per client
            
        Returns:
            Dictionary mapping client_id to recommendations
        """
        print(f"\n🔄 Generating recommendations for {len(self.client_embeddings)} clients...")
        
        all_recommendations = {}
        for client_id in self.client_embeddings.keys():
            recommendations = self.recommend_for_client(client_id, top_k=top_k)
            all_recommendations[client_id] = recommendations
            
        print(f"✅ Generated {len(all_recommendations)} recommendation sets")
        return all_recommendations
        
    def explain_recommendation(
        self,
        client_id: str,
        product_id: str,
        concepts_df: pd.DataFrame,
        concept_col: str = "concept"
    ) -> Dict:
        """
        Explain why a product was recommended to a client.
        
        Args:
            client_id: Client ID
            product_id: Product ID
            concepts_df: DataFrame with client concepts
            concept_col: Name of concept column (or matched_alias)
            
        Returns:
            Explanation dictionary
        """
        # Use matched_alias if concept column not available
        if concept_col not in concepts_df.columns and 'matched_alias' in concepts_df.columns:
            concept_col = 'matched_alias'
            
        # Get client concepts
        client_concepts = concepts_df[
            concepts_df['client_id'] == client_id
        ][concept_col].tolist()
        
        # Get product info
        product_info = self.product_catalog.get(product_id)
        if not product_info:
            return {}
            
        # Compute similarity
        if client_id in self.client_embeddings and product_id in self.product_embeddings:
            client_emb = self.client_embeddings[client_id].reshape(1, -1)
            product_emb = self.product_embeddings[product_id].reshape(1, -1)
            similarity = cosine_similarity(client_emb, product_emb)[0][0]
        else:
            similarity = 0.0
            
        return {
            'client_id': client_id,
            'product_id': product_id,
            'product_name': product_info['name'],
            'similarity_score': float(similarity),
            'client_concepts': client_concepts,
            'product_description': product_info['description'],
            'explanation': f"Based on your interests in {', '.join(client_concepts[:3])}, "
                          f"we recommend {product_info['name']}"
        }
        
    def collaborative_filtering(
        self,
        client_id: str,
        similar_clients: List[str],
        purchase_history: pd.DataFrame,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Recommend products based on what similar clients purchased.
        
        Args:
            client_id: Target client
            similar_clients: List of similar client IDs
            purchase_history: DataFrame with client_id and product_id
            top_k: Number of recommendations
            
        Returns:
            List of recommended products
        """
        # Get products purchased by similar clients
        similar_purchases = purchase_history[
            purchase_history['client_id'].isin(similar_clients)
        ]['product_id'].tolist()
        
        # Get products already purchased by target client
        client_purchases = purchase_history[
            purchase_history['client_id'] == client_id
        ]['product_id'].tolist()
        
        # Count product frequencies (excluding already purchased)
        product_counts = pd.Series(similar_purchases).value_counts()
        recommendations = []
        
        for product_id, count in product_counts.items():
            if product_id in client_purchases:
                continue  # Skip already purchased
                
            if product_id not in self.product_catalog:
                continue
                
            product_info = self.product_catalog[product_id]
            recommendations.append({
                'product_id': product_id,
                'name': product_info['name'],
                'category': product_info['category'],
                'popularity': int(count),
                'reason': f"Purchased by {count} similar clients"
            })
            
        recommendations.sort(key=lambda x: x['popularity'], reverse=True)
        return recommendations[:top_k]
        
    def export_recommendations(
        self,
        recommendations: Dict[str, List[Dict]],
        output_path: Path
    ):
        """
        Export recommendations to JSON.
        
        Args:
            recommendations: Dictionary of recommendations
            output_path: Path to save file
        """
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'num_clients': len(recommendations),
            'recommendations': recommendations
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Exported recommendations to {output_path}")


def main():
    """Test recommendation engine with sample data."""
    print("🎁 Recommendation Engine - Demo")
    print("="*80)
    
    # Create sample product catalog
    products = [
        {
            'product_id': 'P001',
            'name': 'Patek Philippe Nautilus',
            'description': 'Luxury vintage sports watch with heritage design',
            'category': 'Watches',
            'price': 50000
        },
        {
            'product_id': 'P002',
            'name': 'Dior Lady Handbag',
            'description': 'Elegant fashion handbag for modern women',
            'category': 'Fashion',
            'price': 5000
        },
        {
            'product_id': 'P003',
            'name': 'Louis Vuitton Keepall',
            'description': 'Classic travel bag with monogram canvas',
            'category': 'Fashion',
            'price': 2500
        },
        {
            'product_id': 'P004',
            'name': 'Tiffany Diamond Ring',
            'description': 'Luxury engagement ring perfect for special occasions',
            'category': 'Jewelry',
            'price': 15000
        },
        {
            'product_id': 'P005',
            'name': 'TAG Heuer Connected',
            'description': 'Modern smartwatch with luxury design',
            'category': 'Watches',
            'price': 2000
        },
    ]
    
    catalog_df = pd.DataFrame(products)
    
    # Create sample client concepts
    client_concepts = [
        {'client_id': 'CA001', 'concept': 'vintage'},
        {'client_id': 'CA001', 'concept': 'watches'},
        {'client_id': 'CA001', 'concept': 'heritage'},
        {'client_id': 'CA002', 'concept': 'modern'},
        {'client_id': 'CA002', 'concept': 'fashion'},
        {'client_id': 'CA002', 'concept': 'luxury'},
        {'client_id': 'CA003', 'concept': 'jewelry'},
        {'client_id': 'CA003', 'concept': 'gift'},
        {'client_id': 'CA003', 'concept': 'engagement'},
    ]
    
    concepts_df = pd.DataFrame(client_concepts)
    
    # Initialize recommendation engine
    engine = RecommendationEngine()
    
    # Load catalog
    engine.load_product_catalog(catalog_df)
    
    # Create client embeddings
    engine.create_client_embeddings(concepts_df)
    
    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    for client_id in concepts_df['client_id'].unique():
        print(f"\n🎯 Recommendations for {client_id}:")
        
        # Show client concepts
        client_concepts_list = concepts_df[
            concepts_df['client_id'] == client_id
        ]['concept'].tolist()
        print(f"   Interests: {', '.join(client_concepts_list)}")
        
        # Get recommendations
        recommendations = engine.recommend_for_client(client_id, top_k=3)
        
        print(f"   Top recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec['name']} ({rec['category']})")
            print(f"         Score: {rec['similarity_score']:.3f} | Price: ${rec['price']:,}")
            
    # Export
    all_recs = engine.recommend_for_all_clients(top_k=5)
    engine.export_recommendations(all_recs, Path("outputs/recommendations.json"))
    
    print("\n✅ Recommendation engine demo complete!")


if __name__ == "__main__":
    main()
