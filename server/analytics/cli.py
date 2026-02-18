"""
ML Analytics CLI - Run post-extraction analytics

Usage:
    python -m server.analytics.cli clustering --concepts outputs/concepts.csv
    python -m server.analytics.cli predictions --concepts outputs/concepts.csv --labels data/labels.csv
    python -m server.analytics.cli recommendations --concepts outputs/concepts.csv --catalog data/products.csv
    python -m server.analytics.cli all --concepts outputs/concepts.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

from .clustering import ClientClusterer
from .predictions import PredictiveAnalytics
from .recommendations import RecommendationEngine


def run_clustering(args):
    """Run client clustering analysis."""
    print("\n" + "="*80)
    print("CLIENT CLUSTERING & SIMILARITY ANALYSIS")
    print("="*80)
    
    # Load concepts
    concepts_df = pd.read_csv(args.concepts)
    print(f"📊 Loaded {len(concepts_df)} concept matches from {len(concepts_df['client_id'].unique())} clients")
    
    # Initialize clusterer
    clusterer = ClientClusterer(model_name=args.model)
    
    # Compute embeddings
    embeddings = clusterer.compute_embeddings(concepts_df)
    
    # Cluster clients
    n_clusters = args.n_clusters or min(5, len(embeddings) // 20)  # Auto-determine
    clusters = clusterer.cluster_clients(n_clusters=n_clusters, method=args.cluster_method)
    
    # Describe clusters
    print("\n" + "="*80)
    print("CLUSTER DESCRIPTIONS")
    print("="*80)
    
    for cluster_id in sorted(set(clusters.values())):
        desc = clusterer.describe_cluster(cluster_id, concepts_df, top_k=10)
        print(f"\n📊 Cluster {cluster_id}: {desc['size']} clients")
        print(f"   Top concepts:")
        for concept, count in list(desc['top_concepts'].items())[:5]:
            print(f"      • {concept}: {count} mentions")
            
    # Show similarity examples
    print("\n" + "="*80)
    print("SIMILARITY EXAMPLES")
    print("="*80)
    
    sample_clients = list(embeddings.keys())[:3]
    for client_id in sample_clients:
        similar = clusterer.find_similar_clients(client_id, top_k=3)
        print(f"\n🔍 Clients similar to {client_id}:")
        for sim_id, score in similar:
            print(f"   • {sim_id}: {score:.3f} similarity")
            
    # Export results
    output_path = args.output or Path("outputs/clustering_results.json")
    clusterer.export_results(output_path)
    
    print("\n✅ Clustering analysis complete!")


def run_predictions(args):
    """Run predictive analytics."""
    print("\n" + "="*80)
    print("PREDICTIVE ANALYTICS")
    print("="*80)
    
    # Load concepts
    concepts_df = pd.read_csv(args.concepts)
    print(f"📊 Loaded {len(concepts_df)} concept matches")
    
    # Initialize analytics
    analytics = PredictiveAnalytics()
    
    # Create feature matrix
    features_df, feature_names = analytics.create_feature_matrix(concepts_df)
    print(f"✅ Created feature matrix with {len(feature_names)} features")
    
    if args.labels:
        # Load labels and train models
        labels_df = pd.read_csv(args.labels)
        print(f"📊 Loaded labels for {len(labels_df)} clients")
        
        # Train models if labels available
        if 'purchased' in labels_df.columns:
            analytics.train_purchase_predictor(features_df, labels_df)
            
        if 'churned' in labels_df.columns:
            analytics.train_churn_predictor(features_df, labels_df)
            
        if 'lifetime_value' in labels_df.columns:
            analytics.train_clv_predictor(features_df, labels_df)
            
        # Save models
        output_dir = args.output or Path("outputs/predictive_models")
        analytics.save_models(output_dir)
        
    else:
        # No labels - run demo with synthetic data
        print("\n⚠️  No labels provided. Running demo with synthetic data...")
        print("   To train on real data, provide --labels file with columns:")
        print("   - client_id, purchased (0/1), churned (0/1), lifetime_value (float)")
        
        # Generate synthetic labels for demo
        import numpy as np
        np.random.seed(42)
        
        labels = []
        for client_id in concepts_df['client_id'].unique():
            labels.append({
                'client_id': client_id,
                'purchased': np.random.randint(0, 2),
                'churned': np.random.randint(0, 2),
                'lifetime_value': np.random.uniform(1000, 50000)
            })
        labels_df = pd.DataFrame(labels)
        
        # Train on synthetic data
        analytics.train_purchase_predictor(features_df, labels_df)
        analytics.train_churn_predictor(features_df, labels_df)
        analytics.train_clv_predictor(features_df, labels_df)
        
    print("\n✅ Predictive analytics complete!")


def run_recommendations(args):
    """Run recommendation engine."""
    print("\n" + "="*80)
    print("RECOMMENDATION ENGINE")
    print("="*80)
    
    # Load concepts
    concepts_df = pd.read_csv(args.concepts)
    print(f"📊 Loaded {len(concepts_df)} concept matches from {len(concepts_df['client_id'].unique())} clients")
    
    # Initialize recommendation engine
    engine = RecommendationEngine(model_name=args.model)
    
    if args.catalog:
        # Load real product catalog
        catalog_df = pd.read_csv(args.catalog)
        print(f"📦 Loaded {len(catalog_df)} products from catalog")
        engine.load_product_catalog(catalog_df)
    else:
        # Create sample LVMH catalog
        print("\n⚠️  No catalog provided. Using sample LVMH products...")
        print("   To use real catalog, provide --catalog file with columns:")
        print("   - product_id, name, description, category, price")
        
        sample_catalog = pd.DataFrame([
            {
                'product_id': 'LV001',
                'name': 'Louis Vuitton Neverfull MM',
                'description': 'Iconic tote bag in monogram canvas, perfect for everyday luxury',
                'category': 'Handbags',
                'price': 1500
            },
            {
                'product_id': 'TAG001',
                'name': 'TAG Heuer Carrera Chronograph',
                'description': 'Swiss luxury sports chronograph with heritage racing design',
                'category': 'Watches',
                'price': 5500
            },
            {
                'product_id': 'DIOR001',
                'name': 'Dior J\'adore Eau de Parfum',
                'description': 'Elegant floral fragrance, perfect gift for sophisticated women',
                'category': 'Perfume',
                'price': 150
            },
            {
                'product_id': 'BULL001',
                'name': 'Bulgari Serpenti Watch',
                'description': 'Luxury jewelry watch with iconic serpent design, modern elegance',
                'category': 'Jewelry & Watches',
                'price': 8000
            },
            {
                'product_id': 'FENDI001',
                'name': 'Fendi Peekaboo Bag',
                'description': 'Contemporary designer handbag for fashion-forward clients',
                'category': 'Handbags',
                'price': 4500
            },
            {
                'product_id': 'HUBLOT001',
                'name': 'Hublot Big Bang Unico',
                'description': 'Bold modern chronograph for contemporary watch enthusiasts',
                'category': 'Watches',
                'price': 18000
            },
            {
                'product_id': 'CELINE001',
                'name': 'Celine Classic Box Bag',
                'description': 'Minimalist luxury bag, vintage-inspired design',
                'category': 'Handbags',
                'price': 3500
            },
            {
                'product_id': 'ZENITH001',
                'name': 'Zenith Chronomaster Heritage',
                'description': 'Vintage-inspired chronograph with heritage El Primero movement',
                'category': 'Watches',
                'price': 7500
            }
        ])
        
        engine.load_product_catalog(sample_catalog)
        
    # Create client embeddings
    engine.create_client_embeddings(concepts_df)
    
    # Generate recommendations for all clients
    all_recommendations = engine.recommend_for_all_clients(top_k=args.top_k)
    
    # Show sample recommendations
    print("\n" + "="*80)
    print("SAMPLE RECOMMENDATIONS")
    print("="*80)
    
    sample_clients = list(all_recommendations.keys())[:3]
    for client_id in sample_clients:
        print(f"\n🎯 Top recommendations for {client_id}:")
        
        # Show client concepts
        concept_col = 'matched_alias' if 'matched_alias' in concepts_df.columns else 'concept'
        client_concepts = concepts_df[
            concepts_df['client_id'] == client_id
        ][concept_col].unique()[:5]
        print(f"   Based on interests: {', '.join(client_concepts)}")
        
        # Show recommendations
        for i, rec in enumerate(all_recommendations[client_id][:3], 1):
            print(f"\n   {i}. {rec['name']}")
            print(f"      Category: {rec['category']} | Score: {rec['similarity_score']:.3f}")
            if rec['price']:
                print(f"      Price: ${rec['price']:,}")
                
    # Export recommendations
    output_path = args.output or Path("outputs/recommendations.json")
    engine.export_recommendations(all_recommendations, output_path)
    
    print("\n✅ Recommendation engine complete!")


def run_all(args):
    """Run all analytics."""
    print("\n" + "="*80)
    print("RUNNING ALL ML ANALYTICS")
    print("="*80)
    
    # Run clustering
    clustering_args = argparse.Namespace(
        concepts=args.concepts,
        model=args.model,
        n_clusters=args.n_clusters,
        cluster_method='kmeans',
        output=Path("outputs/clustering_results.json")
    )
    run_clustering(clustering_args)
    
    # Run predictions (with synthetic data if no labels)
    predictions_args = argparse.Namespace(
        concepts=args.concepts,
        labels=args.labels,
        output=Path("outputs/predictive_models")
    )
    run_predictions(predictions_args)
    
    # Run recommendations
    recommendations_args = argparse.Namespace(
        concepts=args.concepts,
        catalog=args.catalog,
        model=args.model,
        top_k=5,
        output=Path("outputs/recommendations.json")
    )
    run_recommendations(recommendations_args)
    
    print("\n" + "="*80)
    print("✅ ALL ANALYTICS COMPLETE!")
    print("="*80)
    print("\nOutputs saved to:")
    print("  • outputs/clustering_results.json")
    print("  • outputs/predictive_models/")
    print("  • outputs/recommendations.json")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Analytics for extracted concepts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run clustering analysis
  python -m server.analytics.cli clustering --concepts outputs/concepts.csv
  
  # Train predictive models
  python -m server.analytics.cli predictions --concepts outputs/concepts.csv --labels data/labels.csv
  
  # Generate recommendations
  python -m server.analytics.cli recommendations --concepts outputs/concepts.csv --catalog data/products.csv
  
  # Run all analytics
  python -m server.analytics.cli all --concepts outputs/concepts.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analytics command')
    
    # Clustering command
    cluster_parser = subparsers.add_parser('clustering', help='Client clustering & similarity')
    cluster_parser.add_argument('--concepts', required=True, help='Path to concepts CSV')
    cluster_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model')
    cluster_parser.add_argument('--n-clusters', type=int, help='Number of clusters')
    cluster_parser.add_argument('--cluster-method', default='kmeans', choices=['kmeans', 'dbscan'])
    cluster_parser.add_argument('--output', type=Path, help='Output path')
    
    # Predictions command
    pred_parser = subparsers.add_parser('predictions', help='Predictive analytics')
    pred_parser.add_argument('--concepts', required=True, help='Path to concepts CSV')
    pred_parser.add_argument('--labels', help='Path to labels CSV (client_id, purchased, churned, lifetime_value)')
    pred_parser.add_argument('--output', type=Path, help='Output directory for models')
    
    # Recommendations command
    rec_parser = subparsers.add_parser('recommendations', help='Product recommendations')
    rec_parser.add_argument('--concepts', required=True, help='Path to concepts CSV')
    rec_parser.add_argument('--catalog', help='Path to product catalog CSV')
    rec_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model')
    rec_parser.add_argument('--top-k', type=int, default=5, help='Number of recommendations per client')
    rec_parser.add_argument('--output', type=Path, help='Output path')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run all analytics')
    all_parser.add_argument('--concepts', required=True, help='Path to concepts CSV')
    all_parser.add_argument('--labels', help='Path to labels CSV (optional)')
    all_parser.add_argument('--catalog', help='Path to product catalog CSV (optional)')
    all_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model')
    all_parser.add_argument('--n-clusters', type=int, help='Number of clusters')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Route to appropriate function
    if args.command == 'clustering':
        run_clustering(args)
    elif args.command == 'predictions':
        run_predictions(args)
    elif args.command == 'recommendations':
        run_recommendations(args)
    elif args.command == 'all':
        run_all(args)


if __name__ == "__main__":
    main()
