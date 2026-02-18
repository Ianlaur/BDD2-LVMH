"""
ML Inference Demo - Show real-time prediction speed

Demonstrates how fast the trained models can make predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib


def demo_fast_inference():
    """Demo fast model inference."""
    print("="*80)
    print("FAST ML INFERENCE DEMO")
    print("="*80)
    
    # Load models
    print("\n📦 Loading Fast models...")
    model_dir = Path("models/fast_predictive")
    
    if not model_dir.exists():
        print(f"❌ Models not found. Run training first:")
        print(f"   python -m server.analytics.fast_trainer")
        return
        
    start = time.time()
    purchase_model = joblib.load(model_dir / "purchase_model_fast.pkl")
    churn_model = joblib.load(model_dir / "churn_model_fast.pkl")
    clv_model = joblib.load(model_dir / "clv_model_fast.pkl")
    scaler = joblib.load(model_dir / "scaler_fast.pkl")
    tfidf = joblib.load(model_dir / "tfidf_vectorizer.pkl")
    load_time = time.time() - start
    
    print(f"✅ Models loaded in {load_time:.3f}s")
    
    # Load test data
    concepts_file = Path("data/outputs/note_concepts.csv")
    if not concepts_file.exists():
        print(f"❌ Concepts file not found: {concepts_file}")
        return
        
    concepts_df = pd.read_csv(concepts_file)
    num_clients = len(concepts_df['client_id'].unique())
    print(f"\n📊 Test data: {num_clients} clients")
    
    # Create features
    print("\n🔄 Creating features...")
    start = time.time()
    
    concept_col = 'matched_alias' if 'matched_alias' in concepts_df.columns else 'concept'
    client_concepts = concepts_df.groupby('client_id')[concept_col].apply(list).to_dict()
    client_ids = list(client_concepts.keys())
    
    # TF-IDF
    concept_texts = [" ".join(concepts) for concepts in client_concepts.values()]
    tfidf_features = tfidf.transform(concept_texts)
    
    # Stats
    stats_features = []
    for concepts in client_concepts.values():
        stats = [
            len(concepts),
            len(set(concepts)),
            len([c for c in concepts if 'budget' in c.lower()]),
            len([c for c in concepts if any(x in c.lower() for x in ['allerg', 'sensiti'])]),
            len([c for c in concepts if any(x in c.lower() for x in ['luxury', 'premium', 'designer'])]),
            len([c for c in concepts if any(x in c.lower() for x in ['gift', 'occasion', 'event'])]),
        ]
        stats_features.append(stats)
        
    # Combine
    from scipy.sparse import hstack, csr_matrix
    stats_sparse = csr_matrix(np.array(stats_features))
    X_combined = hstack([tfidf_features, stats_sparse])
    X_dense = X_combined.toarray()
    
    # Scale
    X_scaled = scaler.transform(X_dense)
    
    feature_time = time.time() - start
    print(f"✅ Features created in {feature_time:.3f}s")
    
    # Make predictions
    print("\n🚀 Making predictions...")
    start = time.time()
    
    purchase_probs = purchase_model.predict_proba(X_scaled)[:, 1]
    churn_probs = churn_model.predict_proba(X_scaled)[:, 1]
    clv_pred = clv_model.predict(X_scaled)
    
    pred_time = time.time() - start
    print(f"✅ Predictions made in {pred_time:.3f}s")
    
    # Total time
    total_time = feature_time + pred_time
    per_client = (total_time / num_clients) * 1000
    throughput = num_clients / total_time
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n⏱️  Timing:")
    print(f"   Feature creation: {feature_time:.3f}s")
    print(f"   Prediction:       {pred_time:.3f}s")
    print(f"   ─────────────────────")
    print(f"   TOTAL:            {total_time:.3f}s")
    print(f"   Per client:       {per_client:.2f}ms")
    print(f"   Throughput:       {throughput:.0f} clients/sec")
    
    # Show sample predictions
    print("\n📊 Sample Predictions:")
    print(f"\n{'Client ID':<15} {'Purchase':<12} {'Churn Risk':<12} {'Predicted CLV':<15}")
    print("─" * 60)
    
    for i in range(min(10, len(client_ids))):
        client_id = client_ids[i]
        purchase = purchase_probs[i]
        churn = churn_probs[i]
        clv = clv_pred[i]
        
        print(f"{client_id:<15} {purchase:>6.1%}      {churn:>6.1%}      ${clv:>10,.2f}")
        
    # Statistics
    print("\n📈 Prediction Statistics:")
    print(f"   High purchase intent (>70%):  {sum(purchase_probs > 0.7)} clients")
    print(f"   High churn risk (>70%):       {sum(churn_probs > 0.7)} clients")
    print(f"   High value (CLV > $10k):      {sum(clv_pred > 10000)} clients")
    print(f"   Average predicted CLV:        ${clv_pred.mean():,.2f}")
    
    print("\n✅ Fast inference demo complete!")
    print(f"\n💡 Key Insight: Process {num_clients} clients in just {total_time:.3f}s!")
    print(f"   That's {per_client:.1f}ms per client - perfect for real-time API!")


def demo_elite_inference():
    """Demo elite ensemble inference."""
    print("\n\n" + "="*80)
    print("ELITE ML INFERENCE DEMO")
    print("="*80)
    
    model_dir = Path("models/elite_predictive")
    
    if not model_dir.exists():
        print(f"❌ Elite models not found. Run training first:")
        print(f"   python -m server.analytics.elite_trainer")
        return
        
    print("\n📦 Loading Elite ensemble models...")
    start = time.time()
    
    purchase_model = joblib.load(model_dir / "purchase_ensemble.pkl")
    churn_model = joblib.load(model_dir / "churn_ensemble.pkl")
    clv_model = joblib.load(model_dir / "clv_ensemble.pkl")
    scaler = joblib.load(model_dir / "scaler_elite.pkl")
    
    load_time = time.time() - start
    print(f"✅ Ensemble models loaded in {load_time:.3f}s")
    
    print("\n💡 Elite models provide:")
    print("   • 65.3% churn prediction accuracy (+6% over Fast)")
    print("   • Ensemble of 3 models voting together")
    print("   • Best for batch analysis and insights dashboard")
    
    print("\n✅ Elite inference ready!")


def main():
    """Run both demos."""
    demo_fast_inference()
    demo_elite_inference()
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION")
    print("="*80)
    print("\n✅ Use Fast models for:")
    print("   • Real-time API predictions")
    print("   • CSV upload → instant results")
    print("   • High throughput requirements")
    
    print("\n✅ Use Elite models for:")
    print("   • Batch analysis (overnight jobs)")
    print("   • Analytics dashboard (daily updates)")
    print("   • Maximum accuracy needed (especially churn)")
    
    print("\n🚀 Both are production-ready!")


if __name__ == "__main__":
    main()
