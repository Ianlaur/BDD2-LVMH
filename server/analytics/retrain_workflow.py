"""
Practical Continuous Training Workflow

Shows how to:
1. Collect real labels from user feedback
2. Update models incrementally
3. Full retrain when needed
4. Compare and deploy best model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sys

# Import our continuous trainer
from server.analytics.continuous_trainer import ContinuousTrainer


def create_synthetic_outcomes(days_ago: int = 1) -> pd.DataFrame:
    """
    Simulate real outcomes (in production, this would come from your database).
    
    Args:
        days_ago: How many days ago were these predictions made
        
    Returns:
        DataFrame with actual outcomes
    """
    print(f"\n📊 Simulating actual outcomes from {days_ago} days ago...")
    
    # Load some real client data
    df = pd.read_csv("data/LVMH_Sales_Database.csv")
    
    # Sample 200 clients
    clients = df.sample(min(200, len(df)), random_state=42 + days_ago)
    
    # Simulate actual outcomes (in production, you'd query your database)
    outcomes = []
    
    for _, client in clients.iterrows():
        # Simulate purchase outcome (70% accuracy from our current model)
        concepts = df[df['Client ID'] == client['Client ID']]['Matched Alias'].dropna()
        num_concepts = len(concepts)
        
        # Business logic for actual purchase
        purchased = (
            num_concepts >= 3 and  # Active clients
            np.random.random() < 0.3  # 30% conversion
        )
        
        # Simulate churn (inverse of engagement)
        churned = (
            num_concepts < 2 and
            np.random.random() < 0.5
        )
        
        # Simulate CLV (based on engagement + randomness)
        base_clv = num_concepts * 2000 + np.random.randint(-1000, 3000)
        actual_clv = max(0, base_clv + np.random.normal(0, 1500))
        
        outcomes.append({
            'client_id': client['Client ID'],
            'purchased_actual': 1 if purchased else 0,
            'churned_actual': 1 if churned else 0,
            'clv_actual': actual_clv,
            'date': (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        })
        
    outcomes_df = pd.DataFrame(outcomes)
    
    print(f"   ✓ Created {len(outcomes_df)} actual outcomes")
    print(f"   - Purchased: {outcomes_df['purchased_actual'].sum()} clients")
    print(f"   - Churned: {outcomes_df['churned_actual'].sum()} clients")
    print(f"   - Avg CLV: ${outcomes_df['clv_actual'].mean():.2f}")
    
    return outcomes_df


def extract_features(client_ids: list, concepts_df: pd.DataFrame) -> np.ndarray:
    """
    Extract features for given clients (matching the training format).
    
    Args:
        client_ids: List of client IDs
        concepts_df: DataFrame with all concept matches
        
    Returns:
        Feature matrix
    """
    print(f"\n🔧 Extracting features for {len(client_ids)} clients...")
    
    # Get concepts for each client
    client_concepts = {}
    for client_id in client_ids:
        concepts = concepts_df[concepts_df['Client ID'] == client_id]['Matched Alias'].dropna().tolist()
        client_concepts[client_id] = ' '.join(concepts) if concepts else 'unknown'
        
    # Create TF-IDF features (matching training format)
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    tfidf_features = vectorizer.fit_transform(client_concepts.values()).toarray()
    
    # Add stats features
    stats_features = []
    for client_id in client_ids:
        concepts = concepts_df[concepts_df['Client ID'] == client_id]['Matched Alias'].dropna()
        stats_features.append([
            len(concepts),  # num_concepts
            len(set(concepts)),  # unique_concepts
            len(concepts) / max(1, len(set(concepts))),  # repeat_rate
            1 if len(concepts) > 5 else 0,  # is_active
            1 if len(concepts) < 2 else 0,  # is_at_risk
            len(concepts) * 1000  # engagement_score
        ])
        
    stats_features = np.array(stats_features)
    
    # Combine
    features = np.hstack([tfidf_features, stats_features])
    
    print(f"   ✓ Created feature matrix: {features.shape}")
    
    return features


def workflow_incremental_update():
    """
    Workflow 1: Daily incremental update with new data.
    """
    print("\n" + "="*80)
    print("WORKFLOW 1: INCREMENTAL UPDATE (Daily)")
    print("="*80)
    
    trainer = ContinuousTrainer()
    
    # Step 1: Load yesterday's predictions
    print("\n1️⃣ Loading yesterday's predictions...")
    # In production: predictions_df = pd.read_csv('outputs/predictions_2026-02-09.csv')
    # For demo, we'll simulate
    print("   (In production: Load from outputs/predictions_YYYY-MM-DD.csv)")
    
    # Step 2: Get actual outcomes
    outcomes_df = create_synthetic_outcomes(days_ago=1)
    
    # Step 3: Load existing model
    print("\n2️⃣ Loading existing model...")
    model_path = Path("models/elite_predictive/purchase_model.pkl")
    if not model_path.exists():
        print(f"   ❌ Model not found: {model_path}")
        print("   Run: python -m server.analytics.elite_trainer first")
        return
        
    old_model = joblib.load(model_path)
    print(f"   ✓ Loaded model from {model_path}")
    
    # Step 4: Extract features for new clients
    concepts_df = pd.read_csv("data/LVMH_Sales_Database.csv")
    new_features = extract_features(outcomes_df['client_id'].tolist(), concepts_df)
    new_labels = outcomes_df['purchased_actual'].values
    
    # Step 5: Incremental update
    print("\n3️⃣ Performing incremental update...")
    updated_model, metrics = trainer.incremental_update(
        model=old_model,
        new_features=new_features,
        new_labels=new_labels,
        model_name='purchase'
    )
    
    # Step 6: Save new version
    print("\n4️⃣ Saving updated model...")
    trainer.save_model_version(
        model=updated_model,
        model_name='purchase',
        metrics=metrics,
        version=f"incremental_{datetime.now().strftime('%Y%m%d')}"
    )
    
    print("\n✅ Incremental update complete!")
    print(f"   • New training examples: {len(new_labels)}")
    print(f"   • Training time: {metrics['train_time']:.2f}s")
    print(f"   • Accuracy on new data: {metrics.get('accuracy', 0):.2%}")


def workflow_full_retrain():
    """
    Workflow 2: Weekly full retrain with all accumulated data.
    """
    print("\n" + "="*80)
    print("WORKFLOW 2: FULL RETRAIN (Weekly)")
    print("="*80)
    
    trainer = ContinuousTrainer()
    
    # Step 1: Collect ALL labeled data (7 days)
    print("\n1️⃣ Collecting all labeled data from past 7 days...")
    
    all_outcomes = []
    for days_ago in range(1, 8):
        outcomes = create_synthetic_outcomes(days_ago=days_ago)
        all_outcomes.append(outcomes)
        
    all_outcomes_df = pd.concat(all_outcomes, ignore_index=True)
    print(f"\n   ✓ Total labeled examples: {len(all_outcomes_df)}")
    
    # Step 2: Extract features for ALL clients
    concepts_df = pd.read_csv("data/LVMH_Sales_Database.csv")
    all_features = extract_features(all_outcomes_df['client_id'].tolist(), concepts_df)
    
    # Step 3: Full retrain for each model
    results = {}
    
    for model_type, label_col in [
        ('purchase', 'purchased_actual'),
        ('churn', 'churned_actual'),
        ('clv', 'clv_actual')
    ]:
        print(f"\n2️⃣ Full retrain: {model_type} model...")
        
        all_labels = all_outcomes_df[label_col].values
        
        new_model, metrics = trainer.full_retrain(
            all_features=all_features,
            all_labels=all_labels,
            model_type=model_type
        )
        
        # Save new version
        model_path = trainer.save_model_version(
            model=new_model,
            model_name=model_type,
            metrics=metrics,
            version=f"weekly_{datetime.now().strftime('%Y%m%d')}"
        )
        
        results[model_type] = {
            'model_path': model_path,
            'metrics': metrics
        }
        
    print("\n✅ Full retrain complete!")
    for model_type, result in results.items():
        metrics = result['metrics']
        if 'accuracy' in metrics:
            print(f"   • {model_type}: {metrics['accuracy']:.2%} accuracy")
        else:
            print(f"   • {model_type}: R² {metrics['r2_score']:.3f}")


def workflow_ab_testing():
    """
    Workflow 3: A/B test new model vs production model.
    """
    print("\n" + "="*80)
    print("WORKFLOW 3: A/B TESTING")
    print("="*80)
    
    trainer = ContinuousTrainer()
    
    # Step 1: Get test data
    print("\n1️⃣ Creating test dataset...")
    test_outcomes = create_synthetic_outcomes(days_ago=0)  # Today's data
    
    concepts_df = pd.read_csv("data/LVMH_Sales_Database.csv")
    test_features = extract_features(test_outcomes['client_id'].tolist(), concepts_df)
    test_labels = test_outcomes['purchased_actual'].values
    
    # Step 2: Find models to compare
    print("\n2️⃣ Finding models to compare...")
    
    old_model_path = Path("models/elite_predictive/purchase_model.pkl")
    new_model_path = trainer.get_best_model('purchase')
    
    if not old_model_path.exists():
        print(f"   ❌ Old model not found: {old_model_path}")
        return
        
    if not new_model_path or not new_model_path.exists():
        print("   ❌ No new model found in continuous training history")
        print("   Run workflow_incremental_update() or workflow_full_retrain() first")
        return
        
    print(f"   Old model: {old_model_path}")
    print(f"   New model: {new_model_path}")
    
    # Step 3: Compare
    print("\n3️⃣ Comparing models...")
    comparison = trainer.compare_models(
        old_model_path=old_model_path,
        new_model_path=new_model_path,
        test_features=test_features,
        test_labels=test_labels,
        model_type='purchase'
    )
    
    # Step 4: Decision
    print("\n4️⃣ Deployment decision...")
    if comparison['better']:
        print(f"   ✅ New model is BETTER ({comparison['improvement_pct']:+.1f}%)")
        print("   Recommendation: Deploy to production")
        
        # In production, you would:
        # shutil.copy(new_model_path, 'models/production/purchase_model.pkl')
        print("\n   To deploy:")
        print(f"   cp {new_model_path} models/production/purchase_model.pkl")
    else:
        print(f"   ❌ New model is WORSE ({comparison['improvement_pct']:+.1f}%)")
        print("   Recommendation: Keep current production model")
        
    print("\n✅ A/B testing complete!")


def main():
    """Main demo of all workflows."""
    print("\n🚀 CONTINUOUS TRAINING WORKFLOWS\n")
    
    print("Available workflows:")
    print("  1 - Incremental Update (Daily, Fast)")
    print("  2 - Full Retrain (Weekly, Accurate)")
    print("  3 - A/B Testing (Compare & Deploy)")
    print("  4 - Run All")
    
    choice = input("\nSelect workflow (1-4): ").strip()
    
    if choice == '1':
        workflow_incremental_update()
    elif choice == '2':
        workflow_full_retrain()
    elif choice == '3':
        workflow_ab_testing()
    elif choice == '4':
        workflow_incremental_update()
        workflow_full_retrain()
        workflow_ab_testing()
    else:
        print("❌ Invalid choice")
        sys.exit(1)


if __name__ == "__main__":
    main()
