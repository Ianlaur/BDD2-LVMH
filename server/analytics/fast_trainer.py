"""
Fast & Accurate ML Training - Optimized for Speed

Uses only scikit-learn with smart optimizations:
- Efficient feature engineering (sparse matrices)
- Fast models (RandomForest with n_jobs=-1)
- Batched processing
- Cached embeddings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime
from scipy.sparse import hstack, csr_matrix


class FastMLTrainer:
    """
    Fast ML trainer optimized for speed and accuracy.
    """
    
    def __init__(self):
        """Initialize trainer."""
        self.purchase_model = None
        self.churn_model = None
        self.clv_model = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_fast_features(
        self,
        concepts_df: pd.DataFrame,
        client_id_col: str = "client_id"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features FAST using sparse matrices.
        
        Args:
            concepts_df: DataFrame with client concepts
            client_id_col: Name of client ID column
            
        Returns:
            (feature_df, feature_names) tuple
        """
        print("\n🚀 Creating features (FAST mode)...")
        
        # Use matched_alias if available
        concept_col = 'matched_alias' if 'matched_alias' in concepts_df.columns else 'concept'
        
        # Group concepts by client
        client_concepts = concepts_df.groupby(client_id_col)[concept_col].apply(list).to_dict()
        client_ids = list(client_concepts.keys())
        
        # 1. TF-IDF features (sparse, very fast)
        print("   Creating TF-IDF features...")
        concept_texts = [" ".join(concepts) for concepts in client_concepts.values()]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,  # Limit features for speed
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(concept_texts)
        print(f"   ✓ TF-IDF: {tfidf_features.shape[1]} features")
        
        # 2. Statistical features (fast numpy operations)
        print("   Creating statistical features...")
        stats_features = []
        for concepts in client_concepts.values():
            stats = [
                len(concepts),  # Total concepts
                len(set(concepts)),  # Unique concepts
                len([c for c in concepts if 'budget' in c.lower()]),  # Budget mentions
                len([c for c in concepts if any(x in c.lower() for x in ['allerg', 'sensiti'])]),  # Allergies
                len([c for c in concepts if any(x in c.lower() for x in ['luxury', 'premium', 'designer'])]),  # Luxury
                len([c for c in concepts if any(x in c.lower() for x in ['gift', 'occasion', 'event'])]),  # Gifting
            ]
            stats_features.append(stats)
            
        stats_array = np.array(stats_features)
        stats_sparse = csr_matrix(stats_array)
        print(f"   ✓ Stats: {stats_sparse.shape[1]} features")
        
        # Combine features (sparse matrix concatenation is FAST)
        print("   Combining features...")
        X_combined = hstack([tfidf_features, stats_sparse])
        
        # Create DataFrame with client IDs
        feature_df = pd.DataFrame(
            X_combined.toarray(),  # Convert to dense for compatibility
            columns=[f'feat_{i}' for i in range(X_combined.shape[1])]
        )
        feature_df.insert(0, 'client_id', client_ids)
        
        feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])] + \
                       ['total_concepts', 'unique_concepts', 'budget_mentions', 
                        'allergy_mentions', 'luxury_mentions', 'gift_mentions']
        
        print(f"✅ Created {X_combined.shape[1]} features for {len(client_ids)} clients")
        return feature_df, feature_names
        
    def train_purchase_predictor(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = "purchased"
    ) -> Dict:
        """
        Train purchase prediction model (FAST).
        
        Args:
            features_df: Feature matrix
            labels_df: Labels DataFrame
            label_col: Label column name
            
        Returns:
            Training metrics
        """
        print("\n🚀 Training purchase prediction model (FAST)...")
        
        # Merge features with labels
        data = features_df.merge(labels_df[['client_id', label_col]], on='client_id')
        
        # Prepare data
        X = data.drop(['client_id', label_col], axis=1)
        y = data[label_col]
        
        self.feature_names = X.columns.tolist()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train with optimized parameters for SPEED
        print("   Training RandomForest with parallel processing...")
        self.purchase_model = RandomForestClassifier(
            n_estimators=200,  # More trees = better accuracy
            max_depth=15,  # Deeper = more complex patterns
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,  # Use ALL CPU cores
            random_state=42,
            verbose=0
        )
        
        import time
        start = time.time()
        self.purchase_model.fit(X_train_scaled, y_train)
        train_time = time.time() - start
        
        # Evaluate
        y_pred = self.purchase_model.predict(X_test_scaled)
        y_pred_proba = self.purchase_model.predict_proba(X_test_scaled)
        
        accuracy = (y_pred == y_test).mean()
        
        print(f"✅ Purchase Model trained in {train_time:.2f}s")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase'])}")
        
        # Feature importance
        importances = self.purchase_model.feature_importances_
        top_features = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("\n📊 Top 10 Predictive Features:")
        for feat, imp in top_features:
            print(f"   - {feat}: {imp:.4f}")
            
        return {
            "accuracy": float(accuracy),
            "train_time": train_time,
            "feature_importance": dict(top_features)
        }
        
    def train_churn_predictor(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = "churned"
    ) -> Dict:
        """Train churn prediction model (FAST)."""
        print("\n🚀 Training churn prediction model (FAST)...")
        
        data = features_df.merge(labels_df[['client_id', label_col]], on='client_id')
        X = data.drop(['client_id', label_col], axis=1)
        y = data[label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        import time
        start = time.time()
        
        self.churn_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        self.churn_model.fit(X_train_scaled, y_train)
        
        train_time = time.time() - start
        
        y_pred = self.churn_model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        print(f"✅ Churn Model trained in {train_time:.2f}s")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Active', 'Churned'])}")
        
        return {"accuracy": float(accuracy), "train_time": train_time}
        
    def train_clv_predictor(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = "lifetime_value"
    ) -> Dict:
        """Train CLV prediction model (FAST)."""
        print("\n🚀 Training CLV prediction model (FAST)...")
        
        data = features_df.merge(labels_df[['client_id', label_col]], on='client_id')
        X = data.drop(['client_id', label_col], axis=1)
        y = data[label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        import time
        start = time.time()
        
        self.clv_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        self.clv_model.fit(X_train_scaled, y_train)
        
        train_time = time.time() - start
        
        y_pred = self.clv_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ CLV Model trained in {train_time:.2f}s")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   R² Score: {r2:.3f}")
        
        return {"rmse": float(rmse), "r2_score": float(r2), "train_time": train_time}
        
    def save_models(self, output_dir: Path):
        """Save trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.purchase_model:
            joblib.dump(self.purchase_model, output_dir / "purchase_model_fast.pkl")
        if self.churn_model:
            joblib.dump(self.churn_model, output_dir / "churn_model_fast.pkl")
        if self.clv_model:
            joblib.dump(self.clv_model, output_dir / "clv_model_fast.pkl")
            
        joblib.dump(self.scaler, output_dir / "scaler_fast.pkl")
        joblib.dump(self.tfidf_vectorizer, output_dir / "tfidf_vectorizer.pkl")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "feature_count": len(self.feature_names),
            "models": ["purchase", "churn", "clv"]
        }
        
        with open(output_dir / "metadata_fast.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\n✅ Saved models to {output_dir}")
        
    def load_models(self, model_dir: Path):
        """Load trained models."""
        if (model_dir / "purchase_model_fast.pkl").exists():
            self.purchase_model = joblib.load(model_dir / "purchase_model_fast.pkl")
        if (model_dir / "churn_model_fast.pkl").exists():
            self.churn_model = joblib.load(model_dir / "churn_model_fast.pkl")
        if (model_dir / "clv_model_fast.pkl").exists():
            self.clv_model = joblib.load(model_dir / "clv_model_fast.pkl")
            
        self.scaler = joblib.load(model_dir / "scaler_fast.pkl")
        self.tfidf_vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
        
        print(f"✅ Loaded models from {model_dir}")


def generate_synthetic_labels(concepts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate better synthetic labels based on concept patterns.
    """
    print("\n🎲 Generating synthetic labels (improved logic)...")
    
    np.random.seed(42)
    
    # Use matched_alias if available
    concept_col = 'matched_alias' if 'matched_alias' in concepts_df.columns else 'concept'
    
    labels = []
    for client_id in concepts_df['client_id'].unique():
        # Get client concepts
        client_concepts = concepts_df[
            concepts_df['client_id'] == client_id
        ][concept_col].tolist()
        
        concept_text = " ".join(client_concepts).lower()
        
        # Purchase likelihood based on concepts
        purchase_prob = 0.3  # Base
        
        # Increase likelihood for luxury indicators
        if any(x in concept_text for x in ['luxury', 'designer', 'premium', 'high-end']):
            purchase_prob += 0.25
        if any(x in concept_text for x in ['gift', 'occasion', 'event', 'birthday', 'anniversary']):
            purchase_prob += 0.2
        if 'budget @0' in concept_text or 'no limit' in concept_text:
            purchase_prob += 0.15
        if any(x in concept_text for x in ['interested', 'looking for', 'wants', 'seeks']):
            purchase_prob += 0.1
            
        # Decrease for hesitation
        if any(x in concept_text for x in ['hesitant', 'unsure', 'maybe', 'thinking']):
            purchase_prob -= 0.15
            
        purchase_prob = np.clip(purchase_prob, 0.1, 0.9)
        purchased = 1 if np.random.random() < purchase_prob else 0
        
        # Churn risk (inverse of engagement)
        churn_prob = 1 - purchase_prob
        churned = 1 if np.random.random() < churn_prob else 0
        
        # CLV (higher for luxury buyers)
        clv = 5000 + np.random.normal(0, 2000)
        if any(x in concept_text for x in ['luxury', 'premium', 'designer']):
            clv += 15000
        if 'budget @0' in concept_text:
            clv += 20000
        clv = max(1000, clv)
        
        labels.append({
            'client_id': client_id,
            'purchased': purchased,
            'churned': churned,
            'lifetime_value': clv
        })
        
    labels_df = pd.DataFrame(labels)
    
    print(f"✅ Generated labels:")
    print(f"   Purchase rate: {labels_df['purchased'].mean():.1%}")
    print(f"   Churn rate: {labels_df['churned'].mean():.1%}")
    print(f"   Avg CLV: ${labels_df['lifetime_value'].mean():.2f}")
    
    return labels_df


def main():
    """Test fast training."""
    print("="*80)
    print("FAST ML TRAINING - Optimized for Speed & Accuracy")
    print("="*80)
    
    # Load concepts
    concepts_file = Path("data/outputs/note_concepts.csv")
    if not concepts_file.exists():
        print(f"❌ Concepts file not found: {concepts_file}")
        return
        
    concepts_df = pd.read_csv(concepts_file)
    
    # Join client_id from notes_clean if missing
    if 'client_id' not in concepts_df.columns and 'note_id' in concepts_df.columns:
        notes_path = Path("data/processed/notes_clean.parquet")
        if notes_path.exists():
            notes_df = pd.read_parquet(notes_path)[['note_id', 'client_id']]
            concepts_df = concepts_df.merge(notes_df, on='note_id', how='left')
            print(f"   Joined client_id from notes_clean.parquet")
    
    print(f"📊 Loaded {len(concepts_df)} concept matches from {len(concepts_df['client_id'].unique())} clients")
    
    # Initialize trainer
    trainer = FastMLTrainer()
    
    # Create features
    import time
    start_time = time.time()
    
    features_df, feature_names = trainer.create_fast_features(concepts_df)
    feature_time = time.time() - start_time
    print(f"\n⏱️  Feature creation: {feature_time:.2f}s")
    
    # Generate synthetic labels
    labels_df = generate_synthetic_labels(concepts_df)
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    purchase_metrics = trainer.train_purchase_predictor(features_df, labels_df)
    churn_metrics = trainer.train_churn_predictor(features_df, labels_df)
    clv_metrics = trainer.train_clv_predictor(features_df, labels_df)
    
    total_train_time = purchase_metrics['train_time'] + churn_metrics['train_time'] + clv_metrics['train_time']
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n⏱️  Total Time:")
    print(f"   Feature creation: {feature_time:.2f}s")
    print(f"   Model training: {total_train_time:.2f}s")
    print(f"   TOTAL: {feature_time + total_train_time:.2f}s")
    
    print(f"\n📊 Model Accuracy:")
    print(f"   Purchase: {purchase_metrics['accuracy']:.2%}")
    print(f"   Churn: {churn_metrics['accuracy']:.2%}")
    print(f"   CLV R²: {clv_metrics['r2_score']:.3f}")
    
    # Save models
    trainer.save_models(Path("models/fast_predictive"))
    
    print("\n✅ Fast training complete!")


if __name__ == "__main__":
    main()
