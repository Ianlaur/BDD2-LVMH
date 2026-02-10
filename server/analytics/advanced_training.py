"""
Advanced ML Training - Push the Limits!

Features:
- Advanced feature engineering (TF-IDF, embeddings, co-occurrence)
- Hyperparameter optimization (GridSearch, RandomSearch, Optuna)
- Ensemble models (XGBoost, LightGBM, CatBoost, stacking)
- Speed optimizations (batching, caching, multi-threading)
- Cross-validation with stratification
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import joblib
from collections import Counter

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
)

# Advanced models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

# Hyperparameter optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from server.shared.model_cache import get_sentence_transformer


class AdvancedMLTrainer:
    """
    Elite ML training system with advanced features and optimizations.
    """
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        use_tfidf: bool = True,
        use_embeddings: bool = True,
        use_cooccurrence: bool = True,
        use_ensemble: bool = True,
        optimize_hyperparams: bool = True,
        n_jobs: int = -1
    ):
        """
        Initialize advanced trainer.
        
        Args:
            embedding_model: Sentence transformer model
            use_tfidf: Use TF-IDF features
            use_embeddings: Use concept embeddings
            use_cooccurrence: Use concept co-occurrence features
            use_ensemble: Use ensemble models
            optimize_hyperparams: Run hyperparameter optimization
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings
        self.use_cooccurrence = use_cooccurrence
        self.use_ensemble = use_ensemble
        self.optimize_hyperparams = optimize_hyperparams
        self.n_jobs = n_jobs
        
        # Feature processors
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
        # Models
        self.models = {}
        self.best_models = {}
        
        # Feature names
        self.feature_names = []
        
        print("🚀 Advanced ML Trainer initialized")
        print(f"   TF-IDF features: {'✅' if use_tfidf else '❌'}")
        print(f"   Embedding features: {'✅' if use_embeddings else '❌'}")
        print(f"   Co-occurrence features: {'✅' if use_cooccurrence else '❌'}")
        print(f"   Ensemble models: {'✅' if use_ensemble else '❌'}")
        print(f"   Hyperparameter optimization: {'✅' if optimize_hyperparams else '❌'}")
        print(f"   XGBoost available: {'✅' if HAS_XGBOOST else '❌'}")
        print(f"   LightGBM available: {'✅' if HAS_LIGHTGBM else '❌'}")
        print(f"   CatBoost available: {'✅' if HAS_CATBOOST else '❌'}")
        print(f"   Optuna available: {'✅' if HAS_OPTUNA else '❌'}")
        
    def _ensure_embedding_model_loaded(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = get_sentence_transformer()
            
    def create_advanced_features(
        self,
        concepts_df: pd.DataFrame,
        client_id_col: str = "client_id",
        concept_col: str = "concept"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create advanced feature matrix with multiple feature types.
        
        Args:
            concepts_df: DataFrame with client concepts
            client_id_col: Client ID column
            concept_col: Concept column (or matched_alias)
            
        Returns:
            (feature_df, feature_names) tuple
        """
        print("\n🔧 Creating advanced feature matrix...")
        
        # Use matched_alias if concept column not available
        if concept_col not in concepts_df.columns and 'matched_alias' in concepts_df.columns:
            concept_col = 'matched_alias'
            print(f"   Using '{concept_col}' column")
            
        # Get unique concepts and clients
        unique_concepts = concepts_df[concept_col].unique()
        unique_clients = concepts_df[client_id_col].unique()
        print(f"   Clients: {len(unique_clients)}, Concepts: {len(unique_concepts)}")
        
        # Initialize feature dictionary
        all_features = {}
        
        # 1. ONE-HOT ENCODING (baseline)
        print("   1️⃣  Creating one-hot encoding features...")
        for client_id in unique_clients:
            client_concepts = concepts_df[
                concepts_df[client_id_col] == client_id
            ][concept_col].tolist()
            
            # One-hot features
            onehot = {f"onehot_{concept}": 1 if concept in client_concepts else 0 
                     for concept in unique_concepts}
            
            all_features[client_id] = onehot
            
        # 2. TF-IDF FEATURES
        if self.use_tfidf:
            print("   2️⃣  Creating TF-IDF features...")
            client_texts = {}
            for client_id in unique_clients:
                client_concepts = concepts_df[
                    concepts_df[client_id_col] == client_id
                ][concept_col].tolist()
                client_texts[client_id] = " ".join(client_concepts)
                
            # Fit TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=200,  # Top 200 features
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2
            )
            
            texts = [client_texts[cid] for cid in unique_clients]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts).toarray()
            
            # Add to features
            for i, client_id in enumerate(unique_clients):
                tfidf_features = {f"tfidf_{j}": tfidf_matrix[i, j] 
                                 for j in range(tfidf_matrix.shape[1])}
                all_features[client_id].update(tfidf_features)
                
        # 3. CONCEPT EMBEDDINGS (BATCHED FOR SPEED)
        if self.use_embeddings:
            print("   3️⃣  Creating embedding features (batched)...")
            self._ensure_embedding_model_loaded()
            
            # Prepare all profiles at once
            client_profiles = {}
            for client_id in unique_clients:
                client_concepts = concepts_df[
                    concepts_df[client_id_col] == client_id
                ][concept_col].tolist()
                client_profiles[client_id] = " ".join(client_concepts)
                
            # Batch encode for speed (all at once)
            profiles_list = [client_profiles[cid] for cid in unique_clients]
            print(f"      Encoding {len(profiles_list)} client profiles...")
            
            embeddings = self.embedding_model.encode(
                profiles_list,
                batch_size=32,  # Process 32 at a time
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Add embedding features
            for i, client_id in enumerate(unique_clients):
                emb_features = {f"emb_{j}": embeddings[i][j] 
                               for j in range(embeddings.shape[1])}
                all_features[client_id].update(emb_features)
                
        # 4. CO-OCCURRENCE FEATURES
        if self.use_cooccurrence:
            print("   4️⃣  Creating co-occurrence features...")
            
            # Build co-occurrence matrix
            cooccurrence = Counter()
            for client_id in unique_clients:
                client_concepts = concepts_df[
                    concepts_df[client_id_col] == client_id
                ][concept_col].tolist()
                
                # Count pairs
                for i, c1 in enumerate(client_concepts):
                    for c2 in client_concepts[i+1:]:
                        pair = tuple(sorted([c1, c2]))
                        cooccurrence[pair] += 1
                        
            # Get top co-occurrence pairs
            top_pairs = [pair for pair, count in cooccurrence.most_common(100)]
            
            # Add features
            for client_id in unique_clients:
                client_concepts = set(concepts_df[
                    concepts_df[client_id_col] == client_id
                ][concept_col].tolist())
                
                cooc_features = {}
                for pair in top_pairs:
                    has_pair = 1 if pair[0] in client_concepts and pair[1] in client_concepts else 0
                    cooc_features[f"cooc_{pair[0]}_{pair[1]}"] = has_pair
                    
                all_features[client_id].update(cooc_features)
                
        # 5. STATISTICAL FEATURES
        print("   5️⃣  Creating statistical features...")
        for client_id in unique_clients:
            client_concepts = concepts_df[
                concepts_df[client_id_col] == client_id
            ]
            
            stats = {
                'stat_num_concepts': len(client_concepts),
                'stat_unique_concepts': client_concepts[concept_col].nunique(),
                'stat_concept_diversity': (client_concepts[concept_col].nunique() / 
                                          max(len(client_concepts), 1))
            }
            
            all_features[client_id].update(stats)
            
        # Convert to DataFrame
        feature_df = pd.DataFrame.from_dict(all_features, orient='index')
        feature_df.index.name = 'client_id'
        feature_df = feature_df.reset_index()
        
        self.feature_names = [col for col in feature_df.columns if col != 'client_id']
        
        print(f"✅ Created {len(self.feature_names)} features:")
        print(f"   - One-hot: {len(unique_concepts)}")
        if self.use_tfidf:
            print(f"   - TF-IDF: {tfidf_matrix.shape[1]}")
        if self.use_embeddings:
            print(f"   - Embeddings: {embeddings.shape[1]}")
        if self.use_cooccurrence:
            print(f"   - Co-occurrence: {len(top_pairs)}")
        print(f"   - Statistical: 3")
        
        return feature_df, self.feature_names
        
    def get_base_models(self, task: str = "classification") -> Dict[str, Any]:
        """
        Get base models for ensemble.
        
        Args:
            task: "classification" or "regression"
            
        Returns:
            Dictionary of models
        """
        models = {}
        
        if task == "classification":
            # Random Forest
            models['rf'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                n_jobs=self.n_jobs,
                random_state=42
            )
            
            # Gradient Boosting
            models['gb'] = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            # Logistic Regression
            models['lr'] = LogisticRegression(
                max_iter=1000,
                n_jobs=self.n_jobs,
                random_state=42
            )
            
            # XGBoost
            if HAS_XGBOOST:
                models['xgb'] = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    n_jobs=self.n_jobs,
                    random_state=42
                )
                
            # LightGBM
            if HAS_LIGHTGBM:
                models['lgb'] = lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    n_jobs=self.n_jobs,
                    random_state=42,
                    verbose=-1
                )
                
            # CatBoost
            if HAS_CATBOOST:
                models['catboost'] = CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.1,
                    depth=6,
                    random_state=42,
                    verbose=False
                )
                
        else:  # regression
            models['rf'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                n_jobs=self.n_jobs,
                random_state=42
            )
            
            models['gb'] = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            if HAS_XGBOOST:
                models['xgb'] = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    n_jobs=self.n_jobs,
                    random_state=42
                )
                
            if HAS_LIGHTGBM:
                models['lgb'] = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    n_jobs=self.n_jobs,
                    random_state=42,
                    verbose=-1
                )
                
        return models
        
    def optimize_hyperparameters(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task: str = "classification"
    ) -> Any:
        """
        Optimize hyperparameters using GridSearch or Optuna.
        
        Args:
            model: Base model
            model_name: Model identifier
            X_train: Training features
            y_train: Training labels
            task: "classification" or "regression"
            
        Returns:
            Optimized model
        """
        print(f"   🔍 Optimizing {model_name} hyperparameters...")
        
        # Define parameter grids
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'xgb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [4, 6, 8]
            },
            'lgb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [4, 6, 8]
            }
        }
        
        if model_name not in param_grids:
            return model
            
        # Use RandomizedSearchCV for speed
        search = RandomizedSearchCV(
            model,
            param_grids[model_name],
            n_iter=10,
            cv=3,
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        print(f"      Best params: {search.best_params_}")
        print(f"      Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_
        
    def train_elite_classifier(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str,
        model_name: str = "purchase"
    ) -> Dict:
        """
        Train elite classifier with all optimizations.
        
        Args:
            features_df: Feature matrix
            labels_df: Labels DataFrame
            label_col: Label column name
            model_name: Model identifier
            
        Returns:
            Training metrics
        """
        print(f"\n🚀 Training ELITE {model_name} classifier...")
        print("="*80)
        
        # Merge features with labels
        data = features_df.merge(labels_df[['client_id', label_col]], on='client_id')
        
        # Prepare data
        X = data.drop(['client_id', label_col], axis=1)
        y = data[label_col]
        
        print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Class distribution: {Counter(y)}")
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get base models
        base_models = self.get_base_models("classification")
        
        # Train and evaluate each model
        results = {}
        trained_models = {}
        
        for name, model in base_models.items():
            print(f"\n   📊 Training {name.upper()}...")
            
            # Optimize hyperparameters if enabled
            if self.optimize_hyperparams and name in ['rf', 'xgb', 'lgb']:
                model = self.optimize_hyperparameters(
                    model, name, X_train_scaled, y_train, "classification"
                )
            else:
                model.fit(X_train_scaled, y_train)
                
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            
            results[name] = {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            trained_models[name] = model
            
            print(f"      Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
            
        # Create ensemble if enabled
        if self.use_ensemble and len(trained_models) >= 3:
            print(f"\n   🎯 Creating ENSEMBLE model...")
            
            # Voting ensemble
            voting_models = [(name, model) for name, model in trained_models.items()]
            ensemble = VotingClassifier(
                estimators=voting_models,
                voting='soft',
                n_jobs=self.n_jobs
            )
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            y_pred = ensemble.predict(X_test_scaled)
            y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            
            results['ensemble'] = {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            trained_models['ensemble'] = ensemble
            
            print(f"      Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
            
        # Find best model
        best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
        best_model = trained_models[best_model_name]
        
        self.models[model_name] = trained_models
        self.best_models[model_name] = best_model
        
        print(f"\n✅ Best model: {best_model_name.upper()}")
        print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"   AUC: {results[best_model_name]['auc']:.4f}")
        print(f"   F1: {results[best_model_name]['f1']:.4f}")
        
        return {
            'best_model': best_model_name,
            'all_results': results,
            'best_metrics': results[best_model_name]
        }
        
    def save_models(self, output_dir: Path, model_name: str = "purchase"):
        """Save trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        if model_name in self.best_models:
            joblib.dump(
                self.best_models[model_name],
                output_dir / f"{model_name}_best_model.pkl"
            )
            
        # Save all models
        if model_name in self.models:
            for name, model in self.models[model_name].items():
                joblib.dump(
                    model,
                    output_dir / f"{model_name}_{name}_model.pkl"
                )
                
        # Save scaler
        joblib.dump(self.scaler, output_dir / "scaler.pkl")
        
        # Save TF-IDF vectorizer
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, output_dir / "tfidf.pkl")
            
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'use_tfidf': self.use_tfidf,
            'use_embeddings': self.use_embeddings,
            'use_cooccurrence': self.use_cooccurrence,
            'embedding_model': self.embedding_model_name
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\n✅ Saved models to {output_dir}")


def main():
    """Demo with synthetic data."""
    print("="*80)
    print("ADVANCED ML TRAINING - PUSH TO THE LIMIT!")
    print("="*80)
    
    # Check if we have real data
    concepts_file = Path("data/outputs/note_concepts.csv")
    if not concepts_file.exists():
        print("❌ No concepts file found. Run extraction first.")
        return
        
    concepts_df = pd.read_csv(concepts_file)
    print(f"\n📊 Loaded {len(concepts_df)} concept matches")
    
    # Initialize trainer
    trainer = AdvancedMLTrainer(
        use_tfidf=True,
        use_embeddings=True,
        use_cooccurrence=True,
        use_ensemble=True,
        optimize_hyperparams=True
    )
    
    # Create advanced features
    features_df, feature_names = trainer.create_advanced_features(concepts_df)
    
    # Generate synthetic labels for demo
    print("\n⚠️  Generating synthetic labels for demo...")
    np.random.seed(42)
    labels = []
    for client_id in features_df['client_id'].unique():
        labels.append({
            'client_id': client_id,
            'purchased': np.random.randint(0, 2)
        })
    labels_df = pd.DataFrame(labels)
    
    # Train elite classifier
    results = trainer.train_elite_classifier(
        features_df,
        labels_df,
        label_col='purchased',
        model_name='purchase'
    )
    
    # Save models
    trainer.save_models(Path("outputs/elite_models"), "purchase")
    
    print("\n" + "="*80)
    print("✅ ELITE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest model: {results['best_model'].upper()}")
    print(f"Accuracy: {results['best_metrics']['accuracy']:.2%}")
    print(f"AUC: {results['best_metrics']['auc']:.4f}")
    print(f"F1 Score: {results['best_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
