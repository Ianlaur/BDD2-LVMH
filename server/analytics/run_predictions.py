"""
ML Predictions Pipeline Stage

Adds ML predictions (purchase likelihood, churn risk, CLV) to client profiles.
Integrates trained models into the main pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Tuple, Optional
import time

from server.shared.config import DATA_OUTPUTS, MODELS_DIR, BASE_DIR
from server.shared.utils import log_stage


class PipelinePredictor:
    """
    Integrates ML predictions into the pipeline.
    """
    
    def __init__(self):
        """Initialize predictor with best available models."""
        self.models = {}
        self.model_dir = self._find_best_models()
        
        if self.model_dir:
            self._load_models()
        else:
            log_stage("ml_predict", "No trained models found - skipping predictions")
            
    def _find_best_models(self) -> Optional[Path]:
        """
        Find the best trained models to use.
        Priority: continuous > elite > fast > None
        """
        # Check for continuous training models
        continuous_dir = BASE_DIR / "models" / "continuous"
        if continuous_dir.exists():
            try:
                from server.analytics.continuous_trainer import ContinuousTrainer
                trainer = ContinuousTrainer()
                
                # Try to get best purchase model
                best_model = trainer.get_best_model('purchase')
                if best_model and best_model.exists():
                    log_stage("ml_predict", f"Using continuous training models (best version)")
                    return best_model.parent
            except Exception as e:
                log_stage("ml_predict", f"Could not load continuous models: {e}")
        
        # Check for elite models
        elite_dir = BASE_DIR / "models" / "elite_predictive"
        if elite_dir.exists():
            # Check for models with any naming convention
            model_files = list(elite_dir.glob("*model*.pkl"))
            if model_files:
                log_stage("ml_predict", "Using elite trained models")
                return elite_dir
            
        # Check for fast models
        fast_dir = BASE_DIR / "models" / "fast_predictive"
        if fast_dir.exists():
            # Check for models with any naming convention
            model_files = list(fast_dir.glob("*model*.pkl"))
            if model_files:
                log_stage("ml_predict", "Using fast trained models")
                return fast_dir
            
        return None
        
    def _load_models(self):
        """Load trained models from disk."""
        try:
            for model_name in ['purchase', 'churn', 'clv']:
                # Try different naming conventions
                model_paths = [
                    self.model_dir / f"{model_name}_model.pkl",
                    self.model_dir / f"{model_name}_model_fast.pkl",
                    self.model_dir / f"{model_name}_model_elite.pkl",
                ]
                
                loaded = False
                for model_path in model_paths:
                    if model_path.exists():
                        self.models[model_name] = joblib.load(model_path)
                        log_stage("ml_predict", f"Loaded {model_name} model from {model_path.name}")
                        loaded = True
                        break
                        
                if not loaded:
                    log_stage("ml_predict", f"Warning: {model_name} model not found")
                    
        except Exception as e:
            log_stage("ml_predict", f"Error loading models: {e}")
            self.models = {}
            
    def extract_features(self, concepts_df: pd.DataFrame, 
                        client_profiles_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract features matching the training format.
        
        Args:
            concepts_df: Note concepts data
            client_profiles_df: Client profiles data
            
        Returns:
            (clients_df, feature_matrix) tuple
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Get unique clients
        if 'Client ID' in concepts_df.columns:
            client_ids = concepts_df['Client ID'].unique()
        elif 'client_id' in concepts_df.columns:
            client_ids = concepts_df['client_id'].unique()
        else:
            raise ValueError("No client ID column found in concepts data")
            
        # Build client concept strings
        client_concepts = {}
        for client_id in client_ids:
            if 'Client ID' in concepts_df.columns:
                client_data = concepts_df[concepts_df['Client ID'] == client_id]
            else:
                client_data = concepts_df[concepts_df['client_id'] == client_id]
                
            # Get concepts (try different column names)
            for col in ['Matched Alias', 'matched_alias', 'Concept', 'concept']:
                if col in client_data.columns:
                    concepts = client_data[col].dropna().tolist()
                    break
            else:
                concepts = []
                
            client_concepts[client_id] = ' '.join(concepts) if concepts else 'unknown'
            
        # Create TF-IDF features (match training: 500 or 800 features)
        # Use 500 for fast models, 800 for elite models
        max_features = 800 if 'elite' in str(self.model_dir) else 500
        
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(client_concepts.values())
        
        # Add statistical features (matching training format)
        stats_features = []
        for client_id in client_ids:
            if 'Client ID' in concepts_df.columns:
                client_data = concepts_df[concepts_df['Client ID'] == client_id]
            else:
                client_data = concepts_df[concepts_df['client_id'] == client_id]
                
            # Get concepts
            for col in ['Matched Alias', 'matched_alias', 'Concept', 'concept']:
                if col in client_data.columns:
                    concepts = client_data[col].dropna().tolist()
                    break
            else:
                concepts = []
                
            num_concepts = len(concepts)
            unique_concepts = len(set(concepts))
            repeat_rate = num_concepts / max(1, unique_concepts)
            is_active = 1 if num_concepts > 5 else 0
            is_at_risk = 1 if num_concepts < 2 else 0
            engagement = num_concepts * 1000
            
            stats = [num_concepts, unique_concepts, repeat_rate, 
                    is_active, is_at_risk, engagement]
            
            # Add extended stats for elite models
            if 'elite' in str(self.model_dir):
                diversity = unique_concepts / max(1, num_concepts)
                stats.extend([diversity, num_concepts ** 2, 
                            np.log1p(num_concepts), np.sqrt(num_concepts),
                            1 / max(1, num_concepts), num_concepts * diversity,
                            engagement / max(1, unique_concepts), repeat_rate * engagement])
                            
            stats_features.append(stats)
            
        stats_features = np.array(stats_features)
        
        # Combine TF-IDF + stats
        features = np.hstack([tfidf_matrix.toarray(), stats_features])
        
        # Create client dataframe
        clients_df = pd.DataFrame({
            'client_id': list(client_ids)
        })
        
        log_stage("ml_predict", f"Extracted features: {features.shape}")
        
        return clients_df, features
        
    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions using loaded models.
        
        Args:
            features: Feature matrix
            
        Returns:
            Dictionary of predictions
        """
        if not self.models:
            return {}
            
        predictions = {}
        
        # Purchase prediction
        if 'purchase' in self.models:
            model = self.models['purchase']
            predictions['purchase_prob'] = model.predict_proba(features)[:, 1]
            predictions['will_purchase'] = model.predict(features)
            
        # Churn prediction
        if 'churn' in self.models:
            model = self.models['churn']
            predictions['churn_risk'] = model.predict_proba(features)[:, 1]
            predictions['will_churn'] = model.predict(features)
            
        # CLV prediction
        if 'clv' in self.models:
            model = self.models['clv']
            predictions['predicted_clv'] = model.predict(features)
            
        return predictions
        
    def add_predictions_to_profiles(self, client_profiles_df: pd.DataFrame,
                                   predictions_dict: Dict[str, np.ndarray],
                                   clients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add predictions to client profiles.
        
        Args:
            client_profiles_df: Original client profiles
            predictions_dict: Dictionary of predictions
            clients_df: DataFrame with client IDs (matching prediction order)
            
        Returns:
            Enhanced client profiles with predictions
        """
        # Create predictions dataframe
        pred_df = clients_df.copy()
        
        for key, values in predictions_dict.items():
            pred_df[key] = values
            
        # Merge with client profiles
        # Try different ID column names
        id_col = None
        for col in ['Client ID', 'client_id', 'ClientID']:
            if col in client_profiles_df.columns:
                id_col = col
                break
                
        if id_col:
            # Ensure matching column names for merge
            if id_col != 'client_id':
                pred_df = pred_df.rename(columns={'client_id': id_col})
                
            enhanced_df = client_profiles_df.merge(
                pred_df, 
                on=id_col, 
                how='left'
            )
        else:
            # If no ID column, just append (assumes same order)
            for col in pred_df.columns:
                if col != 'client_id':
                    client_profiles_df[col] = pred_df[col].values
            enhanced_df = client_profiles_df
            
        # Add value segments based on predictions
        if 'predicted_clv' in enhanced_df.columns:
            enhanced_df['value_segment'] = pd.cut(
                enhanced_df['predicted_clv'],
                bins=[-np.inf, 3000, 7000, 15000, np.inf],
                labels=['Low Value', 'Medium Value', 'High Value', 'VIP']
            )
            
        # Add risk segments
        if 'churn_risk' in enhanced_df.columns:
            enhanced_df['risk_segment'] = pd.cut(
                enhanced_df['churn_risk'],
                bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
                labels=['Safe', 'Monitor', 'At Risk', 'Critical']
            )
            
        return enhanced_df


def run_ml_predictions():
    """
    Run ML predictions as a pipeline stage.
    """
    print("\n🤖 Running ML Predictions...")
    start_time = time.time()
    
    try:
        # Load data
        concepts_path = DATA_OUTPUTS / "note_concepts.csv"
        profiles_path = DATA_OUTPUTS / "client_profiles.csv"
        
        if not concepts_path.exists():
            log_stage("ml_predict", "⚠️  note_concepts.csv not found - skipping predictions")
            return
            
        if not profiles_path.exists():
            log_stage("ml_predict", "⚠️  client_profiles.csv not found - skipping predictions")
            return
            
        concepts_df = pd.read_csv(concepts_path)
        profiles_df = pd.read_csv(profiles_path)
        
        log_stage("ml_predict", f"Loaded {len(concepts_df)} concept matches")
        log_stage("ml_predict", f"Loaded {len(profiles_df)} client profiles")
        
        # Initialize predictor
        predictor = PipelinePredictor()
        
        if not predictor.models:
            log_stage("ml_predict", "⚠️  No trained models available")
            log_stage("ml_predict", "   Train models with: python -m server.analytics.elite_trainer")
            return
            
        # Extract features
        clients_df, features = predictor.extract_features(concepts_df, profiles_df)
        
        # Make predictions
        predictions = predictor.predict(features)
        
        if predictions:
            log_stage("ml_predict", f"Generated predictions for {len(clients_df)} clients")
            
            # Show summary
            if 'purchase_prob' in predictions:
                avg_prob = predictions['purchase_prob'].mean()
                high_intent = (predictions['purchase_prob'] > 0.7).sum()
                log_stage("ml_predict", f"  • Purchase: {avg_prob:.1%} avg, {high_intent} high intent")
                
            if 'churn_risk' in predictions:
                avg_risk = predictions['churn_risk'].mean()
                high_risk = (predictions['churn_risk'] > 0.7).sum()
                log_stage("ml_predict", f"  • Churn: {avg_risk:.1%} avg risk, {high_risk} at risk")
                
            if 'predicted_clv' in predictions:
                avg_clv = predictions['predicted_clv'].mean()
                high_value = (predictions['predicted_clv'] > 10000).sum()
                log_stage("ml_predict", f"  • CLV: ${avg_clv:,.0f} avg, {high_value} high value")
            
            # Add predictions to profiles
            enhanced_profiles = predictor.add_predictions_to_profiles(
                profiles_df, predictions, clients_df
            )
            
            # Save enhanced profiles
            output_path = DATA_OUTPUTS / "client_profiles_with_predictions.csv"
            enhanced_profiles.to_csv(output_path, index=False)
            
            # Also update the original profiles file
            enhanced_profiles.to_csv(profiles_path, index=False)
            
            log_stage("ml_predict", f"✓ Saved enhanced profiles to {output_path}")
            
            # Generate predictions report
            _generate_predictions_report(enhanced_profiles, predictions)
            
        duration = time.time() - start_time
        log_stage("ml_predict", f"✓ ML predictions complete in {duration:.2f}s")
        
    except Exception as e:
        log_stage("ml_predict", f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def _generate_predictions_report(enhanced_df: pd.DataFrame, 
                                predictions: Dict[str, np.ndarray]):
    """Generate a summary report of predictions."""
    
    report_path = DATA_OUTPUTS / "ml_predictions_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ML PREDICTIONS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total clients: {len(enhanced_df)}\n\n")
        
        # Purchase predictions
        if 'purchase_prob' in enhanced_df.columns:
            f.write("PURCHASE PREDICTIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average probability: {enhanced_df['purchase_prob'].mean():.1%}\n")
            f.write(f"High intent (>70%): {(enhanced_df['purchase_prob'] > 0.7).sum()}\n")
            f.write(f"Medium intent (40-70%): {((enhanced_df['purchase_prob'] > 0.4) & (enhanced_df['purchase_prob'] <= 0.7)).sum()}\n")
            f.write(f"Low intent (<40%): {(enhanced_df['purchase_prob'] <= 0.4).sum()}\n\n")
            
        # Churn predictions
        if 'churn_risk' in enhanced_df.columns:
            f.write("CHURN PREDICTIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average risk: {enhanced_df['churn_risk'].mean():.1%}\n")
            f.write(f"Critical risk (>80%): {(enhanced_df['churn_risk'] > 0.8).sum()}\n")
            f.write(f"At risk (60-80%): {((enhanced_df['churn_risk'] > 0.6) & (enhanced_df['churn_risk'] <= 0.8)).sum()}\n")
            f.write(f"Monitor (30-60%): {((enhanced_df['churn_risk'] > 0.3) & (enhanced_df['churn_risk'] <= 0.6)).sum()}\n")
            f.write(f"Safe (<30%): {(enhanced_df['churn_risk'] <= 0.3).sum()}\n\n")
            
        # CLV predictions
        if 'predicted_clv' in enhanced_df.columns:
            f.write("CLV PREDICTIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average CLV: ${enhanced_df['predicted_clv'].mean():,.2f}\n")
            f.write(f"Median CLV: ${enhanced_df['predicted_clv'].median():,.2f}\n")
            f.write(f"Total potential value: ${enhanced_df['predicted_clv'].sum():,.2f}\n\n")
            
            if 'value_segment' in enhanced_df.columns:
                f.write("Value Segments:\n")
                for segment in ['VIP', 'High Value', 'Medium Value', 'Low Value']:
                    count = (enhanced_df['value_segment'] == segment).sum()
                    if count > 0:
                        avg_clv = enhanced_df[enhanced_df['value_segment'] == segment]['predicted_clv'].mean()
                        f.write(f"  {segment}: {count} clients (avg ${avg_clv:,.0f})\n")
                f.write("\n")
                
        f.write("="*60 + "\n")
        
    log_stage("ml_predict", f"✓ Generated report: {report_path}")


if __name__ == "__main__":
    run_ml_predictions()
