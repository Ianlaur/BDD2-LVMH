"""
Continuous ML Training System

Supports:
1. Incremental training (add new data without full retraining)
2. Scheduled retraining (daily/weekly with all data)
3. Performance monitoring (track accuracy over time)
4. Model versioning (keep history of improvements)
5. A/B testing (compare old vs new models)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import json
from datetime import datetime
import shutil
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


class ContinuousTrainer:
    """
    Manages continuous training and model updates.
    """
    
    def __init__(self, model_dir: Path = Path("models/continuous")):
        """Initialize continuous trainer."""
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.model_dir / "training_history.json"
        self.training_history = self._load_history()
        
    def _load_history(self) -> List[Dict]:
        """Load training history."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
        
    def _save_history(self):
        """Save training history."""
        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
    def collect_new_labels(
        self,
        predictions_file: Path,
        actual_outcomes_file: Path
    ) -> pd.DataFrame:
        """
        Collect new labeled data from predictions vs actual outcomes.
        
        Args:
            predictions_file: CSV with client_id and predictions
            actual_outcomes_file: CSV with client_id and actual outcomes
            
        Returns:
            DataFrame with new labeled examples
        """
        print("\n📊 Collecting new labeled data...")
        
        # Load predictions
        predictions_df = pd.read_csv(predictions_file)
        
        # Load actual outcomes
        outcomes_df = pd.read_csv(actual_outcomes_file)
        
        # Merge to get labeled examples
        labeled_data = predictions_df.merge(
            outcomes_df, 
            on='client_id', 
            suffixes=('_pred', '_actual')
        )
        
        print(f"   Found {len(labeled_data)} new labeled examples")
        
        # Calculate accuracy of predictions
        if 'purchased_actual' in labeled_data.columns and 'purchased_pred' in labeled_data.columns:
            accuracy = (labeled_data['purchased_pred'] == labeled_data['purchased_actual']).mean()
            print(f"   Current model accuracy: {accuracy:.2%}")
            
        return labeled_data
        
    def incremental_update(
        self,
        model: object,
        new_features: np.ndarray,
        new_labels: np.ndarray,
        model_name: str
    ) -> Tuple[object, Dict]:
        """
        Update model with new data (warm start).
        
        Args:
            model: Existing trained model
            new_features: New feature matrix
            new_labels: New labels
            model_name: Name of model (for logging)
            
        Returns:
            (updated_model, metrics) tuple
        """
        print(f"\n🔄 Incrementally updating {model_name}...")
        
        import time
        start = time.time()
        
        # RandomForest supports warm_start
        if hasattr(model, 'warm_start'):
            model.warm_start = True
            model.n_estimators += 50  # Add 50 more trees
            
        # Fit on new data
        model.fit(new_features, new_labels)
        
        train_time = time.time() - start
        
        # Evaluate on new data
        predictions = model.predict(new_features)
        
        if isinstance(model, (RandomForestClassifier,)):
            accuracy = accuracy_score(new_labels, predictions)
            metric_name = "accuracy"
            metric_value = accuracy
            print(f"   ✓ Updated in {train_time:.2f}s (accuracy on new data: {accuracy:.2%})")
        else:
            rmse = np.sqrt(mean_squared_error(new_labels, predictions))
            r2 = r2_score(new_labels, predictions)
            metric_name = "r2_score"
            metric_value = r2
            print(f"   ✓ Updated in {train_time:.2f}s (R² on new data: {r2:.3f})")
            
        return model, {
            "train_time": train_time,
            metric_name: float(metric_value),
            "new_samples": len(new_labels)
        }
        
    def full_retrain(
        self,
        all_features: np.ndarray,
        all_labels: np.ndarray,
        model_type: str = "purchase"
    ) -> Tuple[object, Dict]:
        """
        Full retrain from scratch with all accumulated data.
        
        Args:
            all_features: All features (old + new)
            all_labels: All labels
            model_type: Type of model to train
            
        Returns:
            (new_model, metrics) tuple
        """
        print(f"\n🔄 Full retraining {model_type} model...")
        
        import time
        from sklearn.model_selection import train_test_split
        
        start = time.time()
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, all_labels, test_size=0.2, random_state=42
        )
        
        # Create new model
        if model_type in ["purchase", "churn"]:
            model = RandomForestClassifier(
                n_estimators=250,
                max_depth=20,
                min_samples_split=4,
                n_jobs=-1,
                random_state=42
            )
        else:  # CLV
            model = RandomForestRegressor(
                n_estimators=250,
                max_depth=20,
                min_samples_split=4,
                n_jobs=-1,
                random_state=42
            )
            
        # Train
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        if model_type in ["purchase", "churn"]:
            accuracy = accuracy_score(y_test, y_pred)
            print(f"   ✓ Trained in {train_time:.2f}s (accuracy: {accuracy:.2%})")
            metrics = {"accuracy": float(accuracy)}
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f"   ✓ Trained in {train_time:.2f}s (R²: {r2:.3f}, RMSE: ${rmse:.2f})")
            metrics = {"r2_score": float(r2), "rmse": float(rmse)}
            
        metrics["train_time"] = train_time
        metrics["total_samples"] = len(all_labels)
        
        return model, metrics
        
    def save_model_version(
        self,
        model: object,
        model_name: str,
        metrics: Dict,
        version: Optional[str] = None
    ) -> Path:
        """
        Save model with version tracking.
        
        Args:
            model: Trained model
            model_name: Name (purchase/churn/clv)
            metrics: Performance metrics
            version: Optional version string (default: timestamp)
            
        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create versioned directory
        version_dir = self.model_dir / f"v_{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "metrics": metrics
        }
        
        metadata_path = version_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update training history
        self.training_history.append(metadata)
        self._save_history()
        
        print(f"   ✓ Saved model version: {version}")
        
        return model_path
        
    def compare_models(
        self,
        old_model_path: Path,
        new_model_path: Path,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        model_type: str
    ) -> Dict:
        """
        Compare old vs new model performance.
        
        Args:
            old_model_path: Path to old model
            new_model_path: Path to new model
            test_features: Test features
            test_labels: Test labels
            model_type: Type of model
            
        Returns:
            Comparison metrics
        """
        print(f"\n🔍 Comparing models on test set ({len(test_labels)} examples)...")
        
        # Load models
        old_model = joblib.load(old_model_path)
        new_model = joblib.load(new_model_path)
        
        # Predict
        old_pred = old_model.predict(test_features)
        new_pred = new_model.predict(test_features)
        
        # Compare
        if model_type in ["purchase", "churn"]:
            old_acc = accuracy_score(test_labels, old_pred)
            new_acc = accuracy_score(test_labels, new_pred)
            
            improvement = (new_acc - old_acc) / old_acc * 100
            
            print(f"   Old model accuracy: {old_acc:.2%}")
            print(f"   New model accuracy: {new_acc:.2%}")
            print(f"   Improvement: {improvement:+.1f}%")
            
            return {
                "old_accuracy": float(old_acc),
                "new_accuracy": float(new_acc),
                "improvement_pct": float(improvement),
                "better": new_acc > old_acc
            }
        else:
            old_rmse = np.sqrt(mean_squared_error(test_labels, old_pred))
            new_rmse = np.sqrt(mean_squared_error(test_labels, new_pred))
            
            old_r2 = r2_score(test_labels, old_pred)
            new_r2 = r2_score(test_labels, new_pred)
            
            improvement = (new_r2 - old_r2) / abs(old_r2) * 100 if old_r2 != 0 else 0
            
            print(f"   Old model R²: {old_r2:.3f}, RMSE: ${old_rmse:.2f}")
            print(f"   New model R²: {new_r2:.3f}, RMSE: ${new_rmse:.2f}")
            print(f"   Improvement: {improvement:+.1f}%")
            
            return {
                "old_r2": float(old_r2),
                "new_r2": float(new_r2),
                "old_rmse": float(old_rmse),
                "new_rmse": float(new_rmse),
                "improvement_pct": float(improvement),
                "better": new_r2 > old_r2
            }
            
    def get_best_model(self, model_name: str) -> Optional[Path]:
        """
        Get path to best performing model version.
        
        Args:
            model_name: Name of model (purchase/churn/clv)
            
        Returns:
            Path to best model or None
        """
        # Filter history for this model
        model_history = [
            h for h in self.training_history 
            if h.get("model_name") == model_name
        ]
        
        if not model_history:
            return None
            
        # Sort by appropriate metric
        if model_name in ["purchase", "churn"]:
            best = max(model_history, key=lambda x: x["metrics"].get("accuracy", 0))
        else:
            best = max(model_history, key=lambda x: x["metrics"].get("r2_score", 0))
            
        version = best["version"]
        return self.model_dir / f"v_{version}" / f"{model_name}_model.pkl"
        
    def plot_training_history(self, model_name: str):
        """
        Plot training history over time.
        
        Args:
            model_name: Name of model to plot
        """
        import matplotlib.pyplot as plt
        
        # Filter history
        model_history = [
            h for h in self.training_history 
            if h.get("model_name") == model_name
        ]
        
        if not model_history:
            print(f"No training history for {model_name}")
            return
            
        # Extract data
        timestamps = [h["timestamp"] for h in model_history]
        
        if model_name in ["purchase", "churn"]:
            values = [h["metrics"].get("accuracy", 0) for h in model_history]
            metric_name = "Accuracy"
        else:
            values = [h["metrics"].get("r2_score", 0) for h in model_history]
            metric_name = "R² Score"
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(values)), values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Training Version')
        plt.ylabel(metric_name)
        plt.title(f'{model_name.title()} Model Performance Over Time')
        plt.grid(True, alpha=0.3)
        
        # Save
        plot_path = self.model_dir / f"{model_name}_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot to {plot_path}")


def demo_continuous_training():
    """
    Demo continuous training workflow.
    """
    print("="*80)
    print("CONTINUOUS ML TRAINING DEMO")
    print("="*80)
    
    print("\n📚 Continuous Training Strategies:\n")
    
    print("1. 🔄 INCREMENTAL UPDATE (Warm Start)")
    print("   • Add new data to existing model")
    print("   • Fast: Only train on new examples")
    print("   • Use: Daily updates with small batches")
    print("   • Example: New 100 clients labeled today")
    
    print("\n2. 🔄 FULL RETRAIN (From Scratch)")
    print("   • Train on ALL data (old + new)")
    print("   • Slower but more accurate")
    print("   • Use: Weekly/monthly with accumulated data")
    print("   • Example: Retrain on all 5000 clients")
    
    print("\n3. 📊 A/B TESTING")
    print("   • Keep old model in production")
    print("   • Test new model on sample traffic")
    print("   • Compare performance metrics")
    print("   • Deploy if better")
    
    print("\n4. 📈 PERFORMANCE MONITORING")
    print("   • Track accuracy over time")
    print("   • Detect model degradation")
    print("   • Auto-trigger retraining")
    print("   • Version control all models")
    
    print("\n" + "="*80)
    print("EXAMPLE WORKFLOW")
    print("="*80)
    
    print("\n📅 Daily (Incremental):")
    print("   9:00 AM - Collect yesterday's predictions")
    print("   9:05 AM - Match with actual outcomes")
    print("   9:10 AM - Incremental update (+50 trees)")
    print("   9:15 AM - Deploy if accuracy improved")
    
    print("\n📅 Weekly (Full Retrain):")
    print("   Sunday 2 AM - Collect all data (7 days)")
    print("   Sunday 2:05 AM - Full retrain from scratch")
    print("   Sunday 2:20 AM - Compare vs production model")
    print("   Sunday 2:25 AM - Deploy new version")
    
    print("\n📅 Monthly (Review):")
    print("   - Plot training history")
    print("   - Analyze error patterns")
    print("   - Add new features if needed")
    print("   - Tune hyperparameters")
    
    # Show example code
    print("\n" + "="*80)
    print("EXAMPLE CODE")
    print("="*80)
    
    print("""
# 1. Load existing model
from server.analytics.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

# 2. Collect new labeled data (daily)
new_data = trainer.collect_new_labels(
    predictions_file='outputs/predictions_2026-02-09.csv',
    actual_outcomes_file='data/outcomes_2026-02-09.csv'
)

# 3. Incremental update (fast)
old_model = joblib.load('models/continuous/v_latest/purchase_model.pkl')
new_model, metrics = trainer.incremental_update(
    model=old_model,
    new_features=new_features,
    new_labels=new_data['purchased_actual'],
    model_name='purchase'
)

# 4. Save new version
trainer.save_model_version(
    model=new_model,
    model_name='purchase',
    metrics=metrics
)

# 5. Compare performance
comparison = trainer.compare_models(
    old_model_path='models/continuous/v_20260209/purchase_model.pkl',
    new_model_path='models/continuous/v_20260210/purchase_model.pkl',
    test_features=test_X,
    test_labels=test_y,
    model_type='purchase'
)

# 6. Deploy if better
if comparison['better']:
    shutil.copy(new_model_path, 'models/production/purchase_model.pkl')
    print("✅ Deployed new model to production!")
""")
    
    print("\n✅ Continuous training system ready!")


if __name__ == "__main__":
    demo_continuous_training()
