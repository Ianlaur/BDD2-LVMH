"""
Predictive Analytics - Purchase Likelihood, Churn Risk, CLV

Uses extracted concepts + historical behavior to predict future outcomes.
Example: Clients with [budget@0, family, gift] → 85% buy within 30 days
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import json
from datetime import datetime
import joblib


class PredictiveAnalytics:
    """
    Predict client behavior based on extracted concepts.
    """
    
    def __init__(self):
        """Initialize predictive analytics system."""
        self.purchase_model = None
        self.churn_model = None
        self.clv_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_feature_matrix(
        self, 
        concepts_df: pd.DataFrame,
        client_id_col: str = "client_id",
        concept_col: str = "concept"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create feature matrix from concepts (one-hot encoding).
        
        Args:
            concepts_df: DataFrame with client concepts
            client_id_col: Name of client ID column
            concept_col: Name of concept column (or matched_alias)
            
        Returns:
            (feature_df, feature_names) tuple
        """
        print("\n🔄 Creating feature matrix from concepts...")
        
        # Use matched_alias if concept column not available
        if concept_col not in concepts_df.columns and 'matched_alias' in concepts_df.columns:
            concept_col = 'matched_alias'
            print(f"   Using '{concept_col}' column for concepts")
        
        # Get unique concepts
        unique_concepts = concepts_df[concept_col].unique()
        print(f"   Found {len(unique_concepts)} unique concepts")
        
        # Create binary features (client has concept or not)
        feature_matrix = {}
        
        for client_id in concepts_df[client_id_col].unique():
            client_concepts = concepts_df[
                concepts_df[client_id_col] == client_id
            ][concept_col].tolist()
            
            # Create feature vector
            features = {concept: 1 if concept in client_concepts else 0 
                       for concept in unique_concepts}
            feature_matrix[client_id] = features
            
        # Convert to DataFrame
        feature_df = pd.DataFrame.from_dict(feature_matrix, orient='index')
        feature_df.index.name = 'client_id'
        feature_df = feature_df.reset_index()
        
        feature_names = list(unique_concepts)
        
        print(f"✅ Created feature matrix: {feature_df.shape}")
        return feature_df, feature_names
        
    def train_purchase_predictor(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = "purchased"
    ) -> Dict:
        """
        Train model to predict purchase likelihood.
        
        Args:
            features_df: Feature matrix (client_id + concept features)
            labels_df: DataFrame with client_id and purchase label
            label_col: Name of label column (0/1 for no/yes)
            
        Returns:
            Training metrics
        """
        print("\n🔄 Training purchase prediction model...")
        
        # Merge features with labels
        data = features_df.merge(labels_df[['client_id', label_col]], on='client_id')
        
        # Prepare data
        X = data.drop(['client_id', label_col], axis=1)
        y = data[label_col]
        
        self.feature_names = X.columns.tolist()
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.purchase_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.purchase_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.purchase_model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        print(f"✅ Purchase Prediction Model trained:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase'])}")
        
        # Feature importance
        importances = self.purchase_model.feature_importances_
        feature_importance = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("\n📊 Top 10 Predictive Concepts:")
        for concept, importance in feature_importance:
            print(f"   - {concept}: {importance:.4f}")
            
        return {
            "accuracy": float(accuracy),
            "feature_importance": dict(feature_importance)
        }
        
    def train_churn_predictor(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = "churned"
    ) -> Dict:
        """
        Train model to predict churn risk.
        
        Args:
            features_df: Feature matrix
            labels_df: DataFrame with churn labels
            label_col: Name of label column (0/1 for active/churned)
            
        Returns:
            Training metrics
        """
        print("\n🔄 Training churn prediction model...")
        
        # Merge features with labels
        data = features_df.merge(labels_df[['client_id', label_col]], on='client_id')
        
        # Prepare data
        X = data.drop(['client_id', label_col], axis=1)
        y = data[label_col]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.churn_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        self.churn_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.churn_model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        print(f"✅ Churn Prediction Model trained:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Active', 'Churned'])}")
        
        return {"accuracy": float(accuracy)}
        
    def train_clv_predictor(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = "lifetime_value"
    ) -> Dict:
        """
        Train model to predict customer lifetime value.
        
        Args:
            features_df: Feature matrix
            labels_df: DataFrame with CLV values
            label_col: Name of label column (continuous value)
            
        Returns:
            Training metrics
        """
        print("\n🔄 Training CLV prediction model...")
        
        # Merge features with labels
        data = features_df.merge(labels_df[['client_id', label_col]], on='client_id')
        
        # Prepare data
        X = data.drop(['client_id', label_col], axis=1)
        y = data[label_col]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.clv_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.clv_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.clv_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ CLV Prediction Model trained:")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   R² Score: {r2:.3f}")
        
        return {
            "rmse": float(rmse),
            "r2_score": float(r2)
        }
        
    def predict_purchase_likelihood(
        self, 
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict purchase likelihood for clients.
        
        Args:
            features_df: Feature matrix with client_id
            
        Returns:
            DataFrame with client_id and purchase_probability
        """
        if self.purchase_model is None:
            raise ValueError("Must train purchase model first")
            
        X = features_df.drop('client_id', axis=1)
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probs = self.purchase_model.predict_proba(X_scaled)[:, 1]  # Probability of purchase
        
        results = features_df[['client_id']].copy()
        results['purchase_probability'] = probs
        results['purchase_prediction'] = (probs > 0.5).astype(int)
        
        return results
        
    def predict_churn_risk(
        self, 
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict churn risk for clients.
        
        Args:
            features_df: Feature matrix with client_id
            
        Returns:
            DataFrame with client_id and churn_risk
        """
        if self.churn_model is None:
            raise ValueError("Must train churn model first")
            
        X = features_df.drop('client_id', axis=1)
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probs = self.churn_model.predict_proba(X_scaled)[:, 1]  # Probability of churn
        
        results = features_df[['client_id']].copy()
        results['churn_risk'] = probs
        results['risk_level'] = pd.cut(
            probs, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        return results
        
    def predict_clv(
        self, 
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict customer lifetime value.
        
        Args:
            features_df: Feature matrix with client_id
            
        Returns:
            DataFrame with client_id and predicted_clv
        """
        if self.clv_model is None:
            raise ValueError("Must train CLV model first")
            
        X = features_df.drop('client_id', axis=1)
        X_scaled = self.scaler.transform(X)
        
        # Predict CLV
        clv_predictions = self.clv_model.predict(X_scaled)
        
        results = features_df[['client_id']].copy()
        results['predicted_clv'] = clv_predictions
        results['value_segment'] = pd.qcut(
            clv_predictions,
            q=4,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )
        
        return results
        
    def save_models(self, output_dir: Path):
        """Save trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.purchase_model:
            joblib.dump(self.purchase_model, output_dir / "purchase_model.pkl")
        if self.churn_model:
            joblib.dump(self.churn_model, output_dir / "churn_model.pkl")
        if self.clv_model:
            joblib.dump(self.clv_model, output_dir / "clv_model.pkl")
            
        joblib.dump(self.scaler, output_dir / "scaler.pkl")
        
        # Save feature names
        with open(output_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)
            
        print(f"✅ Saved models to {output_dir}")
        
    def load_models(self, model_dir: Path):
        """Load trained models."""
        if (model_dir / "purchase_model.pkl").exists():
            self.purchase_model = joblib.load(model_dir / "purchase_model.pkl")
        if (model_dir / "churn_model.pkl").exists():
            self.churn_model = joblib.load(model_dir / "churn_model.pkl")
        if (model_dir / "clv_model.pkl").exists():
            self.clv_model = joblib.load(model_dir / "clv_model.pkl")
            
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        
        with open(model_dir / "feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
            
        print(f"✅ Loaded models from {model_dir}")


def main():
    """Test with synthetic data."""
    print("🔬 Predictive Analytics - Demo with Synthetic Data")
    print("="*80)
    
    # Create synthetic training data
    np.random.seed(42)
    n_clients = 1000
    
    # Synthetic concepts
    concepts = ['luxury', 'budget', 'family', 'gift', 'vintage', 'modern', 
                'watches', 'jewelry', 'fashion', 'art']
    
    # Generate synthetic data
    data = []
    for i in range(n_clients):
        client_id = f"CA{i:04d}"
        # Each client has 2-5 random concepts
        client_concepts = np.random.choice(concepts, size=np.random.randint(2, 6), replace=False)
        for concept in client_concepts:
            data.append({'client_id': client_id, 'concept': concept})
            
    concepts_df = pd.DataFrame(data)
    
    # Create synthetic labels
    # Purchase more likely if has 'luxury' or 'gift'
    labels = []
    for client_id in concepts_df['client_id'].unique():
        client_concepts = concepts_df[concepts_df['client_id'] == client_id]['concept'].tolist()
        
        # Purchase likelihood
        base_prob = 0.3
        if 'luxury' in client_concepts:
            base_prob += 0.3
        if 'gift' in client_concepts:
            base_prob += 0.2
        purchased = 1 if np.random.random() < base_prob else 0
        
        # Churn risk (opposite of purchase)
        churned = 1 if np.random.random() < (1 - base_prob) else 0
        
        # CLV (higher if luxury)
        clv = 5000 + np.random.normal(0, 2000)
        if 'luxury' in client_concepts:
            clv += 10000
        clv = max(0, clv)
        
        labels.append({
            'client_id': client_id,
            'purchased': purchased,
            'churned': churned,
            'lifetime_value': clv
        })
        
    labels_df = pd.DataFrame(labels)
    
    print(f"📊 Generated synthetic data:")
    print(f"   - {len(concepts_df)} concept records")
    print(f"   - {n_clients} clients")
    print(f"   - Purchase rate: {labels_df['purchased'].mean():.1%}")
    print(f"   - Churn rate: {labels_df['churned'].mean():.1%}")
    print(f"   - Avg CLV: ${labels_df['lifetime_value'].mean():.2f}")
    
    # Initialize analytics
    analytics = PredictiveAnalytics()
    
    # Create features
    features_df, feature_names = analytics.create_feature_matrix(concepts_df)
    
    # Train models
    analytics.train_purchase_predictor(features_df, labels_df)
    analytics.train_churn_predictor(features_df, labels_df)
    analytics.train_clv_predictor(features_df, labels_df)
    
    # Save models
    analytics.save_models(Path("outputs/predictive_models"))
    
    print("\n✅ Predictive analytics demo complete!")


if __name__ == "__main__":
    main()
