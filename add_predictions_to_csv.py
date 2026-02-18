"""
Add ML Predictions to Existing CSV

Standalone script to add purchase/churn/CLV predictions to any CSV file.
Can be used independently of the main pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer


def load_best_models(models_dir: Path = Path("models")):
    """Load the best available trained models."""
    
    models = {}
    model_dir = None
    
    # Priority: continuous > elite > fast
    for subdir in ['continuous', 'elite_predictive', 'fast_predictive']:
        potential_dir = models_dir / subdir
        
        if subdir == 'continuous':
            # Find best version
            if potential_dir.exists():
                versions = sorted([d for d in potential_dir.iterdir() if d.is_dir() and d.name.startswith('v_')])
                if versions:
                    model_dir = versions[-1]  # Latest version
        else:
            if potential_dir.exists():
                model_dir = potential_dir
                
        if model_dir:
            break
            
    if not model_dir:
        print("❌ No trained models found!")
        print("   Train models with: python -m server.analytics.elite_trainer")
        return None
        
    print(f"📦 Loading models from: {model_dir}")
    
    # Load models
    for model_name in ['purchase', 'churn', 'clv']:
        model_path = model_dir / f"{model_name}_model.pkl"
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
            print(f"   ✓ Loaded {model_name} model")
        else:
            print(f"   ⚠️  {model_name} model not found")
            
    return models if models else None


def extract_features_from_text(texts: pd.Series, is_elite: bool = False) -> np.ndarray:
    """
    Extract ML features from text data.
    
    Args:
        texts: Series of text data
        is_elite: Whether using elite models (800 features vs 500)
        
    Returns:
        Feature matrix
    """
    max_features = 800 if is_elite else 500
    
    # TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts.fillna('unknown'))
    
    # Statistical features
    stats_features = []
    for text in texts:
        if pd.isna(text) or text == '':
            text = 'unknown'
            
        words = str(text).split()
        num_words = len(words)
        unique_words = len(set(words))
        repeat_rate = num_words / max(1, unique_words)
        is_active = 1 if num_words > 10 else 0
        is_at_risk = 1 if num_words < 3 else 0
        engagement = num_words * 1000
        
        stats = [num_words, unique_words, repeat_rate, 
                is_active, is_at_risk, engagement]
        
        # Extended stats for elite models
        if is_elite:
            diversity = unique_words / max(1, num_words)
            stats.extend([
                diversity,
                num_words ** 2,
                np.log1p(num_words),
                np.sqrt(num_words),
                1 / max(1, num_words),
                num_words * diversity,
                engagement / max(1, unique_words),
                repeat_rate * engagement
            ])
            
        stats_features.append(stats)
        
    stats_features = np.array(stats_features)
    
    # Combine
    features = np.hstack([tfidf_matrix.toarray(), stats_features])
    
    return features


def add_predictions(input_csv: str, 
                   text_column: str,
                   output_csv: Optional[str] = None,
                   id_column: Optional[str] = None):
    """
    Add ML predictions to a CSV file.
    
    Args:
        input_csv: Path to input CSV
        text_column: Name of text column to analyze
        output_csv: Path to output CSV (default: input_with_predictions.csv)
        id_column: Optional ID column name
    """
    print("="*60)
    print("ADD ML PREDICTIONS TO CSV")
    print("="*60)
    print()
    
    # Load CSV
    print(f"📂 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Verify text column exists
    if text_column not in df.columns:
        print(f"❌ Error: Column '{text_column}' not found!")
        print(f"   Available columns: {', '.join(df.columns)}")
        return
        
    print(f"   ✓ Using text column: '{text_column}'")
    
    # Load models
    models = load_best_models()
    if not models:
        return
        
    # Determine if elite models
    is_elite = 'elite' in str(Path("models/elite_predictive"))
    
    # Extract features
    print(f"\n🔧 Extracting features...")
    features = extract_features_from_text(df[text_column], is_elite=is_elite)
    print(f"   ✓ Feature matrix: {features.shape}")
    
    # Make predictions
    print(f"\n🤖 Generating predictions...")
    
    if 'purchase' in models:
        df['purchase_probability'] = models['purchase'].predict_proba(features)[:, 1]
        df['will_purchase'] = models['purchase'].predict(features)
        print(f"   ✓ Purchase predictions:")
        print(f"      - Avg probability: {df['purchase_probability'].mean():.1%}")
        print(f"      - High intent (>70%): {(df['purchase_probability'] > 0.7).sum()}")
        
    if 'churn' in models:
        df['churn_risk'] = models['churn'].predict_proba(features)[:, 1]
        df['will_churn'] = models['churn'].predict(features)
        print(f"   ✓ Churn predictions:")
        print(f"      - Avg risk: {df['churn_risk'].mean():.1%}")
        print(f"      - At risk (>70%): {(df['churn_risk'] > 0.7).sum()}")
        
    if 'clv' in models:
        df['predicted_clv'] = models['clv'].predict(features)
        print(f"   ✓ CLV predictions:")
        print(f"      - Avg CLV: ${df['predicted_clv'].mean():,.0f}")
        print(f"      - High value (>$10k): {(df['predicted_clv'] > 10000).sum()}")
        
        # Add value segments
        df['value_segment'] = pd.cut(
            df['predicted_clv'],
            bins=[-np.inf, 3000, 7000, 15000, np.inf],
            labels=['Low Value', 'Medium Value', 'High Value', 'VIP']
        )
        
    # Add risk segments
    if 'churn_risk' in df.columns:
        df['risk_segment'] = pd.cut(
            df['churn_risk'],
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=['Safe', 'Monitor', 'At Risk', 'Critical']
        )
        
    # Save output
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_with_predictions.csv"
    else:
        output_csv = Path(output_csv)
        
    print(f"\n💾 Saving results...")
    df.to_csv(output_csv, index=False)
    print(f"   ✓ Saved to: {output_csv}")
    print(f"   ✓ Size: {output_csv.stat().st_size / 1024:.1f} KB")
    
    # Show sample
    print(f"\n📊 Sample predictions:")
    
    display_cols = [text_column]
    if id_column and id_column in df.columns:
        display_cols.insert(0, id_column)
        
    pred_cols = [col for col in df.columns if col in [
        'purchase_probability', 'churn_risk', 'predicted_clv', 
        'value_segment', 'risk_segment'
    ]]
    display_cols.extend(pred_cols)
    
    # Show top 5 by CLV if available
    if 'predicted_clv' in df.columns:
        sample = df.nlargest(5, 'predicted_clv')[display_cols]
    else:
        sample = df.head(5)[display_cols]
        
    print(sample.to_string(index=False, max_colwidth=40))
    
    print("\n" + "="*60)
    print("✅ PREDICTIONS COMPLETE!")
    print("="*60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Add ML predictions to CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add predictions to client data
  python add_predictions_to_csv.py data/clients.csv --text-column "notes"
  
  # Specify output file
  python add_predictions_to_csv.py data/clients.csv --text-column "notes" --output results.csv
  
  # With ID column for better display
  python add_predictions_to_csv.py data/clients.csv --text-column "notes" --id-column "client_id"
        """
    )
    
    parser.add_argument('input_csv', help='Input CSV file')
    parser.add_argument('--text-column', '-t', required=True, 
                       help='Name of text column to analyze')
    parser.add_argument('--output', '-o', help='Output CSV file (optional)')
    parser.add_argument('--id-column', '-i', help='ID column name (optional)')
    
    args = parser.parse_args()
    
    add_predictions(
        input_csv=args.input_csv,
        text_column=args.text_column,
        output_csv=args.output,
        id_column=args.id_column
    )


if __name__ == "__main__":
    main()
