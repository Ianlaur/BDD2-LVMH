# ML Analytics Layer - Post-Extraction Intelligence

The ML analytics layer provides advanced insights **after** concept extraction is complete. This approach keeps extraction fast while delivering high-value business intelligence.

## 🎯 Features

### 1. Client Clustering & Similarity
- **What it does**: Groups similar clients even if they use different words
- **Example**: "loves modern art" ≈ "prefers contemporary pieces"
- **Business Value**: Better segmentation, targeted campaigns
- **Speed**: Fast (batch processing)

### 2. Predictive Analytics
- **Purchase Likelihood**: Predict which clients will buy soon
- **Churn Risk**: Identify at-risk clients before they leave
- **Customer Lifetime Value (CLV)**: Estimate long-term value
- **Business Value**: Prioritize high-value clients, prevent churn
- **Speed**: Fast (inference on aggregated data)

### 3. Recommendation Engine
- **Product Matching**: Match clients to products based on concepts
- **Example**: Client with `[vintage, watches, heritage]` → Patek Philippe
- **Collaborative Filtering**: "Clients like you also bought..."
- **Business Value**: Increase conversion, cross-sell opportunities
- **Speed**: Fast (precomputed recommendations)

## 📊 Architecture

```
CSV Upload → Fast Extraction (9.6s) → Immediate Response ✅
    ↓
Background ML Analysis (async) → Insights ready later 📊
    ├── Clustering: Find similar clients
    ├── Predictions: Forecast behavior
    └── Recommendations: Suggest products
```

## 🚀 Quick Start

### Run All Analytics
```bash
# After extraction completes, run analytics on the concepts
python -m server.analytics.cli all --concepts outputs/concepts.csv
```

### Individual Features

#### Client Clustering
```bash
python -m server.analytics.cli clustering \
  --concepts outputs/concepts.csv \
  --n-clusters 5
```

Output:
- `outputs/clustering_results.json`: Client clusters with descriptions
- Cluster descriptions with top concepts
- Similar client mappings

#### Predictive Analytics
```bash
# With real labels (if you have historical data)
python -m server.analytics.cli predictions \
  --concepts outputs/concepts.csv \
  --labels data/labels.csv

# Demo with synthetic data (for testing)
python -m server.analytics.cli predictions \
  --concepts outputs/concepts.csv
```

Labels CSV format:
```csv
client_id,purchased,churned,lifetime_value
CA001,1,0,25000
CA002,0,1,5000
...
```

Output:
- `outputs/predictive_models/purchase_model.pkl`: Purchase predictor
- `outputs/predictive_models/churn_model.pkl`: Churn predictor
- `outputs/predictive_models/clv_model.pkl`: CLV predictor

#### Recommendations
```bash
# With product catalog
python -m server.analytics.cli recommendations \
  --concepts outputs/concepts.csv \
  --catalog data/products.csv \
  --top-k 5

# Demo with sample LVMH products
python -m server.analytics.cli recommendations \
  --concepts outputs/concepts.csv
```

Product catalog format:
```csv
product_id,name,description,category,price
LV001,Louis Vuitton Neverfull,"Iconic tote bag...",Handbags,1500
TAG001,TAG Heuer Carrera,"Swiss luxury watch...",Watches,5500
...
```

Output:
- `outputs/recommendations.json`: Recommendations for all clients

## 📈 Using the Results

### 1. Clustering Results
```python
import json

# Load clustering results
with open('outputs/clustering_results.json', 'r') as f:
    clusters = json.load(f)

# See which cluster a client belongs to
client_cluster = clusters['clusters']['CA001']
print(f"Client CA001 is in cluster {client_cluster}")
```

### 2. Predictive Models
```python
from server.analytics.predictions import PredictiveAnalytics
import pandas as pd

# Load trained models
analytics = PredictiveAnalytics()
analytics.load_models('outputs/predictive_models')

# Get predictions for new clients
features_df, _ = analytics.create_feature_matrix(concepts_df)

# Purchase likelihood
purchase_pred = analytics.predict_purchase_likelihood(features_df)
print(purchase_pred.head())
# Output: client_id | purchase_probability | purchase_prediction

# Churn risk
churn_pred = analytics.predict_churn_risk(features_df)
print(churn_pred.head())
# Output: client_id | churn_risk | risk_level (Low/Medium/High)

# CLV
clv_pred = analytics.predict_clv(features_df)
print(clv_pred.head())
# Output: client_id | predicted_clv | value_segment (Bronze/Silver/Gold/Platinum)
```

### 3. Recommendations
```python
import json

# Load recommendations
with open('outputs/recommendations.json', 'r') as f:
    recs = json.load(f)

# Get recommendations for a client
client_recs = recs['recommendations']['CA001']
for rec in client_recs:
    print(f"{rec['name']} ({rec['category']}): {rec['similarity_score']:.3f}")
```

## 🔄 Integration with Pipeline

Add analytics as an async step after extraction:

```python
# In server/run_all.py

# After Stage 4 (Concept Detection)
concepts_df = pd.read_csv("outputs/concepts.csv")

# Run analytics in background (non-blocking)
import threading

def run_analytics_async():
    from server.analytics.cli import run_all
    args = argparse.Namespace(
        concepts="outputs/concepts.csv",
        labels=None,
        catalog=None,
        model='all-MiniLM-L6-v2',
        n_clusters=5
    )
    run_all(args)

analytics_thread = threading.Thread(target=run_analytics_async)
analytics_thread.start()

print("✅ Analytics running in background...")
print("   Results will be available in outputs/")
```

## 🎯 Why This Architecture?

✅ **Advantages**:
1. **Fast uploads**: Extraction stays fast (9.6s)
2. **Rich insights**: ML provides business-critical predictions
3. **Scalable**: Analytics run async, don't block pipeline
4. **Better UX**: Users get instant extraction, insights appear later
5. **High ROI**: ML where it matters most (decisions, not tagging)

❌ **Alternative (ML during extraction)**:
- 2x slower (20.5s vs 9.6s)
- Blocks uploads
- Lower business value (minor accuracy gain)
- Poor UX (waiting for uploads)

## 📦 Dependencies

Already in `requirements.txt`:
- `sentence-transformers`: Embeddings for similarity
- `scikit-learn`: Clustering and predictive models
- `pandas`, `numpy`: Data processing

## 🧪 Testing

Test each module individually:

```bash
# Test clustering
python -m server.analytics.clustering

# Test predictions (with synthetic data)
python -m server.analytics.predictions

# Test recommendations (with sample catalog)
python -m server.analytics.recommendations
```

## 📊 Expected Performance

| Feature | Processing Time | Scalability |
|---------|----------------|-------------|
| Clustering | ~5s for 300 clients | O(n) batch |
| Predictions | ~2s for 300 clients | O(n) batch |
| Recommendations | ~10s for 300 clients | O(n×m) batch |

All analytics run **after** extraction, so they don't impact upload time.

## 🎓 Learn More

- [ARCHITECTURE.md](../../ARCHITECTURE.md): Overall system design
- [docs/pipeline.md](../../docs/pipeline.md): Extraction pipeline
- [docs/ML_API_INTEGRATION.md](../../docs/ML_API_INTEGRATION.md): API integration

## 💡 Next Steps

1. **Run analytics on your data**: `python -m server.analytics.cli all --concepts outputs/concepts.csv`
2. **Create product catalog**: Add LVMH product data for recommendations
3. **Collect labels**: Add purchase/churn data to train better predictive models
4. **Integrate with dashboard**: Display insights in UI
5. **Schedule regular runs**: Cron job for daily/weekly analytics updates
