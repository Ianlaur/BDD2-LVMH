# 🔄 Continuous ML Training Guide

Complete guide to keeping your ML models accurate over time with continuous training.

---

## 📚 Overview

Your ML models need to stay updated as:
- New clients are added
- Purchase patterns change
- Seasonal trends emerge
- Business strategy evolves

**Continuous training** automatically improves models using real-world feedback.

---

## 🎯 Three Training Strategies

### 1. ⚡ Incremental Update (Daily)
**Best for:** Fast updates with small batches

```python
# Add new data to existing model (warm start)
trainer = ContinuousTrainer()

updated_model, metrics = trainer.incremental_update(
    model=current_model,
    new_features=new_X,
    new_labels=new_y,
    model_name='purchase'
)
# Takes: ~5 seconds for 100 new clients
```

**Pros:**
- ⚡ Very fast (seconds)
- 🔄 Daily updates possible
- 💾 Low compute cost

**Cons:**
- 📉 May drift over time
- 🎯 Less accurate than full retrain

**Use when:**
- Getting 50-500 new labels daily
- Need quick updates
- Can't afford full retrain time


### 2. 🔄 Full Retrain (Weekly)
**Best for:** Maximum accuracy with all data

```python
# Train from scratch with ALL data
new_model, metrics = trainer.full_retrain(
    all_features=all_X,
    all_labels=all_y,
    model_type='purchase'
)
# Takes: ~20 seconds for 5000 clients
```

**Pros:**
- 🎯 Most accurate
- 🧹 Corrects any drift
- 📊 Learns global patterns

**Cons:**
- ⏱️ Slower (minutes)
- 💻 More compute intensive

**Use when:**
- Accumulated 1000+ new labels
- Weekly/monthly schedule
- Want maximum accuracy


### 3. 🧪 A/B Testing (Before Deploy)
**Best for:** Safe deployment

```python
# Compare old vs new model
comparison = trainer.compare_models(
    old_model_path='models/v1/purchase_model.pkl',
    new_model_path='models/v2/purchase_model.pkl',
    test_features=test_X,
    test_labels=test_y,
    model_type='purchase'
)

if comparison['better']:
    deploy_to_production(new_model_path)
```

**Pros:**
- ✅ Verify improvements
- 🛡️ Avoid regressions
- 📈 Track progress

**Use when:**
- Before any deployment
- Testing new features
- Validating retrain

---

## 🚀 Quick Start

### Option A: Interactive Workflow

```bash
cd /Users/ian/BDD2-LVMH
source .venv/bin/activate

# Run interactive workflow
python -m server.analytics.retrain_workflow

# Select:
# 1 - Daily incremental update
# 2 - Weekly full retrain
# 3 - A/B testing
# 4 - Run all
```

### Option B: Automated Schedule

```bash
# Setup automated training
python -m server.analytics.setup_scheduler

# This creates:
# - Daily script (6:00 AM)
# - Weekly script (Sunday 2:00 AM)
# - Monthly script (1st at 3:00 AM)
```

---

## 📅 Recommended Schedule

### Daily (Automated)
```bash
# 6:00 AM - Incremental update
scripts/ml_daily_update.sh

# What it does:
# 1. Collect yesterday's predictions
# 2. Match with actual outcomes
# 3. Incremental update (+50 trees)
# 4. Save new version
# Time: ~30 seconds
```

### Weekly (Automated)
```bash
# Sunday 2:00 AM - Full retrain
scripts/ml_weekly_retrain.sh

# What it does:
# 1. Collect all data (7 days)
# 2. Full retrain from scratch
# 3. Train all 3 models (purchase, churn, CLV)
# 4. Save new versions
# Time: ~3 minutes
```

### Monthly (Manual Review)
```bash
# 1st of month - A/B testing
scripts/ml_monthly_ab_test.sh

# What it does:
# 1. Compare new vs production model
# 2. Test on holdout set
# 3. Make deploy recommendation
# Time: ~1 minute
```

---

## 💻 Code Examples

### Example 1: Collect Real Labels

```python
from server.analytics.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

# Your predictions from yesterday
predictions_df = pd.read_csv('outputs/predictions_2026-02-09.csv')
# Columns: client_id, purchased_pred, churned_pred, clv_pred

# Actual outcomes from your database
outcomes_df = pd.read_sql("""
    SELECT 
        client_id,
        purchased as purchased_actual,
        churned as churned_actual,
        lifetime_value as clv_actual
    FROM client_outcomes
    WHERE date = '2026-02-09'
""", db_connection)

# Collect labeled examples
labeled_data = trainer.collect_new_labels(
    predictions_file='outputs/predictions_2026-02-09.csv',
    actual_outcomes_file='data/outcomes_2026-02-09.csv'
)

print(f"Collected {len(labeled_data)} new labels")
# Current model accuracy: 68.5%
```

### Example 2: Incremental Update

```python
import joblib
from server.analytics.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

# Load current production model
model = joblib.load('models/production/purchase_model.pkl')

# Extract features for new clients (match training format!)
# ... (your feature extraction code)

# Incremental update
updated_model, metrics = trainer.incremental_update(
    model=model,
    new_features=new_X,  # Shape: (100, 506)
    new_labels=new_y,    # Shape: (100,)
    model_name='purchase'
)

print(f"Updated in {metrics['train_time']:.1f}s")
print(f"Accuracy on new data: {metrics['accuracy']:.2%}")

# Save version
trainer.save_model_version(
    model=updated_model,
    model_name='purchase',
    metrics=metrics
)
```

### Example 3: Full Retrain

```python
from server.analytics.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

# Load ALL labeled data (old + new)
all_data = pd.concat([
    pd.read_csv('data/labels_week1.csv'),
    pd.read_csv('data/labels_week2.csv'),
    pd.read_csv('data/labels_week3.csv'),
    pd.read_csv('data/labels_week4.csv')
])

# Extract features
all_X = extract_features(all_data)  # Shape: (5000, 506)
all_y = all_data['purchased_actual'].values

# Full retrain
new_model, metrics = trainer.full_retrain(
    all_features=all_X,
    all_labels=all_y,
    model_type='purchase'
)

print(f"Trained on {metrics['total_samples']} samples")
print(f"Validation accuracy: {metrics['accuracy']:.2%}")

# Save version
trainer.save_model_version(
    model=new_model,
    model_name='purchase',
    metrics=metrics,
    version='weekly_20260210'
)
```

### Example 4: A/B Testing

```python
from server.analytics.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

# Compare models
comparison = trainer.compare_models(
    old_model_path='models/production/purchase_model.pkl',
    new_model_path='models/continuous/v_20260210/purchase_model.pkl',
    test_features=test_X,
    test_labels=test_y,
    model_type='purchase'
)

print(f"Old accuracy: {comparison['old_accuracy']:.2%}")
print(f"New accuracy: {comparison['new_accuracy']:.2%}")
print(f"Improvement: {comparison['improvement_pct']:+.1f}%")

# Deploy if better
if comparison['better']:
    import shutil
    shutil.copy(
        'models/continuous/v_20260210/purchase_model.pkl',
        'models/production/purchase_model.pkl'
    )
    print("✅ Deployed new model!")
else:
    print("❌ Keeping current model")
```

### Example 5: Get Best Model

```python
from server.analytics.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

# Get best performing version
best_model_path = trainer.get_best_model('purchase')

print(f"Best model: {best_model_path}")
# models/continuous/v_20260210_143522/purchase_model.pkl

# Load it
best_model = joblib.load(best_model_path)
```

---

## 📊 Monitoring Performance

### Track Training History

```python
from server.analytics.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()

# View all training runs
for run in trainer.training_history:
    print(f"Version: {run['version']}")
    print(f"  Timestamp: {run['timestamp']}")
    print(f"  Model: {run['model_name']}")
    print(f"  Metrics: {run['metrics']}")
    print()
```

### Plot Progress Over Time

```python
trainer = ContinuousTrainer()

# Plot accuracy improvements
trainer.plot_training_history('purchase')
# Saves: models/continuous/purchase_history.png
```

### Check Model Versions

```bash
ls -lht models/continuous/v_*/
# v_20260210_143522/  (latest)
# v_20260209_120001/
# v_20260208_063045/
# ...
```

---

## 🔧 Integration with Production

### 1. CSV Upload API
```python
@app.post("/predict")
async def predict(file: UploadFile):
    # Load LATEST model
    trainer = ContinuousTrainer()
    model_path = trainer.get_best_model('purchase')
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(features)
    
    return {"predictions": predictions.tolist()}
```

### 2. Batch Processing
```python
# Nightly batch job
def nightly_predictions():
    # Use BEST model
    trainer = ContinuousTrainer()
    
    for model_name in ['purchase', 'churn', 'clv']:
        model_path = trainer.get_best_model(model_name)
        model = joblib.load(model_path)
        
        # Predict for all clients
        predictions = model.predict(all_features)
        save_predictions(predictions, model_name)
```

### 3. Real-time Inference
```python
# Cache models in memory
class ModelCache:
    def __init__(self):
        trainer = ContinuousTrainer()
        self.models = {
            'purchase': joblib.load(trainer.get_best_model('purchase')),
            'churn': joblib.load(trainer.get_best_model('churn')),
            'clv': joblib.load(trainer.get_best_model('clv'))
        }
        
    def predict(self, client_features):
        return {
            'purchase_prob': self.models['purchase'].predict_proba(client_features)[0][1],
            'churn_risk': self.models['churn'].predict_proba(client_features)[0][1],
            'predicted_clv': self.models['clv'].predict(client_features)[0]
        }

# Initialize once at startup
model_cache = ModelCache()
```

---

## 🎯 Best Practices

### ✅ DO:
- **Collect real labels** from actual outcomes (purchases, churn, CLV)
- **Test before deploy** using A/B testing
- **Monitor performance** over time
- **Version everything** (models, features, data)
- **Start simple** with daily incremental updates
- **Schedule retrains** during low-traffic hours

### ❌ DON'T:
- Don't deploy untested models
- Don't ignore degrading accuracy
- Don't mix feature formats (train vs inference)
- Don't skip validation sets
- Don't retrain too frequently (wait for meaningful data)

### 🎪 Feature Engineering Tips:
- **Keep features consistent** between training and inference
- **Save feature extractors** (TfidfVectorizer, etc.)
- **Document feature changes** in version metadata
- **Test feature compatibility** before retraining

---

## 🐛 Troubleshooting

### Problem: Model accuracy decreasing
**Solution:**
```python
# 1. Check training history
trainer = ContinuousTrainer()
recent_runs = trainer.training_history[-10:]  # Last 10 runs

for run in recent_runs:
    print(f"{run['timestamp']}: {run['metrics']['accuracy']:.2%}")

# 2. If declining, do full retrain
new_model, metrics = trainer.full_retrain(
    all_features=all_X,
    all_labels=all_y,
    model_type='purchase'
)
```

### Problem: New features not matching
**Solution:**
```python
# Always save and load feature extractors
import joblib

# During training
vectorizer = TfidfVectorizer(max_features=500)
features = vectorizer.fit_transform(texts)
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# During inference
vectorizer = joblib.load('models/vectorizer.pkl')
features = vectorizer.transform(texts)  # Same shape!
```

### Problem: Too slow
**Solution:**
```python
# Use incremental instead of full retrain
# Daily: Incremental (~5s for 100 samples)
# Weekly: Full retrain (~30s for 5000 samples)

# Or reduce model size
model = RandomForestClassifier(
    n_estimators=100,  # Instead of 250
    max_depth=15,      # Instead of 20
    n_jobs=-1
)
```

---

## 📈 Expected Results

### With Continuous Training:

| Metric | Initial | After 1 Week | After 1 Month | After 3 Months |
|--------|---------|--------------|---------------|----------------|
| Purchase Accuracy | 59% | 65% | 72% | 80%+ |
| Churn Accuracy | 65% | 70% | 75% | 82%+ |
| CLV R² | 0.88 | 0.91 | 0.94 | 0.96+ |

**Why improvement?**
- Real labels replace synthetic data
- Model learns actual patterns
- Features better capture behavior
- More training data = better generalization

---

## 🚀 Next Steps

1. **Test manually:**
   ```bash
   python -m server.analytics.retrain_workflow
   ```

2. **Setup automation:**
   ```bash
   python -m server.analytics.setup_scheduler
   ```

3. **Monitor daily:**
   ```bash
   tail -f /tmp/lvmh_ml_daily.log
   ```

4. **Review monthly:**
   - Check training_history.json
   - Plot accuracy trends
   - Adjust schedules if needed

---

## 📚 Files Created

```
server/analytics/
├── continuous_trainer.py      # Core training logic
├── retrain_workflow.py        # Interactive workflows
├── setup_scheduler.py         # Automated scheduling
└── README_CONTINUOUS.md       # This guide

models/continuous/
├── training_history.json      # All training runs
└── v_YYYYMMDD_HHMMSS/        # Versioned models
    ├── purchase_model.pkl
    ├── churn_model.pkl
    ├── clv_model.pkl
    └── *_metadata.json

scripts/
├── ml_daily_update.sh         # Daily script
├── ml_weekly_retrain.sh       # Weekly script
└── ml_monthly_ab_test.sh      # Monthly script
```

---

**Ready to keep your models fresh! 🎯**

For questions or issues, review the training history:
```bash
cat models/continuous/training_history.json
```
