# 🚀 Quick Start: Connect Your Frontend to LVMH API

## Server is Running! ✅

Your API server is live at **http://localhost:8000**

---

## 📝 Step-by-Step Integration

### 1. Update Your Dashboard Config

Edit `dashboard/src/config.ts`:

```typescript
const API_CONFIG: APIConfig = {
  BASE_URL: 'http://localhost:8000',  // ← Use this
};
```

### 2. Test the API from Browser

Open your browser and try:

- **Swagger UI**: http://localhost:8000/docs
- **Get Predictions**: http://localhost:8000/api/predictions
- **Get Clients**: http://localhost:8000/api/clients
- **Get Dashboard**: http://localhost:8000/api/dashboard-data

### 3. Upload CSV from Frontend

```typescript
const uploadCSV = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/api/upload-csv', {
    method: 'POST',
    body: formData,
  });
  
  const result = await response.json();
  console.log('Upload complete:', result);
};
```

### 4. Check Pipeline Status

```typescript
const checkStatus = async () => {
  const response = await fetch('http://localhost:8000/api/pipeline/status');
  const status = await response.json();
  
  // status = { running: false, last_run: "success", last_error: null }
  return status;
};
```

### 5. Fetch ML Predictions

```typescript
const getPredictions = async () => {
  const response = await fetch('http://localhost:8000/api/predictions');
  const predictions = await response.json();
  
  // predictions = [
  //   {
  //     client_id: "CA001",
  //     purchase_probability: 0.387,
  //     churn_risk: 0.605,
  //     predicted_clv: 6277.50,
  //     value_segment: "Medium Value"
  //   },
  //   ...
  // ]
  
  return predictions;
};
```

---

## 🎯 Performance Metrics You'll See

### Upload Speed
- **Upload time**: ~0.041s for 48KB (400 clients)
- **Throughput**: 9,767 clients/second

### Processing Time
- **Complete pipeline**: ~60 seconds
- **ML predictions only**: 0.39 seconds
- **Per-client**: 0.98ms for ML predictions

### End-to-End
- **Total time**: ~60 seconds (upload + process + generate reports)
- **Throughput**: 6.67 clients/second

---

## 📊 ML Predictions Data Structure

Each client will have these predictions:

```typescript
interface ClientPrediction {
  client_id: string;              // "CA001"
  purchase_probability: number;   // 0.387 (38.7% likely to purchase)
  churn_risk: number;             // 0.605 (60.5% risk of churning)
  predicted_clv: number;          // 6277.50 (predicted lifetime value)
  value_segment: string;          // "VIP" | "High Value" | "Medium Value" | "Low Value"
}
```

### Summary Statistics (400 clients)
- **Average purchase probability**: 38.7%
- **Average churn risk**: 60.5%
- **Average CLV**: $6,277
- **Total potential value**: $2.3M
- **High-risk clients** (>70% churn): 339
- **VIP clients**: 33 ($19,882 avg CLV)

---

## 🎨 Display Ideas for Your Dashboard

### 1. Alert Widget - Churn Risk
```
⚠️ HIGH PRIORITY ALERTS
━━━━━━━━━━━━━━━━━━━━━━
339 clients at risk of churning (>70% risk)
Potential revenue loss: $2.1M

Top 10 at-risk VIP clients:
• CA015: 87% churn risk, $28,500 CLV
• CA042: 83% churn risk, $24,200 CLV
...
```

### 2. Opportunity Widget - Purchase Intent
```
🎯 SALES OPPORTUNITIES
━━━━━━━━━━━━━━━━━━━━━━
70 clients with medium-high purchase intent
Estimated conversion value: $620,000

Ready to buy:
• CA089: 92% purchase probability
• CA127: 88% purchase probability
...
```

### 3. Value Segmentation Chart
```
💎 CLIENT VALUE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━
VIP:          33 clients  ($656K total)
High Value:   TBD clients ($XXX total)
Medium Value: 339 clients ($2.1M total)
Low Value:    TBD clients ($XXX total)
```

---

## 🔄 Real-Time Upload Flow

Here's the complete flow for CSV upload:

```typescript
// 1. User selects CSV file
<input type="file" accept=".csv" onChange={handleUpload} />

// 2. Upload to server
const formData = new FormData();
formData.append('file', file);
await fetch('http://localhost:8000/api/upload-csv', {
  method: 'POST',
  body: formData,
});

// 3. Show processing indicator
setProcessing(true);

// 4. Poll for status every 3 seconds
const interval = setInterval(async () => {
  const status = await fetch('http://localhost:8000/api/pipeline/status').then(r => r.json());
  
  if (!status.running) {
    clearInterval(interval);
    setProcessing(false);
    
    if (status.last_run === 'success') {
      // 5. Refresh dashboard with new data
      loadDashboardData();
      loadPredictions();
    }
  }
}, 3000);
```

---

## 🧪 Test Commands

### Test API from Terminal

```bash
# Upload CSV
curl -X POST -F "file=@data/LVMH_Sales_Database.csv" \
  http://localhost:8000/api/upload-csv

# Check status
curl http://localhost:8000/api/pipeline/status | jq

# Get predictions
curl http://localhost:8000/api/predictions | jq '.[0]'

# Get dashboard data
curl http://localhost:8000/api/dashboard-data | jq '.metrics'

# Run automated test
./test_upload_flow.sh
```

### Expected Response Times

```bash
# All these should return in <0.5s
curl -w "\nTime: %{time_total}s\n" http://localhost:8000/api/predictions
curl -w "\nTime: %{time_total}s\n" http://localhost:8000/api/clients
curl -w "\nTime: %{time_total}s\n" http://localhost:8000/api/dashboard-data
```

---

## 📚 Full Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Performance Report**: `PERFORMANCE_TEST_RESULTS.md`
- **Integration Examples**: `dashboard/FRONTEND_INTEGRATION_EXAMPLES.tsx`
- **ML Integration**: `docs/ML_INTEGRATION_FIXED.md`
- **Continuous Training**: `docs/CONTINUOUS_TRAINING.md`

---

## 🎉 You're Ready!

Your server is running with:
- ✅ 400 clients processed
- ✅ ML predictions available
- ✅ Real-time API endpoints
- ✅ Background pipeline processing
- ✅ Complete documentation

**Start building your frontend now!** 🚀

The API is fast (6.67 clients/second end-to-end) and ready for production use.

---

## 💡 Pro Tips

1. **Cache predictions**: They don't change often, cache for 5 minutes
2. **Show progress**: Use polling to show real-time pipeline progress
3. **Highlight risks**: Make high-churn clients stand out visually
4. **Sort by value**: Show VIP clients first
5. **Add filters**: Let users filter by segment, risk level, etc.

---

**Questions?** Check the Swagger UI at http://localhost:8000/docs
