// Example: How to connect your React dashboard to the API server

import { useState } from 'react';
import API_CONFIG from './config';

// ═══════════════════════════════════════════════════════════════════════
// 1. UPLOAD CSV FILE
// ═══════════════════════════════════════════════════════════════════════

export function UploadCSV() {
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);

  const handleUpload = async (file: File) => {
    setUploading(true);
    
    try {
      // Upload CSV
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_CONFIG.BASE_URL}/api/upload-csv`, {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      console.log('Upload result:', result);
      
      if (result.pipeline_status === 'running') {
        setProcessing(true);
        // Start polling for pipeline status
        pollPipelineStatus();
      }
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  };

  const pollPipelineStatus = async () => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_CONFIG.BASE_URL}/api/pipeline/status`);
        const status = await response.json();
        
        if (!status.running) {
          clearInterval(interval);
          setProcessing(false);
          
          if (status.last_run === 'success') {
            console.log('Pipeline completed successfully!');
            // Refresh dashboard data
            loadDashboardData();
          } else {
            console.error('Pipeline failed:', status.last_error);
          }
        }
      } catch (error) {
        console.error('Status check failed:', error);
      }
    }, 3000); // Poll every 3 seconds
  };

  return (
    <div>
      <input 
        type="file" 
        accept=".csv" 
        onChange={(e) => e.target.files && handleUpload(e.target.files[0])}
        disabled={uploading || processing}
      />
      {uploading && <p>Uploading...</p>}
      {processing && <p>Processing pipeline... This may take ~60 seconds.</p>}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// 2. FETCH ML PREDICTIONS
// ═══════════════════════════════════════════════════════════════════════

export async function fetchPredictions() {
  try {
    const response = await fetch(`${API_CONFIG.BASE_URL}/api/predictions`);
    const predictions = await response.json();
    
    // predictions is an array of:
    // {
    //   client_id: "CA001",
    //   purchase_probability: 0.387,
    //   churn_risk: 0.605,
    //   predicted_clv: 6277.50,
    //   value_segment: "Medium Value"
    // }
    
    return predictions;
  } catch (error) {
    console.error('Failed to fetch predictions:', error);
    return [];
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. FETCH DASHBOARD DATA
// ═══════════════════════════════════════════════════════════════════════

export async function loadDashboardData() {
  try {
    const response = await fetch(`${API_CONFIG.BASE_URL}/api/dashboard-data`);
    const data = await response.json();
    
    // data contains:
    // {
    //   segments: [...],      // 8 client segments
    //   concepts: [...],      // 12 key concepts
    //   clients: [...],       // 400 client profiles
    //   scatter3d: [...],     // 3D visualization data
    //   heatmap: [...],       // Segment-concept heatmap
    //   metrics: { clients: 400, segments: 8 }
    // }
    
    return data;
  } catch (error) {
    console.error('Failed to load dashboard:', error);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. DISPLAY ML PREDICTIONS IN DASHBOARD
// ═══════════════════════════════════════════════════════════════════════

export function PredictionsDashboard() {
  const [predictions, setPredictions] = useState<any[]>([]);

  useEffect(() => {
    fetchPredictions().then(setPredictions);
  }, []);

  // Calculate summary statistics
  const avgPurchase = predictions.reduce((sum, p) => sum + (p.purchase_probability || 0), 0) / predictions.length;
  const avgChurn = predictions.reduce((sum, p) => sum + (p.churn_risk || 0), 0) / predictions.length;
  const totalCLV = predictions.reduce((sum, p) => sum + (p.predicted_clv || 0), 0);
  
  const atRiskCount = predictions.filter(p => p.churn_risk > 0.7).length;
  const vipCount = predictions.filter(p => p.value_segment === 'VIP').length;

  return (
    <div className="predictions-dashboard">
      <h2>ML Predictions Overview</h2>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Average Purchase Intent</h3>
          <p className="metric-value">{(avgPurchase * 100).toFixed(1)}%</p>
        </div>
        
        <div className="metric-card">
          <h3>Average Churn Risk</h3>
          <p className="metric-value">{(avgChurn * 100).toFixed(1)}%</p>
        </div>
        
        <div className="metric-card">
          <h3>Total Potential CLV</h3>
          <p className="metric-value">${totalCLV.toLocaleString()}</p>
        </div>
        
        <div className="metric-card alert">
          <h3>At-Risk Clients</h3>
          <p className="metric-value">{atRiskCount}</p>
          <p className="metric-label">Churn risk > 70%</p>
        </div>
        
        <div className="metric-card vip">
          <h3>VIP Clients</h3>
          <p className="metric-value">{vipCount}</p>
          <p className="metric-label">High value segment</p>
        </div>
      </div>

      {/* List of high-risk clients */}
      <div className="at-risk-list">
        <h3>⚠️ Clients at Risk of Churning</h3>
        <table>
          <thead>
            <tr>
              <th>Client ID</th>
              <th>Churn Risk</th>
              <th>Purchase Probability</th>
              <th>Predicted CLV</th>
              <th>Value Segment</th>
            </tr>
          </thead>
          <tbody>
            {predictions
              .filter(p => p.churn_risk > 0.7)
              .sort((a, b) => (b.predicted_clv || 0) - (a.predicted_clv || 0))
              .slice(0, 10)
              .map(client => (
                <tr key={client.client_id}>
                  <td>{client.client_id}</td>
                  <td className="risk-high">{(client.churn_risk * 100).toFixed(1)}%</td>
                  <td>{(client.purchase_probability * 100).toFixed(1)}%</td>
                  <td>${client.predicted_clv?.toLocaleString()}</td>
                  <td>{client.value_segment}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. PERFORMANCE MONITORING
// ═══════════════════════════════════════════════════════════════════════

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState({
    uploadTime: 0,
    processingTime: 0,
    totalClients: 0,
    throughput: 0
  });

  const measurePerformance = async (file: File) => {
    const uploadStart = Date.now();
    
    // Upload
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_CONFIG.BASE_URL}/api/upload-csv`, {
      method: 'POST',
      body: formData,
    });
    
    const uploadEnd = Date.now();
    const uploadTime = (uploadEnd - uploadStart) / 1000;
    
    // Wait for processing
    const processingStart = Date.now();
    let processing = true;
    
    while (processing) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const statusResponse = await fetch(`${API_CONFIG.BASE_URL}/api/pipeline/status`);
      const status = await statusResponse.json();
      
      if (!status.running) {
        processing = false;
      }
    }
    
    const processingEnd = Date.now();
    const processingTime = (processingEnd - processingStart) / 1000;
    
    // Get results
    const clientsResponse = await fetch(`${API_CONFIG.BASE_URL}/api/clients`);
    const clients = await clientsResponse.json();
    
    setMetrics({
      uploadTime,
      processingTime,
      totalClients: clients.length,
      throughput: clients.length / (uploadTime + processingTime)
    });
  };

  return (
    <div className="performance-monitor">
      <h3>Performance Metrics</h3>
      <ul>
        <li>Upload Time: {metrics.uploadTime.toFixed(2)}s</li>
        <li>Processing Time: {metrics.processingTime.toFixed(2)}s</li>
        <li>Total Clients: {metrics.totalClients}</li>
        <li>Throughput: {metrics.throughput.toFixed(2)} clients/second</li>
      </ul>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// 6. EXAMPLE CSS (optional styling)
// ═══════════════════════════════════════════════════════════════════════

const styles = `
.predictions-dashboard {
  padding: 2rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.metric-card {
  background: #fff;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-card.alert {
  border-left: 4px solid #ef4444;
}

.metric-card.vip {
  border-left: 4px solid #10b981;
}

.metric-value {
  font-size: 2rem;
  font-weight: bold;
  margin: 0.5rem 0;
}

.at-risk-list {
  margin-top: 2rem;
}

.risk-high {
  color: #ef4444;
  font-weight: bold;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid #e5e7eb;
}

th {
  background: #f3f4f6;
  font-weight: 600;
}
`;

// ═══════════════════════════════════════════════════════════════════════
// USAGE EXAMPLE
// ═══════════════════════════════════════════════════════════════════════

/*
// In your App.tsx:

import { UploadCSV, PredictionsDashboard, loadDashboardData } from './api-examples';

function App() {
  return (
    <div className="app">
      <header>
        <h1>LVMH Client Intelligence Dashboard</h1>
        <UploadCSV />
      </header>
      
      <main>
        <PredictionsDashboard />
        {/* ... rest of your dashboard ... *\/}
      </main>
    </div>
  );
}
*/

// ═══════════════════════════════════════════════════════════════════════
// API ENDPOINTS REFERENCE
// ═══════════════════════════════════════════════════════════════════════

/*
Available endpoints:

GET  /api/dashboard-data           → Complete dashboard JSON
GET  /api/clients                  → All 400 client profiles
GET  /api/clients/{id}             → Specific client
GET  /api/predictions              → All ML predictions
GET  /api/predictions/{id}         → Client predictions
POST /api/upload-csv               → Upload CSV + auto-run pipeline
POST /api/pipeline/run             → Manually trigger pipeline
GET  /api/pipeline/status          → Check if pipeline is running
GET  /api/knowledge-graph          → Relationship graph (Cytoscape)
GET  /api/lexicon                  → Domain vocabulary
GET  /api/outputs/{filename}       → Download output files

Interactive docs: http://localhost:8000/docs
*/
