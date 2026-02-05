import { useState, useEffect } from 'react'
import API_CONFIG from './config'

interface UploadResult {
  status: string;
  filename?: string;
  path?: string;
  message?: string;
  pipeline_status?: string;
  processing_time?: number;
}

interface FileUploadProps {
  onUploadSuccess?: (result: UploadResult) => void;
}

interface ProcessingTimings {
  total?: number;
  [key: string]: number | undefined;
}

const Icons = {
  upload: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>,
  check: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg>,
  x: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>,
  loader: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/></svg>,
  timer: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>,
}

export default function FileUpload({ onUploadSuccess }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [processingTime, setProcessingTime] = useState<number | null>(null)
  const [elapsedTime, setElapsedTime] = useState<number>(0)
  const [timings, setTimings] = useState<ProcessingTimings | null>(null)
  const [startTime, setStartTime] = useState<number | null>(null)

  // Live timer while processing
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    if (uploading && startTime) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000))
      }, 100)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [uploading, startTime])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile && selectedFile.name.endsWith('.csv')) {
      setFile(selectedFile)
      setError('')
    } else {
      setError('Please select a CSV file')
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    const droppedFile = e.dataTransfer.files?.[0]
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      setFile(droppedFile)
      setError('')
    } else {
      setError('Please drop a CSV file')
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setUploading(true)
    setMessage('')
    setError('')
    setProcessingTime(null)
    setTimings(null)
    setStartTime(Date.now())
    setElapsedTime(0)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/api/upload-csv?run_pipeline_after=true`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const result = await response.json()
      setMessage(result.message || 'File uploaded and pipeline started!')
      setFile(null)
      
      // Notify parent component of successful upload
      if (onUploadSuccess) {
        onUploadSuccess(result)
      }

      // Poll for pipeline completion
      if (result.pipeline_status === 'running') {
        pollPipelineStatus()
      }
    } catch (err) {
      console.error('Upload error:', err)
      setError(err.message || 'Failed to upload file')
      setUploading(false)
      setStartTime(null)
    }
  }

  const pollPipelineStatus = async () => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${API_CONFIG.BASE_URL}/api/pipeline/status`)
        const status = await response.json()
        
        if (!status.running) {
          const finalTime = startTime ? (Date.now() - startTime) / 1000 : null
          setProcessingTime(finalTime)
          setUploading(false)
          setStartTime(null)
          
          // Fetch the updated data to get timing info
          try {
            const dataResponse = await fetch(`${API_CONFIG.BASE_URL}/api/data`)
            const data = await dataResponse.json()
            if (data.processingInfo?.pipelineTimings) {
              setTimings(data.processingInfo.pipelineTimings)
            }
          } catch (e) {
            console.error('Failed to fetch timing data:', e)
          }
          
          if (status.last_run === 'success') {
            setMessage('✓ Pipeline completed successfully! Refresh to see new data.')
            if (onUploadSuccess) {
              onUploadSuccess({ status: 'completed', processing_time: finalTime || undefined })
            }
          } else if (status.last_run === 'error') {
            setError(`Pipeline failed: ${status.last_error}`)
          }
          return true // Stop polling
        }
        return false // Continue polling
      } catch (err) {
        console.error('Status check error:', err)
        return true // Stop polling on error
      }
    }

    // Poll every 3 seconds for up to 5 minutes
    let attempts = 0
    const maxAttempts = 100
    const interval = setInterval(async () => {
      attempts++
      const shouldStop = await checkStatus()
      if (shouldStop || attempts >= maxAttempts) {
        clearInterval(interval)
        if (attempts >= maxAttempts) {
          setUploading(false)
          setStartTime(null)
        }
      }
    }, 3000)
  }

  return (
    <div className="file-upload-container">
      <div 
        className={`upload-drop-zone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="upload-icon">{Icons.upload}</div>
        <h3>Upload New CSV File</h3>
        <p>Drag and drop a CSV file here, or click to select</p>
        
        <input
          type="file"
          accept=".csv"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="file-input"
        />
        
        <label htmlFor="file-input" className="upload-button">
          Select CSV File
        </label>

        {file && (
          <div className="selected-file">
            <span className="file-icon">📄</span>
            <span className="file-name">{file.name}</span>
            <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
            <button 
              className="remove-file"
              onClick={(e) => { e.preventDefault(); setFile(null) }}
            >
              {Icons.x}
            </button>
          </div>
        )}

        {file && !uploading && (
          <button 
            className="process-button"
            onClick={handleUpload}
          >
            Upload & Process Data
          </button>
        )}

        {uploading && (
          <div className="uploading">
            <span className="loader">{Icons.loader}</span>
            <span>Uploading and processing...</span>
            <div className="live-timer">
              <span className="timer-icon">{Icons.timer}</span>
              <span className="timer-value">{elapsedTime}s</span>
            </div>
          </div>
        )}

        {message && (
          <div className="upload-message success">
            {Icons.check} {message}
          </div>
        )}

        {/* Processing Time Display */}
        {processingTime && !uploading && (
          <div className="processing-result">
            <div className="processing-header">
              <span className="timer-icon-large">{Icons.timer}</span>
              <div className="processing-total">
                <span className="total-label">Temps total de traitement</span>
                <span className="total-value">{processingTime.toFixed(1)}s</span>
              </div>
            </div>
            
            {timings && Object.keys(timings).length > 1 && (
              <div className="timing-breakdown">
                <h4>Détails par étape</h4>
                <div className="timing-bars">
                  {Object.entries(timings)
                    .filter(([key]) => key !== 'total')
                    .map(([stage, time]) => (
                      <div key={stage} className="timing-row">
                        <span className="timing-label">{stage}</span>
                        <div className="timing-bar-bg">
                          <div 
                            className="timing-bar-fill" 
                            style={{ width: `${Math.min(100, ((time || 0) / (timings.total || 1)) * 100)}%` }}
                          />
                        </div>
                        <span className="timing-time">{(time || 0).toFixed(1)}s</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="upload-message error">
            {Icons.x} {error}
          </div>
        )}
      </div>

      <style>{`
        .file-upload-container {
          padding: 1rem;
        }

        .upload-drop-zone {
          border: 2px dashed #d1d5db;
          border-radius: 12px;
          padding: 3rem 2rem;
          text-align: center;
          background: #f9fafb;
          transition: all 0.2s;
        }

        .upload-drop-zone.drag-over {
          border-color: #6366f1;
          background: #eef2ff;
        }

        .upload-icon {
          width: 48px;
          height: 48px;
          margin: 0 auto 1rem;
          color: #6366f1;
        }

        .upload-icon svg {
          width: 100%;
          height: 100%;
        }

        .upload-drop-zone h3 {
          margin: 0 0 0.5rem;
          color: #111827;
          font-size: 1.25rem;
        }

        .upload-drop-zone p {
          color: #6b7280;
          margin: 0 0 1.5rem;
        }

        .upload-button {
          display: inline-block;
          padding: 0.75rem 1.5rem;
          background: #6366f1;
          color: white;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 500;
          transition: background 0.2s;
        }

        .upload-button:hover {
          background: #4f46e5;
        }

        .selected-file {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin: 1rem auto;
          padding: 0.75rem 1rem;
          background: white;
          border-radius: 8px;
          border: 1px solid #e5e7eb;
          max-width: 400px;
        }

        .file-icon {
          font-size: 1.5rem;
        }

        .file-name {
          flex: 1;
          text-align: left;
          font-weight: 500;
          color: #111827;
        }

        .file-size {
          color: #6b7280;
          font-size: 0.875rem;
        }

        .remove-file {
          background: none;
          border: none;
          cursor: pointer;
          padding: 0.25rem;
          color: #ef4444;
          width: 20px;
          height: 20px;
        }

        .remove-file:hover {
          color: #dc2626;
        }

        .process-button {
          margin-top: 1rem;
          padding: 0.875rem 2rem;
          background: #10b981;
          color: white;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .process-button:hover {
          background: #059669;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }

        .uploading {
          margin-top: 1rem;
          display: flex;
          align-items: center;
          gap: 0.75rem;
          justify-content: center;
          color: #6366f1;
          font-weight: 500;
        }

        .loader {
          width: 24px;
          height: 24px;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .upload-message {
          margin-top: 1rem;
          padding: 0.75rem 1rem;
          border-radius: 8px;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          justify-content: center;
          font-weight: 500;
        }

        .upload-message.success {
          background: #d1fae5;
          color: #065f46;
          border: 1px solid #10b981;
        }

        .upload-message.error {
          background: #fee2e2;
          color: #991b1b;
          border: 1px solid #ef4444;
        }

        .upload-message svg {
          width: 20px;
          height: 20px;
        }

        /* Live Timer Styles */
        .live-timer {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          background: linear-gradient(135deg, #6366f1, #8b5cf6);
          color: white;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-weight: 600;
          font-size: 1.1rem;
        }

        .timer-icon {
          width: 20px;
          height: 20px;
        }

        .timer-icon svg {
          width: 100%;
          height: 100%;
        }

        /* Processing Result Display */
        .processing-result {
          margin-top: 1.5rem;
          background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
          border: 1px solid #0ea5e9;
          border-radius: 12px;
          padding: 1.5rem;
        }

        .processing-header {
          display: flex;
          align-items: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .timer-icon-large {
          width: 48px;
          height: 48px;
          color: #0ea5e9;
          background: white;
          padding: 10px;
          border-radius: 12px;
          box-shadow: 0 2px 8px rgba(14, 165, 233, 0.2);
        }

        .timer-icon-large svg {
          width: 100%;
          height: 100%;
        }

        .processing-total {
          display: flex;
          flex-direction: column;
        }

        .total-label {
          font-size: 0.875rem;
          color: #0369a1;
          font-weight: 500;
        }

        .total-value {
          font-size: 2rem;
          font-weight: 700;
          color: #0c4a6e;
        }

        .timing-breakdown {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #bae6fd;
        }

        .timing-breakdown h4 {
          font-size: 0.875rem;
          color: #0369a1;
          margin-bottom: 0.75rem;
        }

        .timing-bars {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .timing-row {
          display: grid;
          grid-template-columns: 100px 1fr 50px;
          align-items: center;
          gap: 0.75rem;
        }

        .timing-label {
          font-size: 0.75rem;
          color: #0369a1;
          font-weight: 500;
          text-transform: capitalize;
        }

        .timing-bar-bg {
          height: 8px;
          background: #e0f2fe;
          border-radius: 4px;
          overflow: hidden;
        }

        .timing-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, #0ea5e9, #6366f1);
          border-radius: 4px;
          transition: width 0.3s ease;
        }

        .timing-time {
          font-size: 0.75rem;
          color: #0c4a6e;
          font-weight: 600;
          text-align: right;
        }
      `}</style>
    </div>
  )
}
