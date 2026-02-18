import { useState, useRef, useEffect } from 'react'
import API_CONFIG from './config'

interface VoiceRecorderProps {
  onRecordingComplete?: (audioBlob: Blob, transcript: string) => void;
  clientId?: string;
  userId?: number;
}

const Icons = {
  mic: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8"/></svg>,
  stop: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>,
  play: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>,
  pause: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>,
  upload: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>,
  trash: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6"/></svg>,
  wave: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v20M8 6v12M16 6v12M4 10v4M20 10v4"/></svg>,
}

export default function VoiceRecorder({ onRecordingComplete, clientId, userId }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [clientIdInput, setClientIdInput] = useState(clientId || '')
  const [language, setLanguage] = useState('FR')
  const [pipelineRunning, setPipelineRunning] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const recognitionRef = useRef<any>(null)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    // Initialize speech recognition if available
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = true
      recognitionRef.current.interimResults = true
      recognitionRef.current.lang = language === 'EN' ? 'en-US' : language === 'FR' ? 'fr-FR' : language === 'IT' ? 'it-IT' : language === 'ES' ? 'es-ES' : language === 'DE' ? 'de-DE' : 'fr-FR'

      recognitionRef.current.onresult = (event: any) => {
        let finalTranscript = ''
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' '
          }
        }
        if (finalTranscript) {
          setTranscript(prev => prev + finalTranscript)
        }
      }

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        if (event.error !== 'no-speech' && event.error !== 'aborted') {
          setError(`Speech recognition error: ${event.error}`)
        }
      }
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      // Initialize MediaRecorder
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const url = URL.createObjectURL(audioBlob)
        setAudioUrl(url)
        setAudioBlob(audioBlob)
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
        
        if (onRecordingComplete) {
          onRecordingComplete(audioBlob, transcript)
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
      setError('')
      setMessage('')
      setTranscript('')
      setDuration(0)

      // Start timer
      timerRef.current = window.setInterval(() => {
        setDuration(prev => prev + 1)
      }, 1000)

      // Start speech recognition
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start()
          setIsTranscribing(true)
        } catch (err) {
          console.warn('Could not start speech recognition:', err)
        }
      }
    } catch (err) {
      console.error('Error accessing microphone:', err)
      setError('Could not access microphone. Please grant permission.')
    }
  }

  const pauseRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.pause()
      setIsPaused(true)
      if (timerRef.current) clearInterval(timerRef.current)
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop()
        } catch (err) {
          console.warn('Error pausing recognition:', err)
        }
      }
    }
  }

  const resumeRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'paused') {
      mediaRecorderRef.current.resume()
      setIsPaused(false)
      timerRef.current = window.setInterval(() => {
        setDuration(prev => prev + 1)
      }, 1000)
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start()
        } catch (err) {
          console.warn('Error resuming recognition:', err)
        }
      }
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      setIsPaused(false)
      if (timerRef.current) clearInterval(timerRef.current)
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop()
          setIsTranscribing(false)
        } catch (err) {
          console.warn('Error stopping recognition:', err)
        }
      }
    }
  }

  const playAudio = () => {
    if (audioRef.current) {
      audioRef.current.play()
      setIsPlaying(true)
    }
  }

  const pauseAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      setIsPlaying(false)
    }
  }

  const deleteRecording = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(null)
    setAudioBlob(null)
    setTranscript('')
    setDuration(0)
    setMessage('')
    setError('')
  }

  const uploadRecording = async () => {
    if (!audioBlob) return

    if (!transcript.trim()) {
      setError('No transcript detected. Please speak clearly before uploading.')
      return
    }

    setUploading(true)
    setError('')
    setMessage('')
    setPipelineRunning(false)

    try {
      const formData = new FormData()
      const filename = `voice_memo_${Date.now()}.webm`
      formData.append('audio', audioBlob, filename)
      formData.append('transcript', transcript)
      formData.append('language', language)
      formData.append('duration_seconds', String(duration))
      if (clientIdInput.trim()) {
        formData.append('client_id', clientIdInput.trim())
      }

      const url = new URL(`${API_CONFIG.BASE_URL}/api/upload-voice-memo`)
      if (userId) url.searchParams.set('user_id', String(userId))
      url.searchParams.set('run_pipeline_after', 'true')

      const response = await fetch(url.toString(), {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => null)
        throw new Error(errData?.detail || `Upload failed: ${response.statusText}`)
      }

      const result = await response.json()
      const cid = result.client_id || clientIdInput || '?'

      if (result.pipeline_status === 'running' || result.pipeline_status === 'queued') {
        setPipelineRunning(true)
        const isKafka = result.pipeline_status === 'queued'
        setMessage(`✓ Voice memo for ${cid} uploaded — ${isKafka ? 'queued' : 'pipeline processing'}…`)

        if (isKafka) {
          // Use SSE (Server-Sent Events) for real-time Kafka updates
          const es = new EventSource(`${API_CONFIG.BASE_URL}/api/events`)

          const handleEvent = (e: MessageEvent) => {
            try {
              const data = JSON.parse(e.data)
              if (data.client_id === cid || data.upload_session_id === String(result.upload_session_id)) {
                if (data.type === 'pipeline_stage') {
                  setMessage(`✓ ${cid}: ${data.stage}…`)
                } else if (data.type === 'pipeline_completed') {
                  es.close()
                  setPipelineRunning(false)
                  setMessage(`✓ Client ${cid} processed and saved! (${data.total_time}s)`)
                  if (onRecordingComplete) onRecordingComplete(audioBlob!, transcript)
                } else if (data.type === 'pipeline_failed') {
                  es.close()
                  setPipelineRunning(false)
                  setError(`Pipeline failed: ${data.error || 'unknown'}`)
                }
              }
            } catch { /* ignore parse errors */ }
          }

          es.onmessage = handleEvent
          es.onerror = () => {
            // SSE connection lost — fall back to polling
            es.close()
            startPolling(cid, result.upload_session_id)
          }

          // Safety timeout
          setTimeout(() => {
            es.close()
            setPipelineRunning(false)
            setMessage(`✓ Memo uploaded — processing may still be running.`)
          }, 300_000)
        } else {
          // Direct mode — poll pipeline status
          startPolling(cid, result.upload_session_id)
        }
      } else {
        setMessage(`✓ Voice memo for ${cid} uploaded!`)
        if (onRecordingComplete) {
          onRecordingComplete(audioBlob!, transcript)
        }
      }
    } catch (err: any) {
      console.error('Upload error:', err)
      setError(err.message || 'Failed to upload voice memo')
    } finally {
      setUploading(false)
    }
  }

  const startPolling = (cid: string, _uploadSessionId?: number) => {
    const pollInterval = setInterval(async () => {
      try {
        const statusRes = await fetch(`${API_CONFIG.BASE_URL}/api/pipeline/status`)
        const statusData = await statusRes.json()
        if (!statusData.running) {
          clearInterval(pollInterval)
          setPipelineRunning(false)
          if (statusData.last_run === 'success') {
            setMessage(`✓ Client ${cid} processed and saved to database!`)
          } else {
            setError(`Pipeline finished with error: ${statusData.last_error || 'unknown'}`)
          }
          if (onRecordingComplete && audioBlob) {
            onRecordingComplete(audioBlob, transcript)
          }
        }
      } catch {
        // keep polling
      }
    }, 2000)

    setTimeout(() => {
      clearInterval(pollInterval)
      setPipelineRunning(false)
      setMessage(`✓ Memo uploaded — pipeline may still be running in background.`)
    }, 300_000)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="voice-recorder">
      <style>{`
        .voice-recorder {
          background: var(--bg-elevated);
          border: 1px solid var(--border-light);
          border-radius: var(--radius-lg);
          padding: 1.5rem;
          box-shadow: var(--shadow-sm);
        }

        .recorder-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 1.5rem;
        }

        .recorder-header svg {
          width: 24px;
          height: 24px;
          color: var(--accent-primary);
        }

        .recorder-header h3 {
          font-size: 1.125rem;
          font-weight: 600;
          margin: 0;
        }

        .recording-controls {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .control-buttons {
          display: flex;
          gap: 0.75rem;
          align-items: center;
          justify-content: center;
          flex-wrap: wrap;
        }

        .recorder-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1.25rem;
          border: none;
          border-radius: var(--radius-md);
          font-size: 0.9375rem;
          font-weight: 500;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .recorder-btn svg {
          width: 18px;
          height: 18px;
        }

        .recorder-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-record {
          background: #ef4444;
          color: white;
        }

        .btn-record:hover:not(:disabled) {
          background: #dc2626;
          transform: translateY(-1px);
        }

        .btn-record.recording {
          animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }

        .btn-stop {
          background: var(--text-primary);
          color: white;
        }

        .btn-stop:hover:not(:disabled) {
          background: var(--bg-dark-hover);
        }

        .btn-pause {
          background: var(--accent-warning);
          color: white;
        }

        .btn-pause:hover:not(:disabled) {
          background: #d97706;
        }

        .btn-play {
          background: var(--accent-success);
          color: white;
        }

        .btn-play:hover:not(:disabled) {
          background: #059669;
        }

        .btn-delete {
          background: var(--bg-tertiary);
          color: var(--text-secondary);
        }

        .btn-delete:hover:not(:disabled) {
          background: var(--border-medium);
        }

        .btn-upload {
          background: var(--accent-primary);
          color: white;
        }

        .btn-upload:hover:not(:disabled) {
          background: var(--accent-primary-dark);
          transform: translateY(-1px);
        }

        .duration-display {
          font-size: 1.5rem;
          font-weight: 600;
          color: var(--text-primary);
          text-align: center;
          font-variant-numeric: tabular-nums;
        }

        .recording-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          justify-content: center;
          color: var(--text-secondary);
          font-size: 0.875rem;
        }

        .recording-indicator svg {
          width: 20px;
          height: 20px;
          animation: wave 1s ease-in-out infinite;
        }

        @keyframes wave {
          0%, 100% { transform: scaleY(1); }
          50% { transform: scaleY(1.3); }
        }

        .transcript-box {
          margin-top: 1rem;
          padding: 1rem;
          background: var(--bg-tertiary);
          border-radius: var(--radius-md);
          border: 1px solid var(--border-light);
        }

        .transcript-box h4 {
          font-size: 0.875rem;
          font-weight: 600;
          color: var(--text-secondary);
          margin: 0 0 0.5rem 0;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .transcript-content {
          font-size: 0.9375rem;
          color: var(--text-primary);
          line-height: 1.6;
          min-height: 3rem;
          max-height: 10rem;
          overflow-y: auto;
        }

        .transcript-content:empty::before {
          content: 'Transcript will appear here as you speak...';
          color: var(--text-muted);
          font-style: italic;
        }

        .message-box {
          padding: 0.75rem 1rem;
          border-radius: var(--radius-md);
          font-size: 0.875rem;
          margin-top: 1rem;
        }

        .message-box.success {
          background: #d1fae5;
          color: #065f46;
          border: 1px solid #6ee7b7;
        }

        .message-box.error {
          background: #fee2e2;
          color: #991b1b;
          border: 1px solid #fca5a5;
        }

        .audio-player {
          display: none;
        }

        .voice-fields {
          display: grid;
          grid-template-columns: 1fr 120px;
          gap: 0.75rem;
          margin-bottom: 1rem;
        }

        .voice-field label {
          display: block;
          font-size: 0.75rem;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 0.25rem;
        }

        .voice-field input,
        .voice-field select {
          width: 100%;
          padding: 0.5rem 0.75rem;
          border: 1px solid var(--border-light);
          border-radius: var(--radius-md);
          background: var(--bg-tertiary);
          color: var(--text-primary);
          font-size: 0.875rem;
        }

        .voice-field input:focus,
        .voice-field select:focus {
          outline: none;
          border-color: var(--accent-primary);
          box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15);
        }

        .voice-field input::placeholder {
          color: var(--text-muted);
        }

        .pipeline-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          border-radius: var(--radius-md);
          background: rgba(99, 102, 241, 0.1);
          border: 1px solid rgba(99, 102, 241, 0.2);
          color: var(--accent-primary);
          font-size: 0.875rem;
          margin-top: 0.75rem;
        }

        .pipeline-indicator .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(99, 102, 241, 0.3);
          border-top-color: var(--accent-primary);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>

      <div className="recorder-header">
        {Icons.mic}
        <h3>Voice Memo Recorder</h3>
      </div>

      <div className="voice-fields">
        <div className="voice-field">
          <label>Client ID (optional)</label>
          <input
            type="text"
            placeholder="Auto-generated if empty (e.g. VM001)"
            value={clientIdInput}
            onChange={e => setClientIdInput(e.target.value)}
            disabled={isRecording || uploading}
          />
        </div>
        <div className="voice-field">
          <label>Language</label>
          <select value={language} onChange={e => setLanguage(e.target.value)} disabled={isRecording || uploading}>
            <option value="FR">French</option>
            <option value="EN">English</option>
            <option value="IT">Italian</option>
            <option value="ES">Spanish</option>
            <option value="DE">German</option>
          </select>
        </div>
      </div>

      <div className="recording-controls">
        <div className="duration-display">{formatTime(duration)}</div>

        {isRecording && (
          <div className="recording-indicator">
            {Icons.wave}
            <span>Recording{isPaused ? ' (Paused)' : ''}...</span>
          </div>
        )}

        <div className="control-buttons">
          {!isRecording && !audioUrl && (
            <button className="recorder-btn btn-record" onClick={startRecording}>
              {Icons.mic}
              Start Recording
            </button>
          )}

          {isRecording && !isPaused && (
            <>
              <button className="recorder-btn btn-pause" onClick={pauseRecording}>
                {Icons.pause}
                Pause
              </button>
              <button className="recorder-btn btn-stop" onClick={stopRecording}>
                {Icons.stop}
                Stop
              </button>
            </>
          )}

          {isRecording && isPaused && (
            <>
              <button className="recorder-btn btn-play" onClick={resumeRecording}>
                {Icons.play}
                Resume
              </button>
              <button className="recorder-btn btn-stop" onClick={stopRecording}>
                {Icons.stop}
                Stop
              </button>
            </>
          )}

          {audioUrl && !isRecording && (
            <>
              {!isPlaying ? (
                <button className="recorder-btn btn-play" onClick={playAudio}>
                  {Icons.play}
                  Play
                </button>
              ) : (
                <button className="recorder-btn btn-pause" onClick={pauseAudio}>
                  {Icons.pause}
                  Pause
                </button>
              )}
              <button 
                className="recorder-btn btn-upload" 
                onClick={uploadRecording}
                disabled={uploading}
              >
                {Icons.upload}
                {uploading ? 'Uploading...' : 'Upload Memo'}
              </button>
              <button className="recorder-btn btn-delete" onClick={deleteRecording}>
                {Icons.trash}
                Delete
              </button>
            </>
          )}
        </div>

        {(transcript || isTranscribing) && (
          <div className="transcript-box">
            <h4>
              Live Transcript
              {isTranscribing && <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>(Auto-transcribing...)</span>}
            </h4>
            <div className="transcript-content">{transcript}</div>
          </div>
        )}

        {message && <div className="message-box success">{message}</div>}
        {error && <div className="message-box error">{error}</div>}
        {pipelineRunning && (
          <div className="pipeline-indicator">
            <div className="spinner" />
            Processing through pipeline — segmenting, extracting concepts, generating actions…
          </div>
        )}
      </div>

      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          className="audio-player"
          onEnded={() => setIsPlaying(false)}
          onPause={() => setIsPlaying(false)}
        />
      )}
    </div>
  )
}
