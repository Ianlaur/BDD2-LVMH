import { useState, FormEvent } from 'react'
import { useAuth } from './AuthContext'

export default function LoginScreen() {
  const { login, error, clearError, isLoading } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [localError, setLocalError] = useState('')

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    clearError()
    setLocalError('')

    if (!username.trim() || !password.trim()) {
      setLocalError('Username and password are required')
      return
    }

    try {
      await login(username.trim(), password)
    } catch {
      // error is already set in AuthContext
    }
  }

  const displayError = localError || error

  return (
    <div className="login-backdrop">
      <div className="login-card">
        <div className="login-header">
          <span className="login-logo">LVMH</span>
          <h1>Client Intelligence</h1>
          <p>Sign in to your account</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          {displayError && (
            <div className="login-error">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              {displayError}
            </div>
          )}

          <div className="login-field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Enter your username"
              autoFocus
              autoComplete="username"
              disabled={isLoading}
            />
          </div>

          <div className="login-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Enter your password"
              autoComplete="current-password"
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            className="login-btn"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <svg className="login-spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                  <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
                </svg>
                Signing in...
              </>
            ) : 'Sign In'}
          </button>
        </form>

        <div className="login-footer">
          <p>Contact your administrator for account access</p>
        </div>
      </div>
    </div>
  )
}
