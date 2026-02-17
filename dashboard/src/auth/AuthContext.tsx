import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { login as apiLogin } from '../services/apiService'

export interface User {
  id: number
  username: string
  display_name: string
  email: string
  role: 'admin' | 'sales' | 'manager' | 'viewer' | 'data-scientist' | 'data-analyst'
}

interface AuthContextType {
  user: User | null
  login: (username: string, password: string) => Promise<void>
  logout: () => void
  isLoading: boolean
  error: string | null
  clearError: () => void
}

const AuthContext = createContext<AuthContextType | null>(null)

const STORAGE_KEY = 'lvmh_user'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Restore session from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as User
        // Reject sessions with missing critical fields (e.g. from a previous bug)
        if (parsed && parsed.id && parsed.username && parsed.display_name) {
          setUser(parsed)
        } else {
          localStorage.removeItem(STORAGE_KEY)
        }
      } catch {
        localStorage.removeItem(STORAGE_KEY)
      }
    }
    setIsLoading(false)
  }, [])

  const login = async (username: string, password: string) => {
    setIsLoading(true)
    setError(null)

    try {
      // Uses apiService which tries server first, falls back to direct DB
      const data = await apiLogin(username, password)

      const loggedInUser: User = {
        id: data.user.id,
        username: data.user.username,
        display_name: data.user.displayName || data.user.display_name || data.user.username,
        email: data.user.email,
        role: data.user.role,
      }

      setUser(loggedInUser)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(loggedInUser))
    } catch (e: any) {
      setError(e.message || 'Login failed')
      throw e
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem(STORAGE_KEY)
  }

  const clearError = () => setError(null)

  return (
    <AuthContext.Provider value={{ user, login, logout, isLoading, error, clearError }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
