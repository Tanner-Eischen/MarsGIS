const DEFAULT_API_BASE = "http://localhost:5000/api/v1"

export const API_BASE = (import.meta.env.VITE_API_URL ?? DEFAULT_API_BASE).replace(/\/+$/, "")
const DEFAULT_WS_BASE = API_BASE.replace(/\/api\/v1$/, "").replace(/^http/i, "ws")
export const WS_BASE = (import.meta.env.VITE_WS_URL ?? DEFAULT_WS_BASE).replace(/\/+$/, "")

export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`
  return `${API_BASE}${normalizedPath}`
}

export function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(apiUrl(path), init)
}

export function wsUrl(path: string): string {
  if (/^wss?:\/\//i.test(path)) {
    return path
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`
  return `${WS_BASE}${normalizedPath}`
}
