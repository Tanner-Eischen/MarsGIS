const isBrowser = typeof window !== "undefined"
const hostname = isBrowser ? window.location.hostname : ""
const isLocalHost = hostname === "localhost" || hostname === "127.0.0.1"
const rawApiBase = (import.meta.env.VITE_API_URL ?? "").trim()

// In hosted deployments, prefer same-origin /api proxy to avoid browser CORS failures.
const DEFAULT_API_BASE = import.meta.env.PROD && !isLocalHost
  ? "/api/v1"
  : "http://localhost:5000/api/v1"

export const API_BASE = (rawApiBase || DEFAULT_API_BASE).replace(/\/+$/, "")

const rawWsBase = (import.meta.env.VITE_WS_URL ?? "").trim()
const DEFAULT_WS_BASE = API_BASE.replace(/\/api\/v1$/, "").replace(/^http/i, "ws")
export const WS_BASE = (rawWsBase || DEFAULT_WS_BASE).replace(/\/+$/, "")

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
