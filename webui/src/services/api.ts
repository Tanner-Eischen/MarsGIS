import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface StatusResponse {
  status: string
  cache: {
    directory: string
    size_bytes: number
    size_mb: number
    file_count: number
  }
  output: {
    directory: string
    file_count: number
  }
  config: {
    data_sources: string[]
  }
}

export interface DownloadRequest {
  dataset: string
  roi: [number, number, number, number] // [lat_min, lat_max, lon_min, lon_max]
  force?: boolean
}

export interface DownloadResponse {
  status: string
  path: string
  cached: boolean
  size_mb?: number
}

export interface AnalysisRequest {
  roi: [number, number, number, number]
  dataset?: string
  threshold?: number
}

export interface SiteCandidate {
  site_id: number
  geometry_type: string
  area_km2: number
  lat: number
  lon: number
  mean_slope_deg: number
  mean_roughness: number
  mean_elevation_m: number
  suitability_score: number
  rank: number
}

export interface AnalysisResponse {
  status: string
  sites: SiteCandidate[]
  top_site_id: number
  top_site_score: number
  output_dir: string
}

export interface NavigationRequest {
  site_id: number
  analysis_dir?: string
  start_lat: number
  start_lon: number
  strategy?: 'safest' | 'balanced' | 'direct'
}

export interface Waypoint {
  waypoint_id: number
  x_meters: number
  y_meters: number
  tolerance_meters: number
}

export interface NavigationResponse {
  status: string
  waypoints: Waypoint[]
  path_length_m?: number
  num_waypoints: number
}

export const getStatus = async (): Promise<StatusResponse> => {
  const response = await api.get<StatusResponse>('/status')
  return response.data
}

export const downloadDEM = async (request: DownloadRequest): Promise<DownloadResponse> => {
  const response = await api.post<DownloadResponse>('/download', request)
  return response.data
}

export const analyzeTerrain = async (request: AnalysisRequest): Promise<AnalysisResponse> => {
  const response = await api.post<AnalysisResponse>('/analyze', {
    dataset: request.dataset || 'mola',
    threshold: request.threshold || 0.7,
    roi: request.roi,
  })
  return response.data
}

export const planNavigation = async (request: NavigationRequest): Promise<NavigationResponse> => {
  const response = await api.post<NavigationResponse>('/navigate', {
    analysis_dir: request.analysis_dir || 'data/output',
    strategy: request.strategy || 'balanced',
    ...request,
  })
  return response.data
}

