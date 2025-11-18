import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 150000, // 2.5 minute timeout for pathfinding (can take 30-120 seconds for large DEMs)
})

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data ? { data: config.data } : '')
    return config
  },
  (error) => {
    console.error('[API] Request error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor for logging
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Response ${response.status} from ${response.config.url}`)
    return response
  },
  (error) => {
    console.error(`[API] Error ${error.response?.status || 'NETWORK'} from ${error.config?.url || 'unknown'}:`, error.message)
    if (error.response) {
      console.error('[API] Error details:', error.response.data)
    }
    return Promise.reject(error)
  }
)

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
  task_id?: string
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
  task_id?: string
}

export interface NavigationRequest {
  site_id: number
  analysis_dir?: string
  start_lat: number
  start_lon: number
  max_waypoint_spacing_m?: number
  max_slope_deg?: number
  strategy?: 'safest' | 'balanced' | 'direct'  // Note: strategy is handled by backend config
  task_id?: string
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
  task_id?: string
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
  const response = await api.post<NavigationResponse>('/navigation/plan-route', {
    site_id: request.site_id,
    analysis_dir: request.analysis_dir || 'data/output',
    start_lat: request.start_lat,
    start_lon: request.start_lon,
    max_waypoint_spacing_m: request.max_waypoint_spacing_m || 100.0,
    max_slope_deg: request.max_slope_deg || 25.0,
  })
  return response.data
}

// Mission Scenarios API
export interface LandingScenarioRequest {
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset?: string
  preset_id?: string
  constraints?: { max_slope_deg?: number; min_area_km2?: number }
  suitability_threshold?: number
  custom_weights?: Record<string, number>
}

export interface LandingScenarioResponse {
  scenario_id: string
  sites: SiteCandidate[]
  top_site: SiteCandidate | null
  metadata: Record<string, any>
}

export interface TraverseScenarioRequest {
  start_site_id: number
  end_site_id: number
  analysis_dir: string
  preset_id?: string
  rover_capabilities?: { max_slope_deg?: number; max_roughness?: number }
  start_lat?: number
  start_lon?: number
  custom_weights?: Record<string, number>
}

export interface TraverseScenarioResponse {
  route_id: string
  waypoints: Waypoint[]
  total_distance_m: number
  estimated_time_h: number
  risk_score: number
  metadata: Record<string, any>
}

export const runLandingScenario = async (request: LandingScenarioRequest): Promise<LandingScenarioResponse> => {
  const response = await api.post<LandingScenarioResponse>('/mission/landing-scenario', request)
  return response.data
}

export const runTraverseScenario = async (request: TraverseScenarioRequest): Promise<TraverseScenarioResponse> => {
  const response = await api.post<TraverseScenarioResponse>('/mission/rover-traverse', request)
  return response.data
}

