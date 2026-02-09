import axios from 'axios'
import { API_BASE } from '../lib/apiBase'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for better user experience
})

const LONG_RUNNING_TIMEOUT_MS = 180000

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
  criteria_weights?: Record<string, number>
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
  const response = await api.post<DownloadResponse>('/download', request, {
    timeout: LONG_RUNNING_TIMEOUT_MS,
  })
  return response.data
}

export const analyzeTerrain = async (request: AnalysisRequest): Promise<AnalysisResponse> => {
  const response = await api.post<AnalysisResponse>('/analyze', {
    dataset: request.dataset || 'mola_200m',
    threshold: request.threshold || 0.7,
    roi: request.roi,
    criteria_weights: request.criteria_weights,
  }, {
    timeout: LONG_RUNNING_TIMEOUT_MS,
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

export interface MultiRouteRequest {
  site_id: number
  analysis_dir: string
  start_lat: number
  start_lon: number
  strategies?: ('safest' | 'balanced' | 'direct')[]
  max_waypoint_spacing_m?: number
  max_slope_deg?: number
  task_id?: string
}

export interface RouteSummary {
  strategy: 'safest' | 'balanced' | 'direct'
  waypoints: Waypoint[]
  num_waypoints: number
  total_distance_m: number
  relative_cost_percent: number
}

export interface MultiRouteResponse {
  routes: RouteSummary[]
  site_id: number
  task_id: string
}

export const planMultipleRoutes = async (request: MultiRouteRequest): Promise<MultiRouteResponse> => {
  const response = await api.post<MultiRouteResponse>('/navigation/plan-routes', {
    site_id: request.site_id,
    analysis_dir: request.analysis_dir,
    start_lat: request.start_lat,
    start_lon: request.start_lon,
    strategies: request.strategies || ['balanced', 'direct'],
    max_waypoint_spacing_m: request.max_waypoint_spacing_m || 100.0,
    max_slope_deg: request.max_slope_deg || 25.0,
    task_id: request.task_id,
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

// Solar Analysis API
export interface SolarAnalysisRequest {
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset?: string
  sun_azimuth?: number
  sun_altitude?: number
  panel_efficiency?: number
  panel_area_m2?: number
  battery_capacity_kwh?: number
  daily_power_needs_kwh?: number
  battery_cost_per_kwh?: number
  mission_duration_days?: number
}

export interface SolarStatistics {
  min: number
  max: number
  mean: number
  std: number
  min_irradiance_w_per_m2: number
  max_irradiance_w_per_m2: number
  mean_irradiance_w_per_m2: number
}

export interface MissionImpactsResponse {
  power_generation_kwh_per_day: number
  power_surplus_kwh_per_day: number
  mission_duration_extension_days: number
  cost_savings_usd: number
  battery_reduction_kwh: number
}

export interface SolarAnalysisResponse {
  solar_potential_map: number[][]
  irradiance_map: number[][]
  statistics: SolarStatistics
  mission_impacts: MissionImpactsResponse
  shape: { rows: number; cols: number }
}

export const analyzeSolarPotential = async (request: SolarAnalysisRequest): Promise<SolarAnalysisResponse> => {
  const response = await api.post<SolarAnalysisResponse>('/solar/analyze', request)
  return response.data
}

export interface DemoModeResponse { enabled: boolean }
export const getDemoMode = async (): Promise<DemoModeResponse> => {
  const response = await api.get<DemoModeResponse>('/demo-mode')
  return response.data
}
export const setDemoMode = async (enabled: boolean): Promise<DemoModeResponse> => {
  const response = await api.post<DemoModeResponse>('/demo-mode', { enabled })
  return response.data
}

export interface ExampleROIItem { id: string; name: string; description: string; bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }; dataset: string }
export const getExampleROIs = async (): Promise<ExampleROIItem[]> => {
  const response = await api.get<ExampleROIItem[]>('/examples/rois')
  return response.data
}

export type OverlayType = 'elevation' | 'solar' | 'hillshade' | 'slope' | 'aspect' | 'roughness' | 'tri'

export interface OverlayOptions {
  colormap?: string
  relief?: number
  sunAzimuth?: number
  sunAltitude?: number
  width?: number
  height?: number
  buffer?: number
}

export interface OverlayImageResponse {
  url: string
  bounds: L.LatLngBounds
  overlayType: OverlayType
  loading: boolean
  error: string | null
}
