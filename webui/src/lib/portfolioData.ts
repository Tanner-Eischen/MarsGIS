/** Contract for the published, precomputed measured-DEM analysis. */
export type LayerId = 'elevation' | 'slope' | 'roughness' | 'hillshade' | 'suitability'

export interface Candidate {
  rank: number
  row: number
  col: number
  score: number
  slope_deg: number
  roughness_m: number
  lon: number
  lat: number
  distance_m: number
  straight_line_m: number
  terrain_cost_m_equivalent: number
  distance_only_route_terrain_cost_m_equivalent: number
}

export interface Layer {
  id: LayerId
  label: string
  unit: string
  min: number
  max: number
  palette: [number, number, number][]
  image: string
  probe_band: number
}

export interface PortfolioScene {
  schema_version: 1
  data_mode: 'observed_dem'
  execution_mode: 'precomputed_reference_analysis' | 'server_computed_analysis'
  asset_base?: string
  source: {
    url: string
    citation: string
    sha256: string
    bytes: number
    instrument: string
    elevation_units?: string
    crs_wkt?: string
    [key: string]: unknown
  }
  grid: {
    shape: [number, number]
    resolution_m: number
    transform: [number, number, number, number, number, number]
    crs_wkt: string
    valid_cells?: number
    nodata?: string
    resampling?: string
    [key: string]: unknown
  }
  configuration: {
    weights: Record<string, number>
    constraints: Record<string, number>
    [key: string]: unknown
  }
  candidates: Candidate[]
  start_pixel: [number, number]
  routes: { rank: number; points: [number, number][] }[]
  layers: Layer[]
  probe: {
    path: string
    layout: 'band-major'
    dtype: 'float32-le'
    shape: [number, number, number]
    nodata: 'NaN'
  }
  downloads: {
    id: string
    label: string
    path: string
    sha256: string
    bytes: number
    media_type: string
  }[]
  artifacts_sha256: Record<string, string>
  [key: string]: unknown
}

const layerIds: LayerId[] = ['elevation', 'slope', 'roughness', 'hillshade', 'suitability']
const hashPattern = /^[a-f0-9]{64}$/i

function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Invalid portfolio scene: ${message}`)
}

function record(value: unknown, name: string): Record<string, unknown> {
  requireValue(value !== null && typeof value === 'object' && !Array.isArray(value), `${name} must be an object`)
  return value as Record<string, unknown>
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function integer(value: unknown, minimum = 0): value is number {
  return finite(value) && Number.isSafeInteger(value) && value >= minimum
}

function text(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

/** Only relative, unencoded local paths are accepted; URLs never leave this origin. */
export function assetUrl(path: string, base: string = '/portfolio/'): string {
  if (base !== '/portfolio/' && !/^\/api\/v1\/portfolio\/files\/[A-Za-z0-9_-]+\/$/.test(base)) {
    throw new Error('Invalid portfolio asset base')
  }
  if (typeof path !== 'string' || !path || !path.split('/').every(segment =>
    /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(segment) && segment !== '.' && segment !== '..'
  )) throw new Error('Invalid portfolio asset path')
  return `${base}${path}`
}

function safePath(value: unknown, name: string): asserts value is string {
  requireValue(typeof value === 'string', `${name} must be a path`)
  try { assetUrl(value) } catch { throw new Error(`Invalid portfolio scene: unsafe ${name}`) }
}

function httpsUrl(value: unknown, name: string) {
  requireValue(text(value), `${name} must be an HTTPS URL`)
  let url: URL
  try { url = new URL(value) } catch { throw new Error(`Invalid portfolio scene: ${name} must be an HTTPS URL`) }
  requireValue(url.protocol === 'https:' && !url.username && !url.password, `${name} must be an HTTPS URL`)
}

/** Validate the metadata before using it to place geometry or allocate probe data. */
export function validateScene(payload: unknown): PortfolioScene {
  const scene = record(payload, 'scene')
  requireValue(scene.schema_version === 1, 'unsupported schema_version')
  requireValue(scene.data_mode === 'observed_dem', 'data_mode must be observed_dem')
  requireValue(['precomputed_reference_analysis', 'server_computed_analysis'].includes(scene.execution_mode as string), 'unsupported execution_mode')
  if (scene.asset_base !== undefined) {
    requireValue(typeof scene.asset_base === 'string', 'asset_base must be a local path')
    try { assetUrl('probe.bin', scene.asset_base) } catch { throw new Error('Invalid portfolio scene: unsafe asset_base') }
  }
  const source = record(scene.source, 'source')
  httpsUrl(source.url, 'source.url')
  httpsUrl(source.citation, 'source.citation')
  requireValue(typeof source.sha256 === 'string' && hashPattern.test(source.sha256), 'invalid source SHA-256')
  requireValue(integer(source.bytes, 1) && text(source.instrument), 'invalid source provenance')

  const grid = record(scene.grid, 'grid')
  requireValue(Array.isArray(grid.shape) && grid.shape.length === 2 && grid.shape.every(n => integer(n, 1)), 'grid.shape must contain positive integer height and width')
  const [height, width] = grid.shape as [number, number]
  requireValue(Number.isSafeInteger(5 * height * width * 4), 'grid is too large')
  requireValue(finite(grid.resolution_m) && grid.resolution_m > 0 && text(grid.crs_wkt), 'invalid grid resolution or CRS')
  requireValue(Array.isArray(grid.transform) && grid.transform.length === 6 && grid.transform.every(finite), 'grid.transform must have six finite affine coefficients')
  const transform = grid.transform as number[]
  requireValue(transform[0] > 0 && transform[4] < 0 && transform[1] === 0 && transform[3] === 0, 'grid must be north-up and unrotated')
  if (grid.valid_cells !== undefined) requireValue(integer(grid.valid_cells) && grid.valid_cells <= height * width, 'invalid valid_cells')

  const configuration = record(scene.configuration, 'configuration')
  const weights = record(configuration.weights, 'configuration.weights')
  requireValue(Object.keys(weights).length > 0 && Object.values(weights).every(n => finite(n) && n >= 0), 'invalid criterion weights')
  const weightSum = (Object.values(weights) as number[]).reduce((sum, n) => sum + n, 0)
  requireValue(Math.abs(weightSum - 1) < 1e-8, 'criterion weights must sum to one')
  const constraints = record(configuration.constraints, 'configuration.constraints')
  requireValue(Object.values(constraints).every(n => finite(n) && n >= 0), 'invalid terrain constraints')

  requireValue(Array.isArray(scene.start_pixel) && scene.start_pixel.length === 2, 'invalid start_pixel')
  const [startRow, startCol] = scene.start_pixel
  requireValue(integer(startRow) && startRow < height && integer(startCol) && startCol < width, 'start_pixel outside grid')
  requireValue(Array.isArray(scene.candidates) && scene.candidates.length > 0, 'candidates must not be empty')
  const ranks = new Set<number>()
  for (const item of scene.candidates) {
    const candidate = record(item, 'candidate')
    requireValue(integer(candidate.rank, 1) && !ranks.has(candidate.rank), 'candidate ranks must be unique positive integers')
    ranks.add(candidate.rank)
    requireValue(integer(candidate.row) && candidate.row < height && integer(candidate.col) && candidate.col < width, 'candidate outside grid')
    requireValue(finite(candidate.score) && candidate.score >= 0 && candidate.score <= 1, 'candidate score must be between zero and one')
    requireValue(finite(candidate.lat) && Math.abs(candidate.lat) <= 90 && finite(candidate.lon) && candidate.lon >= -180 && candidate.lon <= 360, 'invalid candidate longitude/latitude')
    for (const field of ['slope_deg', 'roughness_m', 'distance_m', 'straight_line_m', 'terrain_cost_m_equivalent', 'distance_only_route_terrain_cost_m_equivalent']) {
      requireValue(finite(candidate[field]) && (candidate[field] as number) >= 0, `invalid candidate ${field}`)
    }
  }

  requireValue(Array.isArray(scene.routes) && scene.routes.length === ranks.size, 'routes must match candidates')
  const routeRanks = new Set<number>()
  for (const item of scene.routes) {
    const route = record(item, 'route')
    requireValue(integer(route.rank, 1) && ranks.has(route.rank) && !routeRanks.has(route.rank), 'route ranks must match candidates exactly')
    routeRanks.add(route.rank)
    requireValue(Array.isArray(route.points) && route.points.length >= 2, 'route must have at least two points')
    for (const point of route.points) {
      requireValue(Array.isArray(point) && point.length === 2 && finite(point[0]) && finite(point[1]) && point[0] >= 0 && point[0] < width && point[1] >= 0 && point[1] < height, 'route point outside grid')
    }
    const points = route.points as [number, number][]
    const candidate = (scene.candidates as Candidate[]).find(c => c.rank === route.rank)!
    requireValue(Math.abs(points[0][0] - startCol - 0.5) < 1e-6 && Math.abs(points[0][1] - startRow - 0.5) < 1e-6, 'route start does not match start_pixel')
    const last = points[points.length - 1]
    requireValue(Math.abs(last[0] - candidate.col - 0.5) < 1e-6 && Math.abs(last[1] - candidate.row - 0.5) < 1e-6, 'route endpoint does not match candidate')
  }

  const artifacts = record(scene.artifacts_sha256, 'artifacts_sha256')
  for (const [path, hash] of Object.entries(artifacts)) {
    safePath(path, 'artifact path')
    requireValue(typeof hash === 'string' && hashPattern.test(hash), `invalid artifact SHA-256 for ${path}`)
  }
  const requireArtifact = (path: unknown, name: string) => {
    safePath(path, name)
    requireValue(Object.prototype.hasOwnProperty.call(artifacts, path), `missing artifact checksum for ${path}`)
  }
  requireValue(Array.isArray(scene.layers) && scene.layers.length === 5, 'exactly five layers are required')
  const ids = new Set<string>(), bands = new Set<number>()
  for (const item of scene.layers) {
    const layer = record(item, 'layer')
    requireValue(typeof layer.id === 'string' && layerIds.includes(layer.id as LayerId) && !ids.has(layer.id), 'layer IDs must be unique and supported')
    ids.add(layer.id)
    requireValue(text(layer.label) && typeof layer.unit === 'string' && finite(layer.min) && finite(layer.max) && layer.min <= layer.max, 'invalid layer metadata')
    requireValue(integer(layer.probe_band) && layer.probe_band < 5 && !bands.has(layer.probe_band), 'probe bands must be unique values from zero to four')
    bands.add(layer.probe_band)
    requireValue(Array.isArray(layer.palette) && layer.palette.length >= 2 && layer.palette.every(color => Array.isArray(color) && color.length === 3 && color.every(n => integer(n) && n <= 255)), 'palette must contain RGB colors')
    requireArtifact(layer.image, 'layer.image')
  }
  const probe = record(scene.probe, 'probe')
  requireValue(probe.layout === 'band-major' && probe.dtype === 'float32-le' && probe.nodata === 'NaN', 'unsupported probe encoding')
  requireValue(Array.isArray(probe.shape) && probe.shape.length === 3 && probe.shape[0] === 5 && probe.shape[1] === height && probe.shape[2] === width, 'probe shape must match grid')
  requireArtifact(probe.path, 'probe.path')
  requireValue(Array.isArray(scene.downloads) && scene.downloads.length > 0, 'downloads must not be empty')
  const downloadIds = new Set<string>()
  for (const item of scene.downloads) {
    const download = record(item, 'download')
    requireValue(text(download.id) && !downloadIds.has(download.id), 'download IDs must be unique')
    downloadIds.add(download.id)
    requireArtifact(download.path, 'download.path')
    requireValue(text(download.label) && text(download.media_type) && integer(download.bytes, 1), 'invalid download metadata')
    requireValue(typeof download.sha256 === 'string' && download.sha256 === artifacts[download.path as string], 'download SHA-256 does not match artifact checksum')
  }
  return payload as PortfolioScene
}

export interface PortfolioStatus {
  state: 'idle' | 'running' | 'ready' | 'failed'
  data_available: boolean
  error?: string
  scene_url?: string | null
}

/** Preserve HTTP status so an initial 404 can be handled as an idle analysis. */
export class PortfolioRequestError extends Error {
  constructor(message: string, public status: number) {
    super(message)
    this.name = 'PortfolioRequestError'
  }
}

async function jsonResponse(response: Response, name: string): Promise<unknown> {
  if (!response.ok) throw new PortfolioRequestError(`${name} request failed (${response.status})`, response.status)
  const mediaType = response.headers.get('content-type')?.split(';')[0].trim().toLowerCase() || ''
  if (mediaType !== 'application/json' && !/^application\/[a-z0-9.+-]+\+json$/.test(mediaType)) {
    throw new Error(`${name} is unavailable: expected JSON, received another content type`)
  }
  return response.json()
}

function validateStatus(payload: unknown): PortfolioStatus {
  const status = record(payload, 'portfolio status')
  requireValue(['idle', 'running', 'ready', 'failed'].includes(status.state as string), 'unsupported portfolio state')
  requireValue(typeof status.data_available === 'boolean', 'portfolio status must declare data availability')
  if (status.error !== undefined) requireValue(typeof status.error === 'string', 'invalid portfolio error')
  if (status.scene_url !== undefined && status.scene_url !== null) requireValue(status.scene_url === '/api/v1/portfolio/scene', 'unsupported portfolio scene URL')
  if (status.scene_url === null) requireValue(status.data_available === false, 'null scene URL requires unavailable data')
  return payload as PortfolioStatus
}

export async function getPortfolioStatus(signal?: AbortSignal): Promise<PortfolioStatus> {
  const response = await fetch('/api/v1/portfolio/status', { signal })
  return validateStatus(await jsonResponse(response, 'Portfolio status'))
}

/** Start the single bounded server analysis; no credentials or arbitrary inputs. */
export async function startPortfolioRun(signal?: AbortSignal): Promise<PortfolioStatus> {
  const response = await fetch('/api/v1/portfolio/run', { method: 'POST', signal })
  if (response.status === 409) {
    throw new PortfolioRequestError('A portfolio analysis is already running. Wait for it to finish before starting another run.', 409)
  }
  return validateStatus(await jsonResponse(response, 'Portfolio analysis'))
}

export async function loadScene(signal?: AbortSignal): Promise<PortfolioScene> {
  const response = await fetch('/api/v1/portfolio/scene', { signal })
  return validateScene(await jsonResponse(response, 'Portfolio scene'))
}

export async function loadProbe(scene: PortfolioScene, signal?: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(assetUrl(scene.probe.path, scene.asset_base), { signal })
  if (!response.ok) throw new Error(`Portfolio probe request failed (${response.status})`)
  const buffer = await response.arrayBuffer()
  const expected = scene.grid.shape[0] * scene.grid.shape[1] * 5 * 4
  if (buffer.byteLength !== expected) throw new Error('Portfolio probe has an unexpected byte count')
  if (!globalThis.crypto?.subtle) throw new Error('SHA-256 verification requires Web Crypto in a secure context')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', buffer)
  const hash = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('')
  if (hash !== scene.artifacts_sha256[scene.probe.path]?.toLowerCase()) throw new Error('Portfolio probe SHA-256 verification failed')
  return buffer
}

/** Read the published cell values; no scientific calculations are performed here. */
export function probeCell(scene: PortfolioScene, buffer: ArrayBuffer, row: number, col: number): Record<LayerId, number | null> {
  const [height, width] = scene.grid.shape
  if (!integer(row) || !integer(col) || row >= height || col >= width) throw new RangeError('Probe cell outside grid')
  if (buffer.byteLength !== 5 * height * width * 4) throw new Error('Portfolio probe has an unexpected byte count')
  const view = new DataView(buffer)
  const result = {} as Record<LayerId, number | null>
  for (const layer of scene.layers) {
    const value = view.getFloat32(((layer.probe_band * height + row) * width + col) * 4, true)
    result[layer.id] = Number.isFinite(value) ? value : null
  }
  return result
}
