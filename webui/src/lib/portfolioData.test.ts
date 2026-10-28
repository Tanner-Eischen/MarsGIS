import { webcrypto } from 'node:crypto'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { assetUrl, getPortfolioStatus, startPortfolioRun, PortfolioRequestError, loadProbe, loadScene, probeCell, validateScene, type LayerId, type PortfolioScene } from './portfolioData'

const hash = 'a'.repeat(64)
const ids: LayerId[] = ['elevation', 'slope', 'roughness', 'hillshade', 'suitability']
function fixture(): PortfolioScene {
  return {
    schema_version: 1, data_mode: 'observed_dem', execution_mode: 'precomputed_reference_analysis',
    source: { url: 'https://example.org/dem.tif', citation: 'https://example.org/source', sha256: hash, bytes: 24, instrument: 'CTX stereo' },
    grid: { shape: [2, 3], resolution_m: 20, transform: [20, 0, 100, 0, -20, 200], crs_wkt: 'PROJCRS["Mars"]', valid_cells: 6 },
    configuration: { weights: { slope: 0.7, roughness: 0.3 }, constraints: { max_slope_deg: 15, max_roughness: 20 } },
    candidates: [{ rank: 1, row: 1, col: 2, score: 0.8, slope_deg: 2, roughness_m: 0.1, lon: 77.3, lat: 18.3, distance_m: 48, straight_line_m: 45, terrain_cost_m_equivalent: 63, distance_only_route_terrain_cost_m_equivalent: 65 }],
    start_pixel: [0, 0], routes: [{ rank: 1, points: [[0.5, 0.5], [1.5, 0.5], [2.5, 1.5]] }],
    layers: ids.map((id, probe_band) => ({ id, label: id, unit: 'm', min: 0, max: 1000, palette: [[0, 0, 0], [255, 255, 255]], image: `layers/${id}.png`, probe_band })),
    probe: { path: 'probe.bin', layout: 'band-major', dtype: 'float32-le', shape: [5, 2, 3], nodata: 'NaN' },
    downloads: [{ id: 'report', label: 'Report', path: 'report.md', sha256: hash, bytes: 20, media_type: 'text/markdown' }],
    artifacts_sha256: Object.fromEntries([...ids.map(id => [`layers/${id}.png`, hash]), ['probe.bin', hash], ['report.md', hash]]),
  }
}
function probeBuffer(): ArrayBuffer {
  const buffer = new ArrayBuffer(5 * 2 * 3 * 4), view = new DataView(buffer)
  for (let band = 0; band < 5; band++) for (let index = 0; index < 6; index++) view.setFloat32((band * 6 + index) * 4, band * 100 + index, true)
  view.setFloat32(0, -123.5, true)
  return buffer
}
async function checksum(buffer: ArrayBuffer): Promise<string> {
  const digest = await webcrypto.subtle.digest('SHA-256', buffer)
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}
afterEach(() => { vi.unstubAllGlobals() })

describe('local asset paths', () => {
  it('creates same-origin safe URLs', () => { expect(assetUrl('layers/slope-v1.png')).toBe('/portfolio/layers/slope-v1.png') })
  it.each(['../probe.bin', 'layers/../probe.bin', '/probe.bin', '//example.org/x', 'https://example.org/a', 'javascript:alert(1)', 'layers\\probe.bin', '%2e%2e/probe.bin', 'probe.bin?x=1', 'probe.bin#fragment', 'a//b', ''])('rejects unsafe path %s', path => { expect(() => assetUrl(path)).toThrow('Invalid portfolio asset path') })
})
describe('scene metadata', () => {
  it('accepts complete internally consistent metadata', () => { const scene = fixture(); expect(validateScene(scene)).toBe(scene) })
  it.each([
    (s: PortfolioScene) => { s.schema_version = 2 as 1 },
    (s: PortfolioScene) => { s.data_mode = 'synthetic' as 'observed_dem' },
    (s: PortfolioScene) => { s.grid.shape[1] = 0 },
    (s: PortfolioScene) => { s.grid.transform[0] = NaN },
    (s: PortfolioScene) => { s.grid.transform[1] = 1 },
    (s: PortfolioScene) => { s.grid.transform[4] = 20 },
    (s: PortfolioScene) => { s.configuration.weights.slope = 0.2 },
    (s: PortfolioScene) => { s.candidates[0].score = 1.1 },
    (s: PortfolioScene) => { s.candidates[0].row = 2 },
    (s: PortfolioScene) => { s.candidates[0].distance_m = Infinity },
    (s: PortfolioScene) => { s.routes[0].points[2] = [1.5, 1.5] },
    (s: PortfolioScene) => { s.routes[0].points[0] = [0, 0] },
    (s: PortfolioScene) => { s.routes[0].rank = 2 },
    (s: PortfolioScene) => { s.layers[1].probe_band = 0 },
    (s: PortfolioScene) => { s.layers[1].id = 'elevation' },
    (s: PortfolioScene) => { s.layers[0].palette[0][1] = 256 },
    (s: PortfolioScene) => { s.layers[0].image = '../elevation.png' },
    (s: PortfolioScene) => { s.probe.shape[0] = 4 },
    (s: PortfolioScene) => { s.source.url = 'javascript:alert(1)' },
    (s: PortfolioScene) => { s.source.sha256 = 'not-a-hash' },
    (s: PortfolioScene) => { delete s.artifacts_sha256['probe.bin'] },
    (s: PortfolioScene) => { s.downloads[0].sha256 = 'b'.repeat(64) },
  ])('rejects invalid metadata %#', mutate => { const scene = fixture(); mutate(scene); expect(() => validateScene(scene)).toThrow('Invalid portfolio scene:') })
})
describe('scene loading', () => {
  it('checks JSON content type and forwards cancellation', async () => {
    const request = vi.fn().mockResolvedValue(new Response(JSON.stringify(fixture()), { headers: { 'Content-Type': 'application/json; charset=utf-8' } }))
    vi.stubGlobal('fetch', request)
    const signal = new AbortController().signal
    expect((await loadScene(signal)).data_mode).toBe('observed_dem')
    expect(request).toHaveBeenCalledWith('/api/v1/portfolio/scene', { signal })
  })
  it('rejects HTTP 200 HTML SPA fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>app</html>', { headers: { 'Content-Type': 'text/html' } })))
    await expect(loadScene()).rejects.toThrow('expected JSON')
  })
  it('reports HTTP failures before parsing', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('offline', { status: 503 })))
    await expect(loadScene()).rejects.toThrow('503')
  })
})
describe('probe loading and offsets', () => {
  it('verifies exact bytes using SHA-256 before returning data', async () => {
    const buffer = probeBuffer(), scene = fixture()
    scene.artifacts_sha256['probe.bin'] = await checksum(buffer)
    vi.stubGlobal('crypto', webcrypto)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(buffer)))
    const loaded = await loadProbe(scene)
    expect(Array.from(new Uint8Array(loaded))).toEqual(Array.from(new Uint8Array(buffer)))
  })
  it('rejects same-sized corrupted probe bytes', async () => {
    const buffer = probeBuffer(), scene = fixture()
    scene.artifacts_sha256['probe.bin'] = await checksum(buffer)
    new DataView(buffer).setFloat32(0, 999, true)
    vi.stubGlobal('crypto', webcrypto)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(buffer)))
    await expect(loadProbe(scene)).rejects.toThrow('SHA-256 verification failed')
  })
  it('rejects truncation and HTTP failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new ArrayBuffer(4))))
    await expect(loadProbe(fixture())).rejects.toThrow('byte count')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 404 })))
    await expect(loadProbe(fixture())).rejects.toThrow('404')
  })
  it('reads little-endian band-major row/column offsets', () => {
    expect(probeCell(fixture(), probeBuffer(), 1, 2)).toEqual({ elevation: 5, slope: 105, roughness: 205, hillshade: 305, suitability: 405 })
    expect(probeCell(fixture(), probeBuffer(), 0, 0).elevation).toBe(-123.5)
  })
  it('converts nonfinite nodata to null while retaining measured zero', () => {
    const buffer = probeBuffer(), view = new DataView(buffer)
    view.setFloat32((6 + 5) * 4, NaN, true)
    view.setFloat32((12 + 5) * 4, Infinity, true)
    view.setFloat32(5 * 4, 0, true)
    expect(probeCell(fixture(), buffer, 1, 2)).toMatchObject({ slope: null, roughness: null, elevation: 0 })
  })
  it.each([[-1, 0], [0, -1], [2, 0], [0, 3], [0.5, 0], [NaN, 0]])('rejects invalid pixel (%s,%s)', (row, col) => { expect(() => probeCell(fixture(), probeBuffer(), row, col)).toThrow('outside grid') })
  it('rejects a partial buffer before reading', () => { expect(() => probeCell(fixture(), new ArrayBuffer(24), 0, 0)).toThrow('byte count') })
})

describe('unified server deployment contract', () => {
  it.each(['idle', 'failed'])('accepts the API-shaped %s status without an existing scene', async state => {
    const status = { state, data_available: false, scene_url: null }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(status), { headers: { 'Content-Type': 'application/json' } })))
    expect(await getPortfolioStatus()).toEqual(status)
  })
  it('accepts server execution and scopes assets to an immutable run', () => {
    const scene = fixture()
    scene.execution_mode = 'server_computed_analysis'
    scene.asset_base = '/api/v1/portfolio/files/run_42-abc/'
    expect(validateScene(scene)).toBe(scene)
    expect(assetUrl('layers/slope.png', scene.asset_base)).toBe('/api/v1/portfolio/files/run_42-abc/layers/slope.png')
  })
  it.each(['https://example.org/files/', '//example.org/files/', '/api/v1/portfolio/files/../', '/api/v1/portfolio/files/a/b/', '/api/v1/portfolio/files/%2e%2e/', '/api/v1/portfolio/files/run?x=1/', '/portfolio/../'])('rejects unsafe server asset base %s', base => {
    const scene = fixture()
    scene.asset_base = base
    expect(() => assetUrl('probe.bin', base)).toThrow('asset base')
    expect(() => validateScene(scene)).toThrow('unsafe asset_base')
  })
  it('retains HTTP 404 for an initial idle state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 404 })))
    await expect(loadScene()).rejects.toMatchObject({ status: 404, name: 'PortfolioRequestError' })
  })
  it('loads status with cancellation support', async () => {
    const status = { state: 'ready', data_available: true, scene_url: '/api/v1/portfolio/scene' }
    const request = vi.fn().mockResolvedValue(new Response(JSON.stringify(status), { headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', request)
    const signal = new AbortController().signal
    expect(await getPortfolioStatus(signal)).toEqual(status)
    expect(request).toHaveBeenCalledWith('/api/v1/portfolio/status', { signal })
  })
  it('starts the bounded analysis without parameters', async () => {
    const status = { state: 'running', data_available: true }
    const request = vi.fn().mockResolvedValue(new Response(JSON.stringify(status), { status: 202, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', request)
    const signal = new AbortController().signal
    expect(await startPortfolioRun(signal)).toEqual(status)
    expect(request).toHaveBeenCalledWith('/api/v1/portfolio/run', { method: 'POST', signal })
  })
  it('reports concurrent runs with an actionable conflict error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 409 })))
    await expect(startPortfolioRun()).rejects.toBeInstanceOf(PortfolioRequestError)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 409 })))
    await expect(startPortfolioRun()).rejects.toThrow('Wait for it to finish')
  })
  it('rejects HTML status fallback and invalid status metadata', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html/>', { headers: { 'Content-Type': 'text/html' } })))
    await expect(getPortfolioStatus()).rejects.toThrow('expected JSON')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ state: 'ready', data_available: 'yes' }), { headers: { 'Content-Type': 'application/json' } })))
    await expect(getPortfolioStatus()).rejects.toThrow('data availability')
  })
  it('downloads and verifies probes from the server run asset base', async () => {
    const scene = fixture(), buffer = probeBuffer()
    scene.asset_base = '/api/v1/portfolio/files/run-42/'
    scene.artifacts_sha256['probe.bin'] = await checksum(buffer)
    const request = vi.fn().mockResolvedValue(new Response(buffer))
    vi.stubGlobal('crypto', webcrypto)
    vi.stubGlobal('fetch', request)
    await loadProbe(scene)
    expect(request).toHaveBeenCalledWith('/api/v1/portfolio/files/run-42/probe.bin', { signal: undefined })
  })
})
