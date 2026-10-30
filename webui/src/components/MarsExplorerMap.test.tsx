import type { ReactNode } from 'react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import L from 'leaflet'
import type { ExplorerMetadata, ExplorerRoute } from '../lib/explorer'
import MarsExplorerMap, { MARS_CRS } from './MarsExplorerMap'

const state = vi.hoisted(() => ({
  events: {} as Record<string, (event: any) => void>,
  center: 0, mapProps: {} as any, layers: [] as any[], markers: [] as any[], lines: [] as any[],
  fitBounds: vi.fn(), setView: vi.fn(), invalidateSize: vi.fn(), resize: null as (() => void) | null,
  map: null as any,
}))
vi.mock('react-leaflet', () => ({
  MapContainer: (props: any) => { state.mapProps = props; return <div data-testid="leaflet-map">{props.children}</div> },
  TileLayer: (props: any) => { state.layers.push(props); return null },
  CircleMarker: (props: any) => { state.markers.push(props); return <span>{props.children}</span> },
  Polyline: (props: any) => { state.lines.push(props); return null },
  Rectangle: () => null,
  Tooltip: ({ children }: { children: ReactNode }) => <span>{children}</span>,
  useMapEvents: (events: any) => { state.events = events; return state.map },
}))
const source = { id: 'mola', label: 'Measured MOLA', url: 'https://example.org/mola', sha256: 'a'.repeat(64), radius_m: 3396000, native_resolution_m: 3700 }
const metadata: ExplorerMetadata = {
  global: source, detail: { ...source, id: 'ctx', label: 'Measured CTX', bounds: [77, 18, 78, 19] },
  tiles: { mola: '/api/v1/explorer/tiles/mola/{z}/{x}/{y}.png', ctx: '/api/v1/explorer/tiles/ctx/{z}/{x}/{y}.png' },
  initial_view: { lat: 0, lon: 0, zoom: 1 }, detail_view: { lat: 18.5, lon: 77.5, zoom: 9 }, max_zoom: 12,
}
const result = { path: [{ lat: 1, lon: 179 }, { lat: 2, lon: -179 }], snapped: { start: { lat: 1, lon: 179 }, end: { lat: 2, lon: -179 } } } as ExplorerRoute
const props = { metadata, start: null, end: null, selection: 'start' as const, result: null, command: null, onPick: vi.fn() }

beforeEach(() => {
  state.center = 0; state.layers = []; state.markers = []; state.lines = []; state.resize = null
  state.map = { getCenter: () => ({ lng: state.center }), fitBounds: state.fitBounds, setView: state.setView, invalidateSize: state.invalidateSize, getContainer: () => document.body }
  vi.stubGlobal('ResizeObserver', class { constructor(callback: () => void) { state.resize = callback } observe() {} disconnect() {} })
  vi.stubGlobal('fetch', vi.fn())
})
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals() })

describe('Mars map presentation and interactions', () => {
  it('clones geographic CRS without mutating Leaflet Earth and enables map navigation', () => {
    render(<MarsExplorerMap {...props} />)
    expect(MARS_CRS.R).toBe(3396000)
    expect((L.CRS.EPSG4326 as L.CRS & { R: number }).R).toBe(6371000)
    expect(MARS_CRS.distance(L.latLng(0, 0), L.latLng(0, 1))).toBeCloseTo(3396000 * Math.PI / 180, 7)
    expect(state.mapProps).toMatchObject({ scrollWheelZoom: true, dragging: true, keyboard: true, touchZoom: true, maxZoom: 12, crs: MARS_CRS })
    expect(state.layers[0]).toMatchObject({ url: metadata.tiles.mola, maxNativeZoom: 4, noWrap: false })
    expect(state.layers[1]).toMatchObject({ url: metadata.tiles.ctx, minZoom: 7, maxNativeZoom: 11 })
    expect(screen.getByText(/Drag to pan Mars/)).toBeTruthy()
  })
  it('normalizes clicked world-copy longitudes and preserves selection mode', () => {
    const onPick = vi.fn()
    const { rerender } = render(<MarsExplorerMap {...props} onPick={onPick} />)
    act(() => state.events.click({ latlng: { lat: 20, lng: 437 } }))
    expect(onPick).toHaveBeenLastCalledWith({ lat: 20, lon: 77 }, 'start')
    rerender(<MarsExplorerMap {...props} selection="end" onPick={onPick} />)
    act(() => state.events.click({ latlng: { lat: -3, lng: 181 } }))
    expect(onPick).toHaveBeenLastCalledWith({ lat: -3, lon: -179 }, 'end')
    expect(fetch).not.toHaveBeenCalled()
  })
  it('keeps CTX tile coverage canonical while rendering a copied-world footprint', () => {
    state.center = -283
    render(<MarsExplorerMap {...props} />)
    act(() => state.events.moveend({ target: state.map }))
    expect(state.layers.slice(-1)[0].bounds).toEqual([[18, 77], [19, 78]])
    const layer = new L.GridLayer({ bounds: [[18, 77], [19, 78]], noWrap: false }) as any
    layer._map = {
      options: { crs: MARS_CRS },
      unproject: (point: L.Point, zoom: number) => MARS_CRS.pointToLatLng(point, zoom),
      wrapLatLngBounds: (bounds: L.LatLngBounds) => (MARS_CRS as any).wrapLatLngBounds(bounds),
    }
    layer._globalTileRange = L.bounds([0, 0], [1023, 511])
    const copyCoordinates = Object.assign(L.point(732 - 1024, 203), { z: 9 })
    // Exercise Leaflet's real tile coverage mathematics, without creating a map/browser.
    expect(layer._isValidTile(copyCoordinates)).toBe(true)
    layer.options.bounds = [[18, 77 - 360], [19, 78 - 360]]
    expect(layer._isValidTile(copyCoordinates)).toBe(false)
  })
  it('keeps routes continuous and aligns endpoint markers in the displayed world copy', () => {
    state.center = 180
    render(<MarsExplorerMap {...props} result={result} />)
    act(() => state.events.moveend({ target: state.map }))
    expect(state.lines.slice(-1)[0].positions).toEqual([[1, 179], [2, 181]])
    expect(state.markers.slice(-2).map(marker => marker.center)).toEqual([[1, 179], [2, 181]])
    act(() => { state.center = -180; state.events.moveend({ target: state.map }) })
    expect(state.lines.slice(-1)[0].positions).toEqual([[1, -181], [2, -179]])
    expect(fetch).not.toHaveBeenCalled()
  })
  it('responds once to explicit view commands without resetting on changed points', () => {
    const { rerender } = render(<MarsExplorerMap {...props} command={{ id: 1, kind: 'jezero' }} />)
    expect(state.setView).toHaveBeenCalledTimes(1)
    rerender(<MarsExplorerMap {...props} start={{ lat: 1, lon: 2 }} command={{ id: 1, kind: 'jezero' }} />)
    expect(state.setView).toHaveBeenCalledTimes(1)
    rerender(<MarsExplorerMap {...props} command={{ id: 2, kind: 'global' }} />)
    expect(state.fitBounds).toHaveBeenLastCalledWith([[-90, -180], [90, 180]], { padding: [12, 12] })
    rerender(<MarsExplorerMap {...props} result={result} command={{ id: 3, kind: 'route' }} />)
    expect(state.fitBounds).toHaveBeenLastCalledWith([[1, 179], [2, 181]], { padding: [32, 32], maxZoom: 11 })
  })
  it('invalidates layout on pane resize and offers tile retry without analysis', () => {
    render(<MarsExplorerMap {...props} />)
    expect(state.invalidateSize).toHaveBeenCalledWith({ pan: false })
    act(() => state.resize?.())
    expect(state.invalidateSize).toHaveBeenCalledTimes(2)
    act(() => state.layers[0].eventHandlers.tileerror())
    expect(screen.getByRole('status').textContent).toContain('could not load')
    fireEvent.click(screen.getByRole('button', { name: 'Retry map tiles' }))
    expect(screen.queryByRole('status')).toBeNull()
    expect(fetch).not.toHaveBeenCalled()
  })
})
