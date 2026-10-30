import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ExplorerRoute } from '../lib/explorer'
import TerrainScene3D from './TerrainScene3D'

const state = vi.hoisted(() => ({ props: {} as any, resize: null as (() => void) | null, plotResize: vi.fn() }))
vi.mock('plotly.js/dist/plotly-gl3d.min.js', () => ({ default: { Plots: { resize: state.plotResize } } }))
vi.mock('react-plotly.js/factory', () => ({ default: () => (props: any) => { state.props = props; return <div data-testid="metric-plot" /> } }))
const result = {
  id: 'observed-run', surface: { x_m: [[0, 1000], [0, 1000]], y_m: [[0, 0], [2000, 2000]], elevation_m: [[-2000, null], [-1500, -1000]], display_step: 3 },
  path: [{ lat: 0, lon: 0, x_m: 0, y_m: 0, elevation_m: -2000 }, { lat: 0.01, lon: 0.01, x_m: 1000, y_m: 2000, elevation_m: -1000 }],
} as ExplorerRoute
beforeEach(() => { vi.stubGlobal('ResizeObserver', class { constructor(callback: () => void) { state.resize = callback } observe() {} disconnect() {} }) })
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals() })

describe('3D terrain controls with a mocked renderer', () => {
  it('keeps measured route/surface elevations and NoData unchanged when exaggerating', () => {
    const { rerender } = render(<TerrainScene3D result={result} exaggeration={1} />)
    const data = state.props.data
    expect(data[0].z).toEqual([[-2, null], [-1.5, -1]])
    expect(data[1].z).toEqual([-2, -1])
    expect(state.props.layout.scene.aspectratio.z).toBe(0.5)
    rerender(<TerrainScene3D result={result} exaggeration={20} />)
    expect(state.props.data).toEqual(data)
    expect(state.props.layout.scene.aspectratio.z).toBe(10)
    expect(screen.getByText('Vertical exaggeration ×20')).toBeTruthy()
    expect(state.props.layout.uirevision).toBe('observed-run-0')
  })
  it('exposes orbit/zoom toolbar and resets camera explicitly', () => {
    render(<TerrainScene3D result={result} exaggeration={5} />)
    expect(state.props.config).toMatchObject({ displayModeBar: true, scrollZoom: true, responsive: true })
    expect(state.props.useResizeHandler).toBe(true)
    state.props.layout.scene.camera.eye.x = 99 // Plotly may mutate supplied layout objects.
    fireEvent.click(screen.getByRole('button', { name: 'Reset 3D camera' }))
    expect(state.props.layout.uirevision).toBe('observed-run-1')
    expect(state.props.layout.scene.uirevision).toBe('observed-run-1')
    expect(state.props.layout.scene.camera.eye.x).toBe(1.45)
    expect(screen.getByText(/Surface displayed every 3 analysis cells/)).toBeTruthy()
  })
  it('resizes after pane layout changes and reports WebGL failure with a 2D fallback', () => {
    render(<TerrainScene3D result={result} exaggeration={5} />)
    act(() => state.props.onInitialized({}, document.body))
    act(() => state.resize?.())
    expect(state.plotResize).toHaveBeenCalledWith(document.body)
    act(() => state.props.onError(new Error('WebGL unavailable')))
    expect(screen.getByRole('alert').textContent).toContain('2D map and route results remain available')
  })
})
