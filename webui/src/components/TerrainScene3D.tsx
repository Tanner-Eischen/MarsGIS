import { useEffect, useMemo, useRef, useState } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js/dist/plotly-gl3d.min.js'
import type { ExplorerRoute } from '../lib/explorer'
import { buildTerrainScene, defaultTerrainCamera } from '../lib/terrainScene'

const Plot = createPlotlyComponent(Plotly)

export default function TerrainScene3D({ result, exaggeration }: { result: ExplorerRoute; exaggeration: number }) {
  const [cameraReset, setCameraReset] = useState(0)
  const [plotError, setPlotError] = useState('')
  const graphRef = useRef<HTMLElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const resize = () => {
      if (!graphRef.current) return
      try { Plotly.Plots.resize(graphRef.current) } catch { /* The WebGL scene may be unmounting. */ }
    }
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize)
    if (containerRef.current) observer?.observe(containerRef.current)
    return () => observer?.disconnect()
  }, [])
  const figure = useMemo(() => buildTerrainScene(result, exaggeration), [result, exaggeration])
  const layout = { ...figure.layout, uirevision: `${result.id}-${cameraReset}`, scene: { ...figure.layout.scene, camera: defaultTerrainCamera(), uirevision: `${result.id}-${cameraReset}` } }
  return <div className="terrain-scene-3d" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }} aria-label="Interactive three-dimensional Mars route terrain">
    <div className="terrain-scene-controls">
      <span>Vertical exaggeration ×{Math.max(1, Math.min(100, exaggeration))}</span>
      <button type="button" onClick={() => setCameraReset(value => value + 1)}>Reset 3D camera</button>
    </div>
    <p className="small-copy">Drag to orbit. Use the plot toolbar to pan, zoom or reset. Elevation measurements remain unchanged by exaggeration.</p>
    {plotError && <p role="alert">The 3D renderer could not start: {plotError}. The 2D map and route results remain available.</p>}
    <div ref={containerRef} className="terrain-scene-plot" style={{ flex: 1, minHeight: 0 }}>
    <Plot
      data={figure.data}
      layout={layout}
      config={{ displayModeBar: true, displaylogo: false, responsive: true, scrollZoom: true }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
      onInitialized={(_figure, graph) => { graphRef.current = graph }}
      onPurge={() => { graphRef.current = null }}
      onError={error => setPlotError(error.message)}
    />
    </div>
    <p className="small-copy">Surface displayed every {result.surface.display_step} analysis cells. Gaps mark NoData; route vertices use their measured terrain elevations.</p>
  </div>
}
