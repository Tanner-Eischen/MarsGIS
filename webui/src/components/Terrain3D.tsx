import { useEffect, useState, useRef } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js/dist/plotly.min.js';
import type { PlotlyHTMLElement } from 'plotly.js';
import { use3DTerrain } from '../hooks/use3DMapData';
import { useSitesGeoJson, useWaypointsGeoJson } from '../hooks/useMapData';
import { RoverPosition } from '../hooks/useRoverAnimation';
import { useCameraFollow } from '../hooks/useCameraFollow';

const Plot = createPlotlyComponent(Plotly as any);

const toPlotlyColorScale = (colormap?: string, overlayType?: string) => {
  const source = (colormap || '').toLowerCase();
  if (source === 'terrain') return 'Terrain';
  if (source === 'viridis') return 'Viridis';
  if (source === 'plasma') return 'Plasma';
  if (source === 'magma') return 'Magma';
  if (source === 'cividis') return 'Cividis';
  if (overlayType === 'solar') return 'Plasma';
  return 'Terrain';
};

interface Terrain3DProps {
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  dataset?: string;
  showSites?: boolean;
  showWaypoints?: boolean;
  enableRoverAnimation?: boolean;
  roverPosition?: RoverPosition | null;
  isAnimating?: boolean;
  overlayType?: 'elevation' | 'solar' | 'dust' | 'hillshade' | 'slope' | 'aspect' | 'roughness' | 'tri' | 'viewshed' | 'comms_risk';
  overlayOptions?: {
    colormap?: string;
    relief?: number;
    sunAzimuth?: number;
    sunAltitude?: number;
    width?: number;
    height?: number;
    buffer?: number;
    marsSol?: number;
    season?: string;
    dustStormPeriod?: string;
  };
}

export default function Terrain3D({ 
  roi, 
  dataset = 'hirise', 
  showSites = false, 
  showWaypoints = false, 
  enableRoverAnimation = false, 
  roverPosition: externalRoverPosition = null, 
  isAnimating = false,
  overlayType,
  overlayOptions = {}
}: Terrain3DProps) {
  const { terrainData, metadata, loading: terrainLoading, error: terrainError } = use3DTerrain(roi || null, dataset);
  const waypointsGeoJson = useWaypointsGeoJson(showWaypoints);
  const sitesGeoJson = useSitesGeoJson(showSites);

  const [verticalRelief, setVerticalRelief] = useState(overlayOptions.relief ?? 1.0);
  const [colorScale, setColorScale] = useState(toPlotlyColorScale(overlayOptions.colormap, overlayType));
  const [enableContourLines, setEnableContourLines] = useState(false);
  const [showMeshLayer, setShowMeshLayer] = useState(true);
  const [showPathLayer, setShowPathLayer] = useState(true);
  const [showWaypointLayer, setShowWaypointLayer] = useState(true);
  const [showSiteLayer, setShowSiteLayer] = useState(true);

  const plotRef = useRef<PlotlyHTMLElement | null>(null);

  // Use external rover position if provided, otherwise use internal state
  const roverPosition = externalRoverPosition;

  // Camera follow hook
  useCameraFollow(
    roverPosition,
    enableRoverAnimation && isAnimating,
    plotRef
  );

  useEffect(() => {
    if (typeof overlayOptions.relief === 'number') {
      setVerticalRelief(overlayOptions.relief);
    }
  }, [overlayOptions.relief]);

  useEffect(() => {
    setColorScale(toPlotlyColorScale(overlayOptions.colormap, overlayType));
  }, [overlayOptions.colormap, overlayType]);

  // Keep hook order stable across loading/error/data states.
  // Resolve rover trace index from current Plotly graph data at runtime.
  useEffect(() => {
    if (!roverPosition || !plotRef.current || !enableRoverAnimation || !isAnimating) {
      return;
    }

    const graphDiv = plotRef.current as PlotlyHTMLElement & { data?: any[] };
    const tracesData = Array.isArray(graphDiv.data) ? graphDiv.data : [];
    const roverTraceIndex = tracesData.findIndex((trace: any) => trace?.name === 'Rover');
    if (roverTraceIndex < 0) {
      return;
    }

    Plotly.restyle(
      graphDiv,
      {
        x: [[roverPosition.lon]],
        y: [[roverPosition.lat]],
        z: [[roverPosition.elevation]],
      },
      [roverTraceIndex]
    );
  }, [roverPosition, enableRoverAnimation, isAnimating]);

  if (!roi) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-400">Select a region of interest to view 3D terrain visualization</p>
      </div>
    );
  }

  if (terrainLoading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-300">Loading 3D terrain data...</p>
      </div>
    );
  }

  if (terrainError?.isServiceUnavailable) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="p-4 bg-amber-900/40 border border-amber-600/70 rounded-md">
          <p className="text-amber-200 font-semibold">3D terrain service unavailable</p>
          <p className="text-amber-300 text-sm mt-1">{terrainError.message}</p>
          {terrainError.detail && (
            <p className="text-amber-400/90 text-xs mt-2">{terrainError.detail}</p>
          )}
        </div>
      </div>
    );
  }

  if (terrainError) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="p-4 bg-red-900/50 border border-red-700 rounded-md">
          <p className="text-red-300 font-semibold">Error loading 3D terrain</p>
          <p className="text-red-400 text-sm mt-1">{terrainError.message}</p>
          {terrainError.detail && (
            <p className="text-red-400/90 text-xs mt-2">{terrainError.detail}</p>
          )}
        </div>
      </div>
    );
  }

  if (!terrainData) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-400">No terrain data available</p>
      </div>
    );
  }

  const mesh = terrainData.mesh || terrainData;
  const { x: xGrid, y: yGrid, z: zGrid } = mesh;

  const zScaled = zGrid.map(row => row.map(z => (z || 0) * verticalRelief));

  const traces: any[] = [];

  const surfaceTrace = {
    type: 'surface',
    x: xGrid,
    y: yGrid,
    z: zScaled,
    colorscale: colorScale,
    showscale: true,
    colorbar: { title: 'Elevation (m)', titleside: 'right' },
    hovertemplate: 'Lon: %{x:.4f}<br>Lat: %{y:.4f}<br>Elevation: %{z:.0f} m<extra></extra>',
    contours: {
      z: {
        show: enableContourLines,
        project: { z: true },
        highlightcolor: '#ffffff',
        highlightwidth: 2,
      }
    }
  };
  if (showMeshLayer) traces.push(surfaceTrace);

  const sceneOverlays = terrainData.overlays || null;

  if (sceneOverlays && showPathLayer && Array.isArray(sceneOverlays.paths)) {
    for (const path of sceneOverlays.paths) {
      const coords = Array.isArray(path.coordinates) ? path.coordinates : [];
      if (coords.length < 2) continue;
      traces.push({
        type: 'scatter3d',
        mode: 'lines',
        x: coords.map((c: number[]) => c[0]),
        y: coords.map((c: number[]) => c[1]),
        z: coords.map((c: number[]) => c[2]),
        line: { color: path.properties?.line_color || '#ff0000', width: 6 },
        name: path.properties?.route_type || 'route',
      });
    }
  } else if (showPathLayer && waypointsGeoJson && Array.isArray(waypointsGeoJson.features)) {
    const lonAxis = Array.isArray(xGrid) && xGrid.length > 0 ? xGrid[0] : [];
    const latAxis = Array.isArray(yGrid) && yGrid.length > 0 ? yGrid.map((row: number[]) => row[0]) : [];
    const nearestIndex = (arr: number[], val: number) => {
      if (!arr || arr.length === 0) return 0;
      let idx = 0;
      let best = Math.abs(arr[0] - val);
      for (let i = 1; i < arr.length; i++) {
        const d = Math.abs(arr[i] - val);
        if (d < best) { best = d; idx = i; }
      }
      return idx;
    };

    for (const f of waypointsGeoJson.features) {
      if (f.geometry && f.geometry.type === 'LineString') {
        const coords = f.geometry.coordinates as number[][];
        const xs = coords.map(c => c[0]);
        const ys = coords.map(c => c[1]);
        const zs = coords.map((c) => {
          const j = nearestIndex(lonAxis as number[], c[0]);
          const i = nearestIndex(latAxis as number[], c[1]);
          return zScaled[i] && zScaled[i][j] ? zScaled[i][j] : 0;
        });
        traces.push({
          type: 'scatter3d',
          mode: 'lines',
          x: xs,
          y: ys,
          z: zs,
          line: { color: f.properties?.line_color || '#ff0000', width: 6 },
          name: f.properties?.route_type || 'route',
        });
      }
    }
  }

  if (sceneOverlays && showWaypointLayer && Array.isArray(sceneOverlays.waypoints) && sceneOverlays.waypoints.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: sceneOverlays.waypoints.map((p: any) => p.lon),
      y: sceneOverlays.waypoints.map((p: any) => p.lat),
      z: sceneOverlays.waypoints.map((p: any) => p.z),
      marker: { size: 3, color: '#f97316' },
      name: 'Waypoints',
    });
  }

  if (sceneOverlays && showSiteLayer && Array.isArray(sceneOverlays.sites) && sceneOverlays.sites.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers+text',
      x: sceneOverlays.sites.map((p: any) => p.lon),
      y: sceneOverlays.sites.map((p: any) => p.lat),
      z: sceneOverlays.sites.map((p: any) => p.z),
      text: sceneOverlays.sites.map((p: any) => `Site ${p.properties?.site_id || ''}`),
      textposition: 'top center',
      marker: { size: 6, color: '#22c55e' },
      name: 'Sites',
    });
  } else if (showSiteLayer && sitesGeoJson && Array.isArray(sitesGeoJson.features)) {
    const points = sitesGeoJson.features.filter((f: any) => f.geometry?.type === 'Point');
    if (points.length > 0) {
      traces.push({
        type: 'scatter3d',
        mode: 'markers+text',
        x: points.map((f: any) => f.geometry.coordinates[0]),
        y: points.map((f: any) => f.geometry.coordinates[1]),
        z: points.map(() => 0),
        text: points.map((f: any) => `Site ${f.properties?.site_id || ''}`),
        textposition: 'top center',
        marker: { size: 6, color: '#22c55e' },
        name: 'Sites',
      });
    }
  }

  // Add rover marker trace if animation is enabled and position is available
  if (enableRoverAnimation && roverPosition) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: [roverPosition.lon],
      y: [roverPosition.lat],
      z: [roverPosition.elevation],
      marker: {
        size: 10,
        color: '#ff0000',
        symbol: 'circle',
        line: { color: '#ffffff', width: 2 },
      },
      name: 'Rover',
      showlegend: false,
    });
  }

  const layout = {
    title: '3D Terrain Visualization',
    scene: {
      xaxis: { title: 'Longitude ()' },
      yaxis: { title: 'Latitude ()' },
      zaxis: { title: 'Elevation (m)' },
      aspectratio: { x: 1, y: 1, z: 0.1 * verticalRelief },
      dragmode: enableRoverAnimation && isAnimating ? false : 'orbit',
      camera: {
        // Camera will be controlled programmatically during animation
      },
    },
    autosize: true,
    paper_bgcolor: '#1f2937',
    plot_bgcolor: '#1f2937',
    font: { color: '#ffffff' },
    margin: { l: 0, r: 0, t: 50, b: 0 },
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      {metadata && (
        <div className="mb-4 bg-gray-900/80 border border-cyan-700/50 rounded-md px-3 py-2 text-xs font-mono text-cyan-300">
          <div>REQUESTED: {(metadata.datasetRequested || dataset).toUpperCase()}</div>
          <div>
            RENDERED: {(metadata.datasetUsed || dataset).toUpperCase()}
            {metadata.isFallback ? ' (fallback)' : ''}
          </div>
          {metadata.isFallback && metadata.fallbackReason && (
            <div className="text-amber-300 mt-1">REASON: {metadata.fallbackReason}</div>
          )}
        </div>
      )}

      <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Vertical Relief</label>
            <input type="range" min="0" max="5.0" step="0.1" value={verticalRelief} onChange={e => setVerticalRelief(parseFloat(e.target.value))} className="w-full" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Color Scale</label>
            <select value={colorScale} onChange={e => setColorScale(e.target.value)} className="w-full bg-gray-700 text-white px-3 py-2 rounded-md text-sm">
              <option>Terrain</option>
              <option>Viridis</option>
              <option>Plasma</option>
              <option>Earth</option>
              <option>Magma</option>
              <option>Cividis</option>
            </select>
          </div>
          <div className="flex items-center">
            <input type="checkbox" checked={enableContourLines} onChange={e => setEnableContourLines(e.target.checked)} id="contour-toggle" />
            <label htmlFor="contour-toggle" className="ml-2 text-sm text-gray-300">Show Contours</label>
          </div>
          <div className="flex flex-wrap items-center gap-3 col-span-2 md:col-span-4">
            <label className="text-xs text-gray-300"><input type="checkbox" checked={showMeshLayer} onChange={e => setShowMeshLayer(e.target.checked)} className="mr-1"/>Mesh</label>
            <label className="text-xs text-gray-300"><input type="checkbox" checked={showPathLayer} onChange={e => setShowPathLayer(e.target.checked)} className="mr-1"/>Paths</label>
            <label className="text-xs text-gray-300"><input type="checkbox" checked={showWaypointLayer} onChange={e => setShowWaypointLayer(e.target.checked)} className="mr-1"/>Waypoints</label>
            <label className="text-xs text-gray-300"><input type="checkbox" checked={showSiteLayer} onChange={e => setShowSiteLayer(e.target.checked)} className="mr-1"/>Sites</label>
          </div>
      </div>
      <div className="h-[600px] w-full rounded-md overflow-hidden border border-gray-700">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
          onInitialized={(figure, graphDiv) => {
            plotRef.current = graphDiv as PlotlyHTMLElement;
          }}
        />
      </div>
    </div>
  );
}
