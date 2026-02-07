import { useEffect, useState, useRef } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js/dist/plotly.min.js';
import type { PlotlyHTMLElement } from 'plotly.js';
import { use3DTerrain } from '../hooks/use3DMapData';
import { useSitesGeoJson, useWaypointsGeoJson } from '../hooks/useMapData';
import { RoverPosition } from '../hooks/useRoverAnimation';
import { useCameraFollow } from '../hooks/useCameraFollow';

const Plot = createPlotlyComponent(Plotly as any);

interface Terrain3DProps {
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  dataset?: string;
  showSites?: boolean;
  showWaypoints?: boolean;
  enableRoverAnimation?: boolean;
  roverPosition?: RoverPosition | null;
  isAnimating?: boolean;
  overlayType?: 'elevation' | 'solar' | 'dust' | 'hillshade' | 'slope' | 'aspect' | 'roughness' | 'tri';
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
  dataset = 'mola', 
  showSites = false, 
  showWaypoints = false, 
  enableRoverAnimation = false, 
  roverPosition: externalRoverPosition = null, 
  isAnimating = false,
  overlayType,
  overlayOptions = {}
}: Terrain3DProps) {
  const { terrainData, loading: terrainLoading, error: terrainError } = use3DTerrain(roi, dataset);
  const sitesGeoJson = useSitesGeoJson(showSites);
  const waypointsGeoJson = useWaypointsGeoJson(showWaypoints);

  const [verticalRelief, setVerticalRelief] = useState(1.0);
  const [horizontalScale, setHorizontalScale] = useState(1.0);
  const [colorScale, setColorScale] = useState('Terrain');
  const [enableContourLines, setEnableContourLines] = useState(false);
  
  const plotRef = useRef<PlotlyHTMLElement | null>(null);

  // Use external rover position if provided, otherwise use internal state
  const roverPosition = externalRoverPosition;

  // Camera follow hook
  useCameraFollow(
    roverPosition,
    enableRoverAnimation && isAnimating,
    plotRef
  );

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

  if (terrainError) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="p-4 bg-red-900/50 border border-red-700 rounded-md">
          <p className="text-red-300 font-semibold">Error loading 3D terrain</p>
          <p className="text-red-400 text-sm mt-1">{terrainError}</p>
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

  const { x: xGrid, y: yGrid, z: zGrid } = terrainData;

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
  traces.push(surfaceTrace);

  if (waypointsGeoJson && Array.isArray(waypointsGeoJson.features)) {
    const lonAxis = Array.isArray(xGrid) && xGrid.length > 0 ? xGrid[0] : []
    const latAxis = Array.isArray(yGrid) && yGrid.length > 0 ? yGrid.map((row: number[]) => row[0]) : []
    const nearestIndex = (arr: number[], val: number) => {
      if (!arr || arr.length === 0) return 0
      let idx = 0
      let best = Math.abs(arr[0] - val)
      for (let i = 1; i < arr.length; i++) {
        const d = Math.abs(arr[i] - val)
        if (d < best) { best = d; idx = i }
      }
      return idx
    }
    for (const f of waypointsGeoJson.features) {
      if (f.geometry && f.geometry.type === 'LineString') {
        const coords = f.geometry.coordinates as number[][]
        const xs = coords.map(c => c[0])
        const ys = coords.map(c => c[1])
        const zs = [] as number[]
        for (let k = 0; k < coords.length; k++) {
          const j = nearestIndex(lonAxis as number[], xs[k])
          const i = nearestIndex(latAxis as number[], ys[k])
          const zv = zScaled[i] && zScaled[i][j] ? zScaled[i][j] : 0
          zs.push(zv)
        }
        const color = (f.properties && f.properties.line_color) ? f.properties.line_color : '#ff0000'
        traces.push({
          type: 'scatter3d',
          mode: 'lines',
          x: xs,
          y: ys,
          z: zs,
          line: { color, width: 6 },
          name: f.properties && f.properties.route_type ? f.properties.route_type : 'route'
        })
      }
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

  // Update rover position using Plotly.restyle for performance
  useEffect(() => {
    if (roverPosition && plotRef.current && enableRoverAnimation && isAnimating) {
      // Find rover trace index (should be the last trace)
      const roverTraceIndex = traces.length - 1;
      if (roverTraceIndex >= 0 && traces[roverTraceIndex]?.name === 'Rover') {
        Plotly.restyle(
          plotRef.current,
          {
            x: [[roverPosition.lon]],
            y: [[roverPosition.lat]],
            z: [[roverPosition.elevation]],
          },
          [roverTraceIndex]
        );
      }
    }
  }, [roverPosition, enableRoverAnimation, isAnimating, traces.length]);

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
      <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Vertical Relief</label>
            <input type="range" min="0.1" max="5.0" step="0.1" value={verticalRelief} onChange={e => setVerticalRelief(parseFloat(e.target.value))} className="w-full" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Color Scale</label>
            <select value={colorScale} onChange={e => setColorScale(e.target.value)} className="w-full bg-gray-700 text-white px-3 py-2 rounded-md text-sm">
              <option>Terrain</option>
              <option>Viridis</option>
              <option>Plasma</option>
              <option>Earth</option>
            </select>
          </div>
          <div className="flex items-center">
            <input type="checkbox" checked={enableContourLines} onChange={e => setEnableContourLines(e.target.checked)} id="contour-toggle" />
            <label htmlFor="contour-toggle" className="ml-2 text-sm text-gray-300">Show Contours</label>
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
