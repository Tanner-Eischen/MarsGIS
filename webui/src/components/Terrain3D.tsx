import { useEffect, useRef, useState } from 'react'
import Plot from 'react-plotly.js'

interface Terrain3DProps {
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset?: string
  showSites?: boolean
  showWaypoints?: boolean
}

export default function Terrain3D({ roi, dataset = 'mola', showSites = false, showWaypoints = false }: Terrain3DProps) {
  const [terrainData, setTerrainData] = useState<any>(null)
  const [sitesGeoJson, setSitesGeoJson] = useState<any>(null)
  const [waypointsGeoJson, setWaypointsGeoJson] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [verticalRelief, setVerticalRelief] = useState(1.0)
  const [horizontalScale, setHorizontalScale] = useState(1.0)

  useEffect(() => {
    if (!roi) return

    const load3DTerrain = async () => {
      setLoading(true)
      try {
        // Load 3D terrain data
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
        const terrainUrl = `http://localhost:5000/api/v1/visualization/terrain-3d?dataset=${dataset}&roi=${roiStr}&max_points=50000`
        
        const terrainResponse = await fetch(terrainUrl)
        if (terrainResponse.ok) {
          const data = await terrainResponse.json()
          setTerrainData(data)
        }

        // Load sites if requested
        if (showSites) {
          try {
            const sitesResponse = await fetch('http://localhost:5000/api/v1/visualization/sites-geojson')
            if (sitesResponse.ok) {
              const geojson = await sitesResponse.json()
              setSitesGeoJson(geojson)
            }
          } catch (e) {
            console.warn('Failed to load sites:', e)
          }
        }

        // Load waypoints if requested
        if (showWaypoints) {
          try {
            const waypointsResponse = await fetch('http://localhost:5000/api/v1/visualization/waypoints-geojson')
            if (waypointsResponse.ok) {
              const geojson = await waypointsResponse.json()
              setWaypointsGeoJson(geojson)
            }
          } catch (e) {
            console.warn('Failed to load waypoints:', e)
          }
        }
      } catch (error) {
        console.error('Failed to load 3D terrain:', error)
      } finally {
        setLoading(false)
      }
    }

    load3DTerrain()
  }, [roi, dataset, showSites, showWaypoints])

  if (!roi) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-400">Select a region of interest to view 3D terrain visualization</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-300">Loading 3D terrain data...</p>
      </div>
    )
  }

  if (!terrainData) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-400">No terrain data available</p>
      </div>
    )
  }

  // Apply vertical relief scaling to z values
  const zScaled = terrainData.z.map((row: number[]) => 
    row.map((z: number) => z * verticalRelief)
  )

  // Prepare Plotly surface trace
  const surfaceTrace: any = {
    type: 'surface',
    x: terrainData.x,
    y: terrainData.y,
    z: zScaled,
    colorscale: 'Terrain',
    showscale: true,
    colorbar: {
      title: 'Elevation (m)',
      titleside: 'right',
    },
    hovertemplate: 'Lon: %{x:.4f}°<br>Lat: %{y:.4f}°<br>Elevation: %{z:.0f} m<extra></extra>',
  }

  const traces: any[] = [surfaceTrace]

  // Add site markers if available
  if (sitesGeoJson && sitesGeoJson.features) {
    sitesGeoJson.features.forEach((feature: any) => {
      if (feature.geometry.type === 'Point') {
        const [lon, lat] = feature.geometry.coordinates
        // Find elevation at this point (simplified - would need interpolation)
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: [lon],
          y: [lat],
          z: [feature.properties.mean_elevation_m || 0],
          marker: {
            size: 10,
            color: '#00ff00',
            symbol: 'circle',
          },
          name: `Site ${feature.properties.site_id}`,
          text: [`Site ${feature.properties.site_id}<br>Rank: ${feature.properties.rank}<br>Score: ${(feature.properties.suitability_score * 100).toFixed(1)}%`],
          hovertemplate: '%{text}<extra></extra>',
        })
      }
    })
  }

  // Add waypoint markers if available
  if (waypointsGeoJson && waypointsGeoJson.features) {
    waypointsGeoJson.features.forEach((feature: any) => {
      if (feature.geometry.type === 'Point' && feature.properties.waypoint_id) {
        const [lon, lat] = feature.geometry.coordinates
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: [lon],
          y: [lat],
          z: [0], // Would need elevation lookup
          marker: {
            size: 8,
            color: '#ff0000',
            symbol: 'diamond',
          },
          name: `Waypoint ${feature.properties.waypoint_id}`,
          text: [`Waypoint ${feature.properties.waypoint_id}`],
          hovertemplate: '%{text}<extra></extra>',
        })
      } else if (feature.geometry.type === 'LineString') {
        // Add path line
        const coords = feature.geometry.coordinates
        const lons = coords.map((c: number[]) => c[0])
        const lats = coords.map((c: number[]) => c[1])
        const zs = new Array(coords.length).fill(0) // Would need elevation lookup
        traces.push({
          type: 'scatter3d',
          mode: 'lines',
          x: lons,
          y: lats,
          z: zs,
          line: {
            color: '#ff0000',
            width: 4,
          },
          name: 'Navigation Path',
          hovertemplate: 'Path<extra></extra>',
        })
      }
    })
  }

  const layout = {
    title: '3D Terrain Visualization',
    scene: {
      xaxis: { title: 'Longitude (°)' },
      yaxis: { title: 'Latitude (°)' },
      zaxis: { title: 'Elevation (m)' },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 },
      },
      aspectmode: 'manual' as const,
      aspectratio: {
        x: 1 / horizontalScale,
        y: 1 / horizontalScale,
        z: 1,
      },
    },
    autosize: true,
    paper_bgcolor: '#1f2937',
    plot_bgcolor: '#1f2937',
    font: { color: '#ffffff' },
    margin: { l: 0, r: 0, t: 50, b: 0 },
  }

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="mb-4 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Vertical Relief: {verticalRelief.toFixed(1)}x
          </label>
          <input
            type="range"
            min="0.1"
            max="5.0"
            step="0.1"
            value={verticalRelief}
            onChange={(e) => setVerticalRelief(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0.1x</span>
            <span>5.0x</span>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Horizontal Scale: {horizontalScale.toFixed(1)}x
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={horizontalScale}
            onChange={(e) => setHorizontalScale(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0.5x</span>
            <span>2.0x</span>
          </div>
        </div>
      </div>
      <div className="h-[600px] w-full rounded-md overflow-hidden border border-gray-700">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>
      <div className="mt-4 text-sm text-gray-400">
        <p>Interactive 3D terrain viewer. Use mouse to rotate, zoom, and pan. Green markers = sites, Red markers = waypoints</p>
      </div>
    </div>
  )
}


