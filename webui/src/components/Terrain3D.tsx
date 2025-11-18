import { useEffect, useState } from 'react'
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
  const [error, setError] = useState<string | null>(null)
  const [verticalRelief, setVerticalRelief] = useState(1.0)
  const [horizontalScale, setHorizontalScale] = useState(1.0)

  useEffect(() => {
    if (!roi) return

    const load3DTerrain = async () => {
      setLoading(true)
      setError(null)
      try {
        // Load 3D terrain data
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
        const terrainUrl = `http://localhost:5000/api/v1/visualization/terrain-3d?dataset=${dataset}&roi=${roiStr}&max_points=50000`
        
        console.log('Fetching 3D terrain data from:', terrainUrl)
        const terrainResponse = await fetch(terrainUrl)
        if (terrainResponse.ok) {
          const data = await terrainResponse.json()
          console.log('3D terrain data loaded:', { 
            hasX: !!data.x, 
            hasY: !!data.y, 
            hasZ: !!data.z,
            xShape: data.x?.length,
            yShape: data.y?.length,
            zShape: data.z?.length
          })
          setTerrainData(data)
        } else {
          let errorText = ''
          try {
            const errorJson = await terrainResponse.json()
            errorText = JSON.stringify(errorJson, null, 2)
          } catch {
            errorText = await terrainResponse.text()
          }
          const errorMsg = `Failed to load 3D terrain: ${terrainResponse.status} ${terrainResponse.statusText}.\n${errorText}`
          console.error('Failed to load 3D terrain', {
            status: terrainResponse.status,
            statusText: terrainResponse.statusText,
            error: errorText,
            url: terrainUrl
          })
          setError(errorMsg)
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
        const errorMsg = error instanceof Error ? error.message : 'Unknown error occurred'
        console.error('Failed to load 3D terrain:', error)
        setError(errorMsg)
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

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="p-4 bg-red-900/50 border border-red-700 rounded-md">
          <p className="text-red-300 font-semibold">Error loading 3D terrain</p>
          <p className="text-red-400 text-sm mt-1">{error}</p>
          <p className="text-red-400 text-xs mt-2">Check browser console for details. Make sure the backend server is running on port 5000.</p>
        </div>
      </div>
    )
  }

  if (!terrainData) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-400">No terrain data available</p>
        <p className="text-gray-500 text-sm mt-2">The terrain data may still be loading, or there may be an issue with the API endpoint.</p>
      </div>
    )
  }

  // Apply vertical relief and horizontal scale to terrain
  // terrainData.x and terrainData.y are 2D grids
  const xGrid = terrainData.x as number[][]
  const yGrid = terrainData.y as number[][]
  
  // Extract coordinate vectors for center calculation
  const xCoords = xGrid[0] || []
  const yCoords = yGrid.map((row: number[]) => row[0]) || []
  const centerX = xCoords.length > 0 ? (xCoords[0] + xCoords[xCoords.length - 1]) / 2 : 0
  const centerY = yCoords.length > 0 ? (yCoords[0] + yCoords[yCoords.length - 1]) / 2 : 0
  
  // Apply scaling to z values - ensure no NaN or null values
  const zScaled = terrainData.z.map((row: number[]) => 
    row.map((z: number) => {
      const val = (isNaN(z) || z === null || z === undefined) ? 0 : z
      return val * verticalRelief
    })
  )
  
  // Apply horizontal scale to x and y coordinate grids - ensure valid numbers
  const xScaled = xGrid.map((row: number[]) =>
    row.map((x: number) => {
      const val = (isNaN(x) || x === null || x === undefined) ? centerX : x
      return centerX + (val - centerX) * horizontalScale
    })
  )
  
  const yScaled = yGrid.map((row: number[]) =>
    row.map((y: number) => {
      const val = (isNaN(y) || y === null || y === undefined) ? centerY : y
      return centerY + (val - centerY) * horizontalScale
    })
  )

  // Helper function to get exact terrain surface elevation at a point
  // This samples the already-scaled terrain surface directly
  const getSurfaceElevation = (lon: number, lat: number): number => {
    if (!terrainData || !terrainData.x || !terrainData.y || !terrainData.z) {
      return 0
    }

    // Find the grid cell containing this point using original (unscaled) coordinates
    let xIdx = -1
    let yIdx = -1
    
    // Find x index (longitude) - handle both ascending and descending
    const xAscending = xCoords.length > 1 && xCoords[1] > xCoords[0]
    for (let i = 0; i < xCoords.length - 1; i++) {
      const x1 = xCoords[i]
      const x2 = xCoords[i + 1]
      if (xAscending) {
        if (lon >= x1 && lon <= x2) {
          xIdx = i
          break
        }
      } else {
        if (lon <= x1 && lon >= x2) {
          xIdx = i
          break
        }
      }
    }
    
    // Find y index (latitude) - handle both ascending and descending
    const yAscending = yCoords.length > 1 && yCoords[1] > yCoords[0]
    for (let i = 0; i < yCoords.length - 1; i++) {
      const y1 = yCoords[i]
      const y2 = yCoords[i + 1]
      if (yAscending) {
        if (lat >= y1 && lat <= y2) {
          yIdx = i
          break
        }
      } else {
        if (lat <= y1 && lat >= y2) {
          yIdx = i
          break
        }
      }
    }

    // Use nearest edge if outside grid
    if (xIdx === -1) {
      xIdx = lon < xCoords[0] ? 0 : xCoords.length - 1
    }
    if (yIdx === -1) {
      yIdx = lat < yCoords[0] ? 0 : yCoords.length - 1
    }

    // Get exact elevation from scaled terrain surface
    const zVal = zScaled[yIdx][xIdx]
    return isNaN(zVal) ? 0 : zVal
  }

  // Prepare Plotly surface trace
  const surfaceTrace: any = {
    type: 'surface',
    x: xScaled,
    y: yScaled,
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


  // Add site markers if available - positioned directly on terrain surface
  if (sitesGeoJson && sitesGeoJson.features) {
    sitesGeoJson.features.forEach((feature: any) => {
      if (feature.geometry.type === 'Point') {
        const [lon, lat] = feature.geometry.coordinates
        
        // Apply horizontal scale to coordinates (same as terrain)
        const xGrid = terrainData.x as number[][]
        const yGrid = terrainData.y as number[][]
        const xCoords = xGrid[0] || []
        const yCoords = yGrid.map((row: number[]) => row[0]) || []
        const centerX = xCoords.length > 0 ? (xCoords[0] + xCoords[xCoords.length - 1]) / 2 : 0
        const centerY = yCoords.length > 0 ? (yCoords[0] + yCoords[yCoords.length - 1]) / 2 : 0
        const xScaled = centerX + (lon - centerX) * horizontalScale
        const yScaled = centerY + (lat - centerY) * horizontalScale
        
        // Get exact elevation from scaled terrain surface
        const zSurface = getSurfaceElevation(lon, lat)
        
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: [xScaled],
          y: [yScaled],
          z: [zSurface], // Use exact terrain surface elevation
          marker: {
            size: 10,
            color: '#00ff00',
            symbol: 'circle',
          },
          name: `Site ${feature.properties.site_id}`,
          text: [`Site ${feature.properties.site_id}<br>Rank: ${feature.properties.rank}<br>Score: ${(feature.properties.suitability_score * 100).toFixed(1)}%<br>Elevation: ${(zSurface / verticalRelief).toFixed(0)} m`],
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
        
        // Apply horizontal scale to coordinates (same as terrain)
        const xGrid = terrainData.x as number[][]
        const yGrid = terrainData.y as number[][]
        const xCoords = xGrid[0] || []
        const yCoords = yGrid.map((row: number[]) => row[0]) || []
        const centerX = xCoords.length > 0 ? (xCoords[0] + xCoords[xCoords.length - 1]) / 2 : 0
        const centerY = yCoords.length > 0 ? (yCoords[0] + yCoords[yCoords.length - 1]) / 2 : 0
        const xScaled = centerX + (lon - centerX) * horizontalScale
        const yScaled = centerY + (lat - centerY) * horizontalScale
        
        // Get exact elevation from scaled terrain surface
        const zSurface = getSurfaceElevation(lon, lat)
        
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: [xScaled],
          y: [yScaled],
          z: [zSurface], // Use exact terrain surface elevation
          marker: {
            size: 8,
            color: '#ff0000',
            symbol: 'diamond',
          },
          name: `Waypoint ${feature.properties.waypoint_id}`,
          text: [`Waypoint ${feature.properties.waypoint_id}<br>Elevation: ${(zSurface / verticalRelief).toFixed(0)} m`],
          hovertemplate: '%{text}<extra></extra>',
        })
      } else if (feature.geometry.type === 'LineString') {
        // Add path line positioned directly on terrain surface
        const coords = feature.geometry.coordinates
        const lons = coords.map((c: number[]) => c[0])
        const lats = coords.map((c: number[]) => c[1])
        
        // Get exact elevations from terrain surface for each point
        const zs = coords.map((c: number[]) => getSurfaceElevation(c[0], c[1]))
        
        // Apply horizontal scale to coordinates
        const xGrid = terrainData.x as number[][]
        const yGrid = terrainData.y as number[][]
        const xCoords = xGrid[0] || []
        const yCoords = yGrid.map((row: number[]) => row[0]) || []
        const centerX = xCoords.length > 0 ? (xCoords[0] + xCoords[xCoords.length - 1]) / 2 : 0
        const centerY = yCoords.length > 0 ? (yCoords[0] + yCoords[yCoords.length - 1]) / 2 : 0
        const xScaled = lons.map((lon: number) => centerX + (lon - centerX) * horizontalScale)
        const yScaled = lats.map((lat: number) => centerY + (lat - centerY) * horizontalScale)
        
        traces.push({
          type: 'scatter3d',
          mode: 'lines',
          x: xScaled,
          y: yScaled,
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
      } as any, // Plotly expects this structure but TypeScript types may not match
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


