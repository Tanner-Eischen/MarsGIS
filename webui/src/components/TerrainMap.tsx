import { useEffect, useRef, useState } from 'react'
import { MapContainer, TileLayer, ImageOverlay, Marker, Popup, GeoJSON } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons in React-Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

interface TerrainMapProps {
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset?: string
  showSites?: boolean
  showWaypoints?: boolean
  relief?: number
}


export default function TerrainMap({
  roi,
  dataset = 'mola',
  showSites = false,
  showWaypoints = false,
  relief = 0,
}: TerrainMapProps) {
  const [demImageUrl, setDemImageUrl] = useState<string | null>(null)
  const [imageBounds, setImageBounds] = useState<[[number, number], [number, number]] | null>(null)
  const [sitesGeoJson, setSitesGeoJson] = useState<any>(null)
  const [waypointsGeoJson, setWaypointsGeoJson] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!roi) return

    const loadVisualizations = async () => {
      setLoading(true)
      try {
        // Load DEM image with buffer for seamless panning
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
        const demUrl = `http://localhost:5000/api/v1/visualization/dem-image?dataset=${dataset}&roi=${roiStr}&width=2400&height=1600&colormap=terrain&relief=${relief}&buffer=1.0`
        
        const demResponse = await fetch(demUrl)
        if (demResponse.ok) {
          const blob = await demResponse.blob()
          const url = URL.createObjectURL(blob)
          setDemImageUrl(url)

          // Get bounds from response headers
          const left = parseFloat(demResponse.headers.get('X-Bounds-Left') || String(roi.lon_min))
          const right = parseFloat(demResponse.headers.get('X-Bounds-Right') || String(roi.lon_max))
          const bottom = parseFloat(demResponse.headers.get('X-Bounds-Bottom') || String(roi.lat_min))
          const top = parseFloat(demResponse.headers.get('X-Bounds-Top') || String(roi.lat_max))
          
          setImageBounds([[bottom, left], [top, right]])
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
        console.error('Failed to load visualizations:', error)
      } finally {
        setLoading(false)
      }
    }

    loadVisualizations()
  }, [roi, dataset, showSites, showWaypoints, relief])

  // Default center (Mars equator)
  const center: [number, number] = roi 
    ? [(roi.lat_min + roi.lat_max) / 2, (roi.lon_min + roi.lon_max) / 2]
    : [0, 180]

  const sitesStyle = {
    color: '#00ff00',
    weight: 2,
    opacity: 0.8,
    fillColor: '#00ff00',
    fillOpacity: 0.3,
  }

  const waypointsStyle = {
    color: '#ff0000',
    weight: 3,
    opacity: 0.9,
  }

  const pointToLayer = (feature: any, latlng: L.LatLng) => {
    if (feature.properties.waypoint_id) {
      // Waypoint marker
      return L.circleMarker(latlng, {
        radius: 6,
        fillColor: '#ff0000',
        color: '#ffffff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.8,
      })
    } else {
      // Site marker
      return L.circleMarker(latlng, {
        radius: 8,
        fillColor: '#00ff00',
        color: '#ffffff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.8,
      })
    }
  }

  const onEachFeature = (feature: any, layer: L.Layer) => {
    if (feature.properties && feature.properties.site_id) {
      const props = feature.properties
      const popupContent = `
        <div>
          <strong>Site ${props.site_id}</strong><br/>
          Rank: ${props.rank}<br/>
          Area: ${props.area_km2.toFixed(2)} km²<br/>
          Suitability: ${(props.suitability_score * 100).toFixed(1)}%<br/>
          Slope: ${props.mean_slope_deg.toFixed(2)}°<br/>
          Elevation: ${props.mean_elevation_m.toFixed(0)} m
        </div>
      `
      layer.bindPopup(popupContent)
    } else if (feature.properties && feature.properties.waypoint_id) {
      const props = feature.properties
      const popupContent = `
        <div>
          <strong>Waypoint ${props.waypoint_id}</strong><br/>
          Tolerance: ${props.tolerance_meters.toFixed(1)} m
        </div>
      `
      layer.bindPopup(popupContent)
    }
  }

  if (!roi) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-400">Select a region of interest to view terrain visualization</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      {loading && (
        <div className="mb-4 text-center">
          <p className="text-gray-300">Loading terrain visualization...</p>
        </div>
      )}
      <div className="h-[600px] w-full rounded-md overflow-hidden border border-gray-700">
        <MapContainer
          center={center}
          zoom={6}
          style={{ height: '100%', width: '100%' }}
          crs={L.CRS.Simple} // Use simple CRS for custom image overlay
          maxBounds={imageBounds ? imageBounds : undefined}
          minZoom={3}
          maxZoom={10}
        >
          {imageBounds && demImageUrl && (
            <ImageOverlay 
              url={demImageUrl} 
              bounds={imageBounds} 
              opacity={0.8}
              boundsOptions={{ padding: [0, 0] }}
            />
          )}
          
          {sitesGeoJson && (
            <GeoJSON
              data={sitesGeoJson}
              style={sitesStyle}
              pointToLayer={pointToLayer}
              onEachFeature={onEachFeature}
            />
          )}
          
          {waypointsGeoJson && (
            <GeoJSON
              data={waypointsGeoJson}
              style={waypointsStyle}
              pointToLayer={pointToLayer}
              onEachFeature={onEachFeature}
            />
          )}
        </MapContainer>
      </div>
      <div className="mt-4 text-sm text-gray-400">
        <p>DEM imagery cropped to ROI. Green markers = construction sites, Red markers = navigation waypoints</p>
      </div>
    </div>
  )
}



