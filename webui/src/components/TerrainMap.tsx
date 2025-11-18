import { useEffect, useState, useRef } from 'react'
import { MapContainer, ImageOverlay, GeoJSON, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons in React-Leaflet
// Prevent yellow square from appearing when icons fail to load
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// Note: Yellow squares appear when Leaflet marker icons fail to load.
// We use circleMarker instead of default markers to avoid this issue.

interface TerrainMapProps {
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset?: string
  showSites?: boolean
  showWaypoints?: boolean
  relief?: number
  onSiteSelect?: (siteId: number) => void
  selectedSiteId?: number | null
}


export default function TerrainMap({
  roi,
  dataset = 'mola',
  showSites = false,
  showWaypoints = false,
  relief = 0,
  onSiteSelect,
  selectedSiteId,
}: TerrainMapProps) {
  const [demImageUrl, setDemImageUrl] = useState<string | null>(null)
  const [imageBounds, setImageBounds] = useState<[[number, number], [number, number]] | null>(null)
  const [sitesGeoJson, setSitesGeoJson] = useState<any>(null)
  const [waypointsGeoJson, setWaypointsGeoJson] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Use ref to track blob URL for cleanup
  const demImageUrlRef = useRef<string | null>(null)

  useEffect(() => {
    if (!roi) return

    const loadVisualizations = async () => {
      setLoading(true)
      setError(null)
      
      // Clean up previous image URL before loading new one
      if (demImageUrlRef.current) {
        URL.revokeObjectURL(demImageUrlRef.current)
        demImageUrlRef.current = null
      }
      setDemImageUrl(null)
      setImageBounds(null)
      
      try {
        // Calculate buffer to ensure DEM fills viewport and allows seamless panning
        // Use a moderate buffer to ensure the DEM extends beyond the viewport
        // This allows seamless panning and ensures the image always fills the area
        // Buffer of 1.5 = 150% extension (extends 75% in each direction)
        // Reduced from 3.0 to prevent timeouts and improve loading speed
        const buffer = 1.5 // Moderate extension for seamless panning
        
        // Load DEM image with large buffer for seamless panning and overflow
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
        const demUrl = `http://localhost:5000/api/v1/visualization/dem-image?dataset=${dataset}&roi=${roiStr}&width=2400&height=1600&colormap=terrain&relief=${relief}&buffer=${buffer}`
        
        console.log('Fetching DEM image from:', demUrl)
        // Add timeout to prevent hanging
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 150000) // 2.5 minute timeout
        
        let demResponse: Response
        try {
          demResponse = await fetch(demUrl, { signal: controller.signal })
          clearTimeout(timeoutId)
          console.log('DEM response status:', demResponse.status, demResponse.statusText)
        } catch (fetchError) {
          clearTimeout(timeoutId)
          if (fetchError instanceof Error && fetchError.name === 'AbortError') {
            const errorMsg = 'DEM loading timed out after 2.5 minutes. Try a smaller ROI or check server logs.'
            console.error(errorMsg)
            setError(errorMsg)
            setLoading(false)
            return
          }
          const errorMsg = fetchError instanceof Error ? fetchError.message : 'Failed to fetch DEM image'
          console.error('DEM fetch error:', errorMsg)
          setError(errorMsg)
          setLoading(false)
          return
        }
        
        if (demResponse.ok) {
          const blob = await demResponse.blob()
          console.log('DEM blob received:', { 
            size: blob.size, 
            type: blob.type,
            headers: {
              'content-type': demResponse.headers.get('content-type'),
              'content-length': demResponse.headers.get('content-length'),
            }
          })
          
          // Check if blob is valid image
          if (blob.size === 0) {
            console.error('DEM image blob is empty')
            throw new Error('Empty DEM image received')
          }
          
          // Verify it's actually an image by creating an Image object
          const testImg = new Image()
          const testUrl = URL.createObjectURL(blob)
          await new Promise((resolve, reject) => {
            testImg.onload = () => {
              console.log('DEM image verified:', { width: testImg.width, height: testImg.height })
              URL.revokeObjectURL(testUrl)
              resolve(null)
            }
            testImg.onerror = (e) => {
              console.error('DEM image failed to load as image:', e)
              URL.revokeObjectURL(testUrl)
              reject(new Error('Blob is not a valid image'))
            }
            testImg.src = testUrl
          })
          
          // Clean up previous URL if it exists
          if (demImageUrlRef.current) {
            URL.revokeObjectURL(demImageUrlRef.current)
            demImageUrlRef.current = null
          }
          
          const url = URL.createObjectURL(blob)
          console.log('Created object URL for DEM image:', url.substring(0, 50) + '...')
          demImageUrlRef.current = url
          setDemImageUrl(url)

          // Get bounds from response headers
          // Leaflet ImageOverlay expects bounds as [[south, west], [north, east]]
          // which is [[lat_min, lon_min], [lat_max, lon_max]]
          const left = parseFloat(demResponse.headers.get('X-Bounds-Left') || String(roi.lon_min))
          const right = parseFloat(demResponse.headers.get('X-Bounds-Right') || String(roi.lon_max))
          const bottom = parseFloat(demResponse.headers.get('X-Bounds-Bottom') || String(roi.lat_min))
          const top = parseFloat(demResponse.headers.get('X-Bounds-Top') || String(roi.lat_max))
          
          console.log('Parsed bounds from headers:', { left, right, bottom, top })
          console.log('ROI bounds:', { 
            lat_min: roi.lat_min, 
            lat_max: roi.lat_max, 
            lon_min: roi.lon_min, 
            lon_max: roi.lon_max 
          })
          
          // Validate bounds are reasonable lat/lon values
          // Latitude must be between -90 and 90
          // Longitude must be between -180 and 360
          const isValidLat = (val: number) => !isNaN(val) && -90 <= val && val <= 90
          const isValidLon = (val: number) => !isNaN(val) && -180 <= val && val <= 360
          
          // Validate and fix bounds order
          const latMin = Math.min(bottom, top)
          const latMax = Math.max(bottom, top)
          const lonMin = Math.min(left, right)
          const lonMax = Math.max(left, right)
          
          const boundsValid = isValidLat(latMin) && isValidLat(latMax) && 
                              isValidLon(lonMin) && isValidLon(lonMax) &&
                              latMin < latMax && lonMin < lonMax
          
          if (!boundsValid) {
            console.error('Invalid bounds from headers (not valid lat/lon), using ROI bounds', { 
              left, right, bottom, top,
              latMin, latMax, lonMin, lonMax,
              isValidLatMin: isValidLat(latMin),
              isValidLatMax: isValidLat(latMax),
              isValidLonMin: isValidLon(lonMin),
              isValidLonMax: isValidLon(lonMax)
            })
            // Ensure ROI bounds are also valid
            const roiLatMin = Math.min(roi.lat_min, roi.lat_max)
            const roiLatMax = Math.max(roi.lat_min, roi.lat_max)
            const roiLonMin = Math.min(roi.lon_min, roi.lon_max)
            const roiLonMax = Math.max(roi.lon_min, roi.lon_max)
            const bounds: [[number, number], [number, number]] = [
              [roiLatMin, roiLonMin], 
              [roiLatMax, roiLonMax]
            ]
            console.log('Using ROI bounds:', bounds)
            setImageBounds(bounds)
          } else {
            // Leaflet bounds format: [[south, west], [north, east]]
            // which is [[lat_min, lon_min], [lat_max, lon_max]]
            const bounds: [[number, number], [number, number]] = [[latMin, lonMin], [latMax, lonMax]]
            
            console.log('DEM image loaded successfully', { 
              url: url.substring(0, 50) + '...',
              blobSize: blob.size, 
              bounds,
              boundsFormat: '[[lat_min, lon_min], [lat_max, lon_max]]',
              center: [(latMin + latMax) / 2, (lonMin + lonMax) / 2],
              headers: {
                left: demResponse.headers.get('X-Bounds-Left'),
                right: demResponse.headers.get('X-Bounds-Right'),
                bottom: demResponse.headers.get('X-Bounds-Bottom'),
                top: demResponse.headers.get('X-Bounds-Top'),
              }
            })
            
            setImageBounds(bounds)
          }
        } else {
          // DEM request failed - get error details
          let errorText = ''
          try {
            const errorJson = await demResponse.json()
            errorText = JSON.stringify(errorJson, null, 2)
          } catch {
            try {
              errorText = await demResponse.text()
            } catch {
              errorText = `HTTP ${demResponse.status}: ${demResponse.statusText}`
            }
          }
          const errorMsg = `Failed to load DEM image: ${demResponse.status} ${demResponse.statusText}.\n${errorText}`
          console.error('Failed to load DEM image', {
            status: demResponse.status,
            statusText: demResponse.statusText,
            error: errorText,
            url: demUrl
          })
          setError(errorMsg)
          setLoading(false)
          return // Don't throw, just show error
        }

        // Load sites if requested - make it non-blocking
        if (showSites) {
          // Load sites in parallel, don't block on errors
          fetch('http://localhost:5000/api/v1/visualization/sites-geojson')
            .then(async (sitesResponse) => {
              if (sitesResponse.ok) {
                const geojson = await sitesResponse.json()
                console.log('Sites GeoJSON loaded:', {
                  featureCount: geojson.features?.length,
                  sampleFeature: geojson.features?.[0],
                  sampleCoords: geojson.features?.[0]?.geometry?.coordinates
                })
                setSitesGeoJson(geojson)
              } else {
                console.warn('Failed to load sites:', sitesResponse.status, sitesResponse.statusText)
                // Don't set error, just log - sites are optional
              }
            })
            .catch((e) => {
              console.warn('Failed to load sites:', e)
              // Don't set error, just log - sites are optional
            })
        } else {
          // Clear sites if not requested
          setSitesGeoJson(null)
        }

        // Load waypoints if requested - make it non-blocking with timeout
        if (showWaypoints) {
          // Load waypoints in parallel, don't block on errors
          fetch('http://localhost:5000/api/v1/visualization/waypoints-geojson')
            .then(async (waypointsResponse) => {
              if (waypointsResponse.ok) {
                const geojson = await waypointsResponse.json()
                console.log('Waypoints GeoJSON loaded:', {
                  featureCount: geojson.features?.length,
                  sampleFeature: geojson.features?.[0],
                  sampleCoords: geojson.features?.[0]?.geometry?.coordinates,
                  lineStringCoords: geojson.features?.find((f: any) => f.geometry?.type === 'LineString')?.geometry?.coordinates
                })
                setWaypointsGeoJson(geojson)
              } else {
                console.warn('Failed to load waypoints:', waypointsResponse.status, waypointsResponse.statusText)
                // Don't set error, just log - waypoints are optional
              }
            })
            .catch((e) => {
              console.warn('Failed to load waypoints:', e)
              // Don't set error, just log - waypoints are optional
            })
        } else {
          // Clear waypoints if not requested
          setWaypointsGeoJson(null)
        }
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error occurred'
        console.error('Failed to load visualizations:', error)
        setError(errorMsg)
      } finally {
        setLoading(false)
      }
    }

    loadVisualizations()
    
    // Cleanup function to revoke blob URL when component unmounts or dependencies change
    return () => {
      // Cancel any pending fetch requests
      // (AbortController is handled in the try/catch)
      
      // Note: Don't revoke URL immediately - let it be cleaned up on next render
      // Revoking too early can cause the image to disappear before new one loads
      // The ref will be cleaned up when a new image is loaded
    }
  }, [roi, dataset, showSites, showWaypoints, relief])
  
  // Separate cleanup effect for when component unmounts
  useEffect(() => {
    return () => {
      // Only revoke on unmount, not on dependency changes
      if (demImageUrlRef.current) {
        URL.revokeObjectURL(demImageUrlRef.current)
        demImageUrlRef.current = null
      }
    }
  }, [])

  // Component to fit map bounds when image bounds change
  function FitBounds() {
    const map = useMap()
    useEffect(() => {
      if (imageBounds && demImageUrl) {
        console.log('FitBounds: Fitting map to image bounds', { imageBounds, hasUrl: !!demImageUrl })
        // Small delay to ensure image overlay is rendered first
        const timer = setTimeout(() => {
          try {
            // Validate bounds before fitting
            const [southWest, northEast] = imageBounds
            if (southWest && northEast && 
                Array.isArray(southWest) && Array.isArray(northEast) &&
                southWest.length === 2 && northEast.length === 2) {
              // Create LatLngBounds object
              const bounds = L.latLngBounds(southWest, northEast)
              if (bounds.isValid()) {
                // Fit map to image bounds with minimal padding to fill entire area
                map.fitBounds(bounds, {
                  padding: [0, 0], // No padding - fill entire rendering area
                  maxZoom: 12
                })
                console.log('FitBounds: Map fitted successfully', { bounds: bounds.toBBoxString() })
              } else {
                console.warn('FitBounds: Invalid bounds object', { imageBounds })
              }
            } else {
              console.warn('FitBounds: Invalid bounds format', { imageBounds })
            }
          } catch (e) {
            console.warn('Failed to fit bounds:', e)
          }
        }, 200) // Increased delay to ensure ImageOverlay is fully initialized
        return () => clearTimeout(timer)
      }
    }, [map, imageBounds, demImageUrl])
    return null
  }

  // Debug effect to log when image URL or bounds change
  useEffect(() => {
    if (demImageUrl && imageBounds) {
      console.log('DEM image state updated:', {
        hasUrl: !!demImageUrl,
        hasBounds: !!imageBounds,
        bounds: imageBounds,
        urlPreview: demImageUrl.substring(0, 50) + '...'
      })
    } else {
      console.log('DEM image state:', {
        hasUrl: !!demImageUrl,
        hasBounds: !!imageBounds
      })
    }
  }, [demImageUrl, imageBounds])

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
    weight: 1.5, // Scaled down from 3
    opacity: 0.8,
  }

  const pointToLayer = (feature: any, latlng: L.LatLng) => {
    // Debug: log coordinates to verify they're correct
    const coords = feature.geometry?.coordinates
    if (coords) {
      console.log('Rendering marker:', {
        type: feature.properties.waypoint_id ? 'waypoint' : 'site',
        id: feature.properties.waypoint_id || feature.properties.site_id,
        coords: coords,
        latlng: [latlng.lat, latlng.lng],
        isOrigin: coords[0] === 0 && coords[1] === 0
      })
      
      // Skip invalid coordinates (0,0 or out of bounds) to prevent yellow squares
      if (coords[0] === 0 && coords[1] === 0) {
        console.warn('Warning: Feature has coordinates at origin (0,0), skipping marker:', {
          feature: feature.properties,
          geometry: feature.geometry
        })
        // Return a transparent marker that won't show
        return L.marker(latlng, {
          icon: L.divIcon({
            className: 'hidden-marker',
            html: '',
            iconSize: [0, 0],
          })
        })
      }
      
      // Validate coordinates are within reasonable bounds
      if (!(-180 <= coords[0] && coords[0] <= 360) || !(-90 <= coords[1] && coords[1] <= 90)) {
        console.warn('Warning: Feature has invalid coordinates, skipping marker:', {
          coords,
          feature: feature.properties
        })
        return L.marker(latlng, {
          icon: L.divIcon({
            className: 'hidden-marker',
            html: '',
            iconSize: [0, 0],
          })
        })
      }
    }
    
    if (feature.properties.waypoint_id) {
      // Waypoint marker - use circleMarker to avoid icon loading issues, scaled down
      return L.circleMarker(latlng, {
        radius: 3, // Scaled down from 6
        fillColor: '#ff0000',
        color: '#ffffff',
        weight: 1.5, // Scaled down from 2
        opacity: 1,
        fillOpacity: 0.8,
      })
    } else {
      // Site marker - use circleMarker to avoid icon loading issues
      const siteId = feature.properties?.site_id
      const isSelected = selectedSiteId !== null && selectedSiteId !== undefined && siteId === selectedSiteId
      
      return L.circleMarker(latlng, {
        radius: isSelected ? 12 : 8,
        fillColor: isSelected ? '#ffff00' : '#00ff00',
        color: isSelected ? '#ff0000' : '#ffffff',
        weight: isSelected ? 3 : 2,
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
      
      // Add click handler for site selection
      if (onSiteSelect) {
        layer.on('click', () => {
          onSiteSelect(props.site_id)
        })
        // Add cursor pointer style
        if (layer instanceof L.CircleMarker || layer instanceof L.Marker) {
          layer.setStyle({ cursor: 'pointer' })
        }
      }
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
      {error && (
        <div className="mb-4 p-4 bg-red-900/50 border border-red-700 rounded-md">
          <p className="text-red-300 font-semibold">Error loading visualization</p>
          <p className="text-red-400 text-sm mt-1">{error}</p>
          <p className="text-red-400 text-xs mt-2">Check browser console for details. Make sure the backend server is running on port 5000.</p>
        </div>
      )}
      <div className="h-[600px] w-full rounded-md overflow-hidden border border-gray-700 bg-gray-900">
        <MapContainer
          center={center}
          zoom={6}
          style={{ height: '100%', width: '100%', backgroundColor: '#1a1a1a' }}
          crs={L.CRS.EPSG4326} // Use standard lat/lon CRS
          minZoom={2}
          maxZoom={15}
          zoomControl={true}
          worldCopyJump={false}
          boundsOptions={{ padding: [0, 0] }} // No padding when fitting bounds
        >
          <FitBounds />
          
          {/* DEM Image Overlay - Primary visualization for Mars terrain */}
          {demImageUrl && imageBounds && (() => {
            // Validate bounds before rendering
            const [southWest, northEast] = imageBounds
            const boundsValid = 
              Array.isArray(southWest) && southWest.length === 2 &&
              Array.isArray(northEast) && northEast.length === 2 &&
              typeof southWest[0] === 'number' && typeof southWest[1] === 'number' &&
              typeof northEast[0] === 'number' && typeof northEast[1] === 'number' &&
              southWest[0] < northEast[0] && // lat_min < lat_max
              southWest[1] < northEast[1] && // lon_min < lon_max
              !isNaN(southWest[0]) && !isNaN(southWest[1]) &&
              !isNaN(northEast[0]) && !isNaN(northEast[1]) &&
              isFinite(southWest[0]) && isFinite(southWest[1]) &&
              isFinite(northEast[0]) && isFinite(northEast[1])
            
            if (!boundsValid) {
              console.error('Invalid image bounds, skipping ImageOverlay', { imageBounds })
              return null
            }
            
            // Create stable key that includes URL to force re-render when image changes
            // Use a portion of the URL (blob URLs are unique) plus bounds
            const urlHash = demImageUrl.substring(0, 30).replace(/[^a-zA-Z0-9]/g, '')
            const boundsKey = `${imageBounds[0][0].toFixed(4)}-${imageBounds[0][1].toFixed(4)}-${imageBounds[1][0].toFixed(4)}-${imageBounds[1][1].toFixed(4)}`
            
            // Ensure bounds are in correct format: [[south, west], [north, east]]
            const leafletBounds: [[number, number], [number, number]] = [
              [southWest[0], southWest[1]], // [lat_min, lon_min]
              [northEast[0], northEast[1]]  // [lat_max, lon_max]
            ]
            
            return (
              <ImageOverlay 
                key={`dem-${urlHash}-${boundsKey}`}
                url={demImageUrl} 
                bounds={leafletBounds}
                opacity={1.0}
                zIndex={1}
                interactive={false}
                eventHandlers={{
                  load: () => {
                    console.log('ImageOverlay loaded successfully', { 
                      bounds: leafletBounds,
                      url: demImageUrl.substring(0, 50) + '...'
                    })
                    setError(null) // Clear any previous errors
                  },
                  error: (e) => {
                    console.error('ImageOverlay failed to load:', e)
                    const errorMsg = 'Failed to display DEM image. The image may be corrupted or bounds may be invalid.'
                    console.error('ImageOverlay error details:', {
                      url: demImageUrl.substring(0, 50) + '...',
                      bounds: leafletBounds,
                      error: e
                    })
                    setError(errorMsg)
                  }
                }}
                crossOrigin="anonymous"
              />
            )
          })()}
          
          {/* Debug: Show bounds info if image not loading */}
          {demImageUrl && imageBounds && !loading && (
            <div style={{ display: 'none' }}>
              {/* Hidden debug info */}
              <div>Image URL: {demImageUrl.substring(0, 50)}...</div>
              <div>Bounds: {JSON.stringify(imageBounds)}</div>
            </div>
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



