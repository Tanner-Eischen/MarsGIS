import { useEffect, useState, useRef } from 'react'
import { MapContainer, ImageOverlay, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

interface SolarHeatmapProps {
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset?: string
  solarPotentialMap: number[][] | null
  shape: { rows: number; cols: number } | null
  loading?: boolean
}

function FitBounds({ bounds }: { bounds: [[number, number], [number, number]] }) {
  const map = useMap()
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [20, 20] })
    }
  }, [map, bounds])
  return null
}

export default function SolarHeatmap({
  roi,
  dataset = 'mola',
  solarPotentialMap,
  shape,
  loading = false,
}: SolarHeatmapProps) {
  const [heatmapImageUrl, setHeatmapImageUrl] = useState<string | null>(null)
  const [imageBounds, setImageBounds] = useState<[[number, number], [number, number]] | null>(null)
  const [hoverValue, setHoverValue] = useState<number | null>(null)
  const [hoverPosition, setHoverPosition] = useState<{ x: number; y: number } | null>(null)
  const heatmapImageUrlRef = useRef<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // Calculate center and bounds
  const center: [number, number] = [
    (roi.lat_min + roi.lat_max) / 2,
    (roi.lon_min + roi.lon_max) / 2,
  ]

  const bounds: [[number, number], [number, number]] = [
    [roi.lat_min, roi.lon_min],
    [roi.lat_max, roi.lon_max],
  ]

  // Generate heatmap image from solar potential data
  useEffect(() => {
    if (!solarPotentialMap || !shape) {
      setHeatmapImageUrl(null)
      setImageBounds(null)
      return
    }

    // Clean up previous image URL
    if (heatmapImageUrlRef.current) {
      URL.revokeObjectURL(heatmapImageUrlRef.current)
      heatmapImageUrlRef.current = null
    }

    try {
      const canvas = document.createElement('canvas')
      canvas.width = shape.cols
      canvas.height = shape.rows
      const ctx = canvas.getContext('2d')
      
      if (!ctx) {
        console.error('Failed to get canvas context')
        return
      }

      // Create image data
      const imageData = ctx.createImageData(shape.cols, shape.rows)
      const data = imageData.data

      // Color scale: blue (low) -> yellow -> red (high)
      const getColor = (value: number) => {
        // Clamp value to [0, 1]
        const v = Math.max(0, Math.min(1, value))
        
        if (v < 0.5) {
          // Blue to yellow (0 -> 0.5)
          const t = v * 2
          return {
            r: Math.floor(t * 255),
            g: Math.floor(t * 255),
            b: Math.floor((1 - t) * 255),
            a: 200
          }
        } else {
          // Yellow to red (0.5 -> 1.0)
          const t = (v - 0.5) * 2
          return {
            r: 255,
            g: Math.floor((1 - t) * 255),
            b: 0,
            a: 200
          }
        }
      }

      // Fill image data
      for (let row = 0; row < shape.rows; row++) {
        for (let col = 0; col < shape.cols; col++) {
          const value = solarPotentialMap[row]?.[col] ?? 0
          const color = getColor(value)
          const idx = (row * shape.cols + col) * 4
          data[idx] = color.r
          data[idx + 1] = color.g
          data[idx + 2] = color.b
          data[idx + 3] = color.a
        }
      }

      ctx.putImageData(imageData, 0, 0)
      
      // Convert to blob URL
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob)
          heatmapImageUrlRef.current = url
          setHeatmapImageUrl(url)
          setImageBounds(bounds)
          canvasRef.current = canvas
        }
      }, 'image/png')
    } catch (error) {
      console.error('Failed to generate heatmap image:', error)
      setHeatmapImageUrl(null)
    }
  }, [solarPotentialMap, shape, bounds])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (heatmapImageUrlRef.current) {
        URL.revokeObjectURL(heatmapImageUrlRef.current)
      }
    }
  }, [])

  // Handle mouse move for hover value
  const handleMouseMove = (e: L.LeafletMouseEvent) => {
    if (!solarPotentialMap || !shape || !canvasRef.current || !containerRef.current) return

    const map = e.target
    const point = map.mouseEventToContainerPoint(e.originalEvent)
    const bounds = map.getBounds()
    
    // Convert screen coordinates to data coordinates
    const lat = bounds.getNorth() - (point.y / map.getSize().y) * (bounds.getNorth() - bounds.getSouth())
    const lon = bounds.getWest() + (point.x / map.getSize().x) * (bounds.getEast() - bounds.getWest())
    
    // Convert lat/lon to row/col
    const row = Math.floor((1 - (lat - roi.lat_min) / (roi.lat_max - roi.lat_min)) * shape.rows)
    const col = Math.floor((lon - roi.lon_min) / (roi.lon_max - roi.lon_min) * shape.cols)
    
    if (row >= 0 && row < shape.rows && col >= 0 && col < shape.cols) {
      const value = solarPotentialMap[row]?.[col] ?? 0
      setHoverValue(value)
      
      // Calculate position relative to container
      const containerRect = containerRef.current.getBoundingClientRect()
      const mouseX = e.originalEvent.clientX - containerRect.left
      const mouseY = e.originalEvent.clientY - containerRect.top
      
      // Constrain tooltip to container bounds (with padding)
      const tooltipWidth = 180 // Approximate tooltip width
      const tooltipHeight = 40
      const padding = 10
      
      let tooltipX = mouseX
      let tooltipY = mouseY - tooltipHeight - padding
      
      // Constrain horizontally
      if (tooltipX - tooltipWidth / 2 < padding) {
        tooltipX = tooltipWidth / 2 + padding
      } else if (tooltipX + tooltipWidth / 2 > containerRect.width - padding) {
        tooltipX = containerRect.width - tooltipWidth / 2 - padding
      }
      
      // Constrain vertically
      if (tooltipY < padding) {
        tooltipY = mouseY + padding
      } else if (tooltipY + tooltipHeight > containerRect.height - padding) {
        tooltipY = containerRect.height - tooltipHeight - padding
      }
      
      setHoverPosition({ x: tooltipX, y: tooltipY })
    }
  }

  // const handleMouseLeave = () => {
  //   setHoverValue(null)
  //   setHoverPosition(null)
  // }

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="h-[600px] w-full rounded-md flex items-center justify-center bg-gray-900">
          <p className="text-gray-300">Loading solar potential analysis...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div ref={containerRef} className="h-[600px] w-full rounded-md overflow-hidden border border-gray-700 bg-gray-900 relative" style={{ position: 'relative' }}>
        <MapContainer
          center={center}
          zoom={6}
          style={{ height: '100%', width: '100%', backgroundColor: '#1a1a1a', minHeight: '600px' }}
          crs={L.CRS.EPSG4326}
          minZoom={2}
          maxZoom={15}
          zoomControl={true}
          worldCopyJump={false}
          key={`solar-map-${roi.lat_min}-${roi.lat_max}-${roi.lon_min}-${roi.lon_max}-${dataset}`} // Force remount on ROI/dataset change
        >
          <FitBounds bounds={bounds} />
          
          {heatmapImageUrl && imageBounds && (
            <ImageOverlay
              url={heatmapImageUrl}
              bounds={imageBounds}
              opacity={0.7}
              eventHandlers={{
                mousemove: handleMouseMove
              }}
            />
          )}
        </MapContainer>
        
        {hoverValue !== null && hoverPosition && (
          <div
            className="absolute pointer-events-none bg-gray-900 bg-opacity-90 text-white px-3 py-2 rounded shadow-lg text-sm z-50 whitespace-nowrap"
            style={{
              left: `${hoverPosition.x}px`,
              top: `${hoverPosition.y}px`,
              transform: 'translateX(-50%)',
              maxWidth: '90%',
            }}
          >
            Solar Potential: {(hoverValue * 100).toFixed(1)}%
          </div>
        )}
      </div>
      
      <div className="mt-4 flex items-center justify-between">
        <div className="text-sm text-gray-400">
          <p>Solar Potential Heatmap: Blue (low) → Yellow → Red (high)</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-blue-500"></div>
            <span className="text-xs text-gray-400">Low (0%)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-yellow-500"></div>
            <span className="text-xs text-gray-400">Medium (50%)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-red-500"></div>
            <span className="text-xs text-gray-400">High (100%)</span>
          </div>
        </div>
      </div>
    </div>
  )
}

