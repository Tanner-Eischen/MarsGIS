import { useState, useEffect, useRef } from 'react'
import L from 'leaflet'
import { apiFetch, apiUrl } from '../lib/apiBase'

interface Roi {
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
}

interface OverlayOptions {
  colormap?: string
  relief?: number
  sunAzimuth?: number
  sunAltitude?: number
  width?: number
  height?: number
  buffer?: number
  marsSol?: number
  season?: string
  dustStormPeriod?: string
}

interface ImageState {
  url: string | null
  bounds: L.LatLngBounds | null
  loading: boolean
  error: string | null
}

// Custom hook to fetch and manage the DEM image
export function useDemImage(roi: Roi | null, dataset: string, relief: number): ImageState {
  const [demImage, setDemImage] = useState<ImageState>({
    url: null,
    bounds: null,
    loading: false,
    error: null,
  })
  const imageUrlRef = useRef<string | null>(null)

  useEffect(() => {
    return () => {
      if (imageUrlRef.current) {
        URL.revokeObjectURL(imageUrlRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!roi) return
    let cancelled = false
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 30000)

    const fetchDemImage = async () => {
      setDemImage((prev) => ({ ...prev, loading: true, error: null }))

      try {
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
        const url = apiUrl(
          `/visualization/dem-image?dataset=${dataset}&roi=${roiStr}&width=1200&height=800&colormap=terrain&relief=${relief}&buffer=0.25`
        )

        const response = await fetch(url, { signal: controller.signal })

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`Failed to load DEM image: ${response.status} ${errorText}`)
        }

        const blob = await response.blob()
        if (blob.size === 0) {
          throw new Error('Empty DEM image received')
        }

        if (imageUrlRef.current) {
          URL.revokeObjectURL(imageUrlRef.current)
        }
        const newUrl = URL.createObjectURL(blob)
        imageUrlRef.current = newUrl

        const left = parseFloat(response.headers.get('X-Bounds-Left') || String(roi.lon_min))
        const right = parseFloat(response.headers.get('X-Bounds-Right') || String(roi.lon_max))
        const bottom = parseFloat(response.headers.get('X-Bounds-Bottom') || String(roi.lat_min))
        const top = parseFloat(response.headers.get('X-Bounds-Top') || String(roi.lat_max))

        const bounds = L.latLngBounds([bottom, left], [top, right])
        if (!bounds.isValid()) {
          const fallbackBounds = L.latLngBounds(
            [roi.lat_min, roi.lon_min],
            [roi.lat_max, roi.lon_max]
          )
          if (cancelled) return
          setDemImage({ url: newUrl, bounds: fallbackBounds, loading: false, error: null })
        } else {
          if (cancelled) return
          setDemImage({ url: newUrl, bounds, loading: false, error: null })
        }
      } catch (error: unknown) {
        if (cancelled) return
        if (error instanceof Error && error.name === 'AbortError') {
          setDemImage((prev) => ({ ...prev, loading: false, error: 'DEM loading timed out.' }))
        } else {
          const message = error instanceof Error ? error.message : 'Unknown error'
          setDemImage((prev) => ({ ...prev, loading: false, error: message }))
        }
      } finally {
        clearTimeout(timeoutId)
      }
    }

    fetchDemImage()
    return () => {
      cancelled = true
      controller.abort()
      clearTimeout(timeoutId)
    }
  }, [roi?.lat_min, roi?.lat_max, roi?.lon_min, roi?.lon_max, dataset, relief])

  return demImage
}

function useGeoJson(path: string, enabled: boolean, refreshKey?: number) {
  const [data, setData] = useState<any>(null)

  useEffect(() => {
    if (!enabled) {
      setData(null)
      return
    }

    const fetchData = async () => {
      try {
        const response = await apiFetch(path)
        if (!response.ok) {
          return
        }
        const geojsonData = await response.json()
        setData(geojsonData)
      } catch {
        // Best-effort overlay fetch; map UI handles missing data.
      }
    }

    fetchData()
  }, [path, enabled, refreshKey])

  return data
}

export function useSitesGeoJson(enabled: boolean) {
  return useGeoJson('/visualization/sites-geojson', enabled)
}

export function useWaypointsGeoJson(enabled: boolean, refreshKey?: number) {
  return useGeoJson('/visualization/waypoints-geojson', enabled, refreshKey)
}

// Unified overlay hook
// Note: For caching support, use OverlayLayerContext directly in components
export function useOverlayImage(
  roi: Roi | null,
  dataset: string,
  overlayType: string | null,
  options: OverlayOptions = {}
): ImageState {
  const [overlayImage, setOverlayImage] = useState<ImageState>({
    url: null,
    bounds: null,
    loading: false,
    error: null,
  })
  const imageUrlRef = useRef<string | null>(null)
  const lastRequestKeyRef = useRef<string | null>(null)
  const lastRequestAtRef = useRef<number>(0)

  useEffect(() => {
    return () => {
      if (imageUrlRef.current) {
        URL.revokeObjectURL(imageUrlRef.current)
      }
    }
  }, [])

  const {
    colormap = 'terrain',
    relief = 0,
    sunAzimuth = 315,
    sunAltitude = 45,
    width = 1200,
    height = 800,
    buffer = 0.25,
    marsSol,
    season,
    dustStormPeriod,
  } = options

  useEffect(() => {
    if (!roi || !overlayType) return
    let cancelled = false
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 60000)

    const fetchOverlayImage = async () => {
      setOverlayImage((prev) => ({ ...prev, loading: true, error: null }))

      try {
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
        const params = new URLSearchParams({
          overlay_type: overlayType,
          dataset,
          roi: roiStr,
          colormap,
          relief: String(relief),
          sun_azimuth: String(sunAzimuth),
          sun_altitude: String(sunAltitude),
          width: String(width),
          height: String(height),
          buffer: String(buffer),
        })

        if (marsSol !== undefined) params.append('mars_sol', String(marsSol))
        if (season) params.append('season', season)
        if (dustStormPeriod) params.append('dust_storm_period', dustStormPeriod)

        const requestKey = params.toString()
        const now = Date.now()
        if (lastRequestKeyRef.current === requestKey && now - lastRequestAtRef.current < 750) {
          setOverlayImage((prev) => ({ ...prev, loading: false }))
          return
        }
        lastRequestKeyRef.current = requestKey
        lastRequestAtRef.current = now

        const url = apiUrl(`/visualization/overlay?${params.toString()}`)
        const response = await fetch(url, { signal: controller.signal })

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`Failed to load overlay image: ${response.status} ${errorText}`)
        }

        const blob = await response.blob()
        if (blob.size === 0) {
          throw new Error('Empty overlay image received')
        }

        if (imageUrlRef.current) {
          URL.revokeObjectURL(imageUrlRef.current)
        }
        const newUrl = URL.createObjectURL(blob)
        imageUrlRef.current = newUrl

        const left = parseFloat(response.headers.get('X-Bounds-Left') || String(roi.lon_min))
        const right = parseFloat(response.headers.get('X-Bounds-Right') || String(roi.lon_max))
        const bottom = parseFloat(response.headers.get('X-Bounds-Bottom') || String(roi.lat_min))
        const top = parseFloat(response.headers.get('X-Bounds-Top') || String(roi.lat_max))

        const bounds = L.latLngBounds([bottom, left], [top, right])
        if (!bounds.isValid()) {
          const fallbackBounds = L.latLngBounds(
            [roi.lat_min, roi.lon_min],
            [roi.lat_max, roi.lon_max]
          )
          if (cancelled) return
          setOverlayImage({ url: newUrl, bounds: fallbackBounds, loading: false, error: null })
        } else {
          if (cancelled) return
          setOverlayImage({ url: newUrl, bounds, loading: false, error: null })
        }
      } catch (error: unknown) {
        if (cancelled) return
        if (error instanceof Error && error.name === 'AbortError') {
          setOverlayImage((prev) => ({ ...prev, loading: false, error: 'Overlay loading timed out.' }))
        } else {
          const message = error instanceof Error ? error.message : 'Unknown error'
          setOverlayImage((prev) => ({ ...prev, loading: false, error: message }))
        }
      } finally {
        clearTimeout(timeoutId)
      }
    }

    fetchOverlayImage()
    return () => {
      cancelled = true
      controller.abort()
      clearTimeout(timeoutId)
    }
  }, [
    roi?.lat_min,
    roi?.lat_max,
    roi?.lon_min,
    roi?.lon_max,
    dataset,
    overlayType,
    colormap,
    relief,
    sunAzimuth,
    sunAltitude,
    width,
    height,
    buffer,
    marsSol,
    season,
    dustStormPeriod,
  ])

  return overlayImage
}
