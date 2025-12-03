import { useState, useCallback, useRef, useEffect } from 'react'
import { getOverlayDefinition, type OverlayType, type MarsDataset } from '../config/marsDataSources'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1'

export interface OverlayLayer {
  name: OverlayType
  loaded: boolean
  loading: boolean
  data: OverlayLayerData | null
  error: string | null
}

export interface OverlayLayerData {
  url: string
  bounds: { left: number; right: number; bottom: number; top: number }
  timestamp: string
  metadata: {
    overlayType: OverlayType
    dataset: MarsDataset
    roi: string
    colormap?: string
    relief?: number
    sunAzimuth?: number
    sunAltitude?: number
    width: number
    height: number
    temporalParams?: Record<string, any>
  }
  blobSize: number
}

interface CacheEntry {
  data: OverlayLayerData
  accessTime: number
}

interface UseOverlayLayerManagerOptions {
  maxCacheSize?: number
}

/**
 * Generate cache key for overlay layer
 */
function generateCacheKey(
  overlayType: OverlayType,
  dataset: MarsDataset,
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
  options: {
    colormap?: string
    relief?: number
    sunAzimuth?: number
    sunAltitude?: number
    width?: number
    height?: number
    marsSol?: number
    season?: string
    dustStormPeriod?: string
  } = {}
): string {
  const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
  const parts = [
    overlayType,
    dataset,
    roiStr,
    options.colormap || 'terrain',
    String(options.relief || 0),
    String(options.sunAzimuth || 315),
    String(options.sunAltitude || 45),
    String(options.width || 2400),
    String(options.height || 1600)
  ]
  
  // Add temporal parameters if present
  if (options.marsSol !== undefined) parts.push(`sol${options.marsSol}`)
  if (options.season) parts.push(`season${options.season}`)
  if (options.dustStormPeriod) parts.push(`storm${options.dustStormPeriod}`)
  
  return parts.join('|')
}

/**
 * Hook for managing overlay layers with LRU caching
 */
export function useOverlayLayerManager(options: UseOverlayLayerManagerOptions = {}) {
  const { maxCacheSize = 5 } = options
  
  const [layers, setLayers] = useState<Map<OverlayType, OverlayLayer>>(new Map())
  const cacheRef = useRef<Map<string, CacheEntry>>(new Map())
  const accessOrderRef = useRef<string[]>([])
  const [activeLayerName, setActiveLayerName] = useState<OverlayType | null>(null)
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map())

  // Initialize layers from definitions
  useEffect(() => {
    if (layers.size === 0) {
      const map = new Map<OverlayType, OverlayLayer>()
      const definitions = [
        'elevation', 'solar', 'dust', 'hillshade', 'slope', 
        'aspect', 'roughness', 'tri'
      ] as OverlayType[]
      
      definitions.forEach(name => {
        map.set(name, {
          name,
          loaded: false,
          loading: false,
          data: null,
          error: null
        })
      })
      
      setLayers(map)
    }
  }, [layers.size])

  /**
   * Update access order for LRU cache
   */
  const updateAccessOrder = useCallback((cacheKey: string) => {
    const order = accessOrderRef.current
    const index = order.indexOf(cacheKey)
    if (index > -1) {
      order.splice(index, 1)
    }
    order.push(cacheKey)
  }, [])

  /**
   * Evict least recently used cache entry
   */
  const evictLRU = useCallback(() => {
    const cache = cacheRef.current
    const order = accessOrderRef.current
    
    if (order.length === 0) return
    
    const lruKey = order[0]
    const entry = cache.get(lruKey)
    
    if (entry) {
      // Revoke blob URL
      if (entry.data.url.startsWith('blob:')) {
        URL.revokeObjectURL(entry.data.url)
      }
      
      cache.delete(lruKey)
      order.shift()
      
      console.log(`⚠ Cache full, evicting ${lruKey.substring(0, 20)}...`)
    }
  }, [])

  /**
   * Add data to cache
   */
  const addToCache = useCallback((cacheKey: string, data: OverlayLayerData) => {
    const cache = cacheRef.current
    
    // Check cache size limit
    if (cache.size >= maxCacheSize && !cache.has(cacheKey)) {
      evictLRU()
    }
    
    cache.set(cacheKey, {
      data,
      accessTime: Date.now()
    })
    
    updateAccessOrder(cacheKey)
  }, [maxCacheSize, evictLRU, updateAccessOrder])

  /**
   * Load overlay layer from API or cache
   */
  const loadLayer = useCallback(async (
    overlayType: OverlayType,
    dataset: MarsDataset,
    roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
    overlayOptions: {
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
    } = {}
  ): Promise<OverlayLayerData> => {
    const cacheKey = generateCacheKey(overlayType, dataset, roi, overlayOptions)
    
    // Check cache first
    const cached = cacheRef.current.get(cacheKey)
    if (cached) {
      console.log(`✓ Loading ${overlayType} from cache`)
      updateAccessOrder(cacheKey)
      return cached.data
    }

    // Set loading state
    setLayers(prev => {
      const updated = new Map(prev)
      const layer = updated.get(overlayType) || {
        name: overlayType,
        loaded: false,
        loading: false,
        data: null,
        error: null
      }
      updated.set(overlayType, {
        ...layer,
        loading: true,
        error: null
      })
      return updated
    })

    // Cancel any existing request for this overlay
    const existingController = abortControllersRef.current.get(overlayType)
    if (existingController) {
      existingController.abort()
    }

    const controller = new AbortController()
    abortControllersRef.current.set(overlayType, controller)

    try {
      console.log(`⬇ Fetching ${overlayType} from server...`)
      
      const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
      const params = new URLSearchParams({
        overlay_type: overlayType,
        dataset: dataset,
        roi: roiStr,
        colormap: overlayOptions.colormap || 'terrain',
        relief: String(overlayOptions.relief || 0),
        sun_azimuth: String(overlayOptions.sunAzimuth || 315),
        sun_altitude: String(overlayOptions.sunAltitude || 45),
        width: String(overlayOptions.width || 2400),
        height: String(overlayOptions.height || 1600),
        buffer: String(overlayOptions.buffer || 1.5)
      })

      // Add temporal parameters if present
      if (overlayOptions.marsSol !== undefined) {
        params.append('mars_sol', String(overlayOptions.marsSol))
      }
      if (overlayOptions.season) {
        params.append('season', overlayOptions.season)
      }
      if (overlayOptions.dustStormPeriod) {
        params.append('dust_storm_period', overlayOptions.dustStormPeriod)
      }
      
      const url = `${API_BASE_URL}/visualization/overlay?${params.toString()}`
      const timeoutId = setTimeout(() => controller.abort(), 60000)

      const response = await fetch(url, { signal: controller.signal })
      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to load overlay: ${response.status} ${errorText}`)
      }

      const blob = await response.blob()
      if (blob.size === 0) {
        throw new Error('Empty overlay image received')
      }

      const blobUrl = URL.createObjectURL(blob)
      
      const left = parseFloat(response.headers.get('X-Bounds-Left') || String(roi.lon_min))
      const right = parseFloat(response.headers.get('X-Bounds-Right') || String(roi.lon_max))
      const bottom = parseFloat(response.headers.get('X-Bounds-Bottom') || String(roi.lat_min))
      const top = parseFloat(response.headers.get('X-Bounds-Top') || String(roi.lat_max))

      const data: OverlayLayerData = {
        url: blobUrl,
        bounds: { left, right, bottom, top },
        timestamp: new Date().toISOString(),
        metadata: {
          overlayType,
          dataset,
          roi: roiStr,
          colormap: overlayOptions.colormap,
          relief: overlayOptions.relief,
          sunAzimuth: overlayOptions.sunAzimuth,
          sunAltitude: overlayOptions.sunAltitude,
          width: overlayOptions.width || 2400,
          height: overlayOptions.height || 1600,
          temporalParams: {
            marsSol: overlayOptions.marsSol,
            season: overlayOptions.season,
            dustStormPeriod: overlayOptions.dustStormPeriod
          }
        },
        blobSize: blob.size
      }

      // Store in cache
      addToCache(cacheKey, data)

      // Update layer state
      setLayers(prev => {
        const updated = new Map(prev)
        const layer = updated.get(overlayType) || {
          name: overlayType,
          loaded: false,
          loading: false,
          data: null,
          error: null
        }
        updated.set(overlayType, {
          ...layer,
          loaded: true,
          loading: false,
          data,
          error: null
        })
        return updated
      })

      console.log(`✓ ${overlayType} loaded and cached`)
      return data

    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log(`✗ ${overlayType} loading cancelled`)
      } else {
        console.error(`✗ Error loading ${overlayType}:`, error)
      }

      setLayers(prev => {
        const updated = new Map(prev)
        const layer = updated.get(overlayType) || {
          name: overlayType,
          loaded: false,
          loading: false,
          data: null,
          error: null
        }
        updated.set(overlayType, {
          ...layer,
          loading: false,
          error: error.message || 'Failed to load overlay'
        })
        return updated
      })

      throw error
    } finally {
      abortControllersRef.current.delete(overlayType)
    }
  }, [addToCache, updateAccessOrder])

  /**
   * Switch to a different overlay layer
   */
  const switchLayer = useCallback(async (
    newLayerName: OverlayType,
    dataset: MarsDataset,
    roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
    overlayOptions: {
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
    } = {}
  ) => {
    if (activeLayerName === newLayerName) {
      console.log(`${newLayerName} already active`)
      return
    }

    console.log(`\n🔄 Switching to ${newLayerName}...`)

    try {
      const data = await loadLayer(newLayerName, dataset, roi, overlayOptions)
      setActiveLayerName(newLayerName)
      console.log(`✓ Now displaying ${newLayerName}\n`)
      return data
    } catch (error) {
      console.error('Failed to switch layer:', error)
      throw error
    }
  }, [activeLayerName, loadLayer])

  /**
   * Preload all layers
   */
  const preloadAllLayers = useCallback(async (
    dataset: MarsDataset,
    roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
    overlayOptions: {
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
    } = {}
  ) => {
    console.log('\n🚀 Preloading all layers...')
    const layerNames = Array.from(layers.keys())
    
    const promises = layerNames.map(async (name) => {
      const cacheKey = generateCacheKey(name, dataset, roi, overlayOptions)
      if (!cacheRef.current.has(cacheKey)) {
        try {
          await loadLayer(name, dataset, roi, overlayOptions)
        } catch (error) {
          console.error(`Failed to preload ${name}:`, error)
        }
      }
    })
    
    await Promise.all(promises)
    console.log('✓ All layers preloaded\n')
  }, [layers, loadLayer])

  /**
   * Clear cache
   */
  const clearCache = useCallback(() => {
    console.log('\n🗑️ Clearing cache...')
    
    // Revoke all blob URLs
    cacheRef.current.forEach(entry => {
      if (entry.data.url.startsWith('blob:')) {
        URL.revokeObjectURL(entry.data.url)
      }
    })
    
    cacheRef.current.clear()
    accessOrderRef.current = []
    
    // Update layer states
    setLayers(prev => {
      const updated = new Map(prev)
      updated.forEach((layer, name) => {
        updated.set(name, {
          ...layer,
          loaded: false,
          data: null
        })
      })
      return updated
    })
    
    console.log('✓ Cache cleared\n')
  }, [])

  /**
   * Get cache statistics
   */
  const getCacheStats = useCallback(() => {
    const cache = cacheRef.current
    let totalSize = 0
    
    cache.forEach(entry => {
      totalSize += entry.data.blobSize
    })
    
    return {
      cachedCount: cache.size,
      memoryUsageKB: Math.round(totalSize / 1024),
      activeLayer: activeLayerName
    }
  }, [activeLayerName])

  return {
    layers: Array.from(layers.values()),
    activeLayerName,
    loadLayer,
    switchLayer,
    preloadAllLayers,
    clearCache,
    getCacheStats
  }
}

