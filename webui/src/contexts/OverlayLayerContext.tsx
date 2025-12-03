import React, { createContext, useContext, ReactNode } from 'react'
import { useOverlayLayerManager, type OverlayLayerData, type OverlayType, type MarsDataset } from '../hooks/useOverlayLayerManager'

interface OverlayLayerContextValue {
  layers: ReturnType<typeof useOverlayLayerManager>['layers']
  activeLayerName: OverlayType | null
  loadLayer: (
    overlayType: OverlayType,
    dataset: MarsDataset,
    roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
    overlayOptions?: {
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
  ) => Promise<OverlayLayerData>
  switchLayer: (
    newLayerName: OverlayType,
    dataset: MarsDataset,
    roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
    overlayOptions?: {
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
  ) => Promise<OverlayLayerData | undefined>
  preloadAllLayers: (
    dataset: MarsDataset,
    roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
    overlayOptions?: {
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
  ) => Promise<void>
  clearCache: () => void
  getCacheStats: () => {
    cachedCount: number
    memoryUsageKB: number
    activeLayer: OverlayType | null
  }
}

const OverlayLayerContext = createContext<OverlayLayerContextValue | undefined>(undefined)

interface OverlayLayerProviderProps {
  children: ReactNode
  maxCacheSize?: number
}

export function OverlayLayerProvider({ children, maxCacheSize = 5 }: OverlayLayerProviderProps) {
  const layerManager = useOverlayLayerManager({ maxCacheSize })

  const value: OverlayLayerContextValue = {
    layers: layerManager.layers,
    activeLayerName: layerManager.activeLayerName,
    loadLayer: layerManager.loadLayer,
    switchLayer: layerManager.switchLayer,
    preloadAllLayers: layerManager.preloadAllLayers,
    clearCache: layerManager.clearCache,
    getCacheStats: layerManager.getCacheStats
  }

  return (
    <OverlayLayerContext.Provider value={value}>
      {children}
    </OverlayLayerContext.Provider>
  )
}

export function useOverlayLayerContext(): OverlayLayerContextValue {
  const context = useContext(OverlayLayerContext)
  if (context === undefined) {
    throw new Error('useOverlayLayerContext must be used within an OverlayLayerProvider')
  }
  return context
}

