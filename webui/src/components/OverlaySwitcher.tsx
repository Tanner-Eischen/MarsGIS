import { useState, useEffect } from 'react'
import { useOverlayLayerContext } from '../contexts/OverlayLayerContext'
import { getOverlayDefinition, MARS_OVERLAY_DEFINITIONS, type OverlayType, type MarsDataset } from '../config/marsDataSources'
import LayerStatusBadge from './LayerStatusBadge'

export type OverlayType = 'elevation' | 'solar' | 'dust' | 'hillshade' | 'slope' | 'aspect' | 'roughness' | 'tri'

interface OverlaySwitcherProps {
  overlayType: OverlayType
  onOverlayTypeChange: (type: OverlayType) => void
  colormap: string
  onColormapChange: (colormap: string) => void
  relief: number
  onReliefChange: (relief: number) => void
  sunAzimuth: number
  onSunAzimuthChange: (azimuth: number) => void
  sunAltitude: number
  onSunAltitudeChange: (altitude: number) => void
  dataset?: MarsDataset
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  showLayerList?: boolean
  showCacheStats?: boolean
}

const COLORMAPS = [
  { value: 'terrain', label: 'Terrain' },
  { value: 'viridis', label: 'Viridis' },
  { value: 'plasma', label: 'Plasma' },
  { value: 'inferno', label: 'Inferno' },
  { value: 'magma', label: 'Magma' },
  { value: 'cividis', label: 'Cividis' },
]

export default function OverlaySwitcher({
  overlayType,
  onOverlayTypeChange,
  colormap,
  onColormapChange,
  relief,
  onReliefChange,
  sunAzimuth,
  onSunAzimuthChange,
  sunAltitude,
  onSunAltitudeChange,
  dataset = 'mola',
  roi,
  showLayerList = true,
  showCacheStats = true,
}: OverlaySwitcherProps) {
  const [expanded, setExpanded] = useState(true)
  const [showLayerSection, setShowLayerSection] = useState(true)
  const [showCacheSection, setShowCacheSection] = useState(true)
  
  const overlayContext = useOverlayLayerContext()
  const cacheStats = overlayContext.getCacheStats()

  const showColormap = overlayType !== 'hillshade'
  const showRelief = overlayType === 'elevation'
  const showSunAngles = overlayType === 'solar' || overlayType === 'hillshade' || overlayType === 'dust'

  const activeOverlayDef = getOverlayDefinition(overlayType)
  
  const handlePreloadAll = async () => {
    if (!roi) return
    try {
      await overlayContext.preloadAllLayers(dataset, roi, {
        colormap,
        relief,
        sunAzimuth,
        sunAltitude
      })
    } catch (error) {
      console.error('Failed to preload layers:', error)
    }
  }

  const handleClearCache = () => {
    overlayContext.clearCache()
  }

  const getLayerStatus = (layerName: OverlayType): 'loading' | 'cached' | 'error' | 'idle' => {
    const layer = overlayContext.layers.find(l => l.name === layerName)
    if (!layer) return 'idle'
    if (layer.loading) return 'loading'
    if (layer.loaded && layer.data) return 'cached'
    if (layer.error) return 'error'
    return 'idle'
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Map Overlay</h3>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-gray-400 hover:text-white"
        >
          {expanded ? '▼' : '▶'}
        </button>
      </div>

      {expanded && (
        <div className="space-y-4">
          {/* Layer List */}
          {showLayerList && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wide">
                  Available Layers
                </label>
                <button
                  onClick={() => setShowLayerSection(!showLayerSection)}
                  className="text-gray-400 hover:text-white text-xs"
                >
                  {showLayerSection ? '▼' : '▶'}
                </button>
              </div>
              {showLayerSection && (
                <div className="space-y-1 mb-4">
                  {MARS_OVERLAY_DEFINITIONS.filter(def => 
                    ['elevation', 'solar', 'dust', 'hillshade', 'slope', 'aspect', 'roughness', 'tri'].includes(def.name)
                  ).map((def) => {
                    const isActive = overlayType === def.name
                    const status = getLayerStatus(def.name)
                    return (
                      <button
                        key={def.name}
                        onClick={() => onOverlayTypeChange(def.name)}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-sm transition-colors ${
                          isActive
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600'
                        }`}
                        title={def.description}
                      >
                        <div className="flex items-center gap-2">
                          <span>{def.icon}</span>
                          <span className="font-medium">{def.displayName}</span>
                        </div>
                        <LayerStatusBadge status={status} dataset={dataset} />
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {/* Cache Statistics */}
          {showCacheStats && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wide">
                  Cache Statistics
                </label>
                <button
                  onClick={() => setShowCacheSection(!showCacheSection)}
                  className="text-gray-400 hover:text-white text-xs"
                >
                  {showCacheSection ? '▼' : '▶'}
                </button>
              </div>
              {showCacheSection && (
                <div className="bg-gray-700/50 rounded-md p-3 mb-4 space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Cached Layers:</span>
                    <span className="text-white font-semibold">{cacheStats.cachedCount}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Memory Usage:</span>
                    <span className="text-white font-semibold">{cacheStats.memoryUsageKB.toLocaleString()} KB</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Active Layer:</span>
                    <span className="text-white font-semibold">
                      {cacheStats.activeLayer ? getOverlayDefinition(cacheStats.activeLayer)?.displayName || cacheStats.activeLayer : 'None'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Cache Actions */}
          {showCacheStats && (
            <div className="space-y-2 mb-4">
              <button
                onClick={handlePreloadAll}
                disabled={!roi}
                className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed text-white text-sm rounded-md transition-colors"
              >
                Preload All Layers
              </button>
              <button
                onClick={handleClearCache}
                className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-md transition-colors"
              >
                Clear Cache
              </button>
            </div>
          )}

          {/* Overlay Type Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Overlay Type
            </label>
            <div className="grid grid-cols-2 gap-2">
              {MARS_OVERLAY_DEFINITIONS.filter(def => 
                ['elevation', 'solar', 'dust', 'hillshade', 'slope', 'aspect', 'roughness', 'tri'].includes(def.name)
              ).map((def) => (
                <button
                  key={def.name}
                  onClick={() => onOverlayTypeChange(def.name)}
                  className={`px-3 py-2 rounded-md text-sm text-left transition-colors ${
                    overlayType === def.name
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                  title={def.description}
                >
                  {def.icon} {def.displayName}
                </button>
              ))}
            </div>
          </div>

          {/* Layer Metadata */}
          {activeOverlayDef && (
            <div className="bg-gray-700/30 rounded-md p-2 text-xs text-gray-400">
              <div className="font-medium text-gray-300 mb-1">{activeOverlayDef.displayName}</div>
              <div>{activeOverlayDef.description}</div>
              {activeOverlayDef.resolution && (
                <div className="mt-1">Resolution: {activeOverlayDef.resolution}</div>
              )}
              <div className="mt-1">Dataset: {dataset.toUpperCase()}</div>
            </div>
          )}

          {/* Colormap Selector */}
          {showColormap && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Colormap
              </label>
              <select
                value={colormap}
                onChange={(e) => onColormapChange(e.target.value)}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded-md text-sm"
              >
                {COLORMAPS.map((cm) => (
                  <option key={cm.value} value={cm.value}>
                    {cm.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Relief Slider (for elevation) */}
          {showRelief && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Relief / Hillshade Intensity: {relief.toFixed(1)}x
              </label>
              <input
                type="range"
                min="0"
                max="3"
                step="0.1"
                value={relief}
                onChange={(e) => onReliefChange(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-400 mt-1">
                {relief === 0 ? 'Flat color (no shading)' : `3D shading intensity`}
              </div>
            </div>
          )}

          {/* Sun Angle Controls (for solar and hillshade) */}
          {showSunAngles && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Sun Azimuth: {sunAzimuth.toFixed(0)}°
                </label>
                <input
                  type="range"
                  min="0"
                  max="360"
                  step="1"
                  value={sunAzimuth}
                  onChange={(e) => onSunAzimuthChange(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-gray-400 mt-1">
                  0°=North, 90°=East, 180°=South, 270°=West
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Sun Altitude: {sunAltitude.toFixed(0)}°
                </label>
                <input
                  type="range"
                  min="0"
                  max="90"
                  step="1"
                  value={sunAltitude}
                  onChange={(e) => onSunAltitudeChange(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-gray-400 mt-1">
                  0°=Horizon, 90°=Zenith
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

