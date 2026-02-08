import { useState } from 'react'
import { useOverlayLayerContext } from '../contexts/OverlayLayerContext'
import { getOverlayDefinition, MARS_OVERLAY_DEFINITIONS, type MarsDataset } from '../config/marsDataSources'
import LayerStatusBadge from './LayerStatusBadge'
import { Layers, Eye, EyeOff, Database, RefreshCw, Trash2, Settings, Sun } from 'lucide-react'

export type OverlayType = 'elevation' | 'solar' | 'dust' | 'hillshade' | 'slope' | 'aspect' | 'roughness' | 'tri' | 'viewshed' | 'comms_risk'

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
  { value: 'terrain', label: 'Natural Terrain' },
  { value: 'viridis', label: 'Viridis (Analytical)' },
  { value: 'plasma', label: 'Plasma (Solar/Heat)' },
  { value: 'magma', label: 'Magma (Geological)' },
  { value: 'cividis', label: 'Cividis (Colorblind Safe)' },
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
  const [activeTab, setActiveTab] = useState<'layers' | 'settings'>('layers')
  
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
    <div className="glass-panel rounded-lg overflow-hidden text-sm">
      <div className="bg-gray-800/50 p-3 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2 text-cyan-400">
          <Layers size={16} />
          <h3 className="font-bold tracking-wider">VISUAL_LAYERS</h3>
        </div>
        <div className="flex gap-2">
           <button 
             onClick={() => setActiveTab('layers')}
             className={`p-1 rounded ${activeTab === 'layers' ? 'bg-cyan-900/50 text-cyan-300' : 'text-gray-500 hover:text-gray-300'}`}
           >
             <Layers size={14} />
           </button>
           <button 
             onClick={() => setActiveTab('settings')}
             className={`p-1 rounded ${activeTab === 'settings' ? 'bg-cyan-900/50 text-cyan-300' : 'text-gray-500 hover:text-gray-300'}`}
           >
             <Settings size={14} />
           </button>
           <button
            onClick={() => setExpanded(!expanded)}
            className="text-gray-500 hover:text-white p-1"
          >
            {expanded ? <Eye size={14} /> : <EyeOff size={14} />}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="p-3 space-y-4 bg-gray-900/30">
          {activeTab === 'layers' && (
            <div className="grid grid-cols-2 gap-2">
               {MARS_OVERLAY_DEFINITIONS.filter(def => 
                ['elevation', 'solar', 'viewshed', 'comms_risk', 'hillshade', 'slope', 'roughness', 'tri', 'aspect', 'dust'].includes(def.name)
               ).map((def) => {
                 const isActive = overlayType === def.name
                 const status = getLayerStatus(def.name as OverlayType)
                 const isDisabled = def.comingSoon === true
                 return (
                   <button
                     key={def.name}
                     onClick={() => {
                       if (isDisabled) return
                       onOverlayTypeChange(def.name as OverlayType)
                     }}
                     disabled={isDisabled}
                     className={`relative flex flex-col items-start p-2 rounded border transition-all duration-200 ${
                       isActive
                         ? 'bg-cyan-900/20 border-cyan-500/50 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.1)]'
                         : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-gray-700 hover:border-gray-500'
                     } ${isDisabled ? 'opacity-60 cursor-not-allowed hover:bg-gray-800/50 hover:border-gray-700' : ''}`}
                   >
                     <div className="flex items-center justify-between w-full mb-1">
                        <span className="font-mono text-xs font-bold uppercase">{def.displayName}</span>
                        <LayerStatusBadge status={status} dataset={dataset} />
                     </div>
                     <span className="text-[10px] opacity-70 truncate w-full text-left">{def.description}</span>
                     {isDisabled && (
                       <span className="mt-1 text-[9px] uppercase tracking-wide text-amber-300/80">Coming soon</span>
                     )}
                   </button>
                 )
               })}
            </div>
          )}

          {activeTab === 'settings' && (
             <div className="space-y-4">
                {/* Metadata */}
                {activeOverlayDef && (
                  <div className="p-2 bg-cyan-900/10 border border-cyan-500/20 rounded text-xs text-cyan-200/80 font-mono">
                    <div className="flex justify-between">
                      <span>RES: {activeOverlayDef.resolution || 'N/A'}</span>
                      <span>SRC: {dataset.toUpperCase()}</span>
                    </div>
                  </div>
                )}

                {/* Colormap */}
                {showColormap && (
                  <div>
                    <label className="block text-xs font-bold text-gray-400 mb-1 uppercase">Spectral Palette</label>
                    <select
                      value={colormap}
                      onChange={(e) => onColormapChange(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-600 text-white px-2 py-1 rounded text-xs focus:border-cyan-500 focus:outline-none"
                    >
                      {COLORMAPS.map((cm) => (
                        <option key={cm.value} value={cm.value}>{cm.label}</option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Relief Slider */}
                {showRelief && (
                  <div>
                    <label className="block text-xs font-bold text-gray-400 mb-1 uppercase">Exaggeration: {relief.toFixed(1)}x</label>
                    <input
                      type="range"
                      min="0"
                      max="3"
                      step="0.1"
                      value={relief}
                      onChange={(e) => onReliefChange(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                )}

                {/* Sun Angles */}
                {showSunAngles && (
                  <div className="space-y-2 pt-2 border-t border-gray-700/50">
                    <div className="flex items-center gap-2 text-amber-500 mb-1">
                        <Sun size={12} />
                        <span className="text-xs font-bold uppercase">Solar Ephemeris</span>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-gray-400">
                        <span>Azimuth</span>
                        <span>{sunAzimuth.toFixed(0)}°</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="360"
                        step="15"
                        value={sunAzimuth}
                        onChange={(e) => onSunAzimuthChange(parseFloat(e.target.value))}
                        className="w-full accent-amber-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-gray-400">
                        <span>Altitude</span>
                        <span>{sunAltitude.toFixed(0)}°</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="90"
                        step="5"
                        value={sunAltitude}
                        onChange={(e) => onSunAltitudeChange(parseFloat(e.target.value))}
                        className="w-full accent-amber-500"
                      />
                    </div>
                  </div>
                )}
             </div>
          )}

          {/* System Stats / Cache */}
          {showCacheStats && (
            <div className="pt-2 border-t border-gray-700/50">
               <div className="flex items-center justify-between mb-2">
                 <span className="text-xs font-bold text-gray-500 uppercase">Memory Cache</span>
                 <span className="text-xs font-mono text-cyan-400">{(cacheStats.memoryUsageKB / 1024).toFixed(1)} MB</span>
               </div>
               <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={handlePreloadAll}
                    disabled={!roi}
                    className="flex items-center justify-center gap-1 px-2 py-1.5 bg-gray-800 hover:bg-gray-700 text-xs rounded border border-gray-600 transition-colors disabled:opacity-50"
                  >
                    <Database size={10} />
                    PRELOAD
                  </button>
                  <button
                    onClick={handleClearCache}
                    className="flex items-center justify-center gap-1 px-2 py-1.5 bg-gray-800 hover:bg-red-900/30 text-xs rounded border border-gray-600 hover:border-red-500/50 transition-colors"
                  >
                    <Trash2 size={10} />
                    PURGE
                  </button>
               </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
