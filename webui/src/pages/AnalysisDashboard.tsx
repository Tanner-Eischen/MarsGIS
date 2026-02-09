import { lazy, Suspense, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useGeoPlan } from '../context/GeoPlanContext'
import TerrainMap from '../components/TerrainMap'
import OverlaySwitcher, { OverlayType } from '../components/OverlaySwitcher'
import TerrainAnalysis from './TerrainAnalysis'
import SolarAnalysis from './SolarAnalysis'
import { getStatus } from '../services/api'
import { Layers, Sun, Box, Activity } from 'lucide-react'

const Terrain3D = lazy(() => import('../components/Terrain3D'))

export default function AnalysisDashboard() {
  const [mode, setMode] = useState<'terrain' | 'solar' | '3d'>('terrain')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  
  // Shared Map State
  const [roi, setRoi] = useState({ lat_min: 18.0, lat_max: 18.6, lon_min: 77.0, lon_max: 77.8 })
  const [dataset, setDataset] = useState('mola_200m')
  const [overlayType, setOverlayType] = useState<OverlayType>('elevation')
  const [colormap, setColormap] = useState('terrain')
  const [relief, setRelief] = useState(1.0)
  const [sunAzimuth, setSunAzimuth] = useState(315)
  const [sunAltitude, setSunAltitude] = useState(45)

  // Sync overlay type with mode
  useEffect(() => {
    if (mode === 'solar' && overlayType !== 'solar') {
      setOverlayType('solar')
      setColormap('plasma')
    } else if (mode === 'terrain' && overlayType === 'solar') {
      setOverlayType('elevation')
      setColormap('terrain')
    }
  }, [mode])

  return (
    <div className="flex h-[calc(100vh-4rem)] overflow-hidden bg-gray-900 text-white">
      {/* Sidebar */}
      <div className={`flex flex-col bg-gray-800/90 backdrop-blur border-r border-gray-700 transition-all duration-300 ${sidebarOpen ? 'w-96' : 'w-16'}`}>
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          {sidebarOpen && <h2 className="font-bold text-lg tracking-wider text-cyan-400">ANALYSIS_HUB</h2>}
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-gray-700 rounded text-cyan-500">
            {sidebarOpen ? '<<' : '>>'}
          </button>
        </div>

        {/* Mode Switcher */}
        <div className="flex flex-col p-2 gap-2">
          <button
            onClick={() => setMode('terrain')}
            className={`flex items-center p-3 rounded transition-all ${mode === 'terrain' ? 'bg-cyan-900/50 text-cyan-300 border border-cyan-500/50' : 'hover:bg-gray-700 text-gray-400'}`}
            title="Terrain Analysis"
          >
            <Layers size={20} />
            {sidebarOpen && <span className="ml-3 font-medium">Terrain Analysis</span>}
          </button>
          <button
            onClick={() => setMode('solar')}
            className={`flex items-center p-3 rounded transition-all ${mode === 'solar' ? 'bg-amber-900/50 text-amber-300 border border-amber-500/50' : 'hover:bg-gray-700 text-gray-400'}`}
            title="Solar Potential"
          >
            <Sun size={20} />
            {sidebarOpen && <span className="ml-3 font-medium">Solar Potential</span>}
          </button>
          <button
            onClick={() => setMode('3d')}
            className={`flex items-center p-3 rounded transition-all ${mode === '3d' ? 'bg-purple-900/50 text-purple-300 border border-purple-500/50' : 'hover:bg-gray-700 text-gray-400'}`}
            title="3D Visualization"
          >
            <Box size={20} />
            {sidebarOpen && <span className="ml-3 font-medium">3D Visualization</span>}
          </button>
        </div>

        {/* Contextual Controls */}
        {sidebarOpen && (
          <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
             {/* Shared Overlay Controls */}
             <div className="bg-gray-900/50 p-3 rounded border border-gray-700/50">
                <OverlaySwitcher
                  overlayType={overlayType}
                  onOverlayTypeChange={setOverlayType}
                  colormap={colormap}
                  onColormapChange={setColormap}
                  relief={relief}
                  onReliefChange={setRelief}
                  sunAzimuth={sunAzimuth}
                  onSunAzimuthChange={setSunAzimuth}
                  sunAltitude={sunAltitude}
                  onSunAltitudeChange={setSunAltitude}
                  dataset={dataset as any}
                  roi={roi}
                  showLayerList={false}
                  showCacheStats={false}
                />
             </div>

             {/* Mode Specific Content */}
             <div className="space-y-4">
                {mode === 'terrain' && (
                  <div className="text-sm text-gray-400">
                    <h3 className="text-cyan-400 font-semibold mb-2 uppercase tracking-wider">Terrain Parameters</h3>
                    <TerrainAnalysis 
                      roi={roi} 
                      onRoiChange={setRoi} 
                      dataset={dataset} 
                      onDatasetChange={setDataset} 
                    />
                  </div>
                )}
                
                {mode === 'solar' && (
                   <div className="text-sm text-gray-400">
                   <h3 className="text-amber-400 font-semibold mb-2 uppercase tracking-wider">Solar Config</h3>
                   <SolarAnalysis 
                      roi={roi} 
                      onRoiChange={setRoi} 
                      dataset={dataset} 
                      onDatasetChange={setDataset}
                   />
                 </div>
                )}
             </div>
          </div>
        )}
      </div>

      {/* Main Visualization Area */}
      <div className="flex-1 relative bg-black">
        {mode === '3d' ? (
          <Suspense
            fallback={
              <div className="flex items-center justify-center h-full w-full bg-gray-900 text-cyan-400 font-mono">
                Loading 3D module...
              </div>
            }
          >
            <Terrain3D
              roi={roi}
              dataset={dataset}
              overlayType={overlayType}
              overlayOptions={{
                colormap,
                relief,
                sunAzimuth,
                sunAltitude,
              }}
            />
          </Suspense>
        ) : (
          <TerrainMap
            roi={roi}
            dataset={dataset}
            overlayType={overlayType}
            overlayOptions={{
              colormap,
              relief,
              sunAzimuth,
              sunAltitude
            }}
            showSites={true}
            showWaypoints={true}
          />
        )}
        
        {/* HUD Overlays */}
        <div className="absolute top-4 right-4 bg-gray-900/80 backdrop-blur border border-cyan-500/30 p-2 rounded text-xs font-mono text-cyan-400 pointer-events-none">
          <div>LAT: {(roi.lat_min + roi.lat_max) / 2}°</div>
          <div>LON: {(roi.lon_min + roi.lon_max) / 2}°</div>
          <div>ZOOM: 100%</div>
        </div>
      </div>
    </div>
  )
}
