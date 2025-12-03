import { useState } from 'react'
import TerrainMap from '../components/TerrainMap'
import Terrain3D from '../components/Terrain3D'
import OverlaySwitcher from '../components/OverlaySwitcher'
import RoverAnimationControls from '../components/RoverAnimationControls'
import { OverlayLayerProvider } from '../contexts/OverlayLayerContext'
import { use3DTerrain } from '../hooks/use3DMapData'
import { useWaypointsGeoJson } from '../hooks/useMapData'
import { useRoverAnimation, RoverPosition } from '../hooks/useRoverAnimation'

function VisualizationContent() {
  const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [dataset, setDataset] = useState('mola')
  const [showSites, setShowSites] = useState(true)
  const [showWaypoints, setShowWaypoints] = useState(true)
  const [relief, setRelief] = useState(1.0) // Default to shaded relief
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')
  const [overlayType, setOverlayType] = useState<'elevation' | 'solar' | 'dust' | 'hillshade' | 'slope' | 'aspect' | 'roughness' | 'tri'>('elevation')
  const [colormap, setColormap] = useState('terrain')
  const [sunAzimuth, setSunAzimuth] = useState(315)
  const [sunAltitude, setSunAltitude] = useState(45)
  const [enableRoverAnimation, setEnableRoverAnimation] = useState(false)
  const [roverPosition, setRoverPosition] = useState<RoverPosition | null>(null)

  // Get terrain data and waypoints for rover animation
  const { terrainData } = use3DTerrain(roi, dataset)
  const waypointsGeoJson = useWaypointsGeoJson(showWaypoints)

  // Create rover animation hook
  const roverAnimation = useRoverAnimation(
    enableRoverAnimation && showWaypoints ? waypointsGeoJson : null,
    terrainData,
    (pos) => setRoverPosition(pos)
  )

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Terrain Visualization</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Overlay Controls */}
        <div className="lg:col-span-1">
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
            dataset={dataset}
            roi={roi}
            showLayerList={true}
            showCacheStats={true}
          />
        </div>

        {/* Map Controls */}
        <div className="lg:col-span-2 bg-gray-800 rounded-lg p-6 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">Latitude Min</label>
            <input
              type="number"
              value={roi.lat_min}
              onChange={(e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) })}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Latitude Max</label>
            <input
              type="number"
              value={roi.lat_max}
              onChange={(e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) })}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Longitude Min</label>
            <input
              type="number"
              value={roi.lon_min}
              onChange={(e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) })}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Longitude Max</label>
            <input
              type="number"
              value={roi.lon_max}
              onChange={(e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) })}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              step="0.1"
            />
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div>
            <label className="block text-sm font-medium mb-2">Dataset</label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="bg-gray-700 text-white px-4 py-2 rounded-md"
            >
              <option value="mola">MOLA</option>
              <option value="hirise">HiRISE</option>
              <option value="ctx">CTX</option>
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium mb-2">Relief (3D exaggeration)</label>
            <input
              type="range"
              min={0}
              max={3}
              step={0.1}
              value={relief}
              onChange={(e) => setRelief(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="text-xs text-gray-400 mt-1">
              {relief === 0 ? 'Flat color (no shading)' : `Relief factor: ${relief.toFixed(1)}x`}
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div>
              <label className="block text-sm font-medium mb-2">View Mode</label>
              <select
                value={viewMode}
                onChange={(e) => setViewMode(e.target.value as '2d' | '3d')}
                className="bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                <option value="2d">2D Map</option>
                <option value="3d">3D Terrain</option>
              </select>
            </div>
            <div className="flex items-center space-x-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showSites}
                  onChange={(e) => setShowSites(e.target.checked)}
                  className="mr-2"
                />
                Show Sites
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showWaypoints}
                  onChange={(e) => setShowWaypoints(e.target.checked)}
                  className="mr-2"
                />
                Show Waypoints
              </label>
            </div>
          </div>
        </div>
        </div>
      </div>

      {viewMode === '2d' ? (
        <TerrainMap
          roi={roi}
          dataset={dataset}
          showSites={showSites}
          showWaypoints={showWaypoints}
          relief={relief}
          overlayType={overlayType}
          overlayOptions={{
            colormap,
            relief,
            sunAzimuth,
            sunAltitude
          }}
        />
      ) : (
        <>
          <Terrain3D
            roi={roi}
            dataset={dataset}
            showSites={showSites}
            showWaypoints={showWaypoints}
            enableRoverAnimation={enableRoverAnimation}
            roverPosition={roverPosition}
            isAnimating={roverAnimation.state.isPlaying}
            overlayType={overlayType}
            overlayOptions={{
              colormap,
              relief,
              sunAzimuth,
              sunAltitude
            }}
          />
          {viewMode === '3d' && (
            <div className="mt-4">
              <div className="flex items-center mb-2">
                <input
                  type="checkbox"
                  checked={enableRoverAnimation}
                  onChange={(e) => {
                    setEnableRoverAnimation(e.target.checked)
                    if (!e.target.checked) {
                      roverAnimation.reset()
                      setRoverPosition(null)
                    }
                  }}
                  className="mr-2"
                  disabled={!showWaypoints || !waypointsGeoJson}
                />
                <label className="text-sm font-medium">
                  Enable Rover Animation
                </label>
              </div>
              {enableRoverAnimation && showWaypoints && waypointsGeoJson && terrainData && (
                <RoverAnimationControls
                  waypointsGeoJson={waypointsGeoJson}
                  terrainData={terrainData}
                  animation={roverAnimation}
                />
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function Visualization() {
  return (
    <OverlayLayerProvider maxCacheSize={5}>
      <VisualizationContent />
    </OverlayLayerProvider>
  )
}

