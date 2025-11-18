import { useState } from 'react'
import TerrainMap from '../components/TerrainMap'
import Terrain3D from '../components/Terrain3D'

export default function Visualization() {
  const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [dataset, setDataset] = useState('mola')
  const [showSites, setShowSites] = useState(true)
  const [showWaypoints, setShowWaypoints] = useState(true)
  const [relief, setRelief] = useState(1.0) // Default to shaded relief
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Terrain Visualization</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 space-y-4">
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

      {viewMode === '2d' ? (
        <TerrainMap
          roi={roi}
          dataset={dataset}
          showSites={showSites}
          showWaypoints={showWaypoints}
          relief={relief}
        />
      ) : (
        <Terrain3D
          roi={roi}
          dataset={dataset}
          showSites={showSites}
          showWaypoints={showWaypoints}
        />
      )}
    </div>
  )
}

