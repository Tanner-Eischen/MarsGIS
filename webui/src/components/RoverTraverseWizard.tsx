import { useState, useEffect } from 'react'
import { useGeoPlan } from '../context/GeoPlanContext'
import { planMultipleRoutes, MultiRouteRequest, MultiRouteResponse } from '../services/api'
import TerrainMap from './TerrainMap'
import { apiFetch } from '../lib/apiBase'

interface Preset {
  id: string
  name: string
  description: string
  scope: string
}

interface Site {
  site_id: number
  lat: number
  lon: number
  suitability_score: number
  rank: number
}

export default function RoverTraverseWizard() {
  const [step, setStep] = useState<1 | 2 | 3>(1)
  const [startSiteId, setStartSiteId] = useState<number>(1)
  const [endSiteId, setEndSiteId] = useState<number>(2)
  const [analysisDir, setAnalysisDir] = useState('data/output')
  const [presetId, setPresetId] = useState<string>('balanced')
  const [roverCapabilities, setRoverCapabilities] = useState({ max_slope_deg: 25.0, max_roughness: 1.0 })
  const [presets, setPresets] = useState<Preset[]>([])
  const [sites, setSites] = useState<Site[]>([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<MultiRouteResponse | null>(null)
  const { recommendedLandingSiteId } = useGeoPlan()

  // Fetch route presets on mount
  useEffect(() => {
    apiFetch('/analysis/presets?scope=route')
      .then(res => res.json())
      .then(data => {
        const routePresets = data.route_presets || []
        setPresets(routePresets)
        if (routePresets.length > 0 && !presetId) {
          setPresetId(routePresets[0].id)
        }
      })
      .catch(err => console.error('Failed to load presets:', err))
  }, [])

  // Load sites from analysis results
  useEffect(() => {
    if (analysisDir) {
      apiFetch('/visualization/sites-geojson')
        .then(res => res.json())
        .then(data => {
          const siteList: Site[] = data.features.map((f: any) => ({
            site_id: f.properties.site_id,
            lat: f.geometry.type === 'Point' ? f.geometry.coordinates[1] : f.properties.lat || 0,
            lon: f.geometry.type === 'Point' ? f.geometry.coordinates[0] : f.properties.lon || 0,
            suitability_score: f.properties.suitability_score,
            rank: f.properties.rank,
          }))
          setSites(siteList)
          if (siteList.length > 0) {
            if (!startSiteId) setStartSiteId(siteList[0].site_id)
            if (!endSiteId && siteList.length > 1) setEndSiteId(siteList[1].site_id)
          }
        })
        .catch(err => console.error('Failed to load sites:', err))
    }
  }, [analysisDir])

  useEffect(() => {
    if (recommendedLandingSiteId) setStartSiteId(recommendedLandingSiteId)
  }, [recommendedLandingSiteId])

  const handleRunScenario = async () => {
    setLoading(true)
    try {
      const startSite = sites.find(s => s.site_id === startSiteId)
      const endSite = sites.find(s => s.site_id === endSiteId)
      if (!startSite || !endSite) throw new Error('Invalid start or end site')
      const request: MultiRouteRequest = {
        site_id: endSite.site_id,
        analysis_dir: analysisDir,
        start_lat: startSite.lat,
        start_lon: startSite.lon,
        strategies: ['safest', 'balanced', 'direct'],
        max_slope_deg: roverCapabilities.max_slope_deg,
      }
      const response = await planMultipleRoutes(request)
      setResults(response)
      setStep(3)
    } catch (error: any) {
      console.error('Scenario failed:', error)
      alert(`Scenario failed: ${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Step Indicator */}
      <div className="flex items-center justify-center gap-4">
        <div className={`flex items-center ${step >= 1 ? 'text-blue-400' : 'text-gray-500'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 1 ? 'bg-blue-600' : 'bg-gray-700'}`}>
            1
          </div>
          <span className="ml-2">Select Sites</span>
        </div>
        <div className="w-16 h-0.5 bg-gray-700"></div>
        <div className={`flex items-center ${step >= 2 ? 'text-blue-400' : 'text-gray-500'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? 'bg-blue-600' : 'bg-gray-700'}`}>
            2
          </div>
          <span className="ml-2">Rover & Route</span>
        </div>
        <div className="w-16 h-0.5 bg-gray-700"></div>
        <div className={`flex items-center ${step >= 3 ? 'text-blue-400' : 'text-gray-500'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 3 ? 'bg-blue-600' : 'bg-gray-700'}`}>
            3
          </div>
          <span className="ml-2">Route & Cost</span>
        </div>
      </div>

      {/* Step 1: Site Selection */}
      {step === 1 && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Step 1: Select Start and End Sites</h3>
          
          <div>
            <label className="block text-sm font-medium mb-2">Analysis Directory</label>
            <input
              type="text"
              value={analysisDir}
              onChange={(e) => setAnalysisDir(e.target.value)}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              placeholder="data/output"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Start Site</label>
              <select
                value={startSiteId}
                onChange={(e) => setStartSiteId(parseInt(e.target.value))}
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                {sites.map(site => (
                  <option key={site.site_id} value={site.site_id}>
                    Site {site.site_id} (Rank {site.rank}, Score {site.suitability_score.toFixed(2)})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">End Site</label>
              <select
                value={endSiteId}
                onChange={(e) => setEndSiteId(parseInt(e.target.value))}
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                {sites.map(site => (
                  <option key={site.site_id} value={site.site_id}>
                    Site {site.site_id} (Rank {site.rank}, Score {site.suitability_score.toFixed(2)})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            onClick={() => setStep(2)}
            disabled={sites.length === 0}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold"
          >
            Next: Rover Capabilities
          </button>
        </div>
      )}

      {/* Step 2: Rover Capabilities + Route Preset */}
      {step === 2 && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Step 2: Rover Capabilities and Route Preset</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Route Preset</label>
              <select
                value={presetId}
                onChange={(e) => setPresetId(e.target.value)}
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                {presets.map(preset => (
                  <option key={preset.id} value={preset.id}>{preset.name}</option>
                ))}
              </select>
              {presets.find(p => p.id === presetId) && (
                <p className="text-sm text-gray-400 mt-1">
                  {presets.find(p => p.id === presetId)?.description}
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Max Slope (degrees)</label>
                <input
                  type="number"
                  value={roverCapabilities.max_slope_deg}
                  onChange={(e) => setRoverCapabilities({ ...roverCapabilities, max_slope_deg: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
                  step="1"
                  min="0"
                  max="90"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Max Roughness</label>
                <input
                  type="number"
                  value={roverCapabilities.max_roughness}
                  onChange={(e) => setRoverCapabilities({ ...roverCapabilities, max_roughness: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
                  step="0.1"
                  min="0"
                />
              </div>
            </div>
          </div>

          <div className="flex gap-4">
            <button
              onClick={() => setStep(1)}
              className="flex-1 bg-gray-700 hover:bg-gray-600 p-3 rounded font-semibold"
            >
              Back
            </button>
            <button
              onClick={handleRunScenario}
              disabled={loading}
              className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold"
            >
              {loading ? 'Planning Route...' : 'Plan Route'}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Route Results */}
      {step === 3 && results && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Step 3: Route and Cost Summary</h3>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-semibold text-lg mb-2">Route Metrics</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              {results.routes.map((r) => (
                <div key={r.strategy} className="p-3 rounded border" style={{ borderColor: r.strategy === 'safest' ? '#00ff00' : r.strategy === 'balanced' ? '#1e90ff' : '#ffa500' }}>
                  <div className="font-semibold capitalize">{r.strategy}</div>
                  <div><span className="text-gray-400">Distance:</span> {r.total_distance_m.toFixed(2)} m</div>
                  <div><span className="text-gray-400">Waypoints:</span> {r.num_waypoints}</div>
                  <div><span className="text-gray-400">Relative Cost:</span> {r.relative_cost_percent.toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-semibold mb-2">Waypoints (showing up to 20 per route)</h4>
            <div className="max-h-64 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-600">
                    <th className="text-left p-2">Route</th>
                    <th className="text-left p-2">ID</th>
                    <th className="text-left p-2">X (m)</th>
                    <th className="text-left p-2">Y (m)</th>
                    <th className="text-left p-2">Tolerance (m)</th>
                  </tr>
                </thead>
                <tbody>
                  {results.routes.flatMap(r => r.waypoints.slice(0, 20).map(wp => (
                    <tr key={`${r.strategy}-${wp.waypoint_id}`} className="border-b border-gray-600">
                      <td className="p-2 capitalize">{r.strategy}</td>
                      <td className="p-2">{wp.waypoint_id}</td>
                      <td className="p-2">{wp.x_meters.toFixed(2)}</td>
                      <td className="p-2">{wp.y_meters.toFixed(2)}</td>
                      <td className="p-2">{wp.tolerance_meters.toFixed(2)}</td>
                    </tr>
                  )))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="h-96">
            <TerrainMap
              roi={{ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 }}
              dataset="mola"
              showSites={true}
              showWaypoints={true}
            />
          </div>

          <button
            onClick={() => {
              setStep(1)
              setResults(null)
            }}
            className="w-full bg-blue-600 hover:bg-blue-700 p-3 rounded font-semibold"
          >
            Plan New Route
          </button>
        </div>
      )}
    </div>
  )
}

