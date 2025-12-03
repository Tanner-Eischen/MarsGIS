import { useEffect, useState } from 'react'
import TerrainMap from '../components/TerrainMap'
import SolarImpactPanel from '../components/SolarImpactPanel'
import { analyzeSolarPotential, SolarAnalysisResponse, getExampleROIs, ExampleROIItem } from '../services/api'

interface ROI {
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
}

export default function SolarAnalysis() {
  // Default to Jezero Crater
  const [roi, setROI] = useState<ROI>({
    lat_min: 18.0,
    lat_max: 18.6,
    lon_min: 77.0,
    lon_max: 77.8,
  })
  const [dataset, setDataset] = useState('mola')
  const [examples, setExamples] = useState<ExampleROIItem[]>([])
  useEffect(() => {
    const s = localStorage.getItem('solar.roi')
    const d = localStorage.getItem('solar.dataset')
    if (s) { try { const o = JSON.parse(s); if (o && o.lat_min !== undefined) setROI(o) } catch {} }
    if (d) setDataset(d)
  }, [])
  useEffect(() => { localStorage.setItem('solar.roi', JSON.stringify(roi)) }, [roi])
  useEffect(() => { localStorage.setItem('solar.dataset', dataset) }, [dataset])
  
  // Panel configuration
  const [panelEfficiency, setPanelEfficiency] = useState(0.25)
  const [panelArea, setPanelArea] = useState(100.0)
  
  // Mission parameters
  const [batteryCapacity, setBatteryCapacity] = useState(50.0)
  const [dailyPowerNeeds, setDailyPowerNeeds] = useState(20.0)
  const [batteryCostPerKwh] = useState(1000.0)
  const [missionDuration, setMissionDuration] = useState(500.0)
  
  // Sun position (static for now)
  const [sunAzimuth, setSunAzimuth] = useState(0.0)
  const [sunAltitude, setSunAltitude] = useState(45.0)
  
  // Results
  const [results, setResults] = useState<SolarAnalysisResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await analyzeSolarPotential({
        roi,
        dataset,
        sun_azimuth: sunAzimuth,
        sun_altitude: sunAltitude,
        panel_efficiency: panelEfficiency,
        panel_area_m2: panelArea,
        battery_capacity_kwh: batteryCapacity,
        daily_power_needs_kwh: dailyPowerNeeds,
        battery_cost_per_kwh: batteryCostPerKwh,
        mission_duration_days: missionDuration,
      })
      
      setResults(response)
    } catch (err) {
      console.error('Solar analysis failed:', err)
      setError(err instanceof Error ? err.message : 'Failed to analyze solar potential')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold">Solar Potential Analysis</h2>
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="px-6 py-2 bg-mars-orange hover:bg-orange-600 text-white rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Analyzing...' : 'Analyze Solar Potential'}
        </button>
      </div>
      <div className="flex items-center space-x-2">
        <button
          onClick={async () => { const data = await getExampleROIs(); setExamples(data) }}
          className="px-3 py-2 bg-gray-700 text-white rounded"
        >
          Fill with Example ROI
        </button>
        {examples.length > 0 && (
          <select
            onChange={(e) => {
              const sel = examples.find(x => x.id === e.target.value)
              if (sel) { setROI(sel.bbox); setDataset(sel.dataset) }
            }}
            className="bg-gray-700 text-white px-2 py-2 rounded"
            defaultValue=""
          >
            <option value="" disabled>Select example</option>
            {examples.map(x => (
              <option key={x.id} value={x.id}>{x.name}</option>
            ))}
          </select>
        )}
      </div>

      {error && (
        <div className="p-4 bg-red-900/50 border border-red-700 rounded-md">
          <p className="text-red-300 font-semibold">Error</p>
          <p className="text-red-400 text-sm mt-1">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="text-xl font-semibold">Region of Interest</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Latitude Min</label>
                <input
                  type="number"
                  value={isNaN(roi.lat_min) ? '' : roi.lat_min}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) setROI({ ...roi, lat_min: val })
                  }}
                  className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
                  step="0.1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Latitude Max</label>
                <input
                  type="number"
                  value={isNaN(roi.lat_max) ? '' : roi.lat_max}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) setROI({ ...roi, lat_max: val })
                  }}
                  className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
                  step="0.1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Longitude Min</label>
                <input
                  type="number"
                  value={isNaN(roi.lon_min) ? '' : roi.lon_min}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) setROI({ ...roi, lon_min: val })
                  }}
                  className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
                  step="0.1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Longitude Max</label>
                <input
                  type="number"
                  value={isNaN(roi.lon_max) ? '' : roi.lon_max}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) setROI({ ...roi, lon_max: val })
                  }}
                  className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
                  step="0.1"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                <option value="mola">MOLA</option>
                <option value="hirise">HiRISE</option>
                <option value="ctx">CTX</option>
              </select>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="text-xl font-semibold">Solar Panel Configuration</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Panel Efficiency: {(panelEfficiency * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.5"
                step="0.01"
                value={panelEfficiency}
                onChange={(e) => setPanelEfficiency(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Panel Area: {panelArea.toFixed(0)} m²
              </label>
              <input
                type="range"
                min="10"
                max="500"
                step="10"
                value={panelArea}
                onChange={(e) => setPanelArea(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="text-xl font-semibold">Mission Parameters</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Battery Capacity: {batteryCapacity.toFixed(0)} kWh
              </label>
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={batteryCapacity}
                onChange={(e) => setBatteryCapacity(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Daily Power Needs: {dailyPowerNeeds.toFixed(1)} kWh/day
              </label>
              <input
                type="range"
                min="5"
                max="50"
                step="1"
                value={dailyPowerNeeds}
                onChange={(e) => setDailyPowerNeeds(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Mission Duration: {missionDuration.toFixed(0)} days
              </label>
              <input
                type="range"
                min="100"
                max="1000"
                step="50"
                value={missionDuration}
                onChange={(e) => setMissionDuration(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="text-xl font-semibold">Sun Position</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Sun Azimuth: {sunAzimuth.toFixed(0)}° (0° = North)
              </label>
              <input
                type="range"
                min="0"
                max="360"
                step="15"
                value={sunAzimuth}
                onChange={(e) => setSunAzimuth(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Sun Altitude: {sunAltitude.toFixed(0)}° (0° = Horizon)
              </label>
              <input
                type="range"
                min="0"
                max="90"
                step="5"
                value={sunAltitude}
                onChange={(e) => setSunAltitude(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {results && (
            <>
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Solar Potential Statistics</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Mean Potential</div>
                    <div className="text-2xl font-bold text-white">
                      {(results.statistics.mean * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Max Potential</div>
                    <div className="text-2xl font-bold text-white">
                      {(results.statistics.max * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Mean Irradiance</div>
                    <div className="text-2xl font-bold text-white">
                      {results.statistics.mean_irradiance_w_per_m2.toFixed(0)} <span className="text-sm">W/m²</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Solar Potential Map</h3>
                <div className="h-[600px] w-full rounded-md overflow-hidden border border-gray-700 bg-gray-900">
                  <TerrainMap
                    roi={roi}
                    dataset={dataset}
                    overlayType="solar"
                    overlayOptions={{
                      colormap: 'plasma',
                      sunAzimuth: sunAzimuth,
                      sunAltitude: sunAltitude
                    }}
                    showSites={false}
                    showWaypoints={false}
                  />
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

              <SolarImpactPanel
                impacts={results.mission_impacts}
                dailyPowerNeeds={dailyPowerNeeds}
              />
            </>
          )}

          {!results && !loading && (
            <div className="bg-gray-800 rounded-lg p-12 text-center">
              <p className="text-gray-400 text-lg">
                Configure parameters and click "Analyze Solar Potential" to begin
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

