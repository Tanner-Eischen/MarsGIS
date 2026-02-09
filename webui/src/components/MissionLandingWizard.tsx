import { useState, useEffect } from 'react'
import { runLandingScenario, LandingScenarioRequest, LandingScenarioResponse } from '../services/api'
import TerrainMap from './TerrainMap'
import SaveProjectModal from './SaveProjectModal'
import { apiFetch } from '../lib/apiBase'

interface Preset {
  id: string
  name: string
  description: string
  scope: string
}

export default function MissionLandingWizard() {
  const [step, setStep] = useState<1 | 2 | 3>(1)
  const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [dataset, setDataset] = useState('mola_200m')
  const [presetId, setPresetId] = useState<string>('balanced')
  const [constraints, setConstraints] = useState({ max_slope_deg: 5.0, min_area_km2: 0.5 })
  const [presets, setPresets] = useState<Preset[]>([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<LandingScenarioResponse | null>(null)
  const [showSaveModal, setShowSaveModal] = useState(false)

  // Fetch presets on mount
  useEffect(() => {
    apiFetch('/analysis/presets?scope=site')
      .then(res => res.json())
      .then(data => {
        const sitePresets = data.site_presets || []
        setPresets(sitePresets)
        if (sitePresets.length > 0 && !presetId) {
          setPresetId(sitePresets[0].id)
        }
      })
      .catch(err => console.error('Failed to load presets:', err))
  }, [])

  const handleRunScenario = async () => {
    setLoading(true)
    try {
      const request: LandingScenarioRequest = {
        roi,
        dataset,
        preset_id: presetId,
        constraints,
        suitability_threshold: 0.7,
      }
      
      const response = await runLandingScenario(request)
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
          <span className="ml-2">ROI & Dataset</span>
        </div>
        <div className="w-16 h-0.5 bg-gray-700"></div>
        <div className={`flex items-center ${step >= 2 ? 'text-blue-400' : 'text-gray-500'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? 'bg-blue-600' : 'bg-gray-700'}`}>
            2
          </div>
          <span className="ml-2">Constraints</span>
        </div>
        <div className="w-16 h-0.5 bg-gray-700"></div>
        <div className={`flex items-center ${step >= 3 ? 'text-blue-400' : 'text-gray-500'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 3 ? 'bg-blue-600' : 'bg-gray-700'}`}>
            3
          </div>
          <span className="ml-2">Results</span>
        </div>
      </div>

      {/* Step 1: ROI + Dataset */}
      {step === 1 && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Step 1: Select Region of Interest and Dataset</h3>
          
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

          <div>
            <label className="block text-sm font-medium mb-2">Dataset</label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
            >
              <option value="mola">MOLA</option>
              <option value="mola_200m">MOLA 200m</option>
              <option value="hirise">HiRISE</option>
              <option value="ctx">CTX</option>
            </select>
          </div>

          <button
            onClick={() => setStep(2)}
            className="w-full bg-blue-600 hover:bg-blue-700 p-3 rounded font-semibold"
          >
            Next: Mission Constraints
          </button>
        </div>
      )}

      {/* Step 2: Constraints + Preset */}
      {step === 2 && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Step 2: Mission Constraints and Preset Selection</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Preset</label>
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
                  value={constraints.max_slope_deg}
                  onChange={(e) => setConstraints({ ...constraints, max_slope_deg: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
                  step="0.5"
                  min="0"
                  max="90"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Min Area (km²)</label>
                <input
                  type="number"
                  value={constraints.min_area_km2}
                  onChange={(e) => setConstraints({ ...constraints, min_area_km2: parseFloat(e.target.value) })}
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
              {loading ? 'Running Scenario...' : 'Run Scenario'}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Results */}
      {step === 3 && results && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Step 3: Results</h3>
          
          {results.top_site && (
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-lg mb-2">Top Site</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Site ID:</span> {results.top_site.site_id}
                </div>
                <div>
                  <span className="text-gray-400">Score:</span> {results.top_site.suitability_score.toFixed(3)}
                </div>
                <div>
                  <span className="text-gray-400">Area:</span> {results.top_site.area_km2.toFixed(2)} km²
                </div>
                <div>
                  <span className="text-gray-400">Location:</span> {results.top_site.lat.toFixed(2)}°N, {results.top_site.lon.toFixed(2)}°E
                </div>
              </div>
            </div>
          )}

          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-semibold mb-2">Ranked Sites ({results.sites.length} total)</h4>
            <div className="max-h-64 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-600">
                    <th className="text-left p-2">Rank</th>
                    <th className="text-left p-2">Site ID</th>
                    <th className="text-left p-2">Score</th>
                    <th className="text-left p-2">Area (km²)</th>
                    <th className="text-left p-2">Slope (°)</th>
                  </tr>
                </thead>
                <tbody>
                  {results.sites.slice(0, 10).map(site => (
                    <tr key={site.site_id} className="border-b border-gray-600">
                      <td className="p-2">{site.rank}</td>
                      <td className="p-2">{site.site_id}</td>
                      <td className="p-2">{site.suitability_score.toFixed(3)}</td>
                      <td className="p-2">{site.area_km2.toFixed(2)}</td>
                      <td className="p-2">{site.mean_slope_deg.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="h-96">
            <TerrainMap
              roi={roi}
              dataset={dataset}
              showSites={true}
              showWaypoints={false}
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setShowSaveModal(true)}
              className="flex-1 bg-green-600 hover:bg-green-700 p-3 rounded font-semibold"
            >
              Save as Project
            </button>
            <button
              onClick={() => {
                setStep(1)
                setResults(null)
              }}
              className="flex-1 bg-blue-600 hover:bg-blue-700 p-3 rounded font-semibold"
            >
              Start New Scenario
            </button>
          </div>
        </div>
      )}

      {showSaveModal && results && (
        <SaveProjectModal
          roi={roi}
          dataset={dataset}
          presetId={presetId}
          selectedSites={results.top_site ? [results.top_site.site_id] : []}
          onClose={() => setShowSaveModal(false)}
          onSave={() => setShowSaveModal(false)}
        />
      )}
    </div>
  )
}
