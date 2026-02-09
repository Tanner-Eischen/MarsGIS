import { useState, useEffect } from 'react'
import PresetsSelector from '../components/PresetsSelector'
import AdvancedWeightsPanel from '../components/AdvancedWeightsPanel'
import SiteScoresList from '../components/SiteScoresList'
import ExplainabilityPanel from '../components/ExplainabilityPanel'
import TerrainMap from '../components/TerrainMap'
import SaveProjectModal from '../components/SaveProjectModal'
import ExamplesDrawer from '../components/ExamplesDrawer'
import { apiFetch } from '../lib/apiBase'

interface ROI {
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
}

interface Preset {
  id: string
  name: string
  description: string
  scope: string
  weights: Record<string, number>
}

interface SiteScore {
  site_id: number
  rank: number
  total_score: number
  components: Record<string, number>
  explanation: string
  geometry: any
  centroid_lat: number
  centroid_lon: number
  area_km2: number
}

export default function DecisionLab() {
  const [roi, setROI] = useState<ROI | null>({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [dataset, setDataset] = useState('mola_200m')
  const [selectedPreset, setSelectedPreset] = useState<string>('balanced')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [customWeights, setCustomWeights] = useState<Record<string, number>>({})
  const [sites, setSites] = useState<SiteScore[]>([])
  const [selectedSite, setSelectedSite] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [showExamples, setShowExamples] = useState(false)
  const [explainMap, setExplainMap] = useState(false)

  // Fetch presets on mount
  const [presets, setPresets] = useState<Preset[]>([])
  
  useEffect(() => {
    apiFetch('/analysis/presets')
      .then(res => res.json())
      .then(data => setPresets(data.site_presets || []))
      .catch(err => console.error('Failed to load presets:', err))
  }, [])

  // Run analysis when ROI or preset changes
  const runAnalysis = async () => {
    if (!roi) return
    
    setLoading(true)
    
    try {
      const request = {
        roi,
        dataset,
        preset_id: selectedPreset,
        custom_weights: Object.keys(customWeights).length > 0 ? customWeights : null,
        threshold: 0.6
      }
      
      const response = await apiFetch('/analysis/site-scores', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      })
      
      if (response.ok) {
        const data = await response.json()
        setSites(data)
      } else {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }))
        console.error('Analysis failed:', errorData.detail || response.statusText)
        alert(`Analysis failed: ${errorData.detail || response.statusText}`)
      }
    } catch (error) {
      console.error('Analysis error:', error)
      alert(`Analysis error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col bg-gray-900 text-gray-100" style={{ height: 'calc(100vh - 8rem)' }}>
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-4 mb-4">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">Mars Landing Site Decision Lab</h1>
            <p className="text-gray-400 text-sm">
              Explore landing sites using preset criteria or customize your own
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowExamples(true)}
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold text-sm"
            >
              Load Example
            </button>
            <label className="flex items-center gap-2 bg-gray-700 px-4 py-2 rounded cursor-pointer">
              <input
                type="checkbox"
                checked={explainMap}
                onChange={(e) => setExplainMap(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm">Explain this map</span>
            </label>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Controls */}
        <div className="w-96 bg-gray-800 border-r border-gray-700 overflow-y-auto">
          {/* ROI Selection */}
          <div className="p-4 border-b border-gray-700">
            <h3 className="font-semibold mb-2">Region of Interest</h3>
            <div className="space-y-2 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  placeholder="Lat Min"
                  value={roi?.lat_min || ''}
                  className="bg-gray-700 p-2 rounded text-white"
                  onChange={(e) => setROI(prev => ({ ...prev!, lat_min: parseFloat(e.target.value) }))}
                />
                <input
                  type="number"
                  placeholder="Lat Max"
                  value={roi?.lat_max || ''}
                  className="bg-gray-700 p-2 rounded text-white"
                  onChange={(e) => setROI(prev => ({ ...prev!, lat_max: parseFloat(e.target.value) }))}
                />
                <input
                  type="number"
                  placeholder="Lon Min"
                  value={roi?.lon_min || ''}
                  className="bg-gray-700 p-2 rounded text-white"
                  onChange={(e) => setROI(prev => ({ ...prev!, lon_min: parseFloat(e.target.value) }))}
                />
                <input
                  type="number"
                  placeholder="Lon Max"
                  value={roi?.lon_max || ''}
                  className="bg-gray-700 p-2 rounded text-white"
                  onChange={(e) => setROI(prev => ({ ...prev!, lon_max: parseFloat(e.target.value) }))}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Dataset</label>
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  className="w-full bg-gray-700 p-2 rounded text-white text-sm"
                >
                  <option value="mola">MOLA</option>
                  <option value="mola_200m">MOLA 200m</option>
                  <option value="hirise">HiRISE</option>
                  <option value="ctx">CTX</option>
                </select>
              </div>
            </div>
          </div>

          {/* Preset Selection */}
          <PresetsSelector
            presets={presets}
            selected={selectedPreset}
            onSelect={setSelectedPreset}
          />

          {/* Advanced Weights */}
          <div className="p-4 border-b border-gray-700">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-2"
            >
              <span>{showAdvanced ? '▼' : '▶'}</span>
              <span>Advanced Weights</span>
            </button>
            
            {showAdvanced && (
              <AdvancedWeightsPanel
                weights={customWeights}
                onChange={setCustomWeights}
                presetWeights={presets.find(p => p.id === selectedPreset)?.weights || {}}
              />
            )}
          </div>

          {/* Run Analysis Button */}
          <div className="p-4 space-y-2">
            <button
              onClick={runAnalysis}
              disabled={!roi || loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold"
            >
              {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
            {sites.length > 0 && (
              <>
                <button
                  onClick={() => setShowSaveModal(true)}
                  className="w-full bg-green-600 hover:bg-green-700 p-3 rounded font-semibold"
                >
                  Save as Project
                </button>
                <div className="flex gap-2">
                  <button
                    onClick={async () => {
                      try {
                        const response = await apiFetch('/export/suitability-geotiff', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({
                            roi,
                            dataset,
                            preset_id: selectedPreset
                          })
                        })
                        if (response.ok) {
                          const blob = await response.blob()
                          const url = window.URL.createObjectURL(blob)
                          const a = document.createElement('a')
                          a.href = url
                          a.download = `suitability_${Date.now()}.tif`
                          a.click()
                          window.URL.revokeObjectURL(url)
                        } else {
                          alert('Export failed')
                        }
                      } catch (error) {
                        console.error('Export error:', error)
                        alert('Export failed')
                      }
                    }}
                    className="flex-1 bg-purple-600 hover:bg-purple-700 p-2 rounded text-sm font-semibold"
                  >
                    Export GeoTIFF
                  </button>
                  <button
                    onClick={async () => {
                      try {
                        const response = await apiFetch('/export/report', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ format: 'markdown' })
                        })
                        if (response.ok) {
                          const blob = await response.blob()
                          const url = window.URL.createObjectURL(blob)
                          const a = document.createElement('a')
                          a.href = url
                          a.download = `analysis_report_${Date.now()}.md`
                          a.click()
                          window.URL.revokeObjectURL(url)
                        } else {
                          alert('Report generation failed')
                        }
                      } catch (error) {
                        console.error('Report error:', error)
                        alert('Report generation failed')
                      }
                    }}
                    className="flex-1 bg-orange-600 hover:bg-orange-700 p-2 rounded text-sm font-semibold"
                  >
                    Export Report
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Site Scores List */}
          {sites.length > 0 && (
            <SiteScoresList
              sites={sites}
              selectedSite={selectedSite}
              onSelectSite={setSelectedSite}
            />
          )}
        </div>

        {/* Right Panel: Visualization and Explanation */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 p-4">
            {roi && (
              <TerrainMap
                roi={roi}
                dataset={dataset}
                showSites={sites.length > 0}
                showWaypoints={false}
                onSiteSelect={setSelectedSite}
                selectedSiteId={selectedSite}
              />
            )}
          </div>
          
          {/* Explainability Panel */}
          {selectedSite !== null && (
            <div className="p-4 border-t border-gray-700">
              <ExplainabilityPanel
                site={sites.find(s => s.site_id === selectedSite) || null}
                weights={presets.find(p => p.id === selectedPreset)?.weights || {}}
              />
            </div>
          )}
        </div>
      </div>

      {showSaveModal && (
        <SaveProjectModal
          roi={roi!}
          dataset={dataset}
          presetId={selectedPreset}
          selectedSites={selectedSite ? [selectedSite] : sites.slice(0, 5).map(s => s.site_id)}
          onClose={() => setShowSaveModal(false)}
          onSave={() => setShowSaveModal(false)}
        />
      )}

      {showExamples && (
        <ExamplesDrawer
          isOpen={showExamples}
          onClose={() => setShowExamples(false)}
          onSelectExample={(example) => {
            setROI(example.bbox)
            setDataset(example.dataset)
          }}
        />
      )}

      {explainMap && (
        <div className="fixed bottom-4 right-4 bg-gray-800 border border-gray-700 rounded-lg p-4 max-w-md z-40">
          <h4 className="font-semibold mb-2">Map Explanation</h4>
          <p className="text-sm text-gray-300 mb-2">
            Colors represent suitability scores (0-1), where higher values indicate better landing sites.
          </p>
          <p className="text-sm text-gray-300 mb-2">
            Current preset: <span className="font-semibold">{presets.find(p => p.id === selectedPreset)?.name || selectedPreset}</span>
          </p>
          <p className="text-sm text-gray-300">
            This preset optimizes for: {presets.find(p => p.id === selectedPreset)?.description || 'balanced criteria'}
          </p>
        </div>
      )}
    </div>
  )
}

