import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { runLandingScenario, LandingScenarioRequest, LandingScenarioResponse, SiteCandidate } from '../services/api'
import { ChevronDown, ChevronUp, MapPin, Target, AlertCircle, CheckCircle2 } from 'lucide-react'

// Default ROI for quick testing (Jezero Crater area)
const DEFAULT_ROI = {
  lat_min: 18.0,
  lat_max: 18.6,
  lon_min: 77.0,
  lon_max: 77.8,
}

const DATASETS = [
  { value: 'mola', label: 'MOLA (Global)' },
  { value: 'mola_200m', label: 'MOLA 200m' },
  { value: 'hirise', label: 'HiRISE (High-Res)' },
  { value: 'ctx', label: 'CTX' },
]

export default function LandingScenarioForm() {
  // ROI state
  const [latMin, setLatMin] = useState<number>(18.0)
  const [latMax, setLatMax] = useState<number>(18.6)
  const [lonMin, setLonMin] = useState<number>(77.0)
  const [lonMax, setLonMax] = useState<number>(77.8)

  // Form state
  const [dataset, setDataset] = useState<string>('mola')
  const [suitabilityThreshold, setSuitabilityThreshold] = useState<number>(0.7)
  const [showConstraints, setShowConstraints] = useState<boolean>(false)
  const [maxSlopeDeg, setMaxSlopeDeg] = useState<number | undefined>(undefined)
  const [minAreaKm2, setMinAreaKm2] = useState<number | undefined>(undefined)

  // Results state
  const [results, setResults] = useState<LandingScenarioResponse | null>(null)

  const mutation = useMutation({
    mutationFn: async (request: LandingScenarioRequest) => {
      const response = await runLandingScenario(request)
      return response
    },
    onSuccess: (data) => {
      setResults(data)
    },
    onError: (error: any) => {
      setResults(null)
    },
  })

  const handleUseDefault = () => {
    setLatMin(DEFAULT_ROI.lat_min)
    setLatMax(DEFAULT_ROI.lat_max)
    setLonMin(DEFAULT_ROI.lon_min)
    setLonMax(DEFAULT_ROI.lon_max)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setResults(null)

    const request: LandingScenarioRequest = {
      roi: {
        lat_min: latMin,
        lat_max: latMax,
        lon_min: lonMin,
        lon_max: lonMax,
      },
      dataset,
      suitability_threshold: suitabilityThreshold,
    }

    // Add constraints if specified
    if (maxSlopeDeg !== undefined || minAreaKm2 !== undefined) {
      request.constraints = {}
      if (maxSlopeDeg !== undefined) request.constraints.max_slope_deg = maxSlopeDeg
      if (minAreaKm2 !== undefined) request.constraints.min_area_km2 = minAreaKm2
    }

    mutation.mutate(request)
  }

  return (
    <div className="space-y-6 text-sm">
      {/* Form Section */}
      <form onSubmit={handleSubmit} className="glass-panel p-6 rounded-lg space-y-5">
        <div className="flex items-center justify-between border-b border-gray-700/50 pb-3">
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-purple-400" />
            <h2 className="text-lg font-bold text-white tracking-wider">LANDING_SITE_ANALYSIS</h2>
          </div>
        </div>

        {/* ROI Input Section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-gray-400 uppercase">Region of Interest (ROI)</label>
            <button
              type="button"
              onClick={handleUseDefault}
              className="px-3 py-1 bg-gray-800 text-xs text-cyan-400 border border-cyan-500/30 rounded hover:bg-gray-700 transition-colors"
            >
              Use Default
            </button>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">LAT MIN</label>
              <input
                type="number"
                value={latMin}
                onChange={(e) => setLatMin(parseFloat(e.target.value))}
                className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none"
                step="0.1"
                min="-90"
                max="90"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">LAT MAX</label>
              <input
                type="number"
                value={latMax}
                onChange={(e) => setLatMax(parseFloat(e.target.value))}
                className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none"
                step="0.1"
                min="-90"
                max="90"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">LON MIN</label>
              <input
                type="number"
                value={lonMin}
                onChange={(e) => setLonMin(parseFloat(e.target.value))}
                className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none"
                step="0.1"
                min="0"
                max="360"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">LON MAX</label>
              <input
                type="number"
                value={lonMax}
                onChange={(e) => setLonMax(parseFloat(e.target.value))}
                className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none"
                step="0.1"
                min="0"
                max="360"
              />
            </div>
          </div>
        </div>

        {/* Dataset Selector */}
        <div>
          <label className="block text-xs font-bold text-gray-400 uppercase mb-2">Dataset</label>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 text-white px-3 py-2 rounded text-sm focus:border-cyan-500 focus:outline-none"
          >
            {DATASETS.map((ds) => (
              <option key={ds.value} value={ds.value}>
                {ds.label}
              </option>
            ))}
          </select>
        </div>

        {/* Suitability Threshold Slider */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-bold text-gray-400 uppercase">Suitability Threshold</label>
            <span className="text-sm font-mono text-cyan-400">{suitabilityThreshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={suitabilityThreshold}
            onChange={(e) => setSuitabilityThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0.0 (Permissive)</span>
            <span>1.0 (Strict)</span>
          </div>
        </div>

        {/* Collapsible Constraints Section */}
        <div className="border border-gray-700 rounded-lg overflow-hidden">
          <button
            type="button"
            onClick={() => setShowConstraints(!showConstraints)}
            className="w-full flex items-center justify-between px-4 py-3 bg-gray-800/50 hover:bg-gray-800 transition-colors"
          >
            <span className="text-xs font-bold text-gray-400 uppercase">Advanced Constraints</span>
            {showConstraints ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          {showConstraints && (
            <div className="p-4 bg-gray-900/30 space-y-3">
              <div>
                <label className="block text-xs text-gray-500 mb-1">MAX SLOPE (degrees)</label>
                <input
                  type="number"
                  value={maxSlopeDeg ?? ''}
                  onChange={(e) => setMaxSlopeDeg(e.target.value ? parseFloat(e.target.value) : undefined)}
                  placeholder="e.g., 15"
                  className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none placeholder-gray-600"
                  step="0.5"
                  min="0"
                  max="90"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">MIN AREA (km²)</label>
                <input
                  type="number"
                  value={minAreaKm2 ?? ''}
                  onChange={(e) => setMinAreaKm2(e.target.value ? parseFloat(e.target.value) : undefined)}
                  placeholder="e.g., 1.0"
                  className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none placeholder-gray-600"
                  step="0.1"
                  min="0"
                />
              </div>
            </div>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={mutation.isPending}
          className="w-full bg-green-600 hover:bg-green-500 text-white px-4 py-3 rounded font-bold tracking-wider transition-colors disabled:opacity-50 shadow-[0_0_15px_rgba(34,197,94,0.2)]"
        >
          {mutation.isPending ? 'ANALYZING...' : 'RUN LANDING ANALYSIS'}
        </button>
      </form>

      {/* Error Display */}
      {mutation.isError && (
        <div className="glass-panel p-4 rounded-lg border border-red-500/50 bg-red-900/20">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span className="font-bold">Analysis Failed</span>
          </div>
          <p className="mt-2 text-sm text-red-300">
            {(mutation.error as any)?.response?.data?.detail || (mutation.error as any)?.message || 'An unexpected error occurred'}
          </p>
        </div>
      )}

      {/* Results Section */}
      {results && (
        <div className="space-y-4">
          {/* Scenario ID Badge */}
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 bg-purple-900/50 border border-purple-500/30 rounded text-xs text-purple-300 font-mono">
              SCENARIO: {results.scenario_id.slice(0, 8)}...
            </span>
            <span className="px-3 py-1 bg-cyan-900/50 border border-cyan-500/30 rounded text-xs text-cyan-300">
              {results.sites.length} site{results.sites.length !== 1 ? 's' : ''} found
            </span>
          </div>

          {/* Top Site Card */}
          {results.top_site && (
            <div className="glass-panel p-6 rounded-lg border border-green-500/30 bg-gradient-to-br from-green-900/20 to-transparent">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <h3 className="text-lg font-bold text-green-400 tracking-wider">TOP RECOMMENDATION</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">SITE ID</div>
                  <div className="text-lg font-mono text-white">#{results.top_site.site_id}</div>
                </div>
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">RANK</div>
                  <div className="text-lg font-mono text-green-400">#{results.top_site.rank}</div>
                </div>
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">SUITABILITY</div>
                  <div className="text-lg font-mono text-cyan-400">{(results.top_site.suitability_score * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">AREA</div>
                  <div className="text-lg font-mono text-white">{results.top_site.area_km2.toFixed(2)} km²</div>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">COORDINATES</div>
                  <div className="text-sm font-mono text-gray-300">
                    {results.top_site.lat.toFixed(4)}°, {results.top_site.lon.toFixed(4)}°
                  </div>
                </div>
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">SLOPE</div>
                  <div className="text-sm font-mono text-gray-300">{results.top_site.mean_slope_deg.toFixed(2)}°</div>
                </div>
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">ELEVATION</div>
                  <div className="text-sm font-mono text-gray-300">{results.top_site.mean_elevation_m.toFixed(1)} m</div>
                </div>
                <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">ROUGHNESS</div>
                  <div className="text-sm font-mono text-gray-300">{results.top_site.mean_roughness.toFixed(2)}</div>
                </div>
              </div>
            </div>
          )}

          {/* All Sites Table */}
          <div className="glass-panel p-6 rounded-lg">
            <div className="flex items-center gap-2 mb-4">
              <MapPin className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-bold text-cyan-400 tracking-wider">ALL_CANDIDATE_SITES</h3>
            </div>
            <div className="overflow-x-auto max-h-80 custom-scrollbar">
              <table className="w-full text-xs font-mono">
                <thead className="sticky top-0 bg-gray-900">
                  <tr className="border-b border-gray-700 text-gray-500">
                    <th className="text-left p-2">RANK</th>
                    <th className="text-left p-2">SITE</th>
                    <th className="text-left p-2">SCORE</th>
                    <th className="text-left p-2">AREA (km²)</th>
                    <th className="text-left p-2">SLOPE (°)</th>
                    <th className="text-left p-2">ELEV (m)</th>
                    <th className="text-left p-2">COORDS</th>
                  </tr>
                </thead>
                <tbody>
                  {[...results.sites]
                    .sort((a, b) => a.rank - b.rank)
                    .map((site) => (
                      <tr
                        key={site.site_id}
                        className={`border-b border-gray-800 hover:bg-gray-800/50 ${
                          results.top_site && site.site_id === results.top_site.site_id
                            ? 'bg-green-900/20'
                            : ''
                        }`}
                      >
                        <td className="p-2 text-green-300">#{site.rank}</td>
                        <td className="p-2 text-white">{site.site_id}</td>
                        <td className="p-2">
                          <span
                            className={`${
                              site.suitability_score >= 0.8
                                ? 'text-green-400'
                                : site.suitability_score >= 0.5
                                ? 'text-yellow-400'
                                : 'text-red-400'
                            }`}
                          >
                            {(site.suitability_score * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="p-2 text-gray-400">{site.area_km2.toFixed(2)}</td>
                        <td className="p-2 text-gray-400">{site.mean_slope_deg.toFixed(2)}</td>
                        <td className="p-2 text-gray-400">{site.mean_elevation_m.toFixed(1)}</td>
                        <td className="p-2 text-gray-500">
                          {site.lat.toFixed(3)}°, {site.lon.toFixed(3)}°
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
