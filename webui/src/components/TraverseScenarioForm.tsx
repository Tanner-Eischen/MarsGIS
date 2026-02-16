import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { runTraverseScenario, TraverseScenarioRequest, TraverseScenarioResponse } from '../services/api'
import { ChevronDown, ChevronUp, Route, Clock, AlertCircle, AlertTriangle, MapPin, Navigation } from 'lucide-react'

const ROUTE_PRESETS = [
  { value: '', label: 'No preset (custom)' },
  { value: 'safest', label: 'Safest Route' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'direct', label: 'Most Direct' },
]

export default function TraverseScenarioForm() {
  // Required fields
  const [startSiteId, setStartSiteId] = useState<number>(1)
  const [endSiteId, setEndSiteId] = useState<number>(2)
  const [analysisDir, setAnalysisDir] = useState<string>('data/output')

  // Optional preset
  const [presetId, setPresetId] = useState<string>('')

  // Collapsible: Start Coordinates Override
  const [showCoordinates, setShowCoordinates] = useState<boolean>(false)
  const [startLat, setStartLat] = useState<number | undefined>(undefined)
  const [startLon, setStartLon] = useState<number | undefined>(undefined)

  // Collapsible: Rover Capabilities
  const [showCapabilities, setShowCapabilities] = useState<boolean>(false)
  const [maxSlopeDeg, setMaxSlopeDeg] = useState<number>(25)
  const [maxRoughness, setMaxRoughness] = useState<number>(100)

  // Results state
  const [results, setResults] = useState<TraverseScenarioResponse | null>(null)

  const mutation = useMutation({
    mutationFn: async (request: TraverseScenarioRequest) => {
      const response = await runTraverseScenario(request)
      return response
    },
    onSuccess: (data) => {
      setResults(data)
    },
    onError: () => {
      setResults(null)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setResults(null)

    const request: TraverseScenarioRequest = {
      start_site_id: startSiteId,
      end_site_id: endSiteId,
      analysis_dir: analysisDir,
    }

    // Add preset if selected
    if (presetId) {
      request.preset_id = presetId
    }

    // Add rover capabilities if section is shown
    if (showCapabilities) {
      request.rover_capabilities = {
        max_slope_deg: maxSlopeDeg,
        max_roughness: maxRoughness,
      }
    }

    // Add start coordinates if provided
    if (showCoordinates && startLat !== undefined && startLon !== undefined) {
      request.start_lat = startLat
      request.start_lon = startLon
    }

    mutation.mutate(request)
  }

  // Risk score color coding
  const getRiskColor = (risk: number): string => {
    const percentage = risk * 100
    if (percentage < 30) return 'text-green-400'
    if (percentage < 60) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getRiskBgColor = (risk: number): string => {
    const percentage = risk * 100
    if (percentage < 30) return 'bg-green-900/30 border-green-500/30'
    if (percentage < 60) return 'bg-yellow-900/30 border-yellow-500/30'
    return 'bg-red-900/30 border-red-500/30'
  }

  return (
    <div className="space-y-6 text-sm">
      {/* Form Section */}
      <form onSubmit={handleSubmit} className="glass-panel p-6 rounded-lg space-y-5">
        <div className="flex items-center justify-between border-b border-gray-700/50 pb-3">
          <div className="flex items-center gap-2">
            <Route className="w-5 h-5 text-green-400" />
            <h2 className="text-lg font-bold text-white tracking-wider">ROVER_TRAVERSE_PLANNING</h2>
          </div>
        </div>

        {/* Site Selection */}
        <div className="space-y-3">
          <label className="text-xs font-bold text-gray-400 uppercase">Route Endpoints</label>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">START SITE ID</label>
              <input
                type="number"
                value={startSiteId}
                onChange={(e) => setStartSiteId(parseInt(e.target.value) || 1)}
                className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-green-500 focus:outline-none"
                min="1"
                required
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">END SITE ID</label>
              <input
                type="number"
                value={endSiteId}
                onChange={(e) => setEndSiteId(parseInt(e.target.value) || 2)}
                className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-green-500 focus:outline-none"
                min="1"
                required
              />
            </div>
          </div>
        </div>

        {/* Analysis Directory */}
        <div>
          <label className="block text-xs font-bold text-gray-400 uppercase mb-2">Analysis Directory</label>
          <input
            type="text"
            value={analysisDir}
            onChange={(e) => setAnalysisDir(e.target.value)}
            className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-green-500 focus:outline-none"
            placeholder="data/output"
            required
          />
        </div>

        {/* Route Preset */}
        <div>
          <label className="block text-xs font-bold text-gray-400 uppercase mb-2">Route Preset</label>
          <select
            value={presetId}
            onChange={(e) => setPresetId(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 text-white px-3 py-2 rounded text-sm focus:border-green-500 focus:outline-none"
          >
            {ROUTE_PRESETS.map((preset) => (
              <option key={preset.value} value={preset.value}>
                {preset.label}
              </option>
            ))}
          </select>
        </div>

        {/* Collapsible: Start Coordinates Override */}
        <div className="border border-gray-700 rounded-lg overflow-hidden">
          <button
            type="button"
            onClick={() => setShowCoordinates(!showCoordinates)}
            className="w-full flex items-center justify-between px-4 py-3 bg-gray-800/50 hover:bg-gray-800 transition-colors"
          >
            <span className="text-xs font-bold text-gray-400 uppercase">Start Coordinates Override</span>
            {showCoordinates ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          {showCoordinates && (
            <div className="p-4 bg-gray-900/30 space-y-3">
              <p className="text-xs text-gray-500 mb-2">Override the starting coordinates instead of using site center</p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">START LAT</label>
                  <input
                    type="number"
                    value={startLat ?? ''}
                    onChange={(e) => setStartLat(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="e.g., 18.4"
                    className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-green-500 focus:outline-none placeholder-gray-600"
                    step="0.001"
                    min="-90"
                    max="90"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">START LON</label>
                  <input
                    type="number"
                    value={startLon ?? ''}
                    onChange={(e) => setStartLon(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="e.g., 77.5"
                    className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-green-500 focus:outline-none placeholder-gray-600"
                    step="0.001"
                    min="0"
                    max="360"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Collapsible: Rover Capabilities */}
        <div className="border border-gray-700 rounded-lg overflow-hidden">
          <button
            type="button"
            onClick={() => setShowCapabilities(!showCapabilities)}
            className="w-full flex items-center justify-between px-4 py-3 bg-gray-800/50 hover:bg-gray-800 transition-colors"
          >
            <span className="text-xs font-bold text-gray-400 uppercase">Rover Capabilities</span>
            {showCapabilities ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          {showCapabilities && (
            <div className="p-4 bg-gray-900/30 space-y-3">
              <div>
                <label className="block text-xs text-gray-500 mb-1">MAX SLOPE (degrees)</label>
                <input
                  type="number"
                  value={maxSlopeDeg}
                  onChange={(e) => setMaxSlopeDeg(parseFloat(e.target.value) || 25)}
                  className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-green-500 focus:outline-none"
                  step="0.5"
                  min="0"
                  max="90"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">MAX ROUGHNESS</label>
                <input
                  type="number"
                  value={maxRoughness}
                  onChange={(e) => setMaxRoughness(parseFloat(e.target.value) || 100)}
                  className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-green-500 focus:outline-none"
                  step="1"
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
          {mutation.isPending ? 'PLANNING...' : 'PLAN TRAVERSE'}
        </button>
      </form>

      {/* Error Display */}
      {mutation.isError && (
        <div className="glass-panel p-4 rounded-lg border border-red-500/50 bg-red-900/20">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span className="font-bold">Traverse Planning Failed</span>
          </div>
          <p className="mt-2 text-sm text-red-300">
            {(mutation.error as any)?.response?.data?.detail || (mutation.error as any)?.message || 'An unexpected error occurred'}
          </p>
        </div>
      )}

      {/* Results Section */}
      {results && (
        <div className="space-y-4">
          {/* Route ID and Metadata Badge */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-3 py-1 bg-green-900/50 border border-green-500/30 rounded text-xs text-green-300 font-mono">
              ROUTE: {results.route_id.slice(0, 8)}...
            </span>
            <span className="px-3 py-1 bg-cyan-900/50 border border-cyan-500/30 rounded text-xs text-cyan-300">
              {results.waypoints.length} waypoint{results.waypoints.length !== 1 ? 's' : ''}
            </span>
            {results.metadata?.mode && (
              <span className={`px-3 py-1 rounded text-xs font-mono ${
                results.metadata.mode === 'demo' 
                  ? 'bg-yellow-900/50 border border-yellow-500/30 text-yellow-300'
                  : 'bg-blue-900/50 border border-blue-500/30 text-blue-300'
              }`}>
                {results.metadata.mode.toUpperCase()} MODE
              </span>
            )}
          </div>

          {/* Route Summary Card */}
          <div className={`glass-panel p-6 rounded-lg border ${getRiskBgColor(results.risk_score)}`}>
            <div className="flex items-center gap-2 mb-4">
              <Navigation className="w-5 h-5 text-green-400" />
              <h3 className="text-lg font-bold text-green-400 tracking-wider">ROUTE SUMMARY</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-500 mb-1">ROUTE ID</div>
                <div className="text-sm font-mono text-white truncate" title={results.route_id}>
                  {results.route_id.slice(0, 12)}...
                </div>
              </div>
              <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-500 mb-1">TOTAL DISTANCE</div>
                <div className="text-lg font-mono text-cyan-400">
                  {results.total_distance_m >= 1000
                    ? `${(results.total_distance_m / 1000).toFixed(2)} km`
                    : `${results.total_distance_m.toFixed(1)} m`}
                </div>
              </div>
              <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                <div className="flex items-center gap-1 text-xs text-gray-500 mb-1">
                  <Clock className="w-3 h-3" />
                  EST. TIME
                </div>
                <div className="text-lg font-mono text-white">
                  {results.estimated_time_h < 1
                    ? `${(results.estimated_time_h * 60).toFixed(0)} min`
                    : `${results.estimated_time_h.toFixed(1)} hrs`}
                </div>
              </div>
              <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                <div className="flex items-center gap-1 text-xs text-gray-500 mb-1">
                  <AlertTriangle className="w-3 h-3" />
                  RISK SCORE
                </div>
                <div className={`text-lg font-mono ${getRiskColor(results.risk_score)}`}>
                  {(results.risk_score * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Waypoints Table */}
          <div className="glass-panel p-6 rounded-lg">
            <div className="flex items-center gap-2 mb-4">
              <MapPin className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-bold text-cyan-400 tracking-wider">WAYPOINTS</h3>
            </div>
            <div className="overflow-x-auto max-h-80 custom-scrollbar">
              <table className="w-full text-xs font-mono">
                <thead className="sticky top-0 bg-gray-900">
                  <tr className="border-b border-gray-700 text-gray-500">
                    <th className="text-left p-2">ID</th>
                    <th className="text-left p-2">X (meters)</th>
                    <th className="text-left p-2">Y (meters)</th>
                    <th className="text-left p-2">TOLERANCE (m)</th>
                  </tr>
                </thead>
                <tbody>
                  {results.waypoints.map((waypoint, index) => (
                    <tr
                      key={waypoint.waypoint_id}
                      className={`border-b border-gray-800 hover:bg-gray-800/50 ${
                        index === 0
                          ? 'bg-green-900/20'
                          : index === results.waypoints.length - 1
                          ? 'bg-red-900/20'
                          : ''
                      }`}
                    >
                      <td className="p-2">
                        <span className={`${
                          index === 0
                            ? 'text-green-400'
                            : index === results.waypoints.length - 1
                            ? 'text-red-400'
                            : 'text-white'
                        }`}>
                          #{waypoint.waypoint_id}
                          {index === 0 && <span className="ml-2 text-xs text-green-500">(START)</span>}
                          {index === results.waypoints.length - 1 && <span className="ml-2 text-xs text-red-500">(END)</span>}
                        </span>
                      </td>
                      <td className="p-2 text-gray-400">{waypoint.x_meters.toFixed(2)}</td>
                      <td className="p-2 text-gray-400">{waypoint.y_meters.toFixed(2)}</td>
                      <td className="p-2 text-gray-500">{waypoint.tolerance_meters.toFixed(2)}</td>
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
