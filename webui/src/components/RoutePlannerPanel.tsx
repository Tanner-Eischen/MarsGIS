import { useState, useEffect } from 'react'

interface Site {
  site_id: number
  lat: number
  lon: number
  suitability_score: number
  rank: number
}

interface Preset {
  id: string
  name: string
  description: string
  scope: string
}

interface RoutePlannerPanelProps {
  sites: Site[]
  analysisDir: string
  onPlanRoute: (params: {
    startSiteId: number
    endSiteId: number
    analysisDir: string
    presetId: string
    sunAzimuth?: number
    sunAltitude?: number
  }) => void
  loading: boolean
}

export default function RoutePlannerPanel({
  sites,
  analysisDir,
  onPlanRoute,
  loading
}: RoutePlannerPanelProps) {
  const [startSiteId, setStartSiteId] = useState<number>(sites[0]?.site_id || 1)
  const [endSiteId, setEndSiteId] = useState<number>(sites[1]?.site_id || 2)
  const [presetId, setPresetId] = useState<string>('balanced')
  const [presets, setPresets] = useState<Preset[]>([])
  const [showSunControls, setShowSunControls] = useState(false)
  const [sunAzimuth, setSunAzimuth] = useState(315)
  const [sunAltitude, setSunAltitude] = useState(45)

  useEffect(() => {
    fetch('http://localhost:5000/api/v1/analysis/presets?scope=route')
      .then(res => res.json())
      .then(data => {
        const routePresets = data.route_presets || []
        setPresets(routePresets)
        if (routePresets.length > 0) {
          setPresetId(routePresets[0].id)
        }
      })
      .catch(err => console.error('Failed to load presets:', err))
  }, [])

  const handlePlanRoute = () => {
    onPlanRoute({
      startSiteId,
      endSiteId,
      analysisDir,
      presetId,
      sunAzimuth: showSunControls ? sunAzimuth : undefined,
      sunAltitude: showSunControls ? sunAltitude : undefined,
    })
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Route Planning</h3>
      
      <div>
        <label className="block text-sm font-medium mb-2">Analysis Directory</label>
        <input
          type="text"
          value={analysisDir}
          readOnly
          className="w-full bg-gray-700 text-gray-400 px-4 py-2 rounded-md"
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
                Site {site.site_id} (Rank {site.rank})
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
                Site {site.site_id} (Rank {site.rank})
              </option>
            ))}
          </select>
        </div>
      </div>

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

      <div>
        <button
          onClick={() => setShowSunControls(!showSunControls)}
          className="text-sm text-blue-400 hover:text-blue-300"
        >
          {showSunControls ? '▼' : '▶'} Sun Position Controls
        </button>
        {showSunControls && (
          <div className="mt-2 space-y-2">
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Sun Azimuth: {sunAzimuth}°
              </label>
              <input
                type="range"
                min="0"
                max="360"
                value={sunAzimuth}
                onChange={(e) => setSunAzimuth(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Sun Altitude: {sunAltitude}°
              </label>
              <input
                type="range"
                min="0"
                max="90"
                value={sunAltitude}
                onChange={(e) => setSunAltitude(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        )}
      </div>

      <button
        onClick={handlePlanRoute}
        disabled={loading || sites.length < 2}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold"
      >
        {loading ? 'Planning Route...' : 'Plan Route'}
      </button>
    </div>
  )
}

