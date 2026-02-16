import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { planNavigation, NavigationRequest, getExampleROIs, ExampleROIItem, Waypoint } from '../services/api'
import MLSiteRecommendation from '../components/MLSiteRecommendation'
import { useGeoPlan } from '../context/GeoPlanContext'
import TerrainMap from '../components/TerrainMap'
import { AlertCircle, Box } from 'lucide-react'

// Generate UUID for task tracking
function generateTaskId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

// Convert waypoints from meters to approximate lat/lon ROI
// Mars: ~1 degree latitude ≈ 59.28 km, longitude varies with latitude
const MARS_METERS_PER_DEGREE_LAT = 59280
function computeRouteRoi(
  waypoints: Waypoint[],
  startLat: number,
  startLon: number
): { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null {
  if (!waypoints || waypoints.length === 0) return null

  // Validate input coordinates
  if (!Number.isFinite(startLat) || !Number.isFinite(startLon)) return null

  // Convert meters to approximate lat/lon offsets
  const metersPerDegreeLon = MARS_METERS_PER_DEGREE_LAT * Math.cos((startLat * Math.PI) / 180)

  const lats = waypoints.map((w) => startLat + w.y_meters / MARS_METERS_PER_DEGREE_LAT)
  const lons = waypoints.map((w) => startLon + w.x_meters / metersPerDegreeLon)

  // Filter out invalid coordinates
  const validLats = lats.filter(l => Number.isFinite(l))
  const validLons = lons.filter(l => Number.isFinite(l))

  if (validLats.length === 0 || validLons.length === 0) return null

  const padding = 0.02 // degrees padding around route
  return {
    lat_min: Math.min(...validLats) - padding,
    lat_max: Math.max(...validLats) + padding,
    lon_min: Math.min(...validLons) - padding,
    lon_max: Math.max(...validLons) + padding,
  }
}

export default function NavigationPlanning() {
  const navigate = useNavigate()
  const { landingSites, constructionSites, setRecommendedLandingSiteId } = useGeoPlan()
  const [siteId, setSiteId] = useState(1)
  const [startLat, setStartLat] = useState(40.0)
  const [startLon, setStartLon] = useState(180.0)
  const [strategy, setStrategy] = useState<'safest' | 'balanced' | 'direct'>('balanced')
  const [analysisDir, setAnalysisDir] = useState('data/output')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [examples, setExamples] = useState<ExampleROIItem[]>([])
  const [mlRecs, setMlRecs] = useState<any[] | null>(null)
  const [waypointsRefreshKey, setWaypointsRefreshKey] = useState(0)
  
  useEffect(() => {
    const sLat = localStorage.getItem('nav.startLat')
    const sLon = localStorage.getItem('nav.startLon')
    const dir = localStorage.getItem('nav.analysisDir')
    if (sLat) setStartLat(parseFloat(sLat))
    if (sLon) setStartLon(parseFloat(sLon))
    if (dir) setAnalysisDir(dir)
  }, [])
  useEffect(() => { localStorage.setItem('nav.startLat', String(startLat)) }, [startLat])
  useEffect(() => { localStorage.setItem('nav.startLon', String(startLon)) }, [startLon])
  useEffect(() => { localStorage.setItem('nav.analysisDir', analysisDir) }, [analysisDir])

  const mutation = useMutation({
    mutationFn: async (request: NavigationRequest & { task_id?: string }) => {
      const taskId = generateTaskId()
      const response = await planNavigation({ ...request, task_id: taskId })
      setTaskId(taskId)
      return response
    },
    onSuccess: (data) => {
      if (data.task_id) {
        setTaskId(data.task_id)
      }
      // Trigger waypoints GeoJSON refresh after new route is planned
      setWaypointsRefreshKey((k) => k + 1)
    },
    onError: () => {
      setTaskId(null)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setTaskId(null)
    
    // Validate distance to prevent excessive computation
    const site = constructionSites.find(s => s.site_id === siteId)
    if (site) {
      const latDiff = Math.abs(site.lat - startLat)
      const lonDiff = Math.abs(site.lon - startLon)
      const MAX_SPAN = 2.0  // degrees (~200 km on Mars)
      
      if (latDiff > MAX_SPAN || lonDiff > MAX_SPAN) {
        const distanceKm = Math.sqrt(latDiff**2 + lonDiff**2) * 59
        mutation.reset()
        mutation.mutate({
          site_id: -1, // Trigger error immediately without backend call
          start_lat: startLat,
          start_lon: startLon,
          strategy,
          analysis_dir: analysisDir,
        } as any)
        return
      }
    }
    
    mutation.mutate({
      site_id: siteId,
      start_lat: startLat,
      start_lon: startLon,
      strategy,
      analysis_dir: analysisDir,
    })
  }

  return (
    <div className="space-y-6 text-sm">
      <div className="glass-panel p-6 rounded-lg space-y-3">
        <div className="text-xs font-bold text-gray-400 uppercase mb-2">Start Point Selection</div>
        <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
          <label className="block text-xs text-gray-500 mb-1">AVAILABLE SITES</label>
          <select
            value={landingSites.find(s => Math.abs(s.lat - startLat) < 1e-6 && Math.abs(s.lon - startLon) < 1e-6)?.site_id || ''}
            onChange={(e) => {
              const id = parseInt(e.target.value)
              const site = landingSites.find(s => s.site_id === id)
              if (site) { setStartLat(site.lat); setStartLon(site.lon) }
            }}
            className="w-full bg-gray-800 text-white px-3 py-2 rounded text-sm border border-gray-600 focus:border-cyan-500 focus:outline-none"
          >
            <option value="" disabled>{landingSites.length > 0 ? 'Select starting site...' : 'No landing sites found'}</option>
            {landingSites.map(site => (
              <option key={site.site_id} value={site.site_id}>
                Site {site.site_id} (Rank {site.rank}, Score {site.suitability_score.toFixed(2)})
              </option>
            ))}
          </select>
        </div>
      </div>

      {(landingSites.length > 0 || constructionSites.length > 0) && (
        <div className="glass-panel p-6 rounded-lg">
          <h3 className="text-sm font-bold text-cyan-400 uppercase mb-4">AI Recommendations</h3>
          <MLSiteRecommendation
            candidateSites={(landingSites.length > 0 ? landingSites : constructionSites).map(s => ({
              coordinates: [s.lat, s.lon] as [number, number],
              elevation: s.mean_elevation_m,
              slope_mean: s.mean_slope_deg,
              aspect_mean: 180.0,
              roughness_mean: s.mean_roughness,
              tri_mean: 20.0
            }))}
            destinationCoordinates={(constructionSites.find(s => s.site_id === siteId)) ? [
              constructionSites.find(s => s.site_id === siteId)!.lat,
              constructionSites.find(s => s.site_id === siteId)!.lon
            ] as [number, number] : undefined}
            onRecommendationComplete={(recs) => {
              setMlRecs(recs || [])
            }}
          />
          {mlRecs && mlRecs.length > 0 && (
            <div className="mt-4 flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  const top = mlRecs![0]
                  if (!top) return
                  setStartLat(top.coordinates[0])
                  setStartLon(top.coordinates[1])
                  const pool = landingSites.length > 0 ? landingSites : constructionSites
                  const match = pool.find(s => Math.abs(s.lat - top.coordinates[0]) < 1e-6 && Math.abs(s.lon - top.coordinates[1]) < 1e-6)
                  setRecommendedLandingSiteId(match ? match.site_id : null)
                }}
                className="bg-cyan-700 hover:bg-cyan-600 text-white px-3 py-2 rounded text-xs font-bold uppercase"
              >
                USE TOP RECOMMENDATION
              </button>
            </div>
          )}
        </div>
      )}

      <form onSubmit={handleSubmit} className="glass-panel p-6 rounded-lg space-y-4">
        <div className="flex items-center justify-between border-b border-gray-700/50 pb-2">
          <div className="text-xs font-bold text-gray-400 uppercase">Example Seed</div>
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={async () => { const data = await getExampleROIs(); setExamples(data) }}
              className="px-2 py-1 bg-gray-800 text-xs text-cyan-400 border border-cyan-500/30 rounded hover:bg-gray-700"
            >
              Load Examples
            </button>
            {examples.length > 0 && (
              <select
                onChange={(e) => {
                  const sel = examples.find(x => x.id === e.target.value)
                  if (sel) {
                    const cLat = (sel.bbox.lat_min + sel.bbox.lat_max) / 2
                    const cLon = (sel.bbox.lon_min + sel.bbox.lon_max) / 2
                    setStartLat(cLat)
                    setStartLon(cLon)
                  }
                }}
                className="bg-gray-800 text-white px-2 py-1 rounded text-xs border border-gray-600"
                defaultValue=""
              >
                <option value="" disabled>Select...</option>
                {examples.map(x => (
                  <option key={x.id} value={x.id}>{x.name}</option>
                ))}
              </select>
            )}
          </div>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">TARGET SITE ID</label>
          <input
            type="number"
            value={siteId}
            onChange={(e) => setSiteId(parseInt(e.target.value))}
            className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none"
            min="1"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">START LAT</label>
            <input
              type="number"
              value={startLat}
              onChange={(e) => setStartLat(parseFloat(e.target.value))}
              className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">START LON</label>
            <input
              type="number"
              value={startLon}
              onChange={(e) => setStartLon(parseFloat(e.target.value))}
              className="w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none"
              step="0.1"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">STRATEGY</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as 'safest' | 'balanced' | 'direct')}
            className="w-full bg-gray-800 border border-gray-600 text-white px-3 py-2 rounded text-sm focus:border-cyan-500 focus:outline-none"
          >
            <option value="safest">Safest (Risk Averse)</option>
            <option value="balanced">Balanced (Standard)</option>
            <option value="direct">Direct (Distance Priority)</option>
          </select>
        </div>

      <button
        type="submit"
        disabled={mutation.isPending}
        className="w-full bg-green-600 hover:bg-green-500 text-white px-4 py-3 rounded font-bold tracking-wider transition-colors disabled:opacity-50 shadow-[0_0_15px_rgba(34,197,94,0.2)]"
      >
        {mutation.isPending ? 'CALCULATING...' : 'GENERATE FLIGHTPATH'}
      </button>
      </form>

      {/* Error Display */}
      {mutation.isError && (
        <div className="glass-panel p-4 rounded-lg border border-red-500/50 bg-red-900/20">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span className="font-bold">Navigation Planning Failed</span>
          </div>
          <p className="mt-2 text-sm text-red-300">
            {(mutation.error as any)?.response?.data?.detail || (mutation.error as any)?.message || 'An unexpected error occurred while planning the route. Please check your parameters and try again.'}
          </p>
        </div>
      )}

      {mutation.data && (
        <div className="glass-panel p-6 rounded-lg">
          <h3 className="text-lg font-bold mb-4 text-green-400 tracking-wider">
            FLIGHTPATH_DATA
          </h3>
          {mutation.data.path_length_m && (
             <div className="bg-gray-900 p-3 rounded border border-gray-700 mb-4 flex justify-between items-center">
                <span className="text-xs text-gray-500">EST. DISTANCE</span>
                <span className="text-lg font-mono text-white">{(mutation.data.path_length_m).toFixed(2)} m</span>
             </div>
          )}
          <div className="overflow-x-auto max-h-60 custom-scrollbar">
            <table className="w-full text-xs font-mono">
              <thead className="sticky top-0 bg-gray-900">
                <tr className="border-b border-gray-700 text-gray-500">
                  <th className="text-left p-2">WP</th>
                  <th className="text-left p-2">X</th>
                  <th className="text-left p-2">Y</th>
                  <th className="text-left p-2">TOL</th>
                </tr>
              </thead>
              <tbody>
                {mutation.data.waypoints.map((waypoint) => (
                  <tr key={waypoint.waypoint_id} className="border-b border-gray-800 hover:bg-gray-800/50">
                    <td className="p-2 text-green-300">{waypoint.waypoint_id}</td>
                    <td className="p-2 text-gray-400">{waypoint.x_meters.toFixed(1)}</td>
                    <td className="p-2 text-gray-400">{waypoint.y_meters.toFixed(1)}</td>
                    <td className="p-2 text-gray-500">{waypoint.tolerance_meters.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {mutation.data && (
        <div className="glass-panel p-4 rounded-lg">
          <h3 className="text-sm font-bold text-green-400 uppercase mb-3">ROUTE VISUALIZATION</h3>
          {mutation.data.waypoints && mutation.data.waypoints.length > 0 ? (
            <div className="h-[400px] rounded border border-gray-700 overflow-hidden">
              <TerrainMap
                roi={computeRouteRoi(mutation.data.waypoints, startLat, startLon)}
                dataset="mola"
                showWaypoints={true}
                showSites={true}
                waypointsRefreshKey={waypointsRefreshKey}
              />
            </div>
          ) : (
            <div className="h-[200px] rounded border border-gray-700 flex items-center justify-center bg-gray-900/50">
              <p className="text-gray-400 text-sm">No waypoints generated for this route</p>
            </div>
          )}
          {mutation.data.waypoints && mutation.data.waypoints.length > 0 && (
            <button
              onClick={() => {
                const roi = computeRouteRoi(mutation.data!.waypoints, startLat, startLon)
                if (roi) {
                  sessionStorage.setItem('pendingRoi', JSON.stringify(roi))
                  sessionStorage.setItem('pending3dMode', 'true')
                  navigate('/analysis')
                }
              }}
              className="mt-4 flex items-center justify-center gap-2 w-full bg-purple-600 hover:bg-purple-500 text-white px-4 py-3 rounded font-bold tracking-wider transition-colors"
            >
              <Box size={18} />
              VIEW IN 3D TERRAIN
            </button>
          )}
        </div>
      )}
    </div>
  )
}
