import { useEffect, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { planNavigation, NavigationRequest, getExampleROIs, ExampleROIItem } from '../services/api'
import MLSiteRecommendation from '../components/MLSiteRecommendation'
import { useGeoPlan } from '../context/GeoPlanContext'
import ProgressBar from '../components/ProgressBar'

// Generate UUID for task tracking
function generateTaskId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

export default function NavigationPlanning() {
  const { landingSites, constructionSites, setRecommendedLandingSiteId } = useGeoPlan()
  const [siteId, setSiteId] = useState(1)
  const [startLat, setStartLat] = useState(40.0)
  const [startLon, setStartLon] = useState(180.0)
  const [strategy, setStrategy] = useState<'safest' | 'balanced' | 'direct'>('balanced')
  const [analysisDir, setAnalysisDir] = useState('data/output')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [examples, setExamples] = useState<ExampleROIItem[]>([])
  const [mlRecs, setMlRecs] = useState<any[] | null>(null)
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
      // Generate task_id before making request
      const taskId = generateTaskId()
      // Add task_id to request
      const response = await planNavigation({ ...request, task_id: taskId })
      // Only set taskId after successful API call so WebSocket can connect
      setTaskId(taskId)
      return response
    },
    onSuccess: (data) => {
      // Use task_id from response if available, otherwise keep the one we generated
      if (data.task_id) {
        setTaskId(data.task_id)
      }
    },
    onError: (error: any) => {
      // Clear taskId on error to stop WebSocket connection attempts
      setTaskId(null)
      alert(`Navigation planning failed: ${error.response?.data?.detail || error.message}`)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setTaskId(null) // Reset task_id for new request
    mutation.mutate({
      site_id: siteId,
      start_lat: startLat,
      start_lon: startLon,
      strategy,
      analysis_dir: analysisDir,
    })
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Navigation Planning</h2>
      <div className="bg-gray-800 rounded-lg p-6 space-y-3">
        <div className="text-sm text-gray-300">Select a starting site from the list, or use mission-type ranking to explore options.</div>
        <div>
          <label className="block text-sm font-medium mb-2">Starting Site (Landing)</label>
          <select
            value={landingSites.find(s => Math.abs(s.lat - startLat) < 1e-6 && Math.abs(s.lon - startLon) < 1e-6)?.site_id || ''}
            onChange={(e) => {
              const id = parseInt(e.target.value)
              const site = landingSites.find(s => s.site_id === id)
              if (site) { setStartLat(site.lat); setStartLon(site.lon) }
            }}
            className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
          >
            <option value="" disabled>{landingSites.length > 0 ? 'Select starting site' : 'No landing sites found — run Terrain Analysis'}</option>
            {landingSites.map(site => (
              <option key={site.site_id} value={site.site_id}>
                Site {site.site_id} (Rank {site.rank}, Score {site.suitability_score.toFixed(2)})
              </option>
            ))}
          </select>
        </div>
      </div>
      {(landingSites.length > 0 || constructionSites.length > 0) && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">ML Site Recommendation</h3>
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
                className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded"
              >
                Apply top recommendation as starting site
              </button>
              <span className="text-xs text-gray-400">Or select a starting site from the list above.</span>
            </div>
          )}
          <div className="mt-3 text-xs text-gray-400">Use mission type to explore options. Select a site above to apply as the starting location.</div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-300">Seed from Example ROI</div>
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={async () => { const data = await getExampleROIs(); setExamples(data) }}
              className="px-3 py-2 bg-gray-700 text-white rounded"
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
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Target Site ID</label>
          <input
            type="number"
            value={siteId}
            onChange={(e) => setSiteId(parseInt(e.target.value))}
            className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
            min="1"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">Start Latitude</label>
            <input
              type="number"
              value={startLat}
              onChange={(e) => setStartLat(parseFloat(e.target.value))}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Start Longitude</label>
            <input
              type="number"
              value={startLon}
              onChange={(e) => setStartLon(parseFloat(e.target.value))}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              step="0.1"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Pathfinding Strategy</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as 'safest' | 'balanced' | 'direct')}
            className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
          >
            <option value="safest">Safest (prioritize safety)</option>
            <option value="balanced">Balanced (default)</option>
            <option value="direct">Direct (prioritize distance)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Analysis Directory</label>
          <input
            type="text"
            value={analysisDir}
            onChange={(e) => setAnalysisDir(e.target.value)}
            className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
          />
        </div>

      <button
        type="submit"
        disabled={mutation.isPending}
        className="w-full bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {mutation.isPending ? 'Planning...' : 'Generate Waypoints'}
      </button>
      </form>

      {mutation.data && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">
            Navigation Plan ({mutation.data.num_waypoints} waypoints)
          </h3>
          {mutation.data.path_length_m && (
            <p className="text-gray-300 mb-4">
              Estimated Path Length: {mutation.data.path_length_m.toFixed(2)} meters
            </p>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left p-2">Waypoint ID</th>
                  <th className="text-left p-2">X (meters)</th>
                  <th className="text-left p-2">Y (meters)</th>
                  <th className="text-left p-2">Tolerance (meters)</th>
                </tr>
              </thead>
              <tbody>
                {mutation.data.waypoints.map((waypoint) => (
                  <tr key={waypoint.waypoint_id} className="border-b border-gray-700">
                    <td className="p-2">{waypoint.waypoint_id}</td>
                    <td className="p-2">{waypoint.x_meters.toFixed(2)}</td>
                    <td className="p-2">{waypoint.y_meters.toFixed(2)}</td>
                    <td className="p-2">{waypoint.tolerance_meters.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}




