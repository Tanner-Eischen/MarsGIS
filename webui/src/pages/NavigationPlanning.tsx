import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { planNavigation, NavigationRequest } from '../services/api'
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
  const [siteId, setSiteId] = useState(1)
  const [startLat, setStartLat] = useState(40.0)
  const [startLon, setStartLon] = useState(180.0)
  const [strategy, setStrategy] = useState<'safest' | 'balanced' | 'direct'>('balanced')
  const [analysisDir, setAnalysisDir] = useState('data/output')
  const [taskId, setTaskId] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: async (request: NavigationRequest & { task_id?: string }) => {
      // Generate task_id before making request
      const taskId = generateTaskId()
      // Set task_id immediately so WebSocket can connect
      setTaskId(taskId)
      // Add task_id to request
      return planNavigation({ ...request, task_id: taskId })
    },
    onSuccess: (data) => {
      // Task_id should already be set, but keep this for compatibility
      if (data.task_id && !taskId) {
        setTaskId(data.task_id)
      }
    },
    onError: (error: any) => {
      alert(`Navigation planning failed: ${error.response?.data?.detail || error.message}`)
      setTaskId(null)
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
      
      {taskId && (
        <div className="mb-4">
          <ProgressBar taskId={taskId} onComplete={() => setTaskId(null)} />
        </div>
      )}

      <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-6 space-y-4">
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




