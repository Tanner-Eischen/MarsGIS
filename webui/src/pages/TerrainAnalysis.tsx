import { useEffect, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { analyzeTerrain, AnalysisRequest, getExampleROIs, ExampleROIItem } from '../services/api'
import EnhancedProgressBar from '../components/EnhancedProgressBar'
import AIQueryInterface from '../components/AIQueryInterface'
import { useGeoPlan } from '../context/GeoPlanContext'

// Generate UUID for task tracking
function generateTaskId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

export default function TerrainAnalysis() {
  const { publishSites } = useGeoPlan()
  const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [dataset, setDataset] = useState('mola')
  const [threshold, setThreshold] = useState(0.7)
  const [examples, setExamples] = useState<ExampleROIItem[]>([])
  const [criteriaWeights, setCriteriaWeights] = useState<Record<string, number> | null>(null)
  const [autoRunAfterAI, setAutoRunAfterAI] = useState(true)
  const [siteType, setSiteType] = useState<'landing' | 'construction'>('construction')
  useEffect(() => {
    const s = localStorage.getItem('terrain.roi')
    const d = localStorage.getItem('terrain.dataset')
    const t = localStorage.getItem('terrain.threshold')
    if (s) { try { const o = JSON.parse(s); if (o && o.lat_min !== undefined) setRoi(o) } catch {}
    }
    if (d) setDataset(d)
    if (t) setThreshold(parseFloat(t))
  }, [])
  useEffect(() => { localStorage.setItem('terrain.roi', JSON.stringify(roi)) }, [roi])
  useEffect(() => { localStorage.setItem('terrain.dataset', dataset) }, [dataset])
  useEffect(() => { localStorage.setItem('terrain.threshold', String(threshold)) }, [threshold])
  const [taskId, setTaskId] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: async (request: AnalysisRequest & { task_id?: string }) => {
      // Generate task_id before making request
      const taskId = generateTaskId()
      // Add task_id to request
      const response = await analyzeTerrain({ ...request, task_id: taskId })
      // Only set taskId after successful API call so WebSocket can connect
      setTaskId(taskId)
      return response
    },
    onSuccess: (data) => {
      // Use task_id from response if available, otherwise keep the one we generated
      if (data.task_id) {
        setTaskId(data.task_id)
      }
      if (data.sites && data.sites.length > 0) {
        publishSites(siteType, data.sites)
      }
    },
    onError: (error: any) => {
      // Clear taskId on error to stop WebSocket connection attempts
      setTaskId(null)
      alert(`Analysis failed: ${error.response?.data?.detail || error.message}`)
    },
  })

  const handleAIQueryProcessed = (parameters: any) => {
    console.log("AI Query processed, raw parameters:", parameters);
    // This function now ensures UI fields update and auto-run uses the correct, fresh data.
    
    const newRoi = parameters.roi ? {
      lat_min: parameters.roi.min_lat,
      lat_max: parameters.roi.max_lat,
      lon_min: parameters.roi.min_lon,
      lon_max: parameters.roi.max_lon
    } : null;

    if (newRoi) {
      console.log("Updating ROI state with:", newRoi);
      setRoi(newRoi);
    }
    if (parameters.dataset_preferences?.primary) {
      const newDataset = parameters.dataset_preferences.primary.toLowerCase();
      console.log("Updating dataset state with:", newDataset);
      setDataset(newDataset);
    }
    if (parameters.criteria_weights) {
      console.log("Updating criteria weights state with:", parameters.criteria_weights);
      setCriteriaWeights(parameters.criteria_weights);
    }

    if (autoRunAfterAI) {
      console.log('Auto-running analysis with new parameters...');
      setTaskId(null); // Reset task for new analysis
      
      // We use the `newRoi` variable and other params directly to avoid race conditions with state updates.
      const analysisROI: [number, number, number, number] = newRoi ? [newRoi.lat_min, newRoi.lat_max, newRoi.lon_min, newRoi.lon_max] : [roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max];
      
      mutation.mutate({
        roi: analysisROI,
        dataset: parameters.dataset_preferences?.primary?.toLowerCase() ?? dataset,
        threshold,
        criteria_weights: parameters.criteria_weights ?? criteriaWeights ?? undefined,
      });
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setTaskId(null) // Reset task_id for new request
    mutation.mutate({
      roi: [roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max],
      dataset,
      threshold,
      criteria_weights: criteriaWeights ?? undefined,
    })
  }

  const generateWeightSummary = (weights: Record<string, number>): string => {
    if (!weights || Object.keys(weights).length === 0) {
      return 'No specific criteria weights are being applied. The analysis will use default, balanced priorities.'
    }
    const safetyCriteria = ['slope', 'roughness']
    let safetyWeight = 0
    let scienceWeight = 0
    let topCriterion = ''
    let maxWeight = 0
    for (const [key, value] of Object.entries(weights)) {
      if (value > maxWeight) { maxWeight = value; topCriterion = key }
      if (safetyCriteria.includes(key)) safetyWeight += value
      else scienceWeight += value
    }
    const top = topCriterion.replace(/_/g, ' ')
    if (safetyWeight > scienceWeight * 1.5) return `This analysis strongly prioritizes safety, with a focus on finding areas with low ${top}.`
    if (scienceWeight > safetyWeight * 1.5) return `This analysis strongly prioritizes scientific value, focusing on areas with high ${top}.`
    return `This analysis is taking a balanced approach, with the most influential factor being ${top}.`
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Terrain Analysis</h2>
      
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="mb-3 text-sm text-gray-300">
          Use natural language to set analysis preferences. We apply the extracted weights to compute the suitability heat map and rank sites.
        </div>
        <AIQueryInterface 
          onQueryProcessed={handleAIQueryProcessed}
          className="mb-4"
        />
        <div className="flex items-center justify-between">
          <div className="text-xs text-gray-400">
            {criteriaWeights ? 'AI weights loaded and ready to apply' : 'No AI weights yet'}
          </div>
          <label className="flex items-center space-x-2 text-sm text-gray-300">
            <input type="checkbox" checked={autoRunAfterAI} onChange={(e) => setAutoRunAfterAI(e.target.checked)} />
            <span>Auto-run analysis after AI query</span>
          </label>
        </div>
      </div>
      
      {taskId && (
        <div className="mb-4">
          <EnhancedProgressBar 
            taskId={taskId} 
            title="Terrain Analysis Progress"
            showDetails={true}
            onComplete={() => setTaskId(null)}
            onError={(error) => alert(`Progress tracking error: ${error}`)}
          />
        </div>
      )}

      <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-6 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">Site Type</label>
            <select
              value={siteType}
              onChange={(e) => setSiteType(e.target.value as 'landing' | 'construction')}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
            >
              <option value="construction">Construction Site</option>
              <option value="landing">Landing Site</option>
            </select>
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
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-300">Region of Interest</div>
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={async () => { const data = await getExampleROIs(); setExamples(data) }}
              className="px-3 py-2 bg-gray-700 text-white rounded"
            >
              Fill with Example ROI
            </button>
            {examples.length > 0 && (
              <select
                onChange={(e) => {
                  const sel = examples.find(x => x.id === e.target.value)
                  if (sel) { setRoi(sel.bbox); setDataset(sel.dataset) }
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
          <label className="block text-sm font-medium mb-2">
            Suitability Threshold: {threshold}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Display Applied Criteria Weights */}
        {criteriaWeights && (
          <div className="space-y-2 p-4 bg-gray-900 rounded-md">
            <h4 className="text-md font-semibold text-gray-200">Applied Criteria Weights</h4>
            <p className="text-sm text-gray-300 pb-2 italic">
              {generateWeightSummary(criteriaWeights || {})}
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
              {Object.entries(criteriaWeights).map(([key, value]) => (
                <div key={key} className="bg-gray-700 p-2 rounded-md flex justify-between">
                  <span className="font-bold capitalize text-gray-300">{key.replace(/_/g, ' ')}:</span>
                  <span className="text-white">{Number(value).toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={mutation.isPending}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors disabled:opacity-50"
        >
          {mutation.isPending ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </form>

      {mutation.data && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">
            Analysis Results ({mutation.data.sites.length} sites found)
          </h3>
          <div className="mb-4">
            <p className="text-gray-300">
              Top Site ID: {mutation.data.top_site_id} (Score: {mutation.data.top_site_score.toFixed(3)})
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left p-2">ID</th>
                  <th className="text-left p-2">Rank</th>
                  <th className="text-left p-2">Area (km²)</th>
                  <th className="text-left p-2">Lat</th>
                  <th className="text-left p-2">Lon</th>
                  <th className="text-left p-2">Score</th>
                </tr>
              </thead>
              <tbody>
                {mutation.data.sites.map((site) => (
                  <tr key={site.site_id} className="border-b border-gray-700">
                    <td className="p-2">{site.site_id}</td>
                    <td className="p-2">{site.rank}</td>
                    <td className="p-2">{site.area_km2.toFixed(2)}</td>
                    <td className="p-2">{site.lat.toFixed(4)}</td>
                    <td className="p-2">{site.lon.toFixed(4)}</td>
                    <td className="p-2">{site.suitability_score.toFixed(3)}</td>
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




