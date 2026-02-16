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

interface Props {
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  onRoiChange?: (roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }) => void
  dataset?: string
  onDatasetChange?: (dataset: string) => void
}

export default function TerrainAnalysis({ 
  roi: propRoi, 
  onRoiChange, 
  dataset: propDataset, 
  onDatasetChange 
}: Props) {
  const { publishSites } = useGeoPlan()
  
  const [localRoi, setLocalRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [localDataset, setLocalDataset] = useState('mola_200m')
  
  const roi = propRoi || localRoi
  const setRoi = onRoiChange || setLocalRoi
  const dataset = propDataset || localDataset
  const setDataset = onDatasetChange || setLocalDataset

  const [threshold, setThreshold] = useState(0.7)
  const [examples, setExamples] = useState<ExampleROIItem[]>([])
  const [criteriaWeights, setCriteriaWeights] = useState<Record<string, number> | null>(null)
  const [autoRunAfterAI, setAutoRunAfterAI] = useState(true)
  const [siteType, setSiteType] = useState<'landing' | 'construction'>('construction')
  
  useEffect(() => {
    const s = localStorage.getItem('terrain.roi')
    const d = localStorage.getItem('terrain.dataset')
    const t = localStorage.getItem('terrain.threshold')
    if (s && !propRoi) { try { const o = JSON.parse(s); if (o && o.lat_min !== undefined) setRoi(o) } catch {} }
    if (d && !propDataset) setDataset(d)
    if (t) setThreshold(parseFloat(t))
  }, [])
  
  // Only save to local storage if we are controlling the state locally or if it changes
  useEffect(() => { localStorage.setItem('terrain.roi', JSON.stringify(roi)) }, [roi])
  useEffect(() => { localStorage.setItem('terrain.dataset', dataset) }, [dataset])
  useEffect(() => { localStorage.setItem('terrain.threshold', String(threshold)) }, [threshold])
  
  const [taskId, setTaskId] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: async (request: AnalysisRequest & { task_id?: string }) => {
      const taskId = generateTaskId()
      const response = await analyzeTerrain({ ...request, task_id: taskId })
      setTaskId(taskId)
      return response
    },
    onSuccess: (data) => {
      if (data.task_id) {
        setTaskId(data.task_id)
      }
      if (data.sites && data.sites.length > 0) {
        publishSites(siteType, data.sites)
      }
    },
    onError: (error: any) => {
      setTaskId(null)
      alert(`Analysis failed: ${error.response?.data?.detail || error.message}`)
    },
  })

  const handleAIQueryProcessed = (parameters: any) => {
    if (import.meta.env.DEV) {
      console.log("AI Query processed, raw parameters:", parameters);
    }
    const newRoi = parameters.roi ? {
      lat_min: parameters.roi.min_lat,
      lat_max: parameters.roi.max_lat,
      lon_min: parameters.roi.min_lon,
      lon_max: parameters.roi.max_lon
    } : null;

    if (newRoi) {
      setRoi(newRoi);
    }
    if (parameters.dataset_preferences?.primary) {
      setDataset(parameters.dataset_preferences.primary.toLowerCase());
    }
    if (parameters.criteria_weights) {
      setCriteriaWeights(parameters.criteria_weights);
    }

    if (autoRunAfterAI) {
      setTaskId(null);
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
    setTaskId(null)
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
    <div className="space-y-4 text-sm">
      
      <div className="glass-panel p-4 rounded-lg border border-cyan-500/20">
        <div className="mb-3 text-xs text-cyan-400/90 uppercase tracking-wide font-bold flex justify-between">
          <span>AI Mission Assistant</span>
          <span className="text-[10px] opacity-50">NLP_MODULE_V1</span>
        </div>
        <AIQueryInterface 
          onQueryProcessed={handleAIQueryProcessed}
          className="mb-4"
        />
        <div className="flex items-center justify-between border-t border-gray-700/50 pt-3">
          <div className="text-[10px] text-gray-400">
            {criteriaWeights ? 'AI weights loaded' : 'Standard weights'}
          </div>
          <label className="flex items-center space-x-2 text-[10px] text-gray-300 cursor-pointer hover:text-white">
            <input type="checkbox" checked={autoRunAfterAI} onChange={(e) => setAutoRunAfterAI(e.target.checked)} className="accent-cyan-500"/>
            <span>Auto-execute</span>
          </label>
        </div>
      </div>
      
      {taskId && (
        <div className="mb-4">
          <EnhancedProgressBar 
            taskId={taskId} 
            title="ANALYSIS_PROGRESS"
            showDetails={true}
            onComplete={() => setTaskId(null)}
            onError={(error) => alert(`Progress tracking error: ${error}`)}
          />
        </div>
      )}

      <form onSubmit={handleSubmit} className="glass-panel p-4 rounded-lg space-y-3 border border-gray-700/50">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-[10px] font-bold text-gray-500 mb-1 uppercase">Operation Type</label>
            <select
              value={siteType}
              onChange={(e) => setSiteType(e.target.value as 'landing' | 'construction')}
              className="w-full bg-gray-800 border border-gray-600 text-white px-2 py-1.5 rounded text-xs focus:border-cyan-500 focus:outline-none"
            >
              <option value="construction">Construction</option>
              <option value="landing">Landing</option>
            </select>
          </div>
          <div>
            <label className="block text-[10px] font-bold text-gray-500 mb-1 uppercase">Source Data</label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 text-white px-2 py-1.5 rounded text-xs focus:border-cyan-500 focus:outline-none"
            >
              <option value="mola">MOLA (Global)</option>
              <option value="mola_200m">MOLA 200m (Global)</option>
              <option value="hirise">HiRISE (High Res)</option>
              <option value="ctx">CTX (Medium Res)</option>
            </select>
          </div>
        </div>

        <div className="flex items-center justify-between border-b border-gray-700/50 pb-2">
          <div className="text-[10px] font-bold text-gray-400 uppercase">Target Coordinates</div>
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={async () => { const data = await getExampleROIs(); setExamples(data) }}
              className="px-2 py-1 bg-gray-800 text-[10px] text-cyan-400 border border-cyan-500/30 rounded hover:bg-gray-700"
            >
              Load ROI
            </button>
            {examples.length > 0 && (
              <select
                onChange={(e) => {
                  const sel = examples.find(x => x.id === e.target.value)
                  if (sel) { setRoi(sel.bbox); setDataset(sel.dataset) }
                }}
                className="bg-gray-800 text-white px-2 py-1 rounded text-[10px] border border-gray-600 max-w-[100px]"
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

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-[10px] text-gray-500 mb-0.5">LAT MIN</label>
            <input
              type="number"
              value={roi.lat_min}
              onChange={(e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) })}
              className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-[10px] text-gray-500 mb-0.5">LAT MAX</label>
            <input
              type="number"
              value={roi.lat_max}
              onChange={(e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) })}
              className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-[10px] text-gray-500 mb-0.5">LON MIN</label>
            <input
              type="number"
              value={roi.lon_min}
              onChange={(e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) })}
              className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-[10px] text-gray-500 mb-0.5">LON MAX</label>
            <input
              type="number"
              value={roi.lon_max}
              onChange={(e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) })}
              className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none"
              step="0.1"
            />
          </div>
        </div>

        <div>
          <label className="block text-[10px] font-bold text-gray-400 mb-1 uppercase">
            Suitability Threshold: {threshold}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-cyan-500 h-1.5"
          />
        </div>

        {criteriaWeights && (
          <div className="space-y-2 p-2 bg-gray-900/50 rounded border border-gray-700/50">
            <h4 className="text-[10px] font-bold text-gray-400 uppercase">Active Weights</h4>
            <p className="text-[10px] text-gray-500 italic truncate">
              {generateWeightSummary(criteriaWeights || {})}
            </p>
            <div className="grid grid-cols-2 gap-1 text-[10px]">
              {Object.entries(criteriaWeights).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center bg-gray-800 px-2 py-0.5 rounded">
                  <span className="font-mono text-gray-400 truncate max-w-[70px]">{key.replace(/_/g, ' ').toUpperCase()}</span>
                  <span className="font-bold text-cyan-300">{Number(value).toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={mutation.isPending}
          className="w-full bg-cyan-700 hover:bg-cyan-600 text-white px-4 py-2 rounded text-xs font-bold tracking-wider transition-colors disabled:opacity-50 shadow-[0_0_15px_rgba(6,182,212,0.2)]"
        >
          {mutation.isPending ? 'PROCESSING...' : 'INITIATE ANALYSIS'}
        </button>
      </form>

      {mutation.data && (
        <div className="glass-panel p-4 rounded-lg border border-green-500/20">
          <h3 className="text-xs font-bold mb-2 text-green-400 tracking-wider uppercase">
            Results Matrix
          </h3>
          <div className="mb-3 flex gap-2">
             <div className="bg-gray-900 p-2 rounded border border-gray-700 flex-1">
                <span className="block text-[10px] text-gray-500 uppercase">Sites Found</span>
                <span className="text-lg font-bold text-white">{mutation.data.sites.length}</span>
             </div>
             <div className="bg-gray-900 p-2 rounded border border-gray-700 flex-1">
                <span className="block text-[10px] text-gray-500 uppercase">Top Score</span>
                <span className="text-lg font-bold text-green-400">{mutation.data.top_site_score.toFixed(3)}</span>
             </div>
          </div>
          <div className="overflow-x-auto max-h-40 custom-scrollbar">
            <table className="w-full text-[10px] font-mono">
              <thead className="sticky top-0 bg-gray-900">
                <tr className="border-b border-gray-700 text-gray-500">
                  <th className="text-left p-1">ID</th>
                  <th className="text-left p-1">RANK</th>
                  <th className="text-left p-1">AREA</th>
                  <th className="text-left p-1">SCORE</th>
                </tr>
              </thead>
              <tbody>
                {mutation.data.sites.map((site) => (
                  <tr key={site.site_id} className="border-b border-gray-800 hover:bg-gray-800/50">
                    <td className="p-1 text-cyan-300">{site.site_id}</td>
                    <td className="p-1 text-white">{site.rank}</td>
                    <td className="p-1 text-gray-400">{site.area_km2.toFixed(1)}</td>
                    <td className="p-1 text-green-400 font-bold">{site.suitability_score.toFixed(2)}</td>
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
