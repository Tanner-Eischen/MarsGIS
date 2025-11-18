import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { analyzeTerrain, AnalysisRequest } from '../services/api'

export default function TerrainAnalysis() {
  const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [dataset, setDataset] = useState('mola')
  const [threshold, setThreshold] = useState(0.7)

  const mutation = useMutation({
    mutationFn: (request: AnalysisRequest) => analyzeTerrain(request),
    onError: (error: any) => {
      alert(`Analysis failed: ${error.response?.data?.detail || error.message}`)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({
      roi: [roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max],
      dataset,
      threshold,
    })
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Terrain Analysis</h2>
      
      <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-6 space-y-4">
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




