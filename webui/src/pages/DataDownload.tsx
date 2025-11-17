import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { downloadDEM, DownloadRequest } from '../services/api'

export default function DataDownload() {
  const [dataset, setDataset] = useState('mola')
  const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 })
  const [force, setForce] = useState(false)

  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: (request: DownloadRequest) => downloadDEM(request),
    onSuccess: (data) => {
      setErrorMessage(null)
      alert(`Download ${data.cached ? 'completed (cached)' : 'started'}! Size: ${data.size_mb || 'N/A'} MB`)
    },
    onError: (error: any) => {
      const errorDetail = error.response?.data?.detail || error.message
      setErrorMessage(errorDetail)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({
      dataset,
      roi: [roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max],
      force,
    })
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Download DEM Data</h2>
      
      {(dataset === 'hirise' || dataset === 'ctx') && (
        <div className="bg-yellow-900 border border-yellow-700 rounded-lg p-4">
          <h3 className="font-semibold text-yellow-300 mb-2">Manual Download Required</h3>
          <p className="text-yellow-200 text-sm mb-2">
            {dataset === 'hirise' 
              ? 'HiRISE (1m resolution) datasets require manual download. They are region-specific and require selecting specific observation IDs.'
              : 'CTX (18m resolution) datasets require manual download. They are region-specific and require selecting specific observation IDs.'}
          </p>
          <p className="text-yellow-200 text-sm mb-2">Sources:</p>
          <ul className="text-yellow-200 text-sm list-disc list-inside space-y-1">
            {dataset === 'hirise' ? (
              <>
                <li><a href="https://www.uahirise.org/hiwish/" target="_blank" rel="noopener noreferrer" className="underline">HiRISE PDS</a></li>
                <li><a href="https://s3.amazonaws.com/mars-hirise-pds/" target="_blank" rel="noopener noreferrer" className="underline">AWS S3 HiRISE Archive</a></li>
              </>
            ) : (
              <li><a href="https://ode.rsl.wustl.edu/mars/" target="_blank" rel="noopener noreferrer" className="underline">WUSTL ODE Mars Data</a></li>
            )}
          </ul>
          <p className="text-yellow-200 text-sm mt-2">
            After downloading, place the DEM file in the cache directory or the system will use it automatically if it matches the expected pattern.
          </p>
        </div>
      )}
      
      {errorMessage && (
        <div className="bg-red-900 border border-red-700 rounded-lg p-4">
          <h3 className="font-semibold text-red-300 mb-2">Download Error</h3>
          <pre className="text-red-200 text-sm whitespace-pre-wrap">{errorMessage}</pre>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Dataset</label>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
          >
            <option value="mola">MOLA (463m resolution)</option>
            <option value="hirise">HiRISE (1m resolution)</option>
            <option value="ctx">CTX (18m resolution)</option>
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

        <div className="flex items-center">
          <input
            type="checkbox"
            id="force"
            checked={force}
            onChange={(e) => setForce(e.target.checked)}
            className="mr-2"
          />
          <label htmlFor="force" className="text-sm">Force re-download</label>
        </div>

        <button
          type="submit"
          disabled={mutation.isPending}
          className="w-full bg-mars-orange hover:bg-mars-red text-white px-4 py-2 rounded-md transition-colors disabled:opacity-50"
        >
          {mutation.isPending ? 'Downloading...' : 'Download DEM'}
        </button>
      </form>
    </div>
  )
}



