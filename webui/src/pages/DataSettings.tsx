import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { downloadDEM, DownloadRequest } from '../services/api'
import { Database, Folder, CheckCircle, Download, Cloud } from 'lucide-react'

// --- Components for Tabs (Previously Separate Pages) ---

function DataDownloadComponent() {
  const [dataset, setDataset] = useState('mola_200m')
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
    <div className="glass-panel p-6 rounded-lg max-w-2xl mx-auto">
      <h2 className="text-xl font-bold text-mars-orange mb-4 flex items-center gap-2">
        <Cloud className="w-5 h-5" />
        DATA_ACQUISITION
      </h2>
      
      {dataset === 'hirise' && (
        <div className="bg-amber-900/30 border border-amber-700/50 rounded p-4 mb-4 text-xs">
          <h3 className="font-bold text-amber-400 mb-1">MANUAL_DOWNLOAD_REQUIRED</h3>
          <p className="text-amber-200/80 mb-2">
            HiRISE requires a specific observation/region; it will not auto-download globally.
          </p>
          <ul className="list-disc list-inside space-y-1 text-amber-300/70">
            <li>Source: <a href="https://www.uahirise.org/hiwish/" target="_blank" rel="noopener noreferrer" className="underline hover:text-white">HiRISE PDS</a></li>
          </ul>
        </div>
      )}
      
      {errorMessage && (
        <div className="bg-red-900/30 border border-red-700/50 rounded p-4 mb-4 text-xs text-red-200">
          <h3 className="font-bold text-red-400 mb-1">ERROR</h3>
          <pre className="whitespace-pre-wrap">{errorMessage}</pre>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-bold text-gray-500 mb-1 uppercase">Dataset Source</label>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full bg-gray-800/50 border border-gray-700 text-white px-3 py-2 rounded text-sm focus:border-mars-orange focus:outline-none"
          >
            <option value="hirise">HiRISE (1m)</option>
            <option value="mola_200m">MOLA 200m (Global)</option>
            <option value="mola">MOLA (Global 463m)</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">LAT MIN</label>
            <input type="number" value={roi.lat_min} onChange={(e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) })} className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono" step="0.1" />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">LAT MAX</label>
            <input type="number" value={roi.lat_max} onChange={(e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) })} className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono" step="0.1" />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">LON MIN</label>
            <input type="number" value={roi.lon_min} onChange={(e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) })} className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono" step="0.1" />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">LON MAX</label>
            <input type="number" value={roi.lon_max} onChange={(e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) })} className="w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono" step="0.1" />
          </div>
        </div>

        <div className="flex items-center pt-2">
          <input
            type="checkbox"
            id="force"
            checked={force}
            onChange={(e) => setForce(e.target.checked)}
            className="accent-mars-orange mr-2"
          />
          <label htmlFor="force" className="text-xs text-gray-400">Force re-download (Ignore cache)</label>
        </div>

        <button
          type="submit"
          disabled={mutation.isPending}
          className="w-full bg-mars-orange hover:bg-red-600 text-white px-4 py-3 rounded font-bold tracking-wider transition-colors disabled:opacity-50 flex items-center justify-center gap-2 mt-4"
        >
          {mutation.isPending ? 'DOWNLOADING...' : <><Download size={16} /> INITIATE DOWNLOAD</>}
        </button>
      </form>
    </div>
  )
}

function ProjectsComponent() {
  return (
    <div className="glass-panel p-8 rounded-lg text-center text-gray-400">
      <Folder className="w-12 h-12 mx-auto mb-4 opacity-50" />
      <h3 className="text-lg font-bold text-white mb-2">PROJECT_ARCHIVE</h3>
      <p className="text-sm">Project management module is currently offline.</p>
    </div>
  )
}

function ValidationComponent() {
  return (
    <div className="glass-panel p-8 rounded-lg text-center text-gray-400">
      <CheckCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
      <h3 className="text-lg font-bold text-white mb-2">SYSTEM_DIAGNOSTICS</h3>
      <p className="text-sm">Validation suite is running in background mode.</p>
    </div>
  )
}

// --- Main Page Component ---

export default function DataSettings() {
  const [activeTab, setActiveTab] = useState<'data' | 'projects' | 'validation'>('data')

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white min-h-screen">
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <h1 className="text-2xl font-bold tracking-wider mb-6 text-white flex items-center gap-3">
            <Database className="text-mars-orange" />
            SYSTEM_DATA & SETTINGS
        </h1>
        
        <div className="flex space-x-4 border-b border-gray-700">
          <button
            onClick={() => setActiveTab('data')}
            className={`flex items-center px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'data'
                ? 'border-mars-orange text-mars-orange'
                : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'
            }`}
          >
            <Cloud className="w-4 h-4 mr-2" />
            Data Management
          </button>
          <button
            onClick={() => setActiveTab('projects')}
            className={`flex items-center px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'projects'
                ? 'border-blue-500 text-blue-500'
                : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'
            }`}
          >
            <Folder className="w-4 h-4 mr-2" />
            Projects
          </button>
          <button
            onClick={() => setActiveTab('validation')}
            className={`flex items-center px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'validation'
                ? 'border-green-500 text-green-500'
                : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'
            }`}
          >
            <CheckCircle className="w-4 h-4 mr-2" />
            Validation
          </button>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-auto bg-gray-900">
        <div className="max-w-6xl mx-auto">
            {activeTab === 'data' && <DataDownloadComponent />}
            {activeTab === 'projects' && <ProjectsComponent />}
            {activeTab === 'validation' && <ValidationComponent />}
        </div>
      </div>
    </div>
  )
}
