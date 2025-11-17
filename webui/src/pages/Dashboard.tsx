import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getStatus } from '../services/api'

export default function Dashboard() {
  const { data: status, isLoading } = useQuery({
    queryKey: ['status'],
    queryFn: getStatus,
  })

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-4">Welcome to MarsHab</h2>
        <p className="text-gray-300 mb-4">
          Mars Habitat Site Selection and Rover Navigation System
        </p>
        <div className="flex space-x-4">
          <Link
            to="/download"
            className="bg-mars-orange hover:bg-mars-red text-white px-4 py-2 rounded-md transition-colors"
          >
            Download DEM Data
          </Link>
          <Link
            to="/analyze"
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors"
          >
            Analyze Terrain
          </Link>
          <Link
            to="/navigate"
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition-colors"
          >
            Plan Navigation
          </Link>
        </div>
      </div>

      {isLoading ? (
        <div className="text-center py-8">Loading system status...</div>
      ) : status ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-2">Cache Status</h3>
            <p className="text-gray-400">Files: {status.cache.file_count}</p>
            <p className="text-gray-400">Size: {status.cache.size_mb} MB</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-2">Output Files</h3>
            <p className="text-gray-400">Count: {status.output.file_count}</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-2">Data Sources</h3>
            <p className="text-gray-400">
              {status.config.data_sources.join(', ')}
            </p>
          </div>
        </div>
      ) : null}
    </div>
  )
}




