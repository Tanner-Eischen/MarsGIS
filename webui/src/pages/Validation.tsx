import { useState, useEffect } from 'react'

interface HealthStatus {
  status: string
  checks: Record<string, string>
}

export default function Validation() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 5000) // Check every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const checkHealth = async () => {
    try {
      const [, readyResponse] = await Promise.all([
        fetch('http://localhost:5000/api/v1/health/live'),
        fetch('http://localhost:5000/api/v1/health/ready')
      ])

      if (readyResponse.ok) {
        const data = await readyResponse.json()
        setHealthStatus(data)
      }
    } catch (error) {
      console.error('Health check failed:', error)
      setHealthStatus({
        status: 'error',
        checks: { error: 'Failed to connect to server' }
      })
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready':
      case 'alive':
        return 'text-green-400'
      case 'not_ready':
        return 'text-red-400'
      default:
        return 'text-yellow-400'
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">System Validation</h2>
      
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Health Status</h3>
        
        {loading ? (
          <div className="text-center py-4">Checking health...</div>
        ) : healthStatus ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Overall Status:</span>
              <span className={`font-semibold ${getStatusColor(healthStatus.status)}`}>
                {healthStatus.status.toUpperCase()}
              </span>
            </div>
            
            <div className="border-t border-gray-700 pt-4">
              <h4 className="text-sm font-semibold mb-2">Component Checks:</h4>
              <div className="space-y-2">
                {Object.entries(healthStatus.checks).map(([component, status]) => (
                  <div key={component} className="flex justify-between items-center">
                    <span className="text-sm text-gray-400 capitalize">{component.replace('_', ' ')}:</span>
                    <span className={`text-sm font-medium ${getStatusColor(status)}`}>
                      {status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-4 text-red-400">Failed to check health</div>
        )}
      </div>
    </div>
  )
}

