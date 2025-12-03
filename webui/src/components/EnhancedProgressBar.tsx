import { useState, useEffect } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'

interface ProgressStage {
  name: string
  key: string
  progress: number
  message: string
  estimatedSecondsRemaining?: number
  startTime?: number
  duration?: number
}

interface EnhancedProgressBarProps {
  taskId: string | null
  title?: string
  showDetails?: boolean
  onComplete?: () => void
  onError?: (error: string) => void
}

const STAGE_DEFINITIONS: Record<string, { name: string; description: string; typicalDuration: number }> = {
  dem_loading: {
    name: 'Loading DEM Data',
    description: 'Downloading and processing Mars elevation data',
    typicalDuration: 30
  },
  terrain_metrics: {
    name: 'Analyzing Terrain',
    description: 'Calculating slope, aspect, roughness, and terrain ruggedness',
    typicalDuration: 45
  },
  criteria_extraction: {
    name: 'Extracting Criteria',
    description: 'Processing terrain data for site suitability analysis',
    typicalDuration: 20
  },
  mcdm_evaluation: {
    name: 'Evaluating Suitability',
    description: 'Running multi-criteria decision making algorithms',
    typicalDuration: 15
  },
  site_extraction: {
    name: 'Identifying Sites',
    description: 'Finding and ranking candidate construction sites',
    typicalDuration: 25
  }
}

export default function EnhancedProgressBar({ 
  taskId, 
  title = 'Analysis Progress',
  showDetails = true,
  onComplete,
  onError 
}: EnhancedProgressBarProps) {
  const [stages, setStages] = useState<ProgressStage[]>([])
  const [currentStage, setCurrentStage] = useState<string>('')
  const [overallProgress, setOverallProgress] = useState(0)
  const [startTime, setStartTime] = useState<number | null>(null)
  const [estimatedTotalTime, setEstimatedTotalTime] = useState<number | null>(null)

  const { isConnected } = useWebSocket({
    taskId,
    onProgress: (event) => {
      console.log('[EnhancedProgressBar] Progress update:', event)
      
      setStages(prevStages => {
        const newStages = [...prevStages]
        const stageIndex = newStages.findIndex(s => s.key === event.stage)
        
        if (stageIndex >= 0) {
          // Update existing stage
          newStages[stageIndex] = {
            ...newStages[stageIndex],
            progress: event.progress,
            message: event.message,
            estimatedSecondsRemaining: event.estimated_seconds_remaining
          }
          
          // Mark previous stages as complete
          for (let i = 0; i < stageIndex; i++) {
            if (newStages[i].progress < 1.0) {
              newStages[i].progress = 1.0
              newStages[i].duration = Date.now() - (newStages[i].startTime || Date.now())
            }
          }
        } else {
          // Add new stage
          const stageDef = STAGE_DEFINITIONS[event.stage]
          newStages.push({
            name: stageDef?.name || event.stage,
            key: event.stage,
            progress: event.progress,
            message: event.message,
            estimatedSecondsRemaining: event.estimated_seconds_remaining,
            startTime: Date.now()
          })
        }
        
        return newStages
      })
      
      setCurrentStage(event.stage)
      setOverallProgress(event.progress)
      
      // Calculate total estimated time
      if (event.estimated_seconds_remaining && startTime) {
        const elapsed = (Date.now() - startTime) / 1000
        const total = elapsed + event.estimated_seconds_remaining
        setEstimatedTotalTime(total)
      }
      
      // Check for completion
      if (event.progress >= 1.0) {
        setTimeout(() => onComplete?.(), 500) // Small delay for visual feedback
      }
    },
    onError: (error) => {
      console.error('[EnhancedProgressBar] WebSocket error:', error)
      onError?.(error.message)
    }
  })

  useEffect(() => {
    if (taskId) {
      setStartTime(Date.now())
      setStages([])
      setCurrentStage('')
      setOverallProgress(0)
      setEstimatedTotalTime(null)
    } else {
      setStartTime(null)
    }
  }, [taskId])

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.ceil(seconds)}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.ceil(seconds % 60)
    return `${minutes}m ${remainingSeconds}s`
  }

  const formatDuration = (ms: number): string => {
    const seconds = ms / 1000
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = (seconds % 60).toFixed(1)
    return `${minutes}m ${remainingSeconds}s`
  }

  if (!taskId) {
    return null
  }

  const elapsedTime = startTime ? (Date.now() - startTime) / 1000 : 0
  const remainingTime = estimatedTotalTime ? Math.max(0, estimatedTotalTime - elapsedTime) : null

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`} />
          <span className="text-sm text-gray-400">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Overall Progress */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-300 mb-2">
          <span>Overall Progress</span>
          <span>{(overallProgress * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-3">
          <div 
            className="bg-gradient-to-r from-blue-500 to-green-500 h-3 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${overallProgress * 100}%` }}
          />
        </div>
        {remainingTime !== null && (
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>Elapsed: {formatTime(elapsedTime)}</span>
            <span>Remaining: {formatTime(remainingTime)}</span>
          </div>
        )}
      </div>

      {/* Stage Details */}
      {showDetails && stages.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Analysis Stages</h4>
          {stages.map((stage) => {
            const stageDef = STAGE_DEFINITIONS[stage.key]
            const isActive = stage.key === currentStage
            const isCompleted = stage.progress >= 1.0
            
            return (
              <div key={stage.key} className={`p-3 rounded-lg border ${
                isActive ? 'border-blue-500 bg-blue-900/20' : 
                isCompleted ? 'border-green-500 bg-green-900/10' : 
                'border-gray-600 bg-gray-750'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      isActive ? 'bg-blue-500 animate-pulse' :
                      isCompleted ? 'bg-green-500' :
                      'bg-gray-500'
                    }`} />
                    <span className="text-sm font-medium text-white">
                      {stage.name}
                    </span>
                  </div>
                  <span className="text-xs text-gray-400">
                    {(stage.progress * 100).toFixed(0)}%
                  </span>
                </div>
                
                <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      isActive ? 'bg-blue-500' :
                      isCompleted ? 'bg-green-500' :
                      'bg-gray-500'
                    }`}
                    style={{ width: `${stage.progress * 100}%` }}
                  />
                </div>
                
                <div className="flex justify-between items-center">
                  <p className="text-xs text-gray-300">{stage.message}</p>
                  {stage.duration && (
                    <span className="text-xs text-gray-400">
                      {formatDuration(stage.duration)}
                    </span>
                  )}
                </div>
                
                {stage.estimatedSecondsRemaining && stage.progress < 1.0 && (
                  <div className="text-xs text-gray-400 mt-1">
                    ~{formatTime(stage.estimatedSecondsRemaining)} remaining
                  </div>
                )}
                
                {stageDef && (
                  <p className="text-xs text-gray-500 mt-1">{stageDef.description}</p>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Technical Details */}
      {showDetails && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <details className="text-xs text-gray-400">
            <summary className="cursor-pointer hover:text-gray-300">
              Technical Details
            </summary>
            <div className="mt-2 space-y-1">
              <div>Task ID: <code className="text-xs bg-gray-900 px-1 py-0.5 rounded">{taskId}</code></div>
              <div>Connection: {isConnected ? 'WebSocket Active' : 'WebSocket Disconnected'}</div>
              <div>Stages Completed: {stages.filter(s => s.progress >= 1.0).length}/{stages.length}</div>
              {currentStage && <div>Current Stage: {currentStage}</div>}
            </div>
          </details>
        </div>
      )}
    </div>
  )
}