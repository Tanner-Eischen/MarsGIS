interface RoverAnimationControlsProps {
  waypointsGeoJson: any
  terrainData: { x: number[][]; y: number[][]; z: number[][] } | null
  animation: {
    state: {
      isPlaying: boolean
      speed: number
      progress: number
      currentPosition: any
      duration: number
    }
    play: () => void
    pause: () => void
    reset: () => void
    setSpeed: (speed: number) => void
    seek: (progress: number) => void
  }
}

export default function RoverAnimationControls({
  waypointsGeoJson,
  terrainData,
  animation,
}: RoverAnimationControlsProps) {

  const hasWaypoints =
    waypointsGeoJson &&
    waypointsGeoJson.features &&
    waypointsGeoJson.features.length > 0

  if (!hasWaypoints) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <p className="text-gray-400 text-sm">
          No waypoints available. Generate a route to enable rover animation.
        </p>
      </div>
    )
  }

  const { state, play, pause, reset, setSpeed, seek } = animation

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Rover Animation</h3>
        <button
          onClick={reset}
          className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-md transition-colors"
        >
          Reset
        </button>
      </div>

      {/* Play/Pause Controls */}
      <div className="flex items-center space-x-4">
        <button
          onClick={state.isPlaying ? pause : play}
          className={`px-6 py-2 rounded-md font-medium transition-colors ${
            state.isPlaying
              ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
              : 'bg-green-600 hover:bg-green-700 text-white'
          }`}
        >
          {state.isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>

        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Speed: {state.speed.toFixed(1)}x
          </label>
          <input
            type="range"
            min="0.1"
            max="5"
            step="0.1"
            value={state.speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Progress Bar */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Progress: {Math.round(state.progress * 100)}%
        </label>
        <div className="relative">
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={state.progress}
            onChange={(e) => seek(parseFloat(e.target.value))}
            className="w-full"
          />
          <div
            className="absolute top-0 left-0 h-1 bg-blue-500 rounded"
            style={{ width: `${state.progress * 100}%` }}
          />
        </div>
      </div>

      {/* Current Position Display */}
      {state.currentPosition && (
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-gray-700">
          <div>
            <div className="text-xs text-gray-400">Position</div>
            <div className="text-sm text-white">
              Lat: {state.currentPosition.lat.toFixed(6)}
            </div>
            <div className="text-sm text-white">
              Lon: {state.currentPosition.lon.toFixed(6)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400">Status</div>
            <div className="text-sm text-white">
              Elevation: {Math.round(state.currentPosition.elevation)} m
            </div>
            <div className="text-sm text-white">
              Heading: {Math.round(state.currentPosition.heading)}°
            </div>
          </div>
          <div className="col-span-2">
            <div className="text-xs text-gray-400">Waypoint Progress</div>
            <div className="text-sm text-white">
              Segment {state.currentPosition.waypointIndex + 1} (
              {Math.round(state.currentPosition.progress * 100)}% complete)
            </div>
          </div>
        </div>
      )}

      {!state.currentPosition && (
        <div className="text-sm text-gray-400 text-center py-2">
          Click Play to start animation
        </div>
      )}
    </div>
  )
}

