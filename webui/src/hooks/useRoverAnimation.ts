import { useState, useEffect, useRef, useCallback } from 'react'
import { sampleTerrainElevation, calculateBearing } from '../utils/terrainSampling'

export interface RoverPosition {
  lat: number
  lon: number
  elevation: number
  heading: number // degrees 0-360
  progress: number // 0-1 along current segment
  waypointIndex: number // current waypoint segment (0 to waypoints.length-2)
}

export interface AnimationState {
  isPlaying: boolean
  speed: number // multiplier (1.0 = normal, 2.0 = 2x speed)
  progress: number // 0-1 along entire route
  currentPosition: RoverPosition | null
  duration: number // total animation duration in seconds
}

interface Waypoint {
  lat: number
  lon: number
}

interface TerrainData {
  x: number[][]
  y: number[][]
  z: number[][]
}

/**
 * Extract waypoints from GeoJSON LineString features
 */
function extractWaypoints(waypointsGeoJson: any): Waypoint[] {
  if (!waypointsGeoJson || !waypointsGeoJson.features) {
    return []
  }

  const waypoints: Waypoint[] = []

  for (const feature of waypointsGeoJson.features) {
    if (feature.geometry && feature.geometry.type === 'LineString') {
      const coords = feature.geometry.coordinates as number[][]
      for (const coord of coords) {
        waypoints.push({
          lon: coord[0],
          lat: coord[1],
        })
      }
    }
  }

  return waypoints
}

/**
 * Hook for animating rover along waypoint path
 */
export function useRoverAnimation(
  waypointsGeoJson: any,
  terrainData: TerrainData | null,
  onPositionUpdate: (pos: RoverPosition) => void
) {
  const [state, setState] = useState<AnimationState>({
    isPlaying: false,
    speed: 1.0,
    progress: 0,
    currentPosition: null,
    duration: 30, // default 30 seconds
  })

  const animationFrameRef = useRef<number | null>(null)
  const startTimeRef = useRef<number | null>(null)
  const pausedProgressRef = useRef<number>(0)
  const waypointsRef = useRef<Waypoint[]>([])

  // Extract waypoints when GeoJSON changes
  useEffect(() => {
    if (waypointsGeoJson) {
      waypointsRef.current = extractWaypoints(waypointsGeoJson)
    } else {
      waypointsRef.current = []
    }
  }, [waypointsGeoJson])

  // Calculate total route distance and duration
  useEffect(() => {
    const waypoints = waypointsRef.current
    if (waypoints.length < 2) {
      setState((prev) => ({ ...prev, duration: 0 }))
      return
    }

    // Calculate total distance (simplified - sum of segment distances)
    let totalDistance = 0
    for (let i = 0; i < waypoints.length - 1; i++) {
      const lat1 = waypoints[i].lat
      const lon1 = waypoints[i].lon
      const lat2 = waypoints[i + 1].lat
      const lon2 = waypoints[i + 1].lon

      // Haversine distance approximation
      const dLat = ((lat2 - lat1) * Math.PI) / 180
      const dLon = ((lon2 - lon1) * Math.PI) / 180
      const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos((lat1 * Math.PI) / 180) *
          Math.cos((lat2 * Math.PI) / 180) *
          Math.sin(dLon / 2) *
          Math.sin(dLon / 2)
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
      const distance = 6371000 * c // Earth radius in meters (approximate for Mars)

      totalDistance += distance
    }

    // Estimate duration based on average rover speed (0.5 m/s)
    const avgSpeed = 0.5 // m/s
    const duration = totalDistance / avgSpeed
    setState((prev) => ({ ...prev, duration: Math.max(10, duration) }))
  }, [waypointsGeoJson])

  // Animation loop
  const animate = useCallback(() => {
    const waypoints = waypointsRef.current
    if (waypoints.length < 2 || !terrainData) {
      return
    }

    const now = performance.now()
    if (startTimeRef.current === null) {
      startTimeRef.current = now
    }

    const elapsed = (now - startTimeRef.current) / 1000 // seconds
    const totalDuration = state.duration / state.speed
    const currentProgress = Math.min(1, pausedProgressRef.current + elapsed / totalDuration)

    // Find current segment
    const totalSegments = waypoints.length - 1
    const segmentProgress = currentProgress * totalSegments
    const segmentIndex = Math.floor(segmentProgress)
    const segmentLocalProgress = segmentProgress - segmentIndex

    if (segmentIndex >= totalSegments) {
      // Animation complete
      setState((prev) => ({
        ...prev,
        isPlaying: false,
        progress: 1,
      }))
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
    }

    const wp1 = waypoints[segmentIndex]
    const wp2 = waypoints[segmentIndex + 1]

    // Interpolate position
    const lat = wp1.lat + (wp2.lat - wp1.lat) * segmentLocalProgress
    const lon = wp1.lon + (wp2.lon - wp1.lon) * segmentLocalProgress

    // Sample terrain elevation
    const elevation = sampleTerrainElevation(lat, lon, terrainData)

    // Calculate heading
    const heading = calculateBearing(wp1.lat, wp1.lon, wp2.lat, wp2.lon)

    const position: RoverPosition = {
      lat,
      lon,
      elevation,
      heading,
      progress: segmentLocalProgress,
      waypointIndex: segmentIndex,
    }

    setState((prev) => ({
      ...prev,
      currentPosition: position,
      progress: currentProgress,
    }))

    onPositionUpdate(position)

    if (state.isPlaying) {
      animationFrameRef.current = requestAnimationFrame(animate)
    }
  }, [terrainData, state.duration, state.speed, state.isPlaying, onPositionUpdate])

  // Start animation loop when playing
  useEffect(() => {
    if (state.isPlaying) {
      animationFrameRef.current = requestAnimationFrame(animate)
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [state.isPlaying, animate])

  const play = useCallback(() => {
    if (waypointsRef.current.length < 2) return

    if (!state.isPlaying) {
      startTimeRef.current = performance.now()
      setState((prev) => ({ ...prev, isPlaying: true }))
    }
  }, [state.isPlaying])

  const pause = useCallback(() => {
    if (state.isPlaying) {
      pausedProgressRef.current = state.progress
      startTimeRef.current = null
      setState((prev) => ({ ...prev, isPlaying: false }))
    }
  }, [state.isPlaying, state.progress])

  const reset = useCallback(() => {
    pausedProgressRef.current = 0
    startTimeRef.current = null
    setState((prev) => ({
      ...prev,
      isPlaying: false,
      progress: 0,
      currentPosition: null,
    }))
  }, [])

  const setSpeed = useCallback((speed: number) => {
    setState((prev) => ({ ...prev, speed: Math.max(0.1, Math.min(5, speed)) }))
  }, [])

  const seek = useCallback((progress: number) => {
    const clampedProgress = Math.max(0, Math.min(1, progress))
    pausedProgressRef.current = clampedProgress
    startTimeRef.current = null

    // Update position immediately
    const waypoints = waypointsRef.current
    if (waypoints.length >= 2 && terrainData) {
      const totalSegments = waypoints.length - 1
      const segmentProgress = clampedProgress * totalSegments
      const segmentIndex = Math.floor(segmentProgress)
      const segmentLocalProgress = segmentProgress - segmentIndex

      if (segmentIndex < totalSegments) {
        const wp1 = waypoints[segmentIndex]
        const wp2 = waypoints[segmentIndex + 1]

        const lat = wp1.lat + (wp2.lat - wp1.lat) * segmentLocalProgress
        const lon = wp1.lon + (wp2.lon - wp1.lon) * segmentLocalProgress
        const elevation = sampleTerrainElevation(lat, lon, terrainData)
        const heading = calculateBearing(wp1.lat, wp1.lon, wp2.lat, wp2.lon)

        const position: RoverPosition = {
          lat,
          lon,
          elevation,
          heading,
          progress: segmentLocalProgress,
          waypointIndex: segmentIndex,
        }

        setState((prev) => ({
          ...prev,
          currentPosition: position,
          progress: clampedProgress,
        }))

        onPositionUpdate(position)
      }
    }
  }, [terrainData, onPositionUpdate])

  return {
    state,
    play,
    pause,
    reset,
    setSpeed,
    seek,
  }
}

