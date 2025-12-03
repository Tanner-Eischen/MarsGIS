import { useEffect, useRef, useState, useCallback } from 'react'

interface ProgressEvent {
  task_id: string
  stage: string
  progress: number
  message: string
  estimated_seconds_remaining?: number
}

interface UseWebSocketOptions {
  taskId: string | null
  onProgress?: (event: ProgressEvent) => void
  onError?: (error: Error) => void
  onConnect?: () => void
  onDisconnect?: () => void
}

export function useWebSocket({
  taskId,
  onProgress,
  onError,
  onConnect,
  onDisconnect,
}: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<ProgressEvent | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(() => {
    if (!taskId) {
      return
    }

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // Determine WebSocket URL (use ws:// for localhost, wss:// for production)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.hostname === 'localhost' 
      ? 'localhost:5000' 
      : window.location.host
    const wsUrl = `${protocol}//${host}/ws/progress/${taskId}`
    
    console.log('[WebSocket] Connecting to:', wsUrl)

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[WebSocket] Connected for task:', taskId)
        setIsConnected(true)
        reconnectAttempts.current = 0
        onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          // Handle different message types
          if (data.type === 'connected' || data.type === 'ping' || data.type === 'pong') {
            // Connection/ping messages, ignore
            return
          }

          // Progress event
          if (data.task_id && data.stage && typeof data.progress === 'number') {
            const progressEvent: ProgressEvent = {
              task_id: data.task_id,
              stage: data.stage,
              progress: data.progress,
              message: data.message || '',
              estimated_seconds_remaining: data.estimated_seconds_remaining,
            }
            console.log('[WebSocket] Progress update:', progressEvent)
            setLastEvent(progressEvent)
            onProgress?.(progressEvent)
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
          onError?.(error as Error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        onError?.(new Error('WebSocket connection error'))
      }

      ws.onclose = () => {
        setIsConnected(false)
        onDisconnect?.()

        // Attempt to reconnect if we haven't exceeded max attempts and taskId is still valid
        // Use a more conservative reconnection strategy
        if (reconnectAttempts.current < maxReconnectAttempts && taskId) {
          reconnectAttempts.current++
          const delay = Math.min(2000 * reconnectAttempts.current, 10000) // Linear backoff: 2s, 4s, 6s, 8s, 10s
          reconnectTimeoutRef.current = setTimeout(() => {
            // Check taskId is still valid before reconnecting
            if (taskId && wsRef.current === null) { // Only reconnect if no active connection
              connect()
            }
          }, delay)
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket:', error)
      onError?.(error as Error)
    }
  }, [taskId]) // Remove callback dependencies to prevent infinite re-renders

  useEffect(() => {
    // Reset reconnect attempts when taskId changes
    reconnectAttempts.current = 0
    
    if (taskId) {
      connect()
    } else {
      // If taskId is null, close any existing connection
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      setIsConnected(false)
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      setIsConnected(false)
    }
  }, [taskId, connect])

  return {
    isConnected,
    lastEvent,
    reconnect: connect,
  }
}

