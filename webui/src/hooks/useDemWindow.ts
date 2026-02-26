import { useEffect, useMemo, useRef, useState } from 'react'
import { apiUrl } from '../lib/apiBase'

type DemWindowError = {
  isServiceUnavailable: boolean
  status: number | null
  message: string
  detail: string | null
}

type DemWindowData = {
  width: number
  height: number
  nodata: number | null
  bbox: {
    left: number
    bottom: number
    right: number
    top: number
  }
  elevations: Float32Array
  datasetRequested: string
  datasetUsed: string
  isFallback: boolean
  fallbackReason: string | null
}

type DecodeWorkerSuccess = {
  requestId: number
  ok: true
  width: number
  height: number
  nodata: number | null
  bbox: { left: number; bottom: number; right: number; top: number }
  dataBuffer: ArrayBuffer
}

type DecodeWorkerFailure = {
  requestId: number
  ok: false
  error: string
}

const buildApiError = (status: number, payload: any): DemWindowError => ({
  isServiceUnavailable: status === 503,
  status,
  message: payload?.detail || payload?.error || `Request failed (${status})`,
  detail: payload?.detail ?? null,
})

const parseHeaderBool = (value: string | null) => value === 'true'

export function useDemWindow(
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null,
  dataset: string,
  width = 256,
  height = 256,
  buffer = 0
) {
  const [data, setData] = useState<DemWindowData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<DemWindowError | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const requestIdRef = useRef(0)

  const roiKey = useMemo(
    () =>
      roi
        ? `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}|${dataset}|${width}x${height}|${buffer}`
        : '',
    [roi, dataset, width, height, buffer]
  )

  useEffect(() => {
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL('../workers/demWindowDecoder.worker.ts', import.meta.url), {
        type: 'module',
      })
    }
    return () => {
      workerRef.current?.terminate()
      workerRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!roi || !roiKey) return

    const currentRequestId = ++requestIdRef.current
    const controller = new AbortController()
    setLoading(true)
    setError(null)

    const decodeInWorker = (bufferValue: ArrayBuffer) =>
      new Promise<DecodeWorkerSuccess>((resolve, reject) => {
        const worker = workerRef.current
        if (!worker) {
          reject(new Error('DEM decoder worker unavailable'))
          return
        }

        const onMessage = (event: MessageEvent<DecodeWorkerSuccess | DecodeWorkerFailure>) => {
          const payload = event.data
          if (payload.requestId !== currentRequestId) return
          worker.removeEventListener('message', onMessage)
          if (!payload.ok) {
            reject(new Error('error' in payload ? payload.error : 'DEM decode failed'))
            return
          }
          resolve(payload)
        }

        worker.addEventListener('message', onMessage)
        worker.postMessage({ requestId: currentRequestId, buffer: bufferValue }, [bufferValue])
      })

    const fetchWindow = async () => {
      try {
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`
        const params = new URLSearchParams({
          dataset,
          roi: roiStr,
          width: String(width),
          height: String(height),
          buffer: String(buffer),
        })
        const response = await fetch(apiUrl(`/visualization/dem-window.tif?${params.toString()}`), {
          signal: controller.signal,
        })
        if (!response.ok) {
          let payload: any = null
          try {
            payload = await response.json()
          } catch {
            payload = null
          }
          throw buildApiError(response.status, payload)
        }

        const datasetRequested = response.headers.get('X-Dataset-Requested') || dataset
        const datasetUsed = response.headers.get('X-Dataset-Used') || dataset
        const fallbackReason = response.headers.get('X-Fallback-Reason')
        const isFallback = parseHeaderBool(response.headers.get('X-Fallback-Used'))

        const responseBuffer = await response.arrayBuffer()
        const decoded = await decodeInWorker(responseBuffer)
        if (requestIdRef.current !== currentRequestId) return

        setData({
          width: decoded.width,
          height: decoded.height,
          nodata: decoded.nodata,
          bbox: decoded.bbox,
          elevations: new Float32Array(decoded.dataBuffer),
          datasetRequested,
          datasetUsed,
          isFallback,
          fallbackReason,
        })
      } catch (err: any) {
        if (controller.signal.aborted) return
        if (err?.status) {
          setError(err)
        } else {
          setError({
            isServiceUnavailable: false,
            status: null,
            message: err?.message || 'Failed to fetch DEM window',
            detail: null,
          })
        }
      } finally {
        if (requestIdRef.current === currentRequestId) {
          setLoading(false)
        }
      }
    }

    fetchWindow()

    return () => {
      controller.abort()
    }
  }, [roiKey, roi, dataset, width, height, buffer])

  return { data, loading, error }
}
