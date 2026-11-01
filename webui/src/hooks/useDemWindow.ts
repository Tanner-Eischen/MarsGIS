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

type RawTileResponse = {
  x: number
  y: number
  z: number
  rows: number
  cols: number
  bbox: {
    left: number
    bottom: number
    right: number
    top: number
  }
  nodata: number | null
  values: Float32Array
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

const RAW_TILE_MAX_REQUESTS = 16
const RAW_TILE_MIN_Z = 2
const RAW_TILE_MAX_Z = 10
const RAW_TILE_ZOOM_BY_SPAN = [
  { span: 20, z: 2 },
  { span: 10, z: 3 },
  { span: 5, z: 4 },
  { span: 2, z: 5 },
  { span: 1, z: 6 },
  { span: 0.5, z: 7 },
] as const

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const normalizeLon180 = (lon: number) => {
  let normalized = lon
  while (normalized < -180) normalized += 360
  while (normalized >= 180) normalized -= 360
  return normalized
}

const buildApiError = (status: number, payload: any): DemWindowError => ({
  isServiceUnavailable: status === 503,
  status,
  message: payload?.detail || payload?.error || `Request failed (${status})`,
  detail: payload?.detail ?? null,
})

const parseHeaderBool = (value: string | null) => value === 'true'
const ALLOW_TIFF_FALLBACK = import.meta.env.VITE_ALLOW_DEM_WINDOW_TIFF_FALLBACK === 'true'

const parseNpyFloat32 = (buffer: ArrayBuffer): { rows: number; cols: number; values: Float32Array } => {
  const bytes = new Uint8Array(buffer)
  if (bytes.length < 16) throw new Error('Invalid NPY payload (too small)')
  if (bytes[0] !== 0x93 || bytes[1] !== 0x4e || bytes[2] !== 0x55 || bytes[3] !== 0x4d || bytes[4] !== 0x50 || bytes[5] !== 0x59) {
    throw new Error('Invalid NPY payload (bad magic)')
  }

  const major = bytes[6]
  const headerLengthOffset = major <= 1 ? 8 : 10
  const headerLengthSize = major <= 1 ? 2 : 4
  const headerLengthView = new DataView(buffer, headerLengthOffset, headerLengthSize)
  const headerLength = major <= 1 ? headerLengthView.getUint16(0, true) : headerLengthView.getUint32(0, true)
  const headerStart = major <= 1 ? 10 : 12
  const headerEnd = headerStart + headerLength
  if (headerEnd > bytes.length) throw new Error('Invalid NPY payload (header overflow)')

  const header = new TextDecoder('ascii').decode(bytes.subarray(headerStart, headerEnd))
  const descrMatch = /'descr'\s*:\s*'([^']+)'/.exec(header)
  const shapeMatch = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)
  const fortranMatch = /'fortran_order'\s*:\s*(True|False)/.exec(header)

  if (!descrMatch || !shapeMatch) throw new Error('Invalid NPY header')
  if (fortranMatch?.[1] === 'True') throw new Error('Fortran-order NPY arrays are not supported')

  const dims = shapeMatch[1]
    .split(',')
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
    .map((v) => parseInt(v, 10))
    .filter((v) => Number.isFinite(v))

  if (dims.length !== 2) throw new Error('Expected 2D NPY array')
  const rows = dims[0]
  const cols = dims[1]
  const dtype = descrMatch[1]
  const dataOffset = headerEnd
  const expectedValues = rows * cols

  if (dtype === '<f4' || dtype === '|f4' || dtype === 'f4') {
    const raw = buffer.slice(dataOffset)
    const values = new Float32Array(raw)
    if (values.length < expectedValues) throw new Error('NPY payload shorter than expected')
    return { rows, cols, values: values.subarray(0, expectedValues) }
  }
  if (dtype === '<f8' || dtype === '|f8' || dtype === 'f8') {
    const raw = buffer.slice(dataOffset)
    const values64 = new Float64Array(raw)
    if (values64.length < expectedValues) throw new Error('NPY payload shorter than expected')
    const values32 = new Float32Array(expectedValues)
    for (let i = 0; i < expectedValues; i++) values32[i] = values64[i]
    return { rows, cols, values: values32 }
  }
  throw new Error(`Unsupported NPY dtype: ${dtype}`)
}

const estimateInitialZoom = (roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }) => {
  const latSpan = Math.abs(roi.lat_max - roi.lat_min)
  const lonSpan = Math.abs(roi.lon_max - roi.lon_min)
  const span = Math.max(latSpan, lonSpan)
  for (const item of RAW_TILE_ZOOM_BY_SPAN) {
    if (span >= item.span) return item.z
  }
  return 8
}

const tileBoundsForRoi = (
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
  z: number
): { xMin: number; xMax: number; yMin: number; yMax: number } | null => {
  const latMin = clamp(Math.min(roi.lat_min, roi.lat_max), -90, 90)
  const latMax = clamp(Math.max(roi.lat_min, roi.lat_max), -90, 90)
  const lonMin = normalizeLon180(roi.lon_min)
  const lonMax = normalizeLon180(roi.lon_max)

  if (lonMax < lonMin) return null

  const xTiles = 2 ** (z + 1)
  const yTiles = 2 ** z

  const toX = (lon: number) => clamp(Math.floor(((lon + 180) / 360) * xTiles), 0, xTiles - 1)
  const toY = (lat: number) => clamp(Math.floor(((90 - lat) / 180) * yTiles), 0, yTiles - 1)

  const xMin = toX(lonMin)
  const xMax = toX(lonMax === 180 ? 179.999999 : lonMax)
  const yMin = toY(latMax)
  const yMax = toY(latMin)
  return { xMin, xMax, yMin, yMax }
}

const copyTileToRow = (
  rowTarget: Float32Array,
  rowWidth: number,
  writeOffset: number,
  tile: RawTileResponse,
  rowHeight: number,
  trimLeftCols: number
) => {
  const srcWidth = tile.cols
  const copyWidth = Math.max(0, srcWidth - trimLeftCols)
  for (let r = 0; r < rowHeight; r++) {
    const srcStart = r * srcWidth + trimLeftCols
    const srcEnd = srcStart + copyWidth
    rowTarget.set(tile.values.subarray(srcStart, srcEnd), r * rowWidth + writeOffset)
  }
}

const stitchRawTiles = (tiles: RawTileResponse[]) => {
  if (tiles.length === 0) throw new Error('No raw tiles to stitch')

  const byRow = new Map<number, RawTileResponse[]>()
  for (const tile of tiles) {
    const rowTiles = byRow.get(tile.y) || []
    rowTiles.push(tile)
    byRow.set(tile.y, rowTiles)
  }

  const rowKeys = Array.from(byRow.keys()).sort((a, b) => a - b)
  const rowChunks = rowKeys.map((rowKey) => {
    const rowTiles = (byRow.get(rowKey) || []).sort((a, b) => a.x - b.x)
    const rowHeight = Math.min(...rowTiles.map((t) => t.rows))
    let rowWidth = 0

    for (let i = 0; i < rowTiles.length; i++) {
      const tile = rowTiles[i]
      const prev = i > 0 ? rowTiles[i - 1] : null
      const touching = prev ? Math.abs(prev.bbox.right - tile.bbox.left) < 1e-8 : false
      const trimLeft = touching ? 1 : 0
      rowWidth += Math.max(0, tile.cols - trimLeft)
    }

    const rowData = new Float32Array(rowHeight * rowWidth)
    let offset = 0
    for (let i = 0; i < rowTiles.length; i++) {
      const tile = rowTiles[i]
      const prev = i > 0 ? rowTiles[i - 1] : null
      const touching = prev ? Math.abs(prev.bbox.right - tile.bbox.left) < 1e-8 : false
      const trimLeft = touching ? 1 : 0
      copyTileToRow(rowData, rowWidth, offset, tile, rowHeight, trimLeft)
      offset += Math.max(0, tile.cols - trimLeft)
    }

    return {
      rowData,
      width: rowWidth,
      height: rowHeight,
      top: Math.max(...rowTiles.map((t) => t.bbox.top)),
      bottom: Math.min(...rowTiles.map((t) => t.bbox.bottom)),
      left: rowTiles[0].bbox.left,
      right: rowTiles[rowTiles.length - 1].bbox.right,
    }
  })

  const targetWidth = Math.min(...rowChunks.map((r) => r.width))
  let totalHeight = 0
  for (let i = 0; i < rowChunks.length; i++) {
    const row = rowChunks[i]
    const prev = i > 0 ? rowChunks[i - 1] : null
    const touching = prev ? Math.abs(prev.bottom - row.top) < 1e-8 : false
    const trimTop = touching ? 1 : 0
    totalHeight += Math.max(0, row.height - trimTop)
  }

  const merged = new Float32Array(totalHeight * targetWidth)
  let writeRow = 0
  for (let i = 0; i < rowChunks.length; i++) {
    const row = rowChunks[i]
    const prev = i > 0 ? rowChunks[i - 1] : null
    const touching = prev ? Math.abs(prev.bottom - row.top) < 1e-8 : false
    const trimTop = touching ? 1 : 0
    const copyRows = Math.max(0, row.height - trimTop)

    for (let r = 0; r < copyRows; r++) {
      const srcRow = r + trimTop
      const srcStart = srcRow * row.width
      const srcEnd = srcStart + targetWidth
      merged.set(row.rowData.subarray(srcStart, srcEnd), (writeRow + r) * targetWidth)
    }
    writeRow += copyRows
  }

  return {
    values: merged,
    width: targetWidth,
    height: totalHeight,
    bbox: {
      left: Math.max(...rowChunks.map((r) => r.left)),
      right: Math.min(...rowChunks.map((r) => r.right)),
      top: rowChunks[0].top,
      bottom: rowChunks[rowChunks.length - 1].bottom,
    },
  }
}

const clipRawWindowToRoi = (
  merged: {
    values: Float32Array
    width: number
    height: number
    bbox: { left: number; right: number; bottom: number; top: number }
  },
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
) => {
  const left = merged.bbox.left
  const right = merged.bbox.right
  const top = merged.bbox.top
  const bottom = merged.bbox.bottom

  if (merged.width <= 1 || merged.height <= 1 || right <= left || top <= bottom) {
    return merged
  }

  const roiLonMin = clamp(Math.min(roi.lon_min, roi.lon_max), left, right)
  const roiLonMax = clamp(Math.max(roi.lon_min, roi.lon_max), left, right)
  const roiLatMin = clamp(Math.min(roi.lat_min, roi.lat_max), bottom, top)
  const roiLatMax = clamp(Math.max(roi.lat_min, roi.lat_max), bottom, top)

  const colStart = clamp(Math.floor(((roiLonMin - left) / (right - left)) * (merged.width - 1)), 0, merged.width - 1)
  const colEnd = clamp(Math.ceil(((roiLonMax - left) / (right - left)) * (merged.width - 1)), colStart, merged.width - 1)
  const rowStart = clamp(Math.floor(((top - roiLatMax) / (top - bottom)) * (merged.height - 1)), 0, merged.height - 1)
  const rowEnd = clamp(Math.ceil(((top - roiLatMin) / (top - bottom)) * (merged.height - 1)), rowStart, merged.height - 1)

  const outWidth = Math.max(1, colEnd - colStart + 1)
  const outHeight = Math.max(1, rowEnd - rowStart + 1)
  const out = new Float32Array(outWidth * outHeight)

  for (let r = 0; r < outHeight; r++) {
    const srcRow = rowStart + r
    const srcStart = srcRow * merged.width + colStart
    const srcEnd = srcStart + outWidth
    out.set(merged.values.subarray(srcStart, srcEnd), r * outWidth)
  }

  const toLon = (col: number) => left + (col / Math.max(1, merged.width - 1)) * (right - left)
  const toLat = (row: number) => top - (row / Math.max(1, merged.height - 1)) * (top - bottom)

  return {
    values: out,
    width: outWidth,
    height: outHeight,
    bbox: {
      left: toLon(colStart),
      right: toLon(colEnd),
      top: toLat(rowStart),
      bottom: toLat(rowEnd),
    },
  }
}

const fetchRawTile = async (
  dataset: string,
  z: number,
  x: number,
  y: number,
  signal: AbortSignal
): Promise<RawTileResponse> => {
  const path = `/visualization/tiles/elevation-raw/${encodeURIComponent(dataset)}/${z}/${x}/${y}.npy?stride=1&allow_download=true`
  const response = await fetch(apiUrl(path), { signal })
  if (!response.ok) {
    let detail: string
    try {
      const payload = await response.json()
      detail = payload?.detail || payload?.error || ''
    } catch {
      detail = await response.text()
    }
    const error = new Error(`Raw tile request failed (${response.status}) ${detail}`)
    ;(error as any).status = response.status
    throw error
  }

  const payload = await response.arrayBuffer()
  const decoded = parseNpyFloat32(payload)

  return {
    x,
    y,
    z,
    rows: decoded.rows,
    cols: decoded.cols,
    bbox: {
      left: Number(response.headers.get('X-Bounds-Left') || 0),
      right: Number(response.headers.get('X-Bounds-Right') || 0),
      bottom: Number(response.headers.get('X-Bounds-Bottom') || 0),
      top: Number(response.headers.get('X-Bounds-Top') || 0),
    },
    nodata: (() => {
      const raw = response.headers.get('X-NoData-Value')
      if (raw == null) return null
      const v = Number(raw)
      return Number.isFinite(v) ? v : null
    })(),
    values: decoded.values,
    datasetRequested: response.headers.get('X-Dataset-Requested') || dataset,
    datasetUsed: response.headers.get('X-Dataset-Used') || dataset,
    isFallback: parseHeaderBool(response.headers.get('X-Fallback-Used')),
    fallbackReason: response.headers.get('X-Fallback-Reason'),
  }
}

const fetchRawElevationWindow = async (
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
  dataset: string,
  signal: AbortSignal
): Promise<DemWindowData> => {
  let z = clamp(estimateInitialZoom(roi), RAW_TILE_MIN_Z, RAW_TILE_MAX_Z)
  let lastError: any = null

  for (let attempt = 0; attempt < 6; attempt++) {
    const tileRange = tileBoundsForRoi(roi, z)
    if (!tileRange) throw new Error('Raw tile mode does not support dateline-crossing ROI')

    const xCount = tileRange.xMax - tileRange.xMin + 1
    const yCount = tileRange.yMax - tileRange.yMin + 1
    const tileCount = xCount * yCount
    if (tileCount > RAW_TILE_MAX_REQUESTS) {
      z = Math.max(RAW_TILE_MIN_Z, z - 1)
      continue
    }

    const requests: Promise<RawTileResponse>[] = []
    for (let ty = tileRange.yMin; ty <= tileRange.yMax; ty++) {
      for (let tx = tileRange.xMin; tx <= tileRange.xMax; tx++) {
        requests.push(fetchRawTile(dataset, z, tx, ty, signal))
      }
    }

    try {
      const tiles = await Promise.all(requests)
      const stitched = stitchRawTiles(tiles)
      const clipped = clipRawWindowToRoi(stitched, roi)
      const first = tiles[0]

      return {
        width: clipped.width,
        height: clipped.height,
        nodata: first.nodata,
        bbox: clipped.bbox,
        elevations: clipped.values,
        datasetRequested: first.datasetRequested,
        datasetUsed: first.datasetUsed,
        isFallback: tiles.some((t) => t.isFallback),
        fallbackReason: tiles.map((t) => t.fallbackReason).find((v) => Boolean(v)) || null,
      }
    } catch (error: any) {
      lastError = error
      const status = Number(error?.status)
      if (status === 413 && z < RAW_TILE_MAX_Z) {
        z += 1
        continue
      }
      throw error
    }
  }

  if (lastError) throw lastError
  throw new Error('Could not resolve raw elevation tiles for ROI')
}

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
        // Prefer native raw elevation samples (no averaging/resampling).
        try {
          const rawWindow = await fetchRawElevationWindow(roi, dataset, controller.signal)
          if (requestIdRef.current !== currentRequestId) return
          setData(rawWindow)
          return
        } catch (rawError: any) {
          if (!ALLOW_TIFF_FALLBACK) {
            throw rawError
          }
          // Continue to GeoTIFF fallback only when explicitly enabled.
        }

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
