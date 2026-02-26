import { useMemo } from 'react'
import { useDemWindow } from './useDemWindow'

const clampMaxPoints = (dataset: string, requested: number) => {
  const isLowRes = dataset === 'mola' || dataset === 'mola_200m'
  return isLowRes ? Math.min(requested, 35000) : Math.min(requested, 15000)
}

const buildMeshFromWindow = (
  elevations: Float32Array,
  width: number,
  height: number,
  bbox: { left: number; right: number; bottom: number; top: number },
  nodata: number | null,
  effectiveMaxPoints: number
) => {
  let downsampleFactor = 1
  const totalPoints = width * height
  if (totalPoints > effectiveMaxPoints) {
    downsampleFactor = Math.max(1, Math.ceil(Math.sqrt(totalPoints / effectiveMaxPoints)))
  }

  const sampledWidth = Math.max(1, Math.floor((width - 1) / downsampleFactor) + 1)
  const sampledHeight = Math.max(1, Math.floor((height - 1) / downsampleFactor) + 1)
  const xAxis = Array.from({ length: sampledWidth }, (_, col) => {
    const srcCol = Math.min(width - 1, col * downsampleFactor)
    const t = width <= 1 ? 0 : srcCol / (width - 1)
    return bbox.left + (bbox.right - bbox.left) * t
  })
  const yAxis = Array.from({ length: sampledHeight }, (_, row) => {
    const srcRow = Math.min(height - 1, row * downsampleFactor)
    const t = height <= 1 ? 0 : srcRow / (height - 1)
    return bbox.top + (bbox.bottom - bbox.top) * t
  })

  const z: number[][] = []
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (let row = 0; row < sampledHeight; row++) {
    const srcRow = Math.min(height - 1, row * downsampleFactor)
    const line: number[] = []
    for (let col = 0; col < sampledWidth; col++) {
      const srcCol = Math.min(width - 1, col * downsampleFactor)
      const value = elevations[srcRow * width + srcCol]
      let out = Number(value)
      if (!Number.isFinite(out) || (nodata !== null && out === nodata)) {
        out = Number.NaN
      }
      line.push(out)
      if (Number.isFinite(out)) {
        min = Math.min(min, out)
        max = Math.max(max, out)
      }
    }
    z.push(line)
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0
    max = 0
  }

  // Fill NaNs so Plotly surfaces stay stable.
  for (let row = 0; row < z.length; row++) {
    for (let col = 0; col < z[row].length; col++) {
      if (!Number.isFinite(z[row][col])) {
        z[row][col] = min
      }
    }
  }

  const xGrid = yAxis.map(() => [...xAxis])
  const yGrid = yAxis.map((lat) => Array.from({ length: sampledWidth }, () => lat))

  return {
    mesh: {
      x: xGrid,
      y: yGrid,
      z,
    },
    bounds: {
      lon_min: bbox.left,
      lon_max: bbox.right,
      lat_min: bbox.bottom,
      lat_max: bbox.top,
    },
    mesh_shape: {
      rows: sampledHeight,
      cols: sampledWidth,
      downsample_factor: downsampleFactor,
    },
    elevation_range: { min, max },
  }
}

export function use3DTerrain(roi, dataset, maxPoints = 15000) {
  const effectiveMaxPoints = clampMaxPoints(dataset, maxPoints)
  const side = Math.max(64, Math.min(384, Math.round(Math.sqrt(effectiveMaxPoints))))
  const { data, loading, error } = useDemWindow(roi || null, dataset, side, side, 0)

  const terrainData = useMemo(() => {
    if (!roi || !data) return null
    const meshData = buildMeshFromWindow(
      data.elevations,
      data.width,
      data.height,
      data.bbox,
      data.nodata,
      effectiveMaxPoints
    )
    return {
      ...meshData,
      dataset_requested: data.datasetRequested,
      dataset_used: data.datasetUsed,
      is_fallback: data.isFallback,
      fallback_reason: data.fallbackReason,
      max_points_requested: maxPoints,
      max_points_effective: effectiveMaxPoints,
      z_exaggeration_default: 1.0,
      z_exaggeration_limits: { min: 0.0, max: 5.0 },
      overlays: { paths: [], waypoints: [], sites: [] },
    }
  }, [data, effectiveMaxPoints, maxPoints, roi])

  const metadata = useMemo(() => {
    if (!data) return null
    return {
      datasetRequested: data.datasetRequested,
      datasetUsed: data.datasetUsed,
      isFallback: data.isFallback,
      fallbackReason: data.fallbackReason,
    }
  }, [data])

  return { terrainData, metadata, loading, error }
}
