import { fromArrayBuffer } from 'geotiff'

type DecodeRequest = {
  requestId: number
  buffer: ArrayBuffer
}

type DecodeSuccess = {
  requestId: number
  ok: true
  width: number
  height: number
  nodata: number | null
  bbox: {
    left: number
    bottom: number
    right: number
    top: number
  }
  dataBuffer: ArrayBuffer
}

type DecodeFailure = {
  requestId: number
  ok: false
  error: string
}

self.onmessage = async (event: MessageEvent<DecodeRequest>) => {
  const { requestId, buffer } = event.data
  try {
    const tiff = await fromArrayBuffer(buffer)
    const image = await tiff.getImage()
    const width = image.getWidth()
    const height = image.getHeight()
    const [left, bottom, right, top] = image.getBoundingBox()
    const noDataRaw = image.getGDALNoData()
    const nodata = noDataRaw == null ? null : Number(noDataRaw)
    const rasters = await image.readRasters({ samples: [0] })
    const firstBand = rasters[0]
    const floatData = firstBand instanceof Float32Array ? firstBand : new Float32Array(firstBand)

    const transferable = floatData.buffer.slice(0) as ArrayBuffer
    const payload: DecodeSuccess = {
      requestId,
      ok: true,
      width,
      height,
      nodata: Number.isFinite(nodata ?? NaN) ? (nodata as number) : null,
      bbox: { left, bottom, right, top },
      dataBuffer: transferable,
    }
    ;(self as unknown as Worker).postMessage(payload, [payload.dataBuffer])
  } catch (error) {
    const payload: DecodeFailure = {
      requestId,
      ok: false,
      error: error instanceof Error ? error.message : 'Failed to decode DEM GeoTIFF',
    }
    ;(self as unknown as Worker).postMessage(payload)
  }
}
