import { MapContainer, ImageOverlay, GeoJSON, TileLayer, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useSitesGeoJson, useWaypointsGeoJson, useOverlayImage } from '../hooks/useMapData';
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiUrl } from '../lib/apiBase';
import { Link } from 'react-router-dom'
import EditableRoiRectangle from './EditableRoiRectangle';

// Fix for default marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

interface ElevationSample {
  dataset: string
  dataset_used?: string
  is_fallback?: boolean
  fallback_reason?: string | null
  lat: number
  lon: number
  lon_360: number
  elevation_m: number
  pixel_row: number
  pixel_col: number
  grid_shape: [number, number]
  window_deg: number
  elevation_min_m: number
  elevation_max_m: number
}

type MapDataset = 'hirise' | 'mola_200m' | 'mola'

interface ViewportSample {
  roi: {
    lat_min: number
    lat_max: number
    lon_min: number
    lon_max: number
  }
  width: number
  height: number
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))
const roundTo = (value: number, step: number) => Math.round(value / step) * step
const MAX_OVERLAY_PIXELS = 2_000_000

const normalizeLon360 = (lon: number) => {
  let normalized = lon
  while (normalized < 0) normalized += 360
  while (normalized >= 360) normalized -= 360
  return normalized
}

const normalizeLon180 = (lon: number) => {
  let normalized = lon
  while (normalized < -180) normalized += 360
  while (normalized > 180) normalized -= 360
  return normalized
}

const boundsToViewportSample = (map: L.Map): ViewportSample => {
  const bounds = map.getBounds()
  const size = map.getSize()
  const zoom = map.getZoom()

  const south = clamp(roundTo(bounds.getSouth(), 0.01), -90, 90)
  const north = clamp(roundTo(bounds.getNorth(), 0.01), -90, 90)
  const west = normalizeLon360(roundTo(bounds.getWest(), 0.01))
  const east = normalizeLon360(roundTo(bounds.getEast(), 0.01))
  const lonMin = west
  const lonMax = east > west ? east : clamp(west + 0.01, 0, 360)

  const scale = clamp(1 + (zoom - 5) * 0.4, 1, 4)
  const width = roundTo(clamp(size.x * scale, 800, 4000), 64)
  const height = roundTo(clamp(size.y * scale, 600, 4000), 64)

  return {
    roi: {
      lat_min: south,
      lat_max: north,
      lon_min: lonMin,
      lon_max: lonMax,
    },
    width,
    height,
  }
}

const capSpan = (minValue: number, maxValue: number, maxSpan: number, floor: number, ceil: number) => {
  if (maxValue <= minValue) return [minValue, maxValue] as const
  const span = maxValue - minValue
  if (span <= maxSpan) return [minValue, maxValue] as const
  const center = (minValue + maxValue) / 2
  let nextMin = center - maxSpan / 2
  let nextMax = center + maxSpan / 2
  if (nextMin < floor) {
    const delta = floor - nextMin
    nextMin += delta
    nextMax += delta
  }
  if (nextMax > ceil) {
    const delta = nextMax - ceil
    nextMin -= delta
    nextMax -= delta
  }
  return [clamp(nextMin, floor, ceil), clamp(nextMax, floor, ceil)] as const
}

const capRenderSize = (width: number, height: number) => {
  let w = clamp(width, 800, 1800)
  let h = clamp(height, 600, 1400)
  const pixels = w * h
  if (pixels > MAX_OVERLAY_PIXELS) {
    const scale = Math.sqrt(MAX_OVERLAY_PIXELS / pixels)
    w = roundTo(clamp(w * scale, 800, 1600), 16)
    h = roundTo(clamp(h * scale, 600, 1200), 16)
  }
  return { width: w, height: h }
}

function ViewportTracker({
  onViewportChange,
}: {
  onViewportChange: (sample: ViewportSample) => void
}) {
  const map = useMap()
  const debounceRef = useRef<number | null>(null)

  useEffect(() => {
    const scheduleViewport = (immediate = false) => {
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current)
        debounceRef.current = null
      }
      const run = () => onViewportChange(boundsToViewportSample(map))
      if (immediate) {
        run()
      } else {
        debounceRef.current = window.setTimeout(run, 250)
      }
    }
    const handleViewportEvent = () => scheduleViewport(false)

    scheduleViewport(true)
    map.on('moveend zoomend resize', handleViewportEvent)
    return () => {
      map.off('moveend zoomend resize', handleViewportEvent)
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current)
        debounceRef.current = null
      }
    }
  }, [map, onViewportChange])

  return null
}

function ElevationProbe({
  dataset,
  onPending,
  onResult,
  onError,
}: {
  dataset: string
  onPending: () => void
  onResult: (sample: ElevationSample) => void
  onError: (message: string) => void
}) {
  useMapEvents({
    click: async (event) => {
      const lat = event.latlng.lat
      const lon = event.latlng.lng
      const params = new URLSearchParams({
        dataset,
        lat: String(lat),
        lon: String(lon),
      })
      onPending()
      try {
        const response = await fetch(apiUrl(`/visualization/elevation-at?${params.toString()}`))
        if (!response.ok) {
          const detail = await response.text()
          throw new Error(`Elevation sample failed: ${response.status} ${detail}`)
        }
        const data: ElevationSample = await response.json()
        onResult(data)
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown elevation sampling error'
        onError(message)
      }
    },
  })

  return null
}

function CoverageTracker({
  dataset,
  onCoverageChange,
  onLoadingChange,
}: {
  dataset: string
  onCoverageChange: (coverage: { datasetUsed: string; isFallback: boolean; fallbackReason?: string | null }) => void
  onLoadingChange?: (loading: boolean) => void
}) {
  const map = useMap()

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    const fetchCoverage = async () => {
      onLoadingChange?.(true)
      const bounds = map.getBounds()
      const bbox = [
        bounds.getSouth(),
        bounds.getNorth(),
        normalizeLon180(bounds.getWest()),
        normalizeLon180(bounds.getEast()),
      ]
      const params = new URLSearchParams({
        dataset,
        bbox: bbox.map((v) => v.toFixed(4)).join(','),
      })
      try {
        const response = await fetch(apiUrl(`/visualization/dataset-coverage?${params.toString()}`), {
          signal: controller.signal,
        })
        if (!response.ok) {
          if (response.status === 503) {
            try {
              const data = await response.json()
              if (!cancelled) {
                onCoverageChange({
                  datasetUsed: data.dataset_used || dataset,
                  isFallback: true,
                  fallbackReason: data.fallback_reason || data.error || 'real_dem_unavailable',
                })
              }
            } catch {
              // ignore
            }
          }
          return
        }
        const data = await response.json()
        if (cancelled) return
        onCoverageChange({
          datasetUsed: data.dataset_used || dataset,
          isFallback: !data.available,
          fallbackReason: data.fallback_reason,
        })
      } catch {
        // Ignore coverage errors; map still renders.
      } finally {
        onLoadingChange?.(false)
      }
    }

    if (dataset === 'hirise') {
      fetchCoverage()
      map.on('moveend zoomend', fetchCoverage)
    } else {
      onCoverageChange({ datasetUsed: dataset, isFallback: false })
      onLoadingChange?.(false)
    }

    return () => {
      cancelled = true
      controller.abort()
      map.off('moveend zoomend', fetchCoverage)
    }
  }, [dataset, map, onCoverageChange])

  return null
}

function FitBounds({ bounds, fitKey }) {
  const map = useMap();
  const lastFitKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (bounds && bounds.isValid() && lastFitKeyRef.current !== fitKey) {
      try {
        map.fitBounds(bounds, { padding: [20, 20], maxZoom: 14, animate: true });
        lastFitKeyRef.current = fitKey;
      } catch (e) {
        console.error('Error fitting bounds:', e);
      }
    }
  }, [map, bounds, fitKey]);

  return null;
}

function MapResizeHandler() {
  const map = useMap();
  useEffect(() => {
    const el = map.getContainer();
    const observer = new ResizeObserver(() => {
      map.invalidateSize({ animate: false });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [map]);
  return null;
}

const siteStyle = (isSelected) => ({
  radius: isSelected ? 12 : 8,
  fillColor: isSelected ? '#ffff00' : '#00ff00',
  color: isSelected ? '#ff0000' : '#ffffff',
  weight: isSelected ? 3 : 2,
  opacity: 1,
  fillOpacity: 0.8,
});

const waypointStyle = {
  radius: 3,
  fillColor: '#ff0000',
  color: '#ffffff',
  weight: 1.5,
  opacity: 1,
  fillOpacity: 0.8,
};

const pointToLayer = (feature, latlng, selectedSiteId) => {
  const isWaypoint = !!feature.properties.waypoint_id;
  if (isWaypoint) {
    return L.circleMarker(latlng, waypointStyle);
  }

  const siteId = feature.properties?.site_id;
  const isSelected = selectedSiteId === siteId;
  return L.circleMarker(latlng, siteStyle(isSelected));
};

const onEachFeature = (feature, layer, onSiteSelect) => {
  const props = feature.properties;
  if (props && props.site_id) {
    const popupContent = `
      <div>
        <strong>Site ${props.site_id}</strong><br/>
        Rank: ${props.rank}<br/>
        Area: ${props.area_km2.toFixed(2)} km2<br/>
        Suitability: ${(props.suitability_score * 100).toFixed(1)}%<br/>
        Slope: ${props.mean_slope_deg.toFixed(2)} deg<br/>
        Elevation: ${props.mean_elevation_m.toFixed(0)} m
      </div>
    `;
    layer.bindPopup(popupContent);
    if (onSiteSelect) {
      layer.on('click', () => onSiteSelect(props.site_id));
    }
  }
};

function SitesLayer({ showSites, onSiteSelect, selectedSiteId }) {
  const sitesGeoJson = useSitesGeoJson(showSites);
  if (!sitesGeoJson) return null;

  return (
    <GeoJSON
      data={sitesGeoJson}
      pointToLayer={(f, l) => pointToLayer(f, l, selectedSiteId)}
      onEachFeature={(f, l) => onEachFeature(f, l, onSiteSelect)}
    />
  );
}

function WaypointsLayer({ showWaypoints, selectedSiteId, refreshKey }: { showWaypoints: boolean; selectedSiteId?: number; refreshKey?: number }) {
  const waypointsGeoJson = useWaypointsGeoJson(showWaypoints, refreshKey);
  if (!waypointsGeoJson) return null;

  return (
    <GeoJSON
      key={refreshKey}
      data={waypointsGeoJson}
      pointToLayer={(f, l) => pointToLayer(f, l, selectedSiteId)}
      style={(feature) => ({ color: feature?.properties?.line_color || '#ff0000', weight: 2, opacity: 0.8 })}
    />
  );
}

interface TerrainMapProps {
  roi: any
  dataset?: MapDataset | string
  onDatasetChange?: (dataset: MapDataset) => void
  showSites?: boolean
  showWaypoints?: boolean
  waypointsRefreshKey?: number
  relief?: number
  onSiteSelect?: any
  selectedSiteId?: any
  onRoiChange?: (roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }) => void
  overlayType?:
    | 'elevation'
    | 'solar'
    | 'dust'
    | 'hillshade'
    | 'slope'
    | 'aspect'
    | 'roughness'
    | 'tri'
    | 'viewshed'
    | 'comms_risk'
  overlayOptions?: {
    colormap?: string
    relief?: number
    sunAzimuth?: number
    sunAltitude?: number
    width?: number
    height?: number
    buffer?: number
    marsSol?: number
    season?: string
    dustStormPeriod?: string
  }
}

export default function TerrainMap({
  roi,
  dataset = 'hirise',
  onDatasetChange,
  showSites = false,
  showWaypoints = false,
  waypointsRefreshKey,
  relief = 0,
  onSiteSelect,
  selectedSiteId,
  onRoiChange,
  overlayType,
  overlayOptions = {}
}: TerrainMapProps) {
  const activeOverlayType = overlayType || 'elevation';
  const datasetLower = String(dataset || 'hirise').toLowerCase()
  const [probeLoading, setProbeLoading] = useState(false)
  const [probeError, setProbeError] = useState<string | null>(null)
  const [probeSample, setProbeSample] = useState<ElevationSample | null>(null)
  const [viewportSample, setViewportSample] = useState<ViewportSample | null>(null)
  const [coverageStatus, setCoverageStatus] = useState<{ datasetUsed: string; isFallback: boolean; fallbackReason?: string | null } | null>(null)
  const [coverageLoading, setCoverageLoading] = useState(false)
  const [tilePending, setTilePending] = useState({ basemap: 0, overlay: 0 })
  const [tileErrorCount, setTileErrorCount] = useState(0)
  const [lastTileErrorAt, setLastTileErrorAt] = useState<number | null>(null)
  const useLegacyOverlay = import.meta.env.VITE_USE_LEGACY_IMAGE_OVERLAY === 'true'

  const handleViewportChange = useCallback((next: ViewportSample) => {
    setViewportSample((prev) => {
      if (
        prev &&
        Math.abs(prev.roi.lat_min - next.roi.lat_min) < 0.0001 &&
        Math.abs(prev.roi.lat_max - next.roi.lat_max) < 0.0001 &&
        Math.abs(prev.roi.lon_min - next.roi.lon_min) < 0.0001 &&
        Math.abs(prev.roi.lon_max - next.roi.lon_max) < 0.0001 &&
        prev.width === next.width &&
        prev.height === next.height
      ) {
        return prev
      }
      return next
    })
  }, [])

  const targetRoi = roi
    ? {
        lat_min: clamp(Number(roi.lat_min), -90, 90),
        lat_max: clamp(Number(roi.lat_max), -90, 90),
        lon_min: clamp(normalizeLon360(Number(roi.lon_min)), 0, 360),
        lon_max: clamp(normalizeLon360(Number(roi.lon_max)), 0, 360),
      }
    : null

  let renderRoi = viewportSample?.roi ?? targetRoi ?? null
  if (renderRoi && targetRoi) {
    const expandedTarget = {
      lat_min: clamp(targetRoi.lat_min - 0.2, -90, 90),
      lat_max: clamp(targetRoi.lat_max + 0.2, -90, 90),
      lon_min: clamp(targetRoi.lon_min - 0.2, 0, 360),
      lon_max: clamp(targetRoi.lon_max + 0.2, 0, 360),
    }
    const clipped = {
      lat_min: Math.max(renderRoi.lat_min, expandedTarget.lat_min),
      lat_max: Math.min(renderRoi.lat_max, expandedTarget.lat_max),
      lon_min: Math.max(renderRoi.lon_min, expandedTarget.lon_min),
      lon_max: Math.min(renderRoi.lon_max, expandedTarget.lon_max),
    }
    if (clipped.lat_max > clipped.lat_min && clipped.lon_max > clipped.lon_min) {
      renderRoi = clipped
    } else {
      renderRoi = expandedTarget
    }
  }
  if (renderRoi) {
    const isLowRes = datasetLower === 'mola' || datasetLower === 'mola_200m'
    const maxLatSpan = isLowRes ? 2.5 : 0.8
    const maxLonSpan = isLowRes ? 2.5 : 0.8
    const [latMin, latMax] = capSpan(renderRoi.lat_min, renderRoi.lat_max, maxLatSpan, -90, 90)
    const [lonMin, lonMax] = capSpan(renderRoi.lon_min, renderRoi.lon_max, maxLonSpan, 0, 360)
    renderRoi = {
      lat_min: latMin,
      lat_max: latMax,
      lon_min: lonMin,
      lon_max: lonMax,
    }
  }

  const requestedWidth = overlayOptions.width ?? viewportSample?.width ?? 1400
  const requestedHeight = overlayOptions.height ?? viewportSample?.height ?? 900
  const cappedRender = capRenderSize(requestedWidth, requestedHeight)
  const renderWidth = cappedRender.width
  const renderHeight = cappedRender.height

  const overlayImage = useOverlayImage(
    useLegacyOverlay ? renderRoi : null,
    dataset,
    useLegacyOverlay ? activeOverlayType : null,
    {
      colormap: overlayOptions.colormap || 'terrain',
      relief: overlayOptions.relief ?? relief,
      sunAzimuth: overlayOptions.sunAzimuth || 315,
      sunAltitude: overlayOptions.sunAltitude || 45,
      width: renderWidth,
      height: renderHeight,
      buffer: overlayOptions.buffer ?? 0.05,
      marsSol: overlayOptions.marsSol,
      season: overlayOptions.season,
      dustStormPeriod: overlayOptions.dustStormPeriod,
    }
  )

  const displayImageUrl = overlayImage?.url
  const displayBounds = overlayImage?.bounds
  const displayLoading = overlayImage?.loading
  const displayError = overlayImage?.error

  const centerTuple: [number, number] = roi
    ? [
        (roi.lat_min + roi.lat_max) / 2,
        (() => {
          const lon = (roi.lon_min + roi.lon_max) / 2;
          return lon > 180 ? lon - 360 : lon;
        })(),
      ]
    : [0, 180];

  const fitKey = roi
    ? `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}|${dataset}|${activeOverlayType}`
    : `${dataset}|${activeOverlayType}`;

  // Use global local-raster basemap tiles for robust pan/zoom coverage.
  // Keep analytical overlays separate to avoid stacking duplicate terrain layers.
  const supportedTileOverlays = ['solar', 'dust', 'slope', 'aspect', 'roughness', 'tri']
  const tileOverlayType = supportedTileOverlays.includes(activeOverlayType) ? activeOverlayType : null
  const globalBasemapStyle = (import.meta.env.VITE_GLOBAL_BASEMAP_STYLE || 'blendshade').toLowerCase()
  const basemapTileUrl = apiUrl(
    `/visualization/tiles/basemap/global/{z}/{x}/{y}.png?${new URLSearchParams({
      style: globalBasemapStyle,
      fallback_dataset: 'mola_200m',
      allow_fallback: 'true',
    }).toString()}`
  )

  const roiBounds = roi
    ? L.latLngBounds(
        [roi.lat_min, normalizeLon180(roi.lon_min)],
        [roi.lat_max, normalizeLon180(roi.lon_max)]
      )
    : null

  const totalPendingTiles = tilePending.basemap + tilePending.overlay
  const showActivity = probeLoading || coverageLoading || displayLoading || totalPendingTiles > 0
  const activityLabel = probeLoading
    ? 'Sampling elevation…'
    : coverageLoading
      ? 'Checking dataset coverage…'
      : displayLoading
        ? 'Fetching overlay…'
        : totalPendingTiles > 0
          ? `Loading map tiles… (${totalPendingTiles})`
          : null

  const bumpTilePending = useCallback((kind: 'basemap' | 'overlay', delta: number) => {
    setTilePending((prev) => {
      const next = { ...prev }
      next[kind] = Math.max(0, next[kind] + delta)
      return next
    })
  }, [])

  const recordTileError = useCallback(() => {
    setTileErrorCount((v) => v + 1)
    setLastTileErrorAt(Date.now())
  }, [])

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={centerTuple}
        zoom={6}
        style={{ height: '100%', width: '100%' }}
        className="bg-black"
        scrollWheelZoom={true}
        crs={L.CRS.EPSG4326}
        minZoom={0}
        maxZoom={14}
        maxBounds={[[-90, -180], [90, 180]]}
        worldCopyJump={false}
      >
        <MapResizeHandler />
        <ViewportTracker onViewportChange={handleViewportChange} />
        {onRoiChange && targetRoi && (
          <EditableRoiRectangle roi={targetRoi} onRoiChange={onRoiChange} />
        )}
        <ElevationProbe
          dataset={dataset}
          onPending={() => {
            setProbeLoading(true)
            setProbeError(null)
          }}
          onResult={(sample) => {
            setProbeLoading(false)
            setProbeError(null)
            setProbeSample(sample)
          }}
          onError={(message) => {
            setProbeLoading(false)
            setProbeError(message)
          }}
        />
        {!useLegacyOverlay && (
          <CoverageTracker
            dataset={dataset}
            onCoverageChange={setCoverageStatus}
            onLoadingChange={setCoverageLoading}
          />
        )}

        {!useLegacyOverlay && (
          <>
            <TileLayer
              url={basemapTileUrl}
              tileSize={256}
              keepBuffer={2}
              updateWhenIdle={true}
              minZoom={0}
              maxZoom={14}
              noWrap={true}
              eventHandlers={{
                tileloadstart: () => bumpTilePending('basemap', 1),
                tileload: () => bumpTilePending('basemap', -1),
                tileerror: () => {
                  bumpTilePending('basemap', -1)
                  recordTileError()
                },
              }}
            />
            {tileOverlayType && (
              <TileLayer
                url={apiUrl(
                  `/visualization/tiles/overlay/${tileOverlayType}/${dataset}/{z}/{x}/{y}.png?${new URLSearchParams({
                    colormap: overlayOptions.colormap || 'terrain',
                    relief: String(overlayOptions.relief ?? relief),
                    sun_azimuth: String(overlayOptions.sunAzimuth || 315),
                    sun_altitude: String(overlayOptions.sunAltitude || 45),
                    mars_sol: overlayOptions.marsSol ? String(overlayOptions.marsSol) : '',
                    season: overlayOptions.season || '',
                    dust_storm_period: overlayOptions.dustStormPeriod || '',
                  }).toString()}`
                )}
                opacity={0.72}
                tileSize={256}
                keepBuffer={2}
                updateWhenIdle={true}
                minZoom={0}
                maxZoom={14}
                noWrap={true}
                eventHandlers={{
                  tileloadstart: () => bumpTilePending('overlay', 1),
                  tileload: () => bumpTilePending('overlay', -1),
                  tileerror: () => {
                    bumpTilePending('overlay', -1)
                    recordTileError()
                  },
                }}
              />
            )}
          </>
        )}

        {useLegacyOverlay && displayImageUrl && displayBounds && (
          <>
            <ImageOverlay url={displayImageUrl} bounds={displayBounds} opacity={0.78} />
            <FitBounds bounds={displayBounds} fitKey={fitKey} />
          </>
        )}
        {!useLegacyOverlay && roiBounds && <FitBounds bounds={roiBounds} fitKey={fitKey} />}

        {showSites && (
          <SitesLayer showSites={showSites} onSiteSelect={onSiteSelect} selectedSiteId={selectedSiteId} />
        )}

        {showWaypoints && (
          <WaypointsLayer showWaypoints={showWaypoints} selectedSiteId={selectedSiteId} refreshKey={waypointsRefreshKey} />
        )}
      </MapContainer>

      {useLegacyOverlay && displayLoading && (
        <div className="pointer-events-none absolute top-3 left-3 z-[1000] bg-gray-900/85 border border-cyan-700/60 text-cyan-300 text-xs font-mono px-3 py-2 rounded">
          Loading map data...
        </div>
      )}

      {useLegacyOverlay && displayError && (
        <div className="absolute top-3 right-3 z-[1000] max-w-[420px] bg-red-950/90 border border-red-700/70 text-red-200 text-xs font-mono px-3 py-2 rounded">
          Map data error: {displayError}
        </div>
      )}

      {!useLegacyOverlay && coverageStatus && (
        <div className="pointer-events-none absolute top-3 left-3 z-[1000] bg-gray-900/85 border border-cyan-700/60 text-cyan-300 text-xs font-mono px-3 py-2 rounded">
          <div>Requested: {dataset.toUpperCase()}</div>
          <div>
            Rendered:{' '}
            {coverageStatus.datasetUsed.toUpperCase()}
            {coverageStatus.isFallback ? ' (fallback)' : ''}
          </div>
        </div>
      )}

      {!useLegacyOverlay && (
        <div className="absolute top-3 right-3 z-[1000] flex flex-col gap-2 items-end">
          {onDatasetChange && (
            <div className="bg-gray-900/85 border border-cyan-700/60 text-cyan-200 text-xs font-mono px-3 py-2 rounded">
              <div className="text-[10px] uppercase tracking-wide text-cyan-400/90 mb-1">Dataset</div>
              <select
                value={(datasetLower === 'hirise' || datasetLower === 'mola_200m' || datasetLower === 'mola') ? (datasetLower as MapDataset) : 'mola_200m'}
                onChange={(e) => onDatasetChange(e.target.value as MapDataset)}
                className="pointer-events-auto bg-gray-800 border border-gray-600 text-white px-2 py-1 rounded text-xs focus:border-cyan-500 focus:outline-none"
              >
                <option value="hirise">HiRISE (1m)</option>
                <option value="mola_200m">MOLA 200m (global)</option>
                <option value="mola">MOLA 463m (global)</option>
              </select>
              <div className="mt-1 text-[10px] text-cyan-300/80">
                <Link to="/settings" className="pointer-events-auto underline hover:text-white">
                  Data settings / downloads
                </Link>
              </div>
            </div>
          )}

          {showActivity && activityLabel && (
            <div className="pointer-events-none bg-gray-900/85 border border-gray-700/60 text-gray-100 text-xs font-mono px-3 py-2 rounded flex items-center gap-2">
              <div className="h-3 w-3 rounded-full border-2 border-gray-400 border-t-transparent animate-spin" />
              <div>{activityLabel}</div>
            </div>
          )}

          {tileErrorCount > 0 && lastTileErrorAt && Date.now() - lastTileErrorAt < 12_000 && (
            <div className="bg-red-950/85 border border-red-700/60 text-red-200 text-xs font-mono px-3 py-2 rounded max-w-[360px]">
              <div className="text-[10px] uppercase tracking-wide text-red-300/90 mb-1">Tile load issues</div>
              <div>Some map tiles failed to load.</div>
              <div className="text-[10px] text-red-200/80 mt-1">
                If this persists, open{' '}
                <Link to="/settings" className="pointer-events-auto underline hover:text-white">
                  Settings → Data Management
                </Link>{' '}
                and download MOLA 200m / MOLA for this area.
              </div>
            </div>
          )}
        </div>
      )}

      <div className="pointer-events-none absolute bottom-3 right-3 z-[1000] bg-gray-900/85 border border-cyan-700/60 text-cyan-300 text-xs font-mono px-3 py-2 rounded">
        <div className="font-bold mb-1">DEM Probe (click map)</div>
        {probeLoading && <div>Sampling elevation...</div>}
        {!probeLoading && probeSample && (
          <>
            <div>LAT: {probeSample.lat.toFixed(4)}</div>
            <div>LON: {probeSample.lon_360.toFixed(4)}</div>
            <div>ELEV: {probeSample.elevation_m.toFixed(1)} m</div>
            <div>DATASET: {(probeSample.dataset_used || probeSample.dataset).toUpperCase()}</div>
            {probeSample.is_fallback && <div className="text-amber-300">Fallback active</div>}
          </>
        )}
        {!probeLoading && !probeSample && <div>No sample yet</div>}
        {probeError && <div className="text-red-300 mt-1">Error: {probeError}</div>}
      </div>
    </div>
  );
}
