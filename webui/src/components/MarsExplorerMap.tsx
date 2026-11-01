import { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import { CircleMarker, MapContainer, Polyline, Rectangle, TileLayer, Tooltip, useMapEvents } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import type { ExplorerMetadata, ExplorerRoute, MapCommand, MarsPoint, PickMode } from '../lib/explorer'
import { normalizeLongitude } from '../lib/explorer'
import { longitudeNear, marsDistanceMeters, MARS_RADIUS_M, routeLatLngs } from '../lib/terrainScene'

/** The geographic tile matrix is 2 x 1 at zoom zero; distances use the Mars sphere. */
export const MARS_CRS = L.extend({}, L.CRS.EPSG4326, {
  code: 'MARS:SPHERE_GEODETIC',
  R: MARS_RADIUS_M,
  distance: (a: L.LatLng, b: L.LatLng) => marsDistanceMeters({ lat: a.lat, lon: a.lng }, { lat: b.lat, lon: b.lng }),
}) as L.CRS & { R: number }

export interface MarsExplorerMapProps {
  metadata: ExplorerMetadata
  start: MarsPoint | null
  end: MarsPoint | null
  selection: PickMode
  result: ExplorerRoute | null
  command: MapCommand | null
  onPick: (point: MarsPoint, which: PickMode) => void
}

export default function MarsExplorerMap(props: MarsExplorerMapProps) {
  const [tileError, setTileError] = useState(false)
  const [retry, setRetry] = useState(0)
  return (
    <div className="mars-map" role="region" aria-label="Interactive Mars terrain map" aria-describedby="mars-map-instructions">
      <p id="mars-map-instructions" className="map-instructions">
        Drag to pan Mars; scroll, pinch or use +/− to zoom. Click to place {props.selection === 'start' ? 'start A' : 'destination B'}. Use the coordinate fields to select points with a keyboard.
      </p>
      <MapContainer
        center={[props.metadata.initial_view.lat, props.metadata.initial_view.lon]}
        zoom={props.metadata.initial_view.zoom}
        crs={MARS_CRS}
        minZoom={0} maxZoom={Math.min(12, props.metadata.max_zoom)}
        maxBounds={[[-90, -540], [90, 540]]} maxBoundsViscosity={0.8}
        scrollWheelZoom dragging keyboard touchZoom doubleClickZoom boxZoom
        worldCopyJump={false}
        style={{ height: '100%', width: '100%' }}
      >
        <MapContents {...props} retry={retry} onTileError={() => setTileError(true)} />
      </MapContainer>
      {tileError && <div className="map-tile-notice" role="status">
        Some terrain tiles could not load. <button type="button" onClick={() => { setTileError(false); setRetry(value => value + 1) }}>Retry map tiles</button>
      </div>}
    </div>
  )
}

function MapContents({ metadata, start, end, selection, result, command, onPick, retry, onTileError }: MarsExplorerMapProps & { retry: number; onTileError: () => void }) {
  const [centerLongitude, setCenterLongitude] = useState(metadata.initial_view.lon)
  const handledCommand = useRef<number | null>(null)
  const map = useMapEvents({
    click: event => onPick({ lat: Math.max(-90, Math.min(90, event.latlng.lat)), lon: normalizeLongitude(event.latlng.lng) }, selection),
    moveend: event => setCenterLongitude(event.target.getCenter().lng),
  })
  useEffect(() => {
    const resize = () => { map.invalidateSize({ pan: false }) }
    resize()
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize)
    observer?.observe(map.getContainer())
    window.addEventListener('resize', resize)
    return () => { observer?.disconnect(); window.removeEventListener('resize', resize) }
  }, [map])
  useEffect(() => {
    if (!command || handledCommand.current === command.id) return
    handledCommand.current = command.id
    if (command.kind === 'global') map.fitBounds([[-90, -180], [90, 180]], { padding: [12, 12] })
    else if (command.kind === 'jezero') map.setView([metadata.detail_view.lat, longitudeNear(metadata.detail_view.lon, map.getCenter().lng)], metadata.detail_view.zoom)
    else if (result?.path.length) map.fitBounds(routeLatLngs(result.path, map.getCenter().lng), { padding: [32, 32], maxZoom: 11 })
  }, [command, map, metadata, result])
  const route = result ? routeLatLngs(result.path, centerLongitude) : []
  const marker = (point: MarsPoint | null, which: PickMode): [number, number] | null => {
    if (route.length) return which === 'start' ? route[0] : route[route.length - 1]
    return point ? [point.lat, longitudeNear(point.lon, centerLongitude)] : null
  }
  const a = marker(start, 'start'), b = marker(end, 'end')
  const [west, south, east, north] = metadata.detail.bounds
  const worldShift = 360 * Math.round((centerLongitude - (west + east) / 2) / 360)
  const detailBounds: L.LatLngBoundsExpression = [[south, west + worldShift], [north, east + worldShift]]
  return <>
    <TileLayer key={`mola-${retry}`} url={metadata.tiles.mola} maxNativeZoom={4} maxZoom={12} minZoom={0} noWrap={false} keepBuffer={1} updateWhenIdle attribution={metadata.global.label} eventHandlers={{ tileerror: onTileError }} />
    <TileLayer key={`ctx-${retry}`} url={metadata.tiles.ctx} minZoom={7} maxNativeZoom={11} maxZoom={12} noWrap={false} bounds={[[south, west], [north, east]]} keepBuffer={1} updateWhenIdle attribution={metadata.detail.label} eventHandlers={{ tileerror: onTileError }} />
    <Rectangle bounds={detailBounds} pathOptions={{ color: '#d9bc7e', weight: 1, fill: false, dashArray: '5 5' }} interactive={false} />
    {route.length > 1 && <Polyline positions={route} pathOptions={{ color: '#ffbf76', weight: 4, opacity: 0.95 }} />}
    {a && <CircleMarker center={a} radius={8} pathOptions={{ color: '#fff', fillColor: '#63c6c4', fillOpacity: 1, weight: 2 }} bubblingMouseEvents={false}><Tooltip permanent direction="top">A · start</Tooltip></CircleMarker>}
    {b && <CircleMarker center={b} radius={8} pathOptions={{ color: '#fff', fillColor: '#ffb46a', fillOpacity: 1, weight: 2 }} bubblingMouseEvents={false}><Tooltip permanent direction="top">B · destination</Tooltip></CircleMarker>}
  </>
}
