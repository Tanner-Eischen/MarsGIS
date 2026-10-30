/** Display geometry and units only; all terrain and route analysis stays on the server. */
import type { ExplorerRoute, MarsPoint } from './explorer'

export const MARS_RADIUS_M = 3396000

export function marsDistanceMeters(a: MarsPoint, b: MarsPoint): number {
  const radians = Math.PI / 180
  const dLat = (b.lat - a.lat) * radians
  const dLon = (b.lon - a.lon) * radians
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * radians) * Math.cos(b.lat * radians) * Math.sin(dLon / 2) ** 2
  return 2 * MARS_RADIUS_M * Math.asin(Math.sqrt(Math.max(0, Math.min(1, h))))
}

/** Select the longitude's nearest displayed world copy without changing its location. */
export function longitudeNear(longitude: number, centerLongitude: number): number {
  return longitude + 360 * Math.round((centerLongitude - longitude) / 360)
}

/** Keep short neighboring route segments continuous across the dateline. */
export function routeLatLngs(path: MarsPoint[], centerLongitude: number): [number, number][] {
  if (!path.length) return []
  const longitudes = [path[0].lon]
  for (let index = 1; index < path.length; index++) longitudes.push(longitudeNear(path[index].lon, longitudes[index - 1]))
  const middle = (Math.min(...longitudes) + Math.max(...longitudes)) / 2
  const shift = 360 * Math.round((centerLongitude - middle) / 360)
  return path.map((point, index) => [point.lat, longitudes[index] + shift])
}

export const defaultTerrainCamera = () => ({ eye: { x: 1.45, y: -1.45, z: 1.1 }, up: { x: 0, y: 0, z: 1 } })

/** Plot metric x/y/z in kilometers, leaving elevation measurements unexaggerated. */
export function buildTerrainScene(result: ExplorerRoute, exaggeration: number) {
  const surface = result.surface
  const xs = surface.x_m.flat().filter(Number.isFinite)
  const ys = surface.y_m.flat().filter(Number.isFinite)
  const zs = surface.elevation_m.flat().filter((value): value is number => value !== null && Number.isFinite(value))
  xs.push(...result.path.map(point => point.x_m))
  ys.push(...result.path.map(point => point.y_m))
  zs.push(...result.path.map(point => point.elevation_m))
  if (!xs.length || !ys.length || !zs.length) throw new Error('No measured terrain is available for the 3D view.')
  const span = (values: number[]) => Math.max(...values) - Math.min(...values)
  const xSpan = Math.max(1, span(xs)), ySpan = Math.max(1, span(ys)), zSpan = Math.max(1, span(zs))
  const horizontalSpan = Math.max(xSpan, ySpan)
  const verticalScale = Number.isFinite(exaggeration) ? Math.max(1, Math.min(100, exaggeration)) : 1
  const km = (values: number[][]) => values.map(row => row.map(value => value / 1000))
  const elevationsKm = surface.elevation_m.map(row => row.map(value => value !== null && Number.isFinite(value) ? value / 1000 : null))
  return {
    data: [
      {
        type: 'surface', name: 'Measured terrain', x: km(surface.x_m), y: km(surface.y_m), z: elevationsKm,
        surfacecolor: surface.elevation_m.map(row => [...row]), customdata: surface.elevation_m.map(row => [...row]),
        connectgaps: false, colorscale: [[0, '#261e1d'], [0.25, '#683a2d'], [0.5, '#b16240'], [0.75, '#dea66e'], [1, '#fbefd2']], showscale: true,
        colorbar: { title: { text: 'Elevation (m)' }, len: 0.7 },
        hovertemplate: 'Easting: %{x:.3f} km<br>Northing: %{y:.3f} km<br>Elevation: %{customdata:.1f} m<extra>Measured terrain</extra>',
      },
      {
        type: 'scatter3d', name: 'Calculated route', mode: 'lines',
        x: result.path.map(point => point.x_m / 1000),
        y: result.path.map(point => point.y_m / 1000),
        z: result.path.map(point => point.elevation_m / 1000),
        customdata: result.path.map(point => point.elevation_m),
        connectgaps: false, line: { color: '#ffb46a', width: 6 },
        hovertemplate: 'Easting: %{x:.3f} km<br>Northing: %{y:.3f} km<br>Elevation: %{customdata:.1f} m<extra>Calculated route</extra>',
      },
    ],
    layout: {
      autosize: true, margin: { l: 0, r: 0, t: 15, b: 0 }, paper_bgcolor: '#102329', font: { color: '#edf3ef' },
      showlegend: false, uirevision: result.id,
      scene: {
        xaxis: { title: { text: 'Local easting (km)' } },
        yaxis: { title: { text: 'Local northing (km)' } },
        zaxis: { title: { text: 'Elevation (km)' } },
        aspectmode: 'manual',
        aspectratio: { x: xSpan / horizontalSpan, y: ySpan / horizontalSpan, z: zSpan / horizontalSpan * verticalScale },
        camera: defaultTerrainCamera(),
      },
    },
  }
}
