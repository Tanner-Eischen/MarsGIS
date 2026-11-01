import { describe, expect, it } from 'vitest'
import type { ExplorerRoute } from './explorer'
import { buildTerrainScene, longitudeNear, marsDistanceMeters, MARS_RADIUS_M, routeLatLngs } from './terrainScene'

const route = {
  id: 'metric-test',
  surface: {
    x_m: [[1000, 2000, 3000], [1000, 2000, 3000]],
    y_m: [[2000, 2000, 2000], [3000, 3000, 3000]],
    elevation_m: [[-2000, -1500, null], [-1800, -1300, -1000]], display_step: 2,
  },
  path: [
    { lat: 10, lon: 20, x_m: 1000, y_m: 2000, elevation_m: -2000 },
    { lat: 10.1, lon: 20.1, x_m: 3000, y_m: 3000, elevation_m: -1000 },
  ],
} as ExplorerRoute

describe('planetary display geometry', () => {
  it('measures the Mars sphere rather than Earth', () => {
    expect(MARS_RADIUS_M).toBe(3396000)
    expect(marsDistanceMeters({ lat: 0, lon: 0 }, { lat: 0, lon: 1 })).toBeCloseTo(MARS_RADIUS_M * Math.PI / 180, 7)
    expect(marsDistanceMeters({ lat: 0, lon: 0 }, { lat: 90, lon: 50 })).toBeCloseTo(MARS_RADIUS_M * Math.PI / 2, 7)
    expect(marsDistanceMeters({ lat: 0, lon: 0 }, { lat: 0, lon: 180 })).toBeCloseTo(MARS_RADIUS_M * Math.PI, 7)
    expect(marsDistanceMeters({ lat: 30, lon: 50 }, { lat: 30, lon: 50 })).toBe(0)
  })
  it('keeps dateline routes short and in the nearest world copy', () => {
    const path = [{ lat: 1, lon: 179 }, { lat: 2, lon: -179.5 }, { lat: 3, lon: -178 }]
    expect(routeLatLngs(path, 180)).toEqual([[1, 179], [2, 180.5], [3, 182]])
    expect(routeLatLngs(path, -180)).toEqual([[1, -181], [2, -179.5], [3, -178]])
    expect(routeLatLngs(path, 540)).toEqual([[1, 539], [2, 540.5], [3, 542]])
    expect(routeLatLngs([{ lat: 1, lon: 179 }, { lat: 2, lon: 180.5 }], 180)).toEqual([[1, 179], [2, 180.5]])
    expect(routeLatLngs([], 0)).toEqual([])
    expect(longitudeNear(-179, 181)).toBe(181)
  })
})

describe('metric 3D display', () => {
  it('uses the same metric coordinates and actual elevations for route and surface', () => {
    const { data } = buildTerrainScene(route, 5)
    expect(data[0].x).toEqual([[1, 2, 3], [1, 2, 3]])
    expect(data[0].y).toEqual([[2, 2, 2], [3, 3, 3]])
    expect(data[0].z).toEqual([[-2, -1.5, null], [-1.8, -1.3, -1]])
    expect(data[1].x).toEqual([1, 3])
    expect(data[1].y).toEqual([2, 3])
    expect(data[1].z).toEqual([-2, -1])
    expect(data[1].customdata).toEqual([-2000, -1000])
    expect(data[1].hovertemplate).toContain(' m<extra>')
    expect(data[0].connectgaps).toBe(false)
    expect(data[1].connectgaps).toBe(false)
  })
  it('changes only visual aspect for exaggeration and retains NoData holes', () => {
    const normal = buildTerrainScene(route, 1), exaggerated = buildTerrainScene(route, 10)
    expect(exaggerated.data).toEqual(normal.data)
    expect(normal.layout.scene.aspectratio).toEqual({ x: 1, y: 0.5, z: 0.5 })
    expect(exaggerated.layout.scene.aspectratio).toEqual({ x: 1, y: 0.5, z: 5 })
    expect(exaggerated.data[0].z[0][2]).toBeNull()
    expect(route.surface.elevation_m[0][2]).toBeNull()
  })
  it('bounds exaggeration and gives flat terrain a finite display aspect', () => {
    const flat = { ...route, surface: { ...route.surface, elevation_m: [[0, 0, null], [0, 0, 0]] }, path: route.path.map(p => ({ ...p, elevation_m: 0 })) }
    expect(buildTerrainScene(flat, 0).layout.scene.aspectratio.z).toBeCloseTo(1 / 2000)
    expect(buildTerrainScene(flat, 1000).layout.scene.aspectratio.z).toBeCloseTo(100 / 2000)
    expect(buildTerrainScene(flat, NaN).layout.scene.aspectratio.z).toBeCloseTo(1 / 2000)
  })
})
