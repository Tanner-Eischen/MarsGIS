"""Bounded observed-DEM routing on a Mars sphere and an aligned 3D surface.

The optimum is over the sampled eight-connected graph in the reported regional
window, not the full planet or a continuous rover hazard map. Sources retain
their own reference sphere and vertical datum; CTX and MOLA are never fused.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from scipy import ndimage

from marshab.processing.pathfinding import AStarPathfinder

MAX_GRID_SIDE = 256
MAX_SURFACE_SIDE = 96
MAX_PATH_CELLS = 8192


def spherical_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius: float) -> float:
    """Great-circle distance in meters, stable at seams, poles and antipodes."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlon = phi2 - phi1, math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
    h = min(1.0, max(0.0, h))
    return 2 * radius * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def regional_bounds(
    start: dict, end: dict, minimum_buffer_deg: float = 0.125
) -> tuple[float, float, float, float]:
    """Shortest unwrapped longitude interval plus 15% total buffer, W/S/E/N."""
    for point in (start, end):
        lat, lon = point["lat"], point["lon"]
        if not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("Coordinates require finite latitude -90..90 and longitude -180..180")
    delta = (end["lon"] - start["lon"] + 180) % 360 - 180
    if math.isclose(delta, -180) and end["lon"] > start["lon"]:
        delta = 180.0
    lon2 = start["lon"] + delta
    west, east = min(start["lon"], lon2), max(start["lon"], lon2)
    south, north = min(start["lat"], end["lat"]), max(start["lat"], end["lat"])
    lon_buffer = max(minimum_buffer_deg, (east - west) * 0.075)
    lat_buffer = max(minimum_buffer_deg, (north - south) * 0.075)
    return (
        west - lon_buffer,
        max(-90.0, south - lat_buffer),
        east + lon_buffer,
        min(90.0, north + lat_buffer),
    )


@dataclass
class RoutingGrid:
    elevation: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    x: np.ndarray
    y: np.ndarray
    source: dict
    bounds: tuple[float, float, float, float]
    stride: int
    projection: str
    geographic: bool


def _metric_projection(lat: np.ndarray, lon: np.ndarray, radius: float, bounds: tuple) -> tuple:
    west, south, east, north = bounds
    geographic = CRS.from_proj4(f"+proj=longlat +R={radius} +no_defs +type=crs")
    projected = CRS.from_proj4(
        f"+proj=aeqd +lat_0={(south + north) / 2} +lon_0={(west + east) / 2} "
        f"+R={radius} +units=m +no_defs +type=crs"
    )
    x, y = Transformer.from_crs(geographic, projected, always_xy=True).transform(lon, lat)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Regional metric projection is not finite for these endpoints")
    return np.asarray(x), np.asarray(y), projected.to_wkt()


def _sample_indices(first: int, stop: int, stride: int) -> np.ndarray:
    indices = np.arange(first, stop, stride, dtype=int)
    if indices.size == 1 and stop - first >= 2:
        indices = np.array([first, stop - 1])
    return indices


def _global_grid(data: Any, bounds: tuple) -> RoutingGrid:
    west, south, east, north = bounds
    height, width = data.global_array.shape
    if height < 2 or width < 2:
        raise ValueError("Global DEM requires at least two native cells per dimension")
    dy, dx = 180 / height, 360 / width
    r0, r1 = max(0, math.floor((90 - north) / dy)), min(height, math.ceil((90 - south) / dy))
    if r1 - r0 < 2:
        r0, r1 = max(0, min(r0, height - 2)), min(height, max(r1, r0 + 2))
    c0, c1 = math.floor(west / dx), math.ceil(east / dx)
    c1 = max(c1, c0 + 2)
    stride = max(1, math.ceil(max(r1 - r0, c1 - c0) / MAX_GRID_SIDE))
    rows, cols = _sample_indices(r0, r1, stride), _sample_indices(c0, c1, stride)
    elevation = np.asarray(data.global_array[np.ix_(rows, cols % width)], dtype=float)
    lon, lat = np.meshgrid((cols + 0.5) * dx, 90 - (rows + 0.5) * dy)
    source = copy.deepcopy(data.global_provenance)
    radius = source["radius_m"]
    x, y, projection = _metric_projection(lat, lon, radius, bounds)
    source["analysis_resolution_m"] = source["native_resolution_m"] * stride
    return RoutingGrid(elevation, lat, lon, x, y, source, bounds, stride, projection, True)


def _ctx_grid(data: Any, bounds: tuple) -> RoutingGrid:
    west, south, east, north = bounds
    with rasterio.open(data.ctx_path) as src:
        from rasterio.warp import transform_bounds

        geographic = CRS.from_user_input(src.crs).geodetic_crs
        projected_bounds = transform_bounds(
            geographic, src.crs, west, south, east, north, densify_pts=21
        )
        window = from_bounds(*projected_bounds, src.transform)
        col, row = max(0, math.floor(window.col_off)), max(0, math.floor(window.row_off))
        stop_col = min(src.width, math.ceil(window.col_off + window.width))
        stop_row = min(src.height, math.ceil(window.row_off + window.height))
        window = Window(col, row, stop_col - col, stop_row - row)
        if window.width < 2 or window.height < 2:
            raise ValueError("Selected CTX window contains fewer than two native cells")
        stride = max(1, math.ceil(max(window.width, window.height) / MAX_GRID_SIDE))
        shape = (
            max(2, math.ceil(window.height / stride)),
            max(2, math.ceil(window.width / stride)),
        )
        sampled = src.read(
            1, window=window, out_shape=shape, masked=True, resampling=Resampling.nearest
        )
        elevation = sampled.astype(float).filled(np.nan)
        transform = src.window_transform(window) * rasterio.Affine.scale(
            window.width / shape[1], window.height / shape[0]
        )
        row_indices, col_indices = np.indices(shape)
        xx = transform.c + (col_indices + 0.5) * transform.a
        yy = transform.f + (row_indices + 0.5) * transform.e
        lon, lat = Transformer.from_crs(src.crs, geographic, always_xy=True).transform(xx, yy)
        source = copy.deepcopy(data.ctx_provenance)
        source["analysis_resolution_m"] = float(max(abs(transform.a), abs(transform.e)))
    x, y, projection = _metric_projection(
        np.asarray(lat), np.asarray(lon), source["radius_m"], bounds
    )
    return RoutingGrid(
        elevation, np.asarray(lat), np.asarray(lon), x, y, source, bounds, stride, projection, False
    )


def load_routing_grid(data: Any, start: dict, end: dict) -> RoutingGrid:
    bounds = regional_bounds(start, end)
    # A short CTX traverse keeps a five-native-cell (100m) minimum buffer,
    # avoiding the global minimum's unnecessary decimation of 20m observations.
    mean_lat = (start["lat"] + end["lat"]) / 2
    detail_buffer = math.degrees(
        100 / (data.ctx_provenance["radius_m"] * max(math.cos(math.radians(mean_lat)), 1e-6))
    )
    detail_bounds = regional_bounds(start, end, detail_buffer)
    west, south, east, north = detail_bounds
    ctx_west, ctx_south, ctx_east, ctx_north = data.ctx_bounds
    if ctx_west <= west < east <= ctx_east and ctx_south <= south < north <= ctx_north:
        return _ctx_grid(data, detail_bounds)
    return _global_grid(data, bounds)


def terrain_arrays(
    grid: RoutingGrid, max_slope_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical finite differences; invalid inputs/stencils remain impassable."""
    z = np.where(np.isfinite(grid.elevation), grid.elevation, np.nan)
    if grid.geographic:
        north, east = np.gradient(z, np.radians(grid.lat[:, 0]), np.radians(grid.lon[0]))
        radius = grid.source["radius_m"]
        north = north / radius
        east = east / (radius * np.cos(np.radians(grid.lat)))
    else:
        zr, zc = np.gradient(z)
        xr, xc = np.gradient(grid.x)
        yr, yc = np.gradient(grid.y)
        determinant = xc * yr - xr * yc
        if np.any(np.abs(determinant) < 1e-8):
            raise ValueError("Degenerate metric analysis grid")
        east = (zc * yr - zr * yc) / determinant
        north = (zr * xc - zc * xr) / determinant
    slope = np.degrees(np.arctan(np.hypot(east, north)))
    slope[~np.isfinite(z)] = np.nan
    roughness = ndimage.generic_filter(z, np.std, size=5, mode="reflect")
    scale = 500.0 if grid.geographic else 20.0
    cost = 1 + (slope / 10) ** 2 + (roughness / scale) ** 2
    cost[~np.isfinite(cost) | (slope > max_slope_deg)] = np.inf
    return slope, roughness, cost


class SphericalPathfinder(AStarPathfinder):
    """A* with exact spherical edge lengths and an admissible spherical bound."""

    def __init__(self, costs: np.ndarray, lat: np.ndarray, lon: np.ndarray, radius: float):
        super().__init__(costs)
        if costs.shape != lat.shape or costs.shape != lon.shape:
            raise ValueError("Coordinate arrays must align with costs")
        self.lat, self.lon, self.radius = lat, lon, radius

    def distance(self, a: tuple, b: tuple) -> float:
        return spherical_distance(self.lat[a], self.lon[a], self.lat[b], self.lon[b], self.radius)

    def heuristic(self, a: tuple, b: tuple) -> float:
        return self.distance(a, b) * self.min_cost

    def get_neighbors(self, pos: tuple) -> list:
        return [
            (row, col, self.distance(pos, (row, col)) * self.cost_map[row, col])
            for row, col, _ in super().get_neighbors(pos)
        ]


def _nearest(grid: RoutingGrid, point: dict) -> tuple[int, int]:
    phi, lam = np.radians(grid.lat), np.radians(grid.lon)
    target_phi, target_lam = math.radians(point["lat"]), math.radians(point["lon"])
    h = (
        np.sin((phi - target_phi) / 2) ** 2
        + np.cos(phi) * math.cos(target_phi) * np.sin((lam - target_lam) / 2) ** 2
    )
    return tuple(map(int, np.unravel_index(np.argmin(h), h.shape)))


def calculate_route(
    data: Any, start: dict, end: dict, max_slope_deg: float = 25.0, identifier: str = "route"
) -> dict:
    """Calculate and export one bounded route and its matching display surface."""
    if not math.isfinite(max_slope_deg) or not 0 < max_slope_deg < 90:
        raise ValueError("Maximum slope must be finite and between zero and 90 degrees")
    regional_bounds(start, end)  # Validate coordinates before physical identity checks.
    if (
        spherical_distance(
            start["lat"], start["lon"], end["lat"], end["lon"], data.global_provenance["radius_m"]
        )
        < 1
    ):
        raise ValueError("Endpoints represent the same location (less than one meter apart)")
    grid = load_routing_grid(data, start, end)
    if grid.elevation.size > MAX_GRID_SIDE**2:
        raise ValueError("Analysis grid exceeds the bounded workload")
    slope, _, cost = terrain_arrays(grid, max_slope_deg)
    first, last = _nearest(grid, start), _nearest(grid, end)
    if first == last:
        raise ValueError("Both points snap to the same analysis cell; select points farther apart")
    if not np.isfinite(cost[first]) or not np.isfinite(cost[last]):
        raise ValueError(
            "A snapped endpoint has missing or impassable terrain; select another location"
        )
    solver = SphericalPathfinder(cost, grid.lat, grid.lon, grid.source["radius_m"])
    path = solver.find_path(first, last)
    if path is None:
        raise ValueError("No traversable route exists in the bounded regional window")
    if len(path) > MAX_PATH_CELLS:
        raise ValueError("Route exceeds the bounded output complexity; select a smaller region")
    steps = np.array([solver.distance(a, b) for a, b in zip(path, path[1:], strict=False)])
    distances = np.r_[0.0, np.cumsum(steps)]
    terrain_cost = float(np.dot(steps, [cost[p] for p in path[1:]]))

    def point(pixel: tuple) -> dict:
        return {
            "lat": float(grid.lat[pixel]),
            "lon": float(grid.lon[pixel]),
            "elevation_m": float(grid.elevation[pixel]),
            "x_m": float(grid.x[pixel]),
            "y_m": float(grid.y[pixel]),
            "row": pixel[0],
            "col": pixel[1],
        }

    coordinates = [point(pixel) for pixel in path]
    step = max(1, math.ceil(max(grid.elevation.shape) / MAX_SURFACE_SIDE))
    rr = np.arange(0, grid.elevation.shape[0], step)
    cc = np.arange(0, grid.elevation.shape[1], step)
    display_z = grid.elevation[np.ix_(rr, cc)]
    z = [[float(value) if np.isfinite(value) else None for value in row] for row in display_z]
    limitations = (
        "Optimum only within this sampled regional eight-connected graph; no corner cutting. "
        "Illustrative terrain penalties are not a rover energy or hazard model. "
        "Source vertical reference is retained; Mars coordinates are not Earth WGS84."
    )
    if grid.geographic:
        warning = (
            "Coarse MOLA global topography (about 3.7 km native cells at the equator), "
            "decimated for this bounded analysis; sub-cell rover hazards are unresolved. "
            "Selected points are snapped to sampled DEM cell centers."
        )
    else:
        warning = "CTX terrain is sampled at the reported analysis spacing; points are snapped to analysis cell centers."
    source = {**grid.source, "limitations": limitations}
    geojson = {
        "type": "FeatureCollection",
        "planetary_coordinates": {
            "body": "Mars",
            "radius_m": grid.source["radius_m"],
            "latitude": "planetocentric degrees",
            "longitude": "east-positive degrees, unwrapped across the antimeridian",
            "elevation": "source vertical datum in meters",
            "earth_wgs84": False,
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": identifier,
                    "source": source["id"],
                    "distance_m": float(distances[-1]),
                    "terrain_cost_m_equivalent": terrain_cost,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p["lon"], p["lat"], p["elevation_m"]] for p in coordinates],
                },
            }
        ],
    }
    return {
        "id": identifier,
        "source": source,
        "requested": {"start": dict(start), "end": dict(end)},
        "snapped": {"start": coordinates[0], "end": coordinates[-1]},
        "path": coordinates,
        "geojson": geojson,
        "distance_m": float(distances[-1]),
        "straight_line_m": solver.distance(first, last),
        "requested_straight_line_m": spherical_distance(
            start["lat"], start["lon"], end["lat"], end["lon"], grid.source["radius_m"]
        ),
        "terrain_cost_m_equivalent": terrain_cost,
        "max_slope_deg": float(max(slope[p] for p in path)),
        "mean_slope_deg": float(np.mean([slope[p] for p in path])),
        "grid": {
            "shape": list(grid.elevation.shape),
            "stride": grid.stride,
            "spacing_m": source["analysis_resolution_m"],
            "spacing_note": (
                "Nominal equatorial spacing; physical east-west spacing varies with latitude. "
                "Derivatives and every route edge use actual geographic coordinates."
                if grid.geographic
                else "Nominal source projected spacing; terrain gradients use the local metric grid Jacobian."
            ),
            "window_bounds_unwrapped": list(grid.bounds),
            "projection_wkt": grid.projection,
            "slope_limit_deg": max_slope_deg,
            "straight_line_basis": "snapped endpoints",
            "cost_model": "1+(slope_deg/10)^2+(roughness_m/scale)^2; scale=500m MOLA,20m CTX",
        },
        "surface": {
            "x_m": grid.x[np.ix_(rr, cc)].tolist(),
            "y_m": grid.y[np.ix_(rr, cc)].tolist(),
            "elevation_m": z,
            "display_step": step,
            "row_indices": rr.tolist(),
            "col_indices": cc.tolist(),
        },
        "elevation_profile": [
            {"distance_m": float(distance), "elevation_m": p["elevation_m"]}
            for distance, p in zip(distances, coordinates, strict=True)
        ],
        "warning": warning,
    }
