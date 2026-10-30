"""Independent geometric/numerical contracts for bounded planetary routing."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.transform import from_origin
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from marshab.analysis.planetary_routing import (
    MAX_GRID_SIDE,
    SphericalPathfinder,
    calculate_route,
    load_routing_grid,
    regional_bounds,
    spherical_distance,
    terrain_arrays,
)


@pytest.fixture
def data():
    radius = 3396000.0
    return SimpleNamespace(
        global_array=np.zeros((360, 720)),
        global_provenance={
            "id": "mola16",
            "label": "Synthetic numerical fixture",
            "radius_m": radius,
            "native_resolution_m": 2 * math.pi * radius / 720,
        },
        ctx_provenance={
            "id": "ctx",
            "label": "Synthetic CTX fixture",
            "radius_m": 3396190.0,
            "native_resolution_m": 20.0,
        },
        ctx_bounds=(77, 18, 77, 18),
    )


@pytest.mark.parametrize(
    "points",
    [
        (0, 0, 0, 180),
        (90, 0, 90, 120),
        (-45, 179, 30, -179),
        (89.9, 10, 89.8, 100),
    ],
)
def test_spherical_distances_match_independent_geod(points):
    lat1, lon1, lat2, lon2 = points
    radius = 3396000.0
    expected = Geod(a=radius, b=radius).inv(lon1, lat1, lon2, lat2)[2]
    assert spherical_distance(*points, radius) == pytest.approx(expected, rel=1e-12, abs=1e-6)


def test_global_gradient_on_analytical_geographic_plane(data):
    radius = data.global_provenance["radius_m"]
    phi = np.radians(90 - (np.arange(360) + 0.5) * 0.5)
    lam = np.radians((np.arange(720) + 0.5) * 0.5)
    a, b = 0.02, -0.03
    data.global_array = radius * (a * np.cos(np.radians(30)) * lam[None, :] + b * phi[:, None])
    grid = load_routing_grid(data, {"lat": 25, "lon": 30}, {"lat": 35, "lon": 40})
    slope, _, cost = terrain_arrays(grid, 25)
    expected = np.degrees(
        np.arctan(np.hypot(a * np.cos(np.radians(30)) / np.cos(np.radians(grid.lat)), b))
    )
    np.testing.assert_allclose(slope, expected, atol=1e-11)
    assert np.isfinite(cost).all()


def test_zero_elevation_is_real_and_route_surface_coordinates_align(data):
    result = calculate_route(data, {"lat": 5, "lon": 10}, {"lat": 8, "lon": 15})
    assert all(point["elevation_m"] == 0 for point in result["path"])
    assert all(value == 0 for row in result["surface"]["elevation_m"] for value in row)
    assert result["distance_m"] >= result["straight_line_m"] - 1e-6
    assert result["terrain_cost_m_equivalent"] == pytest.approx(result["distance_m"])
    projection = CRS.from_wkt(result["grid"]["projection_wkt"])
    transform = Transformer.from_crs(projection.geodetic_crs, projection, always_xy=True)
    for point in result["path"]:
        x, y = transform.transform(point["lon"], point["lat"])
        assert point["x_m"] == pytest.approx(x, abs=1e-8)
        assert point["y_m"] == pytest.approx(y, abs=1e-8)
        # This small analysis is displayed without decimation.
        assert result["surface"]["x_m"][point["row"]][point["col"]] == point["x_m"]
        assert result["surface"]["y_m"][point["row"]][point["col"]] == point["y_m"]
    assert result["elevation_profile"][-1]["distance_m"] == result["distance_m"]
    assert result["geojson"]["planetary_coordinates"]["earth_wgs84"] is False


@pytest.mark.parametrize(
    "start,end",
    [
        ({"lat": 1, "lon": 179}, {"lat": 3, "lon": -179}),
        ({"lat": 0, "lon": 0}, {"lat": 0, "lon": 180}),
        ({"lat": 85, "lon": 10}, {"lat": 88, "lon": 70}),
        ({"lat": -89, "lon": -160}, {"lat": 89, "lon": 160}),
    ],
)
def test_dateline_polar_and_antipodal_routes_remain_bounded(data, start, end):
    result = calculate_route(data, start, end)
    assert max(result["grid"]["shape"]) <= MAX_GRID_SIDE
    assert np.prod(result["grid"]["shape"]) <= 65536
    assert len(result["surface"]["elevation_m"]) <= 96
    assert len(result["surface"]["elevation_m"][0]) <= 96
    assert np.isfinite(result["distance_m"])
    assert result["source"]["analysis_resolution_m"] >= result["source"]["native_resolution_m"]
    assert (
        max(
            abs(a["lon"] - b["lon"])
            for a, b in zip(result["path"], result["path"][1:], strict=False)
        )
        < 180
    )


def test_spherical_astar_matches_independent_directed_dijkstra():
    radius = 3396000.0
    lon, lat = np.meshgrid([178, 179, 180, 181, 182], [84, 83, 82, 81])
    costs = np.array(
        [[1, 0.02, 0.02, 0.02, 1], [1, np.inf, 8, 8, 1], [1, 1, 1, 1, 1], [1, 3, 3, 3, 1]]
    )
    rows, cols, weights = [], [], []
    geod = Geod(a=radius, b=radius)
    for r, c in np.ndindex(costs.shape):
        if not np.isfinite(costs[r, c]):
            continue
        for r2, c2 in np.ndindex(costs.shape):
            if max(abs(r - r2), abs(c - c2)) != 1 or not np.isfinite(costs[r2, c2]):
                continue
            if (
                r != r2
                and c != c2
                and not (np.isfinite(costs[r, c2]) and np.isfinite(costs[r2, c]))
            ):
                continue
            rows.append(r * 5 + c)
            cols.append(r2 * 5 + c2)
            weights.append(
                geod.inv(lon[r, c], lat[r, c], lon[r2, c2], lat[r2, c2])[2] * costs[r2, c2]
            )
    graph = csr_matrix((weights, (rows, cols)), shape=(costs.size, costs.size))
    expected = dijkstra(graph, indices=0, directed=True)[-1]
    solver = SphericalPathfinder(costs, lat, lon, radius)
    path = solver.find_path((0, 0), (3, 4))
    actual = sum(solver.distance(a, b) * costs[b] for a, b in zip(path, path[1:], strict=False))
    assert actual == pytest.approx(expected, rel=1e-12)


def test_nodata_obstacles_and_no_path_are_preserved(data):
    data.global_array[173:175, :] = np.nan
    with pytest.raises(ValueError, match="No traversable|endpoint"):
        calculate_route(data, {"lat": 2, "lon": 10}, {"lat": 6, "lon": 15})
    grid = load_routing_grid(data, {"lat": 2, "lon": 10}, {"lat": 6, "lon": 15})
    _, _, cost = terrain_arrays(grid, 25)
    assert np.isinf(cost[~np.isfinite(grid.elevation)]).all()


@pytest.mark.parametrize(
    "start,end",
    [
        ({"lat": 90, "lon": 0}, {"lat": 90, "lon": 120}),
        ({"lat": 2.21, "lon": 10.21}, {"lat": 2.211, "lon": 10.211}),
    ],
)
def test_identical_physical_or_snapped_endpoints_are_rejected(data, start, end):
    with pytest.raises(ValueError, match="same location|same analysis cell"):
        calculate_route(data, start, end)


@pytest.mark.parametrize(
    "point", [{"lat": 91, "lon": 0}, {"lat": 0, "lon": 181}, {"lat": np.nan, "lon": 0}]
)
def test_invalid_geographic_points_are_rejected(point):
    with pytest.raises(ValueError, match="Coordinates"):
        regional_bounds(point, {"lat": 0, "lon": 0})


def test_surface_holes_are_null_and_never_fabricated_zero(data):
    data.global_array[167, 25] = np.nan
    result = calculate_route(data, {"lat": 5, "lon": 10}, {"lat": 10, "lon": 18})
    assert any(value is None for row in result["surface"]["elevation_m"] for value in row)
    assert all(np.isfinite(point["elevation_m"]) for point in result["path"])


def test_ctx_uses_native_20m_for_short_covered_traverse(data, tmp_path):
    radius = 3396190.0
    crs = CRS.from_proj4(f"+proj=eqc +R={radius} +units=m +no_defs +type=crs")
    x0, y0 = radius * np.radians(77.4), radius * np.radians(18.38)
    path = tmp_path / "ctx.tif"
    transform = from_origin(x0 - 2000, y0 + 2000, 20, 20)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(np.zeros((200, 200), dtype="float32"), 1)
    data.ctx_path = path
    data.ctx_bounds = (77.366, 18.346, 77.434, 18.414)
    result = calculate_route(
        data, {"lat": 18.380010, "lon": 77.385212}, {"lat": 18.384727, "lon": 77.411880}
    )
    assert result["source"]["id"] == "ctx"
    assert result["source"]["analysis_resolution_m"] == pytest.approx(20)
    assert result["source"]["radius_m"] == radius
    assert result["grid"]["stride"] == 1
    assert all(point["elevation_m"] == 0 for point in result["path"])
    # A point outside full buffered CTX coverage must use one consistent source.
    outside = calculate_route(data, {"lat": 15, "lon": 75}, {"lat": 19, "lon": 80})
    assert outside["source"]["id"] == "mola16"
