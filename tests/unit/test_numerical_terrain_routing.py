"""Numerical contracts: meters, north-up rasters, and destination-cell route costs."""

from itertools import product

import numpy as np
import pytest
import xarray as xr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from marshab.exceptions import NavigationError
from marshab.processing.pathfinding import AStarPathfinder
from marshab.processing.terrain import TerrainAnalyzer


@pytest.mark.parametrize("a,b", [(0.2, 0.3), (-0.7, 0.1), (1.0, 0.0), (0.0, -0.5)])
@pytest.mark.parametrize("spacing", [10.0, (20.0, 7.0)])
def test_plane_gradient_and_north_clockwise_aspect(a, b, spacing):
    """z=a*east+b*north+c has an exact constant gradient, including edges."""
    sy, sx = (spacing, spacing) if np.isscalar(spacing) else spacing
    rows, cols = np.indices((9, 11))
    elevation_m = a * cols * sx - b * rows * sy + 35.0
    analyzer = TerrainAnalyzer(cell_size_m=spacing)
    expected_slope_deg = np.degrees(np.arctan(np.hypot(a, b)))
    expected_aspect_deg = np.degrees(np.arctan2(-a, -b)) % 360
    np.testing.assert_allclose(analyzer.calculate_slope(elevation_m), expected_slope_deg, atol=1e-12)
    np.testing.assert_allclose(analyzer.calculate_aspect(elevation_m), expected_aspect_deg, atol=1e-12)


def test_roughness_and_tri_on_known_plane():
    """For z=column meters: 3x3 std=sqrt(2/3), eight-neighbor TRI=6/8 m."""
    dem = np.tile(np.arange(9, dtype=float), (9, 1))
    analyzer = TerrainAnalyzer(1.0)
    np.testing.assert_allclose(analyzer.calculate_roughness(dem)[1:-1, 1:-1], np.sqrt(2 / 3))
    np.testing.assert_allclose(analyzer.calculate_tri(dem)[1:-1, 1:-1], 0.75)
    # Reflected boundaries: only three of eight neighbors differ by one meter.
    assert analyzer.calculate_tri(dem)[0, 0] == pytest.approx(3 / 8)
    np.testing.assert_allclose(analyzer.calculate_tri(np.ones((5, 5))), 0.0)


@pytest.mark.parametrize("altitude,expected", [(0, 0), (30, 127), (90, 255)])
def test_flat_hillshade_uses_sun_altitude(altitude, expected):
    actual = TerrainAnalyzer(10).calculate_hillshade(np.ones((7, 7)), altitude=altitude)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_tilted_plane_hillshade_matches_normal_sun_dot_product():
    a, b = 0.3, -0.2
    rows, cols = np.indices((7, 9))
    dem = a * cols * 4 - b * rows * 12
    azimuth, altitude = np.radians([63.0, 29.0])
    sun = np.array([
        np.sin(azimuth) * np.cos(altitude),
        np.cos(azimuth) * np.cos(altitude),
        np.sin(altitude),
    ])
    normal = np.array([-a, -b, 1.0])
    expected = int(np.clip(np.dot(normal, sun) / np.linalg.norm(normal), 0, 1) * 255)
    actual = TerrainAnalyzer((12, 4)).calculate_hillshade(dem, azimuth=63, altitude=29)
    np.testing.assert_allclose(actual, expected, atol=1)  # uint8 quantization


def test_invalid_terrain_stays_invalid_and_hillshade_encodes_zero():
    values = np.zeros((9, 9))
    values[4, 4] = -9999
    dem = xr.DataArray(values, dims=("y", "x"), attrs={"nodata": -9999})
    metrics = TerrainAnalyzer(1.0).analyze(dem)
    for field in (metrics.elevation, metrics.slope, metrics.aspect, metrics.roughness, metrics.tri):
        assert np.isnan(field[4, 4])
    assert np.isnan(metrics.slope[4, 3])  # derivative stencil intersects NoData
    assert np.isnan(metrics.roughness[3, 3])
    assert np.isnan(metrics.tri[3, 3])
    assert metrics.hillshade[4, 4] == 0
    assert TerrainAnalyzer(1).calculate_aspect(np.zeros((3, 3)))[1, 1] == -1


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, (1, 0), (1, 2, 3)])
def test_invalid_terrain_spacing_rejected(spacing):
    with pytest.raises(ValueError, match="spacing"):
        TerrainAnalyzer(spacing)


def _reference_cost(costs, start, goal, cell_size_m):
    """Independent directed graph solved by SciPy Dijkstra, without A* helpers."""
    height, width = costs.shape
    sources, targets, edges = [], [], []
    for row, col in product(range(height), range(width)):
        if not np.isfinite(costs[row, col]):
            continue
        for other_row in range(max(0, row - 1), min(height, row + 2)):
            for other_col in range(max(0, col - 1), min(width, col + 2)):
                if (row, col) == (other_row, other_col) or not np.isfinite(costs[other_row, other_col]):
                    continue
                if row != other_row and col != other_col:
                    if not np.isfinite(costs[row, other_col]) or not np.isfinite(costs[other_row, col]):
                        continue
                sources.append(row * width + col)
                targets.append(other_row * width + other_col)
                edges.append(np.hypot(other_row - row, other_col - col) * cell_size_m * costs[other_row, other_col])
    graph = csr_matrix((edges, (sources, targets)), shape=(costs.size, costs.size))
    return dijkstra(graph, directed=True, indices=start[0] * width + start[1])[goal[0] * width + goal[1]]


@pytest.mark.parametrize("seed", range(8))
def test_astar_objective_equals_independent_shortest_path(seed):
    """Costs below one require a scaled heuristic; barriers forbid corner cuts."""
    rng = np.random.default_rng(seed)
    costs = rng.uniform(0.02, 2.0, (7, 8))
    costs[rng.random(costs.shape) < 0.2] = np.inf
    costs[0, 0], costs[-1, -1] = 1, 1
    start, goal, cell_size_m = (0, 0), (6, 7), 3.0
    expected = _reference_cost(costs, start, goal, cell_size_m)
    path = AStarPathfinder(costs, cell_size_m).find_path(start, goal)
    if np.isinf(expected):
        assert path is None
        return
    assert path is not None
    assert path[0] == start and path[-1] == goal
    actual = 0.0
    for source, target in zip(path, path[1:], strict=False):
        dr, dc = target[0] - source[0], target[1] - source[1]
        assert max(abs(dr), abs(dc)) == 1
        assert np.isfinite(costs[target])
        if dr and dc:
            assert np.isfinite(costs[source[0], target[1]])
            assert np.isfinite(costs[target[0], source[1]])
        actual += np.hypot(dr, dc) * cell_size_m * costs[target]
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_diagonal_cannot_cross_touching_obstacles():
    assert AStarPathfinder(np.array([[1, np.inf], [np.nan, 1]])).find_path((0, 0), (1, 1)) is None


def test_nan_endpoint_is_impassable():
    with pytest.raises(NavigationError, match="impassable"):
        AStarPathfinder(np.array([[np.nan, 1], [1, 1]])).find_path((0, 0), (1, 1))


@pytest.mark.parametrize("bad_cost", [-1, -np.inf])
def test_negative_costs_rejected(bad_cost):
    with pytest.raises(ValueError, match="nonnegative"):
        AStarPathfinder(np.array([[1, bad_cost]]))


@pytest.mark.parametrize("cell_size", [0, -1, np.nan, np.inf])
def test_invalid_route_cell_size_rejected(cell_size):
    with pytest.raises(ValueError, match="Cell size"):
        AStarPathfinder(np.ones((2, 2)), cell_size)


def test_zero_costs_use_dijkstra_heuristic_and_exact_diagonals():
    pathfinder = AStarPathfinder(np.zeros((3, 3)))
    assert pathfinder.heuristic((0, 0), (2, 2)) == 0
    assert pathfinder.find_path((0, 0), (2, 2)) is not None
    neighbors = AStarPathfinder(np.ones((3, 3)), 5).get_neighbors((1, 1))
    assert next(cost for row, col, cost in neighbors if (row, col) == (0, 0)) == pytest.approx(5 * np.sqrt(2), rel=1e-15)
