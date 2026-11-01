"""Exercise independent graph semantics and checksum-consistent corrupt exports."""

import hashlib
import json
import tracemalloc

import numpy as np
import pytest
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from scipy.sparse.csgraph import dijkstra

from marshab.analysis.verification import reconstruct_graph, verify


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def refresh_hashes(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["outputs_sha256"]:
        manifest["outputs_sha256"][name] = hashlib.sha256(
            (directory / name).read_bytes()
        ).hexdigest()
    write_json(manifest_path, manifest)


@pytest.fixture
def archive(tmp_path):
    """A hand-calculated 3x3 constant-cost Mars grid with three optimal routes."""
    crs = CRS.from_proj4("+proj=aeqd +lat_0=18.35 +lon_0=77.35 +R=3396190 +units=m +no_defs")
    transform = from_origin(0, 60, 20, 20)
    inverse = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    multiplier = np.exp(2 / 15) + 5 * (2 / (20 + 1e-6)) ** 2
    terrain = np.stack(
        [np.full((3, 3), value) for value in (0, 1, 0, 2, 0, 100, multiplier)]
    ).astype("float32")
    descriptions = ("elevation", "slope", "aspect", "roughness", "tri", "hillshade", "route_cost")
    for name, bands in (("terrain.tif", terrain), ("suitability.tif", np.full((1, 3, 3), 0.5))):
        with rasterio.open(
            tmp_path / name,
            "w",
            driver="GTiff",
            height=3,
            width=3,
            count=len(bands),
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=-9999,
        ) as dataset:
            dataset.write(bands.astype("float32"))
            if name == "terrain.tif":
                dataset.descriptions = descriptions
    candidates, features = [], []
    for rank, pixels in enumerate(
        ([(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 1), (2, 2)]),
        start=1,
    ):
        coordinates = [
            list(inverse.transform(*transform * (col + 0.5, row + 0.5))) for row, col in pixels
        ]
        length = sum(
            np.hypot(b[0] - a[0], b[1] - a[1]) * 20
            for a, b in zip(pixels, pixels[1:], strict=False)
        )
        candidate = {
            "rank": rank,
            "row": pixels[-1][0],
            "col": pixels[-1][1],
            "lon": coordinates[-1][0],
            "lat": coordinates[-1][1],
            "distance_m": length,
            "straight_line_m": length,
            "terrain_cost_m_equivalent": length * multiplier,
            "distance_only_route_terrain_cost_m_equivalent": length * multiplier,
            "score": 0.5,
        }
        candidates.append(candidate)
        features.append(
            {
                "properties": candidate,
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
    write_json(
        tmp_path / "routes.geojson", {"coordinate_system": "Mars degrees", "features": features}
    )
    write_json(
        tmp_path / "manifest.json",
        {
            "outputs_sha256": {
                name: "" for name in ("terrain.tif", "suitability.tif", "routes.geojson")
            },
            "grid": {
                "shape": [3, 3],
                "crs_wkt": crs.to_wkt(),
                "transform": list(transform)[:6],
                "resolution_m": 20,
                "valid_cells": 9,
            },
            "source": {"crs_wkt": crs.to_wkt()},
            "configuration": {
                "constraints": {"max_slope_deg": 15, "max_roughness": 20},
                "weights": {"slope": 0.7, "roughness": 0.3},
                "connectivity": 8,
                "corner_cutting": False,
                "smoothing": False,
            },
            "start_pixel": [0, 0],
            "candidates": candidates,
            "software": {"git_revision": "synthetic-test-fixture", "dirty": False},
        },
    )
    refresh_hashes(tmp_path)
    return tmp_path


def test_graph_uses_directed_destination_cost_and_retains_zero_edges():
    cost = np.array([[9.0, 2.0]])
    graph = reconstruct_graph(cost, np.ones_like(cost, dtype=bool), 20)
    assert graph[0, 1] == 40 and graph[1, 0] == 180
    assert dijkstra(graph, directed=True, indices=0)[1] == 40
    zero_graph = reconstruct_graph(np.zeros((1, 2)), np.ones((1, 2), dtype=bool), 20)
    assert zero_graph.nnz == 2
    assert dijkstra(zero_graph, directed=True, indices=0)[1] == 0


def test_graph_prohibits_diagonal_corner_cutting():
    cost = np.array([[1.0, np.nan], [np.nan, 1.0]])
    graph = reconstruct_graph(cost, np.isfinite(cost), 20)
    assert graph.nnz == 0
    assert np.isinf(dijkstra(graph, directed=True, indices=0)[3])


def test_graph_all_edges_match_explicit_directed_reference():
    costs = np.array([[0, 2, np.nan, 4], [7, 1, 3, 8], [5, np.nan, 6, 9]], dtype=float)
    passable = np.isfinite(costs)
    graph = reconstruct_graph(costs, passable, 20)
    expected = np.zeros(graph.shape)
    expected_edges = 0
    for row, col in np.ndindex(costs.shape):
        if not passable[row, col]:
            continue
        for dr, dc in np.ndindex(3, 3):
            dr, dc = dr - 1, dc - 1
            nr, nc = row + dr, col + dc
            if not (dr or dc) or not (0 <= nr < 3 and 0 <= nc < 4):
                continue
            if not passable[nr, nc]:
                continue
            if dr and dc and not (passable[nr, col] and passable[row, nc]):
                continue
            expected[row * 4 + col, nr * 4 + nc] = costs[nr, nc] * 20 * np.hypot(dr, dc)
            expected_edges += 1
    np.testing.assert_array_equal(graph.toarray(), expected)
    assert graph.nnz == expected_edges and graph.has_sorted_indices


def test_graph_construction_peak_is_bounded_relative_to_csr_storage():
    # NumPy's traced allocations include its array buffers in the pinned profile.
    # This catches the previous 6x CSR peak caused by retaining all triplets twice.
    costs = np.ones((320, 320), dtype=np.float32)
    passable = np.ones_like(costs, dtype=bool)
    tracemalloc.start()
    try:
        graph = reconstruct_graph(costs, passable, 20)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    storage = graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes
    assert peak >= storage, "Allocation tracing must include sparse array buffers"
    assert peak < 2.5 * storage + 5_000_000


@pytest.mark.parametrize("bad_cost", [-1, np.inf, np.nan])
def test_graph_rejects_invalid_passable_cost(bad_cost):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reconstruct_graph(np.array([[bad_cost]]), np.array([[True]]), 20)


def test_complete_archive_reports_three_optimal_routes(archive):
    report = verify(archive)
    assert report["status"] == "passed"
    assert report["graph"]["directed_edges"] == 40
    assert len(report["routes"]) == 3
    for route in report["routes"]:
        assert route["serialized_route_cost_m_equivalent"] == pytest.approx(
            route["dijkstra_cost_m_equivalent"]
        )


def test_changed_artifact_bytes_fail_checksum(archive):
    with (archive / "routes.geojson").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="Output checksum mismatch: routes.geojson"):
        verify(archive)


def test_malformed_crs_has_specific_failure(archive):
    path = archive / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["grid"]["crs_wkt"] = "not a coordinate reference system"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Invalid or missing declared grid/source CRS"):
        verify(archive)


def test_changed_reference_sphere_fails(archive):
    path = archive / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source"]["crs_wkt"] = CRS.from_epsg(4326).to_wkt()
    write_json(path, manifest)
    with pytest.raises(ValueError, match="reference sphere mismatch"):
        verify(archive)


def test_checksum_consistent_wrong_raster_crs_fails(archive):
    with rasterio.open(archive / "terrain.tif", "r+") as dataset:
        dataset.crs = CRS.from_epsg(4326)
    refresh_hashes(archive)
    with pytest.raises(ValueError, match="Terrain CRS mismatch"):
        verify(archive)


def test_checksum_consistent_missing_geometry_fails(archive):
    path = archive / "routes.geojson"
    routes = json.loads(path.read_text(encoding="utf-8"))
    routes["features"][0].pop("geometry")
    write_json(path, routes)
    refresh_hashes(archive)
    with pytest.raises(ValueError, match="missing or malformed geometry"):
        verify(archive)


@pytest.mark.parametrize("bad_cost, message", [(-1, "nonnegative"), (99, "formula mismatch")])
def test_checksum_consistent_corrupt_cost_fails(archive, bad_cost, message):
    with rasterio.open(archive / "terrain.tif", "r+") as dataset:
        cost = dataset.read(7)
        cost[0, 1] = bad_cost
        dataset.write(cost, 7)
    refresh_hashes(archive)
    with pytest.raises(ValueError, match=message):
        verify(archive)


@pytest.mark.parametrize(
    "coordinates, message",
    [
        ([[0, 0], [float("nan"), 0]], "finite longitude/latitude pairs"),
        ([[1], [2]], "finite longitude/latitude pairs"),
    ],
)
def test_checksum_consistent_malformed_geometry_fails(archive, coordinates, message):
    path = archive / "routes.geojson"
    routes = json.loads(path.read_text(encoding="utf-8"))
    routes["features"][0]["geometry"]["coordinates"] = coordinates
    write_json(path, routes)
    refresh_hashes(archive)
    with pytest.raises(ValueError, match=message):
        verify(archive)


def test_disconnected_geometry_fails_after_hash_refresh(archive):
    path = archive / "routes.geojson"
    routes = json.loads(path.read_text(encoding="utf-8"))
    coordinates = routes["features"][0]["geometry"]["coordinates"]
    routes["features"][0]["geometry"]["coordinates"] = [coordinates[0], coordinates[-1]]
    write_json(path, routes)
    refresh_hashes(archive)
    with pytest.raises(ValueError, match="disconnected or repeated grid steps"):
        verify(archive)


def test_wrong_endpoint_fails_after_hash_refresh(archive):
    path = archive / "routes.geojson"
    routes = json.loads(path.read_text(encoding="utf-8"))
    routes["features"][0]["geometry"]["coordinates"][-1] = routes["features"][0]["geometry"][
        "coordinates"
    ][1]
    write_json(path, routes)
    refresh_hashes(archive)
    with pytest.raises(ValueError, match="candidate endpoint mismatch"):
        verify(archive)


def test_self_consistent_suboptimal_detour_fails(archive):
    """Length/cost, manifest and checksums all agree; only shortest-path checking rejects it."""
    path = archive / "routes.geojson"
    routes = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crs = CRS.from_wkt(manifest["grid"]["crs_wkt"])
    inverse = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    transform = from_origin(0, 60, 20, 20)
    detour = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    coordinates = [
        list(inverse.transform(*transform * (col + 0.5, row + 0.5))) for row, col in detour
    ]
    routes["features"][0]["geometry"]["coordinates"] = coordinates
    candidate = manifest["candidates"][0]
    candidate["distance_m"] *= 2
    candidate["terrain_cost_m_equivalent"] *= 2
    candidate["distance_only_route_terrain_cost_m_equivalent"] *= 2
    routes["features"][0]["properties"] = candidate
    write_json(path, routes)
    write_json(manifest_path, manifest)
    refresh_hashes(archive)
    with pytest.raises(ValueError, match="route is not shortest in directed graph"):
        verify(archive)
