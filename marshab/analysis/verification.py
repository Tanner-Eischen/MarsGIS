"""Independently verify archived rasters, routes and full directed-grid optimality."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, message: str, *, rtol=0, atol=0) -> None:
    require(bool(np.allclose(actual, expected, rtol=rtol, atol=atol)), message)


def reconstruct_graph(cost: np.ndarray, passable: np.ndarray, spacing: float) -> csr_matrix:
    """Construct destination-weighted edges directly, without application graph helpers."""
    require(cost.ndim == 2 and cost.shape == passable.shape, "Graph cost/mask shape mismatch")
    require(np.isfinite(spacing) and spacing > 0, "Graph spacing must be finite and positive")
    require(bool(passable.any()), "Graph has no passable cells")
    require(
        bool(np.isfinite(cost[passable]).all() and (cost[passable] >= 0).all()),
        "Graph costs must be finite and nonnegative on passable cells",
    )
    height, width = cost.shape
    rows, columns = np.nonzero(passable)
    sources, targets, weights = [], [], []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        next_rows, next_columns = rows + dr, columns + dc
        inside = (
            (next_rows >= 0) & (next_rows < height) & (next_columns >= 0) & (next_columns < width)
        )
        r, c = rows[inside], columns[inside]
        nr, nc = next_rows[inside], next_columns[inside]
        permitted = passable[nr, nc]
        if dr and dc:
            permitted &= passable[r + dr, c] & passable[r, c + dc]
        r, c, nr, nc = r[permitted], c[permitted], nr[permitted], nc[permitted]
        sources.append(r * width + c)
        targets.append(nr * width + nc)
        weights.append(cost[nr, nc].astype(float) * spacing * np.hypot(dr, dc))
    return csr_matrix(
        (np.concatenate(weights), (np.concatenate(sources), np.concatenate(targets))),
        shape=(cost.size, cost.size),
    )


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    require(
        {"terrain.tif", "suitability.tif", "routes.geojson"} <= manifest["outputs_sha256"].keys(),
        "Manifest must identify terrain, suitability and route checksums",
    )
    for name, expected in manifest["outputs_sha256"].items():
        require(Path(name).name == name, f"Unsafe output checksum path: {name}")
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Output checksum mismatch: {name}")
    grid = manifest["grid"]
    try:
        declared_crs = CRS.from_wkt(grid["crs_wkt"])
        source_crs = CRS.from_wkt(manifest["source"]["crs_wkt"])
    except (CRSError, KeyError, TypeError) as error:
        raise ValueError("Invalid or missing declared grid/source CRS") from error
    require(declared_crs.is_projected, "Grid CRS must be projected")
    close(
        declared_crs.ellipsoid.semi_major_metre,
        source_crs.ellipsoid.semi_major_metre,
        "Grid/source CRS reference sphere mismatch",
        atol=1e-6,
    )
    require(
        all(axis.unit_name == "metre" for axis in declared_crs.axis_info),
        "Grid CRS axes must use metres",
    )
    with rasterio.open(output / "suitability.tif") as src:
        assert [src.height, src.width] == grid["shape"]
        require(src.crs == rasterio.crs.CRS.from_wkt(grid["crs_wkt"]), "Suitability CRS mismatch")
        np.testing.assert_allclose(tuple(src.transform)[:6], grid["transform"], atol=1e-9, rtol=0)
        scores = src.read(1, masked=True).filled(np.nan)
        transform, crs = src.transform, CRS.from_user_input(src.crs)
        np.testing.assert_allclose(
            [transform.a, -transform.e], grid["resolution_m"], atol=1e-10, rtol=0
        )
        assert transform.b == 0 and transform.d == 0
        assert src.nodata == -9999
        assert np.nanmin(scores) >= 0 and np.nanmax(scores) <= 1
    with rasterio.open(output / "terrain.tif") as src:
        require(src.crs is not None, "Terrain CRS is missing")
        require(CRS.from_user_input(src.crs) == crs, "Terrain CRS mismatch")
        require(src.transform == transform, "Terrain/suitability affine mismatch")
        assert [src.height, src.width] == grid["shape"]
        assert src.descriptions == (
            "elevation",
            "slope",
            "aspect",
            "roughness",
            "tri",
            "hillshade",
            "route_cost",
        )
        elevation, slope, aspect, roughness, tri, shade, cost = src.read(masked=True).filled(np.nan)
    constraints = manifest["configuration"]["constraints"]
    max_slope, max_rough = constraints["max_slope_deg"], constraints["max_roughness"]
    require(
        bool(np.isfinite([max_slope, max_rough]).all() and max_slope > 0 and max_rough >= 0),
        "Terrain thresholds must be finite; slope positive and roughness nonnegative",
    )
    configuration = manifest["configuration"]
    require(
        configuration["connectivity"] == 8
        and configuration["corner_cutting"] is False
        and configuration["smoothing"] is False,
        "Unsupported graph convention: requires eight neighbors, no corner cutting or smoothing",
    )
    expected_cost = (
        np.exp(2 * slope.astype(float) / max_slope)
        + 5 * (roughness.astype(float) / (max_rough + 1e-6)) ** 2
    )
    passable = np.isfinite(scores) & (slope <= max_slope) & (roughness <= max_rough)
    require(
        np.array_equal(np.isfinite(cost), passable), "Serialized route-cost validity mask mismatch"
    )
    require(bool((cost[passable] >= 0).all()), "Serialized route costs must be nonnegative")
    close(
        cost[passable], expected_cost[passable], "Serialized route-cost formula mismatch", rtol=1e-6
    )
    assert np.isfinite(elevation).sum() == grid["valid_cells"]
    for name, weight in manifest["configuration"]["weights"].items():
        assert weight >= 0 and name in {"slope", "roughness"}
    valid = np.isfinite(slope) & np.isfinite(roughness)
    expected_score = np.zeros_like(scores)
    for name, values in (("slope", slope), ("roughness", roughness)):
        low, high = values[valid].min(), values[valid].max()
        normalized = (high - values) / (high - low) if high > low else np.full_like(values, 0.5)
        expected_score += normalized * manifest["configuration"]["weights"][name]
    np.testing.assert_allclose(scores[valid], expected_score[valid], atol=2e-7, rtol=0)
    np.testing.assert_array_equal(np.isfinite(scores), valid)
    routes = json.loads((output / "routes.geojson").read_text(encoding="utf-8"))
    require("Mars" in routes.get("coordinate_system", ""), "Route coordinates must declare Mars")
    require(
        len(routes["features"]) == len(manifest["candidates"]) == 3,
        "Reference must contain exactly three candidate routes",
    )
    start = np.asarray(manifest["start_pixel"])
    require(
        start.shape == (2,)
        and np.issubdtype(start.dtype, np.integer)
        and bool((start >= 0).all())
        and bool((start < scores.shape).all()),
        "Invalid start pixel",
    )
    require(bool(passable[tuple(start)]), "Start pixel is not passable")
    graph = reconstruct_graph(cost, passable, grid["resolution_m"])
    distances = dijkstra(graph, directed=True, indices=int(start[0] * scores.shape[1] + start[1]))
    project = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    route_results = []
    for index, (feature, candidate) in enumerate(
        zip(routes["features"], manifest["candidates"], strict=True), start=1
    ):
        prefix = f"Route {index}"
        require(isinstance(feature, dict), f"{prefix}: malformed feature")
        require(feature["properties"] == candidate, f"{prefix}: properties differ from manifest")
        geometry = feature.get("geometry")
        require(isinstance(geometry, dict), f"{prefix}: missing or malformed geometry")
        require(geometry.get("type") == "LineString", f"{prefix}: geometry must be LineString")
        try:
            coords = np.asarray(geometry.get("coordinates"), dtype=float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{prefix}: malformed coordinates") from error
        require(
            coords.ndim == 2
            and coords.shape[1] == 2
            and len(coords) >= 2
            and bool(np.isfinite(coords).all()),
            f"{prefix}: coordinates must be finite longitude/latitude pairs",
        )
        x, y = project.transform(coords[:, 0], coords[:, 1])
        col, row = (~transform) * (np.asarray(x), np.asarray(y))
        centers = np.column_stack((row - 0.5, col - 0.5))
        close(
            centers, np.rint(centers), f"{prefix}: geometry is not at grid-cell centers", atol=1e-6
        )
        pixels = np.rint(centers).astype(int)
        require(
            (pixels >= 0).all()
            and (pixels[:, 0] < scores.shape[0]).all()
            and (pixels[:, 1] < scores.shape[1]).all(),
            f"{prefix}: geometry lies outside grid",
        )
        require(pixels[0].tolist() == manifest["start_pixel"], f"{prefix}: start endpoint mismatch")
        require(
            pixels[-1].tolist() == [candidate["row"], candidate["col"]],
            f"{prefix}: candidate endpoint mismatch",
        )
        require(bool(passable[tuple(pixels.T)].all()), f"{prefix}: geometry crosses blocked cells")
        steps = np.diff(pixels, axis=0)
        require(
            bool((np.abs(steps) <= 1).all() and (np.abs(steps).sum(axis=1) >= 1).all()),
            f"{prefix}: disconnected or repeated grid steps",
        )
        for (r, c), (dr, dc) in zip(pixels[:-1], steps, strict=True):
            if dr and dc:
                require(
                    bool(passable[r + dr, c] and passable[r, c + dc]),
                    f"{prefix}: diagonal cuts a blocked corner",
                )
        lengths = np.linalg.norm(steps, axis=1) * grid["resolution_m"]
        accumulated = np.dot(lengths, expected_cost[tuple(pixels[1:].T)])
        close(candidate["distance_m"], lengths.sum(), f"{prefix}: route length mismatch", atol=1e-7)
        straight = np.linalg.norm(pixels[-1] - pixels[0]) * grid["resolution_m"]
        np.testing.assert_allclose(candidate["straight_line_m"], straight, atol=1e-7, rtol=0)
        close(
            candidate["terrain_cost_m_equivalent"],
            accumulated,
            f"{prefix}: accumulated destination-cell cost mismatch",
            rtol=1e-6,
        )
        serialized_cost = float(np.dot(lengths, cost[tuple(pixels[1:].T)].astype(float)))
        optimum = float(distances[pixels[-1, 0] * scores.shape[1] + pixels[-1, 1]])
        require(np.isfinite(optimum), f"{prefix}: endpoint is unreachable in reconstructed graph")
        close(
            serialized_cost,
            optimum,
            f"{prefix}: route is not shortest in directed graph",
            rtol=1e-6,
        )
        close(
            candidate["terrain_cost_m_equivalent"],
            optimum,
            f"{prefix}: archived objective differs from directed-graph optimum",
            rtol=1e-6,
        )
        assert (
            candidate["terrain_cost_m_equivalent"]
            <= candidate["distance_only_route_terrain_cost_m_equivalent"] + 1e-5
        )
        assert candidate["distance_m"] >= candidate["straight_line_m"] - 1e-6
        np.testing.assert_allclose(candidate["score"], scores[tuple(pixels[-1])], atol=1e-7)
        np.testing.assert_allclose(coords[-1], [candidate["lon"], candidate["lat"]], atol=1e-10)
        route_results.append(
            {
                "rank": candidate["rank"],
                "endpoint_pixel": pixels[-1].tolist(),
                "length_m": float(lengths.sum()),
                "serialized_route_cost_m_equivalent": serialized_cost,
                "dijkstra_cost_m_equivalent": optimum,
                "archived_cost_m_equivalent": candidate["terrain_cost_m_equivalent"],
                "optimality_relative_tolerance": 1e-6,
                "status": "passed",
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_revision": manifest["software"]["git_revision"],
        "reference_dirty_worktree": manifest["software"]["dirty"],
        "graph": {
            "nodes_including_blocked": graph.shape[0],
            "passable_nodes": int(passable.sum()),
            "directed_edges": graph.nnz,
            "edge_definition": "destination-cell serialized cost times metric step length",
            "corner_cutting": False,
            "solver": "scipy.sparse.csgraph.dijkstra; directed=True",
        },
        "routes": route_results,
        "limitations": [
            "Optimality applies only to the serialized discrete graph and declared tolerance.",
            "No physical mobility, habitat safety, browser or deployment validation is implied.",
            "Historical artifacts were checked in place, not regenerated in a fresh locked environment.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/output/portfolio_demo"))
    parser.add_argument(
        "--compare", type=Path, help="Check exact artifact repeatability against another run"
    )
    parser.add_argument(
        "--report", type=Path, help="Write a new JSON verification report; never overwrite"
    )
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report already exists; retain prior evidence and choose a new path")
    report = verify(args.output)
    if args.compare:
        verify(args.compare)
        first = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        second = json.loads((args.compare / "manifest.json").read_text(encoding="utf-8"))
        assert first["outputs_sha256"] == second["outputs_sha256"], "Repeated output hashes differ"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(
        "Verified checksums, georeferencing, suitability, route geometry, costs and directed-graph optimality"
    )


if __name__ == "__main__":
    main()
