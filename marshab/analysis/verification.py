"""Independently verify exported grid, masks, route geometry and accumulated costs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer


def verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["outputs_sha256"].items():
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Output checksum mismatch: {name}")
    grid = manifest["grid"]
    with rasterio.open(output / "suitability.tif") as src:
        assert [src.height, src.width] == grid["shape"]
        assert src.crs == rasterio.crs.CRS.from_wkt(grid["crs_wkt"])
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
        assert src.transform == transform and CRS.from_user_input(src.crs) == crs
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
    expected_cost = (
        np.exp(2 * slope.astype(float) / max_slope)
        + 5 * (roughness.astype(float) / (max_rough + 1e-6)) ** 2
    )
    passable = np.isfinite(scores) & (slope <= max_slope) & (roughness <= max_rough)
    np.testing.assert_array_equal(np.isfinite(cost), passable)
    np.testing.assert_allclose(cost[passable], expected_cost[passable], rtol=1e-6)
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
    assert "Mars" in routes["coordinate_system"] and len(routes["features"]) == 3
    project = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    for feature, candidate in zip(routes["features"], manifest["candidates"], strict=True):
        assert feature["properties"] == candidate
        assert feature["geometry"]["type"] == "LineString"
        coords = np.asarray(feature["geometry"]["coordinates"])
        x, y = project.transform(coords[:, 0], coords[:, 1])
        col, row = (~transform) * (np.asarray(x), np.asarray(y))
        centers = np.column_stack((row - 0.5, col - 0.5))
        np.testing.assert_allclose(centers, np.rint(centers), atol=1e-6, rtol=0)
        pixels = np.rint(centers).astype(int)
        assert (
            (pixels >= 0).all()
            and (pixels[:, 0] < scores.shape[0]).all()
            and (pixels[:, 1] < scores.shape[1]).all()
        )
        assert pixels[0].tolist() == manifest["start_pixel"]
        assert pixels[-1].tolist() == [candidate["row"], candidate["col"]]
        assert passable[tuple(pixels.T)].all()
        steps = np.diff(pixels, axis=0)
        assert (np.abs(steps) <= 1).all() and (np.abs(steps).sum(axis=1) >= 1).all()
        for (r, c), (dr, dc) in zip(pixels[:-1], steps, strict=True):
            if dr and dc:
                assert passable[r + dr, c] and passable[r, c + dc]
        lengths = np.linalg.norm(steps, axis=1) * grid["resolution_m"]
        accumulated = np.dot(lengths, expected_cost[tuple(pixels[1:].T)])
        np.testing.assert_allclose(candidate["distance_m"], lengths.sum(), atol=1e-7)
        straight = np.linalg.norm(pixels[-1] - pixels[0]) * grid["resolution_m"]
        np.testing.assert_allclose(candidate["straight_line_m"], straight, atol=1e-7, rtol=0)
        np.testing.assert_allclose(candidate["terrain_cost_m_equivalent"], accumulated, rtol=1e-6)
        assert (
            candidate["terrain_cost_m_equivalent"]
            <= candidate["distance_only_route_terrain_cost_m_equivalent"] + 1e-5
        )
        assert candidate["distance_m"] >= candidate["straight_line_m"] - 1e-6
        np.testing.assert_allclose(candidate["score"], scores[tuple(pixels[-1])], atol=1e-7)
        np.testing.assert_allclose(coords[-1], [candidate["lon"], candidate["lat"]], atol=1e-10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/output/portfolio_demo"))
    parser.add_argument(
        "--compare", type=Path, help="Check exact artifact repeatability against another run"
    )
    args = parser.parse_args()
    verify(args.output)
    if args.compare:
        verify(args.compare)
        first = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        second = json.loads((args.compare / "manifest.json").read_text(encoding="utf-8"))
        assert first["outputs_sha256"] == second["outputs_sha256"], "Repeated output hashes differ"
    print("Verified checksums, georeferencing, suitability, route connectivity and terrain costs")


if __name__ == "__main__":
    main()
