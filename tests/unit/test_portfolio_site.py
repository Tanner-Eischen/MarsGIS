"""Static reference packaging preserves scientific exports, pixels and masks."""

import json

import numpy as np
import pytest
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin

from marshab.analysis.portfolio import LOCAL_CRS
from scripts.build_portfolio_site import build_package, digest, verify_package


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def analysis(tmp_path):
    """Tiny generated analysis meeting the real export verifier, without mocks/network."""
    folder = tmp_path / "analysis"
    folder.mkdir()
    transform = from_origin(0, 60, 20, 20)
    values = np.arange(15, dtype=np.float32).reshape(3, 5) / 10
    values[-1, -1] = np.nan
    rough = values / 4
    valid = np.isfinite(values)
    score = (1 - values / np.nanmax(values)).astype(np.float32)
    cost = np.exp(2 * values.astype(float) / 15) + 5 * (rough.astype(float) / (20 + 1e-6)) ** 2
    elevation = 1000 + values
    shade = np.where(valid, 177, np.nan)
    bands = [elevation, values, values, rough, values, shade, cost]
    metadata = {
        "driver": "GTiff",
        "height": 3,
        "width": 5,
        "count": 7,
        "dtype": "float32",
        "crs": LOCAL_CRS,
        "transform": transform,
        "nodata": -9999,
    }
    with rasterio.open(folder / "terrain.tif", "w", **metadata) as dst:
        for number, band in enumerate(bands, 1):
            dst.write(np.where(np.isfinite(band), band, -9999).astype(np.float32), number)
            dst.set_band_description(
                number,
                ("elevation", "slope", "aspect", "roughness", "tri", "hillshade", "route_cost")[
                    number - 1
                ],
            )
    metadata["count"] = 1
    with rasterio.open(folder / "suitability.tif", "w", **metadata) as dst:
        dst.write(np.where(valid, score, -9999), 1)
    inverse = Transformer.from_crs(LOCAL_CRS, LOCAL_CRS.geodetic_crs, always_xy=True)
    candidates, features = [], []
    for rank in range(1, 4):
        x, y = rasterio.transform.xy(transform, [0] * (rank + 1), list(range(rank + 1)))
        lon, lat = inverse.transform(x, y)
        total_cost = float(cost[0, 1 : rank + 1].sum() * 20)
        candidate = {
            "rank": rank,
            "row": 0,
            "col": rank,
            "lon": lon[-1],
            "lat": lat[-1],
            "score": float(score[0, rank]),
            "slope_deg": float(values[0, rank]),
            "roughness_m": float(rough[0, rank]),
            "distance_m": rank * 20,
            "straight_line_m": rank * 20,
            "terrain_cost_m_equivalent": total_cost,
            "distance_only_route_terrain_cost_m_equivalent": total_cost,
        }
        candidates.append(candidate)
        features.append(
            {
                "type": "Feature",
                "properties": candidate,
                "geometry": {
                    "type": "LineString",
                    "coordinates": list(map(list, zip(lon, lat, strict=True))),
                },
            }
        )
    write_json(
        folder / "routes.geojson",
        {"type": "FeatureCollection", "coordinate_system": "Mars sphere", "features": features},
    )
    (folder / "report.md").write_text("# Generated unit-test fixture\n", encoding="utf-8")
    Image.new("RGB", (5, 3)).save(folder / "comparison.png")
    manifest = {
        "data_mode": "observed_dem",
        "source": {"citation": "Generated unit-test surface, not observational evidence"},
        "grid": {
            "shape": [3, 5],
            "transform": tuple(transform)[:6],
            "crs_wkt": LOCAL_CRS.to_wkt(),
            "resolution_m": 20,
            "valid_cells": 14,
        },
        "configuration": {
            "weights": {"slope": 0.7, "roughness": 0.3},
            "constraints": {"max_slope_deg": 15, "max_roughness": 20},
        },
        "candidates": candidates,
        "start_pixel": [0, 0],
        "outputs_sha256": {
            name: digest(folder / name)
            for name in (
                "terrain.tif",
                "suitability.tif",
                "routes.geojson",
                "report.md",
                "comparison.png",
            )
        },
    }
    write_json(folder / "manifest.json", manifest)
    return folder


def test_package_preserves_full_resolution_probe_and_pixel_centers(analysis, tmp_path):
    output = tmp_path / "site"
    scene = build_package(analysis, output)
    verify_package(output)
    assert scene["probe"]["shape"] == [5, 3, 5]
    probe = np.frombuffer((output / "probe.bin").read_bytes(), dtype="<f4").reshape(5, 3, 5)
    np.testing.assert_allclose(probe[:, 0, 1], [1000.1, 0.1, 0.025, 177, 1 - 0.1 / 1.3], atol=3e-5)
    assert np.isnan(probe[:, -1, -1]).all()
    assert scene["routes"][0]["points"] == [[0.5, 0.5], [1.5, 0.5]]
    for layer in scene["layers"]:
        with Image.open(output / layer["image"]) as image:
            assert image.size == (5, 3)
            assert image.getpixel((4, 2))[3] == 0
            assert image.getpixel((0, 0))[3] == 255
    for download in scene["downloads"]:
        assert (output / download["path"]).read_bytes() == (
            analysis / download["path"]
        ).read_bytes()


def test_rejects_unverified_input_before_creating_package(analysis, tmp_path):
    (analysis / "report.md").write_text("corrupt", encoding="utf-8")
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_package(analysis, output)
    assert not output.exists()


def test_rejects_corrupt_packaged_artifact(analysis, tmp_path):
    output = tmp_path / "site"
    build_package(analysis, output)
    (output / "probe.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Package checksum mismatch"):
        verify_package(output)


def test_probe_values_checked_even_if_hash_is_updated(analysis, tmp_path):
    output = tmp_path / "site"
    scene = build_package(analysis, output)
    probe = np.frombuffer((output / "probe.bin").read_bytes(), dtype="<f4").copy()
    probe[0] += 10
    (output / "probe.bin").write_bytes(probe.tobytes())
    scene["artifacts_sha256"]["probe.bin"] = digest(output / "probe.bin")
    write_json(output / "scene.json", scene)
    with pytest.raises(AssertionError):
        verify_package(output)


def test_route_centers_checked_independently_of_scene(analysis, tmp_path):
    output = tmp_path / "site"
    scene = build_package(analysis, output)
    scene["routes"][0]["points"][0][0] = 1.5
    write_json(output / "scene.json", scene)
    with pytest.raises(ValueError, match="route pixels"):
        verify_package(output)


def test_rejects_synthetic_data_mode(analysis, tmp_path):
    path = analysis / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["data_mode"] = "synthetic"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="observed DEM"):
        build_package(analysis, tmp_path / "site")
