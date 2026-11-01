"""Ensure numerical agreement cannot hide validity changes or out-of-budget drift."""

import hashlib
import json
import shutil

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts.compare_reference_numerics import compare, compare_array


@pytest.fixture
def comparison_pair(tmp_path):
    """Minimal serialized comparison fixture; graph verification is a separate gate."""
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    descriptions = ("elevation", "slope", "aspect", "roughness", "tri", "hillshade", "route_cost")
    for name, count in (("terrain.tif", 7), ("suitability.tif", 1)):
        with rasterio.open(
            reference / name,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=count,
            dtype="float32",
            transform=from_origin(0, 40, 20, 20),
            crs="+proj=aeqd +lat_0=18.35 +lon_0=77.35 +R=3396190 +units=m +no_defs",
            nodata=-9999,
        ) as dataset:
            dataset.write(np.ones((count, 2, 2), dtype="float32"))
            if count == 7:
                dataset.descriptions = descriptions
    manifest = {
        "source": {"sha256": "a" * 64},
        "configuration": {"fixed_test_configuration": True},
        "start_pixel": [0, 0],
        "grid": {"shape": [2, 2], "resolution_m": 20, "valid_cells": 4},
        "candidates": [
            {
                "rank": rank,
                "row": 1,
                "col": 1,
                "score": 1,
                "slope_deg": 1,
                "roughness_m": 1,
                "distance_m": 20,
                "straight_line_m": 20,
                "terrain_cost_m_equivalent": 20,
                "distance_only_route_terrain_cost_m_equivalent": 20,
                "lon": 77,
                "lat": 18,
            }
            for rank in (1, 2, 3)
        ],
        "outputs_sha256": {
            name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
            for name in ("terrain.tif", "suitability.tif")
        },
    }
    (reference / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copytree(reference, candidate)
    return reference, candidate


def test_numerical_comparison_accepts_precision_drift_with_identical_mask():
    reference = np.array([0, 1, np.nan])
    candidate = np.array([1e-8, 1 + 1e-8, np.nan])
    summary = compare_array(reference, candidate, "score", rtol=0, atol=2e-7)
    assert summary["valid_values"] == 2
    assert summary["maximum_absolute_difference"] == pytest.approx(1e-8)


def test_numerical_comparison_rejects_changed_validity_despite_large_tolerance():
    with pytest.raises(ValueError, match="finite-data mask mismatch"):
        compare_array(np.array([1, np.nan]), np.array([1, 0]), "score", rtol=1, atol=1e6)


def test_numerical_comparison_rejects_out_of_budget_drift():
    with pytest.raises(ValueError, match="numerical tolerance exceeded"):
        compare_array(np.array([1.0]), np.array([1.01]), "cost", rtol=1e-6, atol=1e-6)


def test_serialized_comparison_includes_all_bands_and_candidates(comparison_pair):
    report = compare(*comparison_pair)
    assert report["status"] == "passed"
    assert len(report["rasters"]) == 8 and len(report["candidates"]) == 3


@pytest.mark.parametrize("count", [2, 4])
def test_both_bundles_with_wrong_candidate_count_fail(comparison_pair, count):
    for folder in comparison_pair:
        path = folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["candidates"] = (
            manifest["candidates"][:2]
            if count == 2
            else manifest["candidates"] + [manifest["candidates"][0]]
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Candidate count differs from three"):
        compare(*comparison_pair)


def test_changed_serialized_bytes_are_rejected(comparison_pair):
    _, candidate = comparison_pair
    with (candidate / "terrain.tif").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        compare(*comparison_pair)
