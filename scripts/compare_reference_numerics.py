"""Compare scientific exports numerically without requiring cross-platform byte identity."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio

BAND_TOLERANCES = {
    "elevation": (0, 0.001),
    "slope": (0, 1e-5),
    "aspect": (0, 1e-5),
    "roughness": (0, 1e-4),
    "tri": (0, 1e-4),
    "hillshade": (0, 0),
    "route_cost": (1e-6, 1e-6),
    "suitability": (0, 2e-7),
}


def compare_array(reference, candidate, label, *, rtol, atol):
    """Require identical validity masks, then apply declared finite-value tolerances."""
    if reference.shape != candidate.shape:
        raise ValueError(f"{label}: shape mismatch")
    valid = np.isfinite(reference)
    if not np.array_equal(valid, np.isfinite(candidate)):
        raise ValueError(f"{label}: finite-data mask mismatch")
    if not valid.any():
        raise ValueError(f"{label}: no finite values")
    if not np.allclose(reference[valid], candidate[valid], rtol=rtol, atol=atol):
        raise ValueError(f"{label}: numerical tolerance exceeded")
    return {
        "valid_values": int(valid.sum()),
        "maximum_absolute_difference": float(np.max(np.abs(reference[valid] - candidate[valid]))),
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "status": "passed",
    }


def read_manifest(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["outputs_sha256"].items():
        if Path(name).name != name:
            raise ValueError(f"Unsafe artifact name: {name}")
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def compare(reference: Path, candidate: Path) -> dict:
    first, second = read_manifest(reference), read_manifest(candidate)
    if first["source"]["sha256"] != second["source"]["sha256"]:
        raise ValueError("Source DEM identity differs")
    if first["configuration"] != second["configuration"]:
        raise ValueError("Scientific configuration differs")
    if first["start_pixel"] != second["start_pixel"]:
        raise ValueError("Start pixel differs")
    for key in ("shape", "resolution_m", "valid_cells"):
        if first["grid"][key] != second["grid"][key]:
            raise ValueError(f"Analysis grid {key} differs")
    rasters = {}
    for name in ("terrain.tif", "suitability.tif"):
        with rasterio.open(reference / name) as left, rasterio.open(candidate / name) as right:
            if left.crs != right.crs:
                raise ValueError(f"{name}: CRS differs")
            if not np.allclose(tuple(left.transform), tuple(right.transform), rtol=0, atol=1e-9):
                raise ValueError(f"{name}: affine differs")
            if left.descriptions != right.descriptions or left.count != right.count:
                raise ValueError(f"{name}: bands differ")
            for index in range(1, left.count + 1):
                label = "suitability" if name == "suitability.tif" else left.descriptions[index - 1]
                rtol, atol = BAND_TOLERANCES[label]
                rasters[f"{name}/{label}"] = compare_array(
                    left.read(index, masked=True).filled(np.nan).astype(float),
                    right.read(index, masked=True).filled(np.nan).astype(float),
                    label,
                    rtol=rtol,
                    atol=atol,
                )
    if len(first["candidates"]) != 3 or len(second["candidates"]) != 3:
        raise ValueError("Candidate count differs from three")
    candidates = []
    for original, fresh in zip(first["candidates"], second["candidates"], strict=True):
        for key in ("rank", "row", "col"):
            if original[key] != fresh[key]:
                raise ValueError(f"Candidate {key} differs")
        metrics = {}
        for key in (
            "score",
            "slope_deg",
            "roughness_m",
            "distance_m",
            "straight_line_m",
            "terrain_cost_m_equivalent",
            "distance_only_route_terrain_cost_m_equivalent",
            "lon",
            "lat",
        ):
            rtol, atol = (1e-6, 1e-6)
            if key == "score":
                rtol, atol = 0, 2e-7
            elif key in {"lon", "lat"}:
                rtol, atol = 0, 1e-10
            metrics[key] = compare_array(
                np.array([original[key]], dtype=float),
                np.array([fresh[key]], dtype=float),
                f"candidate {original['rank']} {key}",
                rtol=rtol,
                atol=atol,
            )
        candidates.append({"rank": original["rank"], "metrics": metrics})
    return {
        "schema_version": 1,
        "status": "passed",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "Numerical agreement of one fixed historical scenario; not cross-platform byte identity",
        "source_sha256": first["source"]["sha256"],
        "reference_manifest_sha256": hashlib.sha256(
            (reference / "manifest.json").read_bytes()
        ).hexdigest(),
        "candidate_manifest_sha256": hashlib.sha256(
            (candidate / "manifest.json").read_bytes()
        ).hexdigest(),
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rasters": rasters,
        "candidates": candidates,
        "limitations": [
            "One fixed CTX scenario does not establish parameter or geographic robustness.",
            "Run the independent geometry/optimality verifier separately for each compared bundle.",
            "PNG/TIFF byte equality is not required or promoted as a cross-platform guarantee.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.report.exists():
        parser.error("Report exists; retain earlier evidence and use a new path")
    result = compare(arguments.reference, arguments.candidate)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    with arguments.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("Numerical raster, candidate and objective agreement passed under declared tolerances")
