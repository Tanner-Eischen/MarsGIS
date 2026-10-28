"""Package verified scientific exports for an API-independent reference viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS, Transformer

from marshab.analysis.verification import verify as verify_analysis

DOWNLOADS = (
    ("terrain", "Terrain GeoTIFF (7 bands)", "terrain.tif", "image/tiff"),
    ("suitability", "Suitability GeoTIFF", "suitability.tif", "image/tiff"),
    ("routes", "Mars routes GeoJSON", "routes.geojson", "application/geo+json"),
    ("report", "Analysis report", "report.md", "text/markdown"),
    ("manifest", "Provenance manifest", "manifest.json", "application/json"),
    ("comparison", "Comparison figure", "comparison.png", "image/png"),
)
LAYER_SPECS = (
    (
        "elevation",
        "Elevation",
        "m",
        [[24, 48, 72], [77, 139, 138], [214, 198, 143], [250, 244, 225]],
    ),
    ("slope", "Slope", "degrees", [[22, 24, 54], [129, 47, 106], [235, 123, 81], [252, 244, 177]]),
    ("roughness", "Roughness", "m", [[17, 44, 62], [42, 126, 142], [244, 187, 81]]),
    ("hillshade", "Hillshade", "0–255", [[0, 0, 0], [255, 255, 255]]),
    (
        "suitability",
        "Suitability",
        "dimensionless",
        [[68, 1, 84], [49, 104, 142], [53, 183, 121], [253, 231, 37]],
    ),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_probes(folder: Path) -> np.ndarray:
    """Read exported float32 bands; do not recalculate terrain or suitability."""
    with rasterio.open(folder / "terrain.tif") as src:
        elevation, slope, roughness, shade = src.read([1, 2, 4, 6], masked=True).filled(np.nan)
    with rasterio.open(folder / "suitability.tif") as src:
        score = src.read(1, masked=True).filled(np.nan)
    return np.asarray([elevation, slope, roughness, shade, score], dtype="<f4")


def rgba(values: np.ndarray, low: float, high: float, palette: list) -> np.ndarray:
    """Linearly interpolate RGB palette knots; make invalid cells transparent."""
    valid = np.isfinite(values)
    normalized = np.zeros(values.shape, dtype=float)
    if high > low:
        normalized[valid] = np.clip((values[valid].astype(float) - low) / (high - low), 0, 1)
    else:
        normalized[valid] = 0.5
    knots = np.linspace(0, 1, len(palette))
    colors = np.asarray(palette)
    rgb = np.stack(
        [np.interp(normalized, knots, colors[:, channel]) for channel in range(3)], axis=-1
    )
    return np.concatenate(
        (np.rint(rgb).astype(np.uint8), (valid * 255).astype(np.uint8)[..., None]), axis=-1
    )


def route_pixels(folder: Path, manifest: dict) -> list[dict]:
    crs = CRS.from_wkt(manifest["grid"]["crs_wkt"])
    project = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    inverse = ~rasterio.Affine(*manifest["grid"]["transform"])
    geojson = json.loads((folder / "routes.geojson").read_text(encoding="utf-8"))
    routes = []
    for feature in geojson["features"]:
        coords = np.asarray(feature["geometry"]["coordinates"])
        x, y = project.transform(coords[:, 0], coords[:, 1])
        col, row = inverse * (np.asarray(x), np.asarray(y))
        points = np.column_stack((col, row))
        np.testing.assert_allclose(points - 0.5, np.rint(points - 0.5), atol=1e-6, rtol=0)
        routes.append(
            {
                "rank": feature["properties"]["rank"],
                "points": (np.rint(points - 0.5) + 0.5).tolist(),
            }
        )
    return routes


def build_package(source: Path, output: Path) -> dict:
    """Verify the analysis first, then package exact exports and rendering assets."""
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Package output must differ from analysis input")
    verify_analysis(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_mode") != "observed_dem":
        raise ValueError("Reference package requires an observed DEM manifest")
    output.mkdir(parents=True, exist_ok=True)
    (output / "layers").mkdir(exist_ok=True)
    downloads = []
    for identifier, label, name, media_type in DOWNLOADS:
        shutil.copyfile(source / name, output / name)
        downloads.append(
            {
                "id": identifier,
                "label": label,
                "path": name,
                "sha256": digest(output / name),
                "bytes": (output / name).stat().st_size,
                "media_type": media_type,
            }
        )
    probes = read_probes(output)
    (output / "probe.bin").write_bytes(probes.tobytes(order="C"))
    layers = []
    for band, (identifier, label, unit, palette) in enumerate(LAYER_SPECS):
        values = probes[band]
        valid_values = values[np.isfinite(values)]
        if not valid_values.size:
            raise ValueError(f"Layer has no valid observations: {identifier}")
        low, high = float(valid_values.min()), float(valid_values.max())
        if identifier == "hillshade":
            low, high = 0.0, 255.0
        image = f"layers/{identifier}.png"
        Image.fromarray(rgba(values, low, high, palette)).save(output / image)
        layers.append(
            {
                "id": identifier,
                "label": label,
                "unit": unit,
                "min": low,
                "max": high,
                "palette": palette,
                "image": image,
                "probe_band": band,
            }
        )
    scene = {
        "schema_version": 1,
        "data_mode": "observed_dem",
        "execution_mode": "precomputed_reference_analysis",
        **{
            name: manifest[name]
            for name in ("source", "grid", "configuration", "candidates", "start_pixel")
        },
        "routes": route_pixels(output, manifest),
        "layers": layers,
        "probe": {
            "path": "probe.bin",
            "layout": "band-major",
            "dtype": "float32-le",
            "shape": list(probes.shape),
            "nodata": "NaN",
        },
        "downloads": downloads,
    }
    paths = (
        [item["path"] for item in downloads] + [layer["image"] for layer in layers] + ["probe.bin"]
    )
    scene["artifacts_sha256"] = {name: digest(output / name) for name in paths}
    (output / "scene.json").write_text(
        json.dumps(scene, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    verify_package(output)
    return scene


def verify_package(output: Path) -> None:
    """Check package bytes, metadata, exact probes, image masks and route centers."""
    scene = json.loads((output / "scene.json").read_text(encoding="utf-8"))
    if (
        scene["schema_version"] != 1
        or scene["data_mode"] != "observed_dem"
        or scene["execution_mode"] != "precomputed_reference_analysis"
    ):
        raise ValueError("Unsupported reference scene contract")
    required = (
        {entry[2] for entry in DOWNLOADS}
        | {f"layers/{entry[0]}.png" for entry in LAYER_SPECS}
        | {"probe.bin"}
    )
    if set(scene["artifacts_sha256"]) != required:
        raise ValueError("Incomplete package hash inventory")
    for name, expected in scene["artifacts_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Package checksum mismatch: {name}")
    verify_analysis(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_mode") != scene["data_mode"]:
        raise ValueError("Scene data mode differs from source manifest")
    for key in ("source", "grid", "configuration", "candidates", "start_pixel"):
        if scene[key] != manifest[key]:
            raise ValueError(f"Scene differs from verified manifest: {key}")
    probes = read_probes(output)
    expected_probe = {
        "path": "probe.bin",
        "layout": "band-major",
        "dtype": "float32-le",
        "shape": list(probes.shape),
        "nodata": "NaN",
    }
    if scene["probe"] != expected_probe:
        raise ValueError("Incorrect probe metadata")
    if (output / "probe.bin").stat().st_size != probes.nbytes:
        raise ValueError("Incorrect probe byte length")
    actual = np.frombuffer((output / "probe.bin").read_bytes(), dtype="<f4").reshape(probes.shape)
    np.testing.assert_array_equal(actual, probes)
    if scene["routes"] != route_pixels(output, manifest):
        raise ValueError("Scene route pixels differ from verified geometry")
    if len(scene["layers"]) != len(LAYER_SPECS):
        raise ValueError("Incorrect layer count")
    for band, (layer, spec) in enumerate(zip(scene["layers"], LAYER_SPECS, strict=True)):
        if (
            layer["id"],
            layer["label"],
            layer["unit"],
            layer["palette"],
            layer["probe_band"],
            layer["image"],
        ) != (*spec, band, f"layers/{spec[0]}.png"):
            raise ValueError("Incorrect layer metadata")
        valid = probes[band][np.isfinite(probes[band])]
        expected_range = (
            (0.0, 255.0) if spec[0] == "hillshade" else (float(valid.min()), float(valid.max()))
        )
        if (layer["min"], layer["max"]) != expected_range:
            raise ValueError("Incorrect layer range")
        with Image.open(output / layer["image"]) as image:
            if image.mode != "RGBA" or image.size != (probes.shape[2], probes.shape[1]):
                raise ValueError("Incorrect layer image dimensions or mode")
            np.testing.assert_array_equal(
                np.asarray(image), rgba(probes[band], *expected_range, layer["palette"])
            )
    if len(scene["downloads"]) != len(DOWNLOADS):
        raise ValueError("Incorrect download count")
    for entry, spec in zip(scene["downloads"], DOWNLOADS, strict=True):
        identifier, label, name, media_type = spec
        expected = {
            "id": identifier,
            "label": label,
            "path": name,
            "sha256": digest(output / name),
            "bytes": (output / name).stat().st_size,
            "media_type": media_type,
        }
        if entry != expected:
            raise ValueError("Download metadata differs from packaged bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/output/portfolio_demo"))
    parser.add_argument("--output", type=Path, default=Path("webui/public/portfolio"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_package(args.output)
    else:
        build_package(args.input, args.output)
    print(f"Verified static reference package: {args.output}")


if __name__ == "__main__":
    main()
