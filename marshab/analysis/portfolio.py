"""Strict measured-DEM example using shared terrain, scoring and routing kernels."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage

from marshab.analysis.export import export_suitability_geotiff
from marshab.models import BoundingBox
from marshab.processing.mcdm import MCDMEvaluator
from marshab.processing.pathfinding import AStarPathfinder
from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface

SOURCE_URL = (
    "https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/"
    "JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif"
)
SOURCE_CITATION = "https://www.usgs.gov/media/images/m20jezerocraterctxdemmosaic20m100jpg"
SOURCE_SHA256 = "16a1aa66c3f48c1bca252795827db7b417876217aff0f5e1aff18027fab2327b"
ROI = BoundingBox(lat_min=18.25, lat_max=18.45, lon_min=77.25, lon_max=77.45)
LOCAL_CRS = CRS.from_proj4("+proj=aeqd +lat_0=18.35 +lon_0=77.35 +R=3396190 +units=m +no_defs")
RESOLUTION_M = 20.0
WEIGHTS = {"slope": 0.7, "roughness": 0.3}
CONSTRAINTS = {"max_slope_deg": 15.0, "max_roughness": 20.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_dem(path: Path) -> Path:
    """Download once and verify pinned bytes every run; never fall back."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        try:
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "MarsHab/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha256(partial) != SOURCE_SHA256:
                raise ValueError("Downloaded DEM checksum differs from the documented source")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("DEM checksum mismatch; refusing unverified input")
    return path


def prepare_grid(path: Path) -> tuple[xr.DataArray, dict]:
    """Reproject the fixed ROI to a 20 m local Mars azimuthal equidistant grid."""
    with rasterio.open(path) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError("Expected a single-band DEM with a machine-readable Mars CRS")
        source_crs = CRS.from_user_input(src.crs)
        if not math.isclose(source_crs.ellipsoid.semi_major_metre, 3396190, abs_tol=1):
            raise ValueError("Expected the documented Mars sphere")
        geographic = source_crs.geodetic_crs
        bounds = (ROI.lon_min, ROI.lat_min, ROI.lon_max, ROI.lat_max)
        sb = transform_bounds(geographic, src.crs, *bounds, densify_pts=21)
        if not (
            src.bounds.left <= sb[0] < sb[2] <= src.bounds.right
            and src.bounds.bottom <= sb[1] < sb[3] <= src.bounds.top
        ):
            raise ValueError("Requested ROI is outside the source DEM")
        left, bottom, right, top = transform_bounds(geographic, LOCAL_CRS, *bounds, densify_pts=21)
        width = math.ceil((right - left) / RESOLUTION_M)
        height = math.ceil((top - bottom) / RESOLUTION_M)
        transform = from_origin(left, top, RESOLUTION_M, RESOLUTION_M)
        grid = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=grid,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=LOCAL_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=1,
        )
        xx, yy = np.meshgrid(
            transform.c + (np.arange(width) + 0.5) * RESOLUTION_M,
            transform.f - (np.arange(height) + 0.5) * RESOLUTION_M,
        )
        lon, lat = Transformer.from_crs(LOCAL_CRS, geographic, always_xy=True).transform(xx, yy)
        grid[
            (lon < ROI.lon_min) | (lon > ROI.lon_max) | (lat < ROI.lat_min) | (lat > ROI.lat_max)
        ] = np.nan
        dem = xr.DataArray(
            grid,
            dims=("y", "x"),
            coords={"x": xx[0], "y": yy[:, 0]},
            attrs={
                "crs": LOCAL_CRS.to_wkt(),
                "transform": tuple(transform)[:6],
                "nodata": np.nan,
                "resolution_m": RESOLUTION_M,
                "elevation_units": "m",
            },
        )
        provenance = {
            "url": SOURCE_URL,
            "citation": SOURCE_CITATION,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "instrument": "NASA MRO CTX stereo, MOLA controlled",
            "crs_wkt": src.crs.to_wkt(),
            "transform": list(src.transform)[:6],
            "shape": [src.height, src.width],
            "nodata": src.nodata,
            "elevation_units": "m above product's MOLA/DeltaGeoid reference",
            "source_pixel_size_m": list(src.res),
        }
    return dem, provenance


def path_statistics(path: list[tuple[int, int]], cost: np.ndarray) -> dict[str, float]:
    """Destination-cell penalties integrated along exact eight-connected steps."""
    lengths = np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1) * RESOLUTION_M
    penalties = np.array([cost[p] for p in path[1:]])
    return {
        "distance_m": float(lengths.sum()),
        "terrain_cost_m_equivalent": float(np.dot(lengths, penalties)),
        "straight_line_m": float(np.linalg.norm(np.subtract(path[-1], path[0])) * RESOLUTION_M),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _software() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", True
    hashes = {
        str(p.relative_to(root)).replace("\\", "/"): sha256(p)
        for folder in (root / "marshab", root / "scripts")
        for p in sorted(folder.rglob("*.py"))
    }
    return {
        "git_revision": revision,
        "dirty": dirty,
        "source_sha256": hashes,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("marshab", "numpy", "scipy", "rasterio", "pyproj", "xarray", "matplotlib")
        },
        "gdal": rasterio.__gdal_version__,
    }


def run_demo(source: Path, output: Path) -> dict:
    """Run the strict real-data example, export artifacts and return the manifest."""
    dem, provenance = prepare_grid(acquire_dem(source))
    metrics = TerrainAnalyzer(RESOLUTION_M).analyze(dem)
    criteria = {"slope": metrics.slope, "roughness": metrics.roughness}
    suitability = MCDMEvaluator.weighted_sum(criteria, WEIGHTS, {k: False for k in criteria})
    cost = generate_cost_surface(metrics.slope, metrics.roughness, **CONSTRAINTS)
    cost[~np.isfinite(suitability)] = np.inf
    labels, count = ndimage.label(np.isfinite(cost))
    if count == 0:
        raise ValueError("No traversable DEM cells")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    connected = labels == sizes.argmax()
    available = np.argwhere(connected)
    start = tuple(map(int, available[np.argmin(np.sum((available - [10, 10]) ** 2, axis=1))]))
    candidates, features = [], []
    selectable = connected.copy()
    rows, cols = np.indices(dem.shape)
    # Avoid selecting boundary cells whose local neighborhoods use reflected padding.
    selectable &= (rows >= 5) & (rows < dem.shape[0] - 5) & (cols >= 5) & (cols < dem.shape[1] - 5)
    selectable &= np.hypot(rows - start[0], cols - start[1]) * RESOLUTION_M >= 1500
    transform = rasterio.Affine(*dem.attrs["transform"])
    to_geo = Transformer.from_crs(LOCAL_CRS, LOCAL_CRS.geodetic_crs, always_xy=True)
    for rank in range(1, 4):
        if not selectable.any():
            raise ValueError("Cannot select three separated candidates")
        target = tuple(
            map(
                int,
                np.unravel_index(np.argmax(np.where(selectable, suitability, -np.inf)), dem.shape),
            )
        )
        path = AStarPathfinder(cost, RESOLUTION_M).find_path(start, target)
        baseline = AStarPathfinder(
            np.where(np.isfinite(cost), 1.0, np.inf), RESOLUTION_M
        ).find_path(start, target)
        if path is None or baseline is None:
            raise ValueError("Candidate is unreachable from the documented start")
        stats = path_statistics(path, cost)
        stats["distance_only_route_terrain_cost_m_equivalent"] = path_statistics(baseline, cost)[
            "terrain_cost_m_equivalent"
        ]
        x, y = rasterio.transform.xy(transform, *target)
        lon, lat = to_geo.transform(x, y)
        candidate = {
            "rank": rank,
            "row": target[0],
            "col": target[1],
            "lon": lon,
            "lat": lat,
            "score": float(suitability[target]),
            "slope_deg": float(metrics.slope[target]),
            "roughness_m": float(metrics.roughness[target]),
            **stats,
        }
        candidates.append(candidate)
        px, py = rasterio.transform.xy(transform, *np.asarray(path).T)
        plon, plat = to_geo.transform(px, py)
        features.append(
            {
                "type": "Feature",
                "properties": candidate,
                "geometry": {
                    "type": "LineString",
                    "coordinates": list(map(list, zip(plon, plat, strict=True))),
                },
            }
        )
        selectable &= np.hypot(rows - target[0], cols - target[1]) * RESOLUTION_M >= 1500
        selectable &= np.hypot(rows - start[0], cols - start[1]) * RESOLUTION_M >= 1500

    output.mkdir(parents=True, exist_ok=True)
    export_suitability_geotiff(
        ROI, "jezero_ctx_20m", WEIGHTS, output / "suitability.tif", dem, suitability
    )
    bands = (
        dem.values,
        metrics.slope,
        metrics.aspect,
        metrics.roughness,
        metrics.tri,
        np.where(np.isfinite(metrics.slope), metrics.hillshade, np.nan),
        cost,
    )
    with rasterio.open(
        output / "terrain.tif",
        "w",
        driver="GTiff",
        height=dem.shape[0],
        width=dem.shape[1],
        count=len(bands),
        dtype="float32",
        crs=LOCAL_CRS,
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        for index, (band, name, unit) in enumerate(
            zip(
                bands,
                ("elevation", "slope", "aspect", "roughness", "tri", "hillshade", "route_cost"),
                ("m", "degree", "degree", "m", "m", "0-255", "dimensionless"),
                strict=True,
            ),
            1,
        ):
            dst.write(np.where(np.isfinite(band), band, -9999).astype(np.float32), index)
            dst.set_band_description(index, name)
            dst.update_tags(index, units=unit)
    _write_json(
        output / "routes.geojson",
        {
            "type": "FeatureCollection",
            "coordinate_system": "Mars sphere R=3396190 m, east longitude/planetocentric latitude (degrees)",
            "features": features,
        },
    )
    _plot(output / "comparison.png", dem, metrics, suitability, features)
    report = [
        "# Jezero terrain comparison",
        "",
        "Measured NASA MRO CTX stereo DEM, distributed by USGS.",
        "",
        "Illustrative terrain-only candidate cells, not habitat engineering recommendations.",
        "",
        "| Rank | Longitude E | Latitude N | Score | Slope (deg) | Roughness (m) | Route (m) | Straight (m) | Terrain cost (m equivalent) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in candidates:
        report.append(
            f"| {c['rank']} | {c['lon']:.5f} | {c['lat']:.5f} | {c['score']:.4f} | {c['slope_deg']:.3f} | {c['roughness_m']:.3f} | {c['distance_m']:.1f} | {c['straight_line_m']:.1f} | {c['terrain_cost_m_equivalent']:.1f} |"
        )
    report += [
        "",
        "Weights: slope 0.70, roughness 0.30; both lower-is-better, min-max normalized over jointly valid cells.",
        "Roughness: 5x5 population standard deviation (100 m footprint). Traversability: slope <=15 degrees and roughness <=20 m.",
        "A*: eight-connected, no corner cutting, destination-cell cost times step length. No smoothing.",
        "Terrain cost is a heuristic penalty in distance-equivalent units, not rover energy or travel time.",
        "Candidates: highest-scoring cells in the largest traversable component, separated by at least 1.5 km.",
        "One common proposed start near the NW corner; scores are relative to this ROI, not probabilities.",
        "Bilinear reprojection to a 20 m local spherical azimuthal equidistant grid; distances are planar.",
        "GeoJSON stores Mars degrees and requires a planetary-aware viewer; it is not Earth RFC 7946 WGS84.",
        "No ice, geology, soil strength, radiation, dust, power, solar or mission economics are inferred.",
        "",
        "![DEM, slope, suitability and routes](comparison.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "data_mode": "observed_dem",
        "source": provenance,
        "roi_degrees": ROI.model_dump(),
        "grid": {
            "crs_wkt": LOCAL_CRS.to_wkt(),
            "transform": list(transform)[:6],
            "shape": list(dem.shape),
            "resolution_m": RESOLUTION_M,
            "valid_cells": int(np.isfinite(dem.values).sum()),
            "resampling": "bilinear",
            "nodata": "NaN",
        },
        "configuration": {
            "weights": WEIGHTS,
            "beneficial": {k: False for k in WEIGHTS},
            "constraints": CONSTRAINTS,
            "roughness_window_cells": 5,
            "candidate_separation_m": 1500,
            "candidate_edge_margin_cells": 5,
            "smoothing": False,
            "connectivity": 8,
            "corner_cutting": False,
            "cost": "1 + exp(2*slope/15)-1 + 5*(roughness/(20+1e-6))^2",
        },
        "start_pixel": list(start),
        "candidates": candidates,
        "software": _software(),
        "outputs_sha256": {
            name: sha256(output / name)
            for name in (
                "suitability.tif",
                "terrain.tif",
                "routes.geojson",
                "comparison.png",
                "report.md",
            )
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _plot(path, dem, metrics, suitability, features):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    left, bottom, right, top = array_bounds(*dem.shape, rasterio.Affine(*dem.attrs["transform"]))
    extent = np.array([left, right, bottom, top]) / 1000
    for ax, array, title, cmap, label in zip(
        axes,
        (dem.values, metrics.slope, suitability),
        ("CTX elevations", "Slope", "Suitability and routes"),
        ("terrain", "magma", "viridis"),
        ("Elevation (m)", "Degrees", "Relative score"),
        strict=True,
    ):
        artist = ax.imshow(array, extent=extent, cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Local easting (km)")
        ax.set_ylabel("Local northing (km)")
        fig.colorbar(artist, ax=ax, label=label, shrink=0.65)
    from_geo = Transformer.from_crs(LOCAL_CRS.geodetic_crs, LOCAL_CRS, always_xy=True)
    for feature, color in zip(features, ("#e63946", "#ff9f1c", "#ffffff"), strict=True):
        coords = np.asarray(feature["geometry"]["coordinates"])
        x, y = from_geo.transform(coords[:, 0], coords[:, 1])
        axes[2].plot(np.asarray(x) / 1000, np.asarray(y) / 1000, color=color, linewidth=1.5)
        axes[2].plot(x[-1] / 1000, y[-1] / 1000, "o", color=color)
        axes[2].annotate(str(feature["properties"]["rank"]), (x[-1] / 1000, y[-1] / 1000))
    fig.suptitle("Jezero crater | measured CTX DEM | terrain-only comparison | 20 m analysis grid")
    fig.savefig(path, dpi=140, metadata={"Software": "MarsHab"})
    plt.close(fig)
