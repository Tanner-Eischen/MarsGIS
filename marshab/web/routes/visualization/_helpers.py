"""Shared helpers and constants for visualization routes."""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from fastapi import HTTPException
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from marshab.config import get_config
from marshab.core.raster_service import normalize_dataset, to_lon360
from marshab.core.tile_cache import TileCache, TileCacheConfig
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

MARS_BASEMAP_TILE_URL = (
    "https://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/"
    "celestia_mars-shaded-16k_global/{z}/{x}/{y}.png"
)

TILE_CACHE = TileCache(
    TileCacheConfig(
        max_entries=int(os.getenv("MARSHAB_TILE_CACHE_MAX", "4000")),
        ttl_seconds=int(os.getenv("MARSHAB_TILE_CACHE_TTL", str(7 * 24 * 60 * 60))),
    )
)
ENABLE_TILE_BASEMAP = os.getenv("MARSHAB_ENABLE_TILE_BASEMAP", "true").lower() in {"1", "true", "yes"}
ENABLE_TILE_OVERLAYS = os.getenv("MARSHAB_ENABLE_TILE_OVERLAYS", "true").lower() in {"1", "true", "yes"}
ORTHO_BASEMAP_PATH_ENV = "MARSHAB_ORTHO_BASEMAP_PATH"
GLOBAL_BASEMAP_PATH_ENV = "MARSHAB_GLOBAL_BASEMAP_PATH"
GLOBAL_BASEMAP_STYLE_ENV = "MARSHAB_GLOBAL_BASEMAP_STYLE"
DEFAULT_GLOBAL_BASEMAP_STYLE = "blendshade"
GLOBAL_BASEMAP_STYLE_FILES: dict[str, str] = {
    "blendshade": "Mars_HRSC_MOLA_BlendShade_Global_200mp_v2.tif",
    "clrshade": "Mars_MGS_MOLA_ClrShade_merge_global_463m.tif",
    "shade": "Mars_MGS_MOLA_Shade_global_463m.tif",
}


def _build_dem_dataarray(raster_result) -> xr.DataArray:
    rows, cols = raster_result.array.shape
    lat_coords = np.linspace(raster_result.bbox_used.lat_max, raster_result.bbox_used.lat_min, rows)
    lon_coords = np.linspace(raster_result.bbox_used.lon_min, raster_result.bbox_used.lon_max, cols)
    data = xr.DataArray(
        raster_result.array,
        dims=("lat", "lon"),
        coords={"lat": lat_coords, "lon": lon_coords},
        attrs={
            "nodata": raster_result.nodata,
            "transform": raster_result.transform,
            "dataset": raster_result.dataset_used,
        },
    )
    return data


def _log_metric(name: str, value: float = 1.0, **fields) -> None:
    logger.info("metric", metric=name, value=value, **fields)


def _resolve_fallback_basemap_dataset(dataset: str) -> str:
    try:
        dataset_lower = normalize_dataset(dataset)
    except ValueError:
        return "mola_200m"
    return dataset_lower if dataset_lower in {"mola", "mola_200m", "hirise"} else "mola_200m"


def _discover_default_orthophoto_source() -> Path | None:
    config = get_config()
    candidate_dirs = [config.paths.cache_dir, Path("/app/data/cache")]
    candidates: list[Path] = []
    explicit_names = ("hirise.tif", "hirise_orthophoto.tif")
    for cache_dir in candidate_dirs:
        for filename in explicit_names:
            candidate = cache_dir / filename
            try:
                if candidate.exists():
                    candidates.append(candidate)
            except Exception:
                continue
    for cache_dir in candidate_dirs:
        try:
            candidates.extend(sorted(cache_dir.glob("hirise*.tif")))
        except Exception:
            continue
    unique_candidates: dict[str, Path] = {}
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        unique_candidates[key] = candidate
    candidates = list(unique_candidates.values())
    if not candidates:
        return None

    def _score(path: Path) -> tuple[int, float]:
        try:
            stat = path.stat()
            return (int(stat.st_size), stat.st_mtime)
        except Exception:
            return (0, 0.0)

    return max(candidates, key=_score)


def _resolve_orthophoto_source_path() -> Path | None:
    raw_path = os.getenv(ORTHO_BASEMAP_PATH_ENV, "").strip()
    if raw_path:
        return Path(raw_path)
    return _discover_default_orthophoto_source()


def _normalize_global_basemap_style(style: str | None) -> str:
    requested = (style or "").strip().lower()
    if requested in GLOBAL_BASEMAP_STYLE_FILES:
        return requested
    env_style = os.getenv(GLOBAL_BASEMAP_STYLE_ENV, DEFAULT_GLOBAL_BASEMAP_STYLE).strip().lower()
    if env_style in GLOBAL_BASEMAP_STYLE_FILES:
        return env_style
    return DEFAULT_GLOBAL_BASEMAP_STYLE


def _resolve_global_basemap_source(style: str | None = None) -> tuple[str, Path] | None:
    env_override = os.getenv(GLOBAL_BASEMAP_PATH_ENV, "").strip()
    if env_override:
        override_path = Path(env_override)
        if override_path.exists():
            return ("custom", override_path)
    resolved_style = _normalize_global_basemap_style(style)
    fallback_order = [resolved_style] + [
        s for s in [DEFAULT_GLOBAL_BASEMAP_STYLE, "clrshade", "shade"] if s != resolved_style
    ]
    config = get_config()
    candidate_dirs = [
        Path.cwd(),
        Path("/app"),
        Path("/opt/render/project/src"),
        config.paths.data_dir,
        config.paths.cache_dir,
    ]
    for style_name in fallback_order:
        filename = GLOBAL_BASEMAP_STYLE_FILES.get(style_name)
        if not filename:
            continue
        for base_dir in candidate_dirs:
            candidate = base_dir / filename
            try:
                if candidate.exists():
                    return (style_name, candidate)
            except Exception:
                continue
    return None


def _window_bounds_for_orthophoto(bbox: BoundingBox, src: rasterio.io.DatasetReader) -> tuple[float, float, float, float]:
    left = float(bbox.lon_min)
    right = float(bbox.lon_max)
    if -180.1 <= float(src.bounds.left) <= 180.1 and -180.1 <= float(src.bounds.right) <= 180.1:
        if left >= 180.0 and right >= 180.0:
            left -= 360.0
            right -= 360.0
        elif right > 180.0 > left:
            right -= 360.0
    return left, float(bbox.lat_min), right, float(bbox.lat_max)


def _transform_bounds_for_source_crs(
    left: float,
    bottom: float,
    right: float,
    top: float,
    src: rasterio.io.DatasetReader,
) -> tuple[float, float, float, float]:
    if src.crs is None or src.crs.is_geographic:
        return left, bottom, right, top
    try:
        transformed = transform_bounds(
            "EPSG:4326",
            src.crs,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
        return tuple(float(v) for v in transformed)
    except Exception as exc:
        logger.warning(
            "Orthophoto CRS transform failed; using geographic bounds",
            error=str(exc),
            source_crs=str(src.crs),
        )
        return left, bottom, right, top


def _extract_central_meridian(src: rasterio.io.DatasetReader) -> float:
    if src.crs is None:
        return 0.0
    try:
        wkt = src.crs.to_wkt()
    except Exception:
        return 0.0
    match = re.search(r'PARAMETER\["central_meridian",\s*([-+]?\d+(?:\.\d+)?)\]', wkt or "")
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except Exception:
        return 0.0


def _project_lonlat_bounds_for_simple_cylindrical(
    bbox: BoundingBox,
    src: rasterio.io.DatasetReader,
) -> tuple[float, float, float, float]:
    src_bounds = src.bounds
    src_width = float(src_bounds.right - src_bounds.left)
    src_height = float(src_bounds.top - src_bounds.bottom)
    if src_width <= 0 or src_height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    central_meridian = _extract_central_meridian(src)
    start_lon = (central_meridian - 180.0) % 360.0
    lon_min = float(to_lon360(bbox.lon_min))
    lon_max = float(to_lon360(bbox.lon_max))

    def _offset(lon: float) -> float:
        value = lon - start_lon
        while value < 0.0:
            value += 360.0
        return value

    off_min = _offset(lon_min)
    off_max = _offset(lon_max)
    if off_max <= off_min:
        off_max += 360.0
    off_min = float(np.clip(off_min, 0.0, 360.0))
    off_max = float(np.clip(off_max, 0.0, 360.0))
    lat_min = float(np.clip(bbox.lat_min, -90.0, 90.0))
    lat_max = float(np.clip(bbox.lat_max, -90.0, 90.0))
    x_left = src_bounds.left + (off_min / 360.0) * src_width
    x_right = src_bounds.left + (off_max / 360.0) * src_width
    y_bottom = src_bounds.bottom + ((lat_min + 90.0) / 180.0) * src_height
    y_top = src_bounds.bottom + ((lat_max + 90.0) / 180.0) * src_height
    return float(min(x_left, x_right)), float(min(y_bottom, y_top)), float(max(x_left, x_right)), float(max(y_bottom, y_top))


def _window_bounds_for_global_basemap(
    bbox: BoundingBox,
    src: rasterio.io.DatasetReader,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = _window_bounds_for_orthophoto(bbox, src)
    if src.crs is None or src.crs.is_geographic:
        return left, bottom, right, top
    try:
        transformed = transform_bounds(
            "EPSG:4326",
            src.crs,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
        transformed_tuple = tuple(float(v) for v in transformed)
        src_bounds = src.bounds
        if max(abs(transformed_tuple[0]), abs(transformed_tuple[2])) < 1000 and max(abs(src_bounds.left), abs(src_bounds.right)) > 100000:
            return _project_lonlat_bounds_for_simple_cylindrical(bbox, src)
        return transformed_tuple
    except Exception:
        return _project_lonlat_bounds_for_simple_cylindrical(bbox, src)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr_float = arr.astype(np.float32)
    valid_mask = np.isfinite(arr_float)
    if not np.any(valid_mask):
        return np.zeros(arr.shape, dtype=np.uint8)
    valid_values = arr_float[valid_mask]
    lo = float(np.percentile(valid_values, 2))
    hi = float(np.percentile(valid_values, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(valid_values))
        hi = float(np.nanmax(valid_values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr_float - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _render_blank_tile(tile_size: int = 256) -> bytes:
    blank = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    img = Image.fromarray(blank, mode="RGB")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def _render_orthophoto_tile(source_path: Path, bbox: BoundingBox, tile_size: int = 256) -> bytes:
    with rasterio.open(source_path) as src:
        left, bottom, right, top = _window_bounds_for_orthophoto(bbox, src)
        left, bottom, right, top = _transform_bounds_for_source_crs(left, bottom, right, top, src)
        if (
            right <= float(src.bounds.left)
            or left >= float(src.bounds.right)
            or top <= float(src.bounds.bottom)
            or bottom >= float(src.bounds.top)
        ):
            return _render_blank_tile(tile_size)
        window = rasterio.windows.from_bounds(
            left,
            bottom,
            right,
            top,
            transform=src.transform,
        )
        band_count = max(1, min(int(src.count), 3))
        band_indexes = list(range(1, band_count + 1))
        data = src.read(
            band_indexes,
            window=window,
            out_shape=(band_count, tile_size, tile_size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
    if data.shape[0] == 1:
        gray = _to_uint8(data[0])
        rgb = np.stack([gray, gray, gray], axis=2)
    elif data.shape[0] >= 3:
        r = _to_uint8(data[0])
        g = _to_uint8(data[1])
        b = _to_uint8(data[2])
        rgb = np.stack([r, g, b], axis=2)
    else:
        gray = _to_uint8(data[0])
        rgb = np.stack([gray, gray, gray], axis=2)
    img = Image.fromarray(rgb, mode="RGB")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def _render_global_basemap_tile(source_path: Path, bbox: BoundingBox, tile_size: int = 256) -> bytes:
    with rasterio.open(source_path) as src:
        left, bottom, right, top = _window_bounds_for_global_basemap(bbox, src)
        if (
            right <= float(src.bounds.left)
            or left >= float(src.bounds.right)
            or top <= float(src.bounds.bottom)
            or bottom >= float(src.bounds.top)
        ):
            return _render_blank_tile(tile_size)
        window = rasterio.windows.from_bounds(
            left,
            bottom,
            right,
            top,
            transform=src.transform,
        )
        band_count = max(1, min(int(src.count), 3))
        band_indexes = list(range(1, band_count + 1))
        data = src.read(
            band_indexes,
            window=window,
            out_shape=(band_count, tile_size, tile_size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
    if data.shape[0] == 1:
        gray = _to_uint8(data[0])
        rgb = np.stack([gray, gray, gray], axis=2)
    else:
        rgb_channels: list[np.ndarray] = []
        for idx in range(3):
            band = data[idx] if idx < data.shape[0] else data[0]
            rgb_channels.append(_to_uint8(band))
        rgb = np.stack(rgb_channels, axis=2)
    img = Image.fromarray(rgb, mode="RGB")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def _parse_roi(roi: str) -> tuple[float, float, float, float]:
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, roi.split(","))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ROI format: {exc}")
    if lat_min >= lat_max:
        raise HTTPException(status_code=400, detail="Invalid ROI: lat_min must be less than lat_max")
    if not (-90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0):
        raise HTTPException(status_code=400, detail="Invalid ROI: latitude must be within [-90, 90]")
    if lon_min == lon_max:
        raise HTTPException(status_code=400, detail="Invalid ROI: longitude span must be non-zero")
    lon_span = abs(lon_max - lon_min)
    if lon_span > 120.0:
        raise HTTPException(status_code=400, detail="Invalid ROI: longitude span too large for 3D visualization")
    if (lat_max - lat_min) > 60.0:
        raise HTTPException(status_code=400, detail="Invalid ROI: latitude span too large for 3D visualization")
    return lat_min, lat_max, lon_min, lon_max


def _sample_mesh_elevation(lon: float, lat: float, lons: np.ndarray, lats: np.ndarray, elevation: np.ndarray) -> float:
    lon_idx = int(np.argmin(np.abs(lons - lon)))
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    return float(elevation[lat_idx, lon_idx])


def _load_sites_features() -> list[dict[str, Any]]:
    import pandas as pd

    config = get_config()
    output_dir = config.paths.output_dir
    sites_file = output_dir / "sites.csv"
    if not sites_file.exists():
        return []
    sites_df = pd.read_csv(sites_file)
    features: list[dict[str, Any]] = []
    for _, row in sites_df.iterrows():
        lon = float(row.get("lon", 0))
        lat = float(row.get("lat", 0))
        if not (-180 <= lon <= 360) or not (-90 <= lat <= 90):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "site_id": int(row.get("site_id", 0)),
                    "rank": int(row.get("rank", 0)),
                    "suitability_score": float(row.get("suitability_score", 0)),
                },
            }
        )
    return features


def _load_waypoint_features() -> list[dict[str, Any]]:
    import pandas as pd

    config = get_config()
    output_dir = config.paths.output_dir
    files = list(output_dir.glob("waypoints_*.csv"))
    if not files:
        return []
    colors = {"safest": "#00ff00", "balanced": "#1e90ff", "direct": "#ffa500"}
    features: list[dict[str, Any]] = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if len(df) == 0:
            continue
        coords = []
        for _, row in df.iterrows():
            lon = row.get("lon", row.get("longitude"))
            lat = row.get("lat", row.get("latitude"))
            if lon is None or lat is None:
                continue
            lon_val = float(lon)
            lat_val = float(lat)
            if not (-180 <= lon_val <= 360) or not (-90 <= lat_val <= 90):
                continue
            coords.append([lon_val, lat_val])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_val, lat_val]},
                    "properties": {
                        "kind": "waypoint",
                        "waypoint_id": int(row.get("waypoint_id", 0)),
                    },
                }
            )
        if len(coords) > 1:
            route_type = fpath.name.split("_")[-1].replace(".csv", "")
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "kind": "path",
                        "route_type": route_type,
                        "line_color": colors.get(route_type, "#ff0000"),
                    },
                }
            )
    return features


def _project_features_to_mesh(
    features: list[dict[str, Any]],
    lons: np.ndarray,
    lats: np.ndarray,
    elevation: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    scene_points: list[dict[str, Any]] = []
    scene_paths: list[dict[str, Any]] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        properties = feature.get("properties") or {}
        if geom_type == "Point":
            lon, lat = geometry.get("coordinates", [None, None])
            if lon is None or lat is None:
                continue
            z = _sample_mesh_elevation(float(lon), float(lat), lons, lats, elevation)
            scene_points.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "z": z,
                    "properties": properties,
                }
            )
        elif geom_type == "LineString":
            coords = geometry.get("coordinates") or []
            projected_coords = []
            for coord in coords:
                lon, lat = coord[0], coord[1]
                z = _sample_mesh_elevation(float(lon), float(lat), lons, lats, elevation)
                projected_coords.append([float(lon), float(lat), z])
            if projected_coords:
                scene_paths.append({"coordinates": projected_coords, "properties": properties})
    return {"points": scene_points, "paths": scene_paths}
