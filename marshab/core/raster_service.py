"""Shared raster access helpers for DEM-driven visualization endpoints."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from marshab.config import get_config
from marshab.core.data_manager import DataManager
from marshab.exceptions import DataError
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

MarsDataset = Literal["mola", "mola_200m", "hirise", "ctx"]


@dataclass
class DatasetResolutionResult:
    dataset_requested: MarsDataset
    dataset_used: MarsDataset
    is_fallback: bool
    fallback_reason: Optional[str]


@dataclass
class RasterWindowResult:
    array: np.ndarray
    bbox_used: BoundingBox
    dataset_requested: MarsDataset
    dataset_used: MarsDataset
    is_fallback: bool
    fallback_reason: Optional[str]
    resolution_m: Optional[float]
    nodata: Optional[float]
    transform: Optional[tuple[float, float, float, float, float, float]]


def normalize_dataset(dataset: str) -> MarsDataset:
    dataset_lower = dataset.lower().strip()
    if dataset_lower not in {"mola", "mola_200m", "hirise", "ctx"}:
        raise ValueError(f"Invalid dataset: {dataset}")
    return dataset_lower  # type: ignore[return-value]


def to_lon360(lon: float) -> float:
    """Normalize longitude to [0, 360)."""
    while lon < 0:
        lon += 360
    while lon >= 360:
        lon -= 360
    return lon


def to_lon180(lon: float) -> float:
    """Normalize longitude to [-180, 180]."""
    lon = to_lon360(lon)
    return lon - 360 if lon > 180 else lon


def compute_tile_bbox_epsg4326(z: int, x: int, y: int) -> BoundingBox:
    """Compute EPSG:4326 tile bbox for a z/x/y tile in geodetic scheme."""
    if z < 0 or x < 0 or y < 0:
        raise ValueError("Invalid tile indices")
    tiles = 2 ** z
    if x >= tiles or y >= tiles:
        raise ValueError("Tile indices out of range")
    lon_span = 360.0 / tiles
    lat_span = 180.0 / tiles
    lon_min = -180.0 + x * lon_span
    lon_max = lon_min + lon_span
    lat_max = 90.0 - y * lat_span
    lat_min = lat_max - lat_span
    return BoundingBox(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


def _cache_has_real_data(path) -> bool:
    try:
        if not path.exists():
            return False
        if path.stat().st_size > 2 * 1024 * 1024:
            return True
        with rasterio.open(path) as src:
            if src.width * src.height >= 500_000:
                return True
            if src.tags().get("SOURCE_URL"):
                return True
    except Exception:
        return False
    return False


def resolve_dataset_with_fallback(requested: MarsDataset, bbox: BoundingBox) -> DatasetResolutionResult:
    """Resolve dataset with fallback rules for HiRISE coverage."""
    if requested != "hirise":
        return DatasetResolutionResult(
            dataset_requested=requested,
            dataset_used=requested,
            is_fallback=False,
            fallback_reason=None,
        )

    data_manager = DataManager()
    roi = BoundingBox(
        lat_min=bbox.lat_min,
        lat_max=bbox.lat_max,
        lon_min=to_lon360(bbox.lon_min),
        lon_max=to_lon360(bbox.lon_max),
    )
    hirise_cache = data_manager._get_cache_path("hirise", roi)
    if _cache_has_real_data(hirise_cache):
        return DatasetResolutionResult(
            dataset_requested=requested,
            dataset_used="hirise",
            is_fallback=False,
            fallback_reason=None,
        )

    mola200_cache = data_manager._get_cache_path("mola_200m", roi)
    if _cache_has_real_data(mola200_cache):
        return DatasetResolutionResult(
            dataset_requested=requested,
            dataset_used="mola_200m",
            is_fallback=True,
            fallback_reason="hirise_unavailable",
        )

    return DatasetResolutionResult(
        dataset_requested=requested,
        dataset_used="mola",
        is_fallback=True,
        fallback_reason="hirise_unavailable",
    )


def _resample_array(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resample 2D array to target shape using PIL."""
    from PIL import Image

    img = Image.fromarray(arr)
    resized = img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
    return np.array(resized)


def load_dem_window(
    dataset: str,
    bbox: BoundingBox,
    target_shape: Optional[tuple[int, int]] = None,
    allow_download: bool = True,
) -> RasterWindowResult:
    """Load DEM window with consistent dataset resolution + fallback behavior."""
    dataset_normalized = normalize_dataset(dataset)
    resolution = resolve_dataset_with_fallback(dataset_normalized, bbox)
    config = get_config()
    data_manager = DataManager()

    def load_from_cache(dataset_id: MarsDataset):
        roi_360 = BoundingBox(
            lat_min=bbox.lat_min,
            lat_max=bbox.lat_max,
            lon_min=to_lon360(bbox.lon_min),
            lon_max=to_lon360(bbox.lon_max),
        )
        cache_path = data_manager._get_cache_path(dataset_id, roi_360)
        if not cache_path.exists():
            raise DataError(f"Dataset cache missing: {dataset_id}")
        dem = data_manager.load_dem(cache_path)
        dem = data_manager.loader.clip_to_roi(dem, roi_360)
        return dem

    try:
        roi_360 = BoundingBox(
            lat_min=bbox.lat_min,
            lat_max=bbox.lat_max,
            lon_min=to_lon360(bbox.lon_min),
            lon_max=to_lon360(bbox.lon_max),
        )
        if allow_download:
            dem = data_manager.get_dem_for_roi(
                roi_360,
                dataset=resolution.dataset_used,
                download=True,
                clip=True,
            )
            allow_synthetic = os.getenv("MARSHAB_ALLOW_SYNTHETIC_TILES", "false").lower() in {"1", "true", "yes"}
            if (
                resolution.dataset_used in {"mola", "mola_200m"}
                and not allow_synthetic
                and not _cache_has_real_data(data_manager._get_cache_path(resolution.dataset_used, roi_360))
            ):
                raise DataError(f"Synthetic fallback disallowed for dataset {resolution.dataset_used}")
        else:
            dem = load_from_cache(resolution.dataset_used)
    except Exception as exc:
        if resolution.dataset_used == "mola_200m":
            logger.warning("Falling back to MOLA after MOLA 200m failure", error=str(exc))
            fallback_resolution = DatasetResolutionResult(
                dataset_requested=resolution.dataset_requested,
                dataset_used="mola",
                is_fallback=True,
                fallback_reason="mola_200m_unavailable",
            )
            if allow_download:
                dem = data_manager.get_dem_for_roi(
                    roi_360,
                    dataset=fallback_resolution.dataset_used,
                    download=True,
                    clip=True,
                )
            else:
                dem = load_from_cache(fallback_resolution.dataset_used)
            resolution = fallback_resolution
        else:
            raise

    array = dem.values.astype(np.float32)
    if target_shape and array.shape[:2] != target_shape:
        array = _resample_array(array, target_shape)
        transform = from_bounds(
            roi_360.lon_min,
            roi_360.lat_min,
            roi_360.lon_max,
            roi_360.lat_max,
            target_shape[1],
            target_shape[0],
        )
        transform_values = transform.to_gdal()
    else:
        transform_values = None
        if hasattr(dem, "attrs"):
            transform_attr = dem.attrs.get("transform")
            if isinstance(transform_attr, (list, tuple)) and len(transform_attr) >= 6:
                transform_values = tuple(transform_attr[:6])

    resolution_m = None
    if resolution.dataset_used in config.data_sources:
        resolution_m = config.data_sources[resolution.dataset_used].resolution_m

    nodata = dem.attrs.get("nodata") if hasattr(dem, "attrs") else None

    return RasterWindowResult(
        array=array,
        bbox_used=bbox,
        dataset_requested=resolution.dataset_requested,
        dataset_used=resolution.dataset_used,
        is_fallback=resolution.is_fallback,
        fallback_reason=resolution.fallback_reason,
        resolution_m=resolution_m,
        nodata=nodata,
        transform=transform_values,
    )


def tile_style_hash(params: dict[str, str]) -> str:
    if not params:
        return "default"
    items = sorted(params.items())
    raw = "&".join(f"{k}={v}" for k, v in items)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def tile_cache_buster() -> str:
    return os.getenv("MARSHAB_DEM_CACHE_BUSTER", str(int(time.time() // 86400)))
