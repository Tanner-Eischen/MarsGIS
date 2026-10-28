"""Windowed COG reader for local MOLA DEM.

Reads elevation windows from a local Cloud-Optimized GeoTIFF using rasterio
windowed reads. Preserves dtype semantics (Int16), nodata (-32768), and returns
RasterWindowResult compatible with the visualization pipeline.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

from marshab.core.raster_service import RasterWindowResult
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

# MOLA COG nodata value (Int16)
MOLA_NODATA = -32768


def _extract_central_meridian(src: rasterio.io.DatasetReader) -> float:
    """Best-effort extraction of central meridian from CRS WKT."""
    if src.crs is None:
        return 0.0
    try:
        wkt = src.crs.to_wkt() or ""
    except Exception:
        return 0.0
    match = re.search(r'PARAMETER\["central_meridian",\s*([-+]?\d+(?:\.\d+)?)\]', wkt)
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
    """Project lon/lat bbox into a Mars simple-cylindrical raster domain."""
    src_bounds = src.bounds
    src_width = float(src_bounds.right - src_bounds.left)
    src_height = float(src_bounds.top - src_bounds.bottom)
    if src_width <= 0 or src_height <= 0:
        return 0.0, 0.0, 0.0, 0.0

    central_meridian = _extract_central_meridian(src)
    start_lon = (central_meridian - 180.0) % 360.0
    lon_min = float(bbox.lon_min % 360.0)
    lon_max = float(bbox.lon_max % 360.0)

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

    return (
        float(min(x_left, x_right)),
        float(min(y_bottom, y_top)),
        float(max(x_left, x_right)),
        float(max(y_bottom, y_top)),
    )


def _source_window_bounds(
    bbox: BoundingBox,
    src: rasterio.io.DatasetReader,
) -> tuple[float, float, float, float]:
    """Map request lon/lat bounds into the source CRS/window bounds."""
    left = float(bbox.lon_min)
    bottom = float(bbox.lat_min)
    right = float(bbox.lon_max)
    top = float(bbox.lat_max)

    if src.crs is None or src.crs.is_geographic:
        # Align to lon domain when source bounds look like [-180, 180].
        if -180.1 <= float(src.bounds.left) <= 180.1 and -180.1 <= float(src.bounds.right) <= 180.1:
            if left >= 180.0 and right >= 180.0:
                left -= 360.0
                right -= 360.0
            elif right > 180.0 > left:
                right -= 360.0
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
        # PROJ can return near-zero values for custom Mars CRS transforms;
        # detect and fall back to deterministic simple-cylindrical mapping.
        if (
            max(abs(transformed_tuple[0]), abs(transformed_tuple[2])) < 1000.0
            and max(abs(src_bounds.left), abs(src_bounds.right)) > 100000.0
        ):
            return _project_lonlat_bounds_for_simple_cylindrical(bbox, src)
        return transformed_tuple
    except Exception:
        return _project_lonlat_bounds_for_simple_cylindrical(bbox, src)


def read_cog_window(
    cog_path: str,
    bbox: BoundingBox,
    target_shape: Optional[tuple[int, int]] = None,
) -> RasterWindowResult:
    """Read windowed elevation from local COG file.

    Args:
        cog_path: Path to COG GeoTIFF
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max in degrees)
        target_shape: Optional (height, width) for resampled output

    Returns:
        RasterWindowResult with array (float32), nodata, transform
    """
    with rasterio.open(cog_path) as src:
        transform = src.transform
        nodata = src.nodata
        if nodata is None:
            nodata = float(MOLA_NODATA)

        # Request bounds in geographic lon/lat.
        left_geo = float(bbox.lon_min)
        bottom_geo = float(bbox.lat_min)
        right_geo = float(bbox.lon_max)
        top_geo = float(bbox.lat_max)

        # Convert request bounds into source CRS bounds for window addressing.
        left_src, bottom_src, right_src, top_src = _source_window_bounds(bbox, src)

        window = window_from_bounds(left_src, bottom_src, right_src, top_src, transform=transform)
        window = window.round_offsets().round_lengths()

        # Clamp to raster bounds
        row_off = max(0, int(window.row_off))
        col_off = max(0, int(window.col_off))
        row_stop = min(src.height, int(window.row_off + window.height))
        col_stop = min(src.width, int(window.col_off + window.width))

        if row_stop <= row_off or col_stop <= col_off:
            # Empty window - return minimal valid result
            h, w = 1, 1
            arr = np.full((h, w), nodata, dtype=np.float32)
            transform_out = from_bounds(
                left_geo, bottom_geo, right_geo, top_geo, w, h
            ).to_gdal()
        elif target_shape:
            # Resample to target shape
            out_h, out_w = target_shape
            arr = src.read(
                1,
                window=Window(col_off, row_off, col_stop - col_off, row_stop - row_off),
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=nodata,
            )
            arr = arr.astype(np.float32)
            transform_out = from_bounds(left_geo, bottom_geo, right_geo, top_geo, out_w, out_h).to_gdal()
        else:
            read_window = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)
            arr = src.read(1, window=read_window).astype(np.float32)
            # Return geographic transform for downstream lon/lat row/col sampling.
            transform_out = from_bounds(
                left_geo,
                bottom_geo,
                right_geo,
                top_geo,
                arr.shape[1],
                arr.shape[0],
            ).to_gdal()

    return RasterWindowResult(
        array=arr,
        bbox_used=bbox,
        dataset_requested="mola",
        dataset_used="mola",
        is_fallback=False,
        fallback_reason=None,
        resolution_m=463.0,
        nodata=float(nodata),
        transform=transform_out,
    )
