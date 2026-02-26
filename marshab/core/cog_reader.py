"""Windowed COG reader for local MOLA DEM.

Reads elevation windows from a local Cloud-Optimized GeoTIFF using rasterio
windowed reads. Preserves dtype semantics (Int16), nodata (-32768), and returns
RasterWindowResult compatible with the visualization pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds

from marshab.core.raster_service import RasterWindowResult
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

# MOLA COG nodata value (Int16)
MOLA_NODATA = -32768


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

        # Geographic bounds: left, bottom, right, top
        left = bbox.lon_min
        bottom = bbox.lat_min
        right = bbox.lon_max
        top = bbox.lat_max

        window = window_from_bounds(left, bottom, right, top, transform=transform)
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
                left, bottom, right, top, w, h
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
            transform_out = from_bounds(left, bottom, right, top, out_w, out_h).to_gdal()
        else:
            read_window = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)
            arr = src.read(1, window=read_window).astype(np.float32)
            win_transform = rasterio.windows.transform(read_window, transform)
            transform_out = tuple(win_transform.to_gdal())

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
