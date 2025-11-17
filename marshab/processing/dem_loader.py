"""DEM loading and raster I/O operations using GDAL/rasterio."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling
import xarray as xr
try:
    import rioxarray  # noqa: F401 - needed to register rio accessor
except ImportError:
    pass  # rioxarray not available, will use fallback methods

from marshab.exceptions import DataError
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class DEMLoader:
    """Handles Mars DEM loading and preprocessing."""
    
    def __init__(self, mars_radius_eq: float = 3396190.0):
        """Initialize DEM loader.
        
        Args:
            mars_radius_eq: Mars equatorial radius in meters
        """
        self.mars_radius_eq = mars_radius_eq
    
    def load(self, path: Path, roi: Optional[BoundingBox] = None) -> xr.DataArray:
        """Load DEM into xarray DataArray with geospatial metadata.

        Args:
            path: Path to GeoTIFF DEM file
            roi: Optional region of interest to clip during loading (memory efficient)

        Returns:
            xarray DataArray with elevation data and coordinates

        Raises:
            DataError: If DEM cannot be loaded or is invalid
        """
        try:
            with rasterio.open(path) as src:
                # Extract spatial info
                transform = src.transform
                crs = src.crs
                nodata = src.nodata
                full_height, full_width = src.height, src.width

                # If ROI provided, use windowed reading to only load the needed region
                if roi is not None:
                    from rasterio.windows import from_bounds, Window
                    # Calculate window for ROI
                    window = from_bounds(
                        roi.lon_min, roi.lat_min, roi.lon_max, roi.lat_max,
                        transform
                    )
                    # Round to integer pixel bounds
                    window = window.round_offsets().round_lengths()
                    
                    # Clamp window to image bounds and ensure minimum size
                    row_start = max(0, int(window.row_off))
                    col_start = max(0, int(window.col_off))
                    row_stop = min(full_height, int(window.row_off + window.height))
                    col_stop = min(full_width, int(window.col_off + window.width))
                    
                    # Ensure we have a valid window (at least 1 pixel)
                    # If window is too small, expand it slightly to ensure we get data
                    if row_stop <= row_start:
                        # Expand by at least 10 pixels or to edge
                        row_stop = min(full_height, row_start + max(10, int(0.01 * full_height)))
                    if col_stop <= col_start:
                        # Expand by at least 10 pixels or to edge
                        col_stop = min(full_width, col_start + max(10, int(0.01 * full_width)))
                    
                    # Create window object
                    read_window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                    
                    # Validate window
                    if read_window.width <= 0 or read_window.height <= 0:
                        logger.warning(
                            "Invalid window calculated, falling back to full DEM read",
                            window=read_window,
                            roi=roi.model_dump()
                        )
                        elevation = src.read(1).astype(np.float32)
                    else:
                        # Read only the windowed region
                        try:
                            elevation = src.read(1, window=read_window).astype(np.float32)
                            
                            # Update transform for the windowed region
                            transform = rasterio.windows.transform(read_window, transform)
                        except Exception as e:
                            logger.warning(
                                "Windowed read failed, falling back to full DEM read",
                                error=str(e),
                                window=read_window
                            )
                            elevation = src.read(1).astype(np.float32)
                    
                    logger.info(
                        "Loading DEM with ROI window",
                        path=str(path),
                        full_shape=(full_height, full_width),
                        window_shape=elevation.shape,
                        window=read_window,
                        crs=str(crs),
                    )
                else:
                    # Read full elevation data
                    elevation = src.read(1).astype(np.float32)
                    
                    logger.info(
                        "Loading DEM",
                        path=str(path),
                        shape=elevation.shape,
                        crs=str(crs),
                        nodata=nodata,
                    )

                height, width = elevation.shape

                # Only create coordinate arrays for the loaded region (much smaller)
                # Use a more memory-efficient approach: calculate coordinates only for corners/edges
                # For large arrays, we'll use a sparse coordinate approach
                if height * width > 10_000_000:  # For very large arrays, skip full coordinate arrays
                    logger.info(
                        "Large DEM detected, using transform-based coordinates",
                        shape=(height, width)
                    )
                    # Create xarray without full coordinate arrays - use transform instead
                    dem = xr.DataArray(
                        elevation,
                        dims=["y", "x"],
                        attrs={
                            "crs": str(crs),
                            "transform": list(transform)[:6],
                            "nodata": nodata,
                            "resolution_m": abs(transform.a),
                        },
                    )
                    # Add rio accessor for geospatial operations
                    dem.rio.write_transform(transform, inplace=True)
                    dem.rio.write_crs(crs, inplace=True)
                else:
                    # For smaller arrays, create full coordinate arrays
                    # Calculate lat/lon coordinates efficiently
                    rows, cols = np.meshgrid(
                        np.arange(height), np.arange(width), indexing="ij"
                    )
                    # rasterio.transform.xy returns (x, y) for each (row, col)
                    # We need to call it correctly to get 2D arrays
                    lons_flat, lats_flat = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
                    
                    # Reshape to match elevation shape
                    lons = np.array(lons_flat).reshape(height, width)
                    lats = np.array(lats_flat).reshape(height, width)

                    # Create xarray DataArray with coordinates
                    dem = xr.DataArray(
                        elevation,
                        dims=["y", "x"],
                        coords={
                            "lat": (["y", "x"], lats),
                            "lon": (["y", "x"], lons),
                        },
                        attrs={
                            "crs": str(crs),
                            "transform": list(transform)[:6],
                            "nodata": nodata,
                            "resolution_m": abs(transform.a),
                        },
                    )
                    # Add rio accessor for geospatial operations (needed for bounds() and clip_box)
                    # Only if rioxarray is available and accessor can be set up
                    try:
                        if hasattr(dem, 'rio'):
                            dem.rio.write_transform(transform, inplace=True)
                            dem.rio.write_crs(crs, inplace=True)
                    except (AttributeError, RuntimeError):
                        # rio accessor not available, will use coordinate arrays as fallback
                        pass

                logger.info(
                    "DEM loaded successfully",
                    shape=elevation.shape,
                    resolution_m=abs(transform.a),
                )

                return dem

        except RasterioIOError as e:
            raise DataError(
                f"Failed to read DEM file: {path}",
                details={"path": str(path), "error": str(e)},
            )
        except Exception as e:
            raise DataError(
                f"Unexpected error loading DEM: {path}",
                details={"path": str(path), "error": str(e)},
            )
    
    def clip_to_roi(self, dem: xr.DataArray, roi: BoundingBox) -> xr.DataArray:
        """Clip DEM raster to region of interest.

        Args:
            dem: Input DEM DataArray
            roi: Bounding box to clip to

        Returns:
            Clipped DEM DataArray

        Raises:
            DataError: If clipping fails
        """
        try:
            logger.info(
                "Clipping DEM to ROI",
                original_shape=dem.shape,
                roi=roi.model_dump(),
            )

            # Use rio accessor if available (more efficient)
            if hasattr(dem, 'rio') and hasattr(dem.rio, 'clip_box'):
                try:
                    # Use rioxarray's clip_box method (most efficient)
                    clipped = dem.rio.clip_box(
                        minx=roi.lon_min,
                        miny=roi.lat_min,
                        maxx=roi.lon_max,
                        maxy=roi.lat_max
                    )
                    logger.info(
                        "DEM clipped using rio.clip_box",
                        clipped_shape=clipped.shape,
                    )
                    return clipped
                except Exception as e:
                    logger.warning("rio.clip_box failed, falling back to coordinate-based clipping", error=str(e))

            # Fallback: Use coordinate-based clipping if coordinates exist
            if "lat" in dem.coords and "lon" in dem.coords:
                # Create mask for pixels within ROI
                lat_mask = (dem.coords["lat"] >= roi.lat_min) & (
                    dem.coords["lat"] <= roi.lat_max
                )
                lon_mask = (dem.coords["lon"] >= roi.lon_min) & (
                    dem.coords["lon"] <= roi.lon_max
                )
                combined_mask = lat_mask & lon_mask

                # Apply mask
                clipped = dem.where(combined_mask, drop=True)

                logger.info(
                    "DEM clipped using coordinate mask",
                    clipped_shape=clipped.shape,
                )
                return clipped
            else:
                # If no coordinates, use transform-based clipping
                # Calculate pixel bounds from transform
                if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
                    bounds = dem.rio.bounds()
                    # Check if ROI overlaps with DEM bounds
                    if (roi.lon_max < bounds.left or roi.lon_min > bounds.right or
                        roi.lat_max < bounds.bottom or roi.lat_min > bounds.top):
                        raise DataError(
                            "ROI does not overlap with DEM bounds",
                            details={"roi": roi.model_dump(), "dem_bounds": bounds}
                        )
                    
                    # Use rio clip_box if available
                    clipped = dem.rio.clip_box(
                        minx=roi.lon_min,
                        miny=roi.lat_min,
                        maxx=roi.lon_max,
                        maxy=roi.lat_max
                    )
                    logger.info(
                        "DEM clipped using transform-based clip_box",
                        clipped_shape=clipped.shape,
                    )
                    return clipped
                else:
                    # Last resort: return as-is with warning
                    logger.warning(
                        "Cannot clip DEM - no coordinates or transform available, returning full DEM",
                        shape=dem.shape
                    )
                    return dem

        except Exception as e:
            raise DataError(
                f"Failed to clip DEM to ROI",
                details={"roi": roi.model_dump(), "error": str(e)},
            )
    
    def resample(
        self,
        dem: xr.DataArray,
        target_resolution_m: float,
        method: str = "bilinear",
    ) -> xr.DataArray:
        """Resample DEM to different resolution.

        Args:
            dem: Input DEM DataArray
            target_resolution_m: Target resolution in meters
            method: Resampling method ('nearest', 'bilinear', 'cubic')

        Returns:
            Resampled DEM DataArray
        """
        current_res = float(dem.attrs.get("resolution_m", 200.0))

        if abs(current_res - target_resolution_m) < 1.0:
            logger.info("DEM already at target resolution")
            return dem

        scale_factor = current_res / target_resolution_m

        logger.info(
            "Resampling DEM",
            from_resolution_m=current_res,
            to_resolution_m=target_resolution_m,
            scale_factor=scale_factor,
            from_shape=dem.shape,
        )

        from scipy.ndimage import zoom

        order = {"nearest": 0, "bilinear": 1, "cubic": 3}.get(method, 1)
        resampled_data = zoom(dem.values, scale_factor, order=order)

        resampled = xr.DataArray(
            resampled_data,
            dims=["y", "x"],
            attrs={**dem.attrs, "resolution_m": target_resolution_m},
        )

        return resampled

