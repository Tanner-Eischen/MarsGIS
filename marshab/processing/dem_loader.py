"""DEM loading and raster I/O operations using GDAL/rasterio."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling
import xarray as xr

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
    
    def load(self, path: Path) -> xr.DataArray:
        """Load DEM into xarray DataArray with geospatial metadata.

        Args:
            path: Path to GeoTIFF DEM file

        Returns:
            xarray DataArray with elevation data and coordinates

        Raises:
            DataError: If DEM cannot be loaded or is invalid
        """
        try:
            with rasterio.open(path) as src:
                # Read elevation data
                elevation = src.read(1).astype(np.float32)

                # Extract spatial info
                height, width = elevation.shape
                transform = src.transform
                crs = src.crs
                nodata = src.nodata

                logger.info(
                    "Loading DEM",
                    path=str(path),
                    shape=(height, width),
                    crs=str(crs),
                    nodata=nodata,
                )

                # Calculate lat/lon coordinates
                rows, cols = np.meshgrid(
                    np.arange(height), np.arange(width), indexing="ij"
                )
                lons, lats = rasterio.transform.xy(transform, rows, cols)

                # Create xarray DataArray with coordinates
                dem = xr.DataArray(
                    elevation,
                    dims=["y", "x"],
                    coords={
                        "lat": (["y", "x"], np.array(lats)),
                        "lon": (["y", "x"], np.array(lons)),
                    },
                    attrs={
                        "crs": str(crs),
                        "transform": list(transform)[:6],
                        "nodata": nodata,
                        "resolution_m": abs(transform.a),
                    },
                )

                logger.info(
                    "DEM loaded successfully",
                    shape=elevation.shape,
                    resolution_m=abs(transform.a),
                    lat_range=(float(np.nanmin(lats)), float(np.nanmax(lats))),
                    lon_range=(float(np.nanmin(lons)), float(np.nanmax(lons))),
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
                "DEM clipped",
                clipped_shape=clipped.shape,
                pixels_retained=int(np.sum(combined_mask.values)),
            )

            return clipped

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

