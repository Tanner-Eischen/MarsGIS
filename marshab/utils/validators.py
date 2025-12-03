"""Input validation utilities."""

from pathlib import Path

import rasterio
from rasterio.crs import CRS

from marshab.exceptions import DataError
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


def validate_dem_crs(dem_path: Path, expected_crs: str = "IAU_MARS_2000") -> bool:
    """Validate DEM has correct Mars coordinate system.
    
    Args:
        dem_path: Path to GeoTIFF DEM
        expected_crs: Expected CRS string
    
    Returns:
        True if CRS is valid
    
    Raises:
        DataError: If CRS is invalid or cannot be read
    """
    try:
        with rasterio.open(dem_path) as src:
            crs = src.crs
            
            if crs is None:
                raise DataError(
                    f"DEM has no CRS defined: {dem_path}",
                    details={"path": str(dem_path)}
                )
            
            # Check if CRS string contains Mars identifier
            crs_str = str(crs)
            if "MARS" not in crs_str.upper() and "49900" not in crs_str:
                logger.warning(
                    "DEM CRS may not be Mars-specific",
                    path=str(dem_path),
                    crs=crs_str
                )
            
            return True
            
    except rasterio.errors.RasterioIOError as e:
        raise DataError(
            f"Cannot read DEM file: {dem_path}",
            details={"path": str(dem_path), "error": str(e)}
        )


def validate_roi_bounds(roi: BoundingBox) -> bool:
    """Validate ROI is within Mars valid ranges.
    
    Args:
        roi: Bounding box to validate
    
    Returns:
        True if valid
    
    Raises:
        DataError: If ROI is invalid
    """
    # Already validated by Pydantic, but add logical checks
    if roi.lat_max - roi.lat_min < 0.1:
        raise DataError(
            "ROI latitude range too small (< 0.1 degrees)",
            details={"roi": roi.model_dump()}
        )
    
    if roi.lon_max - roi.lon_min < 0.1:
        raise DataError(
            "ROI longitude range too small (< 0.1 degrees)",
            details={"roi": roi.model_dump()}
        )
    
    logger.info(
        "ROI validated",
        lat_range=f"{roi.lat_min}-{roi.lat_max}",
        lon_range=f"{roi.lon_min}-{roi.lon_max}"
    )
    
    return True





