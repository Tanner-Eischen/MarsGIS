"""DEM download endpoints."""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from marshab.core.data_manager import DataManager
from marshab.exceptions import DataError
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class DownloadRequest(BaseModel):
    """Request model for DEM download."""

    dataset: str = Field(..., description="Dataset name (mola, hirise, ctx)")
    roi: list[float] = Field(..., description="Region of interest [lat_min, lat_max, lon_min, lon_max]")
    force: bool = Field(False, description="Force re-download even if cached")


class DownloadResponse(BaseModel):
    """Response model for DEM download."""

    status: str
    path: str
    cached: bool
    size_mb: float | None = None


@router.post("/download", response_model=DownloadResponse)
async def download_dem(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Download DEM data for specified region."""
    try:
        # Validate ROI
        if len(request.roi) != 4:
            raise HTTPException(status_code=400, detail="ROI must have 4 values: [lat_min, lat_max, lon_min, lon_max]")
        
        lat_min, lat_max, lon_min, lon_max = request.roi
        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )
        
        # Validate dataset
        valid_datasets = ["mola", "hirise", "ctx"]
        if request.dataset.lower() not in valid_datasets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dataset. Must be one of: {', '.join(valid_datasets)}",
            )
        
        # Check if already cached
        data_manager = DataManager()
        dataset_lower = request.dataset.lower()
        
        # Try to get cached path first
        cached_path = data_manager._get_cache_path(dataset_lower, bbox)
        if cached_path.exists() and not request.force:
            size_mb = cached_path.stat().st_size / (1024 * 1024)
            return DownloadResponse(
                status="success",
                path=str(cached_path),
                cached=True,
                size_mb=round(size_mb, 2),
            )
        
        # Download DEM
        logger.info("Downloading DEM", dataset=dataset_lower, roi=bbox.model_dump())
        dem_path = data_manager.download_dem(dataset_lower, bbox, force=request.force)
        
        size_mb = dem_path.stat().st_size / (1024 * 1024) if dem_path.exists() else None
        
        return DownloadResponse(
            status="success",
            path=str(dem_path),
            cached=False,
            size_mb=round(size_mb, 2) if size_mb else None,
        )
    except DataError as e:
        logger.error("Download failed", error=str(e))
        # Check if this is a manual download requirement
        details = getattr(e, 'details', {})
        if details.get('requires_manual_download'):
            raise HTTPException(
                status_code=400, 
                detail=str(e),
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during download")
        raise HTTPException(status_code=500, detail=str(e))

