"""DEM download endpoints."""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from marshab.core.data_manager import DataManager
from marshab.exceptions import DataError
from marshab.models import BoundingBox
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
from typing import List
from time import time
from pydantic import BaseModel, Field
from fastapi import HTTPException
from marshab.core.data_manager import DataManager
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger
from marshab.exceptions import DataError

logger = get_logger(__name__)

class PrewarmRequest(BaseModel):
    roi: List[float] = Field(..., description="[lat_min, lat_max, lon_min, lon_max]")
    tile_deg: float = Field(5.0, gt=0, description="Tile size in degrees")
    force: bool = Field(False, description="Force re-download even if cached")

class PrewarmResponse(BaseModel):
    status: str
    total_tiles: int
    processed_count: int
    total_duration_s: float

def _tiles_from_roi(roi: List[float], tile_deg: float) -> List[BoundingBox]:
    if len(roi) != 4:
        raise HTTPException(status_code=400, detail={"error":"invalid_roi","hint":"ROI must be [lat_min,lat_max,lon_min,lon_max]"})
    lat_min, lat_max, lon_min, lon_max = roi
    lat_min = max(-90.0, lat_min)
    lat_max = min(90.0, lat_max)
    lon_min = max(0.0, lon_min)
    lon_max = min(360.0, lon_max)

    tiles: List[BoundingBox] = []
    lat = lat_min
    while lat < lat_max:
        next_lat = min(lat + tile_deg, lat_max)
        lon = lon_min
        while lon < lon_max:
            next_lon = min(lon + tile_deg, lon_max)
            tiles.append(BoundingBox(lat_min=lat, lat_max=next_lat, lon_min=lon, lon_max=next_lon))
            lon = next_lon
        lat = next_lat
    return tiles

@router.post("/prewarm/mola-tiles", response_model=PrewarmResponse)
async def prewarm_mola_tiles(request: PrewarmRequest):
    start = time()
    try:
        data_manager = DataManager()
        tiles = _tiles_from_roi(request.roi, request.tile_deg)

        processed = 0
        for bbox in tiles:
            # Trigger download/cache by loading DEM
            _ = data_manager.get_dem_for_roi(bbox, dataset="mola", download=True, clip=True)
            processed += 1

        duration = time() - start
        logger.info("Prewarm complete", tiles=len(tiles), duration_s=duration)
        return PrewarmResponse(status="success", total_tiles=len(tiles), processed_count=processed, total_duration_s=round(duration, 3))
    except DataError as e:
        logger.error("Prewarm failed", error=str(e))
        raise HTTPException(status_code=400, detail={"error":"prewarm_failed","detail":str(e)})
    except Exception as e:
        logger.exception("Unexpected error during prewarm")
        raise HTTPException(status_code=500, detail={"error":"internal_error","detail":str(e)})

class PrewarmExamplesResponse(BaseModel):
    status: str
    processed_count: int
    total_duration_s: float

@router.post("/prewarm/examples", response_model=PrewarmExamplesResponse)
async def prewarm_example_rois():
    start = time()
    try:
        from pathlib import Path as P
        import yaml
        cfg_path = P(__file__).parent.parent.parent / "config" / "example_rois.yaml"
        if not cfg_path.exists():
            raise HTTPException(status_code=404, detail={"error":"examples_not_found","detail":str(cfg_path)})
        with open(cfg_path, 'r') as f:
            data = yaml.safe_load(f)
        examples = data.get("examples", {})
        dm = DataManager()
        processed = 0
        for _, ex in examples.items():
            bbox = ex.get("bbox", {})
            dataset = ex.get("dataset", "mola").lower()
            if not all(k in bbox for k in ("lat_min","lat_max","lon_min","lon_max")):
                continue
            b = BoundingBox(lat_min=float(bbox['lat_min']), lat_max=float(bbox['lat_max']), lon_min=float(bbox['lon_min']), lon_max=float(bbox['lon_max']))
            _ = dm.get_dem_for_roi(b, dataset=dataset, download=True, clip=True)
            processed += 1
        duration = time() - start
        logger.info("Prewarm examples complete", processed=processed, duration_s=duration)
        return PrewarmExamplesResponse(status="success", processed_count=processed, total_duration_s=round(duration, 3))
    except Exception as e:
        logger.exception("Prewarm examples failed")
        raise HTTPException(status_code=500, detail={"error":"internal_error","detail":str(e)})

