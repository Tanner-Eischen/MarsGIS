"""Terrain analysis endpoints."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.exceptions import AnalysisError
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger
from marshab.web.routes.progress import ProgressTracker, generate_task_id

logger = get_logger(__name__)

router = APIRouter()


class AnalysisRequest(BaseModel):
    """Request model for terrain analysis."""

    roi: list[float] = Field(..., description="Region of interest [lat_min, lat_max, lon_min, lon_max]")
    dataset: str = Field("mola", description="Dataset to use (mola, hirise, ctx)")
    threshold: float = Field(0.7, ge=0, le=1, description="Suitability threshold (0-1)")
    task_id: Optional[str] = Field(None, description="Optional task ID for progress tracking (client-generated)")


class SiteCandidateResponse(BaseModel):
    """Response model for site candidate."""

    site_id: int
    geometry_type: str
    area_km2: float
    lat: float
    lon: float
    mean_slope_deg: float
    mean_roughness: float
    mean_elevation_m: float
    suitability_score: float
    rank: int


class AnalysisResponse(BaseModel):
    """Response model for terrain analysis."""

    status: str
    sites: list[SiteCandidateResponse]
    top_site_id: int
    top_site_score: float
    output_dir: str
    task_id: str = Field(None, description="Task ID for progress tracking")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_terrain(request: AnalysisRequest):
    """Run terrain analysis for specified region."""
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
        
        # Use provided task_id or generate new one
        task_id = request.task_id or generate_task_id()
        
        # Create progress tracker (queue will be created when WebSocket connects)
        progress_tracker = ProgressTracker(task_id)
        
        # Progress callback wrapper
        def progress_callback(stage: str, progress: float, message: str):
            progress_tracker.update(stage, progress, message)
        
        # Run analysis
        logger.info("Running terrain analysis", roi=bbox.model_dump(), threshold=request.threshold, task_id=task_id)
        pipeline = AnalysisPipeline()
        results = pipeline.run(
            roi=bbox,
            dataset=request.dataset.lower(),
            threshold=request.threshold,
            progress_callback=progress_callback,
        )
        
        # Convert sites to response format
        sites_response = [
            SiteCandidateResponse(
                site_id=site.site_id,
                geometry_type=site.geometry_type,
                area_km2=site.area_km2,
                lat=site.lat,
                lon=site.lon,
                mean_slope_deg=site.mean_slope_deg,
                mean_roughness=site.mean_roughness,
                mean_elevation_m=site.mean_elevation_m,
                suitability_score=site.suitability_score,
                rank=site.rank,
            )
            for site in results.sites
        ]
        
        # Get output directory
        from marshab.config import get_config
        config = get_config()
        output_dir = str(config.paths.output_dir)
        
        return AnalysisResponse(
            status="success",
            sites=sites_response,
            top_site_id=results.top_site_id,
            top_site_score=results.top_site_score,
            output_dir=output_dir,
            task_id=task_id,
        )
    except AnalysisError as e:
        logger.error("Analysis failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(status_code=500, detail=str(e))




