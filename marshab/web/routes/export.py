"""Data export endpoints."""

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from marshab.analysis.export import export_suitability_geotiff, generate_analysis_report
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.models import BoundingBox
from marshab.config.preset_loader import PresetLoader
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["export"])


class SuitabilityExportRequest(BaseModel):
    """Request for suitability GeoTIFF export."""
    roi: Dict[str, float] = Field(..., description="ROI bounding box")
    dataset: str = Field("mola", description="Dataset name")
    preset_id: Optional[str] = Field(None, description="Preset ID for weights")
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Custom weights")


class ReportExportRequest(BaseModel):
    """Request for analysis report export."""
    project_id: Optional[str] = Field(None, description="Project ID")
    analysis_id: Optional[str] = Field(None, description="Analysis ID")
    format: str = Field("markdown", description="Report format (markdown or html)")


@router.post("/suitability-geotiff")
async def export_suitability(request: SuitabilityExportRequest):
    """Export suitability scores as GeoTIFF file."""
    try:
        # Parse ROI
        roi = BoundingBox(**request.roi)
        
        # Load preset weights if specified
        weights = {}
        if request.preset_id:
            loader = PresetLoader()
            preset = loader.get_preset(request.preset_id, scope="site")
            if preset:
                weights = preset.get_weights_dict()
        
        if request.custom_weights:
            weights.update(request.custom_weights)
        
        # Run analysis to get suitability data
        pipeline = AnalysisPipeline()
        results = pipeline.run(
            roi=roi,
            dataset=request.dataset.lower(),
            threshold=0.5,  # Lower threshold to get full suitability surface
            criteria_weights=weights if weights else None
        )
        
        if results.suitability is None:
            raise HTTPException(
                status_code=500,
                detail="Suitability data not available from analysis"
            )
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"suitability_{timestamp}.tif"
        
        # Export GeoTIFF
        export_path = export_suitability_geotiff(
            roi=roi,
            dataset=request.dataset,
            weights=weights,
            output_path=output_path,
            dem=results.dem,
            suitability=results.suitability
        )
        
        return FileResponse(
            path=str(export_path),
            filename=export_path.name,
            media_type="image/tiff"
        )
        
    except Exception as e:
        logger.error("Suitability export failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def export_report(request: ReportExportRequest):
    """Generate and download analysis report."""
    try:
        # For now, use latest analysis results
        # In future, would load from project_id or analysis_id
        output_dir = Path("data/output")
        sites_file = output_dir / "sites.csv"
        
        if not sites_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Analysis results not found. Run analysis first."
            )
        
        import pandas as pd
        sites_df = pd.read_csv(sites_file)
        
        # Convert to report format
        sites = sites_df.to_dict('records')
        top_site = sites[0] if len(sites) > 0 else None
        metadata = {
            "num_sites": len(sites),
            "top_site_id": top_site["site_id"] if top_site else None,
            "top_site_score": top_site["suitability_score"] if top_site else 0.0,
        }
        
        # Generate report
        report_path = generate_analysis_report(
            project_or_run={"sites": sites, "top_site": top_site, "metadata": metadata},
            format=request.format
        )
        
        media_type = "text/markdown" if request.format == "markdown" else "text/html"
        
        return FileResponse(
            path=str(report_path),
            filename=report_path.name,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Report export failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

