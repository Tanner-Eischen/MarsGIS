"""Site analysis API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.analysis.site_scoring import SiteScoringEngine
from marshab.config.preset_loader import PresetLoader
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


class SiteAnalysisRequest(BaseModel):
    """Request for site analysis."""
    roi: Dict[str, float] = Field(
        ..., 
        description="Bounding box: lat_min, lat_max, lon_min, lon_max"
    )
    dataset: str = Field("mola", description="DEM dataset")
    preset_id: Optional[str] = Field(None, description="Preset ID to use")
    custom_weights: Optional[Dict[str, float]] = Field(
        None, 
        description="Custom criterion weights (overrides preset)"
    )
    threshold: float = Field(0.7, ge=0, le=1)


class SiteScoreResponse(BaseModel):
    """Single site score response."""
    site_id: int
    rank: int
    total_score: float
    components: Dict[str, float]
    explanation: str
    geometry: dict
    centroid_lat: float
    centroid_lon: float
    area_km2: float


class PresetResponse(BaseModel):
    """Preset information response."""
    id: str
    name: str
    description: str
    scope: str
    weights: Dict[str, float]


class PresetsListResponse(BaseModel):
    """List of presets response."""
    site_presets: List[PresetResponse]
    route_presets: List[PresetResponse]


@router.get("/presets", response_model=PresetsListResponse)
async def get_presets():
    """Get list of available presets."""
    try:
        loader = PresetLoader()
        loader.load()
        
        site_presets = [
            PresetResponse(
                id=p.id,
                name=p.name,
                description=p.description,
                scope=p.scope,
                weights=p.get_weights_dict()
            )
            for p in loader.list_presets(scope="site")
        ]
        
        route_presets = [
            PresetResponse(
                id=p.id,
                name=p.name,
                description=p.description,
                scope=p.scope,
                weights=p.get_weights_dict()
            )
            for p in loader.list_presets(scope="route")
        ]
        
        return PresetsListResponse(
            site_presets=site_presets,
            route_presets=route_presets
        )
    except Exception as e:
        logger.error("Failed to load presets", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/site-scores", response_model=List[SiteScoreResponse])
async def analyze_sites(request: SiteAnalysisRequest):
    """Analyze sites with preset or custom weights.
    
    Returns ranked sites with component breakdowns and explanations.
    """
    try:
        # Load preset if specified
        weights = None
        if request.preset_id:
            loader = PresetLoader()
            preset = loader.get_preset(request.preset_id, "site")
            
            if preset is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset: {request.preset_id}"
                )
            
            weights = preset.get_weights_dict()
        
        # Override with custom weights if provided
        if request.custom_weights:
            if weights is None:
                weights = {}
            weights.update(request.custom_weights)
        
        # Run analysis pipeline
        pipeline = AnalysisPipeline()
        roi = BoundingBox(**request.roi)
        
        results = pipeline.run(
            roi,
            dataset=request.dataset,
            threshold=request.threshold,
            criteria_weights=weights
        )
        
        # Get beneficial criteria from preset loader
        loader = PresetLoader()
        loader.load()
        
        beneficial = {}
        for name in weights.keys() if weights else []:
            criterion = loader.get_criterion(name)
            if criterion:
                beneficial[name] = criterion.beneficial
        
        # If no beneficial dict, use defaults
        if not beneficial:
            beneficial = {
                "slope": False,
                "roughness": False,
                "elevation": False,
                "solar_exposure": True,
                "science_value": True
            }
        
        # Score sites with explainability
        scoring_engine = SiteScoringEngine()
        
        # Use criteria from results if available, otherwise use empty dict
        criteria = results.criteria if results.criteria else {}
        
        scored_sites = scoring_engine.score_sites(
            results.sites,
            criteria,
            weights or {},
            beneficial
        )
        
        # Format response
        response = []
        for idx, scored_site in enumerate(scored_sites):
            # Create geometry from polygon_coords if available
            geometry = {}
            if scored_site.site.polygon_coords:
                geometry = {
                    "type": "Polygon",
                    "coordinates": [scored_site.site.polygon_coords]
                }
            else:
                geometry = {
                    "type": "Point",
                    "coordinates": [scored_site.site.lon, scored_site.site.lat]
                }
            
            response.append(SiteScoreResponse(
                site_id=scored_site.site.site_id,
                rank=idx + 1,
                total_score=scored_site.total_score,
                components=scored_site.components,
                explanation=scored_site.explanation,
                geometry=geometry,
                centroid_lat=scored_site.site.lat,
                centroid_lon=scored_site.site.lon,
                area_km2=scored_site.site.area_km2
            ))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Site analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

