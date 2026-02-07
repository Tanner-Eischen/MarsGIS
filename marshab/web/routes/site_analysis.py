"""Site analysis API endpoints."""

import os
from typing import Optional

import rasterio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.analysis.decision_brief import build_decision_brief
from marshab.analysis.site_scoring import SiteScoringEngine
from marshab.config import get_config
from marshab.config.preset_loader import PresetLoader
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.data_manager import DataManager
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


class SiteAnalysisRequest(BaseModel):
    """Request for site analysis."""
    roi: dict[str, float] = Field(
        ...,
        description="Bounding box: lat_min, lat_max, lon_min, lon_max"
    )
    dataset: str = Field("mola", description="DEM dataset")
    preset_id: Optional[str] = Field(None, description="Preset ID to use")
    custom_weights: Optional[dict[str, float]] = Field(
        None,
        description="Custom criterion weights (overrides preset)"
    )
    threshold: float = Field(0.7, ge=0, le=1)


class DecisionBriefRequest(BaseModel):
    """Request for deterministic decision brief generation."""

    roi: dict[str, float] = Field(
        ...,
        description="Bounding box: lat_min, lat_max, lon_min, lon_max"
    )
    dataset: str = Field("mola", description="DEM dataset")
    preset_id: Optional[str] = Field(None, description="Preset ID to use")
    custom_weights: Optional[dict[str, float]] = Field(
        None,
        description="Custom criterion weights (overrides preset)"
    )
    threshold: float = Field(0.6, ge=0, le=1)
    site_id: Optional[int] = Field(None, description="Optional site id; defaults to top-ranked site")
    start_lat: Optional[float] = Field(None, ge=-90, le=90)
    start_lon: Optional[float] = Field(None, ge=0, le=360)


class SiteScoreResponse(BaseModel):
    """Single site score response."""
    site_id: int
    rank: int
    total_score: float
    components: dict[str, float]
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
    weights: dict[str, float]


class PresetsListResponse(BaseModel):
    """List of presets response."""
    site_presets: list[PresetResponse]
    route_presets: list[PresetResponse]


class DecisionBriefResponse(BaseModel):
    """Deterministic decision brief response."""

    site_id: int
    rank: int
    suitability_score: float
    summary: str
    reasons: list[str]
    terrain: dict[str, float]
    route_impacts: dict[str, float | dict[str, float]]
    determinism: dict[str, int | str | None]


def _resolve_weights(request: SiteAnalysisRequest | DecisionBriefRequest) -> dict[str, float]:
    """Resolve weights from preset and custom overrides."""
    weights: dict[str, float] = {}
    if request.preset_id:
        loader = PresetLoader()
        preset = loader.get_preset(request.preset_id, "site")
        if preset is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset: {request.preset_id}"
            )
        weights = preset.get_weights_dict()

    if request.custom_weights:
        weights.update(request.custom_weights)

    return weights


def _infer_data_mode(dataset: str, roi: BoundingBox) -> str:
    """Infer whether current run used real DEM or synthetic fallback."""
    try:
        dm = DataManager()
        dem_path = dm._get_cache_path(dataset, roi)  # noqa: SLF001 - best-effort heuristic
        if not dem_path.exists():
            return "synthetic"
        with rasterio.open(dem_path) as src:
            tags = src.tags()
            if tags.get("SOURCE_URL"):
                return "real_dem"
        return "synthetic"
    except Exception:
        return "synthetic"


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


@router.post("/site-scores", response_model=list[SiteScoreResponse])
async def analyze_sites(request: SiteAnalysisRequest):
    """Analyze sites with preset or custom weights.

    Returns ranked sites with component breakdowns and explanations.
    """
    try:
        weights = _resolve_weights(request)

        # Run analysis pipeline
        pipeline = AnalysisPipeline()
        roi = BoundingBox(**request.roi)

        results = pipeline.run(
            roi,
            dataset=request.dataset,
            threshold=request.threshold,
            criteria_weights=weights or None
        )

        # Persist sites for downstream GeoJSON and navigation flows.
        config = get_config()
        results.save(config.paths.output_dir)

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


@router.post("/decision-brief", response_model=DecisionBriefResponse)
async def generate_decision_brief(request: DecisionBriefRequest):
    """Generate deterministic, rules-based explanation for selected/top site."""
    try:
        weights = _resolve_weights(request)
        roi = BoundingBox(**request.roi)

        pipeline = AnalysisPipeline()
        results = pipeline.run(
            roi,
            dataset=request.dataset,
            threshold=request.threshold,
            criteria_weights=weights or None,
        )
        if not results.sites:
            raise HTTPException(status_code=404, detail="No candidate sites found for ROI")

        config = get_config()
        results.save(config.paths.output_dir)

        selected_site = results.sites[0]
        if request.site_id is not None:
            matching = [site for site in results.sites if site.site_id == request.site_id]
            if not matching:
                raise HTTPException(status_code=404, detail=f"Site id {request.site_id} not found in ranked candidates")
            selected_site = matching[0]

        start_lat = request.start_lat if request.start_lat is not None else (roi.lat_min + roi.lat_max) / 2.0
        start_lon = request.start_lon if request.start_lon is not None else (roi.lon_min + roi.lon_max) / 2.0
        seed_raw = os.getenv("MARSHAB_DEMO_SEED")
        seed = int(seed_raw) if seed_raw and seed_raw.lstrip("-").isdigit() else None

        brief = build_decision_brief(
            selected_site,
            start_lat=start_lat,
            start_lon=start_lon,
            seed=seed,
            data_mode=_infer_data_mode(request.dataset, roi),
        )
        return DecisionBriefResponse(**brief)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Decision brief generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

