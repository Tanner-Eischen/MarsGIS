"""Mission scenario endpoints."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.config import get_config
from marshab.mission.scenarios import (
    LandingScenarioParams,
    TraverseScenarioParams,
    run_landing_site_scenario,
    run_rover_traverse_scenario,
)
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/mission", tags=["mission"])


class LandingScenarioRequest(BaseModel):
    """Request for landing site scenario."""
    roi: dict[str, float] = Field(..., description="ROI as {lat_min, lat_max, lon_min, lon_max}")
    dataset: str = Field("mola", description="Dataset name (mola, mola_200m, hirise, ctx)")
    preset_id: Optional[str] = Field(None, description="Preset ID for criteria weights")
    constraints: Optional[dict[str, float]] = Field(None, description="Mission constraints")
    suitability_threshold: float = Field(0.7, ge=0, le=1, description="Minimum suitability threshold")
    custom_weights: Optional[dict[str, float]] = Field(None, description="Custom criteria weights")


class LandingScenarioResponse(BaseModel):
    """Response from landing site scenario."""
    scenario_id: str
    sites: list[dict]
    top_site: Optional[dict]
    metadata: dict


class TraverseScenarioRequest(BaseModel):
    """Request for rover traverse scenario."""
    start_site_id: int = Field(..., description="Starting site ID")
    end_site_id: int = Field(..., description="Target site ID")
    analysis_dir: str = Field(..., description="Path to analysis results directory")
    preset_id: Optional[str] = Field(None, description="Route preset ID")
    rover_capabilities: Optional[dict[str, float]] = Field(None, description="Rover constraints")
    start_lat: Optional[float] = Field(None, ge=-90, le=90, description="Starting latitude (overrides site)")
    start_lon: Optional[float] = Field(None, ge=0, le=360, description="Starting longitude (overrides site)")
    custom_weights: Optional[dict[str, float]] = Field(None, description="Custom route weights")


class TraverseScenarioResponse(BaseModel):
    """Response from rover traverse scenario."""
    route_id: str
    waypoints: list[dict]
    total_distance_m: float
    estimated_time_h: float
    risk_score: float
    metadata: dict


@router.post("/landing-scenario", response_model=LandingScenarioResponse)
async def create_landing_scenario(request: LandingScenarioRequest):
    """Run landing site selection scenario.

    Orchestrates DEM download, terrain analysis, MCDM evaluation, and site extraction.
    """
    try:
        # Parse ROI
        try:
            roi = BoundingBox(
                lat_min=request.roi["lat_min"],
                lat_max=request.roi["lat_max"],
                lon_min=request.roi["lon_min"],
                lon_max=request.roi["lon_max"],
            )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing ROI field: {e}")

        # Parse constraints
        max_slope_deg = None
        min_area_km2 = None
        if request.constraints:
            max_slope_deg = request.constraints.get("max_slope_deg")
            min_area_km2 = request.constraints.get("min_area_km2")

        config = get_config()
        if getattr(config, "demo_mode", False):
            import uuid

            import numpy as np
            rows, cols = 6, 8
            lats = np.linspace(roi.lat_min, roi.lat_max, rows)
            lons = np.linspace(roi.lon_min, roi.lon_max, cols)
            sites = []
            sid = 1
            for i in range(rows):
                for j in range(cols):
                    lat = float(lats[i])
                    lon = float(lons[j])
                    area = float(max(0.5, (roi.lat_max - roi.lat_min) * (roi.lon_max - roi.lon_min) / (rows * cols)))
                    score = float((i + j) / (rows + cols))
                    site = {
                        "site_id": sid,
                        "geometry_type": "POINT",
                        "area_km2": area,
                        "lat": lat,
                        "lon": lon,
                        "mean_slope_deg": float(3.0 + (i % 3)),
                        "mean_roughness": float(50.0 + (j % 5) * 5.0),
                        "mean_elevation_m": float(-1500.0 + (i - j) * 10.0),
                        "suitability_score": score,
                        "rank": 0,
                        "polygon_coords": None,
                    }
                    sites.append(site)
                    sid += 1
            sites.sort(key=lambda s: s["suitability_score"], reverse=True)
            for idx, s in enumerate(sites, start=1):
                s["rank"] = idx
            top_site = sites[0] if sites else None
            return LandingScenarioResponse(
                scenario_id=str(uuid.uuid4()),
                sites=sites,
                top_site=top_site,
                metadata={"mode": "demo"},
            )
        else:
            params = LandingScenarioParams(
                roi=roi,
                dataset=request.dataset.lower(),
                preset_id=request.preset_id,
                max_slope_deg=max_slope_deg,
                min_area_km2=min_area_km2,
                suitability_threshold=request.suitability_threshold,
                custom_weights=request.custom_weights,
            )
            result = run_landing_site_scenario(params)
            sites_data = [site.model_dump() for site in result.sites]
            top_site_data = result.top_site.model_dump() if result.top_site else None
            return LandingScenarioResponse(
                scenario_id=result.scenario_id,
                sites=sites_data,
                top_site=top_site_data,
                metadata=result.metadata,
            )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Landing scenario failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error in landing scenario")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rover-traverse", response_model=TraverseScenarioResponse)
async def create_rover_traverse(request: TraverseScenarioRequest):
    """Run rover traverse planning scenario.

    Plans route between two sites with cost analysis.
    """
    try:
        config = get_config()
        if getattr(config, "demo_mode", False):
            import uuid

            import numpy as np
            if request.start_lat is None or request.start_lon is None:
                raise HTTPException(status_code=400, detail="start_lat and start_lon required in demo mode")
            if request.end_site_id is None and (request.start_lat is None or request.start_lon is None):
                raise HTTPException(status_code=400, detail="end site coordinates required in demo mode")
            start = np.array([request.start_lon, request.start_lat], dtype=float)
            end = np.array([request.start_lon + 0.2, request.start_lat + 0.2], dtype=float)
            n = 25
            lons = np.linspace(start[0], end[0], n)
            lats = np.linspace(start[1], end[1], n)
            waypoints = []
            total = 0.0
            prev = None
            for idx in range(n):
                lon = float(lons[idx])
                lat = float(lats[idx])
                wp = {"waypoint_id": idx + 1, "lon": lon, "lat": lat, "x_meters": 0.0, "y_meters": 0.0, "tolerance_meters": 5.0}
                waypoints.append(wp)
                if prev is not None:
                    dx = (lon - prev[0]) * 111000.0
                    dy = (lat - prev[1]) * 111000.0
                    total += float((dx * dx + dy * dy) ** 0.5)
                prev = (lon, lat)
            risk = 0.2
            return TraverseScenarioResponse(
                route_id=str(uuid.uuid4()),
                waypoints=waypoints,
                total_distance_m=total,
                estimated_time_h=total / 360.0,
                risk_score=risk,
                metadata={"mode": "demo"},
            )

        analysis_dir = Path(request.analysis_dir)
        if not analysis_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Analysis directory not found: {analysis_dir}"
            )

        # Create scenario params
        params = TraverseScenarioParams(
            start_site_id=request.start_site_id,
            end_site_id=request.end_site_id,
            analysis_dir=analysis_dir,
            preset_id=request.preset_id,
            rover_capabilities=request.rover_capabilities,
            start_lat=request.start_lat,
            start_lon=request.start_lon,
            custom_weights=request.custom_weights,
        )

        # Run scenario
        result = run_rover_traverse_scenario(params)

        # Estimate time (simplified: assume 0.1 m/s average speed)
        estimated_time_h = result.route_metrics.total_distance_m / 360.0  # 0.1 m/s = 360 m/h

        return TraverseScenarioResponse(
            route_id=result.route_id,
            waypoints=result.waypoints,
            total_distance_m=result.route_metrics.total_distance_m,
            estimated_time_h=estimated_time_h,
            risk_score=result.route_metrics.risk_score,
            metadata=result.metadata,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Traverse scenario failed", error=str(e))
        message = str(e)
        status_code = 404 if "not found" in message.lower() else 400
        raise HTTPException(status_code=status_code, detail=message)
    except Exception:
        logger.exception("Unexpected error in traverse scenario")
        raise HTTPException(status_code=500, detail="Internal server error")
