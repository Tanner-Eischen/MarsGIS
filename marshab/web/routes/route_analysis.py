"""Route cost analysis API."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.analysis.route_cost import RouteCostEngine
from marshab.analysis.routing import compute_route_cost, plan_route
from marshab.config.preset_loader import PresetLoader
from marshab.core.data_manager import DataManager
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


class RouteCostRequest(BaseModel):
    """Request for route cost analysis."""
    site_id_start: int
    site_id_end: int
    analysis_dir: str
    preset_id: Optional[str] = None
    custom_weights: Optional[dict[str, float]] = None


class RouteCostResponse(BaseModel):
    """Route cost analysis response."""
    total_cost: float
    distance_m: float
    components: dict[str, float]
    explanation: str
    num_waypoints: int


class RoutePlanRequest(BaseModel):
    """Request for route planning."""
    start_site_id: int = Field(..., description="Starting site ID")
    end_site_id: int = Field(..., description="Target site ID")
    analysis_dir: str = Field(..., description="Path to analysis results directory")
    preset_id: Optional[str] = Field(None, description="Route preset ID")
    custom_weights: Optional[dict[str, float]] = Field(None, description="Custom route weights")
    sun_azimuth: Optional[float] = Field(None, ge=0, le=360, description="Sun azimuth (degrees)")
    sun_altitude: Optional[float] = Field(None, ge=0, le=90, description="Sun altitude (degrees)")


class RoutePlanResponse(BaseModel):
    """Route planning response."""
    route: dict  # GeoJSON LineString
    waypoints: list[dict]
    cost_summary: dict[str, float]


@router.post("/route-plan", response_model=RoutePlanResponse)
async def plan_route_endpoint(request: RoutePlanRequest):
    """Plan route between two sites with constraint awareness."""
    try:
        # Load sites to get coordinates
        import pandas as pd
        sites_csv = Path(request.analysis_dir) / "sites.csv"
        if not sites_csv.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Sites file not found: {sites_csv}"
            )

        sites_df = pd.read_csv(sites_csv)

        start_site = sites_df[sites_df["site_id"] == request.start_site_id]
        end_site = sites_df[sites_df["site_id"] == request.end_site_id]

        if start_site.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Start site {request.start_site_id} not found"
            )
        if end_site.empty:
            raise HTTPException(
                status_code=404,
                detail=f"End site {request.end_site_id} not found"
            )

        start_lat = float(start_site["lat"].iloc[0])
        start_lon = float(start_site["lon"].iloc[0])
        end_lat = float(end_site["lat"].iloc[0])
        end_lon = float(end_site["lon"].iloc[0])

        # Create ROI around route
        roi = BoundingBox(
            lat_min=min(start_lat, end_lat) - 0.1,
            lat_max=max(start_lat, end_lat) + 0.1,
            lon_min=min(start_lon, end_lon) - 0.1,
            lon_max=max(start_lon, end_lon) + 0.1
        )

        # Load DEM
        data_manager = DataManager()
        dem = data_manager.get_dem_for_roi(roi, dataset="mola", download=True, clip=True)

        # Get cell size
        if hasattr(dem, 'rio') and hasattr(dem.rio, 'res'):
            cell_size_m = float(abs(dem.rio.res[0]))
        else:
            cell_size_m = 463.0

        # Convert lat/lon to pixel coordinates
        # This is simplified - would need proper coordinate transformation
        # For now, use approximate conversion
        bounds = dem.rio.bounds() if hasattr(dem, 'rio') else None
        if bounds:
            lon_min, lat_min, lon_max, lat_max = bounds.left, bounds.bottom, bounds.right, bounds.top
            height, width = dem.shape

            start_col = int((start_lon - lon_min) / (lon_max - lon_min) * width)
            start_row = int((lat_max - start_lat) / (lat_max - lat_min) * height)
            end_col = int((end_lon - lon_min) / (lon_max - lon_min) * width)
            end_row = int((lat_max - end_lat) / (lat_max - lat_min) * height)
        else:
            # Fallback: use center of DEM
            height, width = dem.shape
            start_row, start_col = height // 2, width // 4
            end_row, end_col = height // 2, 3 * width // 4

        # Load preset weights
        weights = {}
        if request.preset_id:
            loader = PresetLoader()
            preset = loader.get_preset(request.preset_id, "route")
            if preset:
                weights = preset.get_weights_dict()

        if request.custom_weights:
            weights.update(request.custom_weights)

        # Default weights if none provided
        if not weights:
            weights = {
                "distance": 0.3,
                "slope_penalty": 0.3,
                "roughness_penalty": 0.2,
                "elevation_penalty": 0.2,
            }

        # Plan route
        route = plan_route(
            start=(start_row, start_col),
            end=(end_row, end_col),
            weights=weights,
            dem=dem,
            constraints={"max_slope_deg": 25.0, "enable_smoothing": True},
            sun_azimuth=request.sun_azimuth,
            sun_altitude=request.sun_altitude,
            cell_size_m=cell_size_m
        )

        # Compute cost
        cost_result = compute_route_cost(
            route,
            dem,
            weights,
            sun_azimuth=request.sun_azimuth,
            sun_altitude=request.sun_altitude
        )

        # Convert waypoints to GeoJSON format
        waypoints_geojson = []
        coordinates = []

        for i, (row, col) in enumerate(route.waypoints):
            # Convert pixel to lat/lon (simplified)
            if bounds:
                lon = lon_min + (col / width) * (lon_max - lon_min)
                lat = lat_max - (row / height) * (lat_max - lat_min)
            else:
                lon = start_lon + (i / len(route.waypoints)) * (end_lon - start_lon)
                lat = start_lat + (i / len(route.waypoints)) * (end_lat - start_lat)

            waypoints_geojson.append({
                "waypoint_id": i + 1,
                "lat": float(lat),
                "lon": float(lon),
                "x_meters": col * cell_size_m,
                "y_meters": row * cell_size_m,
            })
            coordinates.append([float(lon), float(lat)])

        # Create LineString
        route_geojson = {
            "type": "LineString",
            "coordinates": coordinates
        }

        return RoutePlanResponse(
            route=route_geojson,
            waypoints=waypoints_geojson,
            cost_summary={
                "distance_m": cost_result.distance_m,
                "slope_cost": cost_result.slope_cost,
                "roughness_cost": cost_result.roughness_cost,
                "shadow_cost": cost_result.shadow_cost,
                "energy_estimate_j": cost_result.energy_estimate_j,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Route planning failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route-cost", response_model=RouteCostResponse)
async def analyze_route_cost(request: RouteCostRequest):
    """Analyze route cost with preset or custom weights."""
    try:
        # Load preset
        weights = {}
        if request.preset_id:
            loader = PresetLoader()
            preset = loader.get_preset(request.preset_id, "route")

            if preset is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset: {request.preset_id}"
                )

            weights = preset.get_weights_dict()

        # Override with custom
        if request.custom_weights:
            weights.update(request.custom_weights)

        # Get waypoints (assuming already generated)
        waypoints_file = Path(request.analysis_dir) / f"waypoints_{request.site_id_start}_to_{request.site_id_end}.csv"

        if not waypoints_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Route waypoints not found. Generate route first."
            )

        import pandas as pd
        waypoints = pd.read_csv(waypoints_file)

        # Analyze cost
        engine = RouteCostEngine()
        breakdown = engine.analyze_route(waypoints, weights)

        return RouteCostResponse(
            total_cost=breakdown.total_cost,
            distance_m=breakdown.distance_m,
            components=breakdown.components,
            explanation=breakdown.explanation,
            num_waypoints=len(waypoints)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Route cost analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

