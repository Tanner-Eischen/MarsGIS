"""Navigation planning endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.config import PathfindingStrategy, get_config
from marshab.core.navigation_engine import NavigationEngine
from marshab.exceptions import NavigationError
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class NavigationRequest(BaseModel):
    """Request model for navigation planning."""

    site_id: int = Field(..., ge=1, description="Target site ID")
    analysis_dir: str = Field("data/output", description="Analysis results directory")
    start_lat: float = Field(..., ge=-90, le=90, description="Start latitude")
    start_lon: float = Field(..., ge=0, le=360, description="Start longitude")
    strategy: str = Field("balanced", description="Pathfinding strategy: safest, balanced, or direct")


class WaypointResponse(BaseModel):
    """Response model for waypoint."""

    waypoint_id: int
    x_meters: float
    y_meters: float
    tolerance_meters: float


class NavigationResponse(BaseModel):
    """Response model for navigation planning."""

    status: str
    waypoints: list[WaypointResponse]
    path_length_m: float | None = None
    num_waypoints: int


@router.post("/navigate", response_model=NavigationResponse)
async def plan_navigation(request: NavigationRequest):
    """Generate rover navigation waypoints to target site."""
    try:
        # Validate strategy
        valid_strategies = ["safest", "balanced", "direct"]
        if request.strategy.lower() not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}",
            )
        
        # Update config with strategy
        config = get_config()
        try:
            strategy_enum = PathfindingStrategy(request.strategy.lower())
            config.navigation.strategy = strategy_enum
        except ValueError:
            logger.warning(f"Invalid strategy '{request.strategy}', using default")
        
        # Validate analysis directory
        analysis_dir = Path(request.analysis_dir)
        if not analysis_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Analysis directory not found: {analysis_dir}",
            )
        
        sites_file = analysis_dir / "sites.csv"
        if not sites_file.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Sites file not found: {sites_file}. Run analysis first.",
            )
        
        # Generate waypoints
        logger.info(
            "Planning navigation",
            site_id=request.site_id,
            start_lat=request.start_lat,
            start_lon=request.start_lon,
            strategy=request.strategy,
        )
        
        engine = NavigationEngine()
        waypoints_df = engine.plan_to_site(
            site_id=request.site_id,
            analysis_dir=analysis_dir,
            start_lat=request.start_lat,
            start_lon=request.start_lon,
        )
        
        # Convert to response format
        waypoints_response = [
            WaypointResponse(
                waypoint_id=int(row["waypoint_id"]),
                x_meters=float(row["x_meters"]),
                y_meters=float(row["y_meters"]),
                tolerance_meters=float(row["tolerance_meters"]),
            )
            for _, row in waypoints_df.iterrows()
        ]
        
        # Calculate approximate path length
        path_length_m = None
        if len(waypoints_response) > 1:
            total_distance = 0.0
            for i in range(len(waypoints_response) - 1):
                wp1 = waypoints_response[i]
                wp2 = waypoints_response[i + 1]
                distance = (
                    (wp2.x_meters - wp1.x_meters) ** 2 + (wp2.y_meters - wp1.y_meters) ** 2
                ) ** 0.5
                total_distance += distance
            path_length_m = round(total_distance, 2)
        
        return NavigationResponse(
            status="success",
            waypoints=waypoints_response,
            path_length_m=path_length_m,
            num_waypoints=len(waypoints_response),
        )
    except NavigationError as e:
        logger.error("Navigation planning failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during navigation planning")
        raise HTTPException(status_code=500, detail=str(e))




