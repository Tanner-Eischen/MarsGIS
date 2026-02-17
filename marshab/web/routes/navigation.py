"""Navigation planning endpoints."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from marshab.config import PathfindingStrategy, get_config
from marshab.core.navigation_engine import NavigationEngine
from marshab.exceptions import NavigationError
from marshab.utils.logging import get_logger
from marshab.web.routes.progress import ProgressTracker, generate_task_id

logger = get_logger(__name__)

router = APIRouter(tags=["navigation"])

# Thread pool for CPU-bound pathfinding operations
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nav")


class NavigationRequest(BaseModel):
    """Request for navigation planning."""
    site_id: int = Field(..., description="Target site ID")
    analysis_dir: str = Field(..., description="Analysis results directory")
    start_lat: float = Field(..., ge=-90, le=90)
    start_lon: float = Field(..., ge=0, le=360)
    max_waypoint_spacing_m: float = Field(250.0, gt=0, description="Spacing between waypoints (m); larger = fewer waypoints")
    max_slope_deg: float = Field(25.0, gt=0, le=90)
    strategy: PathfindingStrategy = Field(
        default=PathfindingStrategy.BALANCED,
        description="Pathfinding strategy"
    )
    task_id: Optional[str] = Field(None, description="Optional task ID for progress tracking (client-generated)")


class NavigationResponse(BaseModel):
    """Response with navigation waypoints."""
    waypoints: list[dict]
    num_waypoints: int
    total_distance_m: float
    site_id: int
    task_id: str = Field(None, description="Task ID for progress tracking")


@router.post("/plan-route", response_model=NavigationResponse)
async def plan_route(request: NavigationRequest):
    """Plan navigation route to target site.

    Returns waypoints in SITE frame (North, East, Down).

    Note: Pathfinding can take 30-60 seconds for large DEMs.
    """
    try:
        logger.info(
            "Navigation planning request",
            site_id=request.site_id,
            start_lat=request.start_lat,
            start_lon=request.start_lon
        )

        # Use provided task_id or generate new one
        task_id = request.task_id or generate_task_id()

        # Create progress tracker (queue will be created when WebSocket connects)
        progress_tracker = ProgressTracker(task_id)

        # Progress callback wrapper (will be called from thread pool)
        def progress_callback(stage: str, progress: float, message: str):
            progress_tracker.update(stage, progress, message)

        engine = NavigationEngine()

        # Run CPU-bound pathfinding in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            waypoints_df = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor,
                    lambda: engine.plan_to_site(
                        site_id=request.site_id,
                        analysis_dir=Path(request.analysis_dir),
                        start_lat=request.start_lat,
                        start_lon=request.start_lon,
                        max_waypoint_spacing_m=request.max_waypoint_spacing_m,
                        max_slope_deg=request.max_slope_deg,
                        progress_callback=progress_callback,
                        strategy=request.strategy
                    )
                ),
                timeout=120.0  # 2 minute timeout for pathfinding
            )
        except asyncio.TimeoutError:
            logger.error("Navigation planning timed out after 120 seconds")
            raise HTTPException(
                status_code=504,
                detail="Navigation planning timed out. The DEM may be too large or the path is too complex. Try a smaller ROI or different start/goal positions."
            )

        # Persist route artifacts for export and map overlay endpoints.
        analysis_dir = Path(request.analysis_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        site_waypoints_file = analysis_dir / f"waypoints_site_{request.site_id}.csv"
        default_waypoints_file = analysis_dir / "waypoints.csv"
        waypoints_df.to_csv(site_waypoints_file, index=False)
        waypoints_df.to_csv(default_waypoints_file, index=False)

        # Also write to output_dir for the /visualization/waypoints-geojson endpoint
        config = get_config()
        output_dir = config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        strategy_waypoints_file = output_dir / f"waypoints_{request.strategy.value}.csv"
        waypoints_df.to_csv(strategy_waypoints_file, index=False)

        if len(waypoints_df) > 0:
            last = waypoints_df.iloc[-1]
            total_distance = float((last['x_meters']**2 + last['y_meters']**2)**0.5)
        else:
            total_distance = 0.0

        logger.info(
            "Navigation planning completed",
            num_waypoints=len(waypoints_df),
            total_distance_m=total_distance
        )

        return NavigationResponse(
            waypoints=waypoints_df.to_dict('records'),
            num_waypoints=len(waypoints_df),
            total_distance_m=total_distance,
            site_id=request.site_id,
            task_id=task_id
        )

    except HTTPException:
        raise
    except NavigationError as e:
        logger.error("Navigation planning failed", error=str(e))
        # Provide more helpful error messages
        error_detail = str(e)
        if "impassable" in error_detail.lower():
            error_detail += " Try adjusting the start position or selecting a different site. The terrain may be too steep or rough at the selected location."
        elif "outside DEM bounds" in error_detail.lower():
            error_detail += " Ensure the start position is within the analyzed region."
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        logger.exception("Unexpected error in navigation planning", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/waypoints-geojson")
async def get_waypoints_geojson(
    analysis_dir: str = Query(..., description="Analysis directory"),
    site_id: int = Query(..., description="Site ID")
):
    """Get waypoints as GeoJSON for visualization."""
    try:
        import pandas as pd

        waypoints_file = Path(analysis_dir) / f"waypoints_site_{site_id}.csv"

        if not waypoints_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Waypoints not found for site {site_id}"
            )

        waypoints_df = pd.read_csv(waypoints_file)

        # Convert to GeoJSON
        features = []
        coordinates = []

        for _, row in waypoints_df.iterrows():
            # Point feature for each waypoint
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row.get('lon', row.get('longitude')), row.get('lat', row.get('latitude'))]
                },
                "properties": {
                    "waypoint_id": int(row['waypoint_id']),
                    "x_meters": float(row.get('x_meters', row.get('x_site'))),
                    "y_meters": float(row.get('y_meters', row.get('y_site'))),
                    "tolerance_meters": float(row.get('tolerance_meters', row.get('tolerance_m')))
                }
            })
            coordinates.append([row.get('lon', row.get('longitude')), row.get('lat', row.get('latitude'))])

        # Add LineString for path
        if len(coordinates) > 1:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "path_type": "navigation_route",
                    "site_id": site_id
                }
            })

        return {
            "type": "FeatureCollection",
            "features": features
        }

    except Exception as e:
        logger.error("Failed to generate waypoints GeoJSON", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))





class RouteSummary(BaseModel):
    strategy: PathfindingStrategy
    waypoints: list[dict]
    num_waypoints: int
    total_distance_m: float
    relative_cost_percent: float


class MultiRouteRequest(BaseModel):
    site_id: int
    analysis_dir: str
    start_lat: float
    start_lon: float
    strategies: list[PathfindingStrategy] = Field(default_factory=lambda: [PathfindingStrategy.BALANCED, PathfindingStrategy.DIRECT])
    max_waypoint_spacing_m: float = Field(250.0, gt=0)
    max_slope_deg: float = Field(25.0, gt=0, le=90)
    task_id: Optional[str] = None


class MultiRouteResponse(BaseModel):
    routes: list[RouteSummary]
    site_id: int
    task_id: str


@router.post("/plan-routes", response_model=MultiRouteResponse)
async def plan_routes(request: MultiRouteRequest):
    try:
        task_id = request.task_id or generate_task_id()
        progress_tracker = ProgressTracker(task_id)
        def progress_callback(stage: str, progress: float, message: str):
            progress_tracker.update(stage, progress, message)
        engine = NavigationEngine()
        loop = asyncio.get_event_loop()
        routes_data = []
        from marshab.analysis.route_cost import RouteCostEngine
        cost_engine = RouteCostEngine()
        config = get_config()
        output_dir = config.paths.output_dir
        for strat in request.strategies:
            try:
                df = await asyncio.wait_for(
                    loop.run_in_executor(
                        _executor,
                        lambda current_strategy=strat: engine.plan_to_site(
                            site_id=request.site_id,
                            analysis_dir=Path(request.analysis_dir),
                            start_lat=request.start_lat,
                            start_lon=request.start_lon,
                            max_waypoint_spacing_m=request.max_waypoint_spacing_m,
                            max_slope_deg=request.max_slope_deg,
                            progress_callback=progress_callback,
                            strategy=current_strategy,
                        )
                    ),
                    timeout=180.0,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Navigation planning timed out")
            if len(df) > 0:
                last = df.iloc[-1]
                dist = float((last['x_meters']**2 + last['y_meters']**2)**0.5)
            else:
                dist = 0.0
            weights = {
                "distance": 1.0,
                "slope_penalty": 1.0,
                "roughness_penalty": 1.0,
                "elevation_penalty": 0.2,
            }
            breakdown = cost_engine.analyze_route(df, weights)
            routes_data.append({
                "strategy": strat,
                "waypoints": df.to_dict('records'),
                "num_waypoints": len(df),
                "total_distance_m": dist,
                "_cost": breakdown.total_cost,
            })
            try:
                fname = output_dir / f"waypoints_{request.site_id}_{strat.value}.csv"
                df.to_csv(fname, index=False)
            except Exception:
                pass
        if not routes_data:
            raise HTTPException(status_code=500, detail="No routes generated")
        min_cost = min(r["_cost"] for r in routes_data)
        for r in routes_data:
            r["relative_cost_percent"] = float((r["_cost"] / (min_cost if min_cost > 0 else 1.0)) * 100.0)
            del r["_cost"]
        return MultiRouteResponse(routes=routes_data, site_id=request.site_id, task_id=task_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in multi-route planning", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
