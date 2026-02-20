"""Enhanced route planning with constraint awareness and shadow calculations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

from marshab.core.raster_service import get_cell_size_from_dem
from marshab.processing.pathfinding import AStarPathfinder, smooth_path
from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Route:
    """Route with waypoints and metadata."""
    waypoints: list[tuple[int, int]]  # Pixel coordinates
    total_distance_m: float
    metadata: dict

    def __post_init__(self):
        """Ensure total_distance_m is set."""
        if self.total_distance_m == 0.0 and len(self.waypoints) > 1:
            # Calculate if not set
            pass  # Will be calculated by caller


@dataclass
class RouteCostResult:
    """Route cost breakdown."""
    distance_m: float
    slope_cost: float
    roughness_cost: float
    shadow_cost: float
    energy_estimate_j: float
    components: dict[str, float]


def calculate_shadow_penalty(
    elevation: np.ndarray,
    sun_azimuth: float,
    sun_altitude: float,
    cell_size_m: float
) -> np.ndarray:
    """Calculate shadow penalty based on sun position and terrain.

    Args:
        elevation: Elevation array (meters)
        sun_azimuth: Sun azimuth angle in degrees (0-360, 0=North)
        sun_altitude: Sun altitude angle in degrees (0-90, 0=horizon)
        cell_size_m: Cell size in meters

    Returns:
        Shadow penalty array (0 = no shadow, 1 = full shadow)
    """
    logger.info(
        "Calculating shadow penalty",
        sun_azimuth=sun_azimuth,
        sun_altitude=sun_altitude
    )

    if elevation.size == 0:
        return np.zeros_like(elevation)

    # Convert to radians
    azimuth_rad = np.deg2rad(sun_azimuth)
    altitude_rad = np.deg2rad(sun_altitude)

    # Calculate gradients (slope and aspect)
    dy, dx = np.gradient(elevation, cell_size_m)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)  # Aspect: 0=North, positive=east

    # Calculate sun direction vector
    sun_x = np.sin(azimuth_rad) * np.cos(altitude_rad)
    sun_y = np.cos(azimuth_rad) * np.cos(altitude_rad)
    sun_z = np.sin(altitude_rad)

    # Calculate surface normal
    # For each cell, compute if it's in shadow
    shadow_penalty = np.zeros_like(elevation, dtype=np.float32)

    # Simple shadow detection: check if slope faces away from sun
    # More sophisticated: ray casting from each cell toward sun
    # For now, use simplified approach based on aspect and slope

    # Calculate angle between surface normal and sun direction
    # Surface normal components (simplified)
    normal_x = -np.sin(aspect) * np.sin(slope)
    normal_y = -np.cos(aspect) * np.sin(slope)
    normal_z = np.cos(slope)

    # Dot product: negative means surface faces away from sun (in shadow)
    dot_product = normal_x * sun_x + normal_y * sun_y + normal_z * sun_z

    # Convert to shadow penalty (0-1)
    # Negative dot product = shadow, positive = illuminated
    shadow_penalty = np.clip(-dot_product, 0, 1)

    # Also check for terrain occlusion (simplified)
    # Cells with high elevation behind them (in sun direction) are in shadow
    # This is a simplified check - full implementation would use ray casting

    logger.info(
        "Shadow penalty calculated",
        mean_shadow=float(np.nanmean(shadow_penalty)),
        max_shadow=float(np.nanmax(shadow_penalty))
    )

    return shadow_penalty


def plan_route(
    start: tuple[float, float],
    end: tuple[float, float],
    weights: dict[str, float],
    dem: xr.DataArray,
    constraints: Optional[dict] = None,
    sun_azimuth: Optional[float] = None,
    sun_altitude: Optional[float] = None,
    cell_size_m: Optional[float] = None
) -> Route:
    """Plan route between two points with constraint awareness.

    Args:
        start: Start point (lat, lon) or (row, col) pixel coordinates
        end: End point (lat, lon) or (row, col) pixel coordinates
        weights: Route cost weights (distance, slope_penalty, roughness_penalty, shadow_penalty)
        dem: DEM DataArray
        constraints: Optional constraints (max_slope_deg, etc.)
        sun_azimuth: Optional sun azimuth for shadow calculation (degrees)
        sun_altitude: Optional sun altitude for shadow calculation (degrees)
        cell_size_m: Optional cell size in meters (for shadow calculation)

    Returns:
        Route with waypoints and metadata
    """
    logger.info("Planning route", start=start, end=end)

    # Get elevation data
    elevation = dem.values.astype(np.float32)

    # Calculate cell size if not provided
    if cell_size_m is None:
        cell_size_m = get_cell_size_from_dem(dem)

    # Analyze terrain
    terrain_analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
    metrics = terrain_analyzer.analyze(dem)

    # Generate base cost surface
    cost_surface = generate_cost_surface(
        slope=metrics.slope,
        roughness=metrics.roughness,
        max_slope_deg=constraints.get("max_slope_deg", 25.0) if constraints else 25.0,
        max_roughness=constraints.get("max_roughness", 20.0) if constraints else 20.0,
        slope_weight=weights.get("slope_penalty", 10.0),
        roughness_weight=weights.get("roughness_penalty", 5.0),
        elevation=elevation,
        cell_size_m=cell_size_m,
        cliff_threshold_m=constraints.get("cliff_threshold_m") if constraints else None,
    )

    # Add shadow penalty if sun position provided
    if sun_azimuth is not None and sun_altitude is not None:
        shadow_penalty = calculate_shadow_penalty(
            elevation,
            sun_azimuth,
            sun_altitude,
            cell_size_m
        )

        # Add shadow cost to cost surface
        shadow_weight = weights.get("shadow_penalty", 5.0)
        cost_surface = cost_surface + shadow_penalty * shadow_weight

    # Convert start/end to pixel coordinates if they're lat/lon
    # For now, assume they're already pixel coordinates (row, col)
    if isinstance(start[0], float) and -90 <= start[0] <= 90:
        # Likely lat/lon, need to convert
        # This would require coordinate transformation
        # For now, assume pixel coordinates
        logger.warning("Lat/lon coordinates detected but conversion not implemented, assuming pixel coordinates")

    start_pixel = (int(start[0]), int(start[1]))
    end_pixel = (int(end[0]), int(end[1]))

    # Validate pixel coordinates
    height, width = cost_surface.shape
    if not (0 <= start_pixel[0] < height and 0 <= start_pixel[1] < width):
        raise ValueError(f"Start pixel out of bounds: {start_pixel}")
    if not (0 <= end_pixel[0] < height and 0 <= end_pixel[1] < width):
        raise ValueError(f"End pixel out of bounds: {end_pixel}")

    # Run A* pathfinding
    pathfinder = AStarPathfinder(cost_surface, cell_size_m=cell_size_m)
    path = pathfinder.find_path(start_pixel, end_pixel)

    if not path:
        raise ValueError("No path found between start and end points")

    # Smooth path if enabled
    if constraints and constraints.get("enable_smoothing", True):
        smoothing_tolerance = constraints.get("smoothing_tolerance", 2.0)
        path = smooth_path(path, cost_surface, tolerance=smoothing_tolerance)

    # Calculate total distance
    total_distance = 0.0
    for i in range(len(path) - 1):
        dx = (path[i+1][1] - path[i][1]) * cell_size_m
        dy = (path[i+1][0] - path[i][0]) * cell_size_m
        total_distance += np.sqrt(dx**2 + dy**2)

    metadata = {
        "num_waypoints": len(path),
        "cell_size_m": cell_size_m,
        "sun_azimuth": sun_azimuth,
        "sun_altitude": sun_altitude,
    }

    logger.info(
        "Route planned",
        num_waypoints=len(path),
        total_distance_m=total_distance
    )

    return Route(
        waypoints=path,
        total_distance_m=total_distance,
        metadata=metadata
    )


def compute_route_cost(
    route: Route,
    dem: xr.DataArray,
    weights: dict[str, float],
    sun_azimuth: Optional[float] = None,
    sun_altitude: Optional[float] = None
) -> RouteCostResult:
    """Compute detailed cost breakdown for a route.

    Args:
        route: Route with waypoints
        dem: DEM DataArray
        weights: Cost component weights
        sun_azimuth: Optional sun azimuth for shadow calculation
        sun_altitude: Optional sun altitude for shadow calculation

    Returns:
        RouteCostResult with cost breakdown
    """
    logger.info("Computing route cost breakdown")

    elevation = dem.values.astype(np.float32)

    # Get cell size
    cell_size_m = get_cell_size_from_dem(dem)

    # Analyze terrain along route
    terrain_analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
    metrics = terrain_analyzer.analyze(dem)

    # Extract values along route
    route_slopes = []
    route_roughness = []
    route_elevations = []

    for waypoint in route.waypoints:
        row, col = waypoint
        if 0 <= row < metrics.slope.shape[0] and 0 <= col < metrics.slope.shape[1]:
            route_slopes.append(metrics.slope[row, col])
            route_roughness.append(metrics.roughness[row, col])
            route_elevations.append(elevation[row, col])

    # Calculate costs
    mean_slope = np.nanmean(route_slopes) if route_slopes else 0.0
    mean_roughness = np.nanmean(route_roughness) if route_roughness else 0.0

    slope_cost = mean_slope * weights.get("slope_penalty", 10.0)
    roughness_cost = mean_roughness * weights.get("roughness_penalty", 5.0)

    # Shadow cost
    shadow_cost = 0.0
    if sun_azimuth is not None and sun_altitude is not None:
        shadow_penalty = calculate_shadow_penalty(
            elevation,
            sun_azimuth,
            sun_altitude,
            cell_size_m
        )

        route_shadows = []
        for waypoint in route.waypoints:
            row, col = waypoint
            if 0 <= row < shadow_penalty.shape[0] and 0 <= col < shadow_penalty.shape[1]:
                route_shadows.append(shadow_penalty[row, col])

        mean_shadow = np.nanmean(route_shadows) if route_shadows else 0.0
        shadow_cost = mean_shadow * weights.get("shadow_penalty", 5.0)

    # Energy estimate (simplified: distance * base_energy + terrain_penalties)
    base_energy_per_m = 100.0  # Joules per meter
    energy_estimate = (
        route.total_distance_m * base_energy_per_m +
        slope_cost * 10.0 +  # Additional energy for slopes
        roughness_cost * 5.0  # Additional energy for roughness
    )

    components = {
        "distance": route.total_distance_m * weights.get("distance", 1.0),
        "slope": slope_cost,
        "roughness": roughness_cost,
        "shadow": shadow_cost,
    }

    logger.info(
        "Route cost computed",
        total_distance_m=route.total_distance_m,
        energy_estimate_j=energy_estimate
    )

    return RouteCostResult(
        distance_m=route.total_distance_m,
        slope_cost=slope_cost,
        roughness_cost=roughness_cost,
        shadow_cost=shadow_cost,
        energy_estimate_j=energy_estimate,
        components=components
    )

