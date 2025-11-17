"""Navigation engine for rover path planning."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import rasterio

from marshab.config import get_config
from marshab.core.data_manager import DataManager
from marshab.exceptions import NavigationError
from marshab.processing.coordinates import CoordinateTransformer
from marshab.processing.pathfinding import AStarPathfinder, smooth_path
from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface
from marshab.types import SiteOrigin, Waypoint
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class NavigationEngine:
    """Generates rover navigation commands and waypoints."""

    def __init__(self):
        """Initialize navigation engine."""
        self.data_manager = DataManager()
        self.config = get_config()
        self.coord_transformer = CoordinateTransformer(
            equatorial_radius=self.config.mars.equatorial_radius_m,
            polar_radius=self.config.mars.polar_radius_m
        )
        logger.info("Initialized NavigationEngine")

    def plan_to_site(
        self,
        site_id: int,
        analysis_dir: Path,
        start_lat: float,
        start_lon: float,
    ) -> pd.DataFrame:
        """Plan navigation path to target site.

        Args:
            site_id: Target site ID
            analysis_dir: Directory containing analysis results
            start_lat: Starting latitude (degrees)
            start_lon: Starting longitude (degrees)

        Returns:
            DataFrame with waypoints (columns: waypoint_id, x_meters, y_meters, tolerance_meters)

        Raises:
            NavigationError: If path planning fails
        """
        logger.info(
            "Planning navigation to site",
            site_id=site_id,
            analysis_dir=str(analysis_dir),
            start_lat=start_lat,
            start_lon=start_lon,
        )

        try:
            # 1. Load site location from analysis results
            sites_csv = analysis_dir / "sites.csv"
            if not sites_csv.exists():
                raise NavigationError(
                    f"Sites file not found: {sites_csv}",
                    details={"analysis_dir": str(analysis_dir)}
                )
            
            sites_df = pd.read_csv(sites_csv)
            site_row = sites_df[sites_df["site_id"] == site_id]
            
            if site_row.empty:
                raise NavigationError(
                    f"Site {site_id} not found in analysis results",
                    details={"available_sites": sites_df["site_id"].tolist()}
                )
            
            site_lat = float(site_row["lat"].iloc[0])
            site_lon = float(site_row["lon"].iloc[0])
            site_elevation = float(site_row["mean_elevation_m"].iloc[0])
            
            logger.info(
                "Loaded site location",
                site_id=site_id,
                lat=site_lat,
                lon=site_lon,
                elevation=site_elevation
            )
            
            # Get start elevation from DEM or use site elevation as approximation
            # For now, use a simple approach: load DEM to get start elevation
            # We need to determine which DEM was used - try to infer from cache
            # For simplicity, assume mola dataset and approximate ROI from site location
            from marshab.types import BoundingBox
            # Create a small ROI around start and site for DEM loading
            roi_size = 0.1  # degrees
            roi = BoundingBox(
                lat_min=min(start_lat, site_lat) - roi_size,
                lat_max=max(start_lat, site_lat) + roi_size,
                lon_min=min(start_lon, site_lon) - roi_size,
                lon_max=max(start_lon, site_lon) + roi_size
            )
            
            # Load DEM for the region (allow download if not cached)
            dem = self.data_manager.get_dem_for_roi(roi, dataset="mola", download=True, clip=True)
            
            # Get cell size
            if "mola" in self.config.data_sources:
                cell_size_m = self.config.data_sources["mola"].resolution_m
            else:
                cell_size_m = 463.0  # Default MOLA resolution
            
            # Get start elevation from DEM (interpolate if needed)
            # For simplicity, use mean elevation in small region around start
            if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
                bounds = dem.rio.bounds()
                # Find pixel coordinates for start location
                start_x_px = int((start_lon - bounds.left) / (bounds.right - bounds.left) * dem.shape[1])
                start_y_px = int((bounds.top - start_lat) / (bounds.top - bounds.bottom) * dem.shape[0])
                start_x_px = max(0, min(dem.shape[1] - 1, start_x_px))
                start_y_px = max(0, min(dem.shape[0] - 1, start_y_px))
                start_elevation = float(dem.values[start_y_px, start_x_px])
            else:
                start_elevation = site_elevation  # Fallback
            
            # 2. Transform coordinates to SITE frame
            site_origin = SiteOrigin(
                lat=start_lat,
                lon=start_lon,
                elevation_m=start_elevation
            )
            
            # Transform site location to SITE frame
            site_x, site_y, site_z = self.coord_transformer.iau_mars_to_site_frame(
                site_lat, site_lon, site_elevation, site_origin
            )
            
            logger.info(
                "Transformed to SITE frame",
                site_x_m=site_x,
                site_y_m=site_y,
                site_z_m=site_z
            )
            
            # 3. Generate cost surface from terrain
            logger.info("Calculating terrain metrics for cost surface")
            terrain_analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
            metrics = terrain_analyzer.analyze(dem)
            
            # Get navigation config and strategy weights
            nav_config = self.config.navigation if hasattr(self.config, 'navigation') else None
            if nav_config is None:
                # Fallback to defaults
                slope_weight = 10.0
                roughness_weight = 5.0
                enable_cliff_detection = False
                cliff_threshold_m = None
                enable_smoothing = False
                smoothing_tolerance = 2.0
                strategy_name = "balanced"
            else:
                weights = nav_config.get_weights_for_strategy()
                slope_weight = weights["slope_weight"]
                roughness_weight = weights["roughness_weight"]
                enable_cliff_detection = True
                cliff_threshold_m = nav_config.cliff_threshold_m
                enable_smoothing = nav_config.enable_smoothing
                smoothing_tolerance = nav_config.smoothing_tolerance
                strategy_name = nav_config.strategy.value if hasattr(nav_config.strategy, 'value') else str(nav_config.strategy)
            
            logger.info(
                "Using pathfinding strategy",
                strategy=strategy_name,
                slope_weight=slope_weight,
                roughness_weight=roughness_weight,
                cliff_detection=enable_cliff_detection,
                path_smoothing=enable_smoothing
            )
            
            # Get elevation array for cliff detection
            elevation_array = dem.values.astype(np.float32) if enable_cliff_detection else None
            
            # Generate cost surface with configurable weights and cliff detection
            cost_surface = generate_cost_surface(
                metrics.slope,
                metrics.roughness,
                max_slope_deg=self.config.analysis.max_slope_deg if hasattr(self.config, 'analysis') else 25.0,
                slope_weight=slope_weight,
                roughness_weight=roughness_weight,
                elevation=elevation_array,
                cell_size_m=cell_size_m if enable_cliff_detection else None,
                cliff_threshold_m=cliff_threshold_m
            )
            
            logger.info(
                "Generated cost surface",
                shape=cost_surface.shape,
                passable_fraction=float(np.sum(np.isfinite(cost_surface)) / cost_surface.size)
            )
            
            # 4. Run A* pathfinding
            # Use the actual terrain cost surface for pathfinding
            # The cost surface is in DEM pixel coordinates, so we need to work in that space
            # and transform the resulting path to SITE frame
            
            # Find pixel coordinates for start and goal in DEM
            # Try rio accessor first, fall back to coordinate arrays
            if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
                try:
                    bounds = dem.rio.bounds()
                    bounds_left = bounds.left
                    bounds_right = bounds.right
                    bounds_bottom = bounds.bottom
                    bounds_top = bounds.top
                except (AttributeError, RuntimeError):
                    bounds = None
            else:
                bounds = None
            
            # Fallback: use ROI bounds directly
            # Since the DEM was windowed to the ROI, the bounds should match
            # (coordinate arrays might be in CRS units like meters, not degrees)
            if bounds is None:
                bounds_left = roi.lon_min
                bounds_right = roi.lon_max
                bounds_bottom = roi.lat_min
                bounds_top = roi.lat_max
            
            logger.debug(
                "DEM bounds for pixel coordinate calculation",
                bounds_left=bounds_left,
                bounds_right=bounds_right,
                bounds_bottom=bounds_bottom,
                bounds_top=bounds_top,
                dem_shape=dem.shape,
                start_lat=start_lat,
                start_lon=start_lon,
                site_lat=site_lat,
                site_lon=site_lon
            )
            
            # Start position in pixel coordinates
            start_x_px = int((start_lon - bounds_left) / (bounds_right - bounds_left) * dem.shape[1])
            start_y_px = int((bounds_top - start_lat) / (bounds_top - bounds_bottom) * dem.shape[0])
            # Goal position in pixel coordinates
            goal_x_px = int((site_lon - bounds_left) / (bounds_right - bounds_left) * dem.shape[1])
            goal_y_px = int((bounds_top - site_lat) / (bounds_top - bounds_bottom) * dem.shape[0])
            
            logger.debug(
                "Pixel coordinates before clamping",
                start_x_px=start_x_px,
                start_y_px=start_y_px,
                goal_x_px=goal_x_px,
                goal_y_px=goal_y_px
            )
            
            # Clamp to valid bounds
            start_x_px = max(0, min(dem.shape[1] - 1, start_x_px))
            start_y_px = max(0, min(dem.shape[0] - 1, start_y_px))
            goal_x_px = max(0, min(dem.shape[1] - 1, goal_x_px))
            goal_y_px = max(0, min(dem.shape[0] - 1, goal_y_px))
            
            logger.debug(
                "Pixel coordinates after clamping",
                start_x_px=start_x_px,
                start_y_px=start_y_px,
                goal_x_px=goal_x_px,
                goal_y_px=goal_y_px,
                pixel_distance_x=abs(goal_x_px - start_x_px),
                pixel_distance_y=abs(goal_y_px - start_y_px)
            )
            
            start_idx = (start_y_px, start_x_px)
            goal_idx = (goal_y_px, goal_x_px)
            
            logger.info(
                "Running A* pathfinding",
                start=start_idx,
                goal=goal_idx,
                cost_surface_shape=cost_surface.shape
            )
            
            # Use actual terrain cost surface for pathfinding
            pathfinder = AStarPathfinder(cost_surface, cell_size_m=cell_size_m)
            
            # First find the full path
            full_path = pathfinder.find_path(start_idx, goal_idx)
            
            if full_path is None:
                logger.warning("No path found, returning direct path")
                # Return direct path if A* fails
                full_path = [start_idx, goal_idx]
            
            # Apply path smoothing if enabled
            if enable_smoothing and len(full_path) > 2:
                logger.info("Applying path smoothing", original_length=len(full_path))
                full_path = smooth_path(full_path, cost_surface, tolerance=smoothing_tolerance)
            
            # Downsample to waypoints (after smoothing)
            waypoint_indices = [full_path[0]]  # Always include start
            max_waypoint_spacing = 50
            for i in range(max_waypoint_spacing, len(full_path), max_waypoint_spacing):
                waypoint_indices.append(full_path[i])
            
            # Always include goal
            if waypoint_indices[-1] != full_path[-1]:
                waypoint_indices.append(full_path[-1])
            
            # Ensure waypoint_indices is a list of tuples
            if not isinstance(waypoint_indices, list):
                logger.warning(f"waypoint_indices is not a list: {type(waypoint_indices)}, converting")
                waypoint_indices = [waypoint_indices] if isinstance(waypoint_indices, (tuple, list)) else [start_idx, goal_idx]
            
            logger.debug(
                "Processing waypoints",
                num_waypoints=len(waypoint_indices),
                first_waypoint=waypoint_indices[0] if waypoint_indices else None,
                waypoint_types=[type(w).__name__ for w in waypoint_indices[:3]]
            )
            
            # 5. Generate waypoints from path
            # Convert pixel coordinates to lat/lon, then to SITE frame
            waypoints_data = {
                "waypoint_id": [],
                "lat": [],
                "lon": [],
                "x_meters": [],
                "y_meters": [],
                "tolerance_meters": [],
            }
            
            for i, waypoint in enumerate(waypoint_indices):
                # Ensure waypoint is a tuple (row, col)
                if isinstance(waypoint, (tuple, list)) and len(waypoint) >= 2:
                    row, col = int(waypoint[0]), int(waypoint[1])
                else:
                    logger.warning(f"Invalid waypoint format: {waypoint}, skipping")
                    continue
                
                # Convert pixel coordinates to lat/lon
                # Use ROI bounds since DEM was windowed to ROI
                lon = bounds_left + (col / dem.shape[1]) * (bounds_right - bounds_left)
                lat = bounds_top - (row / dem.shape[0]) * (bounds_top - bounds_bottom)
                
                # Get elevation at this point
                row_clamped = max(0, min(dem.shape[0] - 1, row))
                col_clamped = max(0, min(dem.shape[1] - 1, col))
                elev = float(dem.values[row_clamped, col_clamped])
                
                # Transform to SITE frame
                x_m, y_m, z_m = self.coord_transformer.iau_mars_to_site_frame(
                    lat, lon, elev, site_origin
                )
                
                waypoints_data["waypoint_id"].append(i + 1)
                waypoints_data["lat"].append(float(lat))
                waypoints_data["lon"].append(float(lon))
                waypoints_data["x_meters"].append(float(x_m))
                waypoints_data["y_meters"].append(float(y_m))
                waypoints_data["tolerance_meters"].append(cell_size_m * 2.0)  # 2 cell tolerance
            
            waypoints_df = pd.DataFrame(waypoints_data)
            
            logger.info(
                "Navigation planning complete",
                num_waypoints=len(waypoints_df),
                total_distance_m=float(np.sqrt(site_x**2 + site_y**2))
            )
            
            return waypoints_df

        except Exception as e:
            raise NavigationError(
                "Navigation planning failed",
                details={"site_id": site_id, "error": str(e)},
            )


