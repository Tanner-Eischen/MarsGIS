from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Callable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist

from marshab.config import get_config, PathfindingStrategy
from marshab.core.data_manager import DataManager
from marshab.exceptions import NavigationError
from marshab.processing.coordinates import CoordinateTransformer
from marshab.processing.pathfinding import AStarPathfinder, smooth_path
from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface
from marshab.models import SiteOrigin, Waypoint, SiteCandidate
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
    
    def _load_analysis_results(self, analysis_dir: Path) -> Dict[str, Any]:
        """Load terrain analysis results from directory.
        
        Args:
            analysis_dir: Directory containing analysis outputs
            
        Returns:
            Dictionary with dem, metrics, sites, suitability, criteria
            
        Raises:
            NavigationError: If analysis results not found
        """
        import pickle
        
        results_file = analysis_dir / "analysis_results.pkl"
        if not results_file.exists():
            raise NavigationError(
                f"Analysis results not found: {results_file}",
                details={"analysis_dir": str(analysis_dir)}
            )
        
        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            logger.info("Loaded analysis results from pickle file")
            return results
        except Exception as e:
            raise NavigationError(
                f"Failed to load analysis results: {e}",
                details={"results_file": str(results_file)}
            )
    
    def _get_site_coordinates(
        self, 
        site_id: int,
        sites: list[SiteCandidate]
    ) -> Tuple[float, float, float]:
        """Get coordinates of site by ID.
        
        Args:
            site_id: Site identifier
            sites: List of Site objects
            
        Returns:
            (latitude, longitude, elevation) tuple
            
        Raises:
            NavigationError: If site not found
        """
        for site in sites:
            if site.site_id == site_id:
                return (site.lat, site.lon, site.mean_elevation_m)
        
        raise NavigationError(
            f"Site {site_id} not found",
            details={"available_sites": [s.site_id for s in sites]}
        )
    
    def _latlon_to_pixel(
        self,
        lat: float,
        lon: float,
        dem: xr.DataArray,
        roi: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[int, int]:
        """Convert lat/lon to pixel coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            dem: DEM DataArray with coordinate info
            
        Returns:
            (row, col) pixel coordinates
        """
        # Try rio accessor first
        if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
            try:
                bounds = dem.rio.bounds()
                bounds_left = bounds.left
                bounds_right = bounds.right
                bounds_bottom = bounds.bottom
                bounds_top = bounds.top
                
                # Convert to pixel coordinates
                col = int((lon - bounds_left) / (bounds_right - bounds_left) * dem.shape[1])
                row = int((bounds_top - lat) / (bounds_top - bounds_bottom) * dem.shape[0])
                
                # Clamp to valid bounds
                row = max(0, min(dem.shape[0] - 1, row))
                col = max(0, min(dem.shape[1] - 1, col))
                
                return int(row), int(col)
            except (AttributeError, RuntimeError):
                pass
        
        # Fallback: use coordinate arrays
        try:
            lat_values = dem.coords.get('lat', None)
            lon_values = dem.coords.get('lon', None)
            
            if lat_values is None or lon_values is None:
                raise NavigationError("DEM missing coordinate information")
            
            lat_vals = lat_values.values
            lon_vals = lon_values.values
            
            # Handle 2D coordinate arrays
            if lat_vals.ndim == 2:
                # Find minimum distance
                dist = (lat_vals - lat)**2 + (lon_vals - lon)**2
                row, col = np.unravel_index(np.argmin(dist), lat_vals.shape)
            else:
                # 1D coordinate arrays
                row = int(np.argmin(np.abs(lat_vals - lat)))
                col = int(np.argmin(np.abs(lon_vals - lon)))
            
            return int(row), int(col)
        except Exception as e:
            raise NavigationError(
                f"Failed to convert lat/lon to pixel: {e}",
                details={"lat": lat, "lon": lon}
            )
    
    def _pixel_to_latlon(
        self,
        row: int,
        col: int,
        dem: xr.DataArray
    ) -> Tuple[float, float]:
        """Convert pixel coordinates to lat/lon.
        
        Args:
            row: Pixel row
            col: Pixel column
            dem: DEM DataArray
            
        Returns:
            (latitude, longitude) tuple
        """
        # Try rio accessor first
        if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
            try:
                bounds = dem.rio.bounds()
                bounds_left = bounds.left
                bounds_right = bounds.right
                bounds_bottom = bounds.bottom
                bounds_top = bounds.top
                
                # Convert pixel to lat/lon
                lon = bounds_left + (col / dem.shape[1]) * (bounds_right - bounds_left)
                lat = bounds_top - (row / dem.shape[0]) * (bounds_top - bounds_bottom)
                
                return float(lat), float(lon)
            except (AttributeError, RuntimeError):
                pass
        
        # Fallback: use coordinate arrays
        try:
            lat_values = dem.coords.get('lat', None)
            lon_values = dem.coords.get('lon', None)
            
            if lat_values is None or lon_values is None:
                raise NavigationError("DEM missing coordinate information")
            
            lat_vals = lat_values.values
            lon_vals = lon_values.values
            
            if lat_vals.ndim == 2:
                lat = float(lat_vals[row, col])
                lon = float(lon_vals[row, col])
            else:
                lat = float(lat_vals[row])
                lon = float(lon_vals[col])
            
            return lat, lon
        except Exception as e:
            raise NavigationError(
                f"Failed to convert pixel to lat/lon: {e}",
                details={"row": row, "col": col}
            )

    def plan_to_site(
        self,
        site_id: int,
        analysis_dir: Path,
        start_lat: float,
        start_lon: float,
        max_waypoint_spacing_m: float = 100.0,
        max_slope_deg: float = 25.0,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        strategy: Optional[PathfindingStrategy] = None
    ) -> pd.DataFrame:
        """Plan navigation route to target site.

        Args:
            site_id: Target site ID
            analysis_dir: Directory with analysis results
            start_lat: Starting latitude
            start_lon: Starting longitude
            max_waypoint_spacing_m: Maximum spacing between waypoints
            max_slope_deg: Maximum traversable slope

        Returns:
            DataFrame with waypoint columns: waypoint_id, x_site, y_site, z_site, tolerance_m
        """
        logger.info(
            "Planning route to site",
            site_id=site_id,
            start_lat=start_lat,
            start_lon=start_lon
        )

        try:
            if progress_callback:
                progress_callback("cost_map_generation", 0.0, "Loading analysis results...")
            
            # Try to load from pickle file first
            try:
                results = self._load_analysis_results(analysis_dir)
                dem = results['dem']
                metrics = results['metrics']
                sites = results['sites']
                logger.info("Using saved analysis results")
            except NavigationError:
                # Fallback to direct DEM loading (backward compatibility)
                logger.warning("Pickle file not found, falling back to direct DEM loading")
                from marshab.models import BoundingBox
                
                # Load site location from CSV
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
            
            # Create ROI and load DEM
            roi_size = 0.1
            roi = BoundingBox(
                lat_min=min(start_lat, site_lat) - roi_size,
                lat_max=max(start_lat, site_lat) + roi_size,
                lon_min=min(start_lon, site_lon) - roi_size,
                lon_max=max(start_lon, site_lon) + roi_size
            )
            
            dem = self.data_manager.get_dem_for_roi(roi, dataset="mola", download=True, clip=True)
            
            # Calculate metrics
            if "mola" in self.config.data_sources:
                cell_size_m = self.config.data_sources["mola"].resolution_m
            else:
                cell_size_m = 463.0
                
            terrain_analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
            metrics = terrain_analyzer.analyze(dem)
            
            # Convert sites from DataFrame to SiteCandidate objects
            sites = []
            for _, row in sites_df.iterrows():
                from marshab.models import SiteCandidate
                site = SiteCandidate(
                    site_id=int(row["site_id"]),
                    geometry_type=row.get("geometry_type", "POINT"),
                    area_km2=float(row["area_km2"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    mean_slope_deg=float(row["mean_slope_deg"]),
                    mean_roughness=float(row["mean_roughness"]),
                    mean_elevation_m=float(row["mean_elevation_m"]),
                    suitability_score=float(row["suitability_score"]),
                    rank=int(row["rank"])
                )
                sites.append(site)
            
            # Get goal site coordinates
            goal_lat, goal_lon, goal_elevation = self._get_site_coordinates(site_id, sites)
            
            logger.info(
                "Route parameters",
                goal_lat=goal_lat,
                goal_lon=goal_lon
            )
            
            # Get cell size from DEM
            if "mola" in self.config.data_sources:
                cell_size_m = self.config.data_sources["mola"].resolution_m
            else:
                cell_size_m = float(dem.attrs.get('resolution_m', 463.0))
            
            # Convert start/goal to pixel coordinates
            start_pixel = self._latlon_to_pixel(start_lat, start_lon, dem)
            goal_pixel = self._latlon_to_pixel(goal_lat, goal_lon, dem)
            
            logger.info(
                "Pixel coordinates",
                start_pixel=start_pixel,
                goal_pixel=goal_pixel,
                dem_shape=dem.shape
            )
            
            # Validate pixel coordinates are in bounds
            if not (0 <= start_pixel[0] < dem.shape[0] and 0 <= start_pixel[1] < dem.shape[1]):
                raise NavigationError(
                    f"Start position ({start_lat}, {start_lon}) is outside DEM bounds",
                    details={"start_pixel": start_pixel, "dem_shape": dem.shape}
                )
            if not (0 <= goal_pixel[0] < dem.shape[0] and 0 <= goal_pixel[1] < dem.shape[1]):
                raise NavigationError(
                    f"Goal position ({goal_lat}, {goal_lon}) is outside DEM bounds",
                    details={"goal_pixel": goal_pixel, "dem_shape": dem.shape}
                )
            
            # Get start elevation from DEM
            start_elevation = float(metrics.elevation[start_pixel[0], start_pixel[1]])
            
            # Check if start position has valid terrain data
            start_slope = metrics.slope[start_pixel[0], start_pixel[1]]
            start_roughness = metrics.roughness[start_pixel[0], start_pixel[1]]
            
            if np.isnan(start_slope) or np.isnan(start_roughness):
                logger.warning(
                    "Start position has NaN terrain values, attempting to use nearest valid cell",
                    start_pixel=start_pixel
                )
                # Try to find nearest valid cell
                valid_mask = np.isfinite(metrics.slope) & np.isfinite(metrics.roughness)
                if not np.any(valid_mask):
                    raise NavigationError(
                        "No valid terrain data in DEM for pathfinding",
                        details={"dem_shape": dem.shape}
                    )
                # Find nearest valid cell
                from scipy.spatial.distance import cdist
                valid_coords = np.argwhere(valid_mask)
                if len(valid_coords) > 0:
                    distances = cdist([start_pixel], valid_coords)
                    nearest_idx = np.argmin(distances)
                    start_pixel = tuple(valid_coords[nearest_idx])
                    start_elevation = float(metrics.elevation[start_pixel[0], start_pixel[1]])
                    logger.info("Adjusted start position to nearest valid cell", new_start_pixel=start_pixel)
            
            # Generate cost surface with navigation config
            if progress_callback:
                progress_callback("cost_map_generation", 0.3, "Generating cost map...")
            nav_config = self.config.navigation
            weights = {
                "slope_weight": nav_config.slope_weight,
                "roughness_weight": nav_config.roughness_weight,
            }
            if strategy is not None:
                if strategy == PathfindingStrategy.SAFEST:
                    weights = {"slope_weight": 50.0, "roughness_weight": 30.0}
                elif strategy == PathfindingStrategy.BALANCED:
                    weights = {"slope_weight": 10.0, "roughness_weight": 5.0}
                elif strategy == PathfindingStrategy.DIRECT:
                    weights = {"slope_weight": 2.0, "roughness_weight": 1.0}
            cost_map = generate_cost_surface(
                metrics.slope,
                metrics.roughness,
                max_slope_deg=max_slope_deg,
                max_roughness=nav_config.max_roughness_m,  # Use configurable max roughness
                elevation=metrics.elevation,
                elevation_penalty_factor=0.1,
                slope_weight=weights["slope_weight"],
                roughness_weight=weights["roughness_weight"],
                cell_size_m=cell_size_m,
                cliff_threshold_m=nav_config.cliff_threshold_m
            )
            if progress_callback:
                progress_callback("cost_map_generation", 0.3, "Cost map generated")
            
            # Validate start/goal are passable before pathfinding
            start_cost = cost_map[start_pixel[0], start_pixel[1]]
            goal_cost = cost_map[goal_pixel[0], goal_pixel[1]]
            
            # Reuse already computed values
            start_slope_val = start_slope
            start_roughness_val = start_roughness
            start_elevation_val = start_elevation
            
            logger.info(
                "Cost validation",
                start_cost=float(start_cost) if np.isfinite(start_cost) else "inf",
                goal_cost=float(goal_cost) if np.isfinite(goal_cost) else "inf",
                start_slope=float(start_slope_val) if np.isfinite(start_slope_val) else "nan",
                start_roughness=float(start_roughness_val) if np.isfinite(start_roughness_val) else "nan",
                max_slope_deg=max_slope_deg,
                max_roughness_m=nav_config.max_roughness_m
            )
            
            if np.isinf(start_cost):
                # Check if start is at DEM edge - edge pixels often have artifacts
                original_start_pixel = start_pixel
                is_edge_pixel = (
                    start_pixel[0] == 0 or start_pixel[0] == cost_map.shape[0] - 1 or
                    start_pixel[1] == 0 or start_pixel[1] == cost_map.shape[1] - 1
                )
                
                # If at edge and only marked impassable due to cliff, try to find nearby passable pixel
                if is_edge_pixel and cell_size_m is not None and nav_config.cliff_threshold_m is not None:
                    from marshab.processing.terrain import detect_cliffs
                    cliff_mask = detect_cliffs(metrics.elevation, cell_size_m, nav_config.cliff_threshold_m)
                    is_cliff = cliff_mask[start_pixel[0], start_pixel[1]]
                    
                    if is_cliff:
                        # Try to find a nearby passable pixel within 3 cells
                        search_radius = 3
                        found_passable = False
                        best_pixel = start_pixel
                        
                        for di in range(-search_radius, search_radius + 1):
                            for dj in range(-search_radius, search_radius + 1):
                                ni, nj = start_pixel[0] + di, start_pixel[1] + dj
                                if (0 <= ni < cost_map.shape[0] and 0 <= nj < cost_map.shape[1] and
                                    np.isfinite(cost_map[ni, nj]) and not cliff_mask[ni, nj]):
                                    # Found a passable pixel nearby
                                    best_pixel = (ni, nj)
                                    found_passable = True
                                    logger.info(
                                        "Found nearby passable pixel for edge start position",
                                        original_pixel=original_start_pixel,
                                        new_pixel=best_pixel,
                                        distance_cells=int(np.sqrt(di**2 + dj**2))
                                    )
                                    break
                            if found_passable:
                                break
                        
                        if found_passable:
                            # Update start pixel to the passable one
                            start_pixel = best_pixel
                            start_cost = cost_map[start_pixel[0], start_pixel[1]]
                            start_slope_val = metrics.slope[start_pixel[0], start_pixel[1]]                                                                             
                            start_roughness_val = metrics.roughness[start_pixel[0], start_pixel[1]]
                            start_elevation_val = metrics.elevation[start_pixel[0], start_pixel[1]]
                            logger.info(
                                "Adjusted start position from edge cliff to nearby passable pixel",
                                original_pixel=original_start_pixel,
                                new_pixel=start_pixel,
                                new_cost=float(start_cost) if np.isfinite(start_cost) else "inf"
                            )
                            # Continue with the adjusted start position
                        else:
                            # No passable pixel found nearby, fall through to error
                            pass
                
                # If still impassable after edge adjustment, provide detailed error
                if np.isinf(start_cost):
                    # Provide more detailed error - check all possible reasons
                    reasons = []
                    
                    # Check for NaN values
                    if np.isnan(start_slope_val):
                        reasons.append("NaN slope")
                    if np.isnan(start_roughness_val):
                        reasons.append("NaN roughness")
                    if np.isnan(start_elevation_val):
                        reasons.append("NaN elevation")
                    
                    # Check slope threshold
                    if not np.isnan(start_slope_val) and start_slope_val > max_slope_deg:
                        reasons.append(f"slope too steep ({start_slope_val:.1f}° > {max_slope_deg}°)")
                    
                    # Check roughness threshold
                    if not np.isnan(start_roughness_val) and start_roughness_val > nav_config.max_roughness_m:
                        reasons.append(f"roughness too high ({start_roughness_val:.2f}m > {nav_config.max_roughness_m}m)")
                    
                    # Check if it's a cliff (elevation change)
                    if cell_size_m is not None and nav_config.cliff_threshold_m is not None:
                        from marshab.processing.terrain import detect_cliffs
                        cliff_mask = detect_cliffs(metrics.elevation, cell_size_m, nav_config.cliff_threshold_m)
                        if cliff_mask[start_pixel[0], start_pixel[1]]:
                            # Get neighbor elevations to show why it's a cliff
                            neighbors = []
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = start_pixel[0] + di, start_pixel[1] + dj
                                    if 0 <= ni < metrics.elevation.shape[0] and 0 <= nj < metrics.elevation.shape[1]:
                                        elev_diff = abs(metrics.elevation[ni, nj] - start_elevation_val)
                                        if elev_diff > nav_config.cliff_threshold_m:
                                            neighbors.append(f"neighbor({ni},{nj}): {elev_diff:.1f}m")
                            edge_note = " (at DEM edge)" if is_edge_pixel else ""
                            reasons.append(f"cliff detected (elevation change > {nav_config.cliff_threshold_m}m){edge_note}" + (f" [{', '.join(neighbors[:3])}]" if neighbors else ""))
                    
                    # If no specific reason found, check the actual cost components
                    if not reasons:
                        reasons.append(f"cost surface marked as impassable (cost={start_cost}, slope={start_slope_val:.2f}°, roughness={start_roughness_val:.2f}m, elevation={start_elevation_val:.1f}m)")
                    
                    raise NavigationError(
                        f"Start position is impassable: {', '.join(reasons) if reasons else 'unknown reason'}",
                        details={
                            "start_lat": start_lat,
                            "start_lon": start_lon,
                            "start_pixel": start_pixel,
                            "is_edge_pixel": is_edge_pixel,
                            "slope": float(start_slope_val) if not np.isnan(start_slope_val) else None,
                            "roughness": float(start_roughness_val) if not np.isnan(start_roughness_val) else None,
                            "max_slope_deg": max_slope_deg,
                            "max_roughness_m": nav_config.max_roughness_m,
                            "cost": float(start_cost) if np.isfinite(start_cost) else "inf"
                        }
                    )
            
            if np.isinf(goal_cost):
                raise NavigationError(
                    "Goal position (site) is impassable - site may be on steep slope or rough terrain",
                    details={
                        "goal_lat": goal_lat,
                        "goal_lon": goal_lon,
                        "goal_pixel": goal_pixel
                    }
                )
            
            # Run A* pathfinding
            resolution_m = float(dem.attrs.get('resolution_m', cell_size_m))
            
            # For very large DEMs, consider downsampling the cost map for pathfinding
            # This significantly speeds up pathfinding while maintaining route quality
            downsample_factor = 1
            original_cost_map = cost_map
            original_start_pixel = start_pixel
            original_goal_pixel = goal_pixel
            
            # If cost map is very large (> 1000x1000), downsample for pathfinding
            if cost_map.shape[0] * cost_map.shape[1] > 1000000:
                downsample_factor = 2  # Downsample by 2x
                logger.info(
                    "Downsampling cost map for pathfinding",
                    original_shape=cost_map.shape,
                    downsample_factor=downsample_factor
                )
                
                # Downsample cost map using max pooling (take worst case in each block)
                # Reshape to allow downsampling
                h, w = cost_map.shape
                new_h, new_w = h // downsample_factor, w // downsample_factor
                
                # For cost maps, we want to preserve impassable areas (inf)
                # Use max pooling to ensure we don't lose impassable regions
                cost_map_downsampled = np.full((new_h, new_w), np.inf)
                
                for i in range(new_h):
                    for j in range(new_w):
                        i_start = i * downsample_factor
                        i_end = min(i_start + downsample_factor, h)
                        j_start = j * downsample_factor
                        j_end = min(j_start + downsample_factor, w)
                        
                        block = cost_map[i_start:i_end, j_start:j_end]
                        # If any cell in block is impassable, mark as impassable
                        if np.any(np.isinf(block)):
                            cost_map_downsampled[i, j] = np.inf
                        else:
                            # Use mean cost for passable blocks
                            cost_map_downsampled[i, j] = np.nanmean(block)
                
                cost_map = cost_map_downsampled
                start_pixel = (start_pixel[0] // downsample_factor, start_pixel[1] // downsample_factor)
                goal_pixel = (goal_pixel[0] // downsample_factor, goal_pixel[1] // downsample_factor)
                
                # Clamp to valid bounds
                start_pixel = (min(start_pixel[0], cost_map.shape[0] - 1), min(start_pixel[1], cost_map.shape[1] - 1))
                goal_pixel = (min(goal_pixel[0], cost_map.shape[0] - 1), min(goal_pixel[1], cost_map.shape[1] - 1))
                
                logger.info(
                    "Downsampled cost map",
                    new_shape=cost_map.shape,
                    new_start_pixel=start_pixel,
                    new_goal_pixel=goal_pixel
                )
            
            pathfinder = AStarPathfinder(cost_map, cell_size_m=resolution_m * downsample_factor)
            
            # Calculate waypoint spacing in pixels (adjusted for downsampling)
            max_spacing_pixels = int(max_waypoint_spacing_m / (resolution_m * downsample_factor))
            
            if progress_callback:
                progress_callback("pathfinding", 0.3, "Running A* pathfinding...")
            
            try:
                waypoint_pixels = pathfinder.find_path_with_waypoints(
                    start_pixel,
                    goal_pixel,
                    max_waypoint_spacing=max_spacing_pixels
                )
                
                # If we downsampled, upsample the waypoints back to original resolution
                if downsample_factor > 1:
                    waypoint_pixels_upsampled = []
                    for row, col in waypoint_pixels:
                        # Convert back to original pixel coordinates
                        orig_row = row * downsample_factor + downsample_factor // 2
                        orig_col = col * downsample_factor + downsample_factor // 2
                        # Clamp to original bounds
                        orig_row = min(orig_row, original_cost_map.shape[0] - 1)
                        orig_col = min(orig_col, original_cost_map.shape[1] - 1)
                        waypoint_pixels_upsampled.append((orig_row, orig_col))
                    
                    waypoint_pixels = waypoint_pixels_upsampled
                    logger.info(
                        "Upsampled waypoints to original resolution",
                        num_waypoints=len(waypoint_pixels),
                        downsample_factor=downsample_factor
                    )
                    
            except NavigationError as e:
                logger.error("Pathfinding failed", error=str(e))
                raise
            
            logger.info(f"Found path with {len(waypoint_pixels)} waypoints")
            
            if progress_callback:
                progress_callback("waypoint_generation", 0.9, "Generating waypoints...")
            
            # Convert waypoints to lat/lon
            waypoints_latlon = []
            for row, col in waypoint_pixels:
                lat, lon = self._pixel_to_latlon(row, col, dem)
                elev = float(metrics.elevation[row, col])
                waypoints_latlon.append((lat, lon, elev))
            
            # Define SITE frame origin at start position
            site_origin = SiteOrigin(
                lat=start_lat,
                lon=start_lon,
                elevation_m=start_elevation
            )
            
            # Transform waypoints to SITE frame
            waypoints_site = []
            for i, (lat, lon, elev) in enumerate(waypoints_latlon):
                x, y, z = self.coord_transformer.iau_mars_to_site_frame(
                    lat, lon, elev, site_origin
                )

                waypoints_site.append({
                    'waypoint_id': i + 1,
                    'x_meters': x,
                    'y_meters': y,
                    'z_site': z,
                    'lat': lat,
                    'lon': lon,
                    'elevation_m': elev,
                    'tolerance_meters': max_waypoint_spacing_m / 2.0
                })
            
            # Create DataFrame
            waypoints_df = pd.DataFrame(waypoints_site)
            
            total_distance = float(
                np.sqrt(waypoints_df['x_meters'].iloc[-1]**2 + 
                       waypoints_df['y_meters'].iloc[-1]**2)
            ) if len(waypoints_df) > 0 else 0.0
            
            logger.info(
                "Navigation plan complete",
                num_waypoints=len(waypoints_df),
                total_distance_m=total_distance
            )
            
            if progress_callback:
                progress_callback("waypoint_generation", 1.0, f"Navigation plan complete: {len(waypoints_df)} waypoints")
            
            return waypoints_df

        except Exception as e:
            raise NavigationError(
                "Navigation planning failed",
                details={"site_id": site_id, "error": str(e)},
            )

