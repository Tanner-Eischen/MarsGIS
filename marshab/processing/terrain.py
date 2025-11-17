"""Terrain analysis functions for Mars DEMs."""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
import xarray as xr

from marshab.types import TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class TerrainAnalyzer:
    """Performs terrain analysis and derivative calculations on Mars DEMs."""
    
    def __init__(self, cell_size_m: float = 200.0):
        """Initialize terrain analyzer.
        
        Args:
            cell_size_m: DEM cell size in meters
        """
        self.cell_size_m = cell_size_m
    
    def calculate_slope(self, dem: np.ndarray) -> np.ndarray:
        """Calculate slope magnitude in degrees.
        
        Uses gradient method: slope = arctan(sqrt(dx² + dy²))
        
        Args:
            dem: Input elevation array
        
        Returns:
            Slope array in degrees (0-90)
        """
        # Check minimum size for gradient calculation (need at least 2x2)
        if dem.shape[0] < 2 or dem.shape[1] < 2:
            logger.warning(
                "DEM too small for slope calculation, returning zeros",
                shape=dem.shape
            )
            return np.zeros_like(dem)
        
        # Calculate gradients
        try:
            dy, dx = np.gradient(dem, self.cell_size_m)
        except ValueError as e:
            logger.warning(
                "Gradient calculation failed, returning zeros",
                error=str(e),
                shape=dem.shape
            )
            return np.zeros_like(dem)
        
        # Calculate slope magnitude
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        logger.debug(
            "Calculated slope",
            mean_deg=float(np.nanmean(slope)),
            max_deg=float(np.nanmax(slope)),
            min_deg=float(np.nanmin(slope)),
        )
        
        return slope
    
    def calculate_aspect(self, dem: np.ndarray) -> np.ndarray:
        """Calculate aspect (slope direction) in degrees from North.
        
        Aspect = 0° = North, 90° = East, 180° = South, 270° = West
        
        Args:
            dem: Input elevation array
        
        Returns:
            Aspect array in degrees (0-360)
        """
        # Check minimum size for gradient calculation (need at least 2x2)
        if dem.shape[0] < 2 or dem.shape[1] < 2:
            logger.warning(
                "DEM too small for aspect calculation, returning zeros",
                shape=dem.shape
            )
            return np.zeros_like(dem)
        
        # Calculate gradients
        try:
            dy, dx = np.gradient(dem, self.cell_size_m)
        except ValueError as e:
            logger.warning(
                "Gradient calculation failed for aspect, returning zeros",
                error=str(e),
                shape=dem.shape
            )
            return np.zeros_like(dem)
        
        # Calculate aspect (direction of maximum slope)
        # atan2 gives angle from east, convert to angle from north
        aspect = np.degrees(np.arctan2(-dx, dy))
        
        # Convert to 0-360 range
        aspect = (aspect + 360) % 360
        
        logger.debug(
            "Calculated aspect",
            valid_pixels=int(np.sum(~np.isnan(aspect))),
        )
        
        return aspect
    
    def calculate_roughness(
        self, dem: np.ndarray, window_size: int = 3
    ) -> np.ndarray:
        """Calculate terrain roughness as local standard deviation.
        
        Args:
            dem: Input elevation array
            window_size: Window size for roughness calculation (odd number)
        
        Returns:
            Roughness array
        """
        # Use generic filter to calculate local standard deviation
        roughness = ndimage.generic_filter(
            dem,
            np.std,
            size=window_size,
            mode='reflect'
        )
        
        logger.debug(
            "Calculated roughness",
            mean=float(np.nanmean(roughness)),
            max=float(np.nanmax(roughness)),
        )
        
        return roughness
    
    def calculate_tri(self, dem: np.ndarray) -> np.ndarray:
        """Calculate Terrain Ruggedness Index (TRI).
        
        TRI = mean absolute elevation difference between center and neighbors
        
        Args:
            dem: Input elevation array
        
        Returns:
            TRI array
        """
        # Create kernel for neighbors
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]) / 8.0
        
        # Convolve to get mean of neighbors
        mean_neighbors = ndimage.convolve(dem, kernel, mode='reflect')
        
        # TRI is absolute difference from neighbors
        tri = np.abs(dem - mean_neighbors)
        
        logger.debug(
            "Calculated TRI",
            mean=float(np.nanmean(tri)),
            max=float(np.nanmax(tri)),
        )
        
        return tri
    
    def analyze(self, dem: xr.DataArray) -> TerrainMetrics:
        """Perform complete terrain analysis on DEM.
        
        Args:
            dem: Input DEM DataArray
        
        Returns:
            TerrainMetrics with all calculated products
        """
        logger.info("Starting terrain analysis", shape=dem.shape)
        
        elevation = dem.values.astype(np.float32)
        
        # Check minimum size
        if elevation.size == 0:
            logger.warning("Empty elevation array, returning zero metrics")
            return TerrainMetrics(
                slope=np.zeros_like(elevation),
                aspect=np.zeros_like(elevation),
                roughness=np.zeros_like(elevation),
                tri=np.zeros_like(elevation)
            )
        
        # Calculate all metrics
        slope = self.calculate_slope(elevation)
        aspect = self.calculate_aspect(elevation)
        roughness = self.calculate_roughness(elevation)
        tri = self.calculate_tri(elevation)
        
        # Safe logging
        try:
            slope_mean = float(np.nanmean(slope)) if slope.size > 0 else 0.0
            slope_max = float(np.nanmax(slope)) if slope.size > 0 and np.any(np.isfinite(slope)) else 0.0
            roughness_mean = float(np.nanmean(roughness)) if roughness.size > 0 else 0.0
            logger.info(
                "Terrain analysis complete",
                slope_mean=slope_mean,
                slope_max=slope_max,
                roughness_mean=roughness_mean,
            )
        except Exception as e:
            logger.warning("Failed to log terrain analysis stats", error=str(e))
        
        return TerrainMetrics(
            slope=slope,
            aspect=aspect,
            roughness=roughness,
            tri=tri
        )


def detect_cliffs(
    elevation: np.ndarray,
    cell_size_m: float,
    threshold_m: float = 10.0,
) -> np.ndarray:
    """Detect cliffs and sudden elevation changes.
    
    Args:
        elevation: Elevation array in meters
        cell_size_m: Size of each cell in meters
        threshold_m: Elevation change threshold for cliff detection (meters)
    
    Returns:
        Boolean mask where True indicates cliff locations
    """
    if elevation.size == 0:
        return np.zeros_like(elevation, dtype=bool)
    
    # Calculate elevation gradient (rate of change)
    # Use gradient to find steep elevation changes
    try:
        dy, dx = np.gradient(elevation, cell_size_m)
    except ValueError:
        # If gradient fails (e.g., array too small), return no cliffs
        logger.warning("Gradient calculation failed for cliff detection, returning no cliffs")
        return np.zeros_like(elevation, dtype=bool)
    
    # Calculate magnitude of elevation change rate
    # This gives us the rate of elevation change per meter
    gradient_magnitude = np.sqrt(dy**2 + dx**2)
    
    # A cliff is where the elevation changes more than threshold_m over ~1-2 cell distances
    # Since gradient is in m/m, we check if gradient exceeds threshold/cell_size
    # For a threshold of 10m over ~1 cell (463m), we'd expect gradient > 10/463 ≈ 0.022
    # But we want to detect changes over 1-2 cells, so use a more conservative threshold
    gradient_threshold = threshold_m / (cell_size_m * 1.5)  # Allow detection over ~1.5 cells
    
    cliff_mask = gradient_magnitude > gradient_threshold
    
    # Also check for absolute elevation differences between adjacent cells
    # This catches very sudden drops/rises
    if elevation.shape[0] > 1 and elevation.shape[1] > 1:
        # Check vertical differences
        vert_diff = np.abs(np.diff(elevation, axis=0))
        vert_cliffs = np.zeros_like(elevation, dtype=bool)
        vert_cliffs[:-1, :] = vert_diff > threshold_m
        vert_cliffs[1:, :] = vert_cliffs[1:, :] | (vert_diff > threshold_m)
        
        # Check horizontal differences
        horiz_diff = np.abs(np.diff(elevation, axis=1))
        horiz_cliffs = np.zeros_like(elevation, dtype=bool)
        horiz_cliffs[:, :-1] = horiz_diff > threshold_m
        horiz_cliffs[:, 1:] = horiz_cliffs[:, 1:] | (horiz_diff > threshold_m)
        
        # Combine all cliff detections
        cliff_mask = cliff_mask | vert_cliffs | horiz_cliffs
    
    num_cliffs = int(np.sum(cliff_mask))
    if num_cliffs > 0:
        logger.info(
            "Detected cliffs",
            num_cliffs=num_cliffs,
            threshold_m=threshold_m,
            fraction=float(num_cliffs / elevation.size)
        )
    
    return cliff_mask


def generate_cost_surface(
    slope: np.ndarray,
    roughness: np.ndarray,
    max_slope_deg: float = 25.0,
    slope_weight: float = 10.0,
    roughness_weight: float = 5.0,
    elevation: Optional[np.ndarray] = None,
    cell_size_m: Optional[float] = None,
    cliff_threshold_m: Optional[float] = None,
) -> np.ndarray:
    """Generate traversability cost surface for pathfinding.
    
    Args:
        slope: Slope array in degrees
        roughness: Roughness array
        max_slope_deg: Maximum traversable slope (degrees)
        slope_weight: Slope cost multiplier
        roughness_weight: Roughness cost multiplier
        elevation: Optional elevation array for cliff detection
        cell_size_m: Optional cell size in meters for cliff detection
        cliff_threshold_m: Optional elevation change threshold for cliff detection (meters)
    
    Returns:
        Cost surface (higher = more difficult, inf = impassable)
    """
    # Initialize base cost
    cost = np.ones_like(slope, dtype=np.float32)
    
    # Add slope cost (exponential increase) with configurable weight
    cost += (slope / 45.0) ** 2 * slope_weight
    
    # Add roughness cost (normalized) with configurable weight
    # Handle empty or all-zero roughness arrays
    roughness_max = np.nanmax(roughness) if roughness.size > 0 and np.any(np.isfinite(roughness)) else 1.0
    roughness_norm = roughness / (roughness_max + 1e-6)
    cost += roughness_norm * roughness_weight
    
    # Detect and mark cliffs if elevation data is provided
    if elevation is not None and cell_size_m is not None and cliff_threshold_m is not None:
        cliff_mask = detect_cliffs(elevation, cell_size_m, cliff_threshold_m)
        # Mark cliffs as impassable (infinite cost)
        cost[cliff_mask] = np.inf
        logger.debug(
            "Applied cliff detection to cost surface",
            num_cliffs=int(np.sum(cliff_mask))
        )
    
    # Mark impassable areas (steep slopes)
    cost[slope > max_slope_deg] = np.inf
    
    # Handle empty cost arrays
    if cost.size > 0:
        finite_mask = np.isfinite(cost)
        passable_fraction = float(np.sum(finite_mask) / cost.size) if cost.size > 0 else 0.0
        finite_costs = cost[finite_mask]
        max_cost = float(np.nanmax(finite_costs)) if finite_costs.size > 0 else 0.0
        logger.info(
            "Generated cost surface",
            passable_fraction=passable_fraction,
            max_cost=max_cost,
            slope_weight=slope_weight,
            roughness_weight=roughness_weight,
        )
    else:
        logger.warning("Empty cost surface generated")
    
    return cost

