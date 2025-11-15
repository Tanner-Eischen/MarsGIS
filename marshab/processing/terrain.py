"""Terrain analysis functions for Mars DEMs."""

from typing import Tuple

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
        # Calculate gradients
        dy, dx = np.gradient(dem, self.cell_size_m)
        
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
        # Calculate gradients
        dy, dx = np.gradient(dem, self.cell_size_m)
        
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
        
        # Calculate all metrics
        slope = self.calculate_slope(elevation)
        aspect = self.calculate_aspect(elevation)
        roughness = self.calculate_roughness(elevation)
        tri = self.calculate_tri(elevation)
        
        logger.info(
            "Terrain analysis complete",
            slope_mean=float(np.nanmean(slope)),
            slope_max=float(np.nanmax(slope)),
            roughness_mean=float(np.nanmean(roughness)),
        )
        
        return TerrainMetrics(
            slope=slope,
            aspect=aspect,
            roughness=roughness,
            tri=tri
        )


def generate_cost_surface(
    slope: np.ndarray,
    roughness: np.ndarray,
    max_slope_deg: float = 25.0,
) -> np.ndarray:
    """Generate traversability cost surface for pathfinding.
    
    Args:
        slope: Slope array in degrees
        roughness: Roughness array
        max_slope_deg: Maximum traversable slope (degrees)
    
    Returns:
        Cost surface (higher = more difficult, inf = impassable)
    """
    # Initialize base cost
    cost = np.ones_like(slope, dtype=np.float32)
    
    # Add slope cost (exponential increase)
    cost += (slope / 45.0) ** 2 * 10.0
    
    # Add roughness cost (normalized)
    roughness_norm = roughness / (np.nanmax(roughness) + 1e-6)
    cost += roughness_norm * 5.0
    
    # Mark impassable areas (steep slopes)
    cost[slope > max_slope_deg] = np.inf
    
    logger.info(
        "Generated cost surface",
        passable_fraction=float(np.sum(np.isfinite(cost)) / cost.size),
        max_cost=float(np.nanmax(cost[np.isfinite(cost)])),
    )
    
    return cost

