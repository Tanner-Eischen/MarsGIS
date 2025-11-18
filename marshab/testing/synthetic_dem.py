"""Synthetic DEM generation for testing."""

from typing import Tuple, List, Dict, Optional
import numpy as np
import xarray as xr

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


def create_synthetic_dem_plane(
    size: Tuple[int, int],
    elevation: float,
    cell_size_m: float = 100.0
) -> xr.DataArray:
    """Create a flat plane DEM.
    
    Args:
        size: (height, width) in pixels
        elevation: Constant elevation in meters
        cell_size_m: Cell size in meters
        
    Returns:
        DataArray with elevation data
    """
    height, width = size
    elevation_array = np.ones((height, width), dtype=np.float32) * elevation
    
    dem = xr.DataArray(
        elevation_array,
        dims=["y", "x"],
        coords={
            "y": np.linspace(0, height * cell_size_m, height),
            "x": np.linspace(0, width * cell_size_m, width),
        }
    )
    
    logger.info("Created synthetic plane DEM", size=size, elevation=elevation)
    return dem


def create_synthetic_dem_hill(
    size: Tuple[int, int],
    center: Tuple[int, int],
    height: float,
    radius: float,
    base_elevation: float = 2000.0,
    cell_size_m: float = 100.0
) -> xr.DataArray:
    """Create a DEM with a single hill.
    
    Args:
        size: (height, width) in pixels
        center: (row, col) of hill center
        height: Hill height in meters
        radius: Hill radius in pixels
        base_elevation: Base elevation in meters
        cell_size_m: Cell size in meters
        
    Returns:
        DataArray with elevation data
    """
    height_px, width_px = size
    center_row, center_col = center
    
    y_coords = np.arange(height_px)
    x_coords = np.arange(width_px)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Distance from center
    dist = np.sqrt((y_grid - center_row)**2 + (x_grid - center_col)**2)
    
    # Gaussian hill
    elevation_array = base_elevation + height * np.exp(-(dist**2) / (2 * (radius**2)))
    elevation_array = elevation_array.astype(np.float32)
    
    dem = xr.DataArray(
        elevation_array,
        dims=["y", "x"],
        coords={
            "y": np.linspace(0, height_px * cell_size_m, height_px),
            "x": np.linspace(0, width_px * cell_size_m, width_px),
        }
    )
    
    logger.info("Created synthetic hill DEM", size=size, center=center, height=height)
    return dem


def create_synthetic_dem_crater(
    size: Tuple[int, int],
    center: Tuple[int, int],
    depth: float,
    radius: float,
    base_elevation: float = 2000.0,
    cell_size_m: float = 100.0
) -> xr.DataArray:
    """Create a DEM with a single crater.
    
    Args:
        size: (height, width) in pixels
        center: (row, col) of crater center
        depth: Crater depth in meters
        radius: Crater radius in pixels
        base_elevation: Base elevation in meters
        cell_size_m: Cell size in meters
        
    Returns:
        DataArray with elevation data
    """
    height_px, width_px = size
    center_row, center_col = center
    
    y_coords = np.arange(height_px)
    x_coords = np.arange(width_px)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Distance from center
    dist = np.sqrt((y_grid - center_row)**2 + (x_grid - center_col)**2)
    
    # Inverted Gaussian (crater)
    elevation_array = base_elevation - depth * np.exp(-(dist**2) / (2 * (radius**2)))
    elevation_array = np.clip(elevation_array, 0, None)  # Don't go below 0
    elevation_array = elevation_array.astype(np.float32)
    
    dem = xr.DataArray(
        elevation_array,
        dims=["y", "x"],
        coords={
            "y": np.linspace(0, height_px * cell_size_m, height_px),
            "x": np.linspace(0, width_px * cell_size_m, width_px),
        }
    )
    
    logger.info("Created synthetic crater DEM", size=size, center=center, depth=depth)
    return dem


def create_synthetic_dem_complex(
    size: Tuple[int, int],
    features: List[Dict],
    base_elevation: float = 2000.0,
    cell_size_m: float = 100.0
) -> xr.DataArray:
    """Create a DEM with multiple features.
    
    Args:
        size: (height, width) in pixels
        features: List of feature dictionaries with keys:
            - type: "hill" or "crater"
            - center: (row, col)
            - height/depth: float
            - radius: float
        base_elevation: Base elevation in meters
        cell_size_m: Cell size in meters
        
    Returns:
        DataArray with elevation data
    """
    # Start with base elevation
    height_px, width_px = size
    elevation_array = np.ones((height_px, width_px), dtype=np.float32) * base_elevation
    
    # Add each feature
    for feature in features:
        feature_type = feature.get("type")
        center = feature.get("center")
        radius = feature.get("radius")
        
        if feature_type == "hill":
            height = feature.get("height", 100.0)
            hill_dem = create_synthetic_dem_hill(size, center, height, radius, 0.0, cell_size_m)
            elevation_array += hill_dem.values - 0.0  # Add hill on top
        
        elif feature_type == "crater":
            depth = feature.get("depth", 100.0)
            crater_dem = create_synthetic_dem_crater(size, center, depth, radius, 0.0, cell_size_m)
            elevation_array += crater_dem.values - 0.0  # Subtract crater
    
    dem = xr.DataArray(
        elevation_array,
        dims=["y", "x"],
        coords={
            "y": np.linspace(0, height_px * cell_size_m, height_px),
            "x": np.linspace(0, width_px * cell_size_m, width_px),
        }
    )
    
    logger.info("Created synthetic complex DEM", size=size, num_features=len(features))
    return dem

