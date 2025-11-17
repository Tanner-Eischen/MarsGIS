"""Common helper functions."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def generate_cache_key(*args: Any) -> str:
    """Generate cache key from arguments.
    
    Args:
        *args: Values to hash for cache key
    
    Returns:
        Hexadecimal cache key string
    """
    content = "_".join(str(arg) for arg in args)
    return hashlib.md5(content.encode()).hexdigest()


def ensure_numpy_array(data: Any) -> np.ndarray:
    """Ensure data is numpy array.
    
    Args:
        data: Input data (array-like)
    
    Returns:
        NumPy array
    """
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def format_area(area_km2: float) -> str:
    """Format area for human-readable display.
    
    Args:
        area_km2: Area in square kilometers
    
    Returns:
        Formatted string (e.g., "1.23 km²" or "456.7 m²")
    """
    if area_km2 >= 1.0:
        return f"{area_km2:.2f} km²"
    else:
        area_m2 = area_km2 * 1_000_000
        return f"{area_m2:.1f} m²"


def format_distance(distance_m: float) -> str:
    """Format distance for human-readable display.
    
    Args:
        distance_m: Distance in meters
    
    Returns:
        Formatted string (e.g., "1.23 km" or "456 m")
    """
    if distance_m >= 1000:
        return f"{distance_m/1000:.2f} km"
    else:
        return f"{distance_m:.0f} m"





