"""MarsHab Site Selector - Mars habitat site selection and rover navigation system."""

__version__ = "0.1.0"
__author__ = "MarsHab Development Team"
__description__ = "Geospatial analysis for Mars habitat construction site selection"

# Export commonly used types and functions
from marshab.config import get_config
from marshab.models import (
    AnalysisConfig,
    BoundingBox,
    SiteOrigin,
    TerrainMetrics,
    Waypoint,
)
from marshab.utils.logging import configure_logging, get_logger

__all__ = [
    "BoundingBox",
    "SiteOrigin",
    "Waypoint",
    "TerrainMetrics",
    "AnalysisConfig",
    "get_config",
    "get_logger",
    "configure_logging",
]

