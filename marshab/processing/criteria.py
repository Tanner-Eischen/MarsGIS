"""Extract and prepare criteria from terrain analysis."""

import numpy as np
import xarray as xr
from typing import Dict

from marshab.models import TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class CriteriaExtractor:
    """Extracts criteria values from terrain analysis results."""
    
    def __init__(self, dem: xr.DataArray, metrics: TerrainMetrics):
        """Initialize criteria extractor.
        
        Args:
            dem: DEM DataArray
            metrics: TerrainMetrics from terrain analysis
        """
        self.dem = dem
        self.metrics = metrics
    
    def extract_slope_criterion(self) -> np.ndarray:
        """Extract slope values as cost criterion."""
        return self.metrics.slope
    
    def extract_roughness_criterion(self) -> np.ndarray:
        """Extract roughness values as cost criterion."""
        return self.metrics.roughness
    
    def extract_elevation_criterion(self) -> np.ndarray:
        """Extract elevation values as cost criterion."""
        return self.metrics.elevation
    
    def calculate_solar_exposure(self) -> np.ndarray:
        """Calculate solar exposure based on slope and aspect.
        
        Assumes equatorial region with optimal sun from north.
        North-facing slopes get maximum exposure.
        
        Returns:
            Solar exposure values in [0, 1] range
        """
        slope_rad = np.radians(self.metrics.slope)
        
        # Handle aspect: -1 for flat areas, convert to radians
        aspect_valid = self.metrics.aspect >= 0
        aspect_rad = np.zeros_like(self.metrics.aspect, dtype=np.float32)
        aspect_rad[aspect_valid] = np.radians(self.metrics.aspect[aspect_valid])
        
        # North-facing slopes get maximum exposure
        # Flat areas also good (cos(0) = 1)
        # Formula: cos(slope) + 0.5 * cos(aspect) * sin(slope)
        # For flat areas (aspect = -1), use cos(slope) only
        exposure = np.cos(slope_rad)
        
        # Add aspect contribution for non-flat areas
        aspect_contribution = 0.5 * np.cos(aspect_rad) * np.sin(slope_rad)
        exposure[aspect_valid] += aspect_contribution[aspect_valid]
        
        # Clip to [0, 1] range
        exposure = np.clip(exposure, 0, 1)
        
        return exposure
    
    def calculate_science_value(self) -> np.ndarray:
        """Placeholder for science value criterion.
        
        In real implementation, would use proximity to:
        - Crater features
        - Mineral deposits
        - Water ice signatures
        
        Returns:
            Uniform science value array (0.5 for now)
        """
        # Uniform for now - to be replaced with actual science data
        return np.ones_like(self.metrics.slope) * 0.5
    
    def extract_all(self) -> Dict[str, np.ndarray]:
        """Extract all criteria into dictionary.
        
        Returns:
            Dictionary mapping criterion names to value arrays
        """
        logger.info("Extracting criteria from terrain analysis")
        
        criteria = {
            "slope": self.extract_slope_criterion(),
            "roughness": self.extract_roughness_criterion(),
            "elevation": self.extract_elevation_criterion(),
            "solar_exposure": self.calculate_solar_exposure(),
            "science_value": self.calculate_science_value()
        }
        
        logger.info(
            "Criteria extracted",
            num_criteria=len(criteria),
            criteria_names=list(criteria.keys())
        )
        
        return criteria

