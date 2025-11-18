"""Route cost analysis with breakdown."""

from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouteCostBreakdown:
    """Route cost with component breakdown."""
    total_cost: float
    distance_m: float
    slope_penalty: float
    roughness_penalty: float
    elevation_penalty: float
    components: Dict[str, float]
    explanation: str


class RouteCostEngine:
    """Analyzes route costs with explainability."""
    
    def analyze_route(
        self,
        waypoints: pd.DataFrame,
        weights: Dict[str, float]
    ) -> RouteCostBreakdown:
        """Analyze route cost with component breakdown.
        
        Args:
            waypoints: Waypoint DataFrame with x_site, y_site (or similar coordinate columns)
            weights: Cost component weights
            
        Returns:
            RouteCostBreakdown with explanation
        """
        logger.info("Analyzing route cost")
        
        # Calculate distance
        distances = []
        for i in range(len(waypoints) - 1):
            # Try different possible column names
            x_col = None
            y_col = None
            
            for col in waypoints.columns:
                if 'x' in col.lower() and ('site' in col.lower() or 'meter' in col.lower()):
                    x_col = col
                if 'y' in col.lower() and ('site' in col.lower() or 'meter' in col.lower()):
                    y_col = col
            
            if x_col is None or y_col is None:
                # Fallback: use first two numeric columns
                numeric_cols = waypoints.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                else:
                    logger.warning("Could not find coordinate columns, using defaults")
                    x_col = 'x_meters' if 'x_meters' in waypoints.columns else 'x'
                    y_col = 'y_meters' if 'y_meters' in waypoints.columns else 'y'
            
            if x_col in waypoints.columns and y_col in waypoints.columns:
                dx = waypoints.iloc[i+1][x_col] - waypoints.iloc[i][x_col]
                dy = waypoints.iloc[i+1][y_col] - waypoints.iloc[i][y_col]
                dist = np.sqrt(dx**2 + dy**2)
                distances.append(dist)
            else:
                logger.warning(f"Coordinate columns not found: {x_col}, {y_col}")
                distances.append(0.0)
        
        total_distance = sum(distances) if distances else 0.0
        
        # Placeholder for terrain penalties
        # In real implementation, would query terrain along path
        slope_penalty = 50.0  # Stub
        roughness_penalty = 30.0  # Stub
        elevation_penalty = 20.0  # Stub
        
        # Calculate weighted cost
        components = {
            "distance": total_distance * weights.get("distance", 0.3),
            "slope": slope_penalty * weights.get("slope_penalty", 0.3),
            "roughness": roughness_penalty * weights.get("roughness_penalty", 0.2),
            "elevation": elevation_penalty * weights.get("elevation_penalty", 0.2)
        }
        
        total_cost = sum(components.values())
        
        # Generate explanation
        explanation = self._generate_explanation(
            total_distance, slope_penalty, roughness_penalty
        )
        
        return RouteCostBreakdown(
            total_cost=total_cost,
            distance_m=total_distance,
            slope_penalty=slope_penalty,
            roughness_penalty=roughness_penalty,
            elevation_penalty=elevation_penalty,
            components=components,
            explanation=explanation
        )
    
    def _generate_explanation(
        self,
        distance: float,
        slope_penalty: float,
        roughness_penalty: float
    ) -> str:
        """Generate route explanation."""
        parts = []
        
        parts.append(f"Total distance: {distance:.0f}m")
        
        if slope_penalty > 100:
            parts.append("includes steep terrain sections")
        elif slope_penalty > 50:
            parts.append("has moderate slope challenges")
        else:
            parts.append("traverses gentle slopes")
        
        if roughness_penalty > 50:
            parts.append("rough surface conditions")
        elif roughness_penalty > 20:
            parts.append("moderate surface roughness")
        else:
            parts.append("smooth terrain")
        
        return "This route " + ", ".join(parts) + "."

