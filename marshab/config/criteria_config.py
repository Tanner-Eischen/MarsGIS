"""Criteria configuration for MCDM analysis."""

from typing import Dict, Literal
from pydantic import BaseModel, Field


class Criterion(BaseModel):
    """Single criterion definition."""
    
    name: str
    display_name: str
    description: str
    beneficial: bool = True  # True if higher values are better
    weight: float = Field(ge=0.0, le=1.0)
    source: Literal["terrain", "derived", "external"]


class CriteriaConfig(BaseModel):
    """Complete criteria configuration."""
    
    criteria: Dict[str, Criterion]
    
    def validate_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(c.weight for c in self.criteria.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Default configuration
DEFAULT_CRITERIA = CriteriaConfig(
    criteria={
        "slope": Criterion(
            name="slope",
            display_name="Slope Safety",
            description="Lower slopes are safer for landing",
            beneficial=False,
            weight=0.30,
            source="terrain"
        ),
        "roughness": Criterion(
            name="roughness",
            display_name="Surface Roughness",
            description="Smoother surfaces are preferred",
            beneficial=False,
            weight=0.25,
            source="terrain"
        ),
        "elevation": Criterion(
            name="elevation",
            display_name="Elevation",
            description="Lower elevations have better atmospheric density",
            beneficial=False,
            weight=0.20,
            source="terrain"
        ),
        "solar_exposure": Criterion(
            name="solar_exposure",
            display_name="Solar Exposure",
            description="Higher solar exposure for power generation",
            beneficial=True,
            weight=0.15,
            source="derived"
        ),
        "science_value": Criterion(
            name="science_value",
            display_name="Science Value",
            description="Proximity to features of scientific interest",
            beneficial=True,
            weight=0.10,
            source="external"
        )
    }
)

