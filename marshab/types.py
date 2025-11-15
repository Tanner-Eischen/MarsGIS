"""Type definitions and data models for MarsHab."""

from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Region of interest bounding box in IAU_MARS planetocentric coordinates."""

    lat_min: float = Field(..., ge=-90, le=90, description="Minimum latitude (degrees)")
    lat_max: float = Field(..., ge=-90, le=90, description="Maximum latitude (degrees)")
    lon_min: float = Field(..., ge=0, le=360, description="Minimum longitude (degrees east)")
    lon_max: float = Field(..., ge=0, le=360, description="Maximum longitude (degrees east)")

    @field_validator("lat_max")
    @classmethod
    def lat_max_greater_than_min(cls, v: float, info) -> float:
        """Ensure lat_max > lat_min."""
        if "lat_min" in info.data and v <= info.data["lat_min"]:
            raise ValueError("lat_max must be greater than lat_min")
        return v

    @field_validator("lon_max")
    @classmethod
    def lon_max_greater_than_min(cls, v: float, info) -> float:
        """Ensure lon_max > lon_min."""
        if "lon_min" in info.data and v <= info.data["lon_min"]:
            raise ValueError("lon_max must be greater than lon_min")
        return v

    def to_tuple(self) -> tuple:
        """Return as (lat_min, lat_max, lon_min, lon_max) tuple."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)


class SiteOrigin(BaseModel):
    """Rover SITE frame origin definition on Mars surface."""

    lat: float = Field(..., ge=-90, le=90, description="Latitude (degrees, planetocentric)")
    lon: float = Field(..., ge=0, le=360, description="Longitude (degrees, east positive)")
    elevation_m: float = Field(..., description="Elevation above areoid datum (meters)")


class Waypoint(BaseModel):
    """Rover navigation waypoint in SITE frame coordinates."""

    waypoint_id: int = Field(..., ge=0, description="Waypoint sequence number")
    x_meters: float = Field(..., description="X coordinate in SITE frame - Northing (meters)")
    y_meters: float = Field(..., description="Y coordinate in SITE frame - Easting (meters)")
    tolerance_meters: float = Field(5.0, gt=0, description="Arrival tolerance radius (meters)")
    heading_deg: float | None = Field(None, ge=0, le=360, description="Desired rover heading (degrees)")

    def distance_from_origin(self) -> float:
        """Calculate distance from SITE frame origin."""
        return float(np.sqrt(self.x_meters**2 + self.y_meters**2))


class TerrainMetrics(NamedTuple):
    """Terrain analysis outputs from DEM processing."""

    slope: np.ndarray  # Slope in degrees (0-90)
    aspect: np.ndarray  # Aspect in degrees (0-360, 0=North)
    roughness: np.ndarray  # Terrain roughness (standard deviation of elevation)
    tri: np.ndarray  # Terrain Ruggedness Index


class CriteriaWeights(BaseModel):
    """MCDM criteria and their importance weights."""

    slope: float = Field(0.30, ge=0, le=1, description="Weight for slope criterion")
    roughness: float = Field(0.25, ge=0, le=1, description="Weight for roughness criterion")
    elevation: float = Field(0.20, ge=0, le=1, description="Weight for elevation criterion")
    solar_exposure: float = Field(0.15, ge=0, le=1, description="Weight for solar exposure")
    resources: float = Field(0.10, ge=0, le=1, description="Weight for resource proximity")

    @field_validator("*", mode="before")
    @classmethod
    def validate_weights(cls, v):
        """Validate weights are positive."""
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError("Weights must be non-negative")
        return v

    def total(self) -> float:
        """Return sum of all weights."""
        return (
            self.slope
            + self.roughness
            + self.elevation
            + self.solar_exposure
            + self.resources
        )

    def normalize(self) -> "CriteriaWeights":
        """Normalize weights to sum to 1.0."""
        total = self.total()
        if total == 0:
            raise ValueError("Total weight cannot be zero")
        return CriteriaWeights(
            slope=self.slope / total,
            roughness=self.roughness / total,
            elevation=self.elevation / total,
            solar_exposure=self.solar_exposure / total,
            resources=self.resources / total,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "slope": self.slope,
            "roughness": self.roughness,
            "elevation": self.elevation,
            "solar_exposure": self.solar_exposure,
            "resources": self.resources,
        }


class AnalysisConfig(BaseModel):
    """Configuration for terrain analysis pipeline."""

    roi: BoundingBox
    criteria_weights: CriteriaWeights = Field(default_factory=CriteriaWeights)
    max_slope_deg: float = Field(5.0, gt=0, description="Maximum traversable slope (degrees)")
    max_roughness: float = Field(0.5, gt=0, description="Maximum traversable roughness")
    min_site_area_km2: float = Field(0.5, gt=0, description="Minimum site area (km²)")
    suitability_threshold: float = Field(0.7, ge=0, le=1, description="Suitability score threshold")


class SiteCandidate(BaseModel):
    """Identified construction site candidate."""

    site_id: int = Field(..., ge=0, description="Unique site identifier")
    geometry_type: str = Field(..., description="WKT geometry type")
    area_km2: float = Field(..., gt=0, description="Site area (km²)")
    mean_slope_deg: float = Field(..., ge=0, description="Mean slope (degrees)")
    mean_roughness: float = Field(..., ge=0, description="Mean roughness")
    mean_elevation_m: float = Field(..., description="Mean elevation (meters)")
    suitability_score: float = Field(..., ge=0, le=1, description="Overall suitability (0-1)")
    rank: int = Field(..., ge=1, description="Rank among candidates (1=best)")

