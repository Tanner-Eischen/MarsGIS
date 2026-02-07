"""Backward-compatible type exports.

Historically, project models were imported from ``marshab.types``.
The canonical models now live in ``marshab.models``.
"""

from pydantic import BaseModel, Field

from marshab.models import (
    BoundingBox,
    CriteriaWeights,
    SiteCandidate,
    SiteOrigin,
    TerrainMetrics,
    Waypoint,
)


class AnalysisConfig(BaseModel):
    """Legacy analysis request model retained for test/backward compatibility."""

    roi: BoundingBox
    criteria_weights: CriteriaWeights = Field(default_factory=CriteriaWeights)
    max_slope_deg: float = Field(5.0, gt=0)
    suitability_threshold: float = Field(0.7, ge=0, le=1)


__all__ = [
    "AnalysisConfig",
    "BoundingBox",
    "CriteriaWeights",
    "SiteCandidate",
    "SiteOrigin",
    "TerrainMetrics",
    "Waypoint",
]
