"""Deterministic decision-brief generation for portfolio workflows."""

from __future__ import annotations

import math
from typing import Any

from marshab.models import SiteCandidate

MARS_RADIUS_M = 3396190.0
BASE_ENERGY_PER_M = 100.0


def _great_circle_distance_m(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    """Compute great-circle distance on Mars."""
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return MARS_RADIUS_M * c


def _slope_label(slope_deg: float) -> str:
    if slope_deg < 3.0:
        return "very gentle"
    if slope_deg < 7.0:
        return "moderate"
    return "steeper"


def _roughness_label(roughness: float) -> str:
    if roughness < 3.0:
        return "smooth"
    if roughness < 7.0:
        return "moderately rough"
    return "rough"


def build_decision_brief(
    site: SiteCandidate,
    *,
    start_lat: float,
    start_lon: float,
    seed: int | None = None,
    data_mode: str = "synthetic",
) -> dict[str, Any]:
    """Create deterministic portfolio-oriented decision brief."""
    route_distance_m = _great_circle_distance_m(
        start_lat,
        start_lon,
        site.lat,
        site.lon,
    )
    estimated_energy_j = route_distance_m * BASE_ENERGY_PER_M + site.mean_slope_deg * 1000.0
    estimated_traverse_hours = route_distance_m / max(0.04, 0.001) / 3600.0  # ~4 cm/s rover speed

    reasons = [
        f"Highest suitability score in current ROI ({site.suitability_score:.3f}).",
        f"Terrain slope is {_slope_label(site.mean_slope_deg)} ({site.mean_slope_deg:.2f} deg).",
        f"Surface roughness is {_roughness_label(site.mean_roughness)} ({site.mean_roughness:.2f}).",
        (
            f"Estimated traverse from start is {route_distance_m:.0f} m, "
            f"with approximate energy demand {estimated_energy_j:.0f} J."
        ),
    ]

    summary = (
        f"Site {site.site_id} is recommended because it maximizes suitability while "
        f"maintaining acceptable terrain and traverse cost for mission operations."
    )

    return {
        "site_id": site.site_id,
        "rank": site.rank,
        "suitability_score": round(site.suitability_score, 6),
        "summary": summary,
        "reasons": reasons,
        "terrain": {
            "mean_slope_deg": round(site.mean_slope_deg, 6),
            "mean_roughness": round(site.mean_roughness, 6),
            "mean_elevation_m": round(site.mean_elevation_m, 6),
        },
        "route_impacts": {
            "distance_m": round(route_distance_m, 3),
            "estimated_energy_j": round(estimated_energy_j, 3),
            "estimated_traverse_hours": round(estimated_traverse_hours, 3),
            "start": {"lat": start_lat, "lon": start_lon},
        },
        "determinism": {
            "seed": seed,
            "data_mode": data_mode,
        },
    }
