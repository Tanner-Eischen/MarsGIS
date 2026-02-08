"""Stable contracts for tile math, DEM error payloads, and raster metadata fields.

Phase 0 locks these shapes/rules so downstream work can proceed without ambiguity.
"""

from __future__ import annotations

import math
from typing import Final, Literal, TypedDict

from pydantic import BaseModel, Field

MarsDataset = Literal["mola", "mola_200m", "hirise", "ctx"]

# EPSG:4326 geodetic tile matrix contract.
EPSG4326_MIN_LON_DEG: Final[float] = -180.0
EPSG4326_MAX_LON_DEG: Final[float] = 180.0
EPSG4326_MIN_LAT_DEG: Final[float] = -90.0
EPSG4326_MAX_LAT_DEG: Final[float] = 90.0
EPSG4326_WORLD_LON_SPAN_DEG: Final[float] = 360.0
EPSG4326_WORLD_LAT_SPAN_DEG: Final[float] = 180.0

# Longitude domain contract.
CLIENT_LON_MIN_DEG: Final[float] = -180.0
CLIENT_LON_MAX_DEG: Final[float] = 180.0
RASTER_LON_MIN_DEG: Final[float] = 0.0
RASTER_LON_MAX_DEG: Final[float] = 360.0

# Shared raster metadata contract keys.
DATASET_REQUESTED_FIELD: Final[str] = "dataset_requested"
DATASET_USED_FIELD: Final[str] = "dataset_used"
IS_FALLBACK_FIELD: Final[str] = "is_fallback"
FALLBACK_REASON_FIELD: Final[str] = "fallback_reason"
RASTER_RESPONSE_METADATA_FIELDS: Final[tuple[str, str, str, str]] = (
    DATASET_REQUESTED_FIELD,
    DATASET_USED_FIELD,
    IS_FALLBACK_FIELD,
    FALLBACK_REASON_FIELD,
)

# 503 payload contract for real-DEM unavailability.
REAL_DEM_UNAVAILABLE_STATUS_CODE: Final[int] = 503
REAL_DEM_UNAVAILABLE_ERROR_CODE: Final[str] = "real_dem_unavailable"


class RasterResponseMetadata(TypedDict):
    """Canonical metadata block for DEM-backed responses."""

    dataset_requested: MarsDataset
    dataset_used: MarsDataset
    is_fallback: bool
    fallback_reason: str | None


class RealDemUnavailablePayload(TypedDict):
    """Canonical payload shape for 503 DEM availability failures."""

    error: Literal["real_dem_unavailable"]
    detail: str
    dataset_requested: MarsDataset
    dataset_used: MarsDataset | None
    is_fallback: bool
    fallback_reason: str | None


class RealDemUnavailableErrorModel(BaseModel):
    """OpenAPI model for 503 `real_dem_unavailable` responses."""

    error: Literal["real_dem_unavailable"] = Field(
        default=REAL_DEM_UNAVAILABLE_ERROR_CODE,
        description="Stable machine-readable error code.",
    )
    detail: str = Field(..., description="Human-readable failure detail.")
    dataset_requested: MarsDataset = Field(..., description="Dataset requested by the caller.")
    dataset_used: MarsDataset | None = Field(
        None,
        description="Dataset selected by fallback resolution, if any.",
    )
    is_fallback: bool = Field(
        False,
        description="Whether a fallback dataset path had already been selected.",
    )
    fallback_reason: str | None = Field(
        None,
        description="Fallback reason code when dataset substitution occurred.",
    )


def _validate_zoom(zoom: int) -> None:
    if zoom < 0:
        raise ValueError("Zoom must be non-negative")


def epsg4326_x_cols(zoom: int) -> int:
    """Return EPSG:4326 matrix width (`x_cols`) for the given zoom: `2^(z+1)`."""
    _validate_zoom(zoom)
    return 2 ** (zoom + 1)


def epsg4326_y_rows(zoom: int) -> int:
    """Return EPSG:4326 matrix height (`y_rows`) for the given zoom: `2^z`."""
    _validate_zoom(zoom)
    return 2 ** zoom


def normalize_lon_for_raster(lon: float) -> float:
    """Normalize longitude to raster-access domain `[0, 360)`."""
    if not math.isfinite(lon):
        return lon
    return lon % RASTER_LON_MAX_DEG


def normalize_lon_for_client(lon: float) -> float:
    """Normalize longitude to client domain `[-180, 180]`."""
    lon_360 = normalize_lon_for_raster(lon)
    return lon_360 - RASTER_LON_MAX_DEG if lon_360 > CLIENT_LON_MAX_DEG else lon_360


def client_lon_bounds_to_raster(lon_min: float, lon_max: float) -> tuple[float, float]:
    """Convert non-wrapping client lon bounds into monotonic raster bounds."""
    lon_span = lon_max - lon_min
    if lon_span <= 0:
        raise ValueError("Longitude span must be positive")
    lon_min_raster = normalize_lon_for_raster(lon_min)
    lon_max_raster = min(RASTER_LON_MAX_DEG, lon_min_raster + lon_span)
    return lon_min_raster, lon_max_raster


def build_raster_response_metadata(
    *,
    dataset_requested: MarsDataset,
    dataset_used: MarsDataset,
    is_fallback: bool,
    fallback_reason: str | None,
) -> RasterResponseMetadata:
    """Build metadata using the locked response field set."""
    return {
        DATASET_REQUESTED_FIELD: dataset_requested,
        DATASET_USED_FIELD: dataset_used,
        IS_FALLBACK_FIELD: is_fallback,
        FALLBACK_REASON_FIELD: fallback_reason,
    }


def build_real_dem_unavailable_payload(
    *,
    detail: str,
    dataset_requested: MarsDataset,
    dataset_used: MarsDataset | None = None,
    is_fallback: bool = False,
    fallback_reason: str | None = None,
) -> RealDemUnavailablePayload:
    """Build a 503 payload using the locked `real_dem_unavailable` schema."""
    payload = RealDemUnavailableErrorModel(
        detail=detail,
        dataset_requested=dataset_requested,
        dataset_used=dataset_used,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
    )
    return payload.model_dump()


REAL_DEM_UNAVAILABLE_RESPONSE_DOC: Final[dict[int, dict[str, object]]] = {
    REAL_DEM_UNAVAILABLE_STATUS_CODE: {
        "model": RealDemUnavailableErrorModel,
        "description": "Requested real DEM coverage is unavailable.",
        "content": {
            "application/json": {
                "example": build_real_dem_unavailable_payload(
                    detail="Real DEM coverage is unavailable for the requested tile.",
                    dataset_requested="hirise",
                    dataset_used="mola",
                    is_fallback=True,
                    fallback_reason="hirise_unavailable",
                )
            }
        },
    }
}

