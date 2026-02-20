"""Shared ROI parsing and BoundingBox construction."""

from __future__ import annotations

from typing import Any, Union

from marshab.models import BoundingBox


def roi_to_bounding_box(
    roi: Union[list[float], dict[str, float], str, tuple[float, float, float, float]],
) -> BoundingBox:
    """Build BoundingBox from ROI in various formats.

    Args:
        roi: Region of interest as:
            - list[float]: [lat_min, lat_max, lon_min, lon_max]
            - dict: {"lat_min", "lat_max", "lon_min", "lon_max"}
            - str: "lat_min,lat_max,lon_min,lon_max"
            - tuple: (lat_min, lat_max, lon_min, lon_max)

    Returns:
        Validated BoundingBox

    Raises:
        ValueError: If format is invalid or values out of range
    """
    if isinstance(roi, (list, tuple)):
        if len(roi) != 4:
            raise ValueError("ROI must have 4 values: [lat_min, lat_max, lon_min, lon_max]")
        lat_min, lat_max, lon_min, lon_max = (float(x) for x in roi)
    elif isinstance(roi, dict):
        try:
            lat_min = float(roi["lat_min"])
            lat_max = float(roi["lat_max"])
            lon_min = float(roi["lon_min"])
            lon_max = float(roi["lon_max"])
        except KeyError as e:
            raise ValueError(f"Missing ROI field: {e}") from e
    elif isinstance(roi, str):
        parts = roi.split(",")
        if len(parts) != 4:
            raise ValueError("ROI string must be 'lat_min,lat_max,lon_min,lon_max'")
        lat_min, lat_max, lon_min, lon_max = (float(x.strip()) for x in parts)
    else:
        raise TypeError(f"Unsupported ROI type: {type(roi)}")

    return BoundingBox(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


def roi_from_two_sites(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    padding: float = 0.1,
) -> BoundingBox:
    """Build BoundingBox encompassing two sites with padding.

    Args:
        lat1, lon1: First site coordinates
        lat2, lon2: Second site coordinates
        padding: Degrees to add on each side (default 0.1)

    Returns:
        BoundingBox covering both sites
    """
    return BoundingBox(
        lat_min=min(lat1, lat2) - padding,
        lat_max=max(lat1, lat2) + padding,
        lon_min=min(lon1, lon2) - padding,
        lon_max=max(lon1, lon2) + padding,
    )
