"""Unit tests for longitude normalization helpers."""

import pytest

from marshab.core.raster_service import bbox_to_lon360, normalize_lon_bounds
from marshab.models import BoundingBox


def test_normalize_lon_bounds_preserves_full_world_extent():
    lon_min, lon_max = normalize_lon_bounds(-180.0, 180.0)

    assert lon_min == pytest.approx(0.0)
    assert lon_max == pytest.approx(360.0)


def test_normalize_lon_bounds_converts_negative_range_without_wrap_failure():
    lon_min, lon_max = normalize_lon_bounds(-22.5, 0.0)

    assert lon_min == pytest.approx(337.5)
    assert lon_max == pytest.approx(360.0)


def test_bbox_to_lon360_keeps_lon_max_360_instead_of_wrapping_to_zero():
    bbox = BoundingBox(lat_min=-10.0, lat_max=10.0, lon_min=180.0, lon_max=360.0)

    normalized = bbox_to_lon360(bbox)

    assert normalized.lon_min == pytest.approx(180.0)
    assert normalized.lon_max == pytest.approx(360.0)


def test_normalize_lon_bounds_rejects_non_increasing_intervals():
    with pytest.raises(ValueError):
        normalize_lon_bounds(20.0, 20.0)
