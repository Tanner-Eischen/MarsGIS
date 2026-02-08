"""Unit tests for geodetic tile-to-bbox conversion."""

import pytest

from marshab.core.raster_service import compute_tile_bbox_epsg4326


def test_z0_western_tile_maps_to_180_360_lon_domain():
    bbox = compute_tile_bbox_epsg4326(0, 0, 0)

    assert bbox.lat_min == pytest.approx(-90.0)
    assert bbox.lat_max == pytest.approx(90.0)
    assert bbox.lon_min == pytest.approx(180.0)
    assert bbox.lon_max == pytest.approx(360.0)


def test_z0_eastern_tile_maps_to_0_180_lon_domain():
    bbox = compute_tile_bbox_epsg4326(0, 1, 0)

    assert bbox.lat_min == pytest.approx(-90.0)
    assert bbox.lat_max == pytest.approx(90.0)
    assert bbox.lon_min == pytest.approx(0.0)
    assert bbox.lon_max == pytest.approx(180.0)


def test_z1_westernmost_tile_maps_to_180_270_lon_domain():
    bbox = compute_tile_bbox_epsg4326(1, 0, 0)

    assert bbox.lat_min == pytest.approx(0.0)
    assert bbox.lat_max == pytest.approx(90.0)
    assert bbox.lon_min == pytest.approx(180.0)
    assert bbox.lon_max == pytest.approx(270.0)


def test_z1_eastern_tile_maps_to_0_90_lon_domain():
    bbox = compute_tile_bbox_epsg4326(1, 2, 0)

    assert bbox.lat_min == pytest.approx(0.0)
    assert bbox.lat_max == pytest.approx(90.0)
    assert bbox.lon_min == pytest.approx(0.0)
    assert bbox.lon_max == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("z", "x", "y"),
    [
        (-1, 0, 0),
        (1, -1, 0),
        (1, 0, -1),
        (1, 4, 0),
        (1, 0, 2),
    ],
)
def test_invalid_tile_indices_raise(z: int, x: int, y: int):
    with pytest.raises(ValueError):
        compute_tile_bbox_epsg4326(z, x, y)
