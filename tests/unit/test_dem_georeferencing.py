"""Numerical tests of Mars DEM metadata, windows, masks, and resampling."""

import numpy as np
import pytest
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import Affine, array_bounds, from_origin

from marshab.exceptions import DataError
from marshab.models import BoundingBox
from marshab.processing.dem_loader import DEMLoader

RADIUS_M = 3396190.0
GEOGRAPHIC = CRS.from_proj4(f"+proj=longlat +a={RADIUS_M} +b={RADIUS_M} +no_defs")
PROJECTED = CRS.from_proj4(
    f"+proj=eqc +lat_ts=0 +lon_0=0 +a={RADIUS_M} +b={RADIUS_M} +units=m +no_defs"
)


def write_dem(tmp_path, transform, crs=GEOGRAPHIC, shape=(8, 10), masked=True):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if masked:
        values[3, 4] = -9999
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        transform=transform,
        crs=crs,
        nodata=-9999,
    ) as dst:
        dst.write(values, 1)
    return path, values


def test_geographic_metadata_masks_and_pixel_centers(tmp_path):
    transform = from_origin(77.0, 19.0, 0.01, 0.02)
    path, values = write_dem(tmp_path, transform)
    dem = DEMLoader().load(path)
    np.testing.assert_allclose(dem.x, 77 + (np.arange(10) + 0.5) * 0.01)
    np.testing.assert_allclose(dem.y, 19 - (np.arange(8) + 0.5) * 0.02)
    np.testing.assert_allclose(dem.lon[0], dem.x)
    np.testing.assert_allclose(dem.lat[:, 0], dem.y)
    assert np.isnan(dem.values[3, 4])
    assert dem.values[0, 0] == values[0, 0]
    assert CRS.from_wkt(dem.attrs["crs"]).equals(GEOGRAPHIC, ignore_axis_order=True)
    assert Affine(*dem.attrs["transform"]) == transform
    expected_y = RADIUS_M * np.deg2rad(0.02)
    expected_x = RADIUS_M * np.deg2rad(0.01) * np.cos(np.deg2rad(18.92))
    assert dem.attrs["resolution_y_m"] == pytest.approx(expected_y)
    assert dem.attrs["resolution_x_m"] == pytest.approx(expected_x)
    assert dem.attrs["resolution_m"] == pytest.approx(np.sqrt(expected_x * expected_y))


def test_projected_roi_is_transformed_and_coordinates_are_geographic(tmp_path):
    transformer = Transformer.from_crs(GEOGRAPHIC, PROJECTED, always_xy=True)
    x0, y0 = transformer.transform(77, 19)
    transform = from_origin(x0, y0, 20, 30)
    path, values = write_dem(tmp_path, transform, PROJECTED, masked=False)
    loader = DEMLoader()
    full = loader.load(path)
    inverse = Transformer.from_crs(PROJECTED, GEOGRAPHIC, always_xy=True)
    left, top = inverse.transform(x0 + 40, y0 - 30)
    right, bottom = inverse.transform(x0 + 100, y0 - 120)
    roi = BoundingBox(lon_min=left, lon_max=right, lat_min=bottom, lat_max=top)
    loaded = loader.load(path, roi=roi)
    clipped = loader.clip_to_roi(full, roi)
    np.testing.assert_array_equal(loaded, values[1:4, 2:5])
    np.testing.assert_array_equal(clipped, loaded)
    assert loaded.attrs["resolution_x_m"] == pytest.approx(20)
    assert loaded.attrs["resolution_y_m"] == pytest.approx(30)
    assert Affine(*loaded.attrs["transform"]) == transform * Affine.translation(2, 1)
    lon, lat = inverse.transform(x0 + 50, y0 - 45)
    assert loaded.lon.values[0, 0] == pytest.approx(lon)
    assert loaded.lat.values[0, 0] == pytest.approx(lat)
    assert loaded.x.values[0] == pytest.approx(x0 + 50)


def test_full_extent_is_preserved_and_partial_windows_include_intersections(tmp_path):
    transform = from_origin(77, 19, 0.01, 0.02)
    path, values = write_dem(tmp_path, transform, masked=False)
    loader = DEMLoader()
    full = loader.load(path)
    roi = BoundingBox(lon_min=77, lon_max=77.1, lat_min=18.84, lat_max=19)
    np.testing.assert_array_equal(loader.load(path, roi=roi), values)
    np.testing.assert_array_equal(loader.clip_to_roi(full, roi), values)
    partial = BoundingBox(lon_min=77.021, lon_max=77.049, lat_min=18.921, lat_max=18.979)
    clipped = loader.clip_to_roi(full, partial)
    np.testing.assert_array_equal(clipped, values[1:4, 2:5])
    assert Affine(*clipped.attrs["transform"]) == transform * Affine.translation(2, 1)


def test_outside_roi_fails_for_load_and_clip(tmp_path):
    path, _ = write_dem(tmp_path, from_origin(77, 19, 0.01, 0.02))
    loader = DEMLoader()
    roi = BoundingBox(lon_min=78, lon_max=79, lat_min=18, lat_max=19)
    with pytest.raises(DataError, match="does not overlap"):
        loader.load(path, roi=roi)
    with pytest.raises(DataError, match="does not overlap"):
        loader.clip_to_roi(loader.load(path), roi)


@pytest.mark.parametrize(
    "transform,crs",
    [
        (from_origin(77, 19, 0.01, 0.02), None),
        (Affine(1, 0.2, 0, 0, -1, 10), PROJECTED),
        (Affine(1, 0, 0, 0, 1, 0), PROJECTED),
    ],
)
def test_missing_rotated_or_south_up_georeferencing_is_rejected(tmp_path, transform, crs):
    path, _ = write_dem(tmp_path, transform, crs)
    with pytest.raises(DataError):
        DEMLoader().load(path)


def test_resampling_preserves_bounds_and_updates_coordinates_and_masks(tmp_path):
    transform = from_origin(1000, 2000, 20, 30)
    path, _ = write_dem(tmp_path, transform, PROJECTED)
    loader = DEMLoader()
    dem = loader.load(path)
    resampled = loader.resample(dem, 10, method="nearest")
    assert resampled.shape == (24, 20)
    new_transform = Affine(*resampled.attrs["transform"])
    np.testing.assert_allclose(
        array_bounds(*resampled.shape, new_transform), array_bounds(*dem.shape, transform)
    )
    assert resampled.x.values[0] == pytest.approx(1005)
    assert resampled.y.values[-1] == pytest.approx(1765)
    assert resampled.attrs["resolution_x_m"] == pytest.approx(10)
    assert resampled.attrs["resolution_y_m"] == pytest.approx(10)
    assert np.isnan(resampled.values[9:12, 8:10]).all()
    tiny = loader.resample(dem, 1e9)
    assert tiny.shape == (1, 1)
    np.testing.assert_allclose(
        array_bounds(*tiny.shape, Affine(*tiny.attrs["transform"])),
        array_bounds(*dem.shape, transform),
    )


@pytest.mark.parametrize("resolution", [0, -1, np.nan, np.inf])
def test_invalid_target_resolution_is_rejected(tmp_path, resolution):
    path, _ = write_dem(tmp_path, from_origin(0, 100, 20, 20), PROJECTED)
    loader = DEMLoader()
    with pytest.raises(DataError, match="finite and positive"):
        loader.resample(loader.load(path), resolution)
