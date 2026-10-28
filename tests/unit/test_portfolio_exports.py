"""Scientific export invariants and Mars coordinate round trips, without network."""

import numpy as np
import pytest
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin

from marshab.analysis.export import export_suitability_geotiff
from marshab.analysis.portfolio import LOCAL_CRS, ROI, acquire_dem, path_statistics


def test_export_uses_source_grid_and_masks(tmp_path):
    transform = from_origin(1234, 5678, 20, 30)
    values = np.arange(12, dtype=float).reshape(3, 4)
    values[0, 0] = np.nan
    dem = xr.DataArray(values, attrs={"crs": LOCAL_CRS.to_wkt(), "transform": tuple(transform)[:6]})
    scores = np.ones((3, 4)) * 0.75
    scores[1, 1] = np.nan
    path = export_suitability_geotiff(ROI, "fixture", {}, tmp_path / "result.tif", dem, scores)
    with rasterio.open(path) as src:
        assert src.transform == transform
        assert src.crs == rasterio.crs.CRS.from_wkt(LOCAL_CRS.to_wkt())
        assert src.shape == (3, 4) and src.nodata == -9999
        result = src.read(1, masked=True)
        assert result.mask[0, 0] and result.mask[1, 1]
        assert result.count() == 10
        np.testing.assert_allclose(result.compressed(), 0.75)
        assert src.xy(2, 3) == (1304, 5603)


def test_export_rejects_invented_georeferencing(tmp_path):
    dem = xr.DataArray(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="explicit CRS"):
        export_suitability_geotiff(ROI, "fixture", {}, tmp_path / "bad.tif", dem, dem.values)
    with pytest.raises(ValueError, match="same two-dimensional"):
        export_suitability_geotiff(ROI, "fixture", {}, tmp_path / "bad.tif", dem, np.ones((3, 3)))


def test_mars_local_projection_roundtrip():
    forward = Transformer.from_crs(LOCAL_CRS.geodetic_crs, LOCAL_CRS, always_xy=True)
    inverse = Transformer.from_crs(LOCAL_CRS, LOCAL_CRS.geodetic_crs, always_xy=True)
    longitude = np.array([77.25, 77.35, 77.45, 77.25, 77.45])
    latitude = np.array([18.25, 18.35, 18.45, 18.45, 18.25])
    x, y = forward.transform(longitude, latitude)
    lon, lat = inverse.transform(x, y)
    np.testing.assert_allclose(lon, longitude, atol=1e-10, rtol=0)
    np.testing.assert_allclose(lat, latitude, atol=1e-10, rtol=0)
    np.testing.assert_allclose([x[1], y[1]], [0, 0], atol=1e-8)


def test_path_distance_and_destination_cost():
    path = [(0, 0), (0, 1), (1, 2)]
    cost = np.array([[99, 2, 3], [1, 4, 5]])
    stats = path_statistics(path, cost)
    assert stats["distance_m"] == pytest.approx(20 * (1 + np.sqrt(2)))
    assert stats["terrain_cost_m_equivalent"] == pytest.approx(20 * (2 + 5 * np.sqrt(2)))
    assert stats["straight_line_m"] == pytest.approx(20 * np.sqrt(5))


def test_pinned_dem_rejects_corrupt_local_file(tmp_path):
    path = tmp_path / "bad.tif"
    path.write_bytes(b"not the documented DEM")
    with pytest.raises(ValueError, match="checksum mismatch"):
        acquire_dem(path)
