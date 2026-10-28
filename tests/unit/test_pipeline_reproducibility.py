"""Regression evidence for independent runs and actual DEM georeferencing."""

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from marshab.config.criteria_config import DEFAULT_CRITERIA
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.raster_service import get_cell_size_from_dem, get_cell_spacing_from_dem
from marshab.models import BoundingBox
from marshab.processing.dem_loader import DEMLoader

RADIUS = 3396190.0
PROJECTED = CRS.from_proj4(f"+proj=eqc +lat_ts=0 +a={RADIUS} +b={RADIUS} +units=m +no_defs")


def configure_pipeline(monkeypatch, tmp_path, test_config, *, plane=False):
    # Deliberately use 20 x 40 m cells under a dataset nominally configured as MOLA.
    origin = Transformer.from_crs(PROJECTED.geodetic_crs, PROJECTED, always_xy=True)
    x0, y0 = origin.transform(77, 19)
    affine = from_origin(x0, y0, 20, 40)
    rows, cols = np.indices((12, 16))
    elevation = 0.02 * cols * 20 - 0.05 * rows * 40
    if not plane:
        elevation += 0.1 * cols**2 + 2 * np.sin(rows)
    path = tmp_path / "local_dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=12,
        width=16,
        count=1,
        dtype="float32",
        crs=PROJECTED,
        transform=affine,
    ) as dst:
        dst.write(elevation.astype(np.float32), 1)
    dem = DEMLoader().load(path)
    test_config.analysis.min_site_area_km2 = 0
    monkeypatch.setattr("marshab.core.analysis_pipeline.get_config", lambda: test_config)
    monkeypatch.setattr(
        "marshab.core.analysis_pipeline.DataManager",
        lambda: SimpleNamespace(get_dem_for_roi=lambda *_args, **_kwargs: dem),
    )
    # Larger than the actual raster window: ROI interpolation would be incorrect.
    roi = BoundingBox(lon_min=76, lon_max=78, lat_min=18, lat_max=20)
    return AnalysisPipeline(), dem, affine, roi


def test_default_custom_default_runs_are_independent(monkeypatch, tmp_path, test_config):
    pipeline, _dem, _affine, roi = configure_pipeline(monkeypatch, tmp_path, test_config)
    defaults_before = DEFAULT_CRITERIA.model_dump()
    first = pipeline.run(roi, threshold=0)
    custom = pipeline.run(
        roi,
        threshold=0,
        criteria_weights={
            "slope": 0,
            "roughness": 0,
            "elevation": 1,
            "solar_exposure": 0,
            "science_value": 0,
        },
    )
    repeated = pipeline.run(roi, threshold=0)
    assert DEFAULT_CRITERIA.model_dump() == defaults_before
    np.testing.assert_array_equal(first.suitability, repeated.suitability)
    assert not np.allclose(first.suitability, custom.suitability)
    assert [site.model_dump() for site in first.sites] == [
        site.model_dump() for site in repeated.sites
    ]


@pytest.mark.parametrize("contours", [True, False])
def test_pipeline_uses_anisotropic_native_spacing_and_actual_centroid(
    monkeypatch, tmp_path, test_config, contours
):
    if not contours:
        monkeypatch.setattr("marshab.core.analysis_pipeline.sk_measure", None)
    pipeline, dem, affine, roi = configure_pipeline(monkeypatch, tmp_path, test_config, plane=True)
    result = pipeline.run(roi, threshold=0)
    expected_slope = np.rad2deg(np.arctan(np.hypot(0.02, 0.05)))
    # Float32 DEM quantization affects derivatives by up to a few microdegrees.
    np.testing.assert_allclose(result.metrics.slope, expected_slope, atol=3e-6)
    assert len(result.sites) == 1
    site = result.sites[0]
    assert site.area_km2 == pytest.approx(dem.size * 20 * 40 / 1e6)
    center_x, center_y = affine * (dem.shape[1] / 2, dem.shape[0] / 2)
    to_geographic = Transformer.from_crs(PROJECTED, PROJECTED.geodetic_crs, always_xy=True)
    expected_lon, expected_lat = to_geographic.transform(center_x, center_y)
    assert site.lon == pytest.approx(expected_lon)
    assert site.lat == pytest.approx(expected_lat)
    # Every polygon point must convert back to the native raster edge envelope.
    assert site.polygon_coords[0] == site.polygon_coords[-1]
    to_native = Transformer.from_crs(PROJECTED.geodetic_crs, PROJECTED, always_xy=True)
    xs, ys = to_native.transform(*np.asarray(site.polygon_coords).T)
    inverse = ~affine
    pixels = [inverse * (x, y) for x, y in zip(xs, ys, strict=True)]
    pixel_cols, pixel_rows = np.asarray(pixels).T
    assert pixel_cols.min() == pytest.approx(0, abs=1e-8)
    assert pixel_cols.max() == pytest.approx(dem.shape[1])
    assert pixel_rows.min() == pytest.approx(0, abs=1e-8)
    assert pixel_rows.max() == pytest.approx(dem.shape[0])


@pytest.mark.parametrize("accessor", ["resolution", "res"])
def test_native_geographic_resolution_is_converted_to_meters(accessor):
    crs = CRS.from_proj4(f"+proj=longlat +a={RADIUS} +b={RADIUS} +no_defs")
    dem = SimpleNamespace(
        attrs={},
        coords={"lat": np.array([60.0])},
        rio=SimpleNamespace(crs=crs, **{accessor: lambda: (0.01, -0.02)}),
    )
    dy, dx = get_cell_spacing_from_dem(dem)
    assert dy == pytest.approx(RADIUS * np.deg2rad(0.02))
    assert dx == pytest.approx(RADIUS * np.deg2rad(0.01) * 0.5)
    assert get_cell_size_from_dem(dem) == pytest.approx(np.sqrt(dx * dy))


def test_physical_metadata_precedes_native_degrees_and_nominal_resolution():
    dem = xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        attrs={
            "resolution_x_m": 20,
            "resolution_y_m": 40,
            "resolution_m": 999,
        },
    )
    assert get_cell_spacing_from_dem(dem) == (40, 20)
    assert get_cell_size_from_dem(dem) == pytest.approx(np.sqrt(800))


def test_native_projected_units_and_unreferenced_default():
    feet = CRS.from_proj4(f"+proj=eqc +a={RADIUS} +b={RADIUS} +units=ft +no_defs")
    dem = SimpleNamespace(attrs={}, rio=SimpleNamespace(crs=feet, res=(10, -20)))
    assert get_cell_spacing_from_dem(dem) == pytest.approx((6.096, 3.048))
    unreferenced = SimpleNamespace(attrs={}, rio=SimpleNamespace(res=(0.01, -0.01)))
    assert get_cell_size_from_dem(unreferenced, default=200) == 200
