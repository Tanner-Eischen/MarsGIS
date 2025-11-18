"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

from marshab.config import Config, PathsConfig, AnalysisConfig
from marshab.types import BoundingBox, SiteOrigin, CriteriaWeights


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Provide test configuration with temporary directories."""
    return Config(
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
        ),
        analysis=AnalysisConfig(
            criteria_weights=CriteriaWeights(),
            max_slope_deg=5.0,
            max_roughness=0.5,
            min_site_area_km2=0.5,
            suitability_threshold=0.7
        )
    )


@pytest.fixture
def test_roi() -> BoundingBox:
    """Provide test region of interest."""
    return BoundingBox(
        lat_min=40.0,
        lat_max=41.0,
        lon_min=180.0,
        lon_max=181.0
    )


@pytest.fixture
def test_site_origin() -> SiteOrigin:
    """Provide test SITE frame origin."""
    return SiteOrigin(
        lat=40.5,
        lon=180.5,
        elevation_m=-2500.0
    )


@pytest.fixture
def synthetic_dem(tmp_path: Path) -> Path:
    """Create synthetic Mars DEM for testing.

    Creates a 100x100 elevation raster with:
    - Random noise
    - Gentle slopes
    - IAU_MARS coordinate system
    - Coverage over test ROI

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to synthetic DEM GeoTIFF file
    """
    np.random.seed(42)

    width, height = 100, 100
    # Mean elevation 1000m with small variations
    elevation = np.random.randn(height, width) * 100 + 1000.0

    # Add gentle slopes
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    elevation += x * 0.5 + y * 0.3

    # Define bounds in IAU_MARS coordinates (covering test ROI)
    bounds = (180.0, 40.0, 181.0, 41.0)  # lon_min, lat_min, lon_max, lat_max
    transform = from_bounds(*bounds, width, height)

    # Write to GeoTIFF - use WGS84 as fallback since EPSG:49900 may not be available
    # In production, Mars CRS would be properly configured
    dem_path = tmp_path / "synthetic_dem.tif"

    try:
        # Try Mars CRS first
        crs = "EPSG:49900"  # Mars 2000 sphere IAU
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=elevation.dtype,
            crs=crs,
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(elevation, 1)
    except rasterio.errors.CRSError:
        # Fallback to WGS84 if Mars CRS not available
        # Use a simple geographic CRS for testing
        crs = None  # No CRS - just use transform
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=elevation.dtype,
            crs=crs,
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(elevation, 1)

    return dem_path


@pytest.fixture
def sample_terrain_data() -> np.ndarray:
    """Provide sample terrain data for testing.

    Returns:
        50x50 array of synthetic elevation data
    """
    np.random.seed(42)
    return np.random.randn(50, 50) * 10 + 100


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config singleton after each test."""
    from marshab.config import reset_config
    yield
    reset_config()

