"""Pytest configuration and shared fixtures."""

import os
import socket
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_bounds

from marshab.config import AnalysisConfig, Config, PathsConfig
from marshab.types import BoundingBox, CriteriaWeights, SiteOrigin


@pytest.fixture(scope="session")
def dem_small_cog(tmp_path_factory) -> Path:
    """Generate the small reader/tile fixture locally, without a binary download.

    This legacy reader fixture deliberately uses CRS_INFO metadata, matching
    scripts/create_dem_fixture.py. Scientific CRS contracts have separate tests.
    """
    path = tmp_path_factory.mktemp("raster-fixtures") / "dem_small.cog.tif"
    rng = np.random.default_rng(42)
    elevation = rng.integers(-2000, 3000, (256, 256), dtype=np.int16)
    elevation[:128, 128:] = -32768
    with rasterio.open(
        path, "w", driver="GTiff", height=256, width=256, count=1,
        dtype="int16", transform=from_bounds(0, 0, 2, 2, 256, 256),
        nodata=-32768, tiled=True, blockxsize=64, blockysize=64, compress="lzw",
    ) as dst:
        dst.write(elevation, 1)
        dst.update_tags(CRS_INFO="EPSG:49900")
    return path


@pytest.fixture(autouse=True)
def isolated_offline_runtime(monkeypatch, tmp_path):
    """No test inherits developer data, deployment settings, or network access.

    Tests that exercise synthetic fallback must opt in explicitly. Downloader
    unit tests replace urlopen with their own local response objects.
    """
    from marshab.config import reset_config
    from marshab.web.routes.visualization._helpers import TILE_CACHE

    for key in list(os.environ):
        if key.startswith("MARSHAB_"):
            monkeypatch.delenv(key)
    config_path = tmp_path / "isolated_config.yaml"
    config_path.write_text(yaml.safe_dump({
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "output"),
        },
    }), encoding="utf-8")
    monkeypatch.setenv("MARSHAB_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MARSHAB_ALLOW_SYNTHETIC_TILES", "false")
    monkeypatch.setenv("MARSHAB_PREWARM_COG", "false")

    def no_download(*args, **kwargs):
        raise OSError("Network disabled in tests; provide a local fixture or mock response")

    original_connect = socket.socket.connect

    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise OSError(f"Network disabled in tests: {address[0]}")
        return original_connect(sock, address)

    monkeypatch.setattr(urllib.request, "urlopen", no_download)
    monkeypatch.setattr(socket.socket, "connect", local_connect)
    original_raster_open = rasterio.open

    def local_raster_open(path, *args, **kwargs):
        # GDAL HTTP I/O bypasses Python's socket layer.
        source = str(path).lower()
        if "://" in source or source.startswith(("/vsicurl/", "/vsis3/", "/vsiaz/", "/vsigs/")):
            raise OSError("Remote raster I/O disabled in tests; provide a local fixture")
        return original_raster_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", local_raster_open)
    TILE_CACHE._cache.clear()
    reset_config()
    yield
    TILE_CACHE._cache.clear()
    reset_config()


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

    # Write to GeoTIFF - define Mars CRS using PROJ string since EPSG:49900 may not be available
    # Mars IAU 2000 sphere: radius = 3396190 m (equatorial)
    dem_path = tmp_path / "synthetic_dem.tif"

    # Define Mars CRS using PROJ string (IAU_MARS_2000 sphere)
    # This is equivalent to EPSG:49900 but uses PROJ string format
    mars_crs_proj = "+proj=longlat +a=3396190 +b=3396190 +no_defs +type=crs"
    mars_crs_epsg = "EPSG:49900"  # For reference

    crs_used = None
    crs_string = None

    try:
        # Try EPSG code first (if available in PROJ database)
        crs_used = mars_crs_epsg
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=elevation.dtype,
            crs=crs_used,
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(elevation, 1)
        crs_string = mars_crs_epsg
    except rasterio.errors.CRSError:
        try:
            # Try PROJ string format
            crs_used = mars_crs_proj
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=elevation.dtype,
                crs=crs_used,
                transform=transform,
                nodata=-9999,
            ) as dst:
                dst.write(elevation, 1)
            crs_string = mars_crs_epsg  # Store EPSG code in metadata even if using PROJ string
        except rasterio.errors.CRSError:
            # Final fallback: no CRS, just use transform
            # Store CRS info in file attributes instead
            crs_used = None
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=elevation.dtype,
                crs=crs_used,
                transform=transform,
                nodata=-9999,
            ) as dst:
                dst.write(elevation, 1)
            crs_string = mars_crs_epsg  # Store in metadata for tests

    # Store CRS info in file metadata for test access
    # Reopen and add CRS metadata
    try:
        with rasterio.open(dem_path, "r+") as dst:
            dst.update_tags(CRS_INFO=crs_string or mars_crs_epsg)
    except Exception:
        pass  # If we can't update tags, that's okay

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

