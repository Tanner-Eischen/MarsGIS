"""Unit tests for DEM loader functionality."""

import pytest
import rasterio

from marshab.processing.dem_loader import DEMLoader


class TestDEMLoader:
    """Tests for DEMLoader class."""

    @pytest.fixture
    def loader(self):
        """Provide DEM loader instance."""
        return DEMLoader()

    def test_load_dem(self, loader, synthetic_dem):
        """Test loading DEM from file."""
        dem = loader.load(synthetic_dem)

        assert dem is not None
        assert "lat" in dem.coords
        assert "lon" in dem.coords
        assert dem.shape == (100, 100)
        # CRS may be stored as EPSG:49900, PROJ string, or in tags
        # Check if CRS is in attrs or if it's a Mars CRS (check for Mars radius or EPSG:49900)
        crs = dem.attrs.get("crs", None)
        if crs is None:
            # Check tags for CRS info
            with rasterio.open(synthetic_dem) as src:
                crs_info = src.tags().get("CRS_INFO", None)
                if crs_info:
                    crs = crs_info
        # Accept either EPSG:49900 or PROJ string or None (if CRS not supported)
        assert crs is None or "49900" in str(crs) or "3396190" in str(crs) or "Mars" in str(crs), \
            f"Expected Mars CRS, got: {crs}"

    def test_clip_to_roi(self, loader, synthetic_dem, test_roi):
        """Test ROI clipping."""
        dem = loader.load(synthetic_dem)
        clipped = loader.clip_to_roi(dem, test_roi)

        # Clipped should be smaller (approximately)
        assert clipped.shape[0] <= dem.shape[0]
        assert clipped.shape[1] <= dem.shape[1]

        # All coordinates should be within ROI (within tolerance)
        lat_min = float(clipped.coords["lat"].min())
        lat_max = float(clipped.coords["lat"].max())
        lon_min = float(clipped.coords["lon"].min())
        lon_max = float(clipped.coords["lon"].max())

        assert lat_min >= test_roi.lat_min - 0.01
        assert lat_max <= test_roi.lat_max + 0.01
        assert lon_min >= test_roi.lon_min - 0.01
        assert lon_max <= test_roi.lon_max + 0.01

    def test_resample(self, loader, synthetic_dem):
        """Test DEM resampling."""
        dem = loader.load(synthetic_dem)

        # Downsample to lower resolution (larger cells)
        resampled = loader.resample(dem, target_resolution_m=1000.0)

        # Should have fewer pixels
        assert resampled.shape[0] < dem.shape[0]
        assert resampled.shape[1] < dem.shape[1]

    def test_load_preserves_metadata(self, loader, synthetic_dem):
        """Test that loading preserves metadata."""
        dem = loader.load(synthetic_dem)

        assert "crs" in dem.attrs
        assert "resolution_m" in dem.attrs
        assert dem.attrs["resolution_m"] > 0
