"""Integration tests for data pipeline."""

import pytest

from marshab.core.data_manager import DataManager
from marshab.types import BoundingBox


class TestDataPipeline:
    """Integration tests for complete data pipeline."""

    def test_load_and_clip_dem(self, synthetic_dem, test_roi):
        """Test complete pipeline: load, validate, clip."""
        dm = DataManager()

        # Load DEM
        dem = dm.load_dem(synthetic_dem)
        assert dem.shape == (100, 100)

        # Clip to ROI
        clipped = dm.loader.clip_to_roi(dem, test_roi)
        assert clipped.shape[0] < dem.shape[0]

    def test_dem_metadata(self, synthetic_dem):
        """Test DEM metadata is preserved."""
        dm = DataManager()
        dem = dm.load_dem(synthetic_dem)

        assert "crs" in dem.attrs
        assert "resolution_m" in dem.attrs
        assert dem.attrs["crs"] == "EPSG:49900"





