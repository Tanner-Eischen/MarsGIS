"""Integration tests for data pipeline."""

import numpy as np
import rasterio
from pyproj import CRS

from marshab.core.data_manager import DataManager


class TestDataPipeline:
    """Integration tests for complete data pipeline."""

    def test_load_and_clip_dem(self, synthetic_dem, test_roi):
        dm = DataManager()
        dem = dm.load_dem(synthetic_dem)
        assert dem.shape == (100, 100)
        # This ROI equals the full source footprint. No edge cells may be lost.
        clipped = dm.loader.clip_to_roi(dem, test_roi)
        assert clipped.shape == dem.shape
        np.testing.assert_array_equal(clipped.values, dem.values)
        assert clipped.attrs["transform"] == dem.attrs["transform"]

    def test_dem_metadata(self, synthetic_dem):
        dem = DataManager().load_dem(synthetic_dem)
        assert dem.attrs["resolution_m"] > 0
        with rasterio.open(synthetic_dem) as src:
            assert CRS.from_user_input(dem.attrs["crs"]) == CRS.from_user_input(src.crs)
            assert dem.attrs["transform"] == list(src.transform)[:6]
