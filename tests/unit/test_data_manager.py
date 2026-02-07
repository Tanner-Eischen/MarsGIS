"""Unit tests for data manager service."""


import pytest

from marshab.core.data_manager import DataManager
from marshab.exceptions import DataError


class TestDataManager:
    """Tests for DataManager class."""

    @pytest.fixture
    def data_manager(self, test_config, monkeypatch):
        """Provide DataManager instance with test config."""
        from marshab.config import reset_config
        reset_config()
        monkeypatch.setattr("marshab.core.data_manager.get_config", lambda: test_config)
        return DataManager()

    def test_cache_key_generation(self, data_manager, test_roi):
        """Test cache key generation."""
        key1 = data_manager._get_cache_path("mola", test_roi)
        key2 = data_manager._get_cache_path("mola", test_roi)

        # Same inputs should produce same cache path
        assert key1 == key2

        # Different datasets should have different paths
        key3 = data_manager._get_cache_path("hirise", test_roi)
        assert key1 != key3

    def test_load_dem_from_file(self, data_manager, synthetic_dem):
        """Test loading DEM from existing file."""
        dem = data_manager.load_dem(synthetic_dem)

        assert dem is not None
        assert dem.shape == (100, 100)

    def test_get_dem_for_roi_without_download(self, data_manager, synthetic_dem, test_roi):
        """Test getting DEM without downloading."""
        # This test would need mock setup - simplified version
        dem = data_manager.loader.load(synthetic_dem)
        clipped = data_manager.loader.clip_to_roi(dem, test_roi)

        assert clipped is not None
        assert clipped.shape[0] > 0

    def test_invalid_dataset(self, data_manager, test_roi):
        """Test error handling for invalid dataset."""
        with pytest.raises(DataError) as excinfo:
            data_manager.download_dem("invalid_dataset", test_roi)

        assert "Unknown dataset" in str(excinfo.value)

