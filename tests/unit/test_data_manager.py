"""Unit tests for data manager service."""


import io

import pytest

from marshab.core.data_manager import DataManager
from marshab.exceptions import DataError
from marshab.models import BoundingBox


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

    def test_download_file_sets_user_agent(self, data_manager, tmp_path, monkeypatch):
        """Test downloader sends explicit User-Agent for public data hosts."""
        captured = {}

        class _FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()

        def _fake_urlopen(req, timeout=0):
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            captured["timeout"] = timeout
            return _FakeResponse(b"mars-dem-bytes")

        monkeypatch.setattr("marshab.core.data_manager.urllib.request.urlopen", _fake_urlopen)

        dest = tmp_path / "downloaded.tif"
        data_manager._download_file("https://example.com/dem.tif", dest, timeout_s=9)  # noqa: SLF001

        assert dest.read_bytes() == b"mars-dem-bytes"
        assert captured["headers"]["user-agent"] == "MarsHab/1.0"
        assert captured["timeout"] == 9

    def test_download_real_mola_tile_prefers_curated_jezero(self, data_manager, tmp_path, monkeypatch):
        """Test Jezero ROI prefers curated tile path before global MOLA download."""
        monkeypatch.delenv("MARSHAB_ENABLE_CURATED_JEZERO_DEM", raising=False)
        monkeypatch.setenv("MARSHAB_ALLOW_SYNTHETIC_TILES", "false")

        calls = {"curated": 0, "global": 0}
        roi = BoundingBox(lat_min=18.0, lat_max=18.6, lon_min=77.0, lon_max=77.8)
        dest = tmp_path / "mola_lat18_lon77.tif"

        def _fake_curated(path):
            calls["curated"] += 1
            path.write_bytes(b"curated")

        def _fake_global(*_args, **_kwargs):
            calls["global"] += 1
            raise AssertionError("Global MOLA should not be attempted for Jezero when curated is enabled by default")

        monkeypatch.setattr(data_manager, "_download_curated_jezero_tile", _fake_curated)
        monkeypatch.setattr(data_manager, "_download_global_dem_if_missing", _fake_global)

        data_manager._download_real_mola_tile(roi, dest)  # noqa: SLF001

        assert calls["curated"] == 1
        assert calls["global"] == 0
        assert dest.read_bytes() == b"curated"
