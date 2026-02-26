"""Unit tests for COG window reader."""

from pathlib import Path

import numpy as np
import pytest

from marshab.core.cog_reader import read_cog_window
from marshab.models import BoundingBox

FIXTURE_COG = Path(__file__).resolve().parent.parent / "fixtures" / "dem_small.cog.tif"


@pytest.fixture
def fixture_cog_path():
    """Path to small COG fixture."""
    return str(FIXTURE_COG)


@pytest.fixture
def bbox_within_fixture():
    """Bounding box within fixture bounds (0-2 deg lon, 0-2 deg lat)."""
    return BoundingBox(lat_min=0.5, lat_max=1.5, lon_min=0.5, lon_max=1.5)


@pytest.fixture
def bbox_full_fixture():
    """Bounding box covering full fixture."""
    return BoundingBox(lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0)


class TestCogReader:
    """Tests for read_cog_window."""

    def test_correct_shape_without_target(self, fixture_cog_path, bbox_within_fixture):
        """Correct shape for bbox without target_shape."""
        result = read_cog_window(fixture_cog_path, bbox_within_fixture)
        assert result.array.ndim == 2
        assert result.array.shape[0] > 0
        assert result.array.shape[1] > 0
        # Window should be a subset of 256x256
        assert result.array.shape[0] <= 256
        assert result.array.shape[1] <= 256

    def test_correct_shape_with_target(self, fixture_cog_path, bbox_within_fixture):
        """Correct shape for bbox with target_shape."""
        target = (100, 150)
        result = read_cog_window(fixture_cog_path, bbox_within_fixture, target_shape=target)
        assert result.array.shape == target

    def test_preserves_nodata(self, fixture_cog_path, bbox_full_fixture):
        """Preserves nodata values (-32768)."""
        result = read_cog_window(fixture_cog_path, bbox_full_fixture)
        assert result.nodata == -32768.0
        # Fixture has nodata in top-right quadrant
        nodata_count = np.sum(result.array == -32768.0)
        assert nodata_count > 0

    def test_non_empty_arrays(self, fixture_cog_path, bbox_within_fixture):
        """Non-empty arrays returned."""
        result = read_cog_window(fixture_cog_path, bbox_within_fixture)
        assert result.array.size > 0
        # Should have some valid (non-nodata) values in this bbox
        valid = np.isfinite(result.array) & (result.array != result.nodata)
        assert np.any(valid)

    def test_valid_transform_when_resampling(self, fixture_cog_path, bbox_within_fixture):
        """Valid transform when resampling to target_shape."""
        target = (64, 64)
        result = read_cog_window(fixture_cog_path, bbox_within_fixture, target_shape=target)
        assert result.transform is not None
        assert len(result.transform) == 6
        # Transform should map to bbox bounds
        a, b, c, d, e, f = result.transform
        # Pixel (0,0) -> (c, f), pixel (64,64) -> (c+64*a, f+64*e)
        assert np.isfinite(c) and np.isfinite(f)

    def test_valid_transform_without_resampling(self, fixture_cog_path, bbox_within_fixture):
        """Valid transform when not resampling."""
        result = read_cog_window(fixture_cog_path, bbox_within_fixture)
        assert result.transform is not None
        assert len(result.transform) == 6

    def test_partial_window_near_edge(self, fixture_cog_path):
        """Handles partial windows near COG edges."""
        # Small bbox at bottom-left corner
        bbox = BoundingBox(lat_min=0.0, lat_max=0.5, lon_min=0.0, lon_max=0.5)
        result = read_cog_window(fixture_cog_path, bbox)
        assert result.array.shape[0] > 0
        assert result.array.shape[1] > 0
        assert result.dataset_used == "mola"
        assert result.resolution_m == 463.0

    def test_dataset_metadata(self, fixture_cog_path, bbox_within_fixture):
        """Result has correct dataset metadata."""
        result = read_cog_window(fixture_cog_path, bbox_within_fixture)
        assert result.dataset_requested == "mola"
        assert result.dataset_used == "mola"
        assert result.is_fallback is False
        assert result.fallback_reason is None
        assert result.resolution_m == 463.0
