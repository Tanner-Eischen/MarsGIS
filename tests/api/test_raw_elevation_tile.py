"""Tests for raw elevation tile endpoint."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from marshab.core.raster_service import RasterWindowResult
from marshab.models import BoundingBox
from marshab.web.api import create_app
from marshab.web.routes.visualization import terrain as visualization_terrain

FIXTURE_COG = Path(__file__).resolve().parent.parent / "fixtures" / "dem_small.cog.tif"


@pytest.fixture
def client_with_cog(monkeypatch):
    monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(FIXTURE_COG))
    monkeypatch.setenv("MARSHAB_MOLA_COG_URL", "")
    app = create_app()
    return TestClient(app)


def _decode_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def test_raw_elevation_tile_returns_npy_with_native_samples(client_with_cog):
    response = client_with_cog.get("/api/v1/visualization/tiles/elevation-raw/mola/7/128/63.npy")
    assert response.status_code == 200
    assert "application/x-npy" in response.headers.get("content-type", "")

    arr = _decode_npy(response.content)
    assert arr.ndim == 2
    assert arr.dtype == np.float32
    assert arr.size > 0

    shape_header = response.headers.get("X-Array-Shape")
    assert shape_header is not None
    rows, cols = map(int, shape_header.split(","))
    assert (rows, cols) == arr.shape
    assert response.headers.get("X-Subsample-Stride") == "1"


def test_raw_elevation_tile_stride_is_nearest_neighbor_decimation(client_with_cog):
    base = client_with_cog.get("/api/v1/visualization/tiles/elevation-raw/mola/7/128/63.npy")
    decimated = client_with_cog.get("/api/v1/visualization/tiles/elevation-raw/mola/7/128/63.npy?stride=2")

    assert base.status_code == 200
    assert decimated.status_code == 200

    arr_base = _decode_npy(base.content)
    arr_decimated = _decode_npy(decimated.content)
    assert np.array_equal(arr_decimated, arr_base[::2, ::2])


def test_raw_elevation_tile_does_not_request_target_shape(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_load_dem_window(dataset, bbox, target_shape=None, allow_download=True):
        captured["dataset"] = dataset
        captured["target_shape"] = target_shape
        captured["allow_download"] = allow_download
        return RasterWindowResult(
            array=np.arange(17 * 29, dtype=np.float32).reshape(17, 29),
            bbox_used=BoundingBox(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0),
            dataset_requested="mola",
            dataset_used="mola",
            is_fallback=False,
            fallback_reason=None,
            resolution_m=463.0,
            nodata=-32768.0,
            transform=None,
        )

    monkeypatch.setattr(visualization_terrain, "load_dem_window", _fake_load_dem_window)
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/visualization/tiles/elevation-raw/mola/5/16/15.npy")
    assert response.status_code == 200
    arr = _decode_npy(response.content)
    assert arr.shape == (17, 29)
    assert captured["target_shape"] is None
