from fastapi.testclient import TestClient

from marshab.exceptions import DataError
from marshab.web.api import app
from marshab.web.routes import visualization

client = TestClient(app)


def _force_real_dem_unavailable(monkeypatch):
    def _raise_real_dem(*_args, **_kwargs):
        raise DataError("real_dem_unavailable")

    monkeypatch.setattr(visualization, "load_dem_window", _raise_real_dem)
    monkeypatch.setattr(visualization, "read_disk_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(visualization.TILE_CACHE, "get", lambda *_args, **_kwargs: None)


def _assert_real_dem_unavailable(response):
    assert response.status_code == 503
    payload = response.json()
    if isinstance(payload, dict) and "error" in payload:
        assert payload["error"] == "real_dem_unavailable"
        return

    detail = payload["detail"]
    if isinstance(detail, dict):
        assert detail.get("error") == "real_dem_unavailable"
    else:
        assert detail == "real_dem_unavailable"


def test_basemap_tile_returns_503_when_real_dem_unavailable(monkeypatch):
    _force_real_dem_unavailable(monkeypatch)

    response = client.get("/api/v1/visualization/tiles/basemap/mola/2/1/1.png")

    _assert_real_dem_unavailable(response)


def test_overlay_tile_returns_503_when_real_dem_unavailable(monkeypatch):
    _force_real_dem_unavailable(monkeypatch)

    response = client.get("/api/v1/visualization/tiles/overlay/elevation/mola/2/1/1.png")

    _assert_real_dem_unavailable(response)


def test_terrain_3d_returns_503_when_real_dem_unavailable(monkeypatch):
    _force_real_dem_unavailable(monkeypatch)

    response = client.get(
        "/api/v1/visualization/terrain-3d",
        params={"dataset": "mola", "roi": "18.0,18.6,77.0,77.8"},
    )

    _assert_real_dem_unavailable(response)
