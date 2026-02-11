from fastapi.testclient import TestClient

from marshab.core.raster_service import REAL_DEM_UNAVAILABLE
from marshab.exceptions import DataError
from marshab.web.api import app
from marshab.web.routes import visualization

client = TestClient(app)


def _assert_real_dem_payload(payload: dict) -> None:
    for key in (
        "error",
        "detail",
        "dataset_requested",
        "dataset_used",
        "is_fallback",
        "fallback_reason",
    ):
        assert key in payload
    assert payload["error"] == REAL_DEM_UNAVAILABLE


def test_openapi_documents_503_real_dem_unavailable():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for path in (
        "/api/v1/visualization/tiles/basemap/{dataset}/{z}/{x}/{y}.png",
        "/api/v1/visualization/tiles/overlay/{overlay_type}/{dataset}/{z}/{x}/{y}.png",
        "/api/v1/visualization/terrain-3d",
    ):
        assert "503" in paths[path]["get"]["responses"]


def test_tile_basemap_real_dem_unavailable_payload(monkeypatch):
    def _raise_unavailable(*_args, **_kwargs):
        raise DataError(REAL_DEM_UNAVAILABLE)

    monkeypatch.setattr(visualization, "load_dem_window", _raise_unavailable)
    response = client.get("/api/v1/visualization/tiles/basemap/mola/0/0/0.png")
    assert response.status_code == 503
    payload = response.json()
    _assert_real_dem_payload(payload)
    assert payload["dataset_requested"] == "mola"
    assert payload["dataset_used"] == "mola"


def test_terrain_3d_real_dem_unavailable_payload(monkeypatch):
    def _raise_unavailable(*_args, **_kwargs):
        raise DataError(REAL_DEM_UNAVAILABLE)

    monkeypatch.setattr(visualization, "load_dem_window", _raise_unavailable)
    response = client.get("/api/v1/visualization/terrain-3d?dataset=mola&roi=18.0,18.6,77.0,77.8")
    assert response.status_code == 503
    payload = response.json()
    _assert_real_dem_payload(payload)
    assert payload["dataset_requested"] == "mola"
    assert payload["dataset_used"] is None


def test_terrain_3d_rejects_invalid_roi_bounds():
    response = client.get("/api/v1/visualization/terrain-3d?dataset=mola&roi=18.6,18.0,77.0,77.8")
    assert response.status_code == 400
    assert "lat_min" in str(response.json()["detail"])


def test_terrain_3d_rejects_oversized_roi():
    response = client.get("/api/v1/visualization/terrain-3d?dataset=mola&roi=-10,60,0,200")
    assert response.status_code == 400
    assert "span" in str(response.json()["detail"]).lower()
