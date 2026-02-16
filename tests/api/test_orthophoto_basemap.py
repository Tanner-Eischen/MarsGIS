from pathlib import Path

import numpy as np
import rasterio
from fastapi.responses import Response
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

from marshab.web.api import app
from marshab.web.routes import visualization
from marshab.web.routes.visualization import basemap as visualization_basemap

client = TestClient(app)


def _write_test_orthophoto(path: Path) -> None:
    width = 512
    height = 256
    xx = np.linspace(0, 255, width, dtype=np.uint8)
    yy = np.linspace(0, 255, height, dtype=np.uint8)

    red = np.tile(xx, (height, 1))
    green = np.tile(yy.reshape(-1, 1), (1, width))
    blue = np.full((height, width), 96, dtype=np.uint8)

    transform = from_bounds(0.0, -90.0, 360.0, 90.0, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        transform=transform,
        crs=None,
    ) as dst:
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(blue, 3)




def _write_projected_test_orthophoto(path: Path) -> None:
    width = 256
    height = 256
    xx = np.linspace(0, 255, width, dtype=np.uint8)
    yy = np.linspace(0, 255, height, dtype=np.uint8)

    red = np.tile(xx, (height, 1))
    green = np.tile(yy.reshape(-1, 1), (1, width))
    blue = np.full((height, width), 64, dtype=np.uint8)

    # WebMercator world extent. Requests are in EPSG:4326 and must be transformed by the API.
    transform = from_bounds(-20037508.34, -20037508.34, 20037508.34, 20037508.34, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        transform=transform,
        crs="EPSG:3857",
    ) as dst:
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(blue, 3)


def test_orthophoto_basemap_renders_from_configured_source(tmp_path, monkeypatch):
    src_path = tmp_path / "orthophoto.tif"
    _write_test_orthophoto(src_path)
    monkeypatch.setenv("MARSHAB_ORTHO_BASEMAP_PATH", str(src_path))

    response = client.get("/api/v1/visualization/tiles/basemap/orthophoto/2/1/1.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-orthophoto-used"] == "true"
    assert response.headers["x-orthophoto-source"] == src_path.name
    assert len(response.content) > 0


def test_orthophoto_basemap_requires_hirise_source_when_path_not_configured(monkeypatch):
    monkeypatch.delenv("MARSHAB_ORTHO_BASEMAP_PATH", raising=False)

    # Mock path resolution to ensure no auto-discovered source is found
    monkeypatch.setattr(visualization_basemap, "_resolve_orthophoto_source_path", lambda: None)

    async def _stub_basemap(*_args, **_kwargs):
        return Response(content=b"dem-fallback", media_type="image/png")

    monkeypatch.setattr(visualization, "get_basemap_tile", _stub_basemap)

    response = client.get("/api/v1/visualization/tiles/basemap/orthophoto/0/0/0.png")

    assert response.status_code == 503
    assert "Orthophoto source unavailable" in response.json()["detail"]


def test_orthophoto_basemap_renders_projected_source(tmp_path, monkeypatch):
    src_path = tmp_path / "orthophoto_3857.tif"
    _write_projected_test_orthophoto(src_path)
    monkeypatch.setenv("MARSHAB_ORTHO_BASEMAP_PATH", str(src_path))

    response = client.get("/api/v1/visualization/tiles/basemap/orthophoto/2/1/1.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-orthophoto-used"] == "true"
    assert response.headers["x-orthophoto-source"] == src_path.name
    assert "x-orthophoto-crs" in response.headers
    assert len(response.content) > 0


def test_orthophoto_basemap_can_opt_in_to_dem_fallback(monkeypatch):
    monkeypatch.delenv("MARSHAB_ORTHO_BASEMAP_PATH", raising=False)

    # Mock path resolution to ensure no auto-discovered source is found
    monkeypatch.setattr(visualization_basemap, "_resolve_orthophoto_source_path", lambda: None)

    async def _stub_basemap(*_args, **_kwargs):
        return Response(content=b"dem-fallback", media_type="image/png")

    # Patch at basemap module level since the function is called within the same module
    monkeypatch.setattr(visualization_basemap, "get_basemap_tile", _stub_basemap)

    response = client.get(
        "/api/v1/visualization/tiles/basemap/orthophoto/0/0/0.png",
        params={"fallback_dataset": "hirise", "allow_dem_fallback": "true"},
    )

    assert response.status_code == 200
    assert response.content == b"dem-fallback"
    assert response.headers["x-orthophoto-used"] == "false"
    assert response.headers["x-orthophoto-reason"] == "path_not_configured"
