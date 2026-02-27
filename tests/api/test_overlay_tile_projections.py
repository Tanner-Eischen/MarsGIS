"""Regression tests for tiled projection overlays."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marshab.web.api import create_app

FIXTURE_COG = Path(__file__).resolve().parent.parent / "fixtures" / "dem_small.cog.tif"


@pytest.fixture
def client_with_cog(monkeypatch):
    """Client with local fixture COG configured to avoid network downloads."""
    monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(FIXTURE_COG))
    monkeypatch.setenv("MARSHAB_MOLA_COG_URL", "")
    app = create_app()
    return TestClient(app)


@pytest.mark.parametrize(
    ("overlay_type", "query"),
    [
        ("hillshade", {"sun_azimuth": "315", "sun_altitude": "45", "relief": "2.5"}),
        ("slope", {}),
        ("aspect", {}),
        ("roughness", {}),
        ("tri", {}),
        ("solar", {"sun_azimuth": "315", "sun_altitude": "45"}),
    ],
)
def test_projection_overlay_tiles_render_png(client_with_cog, overlay_type, query):
    """Each projection tile endpoint should render a non-empty PNG in tile mode."""
    response = client_with_cog.get(
        f"/api/v1/visualization/tiles/overlay/{overlay_type}/mola/2/4/1.png",
        params=query,
    )
    assert response.status_code == 200
    assert "image/png" in response.headers.get("content-type", "")
    assert len(response.content) > 0


def test_hillshade_relief_changes_rendered_tile(client_with_cog):
    """Hillshade relief should affect tile output so users can exaggerate terrain."""
    low_relief = client_with_cog.get(
        "/api/v1/visualization/tiles/overlay/hillshade/mola/2/4/1.png",
        params={"sun_azimuth": "315", "sun_altitude": "45", "relief": "0.0"},
    )
    high_relief = client_with_cog.get(
        "/api/v1/visualization/tiles/overlay/hillshade/mola/2/4/1.png",
        params={"sun_azimuth": "315", "sun_altitude": "45", "relief": "3.0"},
    )

    assert low_relief.status_code == 200
    assert high_relief.status_code == 200
    assert low_relief.content != high_relief.content
