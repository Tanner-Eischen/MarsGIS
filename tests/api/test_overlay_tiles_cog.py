"""API tests for overlay tiles when using local COG."""

import sys

import pytest
from fastapi.testclient import TestClient

from marshab.web.api import create_app


@pytest.fixture
def client_with_cog(monkeypatch, dem_small_cog):
    """Client with MARSHAB_MOLA_COG_PATH pointing to fixture (no downloads)."""
    monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(dem_small_cog))
    monkeypatch.setenv("MARSHAB_MOLA_COG_URL", "")
    app = create_app()
    return TestClient(app)


class TestOverlayTilesCog:
    """Tests for overlay endpoints when COG path is configured."""

    def test_dem_image_returns_png_when_cog_set(self, client_with_cog):
        """Overlay dem-image returns 200, image/png, non-empty body when COG path is set."""
        # ROI within fixture bounds (0-2 deg lon, 0-2 deg lat)
        roi = "0.5,1.5,0.5,1.5"  # lat_min, lat_max, lon_min, lon_max
        response = client_with_cog.get(
            f"/api/v1/visualization/dem-image?dataset=mola&roi={roi}&width=200&height=200"
        )
        assert response.status_code == 200
        assert "image/png" in response.headers.get("content-type", "")
        assert len(response.content) > 0

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="dem-window disk cache uses colons in filename, invalid on Windows",
    )
    def test_dem_window_tif_returns_tiff_when_cog_set(self, client_with_cog):
        """dem-window.tif returns 200, image/tiff when COG path is set."""
        roi = "0.5,1.5,0.5,1.5"
        response = client_with_cog.get(
            f"/api/v1/visualization/dem-window.tif?dataset=mola&roi={roi}&width=200&height=200"
        )
        assert response.status_code == 200
        assert "image/tiff" in response.headers.get("content-type", "")
        assert len(response.content) > 0
