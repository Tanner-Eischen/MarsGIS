import numpy as np
from fastapi.testclient import TestClient

from marshab.core.raster_service import RasterWindowResult
from marshab.models import BoundingBox
from marshab.web.api import app
from marshab.web.routes import visualization

client = TestClient(app)


def test_terrain_3d_response_shape(monkeypatch):
    fake_result = RasterWindowResult(
        array=np.arange(100, dtype=np.float32).reshape(10, 10),
        bbox_used=BoundingBox(lat_min=18.0, lat_max=18.6, lon_min=77.0, lon_max=77.8),
        dataset_requested="mola",
        dataset_used="mola",
        is_fallback=False,
        fallback_reason=None,
        resolution_m=200.0,
        nodata=None,
        transform=(0.08, 0.0, 77.0, 0.0, -0.06, 18.6),
    )
    monkeypatch.setattr(visualization, "load_dem_window", lambda *_args, **_kwargs: fake_result)

    roi = "18.0,18.6,77.0,77.8"
    r = client.get(f"/api/v1/visualization/terrain-3d?dataset=mola&roi={roi}&max_points=10000")
    assert r.status_code == 200
    data = r.json()
    for key in ("x", "y", "z", "bounds", "elevation_range"):
        assert key in data
    for key in ("dataset_requested", "dataset_used", "is_fallback", "fallback_reason"):
        assert key in data
    assert isinstance(data["x"], list)
    assert isinstance(data["y"], list)
    assert isinstance(data["z"], list)
