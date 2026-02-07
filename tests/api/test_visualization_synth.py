from fastapi.testclient import TestClient

from marshab.web.api import app

client = TestClient(app)


def test_terrain_3d_response_shape():
    roi = "18.0,18.6,77.0,77.8"
    r = client.get(f"/api/v1/visualization/terrain-3d?dataset=mola&roi={roi}&max_points=10000")
    assert r.status_code == 200
    data = r.json()
    for key in ("x", "y", "z", "bounds", "elevation_range"):
        assert key in data
    assert isinstance(data["x"], list)
    assert isinstance(data["y"], list)
    assert isinstance(data["z"], list)
