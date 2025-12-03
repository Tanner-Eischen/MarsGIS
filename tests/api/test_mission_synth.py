from fastapi.testclient import TestClient
from marshab.web.api import app

client = TestClient(app)


def enable_demo_mode():
    client.post("/api/v1/demo-mode", json={"enabled": True})


def test_landing_scenario_demo():
    enable_demo_mode()
    body = {
        "roi": {"lat_min": 18.0, "lat_max": 18.6, "lon_min": 77.0, "lon_max": 77.8},
        "dataset": "mola",
        "suitability_threshold": 0.7,
    }
    r = client.post("/api/v1/mission/landing-scenario", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "scenario_id" in data and "sites" in data
    assert isinstance(data["sites"], list) and len(data["sites"]) > 0


def test_rover_traverse_demo():
    enable_demo_mode()
    body = {
        "start_site_id": 1,
        "end_site_id": 2,
        "analysis_dir": "data/output",
        "start_lat": 18.3,
        "start_lon": 77.4,
    }
    r = client.post("/api/v1/mission/rover-traverse", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "route_id" in data and "waypoints" in data
    assert isinstance(data["waypoints"], list) and len(data["waypoints"]) > 0