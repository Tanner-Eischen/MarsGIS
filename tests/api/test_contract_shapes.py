from fastapi.testclient import TestClient

from marshab.web.api import app

client = TestClient(app)


def test_solar_analyze_contract():
    body = {
        "roi": {"lat_min": 18.0, "lat_max": 18.6, "lon_min": 77.0, "lon_max": 77.8},
        "dataset": "mola",
        "sun_azimuth": 180,
        "sun_altitude": 45,
        "panel_efficiency": 0.25,
        "panel_area_m2": 100.0,
        "battery_capacity_kwh": 50.0,
        "daily_power_needs_kwh": 20.0,
        "battery_cost_per_kwh": 1000.0,
        "mission_duration_days": 500.0,
    }
    r = client.post("/api/v1/solar/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    for key in ("solar_potential_map", "irradiance_map", "statistics", "mission_impacts", "shape"):
        assert key in data
    assert isinstance(data["solar_potential_map"], list)
    assert isinstance(data["irradiance_map"], list)
    assert "mean" in data["statistics"]
    assert "power_generation_kwh_per_day" in data["mission_impacts"]
