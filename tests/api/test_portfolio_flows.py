"""Portfolio flow acceptance tests (Week 3/4)."""

from pathlib import Path

from fastapi.testclient import TestClient

from marshab.config import reset_config
from marshab.web.api import app


def _configure_paths(monkeypatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        (
            "paths:\n"
            f"  data_dir: \"{data_dir.as_posix()}\"\n"
            f"  cache_dir: \"{cache_dir.as_posix()}\"\n"
            f"  output_dir: \"{output_dir.as_posix()}\"\n"
            "navigation:\n"
            "  max_roughness_m: 100.0\n"
            "  cliff_threshold_m: 1000.0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MARSHAB_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MARSHAB_DEMO_SEED", "42")
    reset_config()
    return output_dir


def _site_scores_payload() -> dict:
    return {
        "roi": {
            "lat_min": 40.0,
            "lat_max": 40.3,
            "lon_min": 180.0,
            "lon_max": 180.3,
        },
        "dataset": "mola",
        "threshold": 0.5,
        "preset_id": "balanced",
    }


def test_site_selection_and_geojson_overlay(monkeypatch, tmp_path: Path):
    output_dir = _configure_paths(monkeypatch, tmp_path)
    client = TestClient(app)

    scores_response = client.post("/api/v1/analysis/site-scores", json=_site_scores_payload())
    assert scores_response.status_code == 200
    scores = scores_response.json()
    assert len(scores) >= 1

    sites_csv = output_dir / "sites.csv"
    assert sites_csv.exists()

    geojson_response = client.get("/api/v1/visualization/sites-geojson")
    assert geojson_response.status_code == 200
    geojson = geojson_response.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 1


def test_route_planning_export_and_invalid_state(monkeypatch, tmp_path: Path):
    output_dir = _configure_paths(monkeypatch, tmp_path)
    client = TestClient(app)

    scores_response = client.post("/api/v1/analysis/site-scores", json=_site_scores_payload())
    assert scores_response.status_code == 200
    top_site_id = scores_response.json()[0]["site_id"]

    nav_response = None
    last_error = None
    start_candidates = [
        (40.05, 180.05),
        (40.10, 180.10),
        (40.20, 180.20),
        (40.25, 180.25),
    ]
    for start_lat, start_lon in start_candidates:
        candidate = client.post(
            "/api/v1/navigation/plan-route",
            json={
                "site_id": top_site_id,
                "analysis_dir": str(output_dir),
                "start_lat": start_lat,
                "start_lon": start_lon,
                "max_waypoint_spacing_m": 20.0,
                "max_slope_deg": 25.0,
            },
        )
        if candidate.status_code == 200 and candidate.json().get("num_waypoints", 0) > 2:
            nav_response = candidate
            break
        last_error = candidate.json().get("detail")

    assert nav_response is not None, f"Route planning failed for all start candidates: {last_error}"
    nav_data = nav_response.json()
    assert nav_data["num_waypoints"] > 2

    waypoints_file = output_dir / f"waypoints_site_{top_site_id}.csv"
    assert waypoints_file.exists()

    geojson_response = client.get(
        "/api/v1/navigation/waypoints-geojson",
        params={"analysis_dir": str(output_dir), "site_id": top_site_id},
    )
    assert geojson_response.status_code == 200
    waypoints_geojson = geojson_response.json()
    assert waypoints_geojson["type"] == "FeatureCollection"
    assert len(waypoints_geojson["features"]) > 1

    invalid_response = client.post(
        "/api/v1/navigation/plan-route",
        json={
            "site_id": 999999,
            "analysis_dir": str(output_dir),
            "start_lat": 40.15,
            "start_lon": 180.15,
        },
    )
    assert invalid_response.status_code == 400
    assert "detail" in invalid_response.json()


def test_decision_brief_is_deterministic(monkeypatch, tmp_path: Path):
    _configure_paths(monkeypatch, tmp_path)
    client = TestClient(app)
    payload = {
        "roi": {
            "lat_min": 40.0,
            "lat_max": 40.3,
            "lon_min": 180.0,
            "lon_max": 180.3,
        },
        "dataset": "mola",
        "threshold": 0.5,
        "preset_id": "balanced",
        "start_lat": 40.15,
        "start_lon": 180.15,
    }

    first = client.post("/api/v1/analysis/decision-brief", json=payload)
    second = client.post("/api/v1/analysis/decision-brief", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    first_data = first.json()
    second_data = second.json()

    for key in ("site_id", "rank", "suitability_score", "summary", "reasons", "terrain", "route_impacts"):
        assert first_data[key] == second_data[key]

    assert first_data["summary"]
    assert len(first_data["reasons"]) >= 3
    assert "estimated_energy_j" in first_data["route_impacts"]
