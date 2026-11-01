"""Explorer HTTP contracts using local synthetic numerical fixtures, no network."""

import math
import threading
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from marshab.config import Config, PathsConfig
from marshab.core import explorer_service as service_module
from marshab.core import portfolio_service as portfolio_module
from marshab.core.explorer_service import ExplorerService
from marshab.core.planetary_data import tile_bounds
from marshab.core.portfolio_service import AnalysisAlreadyRunningError, PortfolioService
from marshab.core.workload import acquire_workload, release_workload
from marshab.web.routes import explorer

PAYLOAD = {"start": {"lat": 5, "lon": 10}, "end": {"lat": 8, "lon": 15}}


@pytest.fixture
def service():
    radius = 3396000.0
    stream = BytesIO()
    Image.new("RGBA", (2, 2), (100, 50, 30, 255)).save(stream, format="PNG")

    def tile(layer, z, x, y):
        tile_bounds(z, x, y)
        if layer not in {"mola", "ctx"}:
            raise ValueError("Unknown planetary tile layer")
        return stream.getvalue()

    data = SimpleNamespace(
        global_array=np.zeros((360, 720)),
        global_provenance={
            "id": "mola16",
            "label": "Local numerical fixture",
            "radius_m": radius,
            "native_resolution_m": 2 * math.pi * radius / 720,
        },
        ctx_provenance={
            "id": "ctx",
            "label": "Local numerical fixture",
            "radius_m": 3396190,
            "native_resolution_m": 20,
        },
        ctx_bounds=(77, 18, 77, 18),
        metadata=lambda: {"global": {"id": "mola16"}, "detail": {"id": "ctx"}},
        tile=tile,
    )
    return ExplorerService(data)


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(explorer.router)
    app.state.explorer_service = service
    return TestClient(app)


def completed(client, service, payload=PAYLOAD):
    response = client.post("/api/v1/explorer/routes", json=payload)
    assert response.status_code == 202
    initial = response.json()
    service._thread.join(timeout=10)
    assert not service._thread.is_alive()
    return client.get(initial["poll_url"])


def test_metadata_and_tile_transport(client):
    assert client.get("/api/v1/explorer/metadata").json()["global"]["id"] == "mola16"
    tile = client.get("/api/v1/explorer/tiles/mola/0/0/0.png")
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(tile.content)).size == (2, 2)
    assert client.get("/api/v1/explorer/tiles/mola/0/2/0.png").status_code == 400
    assert client.get("/api/v1/explorer/tiles/unknown/0/0/0.png").status_code == 400


def test_route_result_download_and_unknown_identifiers(client, service):
    response = completed(client, service)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    job = response.json()
    assert job["state"] == "ready"
    result = job["result"]
    for field in (
        "id",
        "source",
        "requested",
        "snapped",
        "path",
        "geojson",
        "distance_m",
        "straight_line_m",
        "terrain_cost_m_equivalent",
        "max_slope_deg",
        "mean_slope_deg",
        "grid",
        "surface",
        "elevation_profile",
    ):
        assert field in result
    assert result["requested"] == PAYLOAD
    assert result["source"]["radius_m"] == 3396000
    assert result["path"][0] == result["snapped"]["start"]
    download = client.get(f"/api/v1/explorer/routes/{job['id']}/geojson")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/geo+json")
    assert "attachment" in download.headers["content-disposition"]
    assert download.json() == result["geojson"]
    assert client.get("/api/v1/explorer/routes/unknown").status_code == 404
    assert client.get("/api/v1/explorer/routes/unknown/geojson").status_code == 404


@pytest.mark.parametrize(
    "changes",
    [
        {"start": {"lat": 91, "lon": 0}},
        {"end": {"lat": 0, "lon": -181}},
        {"max_slope_deg": 0},
        {"max_slope_deg": 90},
        {"max_slope_deg": "nan"},
        {"start": {"lat": True, "lon": 0}},
        {"unexpected": True},
        {"request_id": "../outside"},
        {"request_id": "x" * 65},
        {"request_id": 123},
    ],
)
def test_invalid_payloads_rejected_before_work(client, changes):
    assert client.post("/api/v1/explorer/routes", json={**PAYLOAD, **changes}).status_code == 422


def test_same_requested_point_rejected(client):
    assert (
        client.post(
            "/api/v1/explorer/routes", json={"start": PAYLOAD["start"], "end": PAYLOAD["start"]}
        ).status_code
        == 400
    )


def test_pole_identity_and_failed_route_are_reported_without_download(client, service):
    response = completed(
        client, service, {"start": {"lat": 90, "lon": 0}, "end": {"lat": 90, "lon": 120}}
    )
    job = response.json()
    assert job["state"] == "failed" and "same location" in job["error"]
    assert "result" not in job
    assert client.get(f"/api/v1/explorer/routes/{job['id']}/geojson").status_code == 409


def test_one_scientific_workload_no_queue_and_five_result_retention(
    client, service, monkeypatch, tmp_path
):
    entered, release = threading.Event(), threading.Event()
    real_calculation = service_module.calculate_route

    def blocked(*args):
        entered.set()
        assert release.wait(timeout=10)
        return real_calculation(*args)

    monkeypatch.setattr(service_module, "calculate_route", blocked)
    try:
        response = client.post("/api/v1/explorer/routes", json=PAYLOAD)
        assert response.status_code == 202 and entered.wait(timeout=5)
        assert client.post("/api/v1/explorer/routes", json=PAYLOAD).status_code == 409
        assert client.get(response.json()["poll_url"] + "/geojson").status_code == 409
        portfolio = PortfolioService(
            Config.model_construct(paths=PathsConfig(output_dir=tmp_path / "output")),
            reference_dir=tmp_path / "no-reference",
        )
        with pytest.raises(AnalysisAlreadyRunningError, match="scientific"):
            portfolio.start()
    finally:
        release.set()
        service._thread.join(timeout=10)
    monkeypatch.setattr(service_module, "calculate_route", real_calculation)
    initial_id = response.json()["id"]
    ids = [completed(client, service).json()["id"] for _ in range(5)]
    assert client.get(f"/api/v1/explorer/routes/{initial_id}").status_code == 404
    assert all(
        client.get(f"/api/v1/explorer/routes/{identifier}").status_code == 200 for identifier in ids
    )


def test_existing_portfolio_job_owns_same_slot(client, service, monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked(source, output):
        entered.set()
        assert release.wait(timeout=10)
        raise ValueError("Deliberate local worker failure")

    monkeypatch.setattr(portfolio_module, "run_demo", blocked)
    portfolio = PortfolioService(
        Config.model_construct(paths=PathsConfig(output_dir=tmp_path / "output")),
        reference_dir=tmp_path / "no-reference",
    )
    try:
        portfolio.start()
        assert entered.wait(timeout=5)
        assert client.post("/api/v1/explorer/routes", json=PAYLOAD).status_code == 409
    finally:
        release.set()
        portfolio._thread.join(timeout=10)
    assert completed(client, service).json()["state"] == "ready"


def test_worker_start_failure_releases_shared_slot(service, monkeypatch):
    def rejected(*args, **kwargs):
        raise RuntimeError("No worker resources")

    monkeypatch.setattr(threading.Thread, "start", rejected)
    with pytest.raises(RuntimeError, match="No worker"):
        service.start(PAYLOAD["start"], PAYLOAD["end"], 25)
    assert acquire_workload()
    release_workload()
    assert not service._jobs


def test_client_request_identity_matches_job_result_and_export(client, service):
    job = completed(client, service, {**PAYLOAD, "request_id": "selection-123"}).json()
    assert job["request_id"] == job["result"]["request_id"] == "selection-123"
    exported = client.get(f"/api/v1/explorer/routes/{job['id']}/geojson").json()
    assert exported["request_id"] == job["request_id"]
    assert exported["run_id"] == job["result"]["id"] == job["id"]


def test_wrong_calculated_identity_fails_without_publishing_result(client, service, monkeypatch):
    monkeypatch.setattr(service_module, "calculate_route", lambda *args: {"id": "another-job"})
    job = completed(client, service, {**PAYLOAD, "request_id": "selection-failed"}).json()
    assert job["state"] == "failed" and "identity differs" in job["error"]
    assert job["request_id"] == "selection-failed" and "result" not in job
    assert client.get(f"/api/v1/explorer/routes/{job['id']}/geojson").status_code == 409


def test_missing_pinned_data_returns_503_not_synthetic(monkeypatch):
    def unavailable(config):
        raise FileNotFoundError("Provision the pinned planetary DEM")

    monkeypatch.setattr(explorer, "from_config", unavailable)
    app = FastAPI()
    app.include_router(explorer.router)
    response = TestClient(app).get("/api/v1/explorer/metadata")
    assert response.status_code == 503
    assert "pinned" in response.json()["detail"]


def test_portfolio_thread_construction_failure_releases_shared_slot(monkeypatch, tmp_path):
    service = PortfolioService(
        Config.model_construct(paths=PathsConfig(output_dir=tmp_path / "output")),
        reference_dir=tmp_path / "no-reference",
    )

    def rejected(*args, **kwargs):
        raise RuntimeError("Thread construction failed")

    monkeypatch.setattr(threading, "Thread", rejected)
    with pytest.raises(RuntimeError, match="construction"):
        service.start()
    assert acquire_workload()
    release_workload()
    assert service.status()["state"] == "failed"
