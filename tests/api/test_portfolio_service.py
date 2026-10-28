"""Observed-DEM API job orchestration over real verified reference artifacts."""

import shutil
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marshab.analysis.site_package import DOWNLOADS
from marshab.config import Config, PathsConfig
from marshab.core import portfolio_service as service_module
from marshab.core.portfolio_service import PortfolioService
from marshab.web.routes.portfolio import router

REFERENCE = Path(__file__).resolve().parents[2] / "webui" / "public" / "portfolio"


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.delenv("MARSHAB_REFERENCE_DEM_PATH", raising=False)
    monkeypatch.delenv("MARSHAB_PORTFOLIO_DEM_PATH", raising=False)
    config = Config.model_construct(
        paths=PathsConfig(
            data_dir=tmp_path, cache_dir=tmp_path / "cache", output_dir=tmp_path / "output"
        )
    )
    return PortfolioService(config, reference_dir=REFERENCE)


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(router)
    app.state.portfolio_service = service
    return TestClient(app)


def copy_analysis(source, output):
    """Mock only the costly numerical job, keeping actual export/package verification."""
    output.mkdir(parents=True)
    for _, _, filename, _ in DOWNLOADS:
        shutil.copyfile(REFERENCE / filename, output / filename)


def complete(service):
    service._thread.join(timeout=15)
    assert not service._thread.is_alive(), "Worker did not complete"


def test_verified_initial_reference_and_allowlisted_download(client):
    status = client.get("/api/v1/portfolio/status").json()
    assert status["state"] == "ready" and status["data_available"]
    assert len(status["source"]["sha256"]) == 64
    scene = client.get(status["scene_url"]).json()
    assert scene["execution_mode"] == "precomputed_reference_analysis"
    assert scene["asset_base"].startswith("/api/v1/portfolio/files/reference-")
    response = client.get(scene["asset_base"] + "manifest.json")
    assert response.status_code == 200
    assert response.content == (REFERENCE / "manifest.json").read_bytes()
    assert "immutable" in response.headers["cache-control"]
    for path in (
        "missing.txt",
        "scene.json",
        "../config.py",
        "layers/../../config.py",
        "C:%5CWindows%5Cwin.ini",
    ):
        assert client.get(scene["asset_base"] + path).status_code == 404
    assert client.get("/api/v1/portfolio/files/unknown/probe.bin").status_code == 404


def test_single_concurrent_job_and_atomic_publication(client, service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(source, output):
        calls.append((source, output))
        entered.set()
        assert release.wait(timeout=10)
        copy_analysis(source, output)

    monkeypatch.setattr(service_module, "run_demo", blocked)
    old = client.get("/api/v1/portfolio/scene").json()
    try:
        response = client.post("/api/v1/portfolio/run")
        assert response.status_code == 202
        assert entered.wait(timeout=5)
        status = client.get("/api/v1/portfolio/status").json()
        assert status["state"] == "running" and status["data_available"]
        duplicate = client.post("/api/v1/portfolio/run")
        assert duplicate.status_code == 409
        assert duplicate.json()["state"] == "running"
        assert client.get("/api/v1/portfolio/scene").json() == old
    finally:
        release.set()
        complete(service)
    assert len(calls) == 1 and calls[0][0] == service.source_path
    assert client.get("/api/v1/portfolio/status").json()["state"] == "ready"
    fresh = client.get("/api/v1/portfolio/scene").json()
    assert fresh["execution_mode"] == "server_computed_analysis"
    assert fresh["run_id"] != old["run_id"] and fresh["computed_at_utc"]
    assert fresh["asset_base"] != old["asset_base"]
    assert client.get(old["asset_base"] + "probe.bin").status_code == 200
    assert (
        client.get(fresh["asset_base"] + "probe.bin").content
        == (REFERENCE / "probe.bin").read_bytes()
    )


def test_failed_job_retains_last_good_result(client, service, monkeypatch):
    old = client.get("/api/v1/portfolio/scene").json()

    def failed(source, output):
        raise ValueError("Pinned DEM checksum mismatch")

    monkeypatch.setattr(service_module, "run_demo", failed)
    assert client.post("/api/v1/portfolio/run").status_code == 202
    complete(service)
    status = client.get("/api/v1/portfolio/status").json()
    assert status["state"] == "failed" and status["data_available"]
    assert "checksum mismatch" in status["error"]
    assert client.get("/api/v1/portfolio/scene").json() == old


def test_prior_live_run_files_survive_new_run_and_tampering_is_rejected(
    client, service, monkeypatch
):
    monkeypatch.setattr(service_module, "run_demo", copy_analysis)
    assert client.post("/api/v1/portfolio/run").status_code == 202
    complete(service)
    first = client.get("/api/v1/portfolio/scene").json()
    assert client.post("/api/v1/portfolio/run").status_code == 202
    complete(service)
    second = client.get("/api/v1/portfolio/scene").json()
    assert first["run_id"] != second["run_id"]
    assert client.get(first["asset_base"] + "report.md").status_code == 200
    artifact = service.file(second["run_id"], "report.md")
    artifact.write_text("changed after verification", encoding="utf-8")
    assert client.get(second["asset_base"] + "report.md").status_code == 409


def test_invalid_reference_does_not_claim_available_data(service, tmp_path):
    config = Config.model_construct(paths=PathsConfig(output_dir=tmp_path / "jobs"))
    invalid = PortfolioService(config, reference_dir=tmp_path / "missing")
    assert invalid.status()["state"] == "failed"
    assert invalid.status()["data_available"] is False
    with pytest.raises(FileNotFoundError):
        invalid.scene()


def test_lazy_app_service_initializes_idle_once(monkeypatch, tmp_path):
    from marshab.web.routes import portfolio

    config = Config.model_construct(paths=PathsConfig(cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(portfolio, "get_config", lambda: config)
    monkeypatch.setattr(PortfolioService, "_find_reference", staticmethod(lambda _: None))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get("/api/v1/portfolio/status")
    assert response.json()["state"] == "idle"
    assert response.json()["data_available"] is False
    first_service = app.state.portfolio_service
    assert client.get("/api/v1/portfolio/status").status_code == 200
    assert app.state.portfolio_service is first_service
    assert client.get("/api/v1/portfolio/scene").status_code == 404


def test_package_verification_failure_does_not_publish_partial_run(client, service, monkeypatch):
    original = client.get("/api/v1/portfolio/scene").json()
    monkeypatch.setattr(service_module, "run_demo", copy_analysis)

    def rejected(source, output):
        raise ValueError("Package verification rejected outputs")

    monkeypatch.setattr(service_module, "build_package", rejected)
    assert client.post("/api/v1/portfolio/run").status_code == 202
    complete(service)
    status = client.get("/api/v1/portfolio/status").json()
    assert status["state"] == "failed" and status["data_available"]
    assert "verification rejected" in status["error"]
    assert client.get("/api/v1/portfolio/scene").json() == original


def test_only_five_successful_live_runs_retained_with_reference(client, service, monkeypatch):
    monkeypatch.setattr(service_module, "run_demo", copy_analysis)
    reference = client.get("/api/v1/portfolio/scene").json()
    scenes = []
    for _ in range(6):
        assert client.post("/api/v1/portfolio/run").status_code == 202
        complete(service)
        assert service.status()["state"] == "ready"
        scenes.append(client.get("/api/v1/portfolio/scene").json())
    assert service.status()["retained_live_runs"] == 5
    assert client.get(scenes[0]["asset_base"] + "report.md").status_code == 404
    assert not (service.output_root / scenes[0]["run_id"]).exists()
    for scene in scenes[1:]:
        assert client.get(scene["asset_base"] + "report.md").status_code == 200
        assert (service.output_root / scene["run_id"]).is_dir()
    assert client.get(reference["asset_base"] + "report.md").status_code == 200
    assert client.get("/api/v1/portfolio/scene").json()["run_id"] == scenes[-1]["run_id"]


def test_failed_partial_output_is_removed_without_touching_source(client, service, monkeypatch):
    folders = []
    service.source_path.parent.mkdir(parents=True)
    service.source_path.write_bytes(b"source survives failure cleanup")

    def failed(source, output):
        folders.append(output.parent)
        output.mkdir(parents=True)
        (output / "partial.txt").write_text("incomplete", encoding="utf-8")
        raise ValueError("Deliberate incomplete calculation")

    monkeypatch.setattr(service_module, "run_demo", failed)
    assert client.post("/api/v1/portfolio/run").status_code == 202
    complete(service)
    assert service.status()["state"] == "failed"
    assert not folders[0].exists()
    assert service.source_path.read_bytes() == b"source survives failure cleanup"
    assert service.status()["data_available"]


def test_cleanup_refuses_protected_identifiers_and_traversal(service):
    for identifier in (service.scene()["run_id"], "../outside", "not-a-uuid"):
        with pytest.raises(ValueError, match="protected or non-UUID"):
            service._remove_run_folder(identifier)


def test_cleanup_rejects_alias_escaping_output_root(service, tmp_path, monkeypatch):
    identifier = "a" * 32
    target = service.output_root / identifier
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    original = Path.resolve

    def aliased(path, *args, **kwargs):
        return outside if path == target else original(path, *args, **kwargs)

    # Simulate resolution through a symlink/junction without Windows symlink privileges.
    monkeypatch.setattr(Path, "resolve", aliased)
    with pytest.raises(ValueError, match="aliased run"):
        service._remove_run_folder(identifier)
    assert marker.read_text(encoding="utf-8") == "preserve"
