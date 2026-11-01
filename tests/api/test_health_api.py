"""Liveness, source readiness and actual-byte identity without provisioning."""

import hashlib
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from marshab.config import Config, PathsConfig
from marshab.web import build_identity as identity_module
from marshab.web.api import create_app
from marshab.web.routes import explorer, health


@pytest.fixture(autouse=True)
def clear_identity_cache():
    identity_module._identity.cache_clear()
    yield
    identity_module._identity.cache_clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    paths = PathsConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "output")
    paths.cache_dir.mkdir()
    paths.output_dir.mkdir()
    config = Config.model_construct(paths=paths)
    monkeypatch.setattr(health, "get_config", lambda: config)
    monkeypatch.setenv("MARSHAB_GLOBAL_DEM_PATH", str(tmp_path / "missing-mola.img"))
    monkeypatch.setenv("MARSHAB_REFERENCE_DEM_PATH", str(tmp_path / "missing-ctx.tif"))

    attempts = []

    def forbid_download(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("Health requests must never download source data")

    monkeypatch.setattr(urllib.request, "urlopen", forbid_download)
    yield TestClient(create_app())
    assert attempts == [], "Health attempted provisioning even if its exception was caught"


def test_liveness_has_identity_without_opening_sources(client, monkeypatch):
    def forbid_source(*args):
        raise AssertionError("Liveness must not initialize sources")

    monkeypatch.setattr(health, "get_service", forbid_source)
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200 and response.json()["status"] == "alive"
    build = response.json()["build"]
    assert len(build["backend_sha256"]) == 64 and build["backend_python_files"] > 90


def test_ready_rejects_missing_sources_without_download_or_machine_path(client):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["planetary_sources"] == "not_ready"
    assert "missing-mola" not in response.text and "Temp" not in response.text


def test_readiness_reuses_verified_store_and_reports_source_identity(client, monkeypatch):
    calls = []

    def provisioned(config):
        calls.append(config)
        return SimpleNamespace(
            metadata=lambda: {
                "global": {"id": "mola16", "sha256": "a" * 64},
                "detail": {"id": "ctx", "sha256": "b" * 64},
            }
        )

    monkeypatch.setattr(explorer, "from_config", provisioned)
    for _ in range(2):
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200 and response.json()["sources"]["status"] == "verified"
        assert response.json()["sources"]["global"]["sha256"] == "a" * 64
    assert len(calls) == 1


def test_invalid_source_error_does_not_leak_private_details(client, monkeypatch):
    def corrupt(*args):
        raise ValueError("Checksum mismatch at C:/Users/private/source.tif")

    monkeypatch.setattr(health, "get_service", corrupt)
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 503 and "private" not in response.text


def test_declared_ancestry_never_changes_actual_byte_fingerprint(monkeypatch):
    monkeypatch.setenv("MARSHAB_SOURCE_COMMIT", "a" * 40)
    first = identity_module.build_identity()
    monkeypatch.setenv("MARSHAB_SOURCE_COMMIT", "b" * 40)
    identity_module._identity.cache_clear()
    second = identity_module.build_identity()
    assert first["declared_source_commit"] != second["declared_source_commit"]
    assert first["backend_sha256"] == second["backend_sha256"]
    root = Path(identity_module.__file__).resolve().parents[1]
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    expected = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first["backend_sha256"] == expected
    second["backend_sha256"] = "changed externally"
    assert identity_module.build_identity()["backend_sha256"] == expected
