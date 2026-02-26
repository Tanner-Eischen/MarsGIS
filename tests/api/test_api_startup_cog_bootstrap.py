"""Startup behavior tests for MOLA COG bootstrap modes."""

import pytest
from fastapi.testclient import TestClient

from marshab.web.api import create_app


def test_startup_continues_when_cog_bootstrap_fails_if_not_required(monkeypatch, tmp_path):
    """Blocking bootstrap failures are non-fatal unless explicitly required."""
    missing_cog = tmp_path / "missing.cog.tif"
    monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(missing_cog))
    monkeypatch.setenv("MARSHAB_MOLA_COG_BOOTSTRAP_MODE", "blocking")
    monkeypatch.setenv("MARSHAB_MOLA_COG_REQUIRED", "false")
    monkeypatch.setenv("MARSHAB_PREWARM_COG", "false")

    def _boom():
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr("marshab.core.datasets.bootstrap_cog.ensure_mola_cog_present", _boom)

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


def test_startup_fails_when_cog_bootstrap_fails_and_required(monkeypatch, tmp_path):
    """Blocking bootstrap failures should fail startup when required=true."""
    missing_cog = tmp_path / "missing.cog.tif"
    monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(missing_cog))
    monkeypatch.setenv("MARSHAB_MOLA_COG_BOOTSTRAP_MODE", "blocking")
    monkeypatch.setenv("MARSHAB_MOLA_COG_REQUIRED", "true")
    monkeypatch.setenv("MARSHAB_PREWARM_COG", "false")

    def _boom():
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr("marshab.core.datasets.bootstrap_cog.ensure_mola_cog_present", _boom)

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        with TestClient(create_app()):
            pass


def test_background_mode_never_blocks_startup(monkeypatch, tmp_path):
    """Background mode should allow startup even if bootstrap worker errors."""
    missing_cog = tmp_path / "missing.cog.tif"
    monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(missing_cog))
    monkeypatch.setenv("MARSHAB_MOLA_COG_BOOTSTRAP_MODE", "background")
    monkeypatch.setenv("MARSHAB_MOLA_COG_REQUIRED", "false")
    monkeypatch.setenv("MARSHAB_PREWARM_COG", "false")

    def _boom():
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr("marshab.core.datasets.bootstrap_cog.ensure_mola_cog_present", _boom)

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
