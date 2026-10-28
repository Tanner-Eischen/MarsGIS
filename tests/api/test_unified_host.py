"""Same-origin React/API hosting without browser or network automation."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marshab.web.api import create_app


@pytest.fixture
def host(monkeypatch, tmp_path):
    dist = tmp_path / "client"
    (dist / "assets").mkdir(parents=True)
    (dist / "portfolio").mkdir()
    (dist / "index.html").write_text("<!doctype html><title>MarsHab client</title>", encoding="utf-8")
    (dist / "assets" / "main.js").write_text("export const app = 'MarsHab'", encoding="utf-8")
    (dist / "portfolio" / "scene.json").write_text('{"reference": true}', encoding="utf-8")
    (tmp_path / "private.txt").write_text("PRIVATE_OUTSIDE_BUILD", encoding="utf-8")
    monkeypatch.setenv("MARSHAB_WEBUI_DIST", str(dist))
    with TestClient(create_app()) as client:
        yield client, dist


@pytest.mark.parametrize("path", ["/", "/compare", "/review/sites/3"])
def test_frontend_root_and_deep_links(host, path):
    client, _ = host
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "MarsHab client" in response.text


def test_files_are_served_with_their_content_and_media_type(host):
    client, _ = host
    script = client.get("/assets/main.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert script.text == "export const app = 'MarsHab'"
    reference = client.get("/portfolio/scene.json")
    assert reference.status_code == 200
    assert reference.json() == {"reference": True}


@pytest.mark.parametrize("path", ["/api", "/api/missing", "/api/v1/missing", "/ws", "/ws/missing"])
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_unknown_service_urls_keep_json_404(host, path, method):
    client, _ = host
    response = client.request(method, path)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Not found"}


@pytest.mark.parametrize("path", [
    "/assets/missing.js", "/assets/missing", "/assets", "/portfolio/missing",
    "/portfolio/missing.json", "/missing.css", "/nested/missing.tif",
])
def test_missing_resources_are_not_spa_documents(host, path):
    client, _ = host
    response = client.get(path)
    assert response.status_code == 404
    assert "MarsHab client" not in response.text


@pytest.mark.parametrize("path", [
    "/%2e%2e/private.txt", "/assets/%2e%2e/%2e%2e/private.txt",
    "/%2e%2e%5cprivate.txt", "/C%3A/private.txt", "/%00private.txt",
])
def test_traversal_cannot_read_outside_build(host, path):
    client, _ = host
    response = client.get(path)
    assert response.status_code == 404
    assert "PRIVATE_OUTSIDE_BUILD" not in response.text


def test_registered_api_and_openapi_take_precedence(host):
    client, _ = host
    health = client.get("/api/v1/health/live")
    assert health.status_code == 200
    assert health.json()["status"] == "alive"
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/api/v1/portfolio/status" in schema.json()["paths"]
    assert "/{resource}" not in schema.json()["paths"]


def test_spa_fallback_is_get_only(host):
    client, _ = host
    response = client.post("/review/sites/3")
    assert response.status_code == 405
    assert response.headers["content-type"].startswith("application/json")


def test_api_remains_available_when_frontend_is_not_built(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MARSHAB_WEBUI_DIST", str(tmp_path / "not-built"))
    client = TestClient(create_app())
    assert client.get("/api/v1/health/live").status_code == 200
    response = client.get("/")
    assert response.status_code == 404
    assert response.json() == {"detail": "Frontend build not available"}


def test_unified_production_exposes_only_validated_api(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MARSHAB_ENABLE_EXPERIMENTAL_API", "false")
    monkeypatch.setenv("MARSHAB_WEBUI_DIST", str(tmp_path / "missing"))
    client = TestClient(create_app())
    paths = client.get("/openapi.json").json()["paths"]
    assert paths
    assert all(path.startswith(("/api/v1/portfolio/", "/api/v1/health/")) for path in paths)
    assert client.get("/api/v1/health/live").status_code == 200
    for path in ("/api/v1/download", "/api/v1/analyze", "/api/v1/solar/analyze"):
        response = client.post(path, json={})
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
