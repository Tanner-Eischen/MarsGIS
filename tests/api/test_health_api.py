"""API tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient

from marshab.web.api import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestHealthAPI:
    """Tests for health check endpoints."""

    def test_health_live(self, client):
        """Test GET /api/v1/health/live."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_health_ready(self, client):
        """Test GET /api/v1/health/ready."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)

