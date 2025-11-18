"""API tests for project endpoints."""

import pytest
from fastapi.testclient import TestClient

from marshab.web.api import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestProjectsAPI:
    """Tests for project API endpoints."""
    
    def test_list_projects(self, client):
        """Test GET /api/v1/projects."""
        response = client.get("/api/v1/projects")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_create_project(self, client):
        """Test POST /api/v1/projects."""
        request_data = {
            "name": "Test Project",
            "description": "Test description",
            "roi": {"lat_min": 40.0, "lat_max": 41.0, "lon_min": 180.0, "lon_max": 181.0},
            "dataset": "mola",
            "preset_id": "balanced",
            "selected_sites": [1, 2],
            "routes": [],
            "metadata": {}
        }
        
        response = client.post("/api/v1/projects", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
        assert "id" in data
    
    def test_get_project(self, client):
        """Test GET /api/v1/projects/{project_id}."""
        # First create a project
        request_data = {
            "name": "Get Test Project",
            "description": "",
            "roi": {"lat_min": 40.0, "lat_max": 41.0, "lon_min": 180.0, "lon_max": 181.0},
            "dataset": "mola",
            "preset_id": None,
            "selected_sites": [],
            "routes": [],
            "metadata": {}
        }
        
        create_response = client.post("/api/v1/projects", json=request_data)
        project_id = create_response.json()["id"]
        
        # Then get it
        response = client.get(f"/api/v1/projects/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project_id
        assert data["name"] == "Get Test Project"
    
    def test_delete_project(self, client):
        """Test DELETE /api/v1/projects/{project_id}."""
        # First create a project
        request_data = {
            "name": "Delete Test Project",
            "description": "",
            "roi": {"lat_min": 40.0, "lat_max": 41.0, "lon_min": 180.0, "lon_max": 181.0},
            "dataset": "mola",
            "preset_id": None,
            "selected_sites": [],
            "routes": [],
            "metadata": {}
        }
        
        create_response = client.post("/api/v1/projects", json=request_data)
        project_id = create_response.json()["id"]
        
        # Then delete it
        response = client.delete(f"/api/v1/projects/{project_id}")
        assert response.status_code == 200
        
        # Verify it's gone
        get_response = client.get(f"/api/v1/projects/{project_id}")
        assert get_response.status_code == 404

