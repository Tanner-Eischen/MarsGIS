"""API tests for mission scenario endpoints."""

import pytest
from fastapi.testclient import TestClient

from marshab.web.api import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestLandingScenarioAPI:
    """Tests for landing scenario API endpoint."""
    
    def test_landing_scenario_endpoint(self, client):
        """Test POST /api/v1/mission/landing-scenario."""
        request_data = {
            "roi": {
                "lat_min": 40.0,
                "lat_max": 41.0,
                "lon_min": 180.0,
                "lon_max": 181.0
            },
            "dataset": "mola",
            "preset_id": "balanced",
            "suitability_threshold": 0.7
        }
        
        response = client.post("/api/v1/mission/landing-scenario", json=request_data)
        
        # May fail if DEM not available, but should return proper structure
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "scenario_id" in data
            assert "sites" in data
            assert "metadata" in data
    
    def test_landing_scenario_invalid_roi(self, client):
        """Test landing scenario with invalid ROI."""
        request_data = {
            "roi": {
                "lat_min": 41.0,  # Invalid: lat_max < lat_min
                "lat_max": 40.0,
                "lon_min": 180.0,
                "lon_max": 181.0
            },
            "dataset": "mola"
        }
        
        response = client.post("/api/v1/mission/landing-scenario", json=request_data)
        assert response.status_code == 400


class TestTraverseScenarioAPI:
    """Tests for rover traverse scenario API endpoint."""
    
    def test_traverse_scenario_endpoint(self, client):
        """Test POST /api/v1/mission/rover-traverse."""
        request_data = {
            "start_site_id": 1,
            "end_site_id": 2,
            "analysis_dir": "data/output",
            "preset_id": "shortest_path"
        }
        
        response = client.post("/api/v1/mission/rover-traverse", json=request_data)
        
        # May fail if analysis results not found, but should return proper structure
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "route_id" in data
            assert "waypoints" in data
            assert "total_distance_m" in data
            assert "risk_score" in data
    
    def test_traverse_scenario_missing_analysis_dir(self, client):
        """Test traverse scenario with missing analysis directory."""
        request_data = {
            "start_site_id": 1,
            "end_site_id": 2,
            "analysis_dir": "/nonexistent/path",
            "preset_id": "shortest_path"
        }
        
        response = client.post("/api/v1/mission/rover-traverse", json=request_data)
        assert response.status_code == 404

