"""Unit tests for enhanced route planning."""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock

from marshab.analysis.routing import (
    plan_route,
    compute_route_cost,
    calculate_shadow_penalty,
    Route,
    RouteCostResult,
)


class TestShadowPenalty:
    """Tests for shadow penalty calculation."""
    
    def test_shadow_penalty_basic(self):
        """Test basic shadow penalty calculation."""
        # Create simple elevation array (flat)
        elevation = np.ones((10, 10)) * 1000.0
        cell_size_m = 100.0
        
        shadow = calculate_shadow_penalty(
            elevation,
            sun_azimuth=315.0,  # Northwest
            sun_altitude=45.0,
            cell_size_m=cell_size_m
        )
        
        assert shadow.shape == elevation.shape
        assert np.all(shadow >= 0)
        assert np.all(shadow <= 1)
    
    def test_shadow_penalty_empty_array(self):
        """Test shadow penalty with empty array."""
        elevation = np.array([])
        shadow = calculate_shadow_penalty(
            elevation,
            sun_azimuth=315.0,
            sun_altitude=45.0,
            cell_size_m=100.0
        )
        
        assert shadow.size == 0


class TestRoutePlanning:
    """Tests for route planning."""
    
    def test_plan_route_basic(self, mock_dem):
        """Test basic route planning."""
        start = (5, 5)
        end = (95, 95)
        weights = {
            "distance": 0.3,
            "slope_penalty": 0.3,
            "roughness_penalty": 0.2,
            "elevation_penalty": 0.2,
        }
        
        route = plan_route(
            start=start,
            end=end,
            weights=weights,
            dem=mock_dem,
            cell_size_m=100.0
        )
        
        assert isinstance(route, Route)
        assert len(route.waypoints) > 0
        assert route.total_distance_m > 0
        assert start in route.waypoints or route.waypoints[0] == start
        assert end in route.waypoints or route.waypoints[-1] == end
    
    def test_plan_route_with_shadow(self, mock_dem):
        """Test route planning with shadow calculation."""
        start = (5, 5)
        end = (95, 95)
        weights = {
            "distance": 0.3,
            "slope_penalty": 0.3,
            "roughness_penalty": 0.2,
            "shadow_penalty": 0.2,
        }
        
        route = plan_route(
            start=start,
            end=end,
            weights=weights,
            dem=mock_dem,
            sun_azimuth=315.0,
            sun_altitude=45.0,
            cell_size_m=100.0
        )
        
        assert isinstance(route, Route)
        assert "sun_azimuth" in route.metadata
        assert route.metadata["sun_azimuth"] == 315.0


class TestRouteCost:
    """Tests for route cost calculation."""
    
    def test_compute_route_cost(self, mock_dem):
        """Test route cost computation."""
        waypoints = [(0, 0), (10, 10), (20, 20), (30, 30)]
        route = Route(
            waypoints=waypoints,
            total_distance_m=4242.6,  # Approximate
            metadata={}
        )
        
        weights = {
            "distance": 0.3,
            "slope_penalty": 0.3,
            "roughness_penalty": 0.2,
            "shadow_penalty": 0.2,
        }
        
        cost = compute_route_cost(
            route,
            mock_dem,
            weights
        )
        
        assert isinstance(cost, RouteCostResult)
        assert cost.distance_m > 0
        assert cost.energy_estimate_j > 0
        assert "distance" in cost.components
        assert "slope" in cost.components


@pytest.fixture
def mock_dem():
    """Create mock DEM for testing with gentle slopes."""
    np.random.seed(42)  # Reproducible
    # Create a simple 100x100 DEM with gentle slopes
    # Use smooth gradient instead of random to avoid impassable terrain
    y, x = np.meshgrid(np.linspace(40.0, 41.0, 100), np.linspace(180.0, 181.0, 100), indexing='ij')
    # Create gentle elevation gradient (max ~100m variation over 100 pixels = ~1m per pixel)
    elevation = 2000.0 + (x - 180.0) * 50.0 + (y - 40.0) * 30.0
    # Add small random noise (max 5m)
    elevation += np.random.randn(100, 100) * 5.0
    
    dem = xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={
            "y": np.linspace(40.0, 41.0, 100),
            "x": np.linspace(180.0, 181.0, 100),
        }
    )
    # Note: rio accessor not needed for routing tests
    return dem

