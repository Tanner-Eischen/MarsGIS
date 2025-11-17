"""Unit tests for navigation engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marshab.core.navigation_engine import NavigationEngine
from marshab.exceptions import NavigationError
from marshab.types import BoundingBox, SiteOrigin


class TestNavigationEngine:
    """Tests for NavigationEngine class."""

    @pytest.fixture
    def engine(self, test_config, monkeypatch):
        """Provide NavigationEngine instance with test config."""
        from marshab.config import reset_config
        reset_config()
        monkeypatch.setattr("marshab.core.navigation_engine.get_config", lambda: test_config)
        return NavigationEngine()

    @pytest.fixture
    def mock_sites_csv(self, tmp_path):
        """Create a mock sites.csv file."""
        sites_data = {
            "site_id": [1, 2],
            "geometry_type": ["POLYGON", "POLYGON"],
            "area_km2": [1.5, 1.2],
            "lat": [40.5, 40.6],
            "lon": [180.5, 180.6],
            "mean_slope_deg": [2.0, 3.0],
            "mean_roughness": [0.2, 0.3],
            "mean_elevation_m": [1000.0, 1050.0],
            "suitability_score": [0.85, 0.75],
            "rank": [1, 2]
        }
        df = pd.DataFrame(sites_data)
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        sites_file = analysis_dir / "sites.csv"
        df.to_csv(sites_file, index=False)
        return analysis_dir

    @pytest.fixture
    def mock_dem(self):
        """Create a mock DEM DataArray."""
        elevation = np.random.randn(100, 100) * 50 + 1000.0
        dem = xr.DataArray(
            elevation,
            dims=["y", "x"],
            coords={"y": np.arange(100), "x": np.arange(100)}
        )
        dem.rio = MagicMock()
        dem.rio.bounds.return_value = (180.0, 40.0, 181.0, 41.0)
        return dem

    def test_plan_to_site_loads_site(self, engine, mock_sites_csv):
        """Test that plan_to_site loads site from CSV."""
        # Mock data manager and other dependencies
        with patch.object(engine.data_manager, 'get_dem_for_roi') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.shape = (100, 100)
            mock_dem.values = np.random.randn(100, 100) * 50 + 1000.0
            mock_dem.rio = MagicMock()
            mock_dem.rio.bounds.return_value = (180.0, 40.0, 181.0, 41.0)
            mock_get_dem.return_value = mock_dem
            
            # Mock terrain analyzer
            with patch('marshab.core.navigation_engine.TerrainAnalyzer') as mock_terrain:
                mock_analyzer = MagicMock()
                mock_terrain.return_value = mock_analyzer
                from marshab.types import TerrainMetrics
                mock_metrics = TerrainMetrics(
                    slope=np.ones((100, 100)) * 5.0,
                    aspect=np.random.rand(100, 100) * 360.0,
                    roughness=np.ones((100, 100)) * 0.3,
                    tri=np.ones((100, 100)) * 2.0
                )
                mock_analyzer.analyze.return_value = mock_metrics
                
                # Mock pathfinder
                with patch('marshab.core.navigation_engine.AStarPathfinder') as mock_pathfinder:
                    mock_pf_instance = MagicMock()
                    mock_pathfinder.return_value = mock_pf_instance
                    # Return a simple path
                    mock_pf_instance.find_path_with_waypoints.return_value = [
                        (50, 50), (55, 55), (60, 60)
                    ]
                    
                    # Add config data source
                    from marshab.config import DataSource
                    engine.config.data_sources = {
                        "mola": DataSource(url="http://test.com/dem.tif", resolution_m=463.0)
                    }
                    
                    waypoints = engine.plan_to_site(
                        site_id=1,
                        analysis_dir=mock_sites_csv,
                        start_lat=40.5,
                        start_lon=180.5
                    )
                    
                    # Verify waypoints DataFrame
                    assert isinstance(waypoints, pd.DataFrame)
                    assert len(waypoints) > 0
                    assert "waypoint_id" in waypoints.columns
                    assert "x_meters" in waypoints.columns
                    assert "y_meters" in waypoints.columns
                    assert "tolerance_meters" in waypoints.columns

    def test_plan_to_site_site_not_found(self, engine, mock_sites_csv):
        """Test error when site ID not found."""
        with pytest.raises(NavigationError) as exc_info:
            engine.plan_to_site(
                site_id=999,  # Non-existent site
                analysis_dir=mock_sites_csv,
                start_lat=40.5,
                start_lon=180.5
            )
        
        assert "not found" in str(exc_info.value).lower()

    def test_plan_to_site_missing_sites_file(self, engine, tmp_path):
        """Test error when sites.csv doesn't exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(NavigationError) as exc_info:
            engine.plan_to_site(
                site_id=1,
                analysis_dir=empty_dir,
                start_lat=40.5,
                start_lon=180.5
            )
        
        assert "not found" in str(exc_info.value).lower() or "Sites file" in str(exc_info.value)

    def test_plan_to_site_coordinate_transformation(self, engine, mock_sites_csv):
        """Test coordinate transformation to SITE frame."""
        with patch.object(engine.data_manager, 'get_dem_for_roi') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.shape = (100, 100)
            mock_dem.values = np.random.randn(100, 100) * 50 + 1000.0
            mock_dem.rio = MagicMock()
            mock_dem.rio.bounds.return_value = (180.0, 40.0, 181.0, 41.0)
            mock_get_dem.return_value = mock_dem
            
            # Mock coordinate transformer
            with patch.object(engine.coord_transformer, 'iau_mars_to_site_frame') as mock_transform:
                mock_transform.return_value = (100.0, 200.0, -5.0)  # x, y, z in SITE frame
                
                # Mock terrain and pathfinding
                with patch('marshab.core.navigation_engine.TerrainAnalyzer') as mock_terrain, \
                     patch('marshab.core.navigation_engine.AStarPathfinder') as mock_pathfinder:
                    
                    mock_analyzer = MagicMock()
                    mock_terrain.return_value = mock_analyzer
                    from marshab.types import TerrainMetrics
                    mock_metrics = TerrainMetrics(
                        slope=np.ones((100, 100)) * 5.0,
                        aspect=np.random.rand(100, 100) * 360.0,
                        roughness=np.ones((100, 100)) * 0.3,
                        tri=np.ones((100, 100)) * 2.0
                    )
                    mock_analyzer.analyze.return_value = mock_metrics
                    
                    mock_pf_instance = MagicMock()
                    mock_pathfinder.return_value = mock_pf_instance
                    mock_pf_instance.find_path_with_waypoints.return_value = [(50, 50), (60, 60)]
                    
                    from marshab.config import DataSource
                    engine.config.data_sources = {
                        "mola": DataSource(url="http://test.com/dem.tif", resolution_m=463.0)
                    }
                    
                    waypoints = engine.plan_to_site(
                        site_id=1,
                        analysis_dir=mock_sites_csv,
                        start_lat=40.5,
                        start_lon=180.5
                    )
                    
                    # Verify coordinate transformation was called
                    assert mock_transform.called
                    assert len(waypoints) > 0

    def test_plan_to_site_pathfinding_integration(self, engine, mock_sites_csv):
        """Test A* pathfinding integration."""
        with patch.object(engine.data_manager, 'get_dem_for_roi') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.shape = (100, 100)
            mock_dem.values = np.random.randn(100, 100) * 50 + 1000.0
            mock_dem.rio = MagicMock()
            mock_dem.rio.bounds.return_value = (180.0, 40.0, 181.0, 41.0)
            mock_get_dem.return_value = mock_dem
            
            with patch('marshab.core.navigation_engine.TerrainAnalyzer') as mock_terrain, \
                 patch('marshab.core.navigation_engine.AStarPathfinder') as mock_pathfinder:
                
                mock_analyzer = MagicMock()
                mock_terrain.return_value = mock_analyzer
                from marshab.types import TerrainMetrics
                mock_metrics = TerrainMetrics(
                    slope=np.ones((100, 100)) * 5.0,
                    aspect=np.random.rand(100, 100) * 360.0,
                    roughness=np.ones((100, 100)) * 0.3,
                    tri=np.ones((100, 100)) * 2.0
                )
                mock_analyzer.analyze.return_value = mock_metrics
                
                mock_pf_instance = MagicMock()
                mock_pathfinder.return_value = mock_pf_instance
                # Return a path with multiple waypoints
                test_path = [(50, 50), (52, 52), (54, 54), (56, 56), (60, 60)]
                mock_pf_instance.find_path_with_waypoints.return_value = test_path
                
                from marshab.config import DataSource
                engine.config.data_sources = {
                    "mola": DataSource(url="http://test.com/dem.tif", resolution_m=463.0)
                }
                
                waypoints = engine.plan_to_site(
                    site_id=1,
                    analysis_dir=mock_sites_csv,
                    start_lat=40.5,
                    start_lon=180.5
                )
                
                # Verify pathfinder was called
                assert mock_pathfinder.called
                mock_pf_instance.find_path_with_waypoints.assert_called_once()
                
                # Verify waypoints match path length (or downsampled)
                assert len(waypoints) > 0
                assert len(waypoints) <= len(test_path)




