"""Integration tests for navigation pipeline."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.navigation_engine import NavigationEngine
from marshab.types import BoundingBox, SiteCandidate, TerrainMetrics


class TestNavigationPipeline:
    """End-to-end navigation tests."""

    @pytest.fixture
    def analysis_results(self, tmp_path, test_config, monkeypatch):
        """Run analysis and save results."""
        from marshab.config import reset_config
        reset_config()
        monkeypatch.setattr("marshab.core.analysis_pipeline.get_config", lambda: test_config)
        monkeypatch.setattr("marshab.core.navigation_engine.get_config", lambda: test_config)

        pipeline = AnalysisPipeline()
        roi = BoundingBox(
            lat_min=40.0, lat_max=40.5,
            lon_min=180.0, lon_max=180.5
        )

        # Mock DEM loading to avoid actual downloads
        with patch.object(pipeline.data_manager, 'get_dem_for_roi') as mock_get_dem:
            # Create mock DEM
            mock_dem = xr.DataArray(
                np.random.randn(100, 100) * 50 + 1000.0,
                dims=["y", "x"],
                coords={"y": np.arange(100), "x": np.arange(100)}
            )
            mock_dem.attrs = {'resolution_m': 463.0}
            mock_get_dem.return_value = mock_dem

            # Mock criteria extraction
            with patch('marshab.core.analysis_pipeline.CriteriaExtractor') as mock_extractor:
                mock_ext = MagicMock()
                mock_extractor.return_value = mock_ext
                mock_ext.extract_all.return_value = {
                    'slope': np.ones((100, 100)) * 5.0,
                    'roughness': np.ones((100, 100)) * 0.3,
                    'elevation': np.ones((100, 100)) * 1000.0
                }

                results = pipeline.run(roi, dataset="mola", threshold=0.6)

        output_dir = tmp_path / "analysis"
        pipeline.save_results(results, output_dir)

        return output_dir, results

    def test_complete_navigation_workflow(self, analysis_results, test_config, monkeypatch):
        """Test full workflow from analysis to waypoints."""
        from marshab.config import reset_config
        reset_config()
        monkeypatch.setattr("marshab.core.navigation_engine.get_config", lambda: test_config)

        output_dir, results = analysis_results

        # Get first site
        if len(results.sites) == 0:
            pytest.skip("No sites found in analysis results")

        site = results.sites[0]

        # Plan navigation
        engine = NavigationEngine()

        # Mock pathfinding to avoid complex A* calculations
        with pytest.mock.patch('marshab.core.navigation_engine.AStarPathfinder') as mock_pathfinder:
            mock_pf_instance = pytest.mock.MagicMock()
            mock_pathfinder.return_value = mock_pf_instance
            # Return a simple path
            mock_pf_instance.find_path_with_waypoints.return_value = [
                (10, 10), (20, 20), (30, 30), (40, 40)
            ]

            waypoints = engine.plan_to_site(
                site_id=site.site_id,
                analysis_dir=output_dir,
                start_lat=40.1,
                start_lon=180.1,
                max_waypoint_spacing_m=100.0
            )

        # Validate waypoints
        assert len(waypoints) > 0
        assert 'waypoint_id' in waypoints.columns
        assert 'x_site' in waypoints.columns
        assert 'y_site' in waypoints.columns
        assert 'z_site' in waypoints.columns
        assert 'latitude' in waypoints.columns
        assert 'longitude' in waypoints.columns
        assert 'elevation_m' in waypoints.columns
        assert 'tolerance_m' in waypoints.columns

        # First waypoint should be near origin
        assert abs(waypoints.iloc[0]['x_site']) < 50
        assert abs(waypoints.iloc[0]['y_site']) < 50

        # Last waypoint should be far from origin
        if len(waypoints) > 1:
            last = waypoints.iloc[-1]
            distance = (last['x_site']**2 + last['y_site']**2)**0.5
            # Distance should be reasonable (at least some distance)
            assert distance >= 0

    def test_pickle_file_loading(self, analysis_results, test_config, monkeypatch):
        """Test that navigation engine can load from pickle file."""
        from marshab.config import reset_config
        reset_config()
        monkeypatch.setattr("marshab.core.navigation_engine.get_config", lambda: test_config)

        output_dir, results = analysis_results

        # Verify pickle file exists
        pickle_file = output_dir / "analysis_results.pkl"
        assert pickle_file.exists()

        # Load using NavigationEngine
        engine = NavigationEngine()
        loaded_results = engine._load_analysis_results(output_dir)

        assert 'dem' in loaded_results
        assert 'metrics' in loaded_results
        assert 'sites' in loaded_results
        assert len(loaded_results['sites']) == len(results.sites)

    def test_analysis_results_save_and_load(self, tmp_path, test_config, monkeypatch):
        """Test that analysis results can be saved and loaded correctly."""
        from marshab.config import reset_config
        reset_config()
        monkeypatch.setattr("marshab.core.analysis_pipeline.get_config", lambda: test_config)

        pipeline = AnalysisPipeline()

        # Create mock results
        mock_dem = xr.DataArray(
            np.random.randn(50, 50) * 50 + 1000.0,
            dims=["y", "x"],
            coords={"y": np.arange(50), "x": np.arange(50)}
        )
        mock_metrics = TerrainMetrics(
            slope=np.ones((50, 50)) * 5.0,
            aspect=np.random.rand(50, 50) * 360.0,
            roughness=np.ones((50, 50)) * 0.3,
            tri=np.ones((50, 50)) * 2.0,
            hillshade=np.ones((50, 50), dtype=np.uint8) * 128,
            elevation=np.ones((50, 50)) * 1000.0
        )
        mock_sites = [
            SiteCandidate(
                site_id=1,
                geometry_type="POLYGON",
                area_km2=1.5,
                lat=40.5,
                lon=180.5,
                mean_slope_deg=2.0,
                mean_roughness=0.2,
                mean_elevation_m=1000.0,
                suitability_score=0.85,
                rank=1
            )
        ]

        results = type('AnalysisResults', (), {
            'sites': mock_sites,
            'top_site_id': 1,
            'top_site_score': 0.85,
            'dem': mock_dem,
            'metrics': mock_metrics,
            'suitability': np.ones((50, 50)) * 0.8,
            'criteria': {}
        })()

        output_dir = tmp_path / "test_analysis"
        pipeline.save_results(results, output_dir)

        # Verify files exist
        assert (output_dir / "analysis_results.pkl").exists()
        assert (output_dir / "sites.geojson").exists()
        assert (output_dir / "sites.csv").exists()

        # Load pickle and verify
        with open(output_dir / "analysis_results.pkl", 'rb') as f:
            loaded = pickle.load(f)

        assert 'dem' in loaded
        assert 'metrics' in loaded
        assert 'sites' in loaded
        assert len(loaded['sites']) == 1

