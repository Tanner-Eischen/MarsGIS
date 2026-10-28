"""Integration tests for navigation pipeline."""

import pickle

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_bounds

from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.data_manager import DataManager
from marshab.core.navigation_engine import NavigationEngine
from marshab.processing.dem_loader import DEMLoader
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

        # Only data acquisition is replaced: terrain, scoring, extraction and
        # routing all run on the same georeferenced, gentle synthetic surface.
        row, col = np.indices((100, 100))
        elevation = 1000 + 50 * np.exp(-((row - 45)**2 + (col - 50)**2) / 250)
        dem_path = tmp_path / "navigation_fixture.tif"
        with rasterio.open(
            dem_path, "w", driver="GTiff", height=100, width=100, count=1,
            dtype="float32", crs="+proj=longlat +a=3396190 +b=3396190 +no_defs +type=crs",
            transform=from_bounds(180, 40, 180.5, 40.5, 100, 100), nodata=-9999,
        ) as dst:
            dst.write(elevation.astype("float32"), 1)
            dst.update_tags(DATA_MODE="synthetic")
        dem = DEMLoader().load(dem_path)
        monkeypatch.setattr(DataManager, "get_dem_for_roi", lambda *args, **kwargs: dem)
        test_config.navigation.enable_smoothing = False
        results = pipeline.run(roi, dataset="mola", threshold=0.6)
        assert results.sites, "Known gentle fixture should yield candidate sites"

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
        assert results.sites

        site = results.sites[0]

        # Plan navigation
        engine = NavigationEngine()

        # Start at a known pixel center so SITE coordinates have no snapping
        # offset. Exercise the real A* solver, not a mocked waypoint list.
        start_lat = float(results.dem.coords["lat"].values[80, 20])
        start_lon = float(results.dem.coords["lon"].values[80, 20])
        waypoints = engine.plan_to_site(
            site_id=site.site_id,
            analysis_dir=output_dir,
            start_lat=start_lat,
            start_lon=start_lon,
            max_waypoint_spacing_m=100.0,
        )

        # Validate waypoints
        assert len(waypoints) > 2
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
            assert distance > results.dem.attrs["resolution_m"]

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

