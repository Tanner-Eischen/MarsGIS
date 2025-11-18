"""Unit tests for analysis pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import pytest
import xarray as xr

from marshab.core.analysis_pipeline import AnalysisPipeline, AnalysisResults
from marshab.exceptions import AnalysisError
from marshab.types import BoundingBox, SiteCandidate


class TestAnalysisPipeline:
    """Tests for AnalysisPipeline class."""

    @pytest.fixture
    def pipeline(self, test_config, monkeypatch):
        """Provide AnalysisPipeline instance with test config."""
        from marshab.config import reset_config
        reset_config()
        monkeypatch.setattr("marshab.core.analysis_pipeline.get_config", lambda: test_config)
        return AnalysisPipeline()

    @pytest.fixture
    def mock_dem(self):
        """Create a mock DEM DataArray."""
        # Create synthetic elevation data
        elevation = np.random.randn(100, 100) * 50 + 1000.0
        # Make some areas suitable (low slope, low roughness)
        elevation[30:50, 30:50] = 1000.0  # Flat area
        
        dem = xr.DataArray(
            elevation,
            dims=["y", "x"],
            coords={"y": np.arange(100), "x": np.arange(100)}
        )
        # Mock rio accessor using object.__setattr__ to bypass xarray restrictions
        # This is needed because xarray DataArray doesn't allow setting arbitrary attributes
        rio_mock = MagicMock()
        rio_mock.bounds.return_value = type('Bounds', (), {
            'left': 180.0,
            'right': 181.0,
            'bottom': 40.0,
            'top': 41.0
        })()
        rio_mock.res = (0.01, 0.01)
        object.__setattr__(dem, 'rio', rio_mock)
        return dem

    @pytest.fixture
    def test_roi(self):
        """Provide test region of interest."""
        return BoundingBox(
            lat_min=40.0,
            lat_max=41.0,
            lon_min=180.0,
            lon_max=181.0
        )

    @patch('marshab.core.analysis_pipeline.TerrainAnalyzer')
    @patch('marshab.core.analysis_pipeline.MCDMEvaluator')
    def test_run_pipeline_basic(self, mock_mcdm, mock_terrain, pipeline, test_roi, mock_dem, test_config):
        """Test basic pipeline execution."""
        # Mock data manager
        pipeline.data_manager.get_dem_for_roi = MagicMock(return_value=mock_dem)
        
        # Mock terrain analyzer
        mock_analyzer_instance = MagicMock()
        mock_terrain.return_value = mock_analyzer_instance
        
        # Create mock terrain metrics
        from marshab.types import TerrainMetrics
        mock_metrics = TerrainMetrics(
            slope=np.random.rand(100, 100) * 5.0,  # Low slopes
            aspect=np.random.rand(100, 100) * 360.0,
            roughness=np.random.rand(100, 100) * 0.3,  # Low roughness
            tri=np.random.rand(100, 100) * 2.0,
            hillshade=np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            elevation=np.random.rand(100, 100) * 1000.0 + 2000.0
        )
        mock_analyzer_instance.analyze.return_value = mock_metrics
        
        # Mock CriteriaExtractor
        with patch('marshab.core.analysis_pipeline.CriteriaExtractor') as mock_extractor:
            mock_extractor_instance = MagicMock()
            mock_extractor.return_value = mock_extractor_instance
            # Return criteria dict
            mock_extractor_instance.extract_all.return_value = {
                "slope": mock_metrics.slope,
                "roughness": mock_metrics.roughness,
                "elevation": mock_metrics.elevation,
                "solar_exposure": np.random.rand(100, 100),
                "science_value": np.ones((100, 100)) * 0.5
            }
            
            # Mock MCDM evaluator
            # Create suitability scores - some areas above threshold
            suitability = np.random.rand(100, 100) * 0.3 + 0.7  # Most above 0.7
            suitability[0:20, 0:20] = 0.5  # Some below threshold
            mock_mcdm.evaluate.return_value = suitability
            
            # Add config data source
            from marshab.config import DataSource
            test_config.data_sources = {
                "mola": DataSource(url="http://test.com/dem.tif", resolution_m=463.0)
            }
            test_config.analysis.min_site_area_km2 = 0.1  # Lower threshold for testing
            
            # Run pipeline
            results = pipeline.run(test_roi, dataset="mola", threshold=0.7)
            
            # Verify results
            assert isinstance(results, AnalysisResults)
            assert len(results.sites) > 0  # Should find some sites
            assert results.top_site_id > 0
            assert results.top_site_score > 0.0
            
            # Verify terrain analyzer was called
            mock_analyzer_instance.analyze.assert_called_once()
            
            # Verify MCDM was called
            mock_mcdm.evaluate.assert_called_once()

    def test_run_pipeline_no_sites_found(self, pipeline, test_roi, mock_dem, test_config):
        """Test pipeline when no suitable sites are found."""
        # Mock data manager
        pipeline.data_manager.get_dem_for_roi = MagicMock(return_value=mock_dem)
        
        # Create terrain metrics with high slopes (unsuitable)
        from marshab.types import TerrainMetrics
        high_slope = np.ones((100, 100)) * 30.0  # Very steep
        metrics = TerrainMetrics(
            slope=high_slope,
            aspect=np.random.rand(100, 100) * 360.0,
            roughness=np.ones((100, 100)) * 1.0,  # High roughness
            tri=np.ones((100, 100)) * 5.0,
            hillshade=np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            elevation=np.ones((100, 100)) * 3000.0
        )
        
        with patch('marshab.core.analysis_pipeline.TerrainAnalyzer') as mock_terrain:
            mock_analyzer = MagicMock()
            mock_terrain.return_value = mock_analyzer
            mock_analyzer.analyze.return_value = metrics
            
            # Mock CriteriaExtractor
            with patch('marshab.core.analysis_pipeline.CriteriaExtractor') as mock_extractor:
                mock_extractor_instance = MagicMock()
                mock_extractor.return_value = mock_extractor_instance
                mock_extractor_instance.extract_all.return_value = {
                    "slope": metrics.slope,
                    "roughness": metrics.roughness,
                    "elevation": metrics.elevation,
                    "solar_exposure": np.random.rand(100, 100),
                    "science_value": np.ones((100, 100)) * 0.5
                }
                # MCDM returns low suitability
                with patch('marshab.core.analysis_pipeline.MCDMEvaluator') as mock_mcdm:
                    low_suitability = np.ones((100, 100)) * 0.3  # All below threshold
                    mock_mcdm.evaluate.return_value = low_suitability
                
                # Add config
                from marshab.config import DataSource
                test_config.data_sources = {"mola": DataSource(url="http://test.com/dem.tif", resolution_m=463.0)}
                
                results = pipeline.run(test_roi, dataset="mola", threshold=0.7)
                
                # Should return empty results
                assert len(results.sites) == 0
                assert results.top_site_id == 0
                assert results.top_site_score == 0.0

    def test_analysis_results_save(self, tmp_path):
        """Test saving analysis results."""
        from marshab.types import SiteCandidate
        
        sites = [
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
            ),
            SiteCandidate(
                site_id=2,
                geometry_type="POLYGON",
                area_km2=1.2,
                lat=40.6,
                lon=180.6,
                mean_slope_deg=3.0,
                mean_roughness=0.3,
                mean_elevation_m=1050.0,
                suitability_score=0.75,
                rank=2
            )
        ]
        
        results = AnalysisResults(
            sites=sites,
            top_site_id=1,
            top_site_score=0.85
        )
        
        output_dir = tmp_path / "output"
        results.save(output_dir)
        
        # Verify file was created
        sites_file = output_dir / "sites.csv"
        assert sites_file.exists()
        
        # Verify content
        import pandas as pd
        df = pd.read_csv(sites_file)
        assert len(df) == 2
        assert df["site_id"].tolist() == [1, 2]
        assert df["suitability_score"].iloc[0] == 0.85

    def test_run_pipeline_error_handling(self, pipeline, test_roi):
        """Test error handling in pipeline."""
        # Mock data manager to raise error
        pipeline.data_manager.get_dem_for_roi = MagicMock(side_effect=Exception("DEM load failed"))
        
        with pytest.raises(AnalysisError) as exc_info:
            pipeline.run(test_roi, dataset="mola")
        
        assert "Analysis pipeline failed" in str(exc_info.value)




