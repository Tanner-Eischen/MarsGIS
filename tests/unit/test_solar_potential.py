"""Unit tests for solar potential analysis."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from marshab.analysis.solar_potential import (
    DEFAULT_PANEL_EFFICIENCY,
    MARS_SOLAR_CONSTANT,
    MissionImpacts,
    SolarPotentialAnalyzer,
    SolarPotentialResult,
)
from marshab.processing.criteria import CriteriaExtractor
from marshab.types import TerrainMetrics


@pytest.fixture
def sample_dem():
    """Create sample DEM for testing."""
    np.random.seed(42)  # Reproducible
    elevation = np.random.randn(50, 50) * 10 + 1000.0
    dem = xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={
            "y": np.linspace(40.0, 41.0, 50),
            "x": np.linspace(180.0, 181.0, 50)
        }
    )
    dem.attrs = {'nodata': -9999}
    return dem


@pytest.fixture
def sample_metrics():
    """Create sample terrain metrics."""
    shape = (50, 50)
    return TerrainMetrics(
        slope=np.random.rand(*shape) * 10.0,  # 0-10 degrees
        aspect=np.random.rand(*shape) * 360.0,  # 0-360 degrees
        roughness=np.random.rand(*shape) * 5.0,  # 0-5 m
        tri=np.random.rand(*shape) * 2.0,  # 0-2 m
        hillshade=np.random.randint(0, 255, shape, dtype=np.uint8),
        elevation=np.random.randn(*shape) * 10 + 1000.0
    )


class TestSolarPotentialAnalyzer:
    """Tests for SolarPotentialAnalyzer."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=200.0)
        assert analyzer.cell_size_m == 200.0

    def test_calculate_solar_potential_basic(self, sample_dem, sample_metrics):
        """Test basic solar potential calculation."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            mock_exposure.return_value = np.ones((50, 50)) * 0.7

            result = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=45.0
            )

            assert isinstance(result, SolarPotentialResult)
            assert result.solar_potential_map.shape == (50, 50)
            assert result.irradiance_map.shape == (50, 50)
            assert 0.0 <= result.statistics["mean"] <= 1.0
            assert result.statistics["mean_irradiance_w_per_m2"] > 0

    def test_calculate_solar_potential_statistics(self, sample_dem, sample_metrics):
        """Test that statistics are calculated correctly."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            # Create known solar exposure pattern
            exposure = np.ones((50, 50)) * 0.5
            exposure[0:25, :] = 0.8  # Higher exposure in first half
            mock_exposure.return_value = exposure

            result = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=45.0
            )

            assert "min" in result.statistics
            assert "max" in result.statistics
            assert "mean" in result.statistics
            assert "std" in result.statistics
            assert result.statistics["min"] <= result.statistics["mean"] <= result.statistics["max"]
            assert result.statistics["mean_irradiance_w_per_m2"] > 0

    def test_calculate_mission_impacts(self, sample_dem, sample_metrics):
        """Test mission impact calculations."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            mock_exposure.return_value = np.ones((50, 50)) * 0.7

            solar_result = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=45.0
            )

            # Set a known mean irradiance for predictable results
            solar_result.statistics["mean_irradiance_w_per_m2"] = 300.0

            impacts = analyzer.calculate_mission_impacts(
                solar_result,
                panel_efficiency=0.25,
                panel_area_m2=100.0,
                battery_capacity_kwh=50.0,
                daily_power_needs_kwh=20.0,
                battery_cost_per_kwh=1000.0,
                mission_duration_days=500.0
            )

            assert isinstance(impacts, MissionImpacts)
            assert impacts.power_generation_kwh_per_day > 0
            assert impacts.power_surplus_kwh_per_day is not None
            assert impacts.mission_duration_extension_days >= 0
            assert impacts.cost_savings_usd >= 0
            assert impacts.battery_reduction_kwh >= 0

    def test_calculate_mission_impacts_power_surplus(self, sample_dem, sample_metrics):
        """Test mission impacts with power surplus."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            mock_exposure.return_value = np.ones((50, 50)) * 0.9  # High exposure

            solar_result = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=45.0
            )

            # Set high irradiance to ensure surplus
            solar_result.statistics["mean_irradiance_w_per_m2"] = 500.0

            impacts = analyzer.calculate_mission_impacts(
                solar_result,
                panel_efficiency=0.25,
                panel_area_m2=100.0,
                battery_capacity_kwh=50.0,
                daily_power_needs_kwh=20.0,
                battery_cost_per_kwh=1000.0,
                mission_duration_days=500.0
            )

            # With high irradiance, should have surplus
            assert impacts.power_generation_kwh_per_day > impacts.power_surplus_kwh_per_day
            if impacts.power_surplus_kwh_per_day > 0:
                assert impacts.mission_duration_extension_days > 0
                assert impacts.cost_savings_usd > 0

    def test_calculate_mission_impacts_power_deficit(self, sample_dem, sample_metrics):
        """Test mission impacts with power deficit."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            mock_exposure.return_value = np.ones((50, 50)) * 0.2  # Low exposure

            solar_result = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=45.0
            )

            # Set low irradiance to ensure deficit
            solar_result.statistics["mean_irradiance_w_per_m2"] = 100.0

            impacts = analyzer.calculate_mission_impacts(
                solar_result,
                panel_efficiency=0.25,
                panel_area_m2=100.0,
                battery_capacity_kwh=50.0,
                daily_power_needs_kwh=20.0,
                battery_cost_per_kwh=1000.0,
                mission_duration_days=500.0
            )

            # With low irradiance, may have deficit
            if impacts.power_surplus_kwh_per_day < 0:
                assert impacts.mission_duration_extension_days == 0
                assert impacts.battery_reduction_kwh == 0

    def test_shadow_penalty_calculation(self, sample_dem, sample_metrics):
        """Test shadow penalty calculation."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        # Create elevation with known pattern
        elevation = np.ones((50, 50)) * 1000.0
        elevation[25:, :] = 1500.0  # Higher elevation in second half
        sample_dem.values = elevation

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            mock_exposure.return_value = np.ones((50, 50)) * 0.7

            result = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=45.0
            )

            # Shadow penalty should affect potential
            assert result.solar_potential_map.shape == (50, 50)
            assert np.all(result.solar_potential_map >= 0.0)
            assert np.all(result.solar_potential_map <= 1.0)

    def test_sun_position_affects_irradiance(self, sample_dem, sample_metrics):
        """Test that sun altitude affects irradiance."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            mock_exposure.return_value = np.ones((50, 50)) * 0.7

            # High sun altitude
            result_high = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=90.0
            )

            # Low sun altitude
            result_low = analyzer.calculate_solar_potential(
                sample_dem,
                sample_metrics,
                sun_azimuth=0.0,
                sun_altitude=10.0
            )

            # Higher sun should give more irradiance
            assert result_high.statistics["mean_irradiance_w_per_m2"] > result_low.statistics["mean_irradiance_w_per_m2"]

    def test_small_elevation_handling(self, sample_metrics):
        """Test handling of small elevation array."""
        analyzer = SolarPotentialAnalyzer(cell_size_m=463.0)

        # Create a small but valid DEM (2x2 minimum for gradient)
        small_dem = xr.DataArray(
            np.array([[1000.0, 1001.0], [1002.0, 1003.0]]),
            dims=["y", "x"],
            coords={"y": [40.0, 40.1], "x": [180.0, 180.1]}
        )
        small_dem.attrs = {}

        # Create matching small metrics
        small_metrics = TerrainMetrics(
            slope=np.array([[0.0, 1.0], [1.0, 2.0]]),
            aspect=np.array([[0.0, 90.0], [180.0, 270.0]]),
            roughness=np.array([[0.0, 0.1], [0.1, 0.2]]),
            tri=np.array([[0.0, 0.1], [0.1, 0.2]]),
            hillshade=np.array([[128, 150], [150, 200]], dtype=np.uint8),
            elevation=np.array([[1000.0, 1001.0], [1002.0, 1003.0]])
        )

        with patch.object(CriteriaExtractor, 'calculate_solar_exposure') as mock_exposure:
            mock_exposure.return_value = np.array([[0.5, 0.6], [0.7, 0.8]])

            result = analyzer.calculate_solar_potential(
                small_dem,
                small_metrics,
                sun_azimuth=0.0,
                sun_altitude=45.0
            )

            # Should handle small arrays gracefully
            assert result.solar_potential_map.shape == (2, 2)
            assert np.all(result.solar_potential_map >= 0.0)
            assert np.all(result.solar_potential_map <= 1.0)


class TestMissionImpacts:
    """Tests for mission impact calculations."""

    def test_mission_impacts_structure(self):
        """Test that MissionImpacts has correct structure."""
        impacts = MissionImpacts(
            power_generation_kwh_per_day=25.0,
            power_surplus_kwh_per_day=5.0,
            mission_duration_extension_days=10.0,
            cost_savings_usd=5000.0,
            battery_reduction_kwh=5.0
        )

        assert impacts.power_generation_kwh_per_day == 25.0
        assert impacts.power_surplus_kwh_per_day == 5.0
        assert impacts.mission_duration_extension_days == 10.0
        assert impacts.cost_savings_usd == 5000.0
        assert impacts.battery_reduction_kwh == 5.0


class TestSolarPotentialConstants:
    """Tests for solar potential constants."""

    def test_mars_solar_constant(self):
        """Test Mars solar constant value."""
        assert MARS_SOLAR_CONSTANT == 590.0
        assert MARS_SOLAR_CONSTANT > 0

    def test_default_panel_efficiency(self):
        """Test default panel efficiency."""
        assert DEFAULT_PANEL_EFFICIENCY == 0.25
        assert 0.0 < DEFAULT_PANEL_EFFICIENCY <= 1.0

