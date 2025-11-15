"""Unit tests for terrain analysis."""

import numpy as np
import pytest
import xarray as xr

from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface


def test_calculate_slope():
    """Test slope calculation."""
    # Create simple sloped surface
    dem = np.tile(np.arange(50), (50, 1)).T  # Constant slope in one direction
    
    analyzer = TerrainAnalyzer(cell_size_m=100.0)
    slope = analyzer.calculate_slope(dem.astype(float))
    
    assert slope.shape == dem.shape
    assert np.all(slope >= 0)
    assert np.all(slope <= 90)
    
    # Slope should be roughly constant (not exactly due to edge effects)
    assert np.std(slope[5:-5, 5:-5]) < 1.0


def test_calculate_aspect():
    """Test aspect calculation."""
    dem = np.random.randn(50, 50) * 10
    
    analyzer = TerrainAnalyzer()
    aspect = analyzer.calculate_aspect(dem)
    
    assert aspect.shape == dem.shape
    assert np.all(aspect >= 0)
    assert np.all(aspect <= 360)


def test_calculate_roughness():
    """Test roughness calculation."""
    # Smooth surface
    smooth = np.ones((50, 50)) * 100.0
    
    # Rough surface
    rough = np.random.randn(50, 50) * 50 + 100.0
    
    analyzer = TerrainAnalyzer()
    
    roughness_smooth = analyzer.calculate_roughness(smooth)
    roughness_rough = analyzer.calculate_roughness(rough)
    
    # Rough surface should have higher roughness
    assert np.mean(roughness_rough) > np.mean(roughness_smooth)


def test_calculate_tri():
    """Test Terrain Ruggedness Index calculation."""
    # Create test DEM with known variation
    dem = np.random.randn(50, 50) * 10 + 100
    
    analyzer = TerrainAnalyzer()
    tri = analyzer.calculate_tri(dem)
    
    assert tri.shape == dem.shape
    assert np.all(tri >= 0)
    
    # TRI should be non-zero for varied terrain
    assert np.mean(tri) > 0


def test_terrain_analyzer_full(sample_terrain_data):
    """Test complete terrain analysis."""
    analyzer = TerrainAnalyzer(cell_size_m=200.0)
    
    dem = xr.DataArray(sample_terrain_data)
    
    metrics = analyzer.analyze(dem)
    
    assert metrics.slope.shape == sample_terrain_data.shape
    assert metrics.aspect.shape == sample_terrain_data.shape
    assert metrics.roughness.shape == sample_terrain_data.shape
    assert metrics.tri.shape == sample_terrain_data.shape
    
    # Verify all metrics are valid
    assert np.all(metrics.slope >= 0)
    assert np.all(metrics.slope <= 90)
    assert np.all(metrics.aspect >= 0)
    assert np.all(metrics.aspect <= 360)
    assert np.all(metrics.roughness >= 0)
    assert np.all(metrics.tri >= 0)


def test_generate_cost_surface():
    """Test cost surface generation."""
    slope = np.random.rand(50, 50) * 30  # 0-30 degrees
    roughness = np.random.rand(50, 50) * 0.5
    
    cost = generate_cost_surface(slope, roughness, max_slope_deg=25.0)
    
    # High slopes should be impassable
    assert np.all(np.isinf(cost[slope > 25.0]))
    
    # Other areas should have finite cost
    assert np.all(np.isfinite(cost[slope <= 25.0]))
    
    # Cost should be positive
    assert np.all(cost[slope <= 25.0] > 0)


def test_generate_cost_surface_edge_cases():
    """Test cost surface with edge cases."""
    # Zero slope, zero roughness
    slope = np.zeros((10, 10))
    roughness = np.zeros((10, 10))
    
    cost = generate_cost_surface(slope, roughness, max_slope_deg=25.0)
    
    # Should have base cost of 1.0
    assert np.allclose(cost, 1.0)
    
    # All slopes above max
    slope = np.ones((10, 10)) * 30.0
    roughness = np.zeros((10, 10))
    
    cost = generate_cost_surface(slope, roughness, max_slope_deg=25.0)
    
    # All should be impassable
    assert np.all(np.isinf(cost))

