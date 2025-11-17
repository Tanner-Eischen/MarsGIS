"""Unit tests for terrain analysis."""

import numpy as np
import pytest
import xarray as xr

from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface, detect_cliffs


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


def test_detect_cliffs():
    """Test cliff detection."""
    # Create elevation with a sudden drop (cliff)
    elevation = np.ones((20, 20)) * 1000.0
    # Create a cliff: sudden drop of 15m
    elevation[10:, :] = 985.0
    
    cell_size_m = 100.0
    threshold_m = 10.0
    
    cliff_mask = detect_cliffs(elevation, cell_size_m, threshold_m)
    
    # Should detect cliffs near the transition
    assert np.any(cliff_mask)
    # Cliffs should be near row 10 (transition point)
    assert np.any(cliff_mask[9:12, :])


def test_detect_cliffs_no_cliffs():
    """Test cliff detection with smooth terrain."""
    # Smooth elevation gradient
    elevation = np.linspace(1000, 1010, 20).reshape(20, 1)
    elevation = np.tile(elevation, (1, 20))
    
    cell_size_m = 100.0
    threshold_m = 10.0
    
    cliff_mask = detect_cliffs(elevation, cell_size_m, threshold_m)
    
    # Should not detect cliffs in smooth terrain
    assert not np.any(cliff_mask)


def test_generate_cost_surface_with_weights():
    """Test cost surface generation with custom weights."""
    slope = np.ones((10, 10)) * 10.0  # 10 degrees
    roughness = np.ones((10, 10)) * 0.5
    
    # Low weights
    cost_low = generate_cost_surface(
        slope, roughness, max_slope_deg=25.0,
        slope_weight=1.0, roughness_weight=1.0
    )
    
    # High weights
    cost_high = generate_cost_surface(
        slope, roughness, max_slope_deg=25.0,
        slope_weight=50.0, roughness_weight=30.0
    )
    
    # High weights should produce higher costs
    assert np.mean(cost_high) > np.mean(cost_low)


def test_generate_cost_surface_with_cliffs():
    """Test cost surface generation with cliff detection."""
    slope = np.ones((20, 20)) * 5.0
    roughness = np.ones((20, 20)) * 0.1
    
    # Create elevation with a cliff
    elevation = np.ones((20, 20)) * 1000.0
    elevation[10:, :] = 980.0  # 20m drop
    
    cell_size_m = 100.0
    cliff_threshold_m = 10.0
    
    cost = generate_cost_surface(
        slope, roughness, max_slope_deg=25.0,
        elevation=elevation,
        cell_size_m=cell_size_m,
        cliff_threshold_m=cliff_threshold_m
    )
    
    # Cliffs should be marked as impassable
    assert np.any(np.isinf(cost))
    # Areas without cliffs should be passable
    assert np.any(np.isfinite(cost))

