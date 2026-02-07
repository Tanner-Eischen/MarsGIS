"""Unit tests for synthetic DEM generation."""

import numpy as np

from marshab.testing.synthetic_dem import (
    create_synthetic_dem_complex,
    create_synthetic_dem_crater,
    create_synthetic_dem_hill,
    create_synthetic_dem_plane,
)


class TestSyntheticDEM:
    """Tests for synthetic DEM generation."""

    def test_create_plane(self):
        """Test creating a flat plane DEM."""
        dem = create_synthetic_dem_plane(
            size=(100, 100),
            elevation=2000.0,
            cell_size_m=100.0
        )

        assert dem.shape == (100, 100)
        assert np.allclose(dem.values, 2000.0)

    def test_create_hill(self):
        """Test creating a hill DEM."""
        dem = create_synthetic_dem_hill(
            size=(100, 100),
            center=(50, 50),
            height=500.0,
            radius=20.0,
            base_elevation=2000.0
        )

        assert dem.shape == (100, 100)
        # Center should be highest
        center_elevation = dem.values[50, 50]
        edge_elevation = dem.values[0, 0]
        assert center_elevation > edge_elevation
        assert center_elevation > 2000.0

    def test_create_crater(self):
        """Test creating a crater DEM."""
        dem = create_synthetic_dem_crater(
            size=(100, 100),
            center=(50, 50),
            depth=300.0,
            radius=15.0,
            base_elevation=2000.0
        )

        assert dem.shape == (100, 100)
        # Center should be lowest
        center_elevation = dem.values[50, 50]
        edge_elevation = dem.values[0, 0]
        assert center_elevation < edge_elevation
        assert center_elevation < 2000.0

    def test_create_complex(self):
        """Test creating a complex DEM with multiple features."""
        features = [
            {"type": "hill", "center": (30, 30), "height": 200.0, "radius": 10.0},
            {"type": "crater", "center": (70, 70), "depth": 150.0, "radius": 8.0}
        ]

        dem = create_synthetic_dem_complex(
            size=(100, 100),
            features=features,
            base_elevation=2000.0
        )

        assert dem.shape == (100, 100)
        # Hill location should be elevated
        assert dem.values[30, 30] > 2000.0
        # Crater location should be depressed
        assert dem.values[70, 70] < 2000.0

