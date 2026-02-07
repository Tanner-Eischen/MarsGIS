"""Unit tests for plugin system."""


from marshab.plugins import PLUGIN_REGISTRY, register_criterion_plugin
from marshab.plugins.example_plugin import ExampleCriterionPlugin


class TestPluginRegistry:
    """Tests for plugin registry."""

    def test_register_criterion_plugin(self):
        """Test registering a criterion plugin."""
        # Clear registry
        PLUGIN_REGISTRY["criteria"].clear()

        plugin = ExampleCriterionPlugin()
        register_criterion_plugin(plugin)

        assert len(PLUGIN_REGISTRY["criteria"]) > 0
        assert "example_metric" in PLUGIN_REGISTRY["criteria"]

    def test_example_plugin_get_criteria(self):
        """Test example plugin returns criteria."""
        plugin = ExampleCriterionPlugin()
        criteria = plugin.get_criteria()

        assert len(criteria) == 1
        assert criteria[0]["id"] == "example_metric"
        assert criteria[0]["name"] == "Example Metric"

    def test_example_plugin_calculate(self):
        """Test example plugin calculation."""
        import numpy as np
        import xarray as xr

        plugin = ExampleCriterionPlugin()

        # Create mock DEM
        elevation = np.array([[1000, 2000], [1500, 2500]])
        dem = xr.DataArray(elevation, dims=["y", "x"])

        result = plugin.calculate(dem, {})

        assert result.shape == elevation.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)

