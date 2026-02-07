"""Tests for preset loader."""


import pytest

from marshab.config.preset_loader import PresetLoader


class TestPresetLoader:
    """Tests for PresetLoader class."""

    @pytest.fixture
    def loader(self):
        """Provide PresetLoader with test config."""
        return PresetLoader()

    def test_load_presets(self, loader):
        """Test loading presets from YAML."""
        config = loader.load()

        assert config is not None
        assert len(config.site_presets) > 0
        assert len(config.route_presets) > 0
        assert len(config.criteria) > 0

    def test_get_site_preset(self, loader):
        """Test retrieving site preset."""
        loader.load()
        preset = loader.get_preset("safe_landing", "site")

        assert preset is not None
        assert preset.name == "Safe Landing"
        assert preset.scope == "site"
        assert "slope" in preset.get_weights_dict()

    def test_weights_validation(self, loader):
        """Test preset weights sum validation."""
        loader.load()
        preset = loader.get_preset("balanced", "site")

        assert preset.validate_weights_sum()

    def test_list_presets(self, loader):
        """Test listing all presets."""
        loader.load()

        all_presets = loader.list_presets()
        assert len(all_presets) > 0

        site_presets = loader.list_presets(scope="site")
        route_presets = loader.list_presets(scope="route")

        assert len(site_presets) > 0
        assert len(route_presets) > 0

    def test_get_criterion(self, loader):
        """Test retrieving criterion definition."""
        loader.load()
        criterion = loader.get_criterion("slope")

        assert criterion is not None
        assert criterion.display_name == "Slope Safety"
        assert criterion.beneficial is False
        assert criterion.unit == "degrees"

    def test_get_nonexistent_preset(self, loader):
        """Test retrieving nonexistent preset returns None."""
        loader.load()
        preset = loader.get_preset("nonexistent", "site")

        assert preset is None

    def test_get_weights_dict(self, loader):
        """Test getting weights as dictionary."""
        loader.load()
        preset = loader.get_preset("safe_landing", "site")

        weights = preset.get_weights_dict()
        assert isinstance(weights, dict)
        assert "slope" in weights
        assert weights["slope"] == 0.40

