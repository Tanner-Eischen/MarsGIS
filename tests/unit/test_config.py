"""Unit tests for configuration management."""

from pathlib import Path
import pytest
import yaml

from marshab.config import Config, PathsConfig, NavigationConfig, PathfindingStrategy
from marshab.exceptions import ConfigurationError


def test_default_config():
    """Test default configuration loads."""
    config = Config()
    
    assert config.mars.equatorial_radius_m == 3396190.0
    assert config.mars.crs == "IAU_MARS_2000"
    assert config.logging.level == "INFO"


def test_config_from_yaml(tmp_path: Path):
    """Test loading configuration from YAML file."""
    config_file = tmp_path / "test_config.yaml"
    
    config_data = {
        "mars": {"equatorial_radius_m": 3396190.0},
        "logging": {"level": "DEBUG"},
        "paths": {
            "data_dir": "test_data",
            "cache_dir": "test_cache",
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    config = Config.from_yaml(config_file)
    
    assert config.logging.level == "DEBUG"
    assert config.paths.data_dir == Path("test_data")


def test_paths_create_directories(tmp_path: Path):
    """Test directory creation."""
    paths = PathsConfig(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )

    paths.create_directories()

    assert paths.data_dir.exists()
    assert paths.cache_dir.exists()
    assert paths.output_dir.exists()


def test_config_file_not_found():
    """Test error when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(Path("nonexistent.yaml"))


def test_navigation_config_defaults():
    """Test NavigationConfig default values."""
    nav_config = NavigationConfig()
    
    assert nav_config.strategy == PathfindingStrategy.BALANCED
    assert nav_config.slope_weight == 10.0
    assert nav_config.roughness_weight == 5.0
    assert nav_config.enable_smoothing is True
    assert nav_config.cliff_threshold_m == 10.0


def test_navigation_config_strategy_presets():
    """Test strategy preset weights."""
    # Safest strategy
    nav_safest = NavigationConfig(strategy=PathfindingStrategy.SAFEST)
    weights_safest = nav_safest.get_weights_for_strategy()
    assert weights_safest["slope_weight"] == 50.0
    assert weights_safest["roughness_weight"] == 30.0
    assert weights_safest["distance_weight"] == 1.0
    
    # Balanced strategy
    nav_balanced = NavigationConfig(strategy=PathfindingStrategy.BALANCED)
    weights_balanced = nav_balanced.get_weights_for_strategy()
    assert weights_balanced["slope_weight"] == 10.0
    assert weights_balanced["roughness_weight"] == 5.0
    assert weights_balanced["distance_weight"] == 1.0
    
    # Direct strategy
    nav_direct = NavigationConfig(strategy=PathfindingStrategy.DIRECT)
    weights_direct = nav_direct.get_weights_for_strategy()
    assert weights_direct["slope_weight"] == 2.0
    assert weights_direct["roughness_weight"] == 1.0
    assert weights_direct["distance_weight"] == 2.0


def test_config_with_navigation(tmp_path: Path):
    """Test loading config with navigation section."""
    config_file = tmp_path / "test_config.yaml"
    
    config_data = {
        "navigation": {
            "strategy": "safest",
            "slope_weight": 50.0,
            "roughness_weight": 30.0,
            "enable_smoothing": True,
            "cliff_threshold_m": 15.0
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    config = Config.from_yaml(config_file)
    
    assert config.navigation.strategy == PathfindingStrategy.SAFEST
    assert config.navigation.slope_weight == 50.0
    assert config.navigation.enable_smoothing is True
