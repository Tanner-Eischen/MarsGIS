"""Unit tests for configuration management."""

from pathlib import Path
import pytest
import yaml

from marshab.config import Config, PathsConfig
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
