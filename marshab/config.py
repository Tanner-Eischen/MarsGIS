"""Configuration management with YAML support."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MarsParameters(BaseModel):
    """Mars planetary parameters."""

    equatorial_radius_m: float = Field(
        3396190.0, description="Mars equatorial radius (meters)"
    )
    polar_radius_m: float = Field(3376200.0, description="Mars polar radius (meters)")
    crs: str = Field("IAU_MARS_2000", description="Mars coordinate reference system")
    datum: str = Field("D_Mars_2000", description="Mars geodetic datum")


class DataSource(BaseModel):
    """External data source configuration."""

    url: str = Field(..., description="URL to data source")
    resolution_m: float = Field(..., description="Data resolution (meters/pixel)")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field("INFO", description="Log level (DEBUG/INFO/WARNING/ERROR)")
    format: str = Field("console", description="Log format (console/json)")
    file: Optional[Path] = Field(None, description="Optional log file path")


class PathsConfig(BaseModel):
    """File system paths configuration."""

    data_dir: Path = Field(Path("data"), description="Data directory")
    cache_dir: Path = Field(Path("data/cache"), description="Cache directory")
    output_dir: Path = Field(Path("data/output"), description="Output directory")
    spice_kernels: Path = Field(
        Path("/usr/local/share/spice"), description="SPICE kernel directory"
    )

    def create_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for path in [self.data_dir, self.cache_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Config(BaseSettings):
    """Main application configuration."""

    mars: MarsParameters = Field(default_factory=MarsParameters)
    data_sources: dict[str, DataSource] = Field(default_factory=dict)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    class Settings:
        env_prefix = "MARSHAB_"
        env_nested_delimiter = "__"
        case_sensitive = False

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from file or environment.

        Priority:
        1. MARSHAB_CONFIG_PATH environment variable
        2. ./marshab_config.yaml in current directory
        3. ~/.config/marshab/config.yaml in home directory
        4. Default configuration with environment overrides
        """
        config_path = os.getenv("MARSHAB_CONFIG_PATH")

        if config_path and Path(config_path).exists():
            return cls.from_yaml(Path(config_path))

        # Check default locations
        default_paths = [
            Path("marshab_config.yaml"),
            Path.home() / ".config" / "marshab" / "config.yaml",
        ]

        for path in default_paths:
            if path.exists():
                return cls.from_yaml(path)

        # Fall back to defaults with environment overrides
        return cls()


# Global config instance (singleton pattern)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance (lazy-loaded singleton).

    Returns:
        Config instance, creating and caching it on first call
    """
    global _config
    if _config is None:
        _config = Config.load()
        _config.paths.create_directories()
    return _config


def reset_config() -> None:
    """Reset global config instance (for testing)."""
    global _config
    _config = None

