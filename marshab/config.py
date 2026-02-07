"""Configuration management with YAML support."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from marshab.models import CriteriaWeights


class PathfindingStrategy(str, Enum):
    """Pathfinding strategy presets."""

    SAFEST = "safest"
    BALANCED = "balanced"
    DIRECT = "direct"


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


class AnalysisConfig(BaseModel):
    """Configuration for terrain analysis pipeline."""

    criteria_weights: CriteriaWeights = Field(default_factory=CriteriaWeights)
    max_slope_deg: float = Field(5.0, gt=0, description="Maximum traversable slope (degrees)")
    max_roughness: float = Field(0.5, gt=0, description="Maximum traversable roughness")
    min_site_area_km2: float = Field(0.5, gt=0, description="Minimum site area (km²)")
    suitability_threshold: float = Field(0.7, ge=0, le=1, description="Suitability score threshold")


class NavigationConfig(BaseModel):
    """Configuration for rover navigation and pathfinding."""

    strategy: PathfindingStrategy = Field(
        PathfindingStrategy.BALANCED,
        description="Pathfinding strategy: safest (prioritize safety), balanced (default), direct (prioritize distance)"
    )
    slope_weight: float = Field(10.0, gt=0, description="Slope cost multiplier for pathfinding")
    roughness_weight: float = Field(5.0, gt=0, description="Roughness cost multiplier for pathfinding")
    distance_weight: float = Field(1.0, gt=0, description="Distance cost multiplier for pathfinding")
    enable_smoothing: bool = Field(True, description="Enable path smoothing to reduce waypoints")
    smoothing_tolerance: float = Field(2.0, gt=0, description="Path smoothing tolerance (cells)")
    cliff_threshold_m: float = Field(10.0, gt=0, description="Elevation change threshold for cliff detection (meters)")
    max_roughness_m: float = Field(20.0, gt=0, description="Maximum traversable roughness in meters (roughness is std dev of elevation)")

    def get_weights_for_strategy(self) -> dict[str, float]:
        """Get weight configuration based on selected strategy.

        Returns:
            Dictionary with slope_weight, roughness_weight, distance_weight
        """
        presets = {
            PathfindingStrategy.SAFEST: {
                "slope_weight": 50.0,
                "roughness_weight": 30.0,
                "distance_weight": 1.0,
            },
            PathfindingStrategy.BALANCED: {
                "slope_weight": 10.0,
                "roughness_weight": 5.0,
                "distance_weight": 1.0,
            },
            PathfindingStrategy.DIRECT: {
                "slope_weight": 2.0,
                "roughness_weight": 1.0,
                "distance_weight": 2.0,
            },
        }

        if self.strategy in presets:
            return presets[self.strategy]
        else:
            # Use configured values if strategy is custom
            return {
                "slope_weight": self.slope_weight,
                "roughness_weight": self.roughness_weight,
                "distance_weight": self.distance_weight,
            }


class Config(BaseSettings):
    """Main application configuration."""

    mars: MarsParameters = Field(default_factory=MarsParameters)
    data_sources: dict[str, DataSource] = Field(default_factory=dict)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    demo_mode: bool = Field(False)

    model_config = SettingsConfigDict(
        env_prefix="MARSHAB_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from YAML
    )

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

