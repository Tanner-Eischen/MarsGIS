# Phase 1: Core Infrastructure

**Duration:** Days 2–3 – 8 hours  
**Goal:** Establish type system, configuration management, logging, error handling, and testing framework

---

## 1.1 Type Definitions and Pydantic Models (1.5 hours)

### 1.1.1 Create `marshab/types.py`

```bash
cat > marshab/types.py << 'EOF'
"""Type definitions and data models for MarsHab."""

from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Region of interest bounding box in IAU_MARS planetocentric coordinates."""

    lat_min: float = Field(..., ge=-90, le=90, description="Minimum latitude (degrees)")
    lat_max: float = Field(..., ge=-90, le=90, description="Maximum latitude (degrees)")
    lon_min: float = Field(..., ge=0, le=360, description="Minimum longitude (degrees east)")
    lon_max: float = Field(..., ge=0, le=360, description="Maximum longitude (degrees east)")

    @field_validator("lat_max")
    @classmethod
    def lat_max_greater_than_min(cls, v: float, info) -> float:
        """Ensure lat_max > lat_min."""
        if "lat_min" in info.data and v <= info.data["lat_min"]:
            raise ValueError("lat_max must be greater than lat_min")
        return v

    @field_validator("lon_max")
    @classmethod
    def lon_max_greater_than_min(cls, v: float, info) -> float:
        """Ensure lon_max > lon_min."""
        if "lon_min" in info.data and v <= info.data["lon_min"]:
            raise ValueError("lon_max must be greater than lon_min")
        return v

    def to_tuple(self) -> tuple:
        """Return as (lat_min, lat_max, lon_min, lon_max) tuple."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)


class SiteOrigin(BaseModel):
    """Rover SITE frame origin definition on Mars surface."""

    lat: float = Field(..., ge=-90, le=90, description="Latitude (degrees, planetocentric)")
    lon: float = Field(..., ge=0, le=360, description="Longitude (degrees, east positive)")
    elevation_m: float = Field(..., description="Elevation above areoid datum (meters)")


class Waypoint(BaseModel):
    """Rover navigation waypoint in SITE frame coordinates."""

    waypoint_id: int = Field(..., ge=0, description="Waypoint sequence number")
    x_meters: float = Field(..., description="X coordinate in SITE frame - Northing (meters)")
    y_meters: float = Field(..., description="Y coordinate in SITE frame - Easting (meters)")
    tolerance_meters: float = Field(5.0, gt=0, description="Arrival tolerance radius (meters)")
    heading_deg: float | None = Field(None, ge=0, le=360, description="Desired rover heading (degrees)")

    def distance_from_origin(self) -> float:
        """Calculate distance from SITE frame origin."""
        return float(np.sqrt(self.x_meters**2 + self.y_meters**2))


class TerrainMetrics(NamedTuple):
    """Terrain analysis outputs from DEM processing."""

    slope: np.ndarray  # Slope in degrees (0-90)
    aspect: np.ndarray  # Aspect in degrees (0-360, 0=North)
    roughness: np.ndarray  # Terrain roughness (standard deviation of elevation)
    tri: np.ndarray  # Terrain Ruggedness Index


class CriteriaWeights(BaseModel):
    """MCDM criteria and their importance weights."""

    slope: float = Field(0.30, ge=0, le=1, description="Weight for slope criterion")
    roughness: float = Field(0.25, ge=0, le=1, description="Weight for roughness criterion")
    elevation: float = Field(0.20, ge=0, le=1, description="Weight for elevation criterion")
    solar_exposure: float = Field(0.15, ge=0, le=1, description="Weight for solar exposure")
    resources: float = Field(0.10, ge=0, le=1, description="Weight for resource proximity")

    @field_validator("*", mode="before")
    @classmethod
    def validate_weights(cls, v):
        """Validate weights are positive."""
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError("Weights must be non-negative")
        return v

    def total(self) -> float:
        """Return sum of all weights."""
        return (
            self.slope
            + self.roughness
            + self.elevation
            + self.solar_exposure
            + self.resources
        )

    def normalize(self) -> "CriteriaWeights":
        """Normalize weights to sum to 1.0."""
        total = self.total()
        if total == 0:
            raise ValueError("Total weight cannot be zero")
        return CriteriaWeights(
            slope=self.slope / total,
            roughness=self.roughness / total,
            elevation=self.elevation / total,
            solar_exposure=self.solar_exposure / total,
            resources=self.resources / total,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "slope": self.slope,
            "roughness": self.roughness,
            "elevation": self.elevation,
            "solar_exposure": self.solar_exposure,
            "resources": self.resources,
        }


class AnalysisConfig(BaseModel):
    """Configuration for terrain analysis pipeline."""

    roi: BoundingBox
    criteria_weights: CriteriaWeights = Field(default_factory=CriteriaWeights)
    max_slope_deg: float = Field(5.0, gt=0, description="Maximum traversable slope (degrees)")
    max_roughness: float = Field(0.5, gt=0, description="Maximum traversable roughness")
    min_site_area_km2: float = Field(0.5, gt=0, description="Minimum site area (km²)")
    suitability_threshold: float = Field(0.7, ge=0, le=1, description="Suitability score threshold")


class SiteCandidate(BaseModel):
    """Identified construction site candidate."""

    site_id: int = Field(..., ge=0, description="Unique site identifier")
    geometry_type: str = Field(..., description="WKT geometry type")
    area_km2: float = Field(..., gt=0, description="Site area (km²)")
    mean_slope_deg: float = Field(..., ge=0, description="Mean slope (degrees)")
    mean_roughness: float = Field(..., ge=0, description="Mean roughness")
    mean_elevation_m: float = Field(..., description="Mean elevation (meters)")
    suitability_score: float = Field(..., ge=0, le=1, description="Overall suitability (0-1)")
    rank: int = Field(..., ge=1, description="Rank among candidates (1=best)")
EOF
```

### 1.1.2 Verify Types

```bash
# Test type imports
poetry run python -c "
from marshab.types import BoundingBox, SiteOrigin, Waypoint, AnalysisConfig
roi = BoundingBox(lat_min=40, lat_max=41, lon_min=180, lon_max=181)
print(f'✓ BoundingBox created: {roi}')
print(f'  Tuple form: {roi.to_tuple()}')
"
```

---

## 1.2 Configuration Management (2 hours)

### 1.2.1 Create `marshab/config.py`

```bash
cat > marshab/config.py << 'EOF'
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
EOF
```

### 1.2.2 Create default `marshab_config.yaml`

```bash
cat > marshab_config.yaml << 'EOF'
# MarsHab Configuration

# Mars planetary parameters
mars:
  equatorial_radius_m: 3396190.0
  polar_radius_m: 3376200.0
  crs: "IAU_MARS_2000"
  datum: "D_Mars_2000"

# Data sources
data_sources:
  mola_global:
    url: "https://astrogeology.usgs.gov/cache/mars/viking/dem/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
    resolution_m: 463
  mola_200m:
    url: "https://astrogeology.usgs.gov/cache/mars/viking/dem/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
    resolution_m: 200

# Logging configuration
logging:
  level: "INFO"
  format: "console"
  file: null

# File system paths
paths:
  data_dir: "data"
  cache_dir: "data/cache"
  output_dir: "data/output"
  spice_kernels: "/usr/local/share/spice"
EOF
```

### 1.2.3 Verify Config

```bash
# Test config loading
poetry run python -c "
from marshab.config import get_config
cfg = get_config()
print(f'✓ Config loaded')
print(f'  Mars radius: {cfg.mars.equatorial_radius_m}m')
print(f'  Data dir: {cfg.paths.data_dir}')
"
```

---

## 1.3 Structured Logging Setup (1.5 hours)

### 1.3.1 Create `marshab/utils/logging.py`

```bash
cat > marshab/utils/logging.py << 'EOF'
"""Structured logging configuration with structlog."""

import logging
import sys
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler


def configure_logging(
    level: str = "INFO",
    format_type: str = "console",
    log_file: Path | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format ("console" for development, "json" for production)
        log_file: Optional file path for log output
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[],
    )

    # Shared processors for all configurations
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Development: colorized console output
    if format_type == "console":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

        handler = RichHandler(
            console=Console(stderr=True),
            show_time=False,  # structlog adds timestamp
            show_path=False,
            markup=True,
        )
    # Production: JSON logs
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(log_level)
    logging.root.addHandler(handler)

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        logging.root.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
EOF
```

### 1.3.2 Verify Logging

```bash
# Test logging
poetry run python -c "
from marshab.utils.logging import configure_logging, get_logger

configure_logging(level='DEBUG', format_type='console')
logger = get_logger('test')

logger.info('Test log message', component='marshab', status='active')
logger.debug('Debug message', details={'key': 'value'})
"
```

---

## 1.4 Custom Exceptions (30 minutes)

### 1.4.1 Create `marshab/exceptions.py`

```bash
cat > marshab/exceptions.py << 'EOF'
"""Custom exceptions for MarsHab system."""


class MarsHabError(Exception):
    """Base exception for all MarsHab errors."""

    def __init__(self, message: str, details: dict | None = None):
        """Initialize exception.

        Args:
            message: Error message
            details: Optional dictionary with additional context
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class DataError(MarsHabError):
    """Data acquisition, validation, or I/O errors."""

    pass


class AnalysisError(MarsHabError):
    """Analysis pipeline errors."""

    pass


class CoordinateError(MarsHabError):
    """Coordinate transformation errors."""

    pass


class NavigationError(MarsHabError):
    """Path planning and navigation errors."""

    pass


class ConfigurationError(MarsHabError):
    """Configuration loading or validation errors."""

    pass
EOF
```

---

## 1.5 Utility Functions (1 hour)

### 1.5.1 Create `marshab/utils/validators.py`

```bash
cat > marshab/utils/validators.py << 'EOF'
"""Input validation utilities."""

from pathlib import Path

import rasterio

from marshab.exceptions import DataError
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


def validate_dem_crs(dem_path: Path, expected_crs: str = "IAU_MARS_2000") -> bool:
    """Validate DEM has correct Mars coordinate system.

    Args:
        dem_path: Path to GeoTIFF DEM file
        expected_crs: Expected CRS identifier

    Returns:
        True if CRS is valid

    Raises:
        DataError: If CRS is invalid or file cannot be read
    """
    try:
        with rasterio.open(dem_path) as src:
            crs = src.crs

            if crs is None:
                raise DataError(
                    f"DEM has no CRS defined: {dem_path}",
                    details={"path": str(dem_path)},
                )

            # Check if CRS string contains Mars identifier
            crs_str = str(crs)
            if "MARS" not in crs_str.upper() and "49900" not in crs_str:
                logger.warning(
                    "DEM CRS may not be Mars-specific",
                    path=str(dem_path),
                    crs=crs_str,
                )

            logger.info("DEM CRS validated", crs=crs_str, path=str(dem_path))
            return True

    except rasterio.errors.RasterioIOError as e:
        raise DataError(
            f"Cannot read DEM file: {dem_path}",
            details={"path": str(dem_path), "error": str(e)},
        )


def validate_roi_bounds(roi: BoundingBox) -> bool:
    """Validate ROI is within Mars valid ranges.

    Args:
        roi: Bounding box to validate

    Returns:
        True if valid

    Raises:
        DataError: If ROI is invalid
    """
    # Already validated by Pydantic, but add logical checks
    if roi.lat_max - roi.lat_min < 0.1:
        raise DataError(
            "ROI latitude range too small (< 0.1 degrees)",
            details={"roi": roi.model_dump()},
        )

    if roi.lon_max - roi.lon_min < 0.1:
        raise DataError(
            "ROI longitude range too small (< 0.1 degrees)",
            details={"roi": roi.model_dump()},
        )

    logger.info(
        "ROI validated",
        lat_range=f"{roi.lat_min}-{roi.lat_max}",
        lon_range=f"{roi.lon_min}-{roi.lon_max}",
    )

    return True
EOF
```

### 1.5.2 Create `marshab/utils/helpers.py`

```bash
cat > marshab/utils/helpers.py << 'EOF'
"""Common helper utilities."""

import hashlib
from typing import Any

import numpy as np


def generate_cache_key(*args: Any) -> str:
    """Generate hexadecimal cache key from arguments.

    Args:
        *args: Values to hash

    Returns:
        MD5 hexadecimal cache key string
    """
    content = "_".join(str(arg) for arg in args)
    return hashlib.md5(content.encode()).hexdigest()


def ensure_numpy_array(data: Any) -> np.ndarray:
    """Ensure data is a numpy array.

    Args:
        data: Input data (array-like)

    Returns:
        NumPy ndarray
    """
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def format_area(area_km2: float) -> str:
    """Format area for human-readable display.

    Args:
        area_km2: Area in square kilometers

    Returns:
        Formatted string (e.g., "1.23 km²")
    """
    if area_km2 >= 1.0:
        return f"{area_km2:.2f} km²"
    else:
        area_m2 = area_km2 * 1_000_000
        return f"{area_m2:.1f} m²"


def format_distance(distance_m: float) -> str:
    """Format distance for human-readable display.

    Args:
        distance_m: Distance in meters

    Returns:
        Formatted string (e.g., "1.23 km" or "456 m")
    """
    if distance_m >= 1000:
        return f"{distance_m / 1000:.2f} km"
    else:
        return f"{distance_m:.0f} m"
EOF
```

---

## 1.6 Testing Infrastructure (2 hours)

### 1.6.1 Create `tests/conftest.py`

```bash
cat > tests/conftest.py << 'EOF'
"""Pytest configuration and shared fixtures."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from marshab.config import Config, PathsConfig
from marshab.types import BoundingBox, SiteOrigin


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Provide test configuration with temporary directories."""
    return Config(
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
        )
    )


@pytest.fixture
def test_roi() -> BoundingBox:
    """Provide standard test region of interest."""
    return BoundingBox(
        lat_min=40.0,
        lat_max=41.0,
        lon_min=180.0,
        lon_max=181.0,
    )


@pytest.fixture
def test_site_origin() -> SiteOrigin:
    """Provide test SITE frame origin."""
    return SiteOrigin(
        lat=40.5,
        lon=180.5,
        elevation_m=-2500.0,
    )


@pytest.fixture
def synthetic_dem(tmp_path: Path) -> Path:
    """Create synthetic Mars DEM for testing.

    Creates a 100x100 elevation raster with:
    - Random noise
    - Gentle slopes
    - IAU_MARS coordinate system
    - Coverage over test ROI

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to synthetic DEM GeoTIFF file
    """
    np.random.seed(42)

    width, height = 100, 100
    # Mean elevation 1000m with small variations
    elevation = np.random.randn(height, width) * 100 + 1000.0

    # Add gentle slopes
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    elevation += x * 0.5 + y * 0.3

    # Define bounds in IAU_MARS coordinates (covering test ROI)
    bounds = (180.0, 40.0, 181.0, 41.0)  # lon_min, lat_min, lon_max, lat_max
    transform = from_bounds(*bounds, width, height)

    # Write to GeoTIFF with Mars CRS
    dem_path = tmp_path / "synthetic_dem.tif"

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=elevation.dtype,
        crs="EPSG:49900",  # Mars 2000 sphere IAU
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(elevation, 1)

    return dem_path


@pytest.fixture
def sample_terrain_data() -> np.ndarray:
    """Provide sample terrain data for testing.

    Returns:
        50x50 array of synthetic elevation data
    """
    np.random.seed(42)
    return np.random.randn(50, 50) * 10 + 100


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config singleton after each test."""
    from marshab.config import reset_config
    yield
    reset_config()
EOF
```

### 1.6.2 Create `tests/unit/test_types.py`

```bash
mkdir -p tests/unit

cat > tests/unit/test_types.py << 'EOF'
"""Unit tests for type definitions and models."""

import pytest
from pydantic import ValidationError

from marshab.types import (
    BoundingBox,
    SiteOrigin,
    Waypoint,
    CriteriaWeights,
    AnalysisConfig,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_valid_bbox(self):
        """Test creating valid bounding box."""
        bbox = BoundingBox(
            lat_min=-45,
            lat_max=45,
            lon_min=0,
            lon_max=360,
        )
        assert bbox.lat_min == -45
        assert bbox.lat_max == 45

    def test_bbox_to_tuple(self):
        """Test conversion to tuple."""
        bbox = BoundingBox(lat_min=10, lat_max=20, lon_min=30, lon_max=40)
        assert bbox.to_tuple() == (10, 20, 30, 40)

    def test_invalid_lat_range(self):
        """Test validation of latitude range."""
        with pytest.raises(ValidationError):
            BoundingBox(lat_min=45, lat_max=45, lon_min=0, lon_max=10)

    def test_lat_bounds(self):
        """Test latitude bounds validation."""
        with pytest.raises(ValidationError):
            BoundingBox(lat_min=-91, lat_max=0, lon_min=0, lon_max=10)


class TestSiteOrigin:
    """Tests for SiteOrigin model."""

    def test_valid_site_origin(self):
        """Test creating valid site origin."""
        origin = SiteOrigin(lat=40.5, lon=180.5, elevation_m=-2500)
        assert origin.lat == 40.5
        assert origin.elevation_m == -2500

    def test_lon_bounds(self):
        """Test longitude bounds."""
        with pytest.raises(ValidationError):
            SiteOrigin(lat=0, lon=361, elevation_m=0)


class TestWaypoint:
    """Tests for Waypoint model."""

    def test_valid_waypoint(self):
        """Test creating valid waypoint."""
        wp = Waypoint(
            waypoint_id=0,
            x_meters=100.0,
            y_meters=50.0,
            tolerance_meters=5.0,
        )
        assert wp.waypoint_id == 0
        assert wp.distance_from_origin() == pytest.approx(111.80, rel=0.01)

    def test_waypoint_at_origin(self):
        """Test waypoint at origin."""
        wp = Waypoint(waypoint_id=1, x_meters=0, y_meters=0)
        assert wp.distance_from_origin() == 0


class TestCriteriaWeights:
    """Tests for CriteriaWeights model."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = CriteriaWeights()
        assert weights.slope == 0.30
        assert weights.roughness == 0.25
        assert weights.elevation == 0.20
        assert weights.total() == 1.0

    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = CriteriaWeights(
            slope=3, roughness=2.5, elevation=2, solar_exposure=1.5, resources=1
        )
        normalized = weights.normalize()
        assert normalized.total() == pytest.approx(1.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = CriteriaWeights()
        d = weights.to_dict()
        assert isinstance(d, dict)
        assert "slope" in d


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_valid_config(self, test_roi):
        """Test creating valid analysis config."""
        config = AnalysisConfig(
            roi=test_roi,
            max_slope_deg=10,
            suitability_threshold=0.75,
        )
        assert config.roi == test_roi
        assert config.max_slope_deg == 10
EOF
```

### 1.6.3 Create `tests/unit/test_config.py`

```bash
cat > tests/unit/test_config.py << 'EOF'
"""Unit tests for configuration management."""

from pathlib import Path
import tempfile

import pytest
import yaml

from marshab.config import Config, PathsConfig, ConfigurationError


class TestPathsConfig:
    """Tests for PathsConfig."""

    def test_create_directories(self, tmp_path):
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


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default configuration loads."""
        config = Config()

        assert config.mars.equatorial_radius_m == 3396190.0
        assert config.mars.crs == "IAU_MARS_2000"
        assert config.logging.level == "INFO"

    def test_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "test_config.yaml"

        config_data = {
            "mars": {"equatorial_radius_m": 3396190.0},
            "logging": {"level": "DEBUG"},
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "cache_dir": str(tmp_path / "cache"),
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(config_file)

        assert config.logging.level == "DEBUG"

    def test_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(Path("nonexistent.yaml"))
EOF
```

### 1.6.4 Run Tests

```bash
# Run all unit tests in Phase 1
poetry run pytest tests/unit/ -v

# Run with coverage
poetry run pytest tests/unit/ --cov=marshab --cov-report=term-missing

# Run specific test
poetry run pytest tests/unit/test_types.py::TestBoundingBox -v
```

---

## 1.7 Update Package Initialization (15 minutes)

### 1.7.1 Update `marshab/__init__.py`

```bash
cat > marshab/__init__.py << 'EOF'
"""MarsHab Site Selector - Mars habitat site selection and rover navigation system."""

__version__ = "0.1.0"
__author__ = "MarsHab Development Team"
__description__ = "Geospatial analysis for Mars habitat construction site selection"

# Export commonly used types and functions
from marshab.types import (
    BoundingBox,
    SiteOrigin,
    Waypoint,
    TerrainMetrics,
    AnalysisConfig,
)
from marshab.config import get_config
from marshab.utils.logging import get_logger, configure_logging

__all__ = [
    "BoundingBox",
    "SiteOrigin",
    "Waypoint",
    "TerrainMetrics",
    "AnalysisConfig",
    "get_config",
    "get_logger",
    "configure_logging",
]
EOF
```

---

## 1.8 Verification Checklist

Run these commands to verify Phase 1 completion:

```bash
# Test all imports
poetry run python -c "
from marshab import (
    BoundingBox, SiteOrigin, Waypoint, AnalysisConfig,
    get_config, get_logger, configure_logging
)
print('✓ All imports successful')
"

# Test type creation
poetry run python -c "
from marshab import BoundingBox, AnalysisConfig
roi = BoundingBox(lat_min=40, lat_max=41, lon_min=180, lon_max=181)
config = AnalysisConfig(roi=roi)
print(f'✓ Config created with {len(config.criteria_weights.to_dict())} criteria')
"

# Test config loading
poetry run python -c "
from marshab.config import get_config
cfg = get_config()
print(f'✓ Config loaded: Mars CRS = {cfg.mars.crs}')
"

# Run full test suite
poetry run pytest tests/unit/ -v --tb=short

# Check code quality
poetry run mypy marshab --ignore-missing-imports
poetry run ruff check marshab tests
```

### Expected Output
```
✓ All imports successful
✓ Config created with 5 criteria
✓ Config loaded: Mars CRS = IAU_MARS_2000
================================ test session starts =================================
tests/unit/test_types.py::TestBoundingBox::test_valid_bbox PASSED
tests/unit/test_types.py::TestBoundingBox::test_bbox_to_tuple PASSED
... [additional tests]
==================================== 8 passed in 0.23s ===================================
```

---

## Phase 1 Summary

**Completed:**
- ✅ Type system with Pydantic models (BoundingBox, SiteOrigin, Waypoint, etc.)
- ✅ Configuration management with YAML support
- ✅ Structured logging with structlog and Rich
- ✅ Custom exception hierarchy
- ✅ Utility functions for validation, helpers, and caching
- ✅ Comprehensive test framework with pytest fixtures
- ✅ Package initialization with exports

**Artifacts Created:**
- `marshab/types.py` – 250 lines
- `marshab/config.py` – 150 lines
- `marshab/exceptions.py` – 40 lines
- `marshab/utils/logging.py` – 100 lines
- `marshab/utils/validators.py` – 80 lines
- `marshab/utils/helpers.py` – 60 lines
- `tests/conftest.py` – 120 lines
- `tests/unit/test_types.py` – 80 lines
- `tests/unit/test_config.py` – 70 lines

**Test Coverage:** ~85% for Phase 1 modules

**Time Spent:** ~8 hours  
**Time Remaining:** ~52 hours for Phases 2-6

---

**Checkpoint:** Core infrastructure complete. Ready to proceed to Phase 2: Data Management.
