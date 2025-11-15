# MarsHab Site Selector - Architecture Document

## Executive Summary

**MarsHab Site Selector** is a production-grade geospatial analysis system for identifying optimal Mars habitat construction sites and generating autonomous rover navigation waypoints. The system processes multi-resolution Mars terrain data, applies multi-criteria decision making (MCDM) algorithms, and outputs actionable navigation commands in Mars coordinate systems.

**Key Technical Decisions:**
- Python 3.11+ for modern type hints and performance
- Modular architecture with clear separation of concerns
- Docker containerization for reproducible environments
- SPICE toolkit integration for coordinate transformations
- CLI-first interface using Typer for automation
- Comprehensive test coverage with pytest
- Structured logging with structlog for production observability

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer (Typer)                        │
│  Commands: download | analyze | navigate | visualize | pipeline  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                      Core Service Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ Data Manager │  │   Analysis   │  │  Navigation Engine  │  │
│  │              │  │   Pipeline   │  │                     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                    Processing Modules                            │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ DEM Loader  │ │   Terrain    │ │  Coordinate Transform   │ │
│  │   (GDAL)    │ │  Analytics   │ │      (SPICE)            │ │
│  └─────────────┘ └──────────────┘ └──────────────────────────┘ │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │  MCDM/TOPSIS│ │ Pathfinding  │ │   Visualization         │ │
│  │             │ │   (A*/D*)    │ │   (Plotly/Matplotlib)   │ │
│  └─────────────┘ └──────────────┘ └──────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                      Data Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Raw DEMs    │  │  Processed   │  │   Output Products   │  │
│  │   (GeoTIFF)  │  │   Rasters    │  │  (GeoJSON, CSV)     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. CLI Layer
- **Purpose**: User interface and automation entry points
- **Technology**: Typer (Click-based)
- **Responsibilities**:
  - Command parsing and validation
  - Progress indication for long-running operations
  - Error handling and user feedback
  - Configuration loading

#### 2. Core Service Layer
- **Data Manager**: Handles all data I/O operations
  - Download Mars DEM datasets
  - Cache management
  - Data validation
  - Format conversion

- **Analysis Pipeline**: Orchestrates geospatial analysis
  - ROI extraction
  - Terrain derivative calculation
  - Multi-criteria evaluation
  - Site ranking

- **Navigation Engine**: Generates rover navigation commands
  - Coordinate frame transformations
  - Waypoint generation
  - Path planning and optimization
  - Obstacle avoidance

#### 3. Processing Modules
- **DEM Loader**: GDAL-based raster operations
- **Terrain Analytics**: Slope, aspect, roughness, TRI calculations
- **Coordinate Transform**: SPICE toolkit integration for Mars frames
- **MCDM/TOPSIS**: Multi-criteria decision making algorithms
- **Pathfinding**: A* and D*-Lite implementations
- **Visualization**: Interactive 3D terrain and path visualization

#### 4. Data Layer
- **Raw Data**: Original Mars DEMs (MOLA, HiRISE, CTX)
- **Processed Data**: Derived products (slope, suitability maps)
- **Output Products**: Final deliverables (GeoJSON, CSV, HTML)

---

## Data Flow Architecture

### Primary Pipeline Flow

```
1. Data Acquisition
   ↓
   Download MOLA/HiRISE DEMs → Validate CRS → Cache locally

2. Region of Interest Definition
   ↓
   User specifies bounding box → Clip DEM → Validate resolution

3. Terrain Analysis
   ↓
   Calculate slope → Calculate roughness → Calculate aspect
   ↓
   Generate cost surfaces → Combine metrics

4. Site Suitability Analysis
   ↓
   Normalize criteria → Apply weights (MCDM) → Generate score raster
   ↓
   Threshold and label → Extract polygons → Rank sites

5. Coordinate Transformation
   ↓
   IAU_MARS global coords → SPICE transformation → SITE frame local coords

6. Navigation Planning
   ↓
   Start/goal definition → Pathfinding (A*) → Waypoint generation
   ↓
   Export rover commands

7. Visualization & Reporting
   ↓
   Generate 3D terrain → Overlay paths → Export HTML/GeoJSON
```

---

## Technology Stack

### Core Dependencies

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| **Geospatial** | GDAL | 3.8+ | Raster I/O and projection |
| | Rasterio | 1.3+ | Pythonic GDAL wrapper |
| | GeoPandas | 0.14+ | Vector operations |
| | Shapely | 2.0+ | Geometric operations |
| | Fiona | 1.9+ | Vector I/O |
| **Planetary** | SpiceyPy | 6.0+ | SPICE toolkit wrapper |
| **Scientific** | NumPy | 1.26+ | Array operations |
| | SciPy | 1.11+ | Scientific computing |
| | pandas | 2.1+ | Data manipulation |
| **Visualization** | Plotly | 5.18+ | Interactive 3D plots |
| | Matplotlib | 3.8+ | Static visualizations |
| **CLI** | Typer | 0.9+ | CLI framework |
| | Rich | 13.7+ | Terminal formatting |
| **Logging** | structlog | 24.1+ | Structured logging |
| **Testing** | pytest | 7.4+ | Test framework |
| | pytest-cov | 4.1+ | Coverage reporting |
| **Development** | ruff | 0.1+ | Linting/formatting |
| | mypy | 1.7+ | Type checking |

### Development Tools

- **Package Manager**: Poetry (dependency management)
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Documentation**: MkDocs + mkdocs-material
- **Pre-commit**: Black, isort, ruff, mypy

---

## Project Structure

```
marshab-site-selector/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # CI/CD pipeline
│       └── docs.yml                  # Documentation deployment
├── data/                             # Git-ignored data directory
│   ├── raw/                          # Original Mars DEMs
│   ├── processed/                    # Derived rasters
│   ├── cache/                        # Downloaded file cache
│   └── output/                       # Final products
├── docs/                             # MkDocs documentation
│   ├── index.md
│   ├── user-guide.md
│   └── api-reference.md
├── tests/
│   ├── conftest.py                   # Pytest fixtures
│   ├── unit/
│   │   ├── test_terrain.py
│   │   ├── test_coordinates.py
│   │   └── test_pathfinding.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_cli.py
│   └── data/                         # Test fixtures
│       └── sample_dem.tif
├── marshab/                          # Main package
│   ├── __init__.py
│   ├── __main__.py                   # Entry point: python -m marshab
│   ├── cli.py                        # Typer CLI commands
│   ├── config.py                     # Configuration management
│   ├── exceptions.py                 # Custom exceptions
│   ├── types.py                      # Type definitions
│   ├── core/                         # Core business logic
│   │   ├── __init__.py
│   │   ├── data_manager.py           # Data acquisition and caching
│   │   ├── analysis_pipeline.py      # Analysis orchestration
│   │   └── navigation_engine.py      # Navigation planning
│   ├── processing/                   # Processing modules
│   │   ├── __init__.py
│   │   ├── dem_loader.py             # Raster I/O
│   │   ├── terrain.py                # Terrain analytics
│   │   ├── coordinates.py            # Coordinate transformations
│   │   ├── mcdm.py                   # Multi-criteria decision making
│   │   ├── pathfinding.py            # A* and D*-Lite
│   │   └── visualization.py          # Plotting functions
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── logging.py                # Structured logging setup
│       ├── validators.py             # Input validation
│       └── helpers.py                # Common helpers
├── scripts/                          # Standalone scripts
│   ├── download_dem.sh               # Download MOLA data
│   └── setup_spice_kernels.sh        # Download SPICE kernels
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                    # Poetry dependencies
├── README.md
└── mkdocs.yml                        # Documentation config
```

---

## Configuration Management

### Configuration File Format

**`marshab_config.yaml`**:
```yaml
# Mars Coordinate System Parameters
mars:
  equatorial_radius_m: 3396190.0
  polar_radius_m: 3376200.0
  crs: "IAU_MARS_2000"
  datum: "D_Mars_2000"

# Data Sources
data_sources:
  mola_global:
    url: "https://astrogeology.usgs.gov/..."
    resolution_m: 463
  hirise:
    base_url: "https://s3.amazonaws.com/..."
    resolution_m: 1.0

# Analysis Parameters
analysis:
  roi:
    lat_min: 35.0
    lat_max: 45.0
    lon_min: 180.0
    lon_max: 200.0
  criteria_weights:
    slope: 0.30
    roughness: 0.25
    elevation: 0.20
    solar_exposure: 0.15
    resources: 0.10
  thresholds:
    max_slope_deg: 5.0
    max_roughness: 0.5
    min_site_area_km2: 0.5

# Navigation Parameters
navigation:
  site_frame_origin:
    lat: 40.0
    lon: 190.0
    elevation_m: -2500.0
  waypoint_spacing_m: 50.0
  tolerance_m: 5.0
  max_slope_traversable_deg: 25.0

# Logging
logging:
  level: "INFO"
  format: "json"
  file: "logs/marshab.log"

# Paths
paths:
  data_dir: "data"
  cache_dir: "data/cache"
  output_dir: "data/output"
  spice_kernels: "/usr/local/share/spice"
```

### Environment Variables

```bash
# Required
MARSHAB_CONFIG_PATH=/path/to/marshab_config.yaml

# Optional
MARSHAB_LOG_LEVEL=DEBUG
MARSHAB_DATA_DIR=/data/mars
MARSHAB_CACHE_ENABLED=true
MARSHAB_PARALLEL_WORKERS=4
```

---

## API Design

### Core Services API

#### DataManager
```python
class DataManager:
    """Manages Mars DEM data acquisition and caching."""
    
    def download_dem(
        self, 
        dataset: Literal["mola", "hirise", "ctx"],
        roi: BoundingBox,
        resolution_m: float | None = None
    ) -> Path:
        """Download DEM covering ROI with specified resolution."""
    
    def load_dem(self, path: Path) -> xr.DataArray:
        """Load DEM into xarray DataArray with coordinate info."""
    
    def clip_to_roi(
        self, 
        dem: xr.DataArray, 
        roi: BoundingBox
    ) -> xr.DataArray:
        """Clip DEM raster to region of interest."""
```

#### AnalysisPipeline
```python
class AnalysisPipeline:
    """Orchestrates geospatial analysis workflow."""
    
    def calculate_terrain_metrics(
        self, 
        dem: xr.DataArray
    ) -> TerrainMetrics:
        """Calculate slope, aspect, roughness, TRI."""
    
    def evaluate_suitability(
        self,
        metrics: TerrainMetrics,
        weights: dict[str, float]
    ) -> xr.DataArray:
        """Apply MCDM to generate suitability score raster."""
    
    def extract_sites(
        self,
        suitability: xr.DataArray,
        threshold: float = 0.7
    ) -> gpd.GeoDataFrame:
        """Extract and rank candidate construction sites."""
```

#### NavigationEngine
```python
class NavigationEngine:
    """Generates rover navigation commands."""
    
    def transform_to_site_frame(
        self,
        lat: float,
        lon: float,
        elevation: float,
        site_origin: SiteOrigin
    ) -> tuple[float, float, float]:
        """Transform IAU_MARS coordinates to SITE frame."""
    
    def plan_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        cost_map: np.ndarray
    ) -> list[tuple[float, float]]:
        """Plan optimal path using A* pathfinding."""
    
    def generate_waypoints(
        self,
        path: list[tuple[float, float]],
        spacing_m: float = 50.0
    ) -> list[Waypoint]:
        """Generate rover waypoints from path."""
```

---

## Security & Error Handling

### Error Handling Strategy

1. **Custom Exception Hierarchy**
```python
class MarsHabError(Exception):
    """Base exception for all MarsHab errors."""

class DataError(MarsHabError):
    """Data acquisition or validation errors."""

class AnalysisError(MarsHabError):
    """Analysis pipeline errors."""

class CoordinateError(MarsHabError):
    """Coordinate transformation errors."""

class NavigationError(MarsHabError):
    """Path planning and navigation errors."""
```

2. **Validation at Boundaries**
- All user inputs validated with Pydantic models
- DEM CRS validated against IAU_MARS standard
- ROI bounds checked against Mars valid ranges
- Waypoint paths validated for traversability

3. **Graceful Degradation**
- If HiRISE unavailable, fallback to MOLA
- If SPICE kernels missing, use simplified transforms
- If pathfinding fails, return direct line with warnings

### Logging Strategy

- **Development**: Human-readable console logs (colored)
- **Production**: Structured JSON logs to file + stdout
- **Log Levels**:
  - DEBUG: Detailed diagnostic info
  - INFO: Major pipeline steps
  - WARNING: Potential issues (low quality data)
  - ERROR: Recoverable errors
  - CRITICAL: Unrecoverable errors

---

## Performance Considerations

### Optimization Strategies

1. **Lazy Loading**: Use dask-backed xarray for large DEMs
2. **Parallel Processing**: Process terrain metrics in parallel (NumPy/Numba)
3. **Caching**: Cache downloaded DEMs and processed products
4. **Chunked Processing**: Process large rasters in tiles
5. **Vectorization**: NumPy vectorized operations over loops

### Expected Performance

| Operation | Input Size | Time | Memory |
|-----------|------------|------|--------|
| DEM Download (MOLA) | 1° × 1° | ~30s | 100MB |
| Terrain Analysis | 2000×2000 px | ~5s | 500MB |
| MCDM Evaluation | 2000×2000 px | ~2s | 200MB |
| Pathfinding (A*) | 10km path | ~10s | 50MB |
| Full Pipeline | Regional analysis | ~5min | 2GB |

---

## Testing Strategy

### Test Coverage Targets
- **Unit Tests**: 80%+ coverage
- **Integration Tests**: All major workflows
- **End-to-End Tests**: Full pipeline execution

### Test Fixtures
- Small synthetic DEMs for fast unit tests
- Real Mars DEM samples for integration tests
- Mocked SPICE kernels for coordinate tests

### Continuous Integration
- Run tests on every commit (GitHub Actions)
- Coverage reporting to Codecov
- Type checking with mypy
- Linting with ruff

---

## Deployment

### Docker Deployment

**Single-command execution**:
```bash
docker-compose run marshab pipeline \
  --roi "35,45,180,200" \
  --output data/output/results
```

**Production deployment**:
- Mount data volumes for persistent storage
- Set environment variables for configuration
- Use Docker secrets for credentials

### Native Installation

```bash
# Install with pip
pip install marshab-site-selector

# Run CLI
marshab pipeline --roi "35,45,180,200"
```

---

## Future Enhancements

### Phase 2 Features
- Real-time rover telemetry integration
- Multi-agent rover coordination
- Machine learning for terrain classification
- Integration with Mars Trek API
- Web-based visualization dashboard

### Phase 3 Features
- Support for lunar coordinate systems
- Autonomous construction equipment simulation
- ISRU resource optimization
- Habitat placement optimization
- Radiation exposure mapping

---

## Appendix: Coordinate System Reference

### IAU 2000 Mars Reference Frame

- **Reference Ellipsoid**: Oblate ellipsoid
  - Equatorial radius: 3,396.19 km
  - Polar radius: 3,376.2 km
- **Prime Meridian**: Airy-0 crater
- **Latitude Convention**: Planetocentric
- **Longitude Convention**: Positive East (0-360°)
- **Vertical Datum**: Areoid (equipotential surface)

### Rover Coordinate Frames

1. **SITE Frame** (Mars-fixed)
   - Origin: Declared location on Mars surface
   - +X: North, +Y: East, +Z: Down

2. **RNAV Frame** (Rover-fixed)
   - Origin: Rover center
   - +X: Forward, +Y: Right, +Z: Down

3. **Local Level (LL) Frame**
   - Origin: Rover location
   - Axes: Mars-fixed (like SITE)

### SPICE Kernel Requirements

- **LSK**: Leap second kernel (naif0012.tls)
- **PCK**: Mars physical constants (pck00010.tpc)
- **FK**: Mars reference frames (mars_iau2000_v0.tf)
- **SPK**: Mars ephemeris (mar097.bsp)

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-15  
**Author**: MarsHab Development Team
