# MarsHab Site Selector

A production-grade geospatial analysis system for identifying optimal Mars habitat construction sites and generating autonomous rover navigation waypoints. The system processes multi-resolution Mars terrain data, applies multi-criteria decision making (MCDM) algorithms, and outputs actionable navigation commands in Mars coordinate systems.

**Status:** Portfolio Week 3 Hardening

**Implementation Note:** Portfolio mode is scoped to three flagship flows: site selection, route planning, and decision brief generation.

## Features

- **Multi-Resolution DEM Processing**: Support for MOLA, HiRISE, and CTX Mars elevation datasets
- **Terrain Analysis**: Automated calculation of slope, aspect, roughness, and terrain ruggedness index (TRI)
- **Site Suitability Analysis**: Multi-criteria decision making (MCDM) with weighted sum and TOPSIS methods
- **Rover Navigation Planning**: A* pathfinding algorithm with coordinate transformations to Mars SITE frame
- **Solar Potential Analysis**: Calculate solar energy generation and mission cost savings from terrain data
- **3D Terrain Visualization**: Interactive 3D visualization with Plotly.js showing sites and routes
- **Cost Analysis**: Route cost breakdown and solar-powered mission cost savings calculations
- **Web UI**: Modern React-based interface for all features
- **Coordinate Transformations**: IAU_MARS to rover SITE frame transformations
- **CLI Interface**: Comprehensive command-line interface for automation and scripting
- **Docker Support**: Containerized deployment for reproducible environments

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Development](#development)
- [Known Limits](#known-limits)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- Docker and Docker Compose (optional, for containerized deployment)
- 16GB RAM recommended for large datasets
- 10GB disk space for data cache

### Option 1: Native Installation with Poetry (Recommended for Development)

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone <repository-url>
cd MarsGIS

# Install dependencies
poetry install

# Verify installation
poetry run marshab --version
```

### Option 2: Docker (Recommended for Production)

```bash
# Build the Docker image
docker-compose build

# Run CLI command in container
docker run --rm marshab:latest marshab --help

# Run API server in container
docker run --rm -p 5000:5000 marshab:latest python -m marshab.web.server
```

## Quick Start

### Web UI (Recommended for Interactive Use)

1. **Start Backend**: `poetry run python -m marshab.web.server`
2. **Start Frontend**: `cd webui && npm run dev`
3. **Open Browser**: http://localhost:4000
4. **Follow Demo**: See `DEMO_WORKFLOW.md` for complete step-by-step guide

### Deterministic Demo Mode

For repeatable portfolio outputs in synthetic fallback mode, set:

```bash
export MARSHAB_DEMO_SEED=42
```

This enforces deterministic synthetic terrain generation when real DEM data is unavailable.

### Install Real DEM (Jezero)

To use a real DEM tile instead of synthetic fallback for Jezero-area demos:

```bash
poetry run python scripts/setup_real_dem.py --force-download
```

This prepares `data/cache/mola_lat18_lon77.tif` for ROI ranges near
`lat 18.x, lon 77.x` (for example `18.25,18.45,77.25,77.45`).

### Run Portfolio Demo (Week 3)

Run the fixed three-flow portfolio walkthrough and write reproducible artifacts:

```bash
poetry run python scripts/run_portfolio_demo.py
```

Outputs are written to `data/output/portfolio_demo`:
- `sites_ranked.json`
- `sites_overlay.geojson`
- `route_waypoints.csv`
- `decision_brief.json`
- `demo_manifest.json`

For API-based decision brief generation (rules-based, deterministic):

```bash
curl -X POST http://localhost:5000/api/v1/analysis/decision-brief \
  -H "Content-Type: application/json" \
  -d '{"roi":{"lat_min":40.0,"lat_max":40.3,"lon_min":180.0,"lon_max":180.3},"dataset":"mola","threshold":0.5}'
```

### CLI (For Automation)

```bash
# Run a complete analysis pipeline
poetry run marshab pipeline \
  --roi "40,41,180,181" \
  --output data/output

# Or use Docker
docker-compose run marshab pipeline \
  --roi "40,41,180,181" \
  --output /app/data/output
```

This command will:
1. Download Mars DEM data for the specified region
2. Analyze terrain characteristics
3. Identify suitable construction sites
4. Generate navigation waypoints to the top-ranked site

### Demo Workflow

For a complete demonstration showcasing all features (3D visualization, solar analysis, cost savings), see **DEMO_WORKFLOW.md**.

Pre-load demo data:
```powershell
.\scripts\prepare_demo_data.ps1
```

## Usage

### CLI Commands

MarsHab provides several CLI commands for different workflows:

#### Download DEM Data

Download Mars DEM data for a specified region:

```bash
poetry run marshab download mola \
  --roi "40,41,180,181" \
  --force

# Available datasets: mola, hirise, ctx
# ROI format: "lat_min,lat_max,lon_min,lon_max"
```

**Options:**
- `dataset`: Dataset to download (`mola`, `hirise`, or `ctx`)
- `--roi`: Region of interest as `lat_min,lat_max,lon_min,lon_max`
- `--force`: Force re-download even if cached

#### Analyze Terrain

Analyze terrain and identify construction sites:

```bash
poetry run marshab analyze \
  --roi "40,41,180,181" \
  --dataset mola \
  --output data/output \
  --threshold 0.7
```

**Options:**
- `--roi`: Region of interest (required)
- `--dataset`: Dataset to use (`mola`, `hirise`, or `ctx`, default: `mola`)
- `--output`: Output directory (default: `data/output`)
- `--threshold`: Suitability threshold (0-1, default: 0.7)

**Output:**
- Suitability score raster
- Candidate site polygons (GeoJSON)
- Analysis summary report

#### Generate Navigation Waypoints

Generate rover navigation waypoints to a target site:

```bash
poetry run marshab navigate 1 \
  --analysis data/output \
  --start-lat 40.5 \
  --start-lon 180.5 \
  --output waypoints.csv
```

**Options:**
- `site_id`: Target site ID from analysis results
- `--analysis`: Analysis results directory
- `--start-lat`: Starting latitude (degrees)
- `--start-lon`: Starting longitude (degrees)
- `--output`: Waypoint output file (default: `waypoints.csv`)

**Output:**
- CSV file with waypoint coordinates in SITE frame
- Each waypoint includes: X (North), Y (East), tolerance

#### Complete Pipeline

Run the complete analysis and navigation pipeline:

```bash
poetry run marshab pipeline \
  --roi "40,41,180,181" \
  --dataset mola \
  --output data/output \
  --verbose
```

This combines all steps: download → analyze → navigate to top site.

### Global Options

All commands support these global options:

- `--version, -v`: Show version and exit
- `--verbose, -V`: Enable verbose logging
- `--config, -c`: Path to configuration file

### Example Workflows

#### Basic Site Selection

```bash
# 1. Download data
poetry run marshab download mola --roi "35,45,180,200"

# 2. Analyze terrain
poetry run marshab analyze --roi "35,45,180,200" --output results/

# 3. Review results in results/sites.geojson
```

#### Navigation Planning

```bash
# 1. Run analysis
poetry run marshab analyze --roi "40,41,180,181" --output nav_results/

# 2. Generate waypoints to site ID 1
poetry run marshab navigate 1 \
  --analysis nav_results/ \
  --start-lat 40.2 \
  --start-lon 180.2 \
  --output mission_waypoints.csv
```

## Configuration

MarsHab uses YAML configuration files. Configuration is loaded in this order:

1. `MARSHAB_CONFIG_PATH` environment variable
2. `./marshab_config.yaml` in current directory
3. `~/.config/marshab/config.yaml` in home directory
4. Default configuration with environment variable overrides

### Configuration File Format

Create `marshab_config.yaml`:

```yaml
# Mars Coordinate System Parameters
mars:
  equatorial_radius_m: 3396190.0
  polar_radius_m: 3376200.0
  crs: "IAU_MARS_2000"
  datum: "D_Mars_2000"

# Data Sources
data_sources:
  mola:
    url: "https://astrogeology.usgs.gov/cache/mars/viking/dem/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
    resolution_m: 463
  hirise:
    url: "https://s3.amazonaws.com/hirise-pds/PDS/..."
    resolution_m: 1.0

# Analysis Parameters
analysis:
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
  waypoint_spacing_m: 50.0
  tolerance_m: 5.0
  max_slope_traversable_deg: 25.0

# Logging
logging:
  level: "INFO"
  format: "console"
  file: null

# File System Paths
paths:
  data_dir: "data"
  cache_dir: "data/cache"
  output_dir: "data/output"
  spice_kernels: "/usr/local/share/spice"
```

### Environment Variables

You can override configuration with environment variables:

```bash
export MARSHAB_CONFIG_PATH=/path/to/config.yaml
export MARSHAB_LOG_LEVEL=DEBUG
export MARSHAB_DATA_DIR=/data/mars
export MARSHAB_CACHE_ENABLED=true
export MARSHAB_DEMO_SEED=42
```

For cloud deployment (Render/Fly), run the backend with:

```bash
python -m marshab.web.server
```

## Deployment

### Frontend (Vercel)

- Build command: `npm run build` (in `webui/`)
- Output directory: `webui/dist`
- Config file: `webui/vercel.json`
- Environment variables:
  - `VITE_API_URL=https://<your-backend>/api/v1`
  - `VITE_WS_URL=wss://<your-backend>`

### Backend (Render/Fly Docker)

- Container start command: `python -m marshab.web.server`
- Render blueprint: `render.yaml`
- `Dockerfile` entrypoint is command-flexible so the same image can run CLI and API commands.
- CORS for deployed frontend:
  - Set `MARSHAB_CORS_ORIGINS=https://<your-vercel-domain>`
- If free-tier limits are tight, run cloud demo in synthetic mode and document local real-DEM setup.

## API Reference

### Core Services

#### DataManager

Manages Mars DEM data acquisition, caching, and loading.

```python
from marshab.core.data_manager import DataManager
from marshab.types import BoundingBox

dm = DataManager()

# Download DEM
roi = BoundingBox(lat_min=40, lat_max=41, lon_min=180, lon_max=181)
dem_path = dm.download_dem("mola", roi)

# Load DEM
dem = dm.load_dem(dem_path)

# Get DEM for ROI (downloads if needed)
dem = dm.get_dem_for_roi(roi, dataset="mola", download=True)
```

#### AnalysisPipeline

Orchestrates geospatial analysis workflow.

```python
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.types import BoundingBox

pipeline = AnalysisPipeline()
roi = BoundingBox(lat_min=40, lat_max=41, lon_min=180, lon_max=181)

results = pipeline.run(roi, dataset="mola", threshold=0.7)
results.save("output/")
```

#### NavigationEngine

Generates rover navigation commands.

```python
from marshab.core.navigation_engine import NavigationEngine

engine = NavigationEngine()
waypoints = engine.plan_to_site(
    site_id=1,
    analysis_dir="output/",
    start_lat=40.5,
    start_lon=180.5
)
waypoints.to_csv("waypoints.csv")
```

### Processing Modules

#### TerrainAnalyzer

Calculate terrain metrics from DEMs.

```python
from marshab.processing.terrain import TerrainAnalyzer
import xarray as xr

analyzer = TerrainAnalyzer(cell_size_m=200.0)
metrics = analyzer.analyze(dem)  # Returns TerrainMetrics

# Access individual metrics
slope = metrics.slope
roughness = metrics.roughness
```

#### CoordinateTransformer

Transform between Mars coordinate frames.

```python
from marshab.processing.coordinates import CoordinateTransformer
from marshab.types import SiteOrigin

transformer = CoordinateTransformer()
site_origin = SiteOrigin(lat=40.0, lon=180.0, elevation_m=-2500.0)

# Transform to SITE frame
x, y, z = transformer.iau_mars_to_site_frame(
    lat_deg=40.1,
    lon_deg=180.1,
    elevation_m=-2500.0,
    site_origin=site_origin
)
```

#### AStarPathfinder

Path planning for rover navigation.

```python
from marshab.processing.pathfinding import AStarPathfinder
import numpy as np

cost_map = np.ones((100, 100))  # Traversability cost map
pathfinder = AStarPathfinder(cost_map, cell_size_m=1.0)

path = pathfinder.find_path(start=(0, 0), goal=(99, 99))
waypoints = pathfinder.find_path_with_waypoints(
    start=(0, 0),
    goal=(99, 99),
    max_waypoint_spacing=50
)
```

### Type Definitions

Key types from `marshab.types`:

- `BoundingBox`: Region of interest bounding box
- `SiteOrigin`: Rover SITE frame origin definition
- `Waypoint`: Navigation waypoint in SITE frame
- `TerrainMetrics`: Terrain analysis outputs
- `CriteriaWeights`: MCDM criteria weights

## Development

### Current Implementation Status

The following components are **stub implementations** that return empty results:

- **`AnalysisPipeline.run()`** (lines 87-113 in `marshab/core/analysis_pipeline.py`): Returns empty sites list
- **`NavigationEngine.plan_to_site()`** (lines 50-74 in `marshab/core/navigation_engine.py`): Returns empty waypoints DataFrame

**What works:**
- ✅ CLI interface and command parsing
- ✅ Data download and caching (MOLA DEM successfully cached)
- ✅ Configuration management
- ✅ Logging and error handling
- ✅ Docker containerization
- ✅ File I/O and output generation

**What needs implementation:**
- ⚠️ Terrain analysis (slope, roughness, TRI calculations)
- ⚠️ MCDM site selection algorithm
- ⚠️ Pathfinding (A* algorithm integration)
- ⚠️ Coordinate transformations (IAU_MARS to SITE frame)

The pipeline runs successfully end-to-end but returns empty results until these components are fully implemented.

### Project Structure

```
MarsGIS/
├── marshab/                  # Main package
│   ├── __init__.py
│   ├── cli.py               # CLI interface
│   ├── config.py            # Configuration management
│   ├── exceptions.py        # Custom exceptions
│   ├── types.py             # Type definitions
│   ├── core/                # Core services
│   │   ├── data_manager.py
│   │   ├── analysis_pipeline.py
│   │   └── navigation_engine.py
│   ├── processing/          # Processing modules
│   │   ├── dem_loader.py
│   │   ├── terrain.py
│   │   ├── coordinates.py
│   │   ├── mcdm.py
│   │   └── pathfinding.py
│   └── utils/               # Utilities
│       ├── logging.py
│       ├── validators.py
│       └── helpers.py
├── tests/                   # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── data/                    # Data directory (git-ignored)
│   ├── raw/                 # Raw DEMs
│   ├── processed/           # Processed rasters
│   ├── cache/               # Cached downloads
│   └── output/              # Analysis outputs
├── docs/                    # Documentation
├── scripts/                 # Helper scripts
├── .github/workflows/       # CI/CD workflows
├── docker-compose.yml       # Docker configuration
├── Dockerfile               # Docker image definition
├── pyproject.toml           # Poetry dependencies
└── README.md                # This file
```

### Running Tests

```bash
# Run all tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=marshab --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_terrain.py -v

# Run integration tests
poetry run pytest tests/integration/ -v
```

### Type Checking

```bash
# Run mypy type checker
poetry run mypy marshab

# Check specific file
poetry run mypy marshab/core/data_manager.py
```

### Linting and Formatting

```bash
# Check code style
poetry run ruff check marshab tests

# Auto-fix issues
poetry run ruff check marshab tests --fix

# Format code
poetry run ruff format marshab tests
```

### Docker Development

```bash
# Build image
docker-compose build

# Run development shell
docker-compose run dev

# Run tests in container
docker-compose run marshab pytest

# Run CLI in container
docker-compose run marshab pipeline --roi "40,41,180,181"
```

## Known Limits

- Real DEM setup currently ships one curated Jezero-area tile (`data/cache/mola_lat18_lon77.tif`).
- The Week 3 CI gate is intentionally smoke-focused while broader legacy lint/test debt is being reduced.
- Frontend production bundle is currently large and should be further split for faster initial load.

## Troubleshooting

### Common Issues

#### DEM Download Fails

**Problem**: `DataError: Failed to download DEM`

**Solutions**:
- Check internet connection
- Verify data source URLs in configuration
- Check disk space in cache directory
- Try with `--force` flag to re-download

#### Memory Errors

**Problem**: Out of memory when processing large DEMs

**Solutions**:
- Use lower resolution datasets (MOLA instead of HiRISE)
- Reduce ROI size
- Increase system RAM or use swap space
- Process in smaller chunks

#### Coordinate Transformation Errors

**Problem**: `CoordinateError: Failed to transform coordinates`

**Solutions**:
- Verify ROI bounds are valid (-90 to 90 lat, 0 to 360 lon)
- Check that site origin is within ROI
- Ensure SPICE kernels are installed (if using SPICE)

#### Pathfinding Returns None

**Problem**: No path found between start and goal

**Solutions**:
- Check that start and goal positions are not impassable (infinite cost)
- Verify cost map has valid passable routes
- Try different start/goal positions
- Check terrain constraints (max slope, roughness)

### Getting Help

- Check the [Architecture Document](mars-site-architecture.md) for system design details
- Review [Implementation Plan](phases2-6-detailed.md) for development phases
- Open an issue on GitHub for bugs or feature requests

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow code style** (ruff formatting, type hints)
4. **Update documentation** for new features
5. **Run tests** before submitting PR

### Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd MarsGIS

# Install dependencies
poetry install

# Install pre-commit hooks
pre-commit install

# Run tests
poetry run pytest
```

### Code Style

- Use type hints for all function signatures
- Follow PEP 8 style guide (enforced by ruff)
- Write docstrings for all public functions/classes
- Keep test coverage above 80%

## License

MIT License - see LICENSE file for details

## Authors

MarsHab Development Team

## Acknowledgments

- Mars elevation data from NASA/USGS
- SPICE toolkit for coordinate transformations
- Open source geospatial libraries (GDAL, Rasterio, GeoPandas)

---

**Version**: 0.1.0  
**Last Updated**: 2025-01-XX
