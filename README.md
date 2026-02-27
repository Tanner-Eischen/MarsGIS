# MarsHab Site Selector

MarsHab is a Mars terrain intelligence platform for planning where to build, how to traverse, and how to operate safely.
It combines real digital elevation models (DEMs), terrain analytics, multi-criteria scoring, and route planning into one API + web app.

## Live Deployment

- Frontend: https://marshab-web.onrender.com/
- Backend API: https://marshab-api.onrender.com
- API Docs (OpenAPI): https://marshab-api.onrender.com/docs

## What It Is

MarsHab is a geospatial decision system built for Mars mission planning workflows:

- Site selection for habitat construction
- Terrain-aware rover routing
- Elevation-driven map products (hillshade, slope, roughness, TRI)
- Solar potential analysis and mission support overlays

## What Problem It Solves

Mars mission planning usually requires multiple disconnected tools. MarsHab unifies them so teams can:

- Identify safer build zones faster
- Avoid manually stitching DEM analysis and route planning
- Use the same data stack from map view to numeric elevation probe
- Expose planning workflows through a production API for automation

## How It Works

1. Load DEM windows from configured datasets (MOLA, MOLA 200m, HiRISE, CTX).
2. Compute terrain derivatives (slope, aspect, roughness, TRI, hillshade).
3. Score candidate regions using weighted criteria.
4. Generate routes and waypoints to selected targets.
5. Serve map tiles, overlays, 3D terrain payloads, and analysis artifacts through REST endpoints.

## Main Features

- DEM ingestion with cache-aware window extraction
- Global basemap + overlay tile endpoints for smooth pan/zoom
- Terrain projection overlays: `elevation`, `hillshade`, `slope`, `aspect`, `roughness`, `tri`, `solar`, `dust`
- Elevation sampling endpoint for exact numeric probe values at clicked coordinates
- 3D terrain payload endpoint for rotatable mesh visualization
- Site scoring and decision-brief generation
- Single-route and multi-route rover planning
- Solar analysis endpoint and map overlays
- Project CRUD endpoints for saved planning runs

## Architecture

- `webui/`: React + Leaflet + Plotly frontend
- `marshab/web/`: FastAPI routes and API composition
- `marshab/core/`: DEM loading, raster services, caching, analysis orchestration
- `marshab/processing/`: Terrain metrics, pathfinding, coordinate transforms
- `data/`: local cache, outputs, and generated artifacts

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- Poetry
- Node.js 18+

### Backend

```bash
poetry install
poetry run python -m marshab.web.server
```

### Frontend

```bash
cd webui
npm install
npm run dev
```

Open `http://localhost:4000` for local UI development.

## Deployment (Render)

### Frontend

Deploy `webui` as a static site on Render and point it to the API service.

Required frontend env var:

- `VITE_API_URL=https://marshab-api.onrender.com/api/v1`

### Backend

Deploy API service with `render.yaml` / Docker.

Recommended backend env vars:

- `MARSHAB_CORS_ORIGINS=https://marshab-web.onrender.com`
- `MARSHAB_MOLA_COG_PATH=/app/data/cache/mola_global_463m.cog.tif`
- `MARSHAB_MOLA_COG_URL=<direct-binary-url-to-cog>`
- `MARSHAB_MOLA_COG_BOOTSTRAP_MODE=blocking`
- `MARSHAB_MOLA_COG_REQUIRED=true`
- `MARSHAB_PREWARM_COG=true`
- `MARSHAB_WEB_HOST=0.0.0.0`
- `MARSHAB_WEB_RELOAD=false`

Notes:

- Use a direct binary URL for COG downloads (not a release HTML page).
- Persist DEMs on the mounted data disk path so files survive restarts.

## Key API Endpoints

Base URL: `/api/v1`

### Health and Status

- `GET /health/live` - liveness probe
- `GET /health/ready` - readiness check
- `GET /status` - service and cache status
- `GET /demo-mode` - current demo-mode flag
- `POST /demo-mode` - update demo-mode flag

### Analysis

- `POST /analyze` - run primary analysis pipeline
- `POST /analysis/site-scores` - compute candidate site scores
- `POST /analysis/decision-brief` - generate ranked decision brief
- `POST /analysis/route-plan` - route planning analysis payload
- `POST /analysis/route-cost` - route cost computation

### Navigation

- `POST /navigation/plan-route` - single route generation
- `POST /navigation/plan-routes` - multi-strategy route generation
- `GET /navigation/waypoints-geojson` - route output as GeoJSON

### Visualization

- `GET /visualization/elevation-at` - sample elevation at lat/lon
- `GET /visualization/terrain-3d` - 3D terrain scene payload
- `GET /visualization/dem-window.tif` - float32 DEM GeoTIFF window
- `GET /visualization/dem-image` - DEM preview image
- `GET /visualization/overlay` - generated overlay image
- `GET /visualization/tiles/basemap/global/{z}/{x}/{y}.png` - global basemap tiles
- `GET /visualization/tiles/basemap/{dataset}/{z}/{x}/{y}.png` - DEM-backed basemap tiles
- `GET /visualization/tiles/overlay/{overlay_type}/{dataset}/{z}/{x}/{y}.png` - overlay tiles
- `GET /visualization/sites-geojson` - site features
- `GET /visualization/waypoints-geojson` - waypoint features
- `GET /visualization/dataset-coverage` - dataset availability/fallback metadata

### Solar and Mission

- `POST /solar/analyze` - solar irradiance and mission-energy analysis
- `GET /solar/health` - solar service health
- `POST /mission/landing-scenario` - landing scenario evaluation
- `POST /mission/rover-traverse` - traverse scenario evaluation

### Data and Export

- `POST /download` - dataset fetch/caching workflows
- `POST /download/prewarm/mola-tiles` - prewarm MOLA tiles
- `POST /download/prewarm/examples` - prewarm examples
- `POST /export/suitability-geotiff` - export suitability raster
- `POST /export/report` - export analysis report

### AI and Projects

- `POST /ai-query` - AI-assisted query endpoint
- `GET /ai-query/examples` - AI query examples
- `GET /ai-query/capabilities` - AI capabilities metadata
- `GET /projects` - list projects
- `POST /projects` - create project
- `GET /projects/{project_id}` - get project
- `DELETE /projects/{project_id}` - delete project

### WebSocket

- `WS /ws/progress/{task_id}` - real-time long-running task progress stream

## CLI Examples

```bash
# Full pipeline
poetry run marshab pipeline --roi "18.0,18.6,77.0,77.8" --output data/output

# Download DEM window sources
poetry run marshab download mola --roi "18.0,18.6,77.0,77.8"

# Terrain/site analysis only
poetry run marshab analyze --roi "18.0,18.6,77.0,77.8" --dataset mola
```

## Development

```bash
# Tests
poetry run pytest -v

# Lint
poetry run ruff check marshab tests

# Format
poetry run ruff format marshab tests
```

## License

MIT
