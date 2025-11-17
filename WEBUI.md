# MarsHab Web UI

Modern web interface for the MarsHab Mars Habitat Site Selection and Rover Navigation System.

## Overview

The web UI provides a user-friendly interface for all CLI functionality:

- **Dashboard**: System status and quick actions
- **Data Download**: Download Mars DEM data (MOLA, HiRISE, CTX) with ROI selection
- **Terrain Analysis**: Run site suitability analysis with configurable parameters
- **Navigation Planning**: Generate rover waypoints with pathfinding strategies (safest, balanced, direct)
- **Visualization**: Interactive 3D terrain and path visualization (coming soon)

## Architecture

### Backend (FastAPI)

- **Location**: `marshab/web/`
- **API Server**: FastAPI with async support
- **Endpoints**:
  - `GET /api/v1/status` - System status
  - `POST /api/v1/download` - Download DEM data
  - `POST /api/v1/analyze` - Run terrain analysis
  - `POST /api/v1/navigate` - Generate navigation waypoints
  - `GET /api/v1/visualization/*` - Export data files

### Frontend (React + TypeScript)

- **Location**: `webui/`
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: TanStack Query (React Query)
- **Routing**: React Router

## Quick Start

### Prerequisites

- Python 3.11+ with Poetry
- Node.js 18+ and npm

### Starting the Backend

```bash
# Using Poetry
poetry run python -m marshab.web.server

# Or using the script
./scripts/start_web.sh  # Linux/Mac
.\scripts\start_web.ps1  # Windows PowerShell
```

The API server will start on `http://localhost:5000`

### Starting the Frontend

```bash
cd webui
npm install
npm run dev
```

The frontend will start on `http://localhost:4000` and automatically proxy API requests to the backend.

## Development

### Backend Development

The FastAPI server supports auto-reload in development mode. API routes are organized in `marshab/web/routes/`:

- `status.py` - System status endpoints
- `download.py` - DEM download endpoints
- `analysis.py` - Terrain analysis endpoints
- `navigation.py` - Navigation planning endpoints
- `visualization.py` - Data export endpoints

### Frontend Development

The frontend uses Vite for fast HMR (Hot Module Replacement). Key directories:

- `src/pages/` - Page components
- `src/components/` - Reusable components
- `src/services/` - API client and services

### API Documentation

Once the backend is running, visit `http://localhost:5000/docs` for interactive API documentation (Swagger UI).

## Production Deployment

### Building the Frontend

```bash
cd webui
npm run build
```

The built files will be in `webui/dist/`. You can serve these with any static file server or configure FastAPI to serve them.

### Docker Deployment

The web UI can be containerized alongside the backend. Update `Dockerfile` to include:

1. Install Node.js dependencies
2. Build the frontend
3. Serve static files via FastAPI

## Features

### Current Implementation

- ✅ RESTful API with FastAPI
- ✅ React frontend with TypeScript
- ✅ Data download interface
- ✅ Terrain analysis interface
- ✅ Navigation planning interface
- ✅ System status dashboard
- ✅ Responsive design with Tailwind CSS

### Coming Soon

- 🔄 Real-time progress updates via WebSocket
- 🔄 Interactive map for ROI selection (Leaflet/Mapbox)
- 🔄 3D terrain visualization (Plotly.js)
- 🔄 Path visualization on terrain maps
- 🔄 Export functionality (CSV, GeoJSON, PNG)

## Troubleshooting

### Backend won't start

- Check that all dependencies are installed: `poetry install`
- Verify port 5000 is not in use
- Check logs for error messages

### Frontend can't connect to API

- Ensure backend is running on `http://localhost:5000`
- Check browser console for CORS errors
- Verify proxy configuration in `vite.config.ts`

### API returns 404

- Check that the endpoint path matches the route definition
- Verify the API server is running and accessible
- Check API documentation at `/docs` endpoint

