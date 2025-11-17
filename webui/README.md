# MarsHab Web UI

Modern web interface for MarsHab Mars Habitat Site Selection and Rover Navigation System.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The frontend runs on `http://localhost:4000` and proxies API requests to `http://localhost:5000`.

## Features

- **Dashboard**: System status and quick actions
- **Data Download**: Download Mars DEM data (MOLA, HiRISE, CTX)
- **Terrain Analysis**: Run site suitability analysis
- **Navigation Planning**: Generate rover waypoints with pathfinding strategies
- **Visualization**: Interactive 3D terrain and path visualization (coming soon)

