# Quick Start Guide - MarsHab

## ✅ Docker is Ready!

Your Docker image has been built successfully. Here's how to use it:

## Start the Application

### Option 1: Run Commands Directly

```powershell
# Show help
docker-compose run --rm marshab --help

# Run a complete pipeline
docker-compose run --rm marshab pipeline --roi "40,41,180,181" --output /app/data/output

# Download DEM data
docker-compose run --rm marshab download mola --roi "40,41,180,181"

# Analyze terrain
docker-compose run --rm marshab analyze --roi "40,41,180,181" --output /app/data/output
```

### Option 2: Interactive Development Shell

```powershell
# Start an interactive development container
docker-compose run --rm dev

# Inside the container, you can run:
marshab --help
marshab pipeline --roi "40,41,180,181"
python -m pytest
```

## Quick Test

Try this to verify everything works:

```powershell
docker-compose run --rm marshab --version
```

## Available Services

- **marshab**: Main application container (runs commands)
- **dev**: Development container with interactive shell

## Data Persistence

All data is stored in:
- `./data/cache/` - Downloaded DEM files
- `./data/output/` - Analysis results
- `./data/processed/` - Processed rasters

These directories are mounted as volumes, so data persists between container runs.

## Next Steps

1. Try a simple command: `docker-compose run --rm marshab --help`
2. Download test data: `docker-compose run --rm marshab download mola --roi "40,41,180,181"`
3. Run analysis: `docker-compose run --rm marshab analyze --roi "40,41,180,181"`

---

**Note**: The `--rm` flag automatically removes the container after it exits, keeping your system clean.




