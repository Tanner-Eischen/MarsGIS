# Starting MarsHab Locally

MarsHab is a CLI-based application. Here are the ways to run it locally:

## Option 1: Using Poetry (Native Installation)

### Quick Start

```powershell
# 1. Add Poetry to PATH (if not already done)
$env:Path += ";$env:APPDATA\Python\Scripts"

# 2. Install dependencies
poetry install

# 3. Run the application
poetry run marshab --help
```

### Available Commands

```powershell
# Show all available commands
poetry run marshab --help

# Download DEM data
poetry run marshab download mola --roi "40,41,180,181"

# Analyze terrain
poetry run marshab analyze --roi "40,41,180,181" --output data/output

# Run complete pipeline
poetry run marshab pipeline --roi "40,41,180,181" --output data/output
```

## Option 2: Using Docker (Recommended)

### Start Development Container

```powershell
# Build the Docker image (first time only)
docker-compose build

# Start development container with interactive shell
docker-compose run dev

# Inside the container, you can run:
marshab --help
marshab pipeline --roi "40,41,180,181"
```

### Run Commands Directly in Docker

```powershell
# Run a single command
docker-compose run marshab pipeline --roi "40,41,180,181" --output /app/data/output

# Run with help
docker-compose run marshab --help
```

## Option 3: Using Python Module

```powershell
# Activate Poetry shell
poetry shell

# Run as Python module
python -m marshab --help
```

## Quick Test

To verify everything is working:

```powershell
# Test with Poetry
poetry run marshab --version

# Or with Docker
docker-compose run marshab --version
```

## Troubleshooting

### Poetry not found
```powershell
$env:Path += ";$env:APPDATA\Python\Scripts"
```

### Docker not running
Make sure Docker Desktop is running on Windows.

### Dependencies not installed
```powershell
poetry install
```

## Next Steps

1. Try a simple command: `poetry run marshab --help`
2. Download test data: `poetry run marshab download mola --roi "40,41,180,181"`
3. Run analysis: `poetry run marshab analyze --roi "40,41,180,181"`




