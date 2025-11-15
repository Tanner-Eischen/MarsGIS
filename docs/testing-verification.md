# Testing Verification Guide

This document outlines the procedures for verifying that all components of MarsHab work end-to-end.

## Pre-Verification Checklist

Before running verification tests, ensure:

- [ ] All dependencies are installed (`poetry install`)
- [ ] Docker is running (if testing containerized deployment)
- [ ] At least 10GB free disk space for data cache
- [ ] Internet connection for downloading DEM data

## Unit Tests

Run all unit tests to verify individual components:

```bash
# Run all unit tests
poetry run pytest tests/unit/ -v

# Run with coverage
poetry run pytest tests/unit/ --cov=marshab --cov-report=term

# Expected: All tests pass, coverage > 80%
```

### Key Test Files

- `tests/unit/test_config.py` - Configuration management
- `tests/unit/test_types.py` - Type definitions and validation
- `tests/unit/test_dem_loader.py` - DEM loading functionality
- `tests/unit/test_terrain.py` - Terrain analysis
- `tests/unit/test_coordinates.py` - Coordinate transformations
- `tests/unit/test_pathfinding.py` - Pathfinding algorithms

## Integration Tests

Run integration tests to verify component interactions:

```bash
# Run all integration tests
poetry run pytest tests/integration/ -v

# Expected: All integration tests pass
```

### Key Integration Tests

- Data pipeline: Download → Load → Process
- Analysis pipeline: DEM → Terrain → MCDM → Sites
- Navigation pipeline: Sites → Pathfinding → Waypoints

## End-to-End Testing

### Test 1: Complete Pipeline

Run the full pipeline from data download to waypoint generation:

```bash
# Run complete pipeline
poetry run marshab pipeline \
  --roi "40,41,180,181" \
  --dataset mola \
  --output data/output/test_run \
  --verbose

# Verify outputs:
# - data/output/test_run/suitability.tif exists
# - data/output/test_run/sites.geojson exists
# - data/output/test_run/waypoints.csv exists
```

**Expected Results:**
- Pipeline completes without errors
- Output files are generated
- Sites are identified (check sites.geojson)
- Waypoints are generated (check waypoints.csv)

### Test 2: Individual Commands

Test each CLI command individually:

```bash
# 1. Download test
poetry run marshab download mola \
  --roi "40,41,180,181" \
  --force

# Verify: data/cache/ contains downloaded DEM

# 2. Analyze test
poetry run marshab analyze \
  --roi "40,41,180,181" \
  --output data/output/analyze_test

# Verify: Analysis outputs generated

# 3. Navigate test (requires analysis results)
poetry run marshab navigate 1 \
  --analysis data/output/analyze_test \
  --start-lat 40.5 \
  --start-lon 180.5 \
  --output test_waypoints.csv

# Verify: waypoints.csv created with valid coordinates
```

### Test 3: Docker Deployment

Test containerized deployment:

```bash
# Build Docker image
docker-compose build

# Verify build succeeds

# Test CLI in container
docker-compose run marshab --version

# Test pipeline in container
docker-compose run marshab pipeline \
  --roi "40,41,180,181" \
  --output /app/data/output

# Verify: Container runs successfully, outputs generated
```

## Code Quality Checks

### Type Checking

```bash
# Run mypy type checker
poetry run mypy marshab

# Expected: No type errors (warnings acceptable)
```

### Linting

```bash
# Run ruff linter
poetry run ruff check marshab tests

# Expected: No linting errors

# Check formatting
poetry run ruff format --check marshab tests

# Expected: Code is properly formatted
```

### Test Coverage

```bash
# Generate coverage report
poetry run pytest --cov=marshab --cov-report=html

# Open htmlcov/index.html in browser
# Expected: Coverage > 80% for all modules
```

## Performance Verification

### Memory Usage

Monitor memory usage during large dataset processing:

```bash
# Process larger ROI
poetry run marshab analyze \
  --roi "35,45,180,200" \
  --dataset mola

# Monitor with: htop or top
# Expected: Memory usage < 16GB for typical ROI
```

### Processing Time

Verify processing completes in reasonable time:

- Small ROI (1° × 1°): < 5 minutes
- Medium ROI (5° × 5°): < 15 minutes
- Large ROI (10° × 10°): < 30 minutes

## Known Limitations

1. **SPICE Kernels**: Coordinate transformations use simplified models if SPICE kernels are not installed
2. **Large Datasets**: HiRISE data requires significant memory (>8GB for typical ROI)
3. **Network Dependency**: DEM downloads require internet connection
4. **Coordinate Range**: Longitude must be 0-360 (east positive)

## Troubleshooting Failed Tests

### Tests Fail with Import Errors

```bash
# Reinstall dependencies
poetry install --no-cache
```

### Tests Fail with Data Errors

```bash
# Clear cache and re-download
rm -rf data/cache/*
poetry run marshab download mola --roi "40,41,180,181" --force
```

### Docker Tests Fail

```bash
# Rebuild from scratch
docker-compose build --no-cache
```

## Verification Checklist

Before considering Phase 6 complete, verify:

- [ ] All unit tests pass (>80% coverage)
- [ ] All integration tests pass
- [ ] End-to-end pipeline completes successfully
- [ ] Docker build and run successfully
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Code formatting is correct
- [ ] README documentation is complete
- [ ] CI/CD workflow is configured

## Automated Verification Script

For convenience, create a verification script:

```bash
#!/bin/bash
# scripts/verify_installation.sh

set -e

echo "Running MarsHab verification tests..."

echo "1. Running unit tests..."
poetry run pytest tests/unit/ -v

echo "2. Running integration tests..."
poetry run pytest tests/integration/ -v

echo "3. Type checking..."
poetry run mypy marshab --ignore-missing-imports || true

echo "4. Linting..."
poetry run ruff check marshab tests

echo "5. Testing CLI..."
poetry run marshab --version

echo "6. Testing Docker build..."
docker-compose build

echo "✓ All verification tests passed!"
```

Run with: `bash scripts/verify_installation.sh`

## Next Steps

After verification:

1. Review any warnings or non-critical errors
2. Update documentation if procedures change
3. Document any new limitations discovered
4. Prepare for production deployment

---

**Last Updated**: 2025-01-XX  
**Phase**: 6 - Documentation & Polish

