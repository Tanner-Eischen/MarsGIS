# Testing Guide

This guide explains how to run tests for the MarsHab application.

## Prerequisites

Make sure you have installed the development dependencies:

```bash
poetry install
```

This installs `pytest`, `pytest-cov`, and other testing tools.

## Running Tests

### Option 1: Run All Tests

```bash
# Using Poetry (recommended)
poetry run pytest

# Or if Poetry is in your PATH
pytest
```

### Option 2: Run Tests with Verbose Output

```bash
poetry run pytest -v
```

### Option 3: Run Specific Test Files

```bash
# Test analysis pipeline
poetry run pytest tests/unit/test_analysis_pipeline.py -v

# Test navigation engine
poetry run pytest tests/unit/test_navigation_engine.py -v

# Test terrain analysis
poetry run pytest tests/unit/test_terrain.py -v
```

### Option 4: Run Tests with Coverage

```bash
# Generate coverage report
poetry run pytest --cov=marshab --cov-report=html --cov-report=term

# View HTML report (opens in browser)
# Windows: start htmlcov/index.html
# Linux/Mac: open htmlcov/index.html
```

### Option 5: Run Specific Test Functions

```bash
# Run a specific test function
poetry run pytest tests/unit/test_analysis_pipeline.py::TestAnalysisPipeline::test_run_pipeline_basic -v
```

### Option 6: Run Tests in Docker

```bash
# Build and run tests in Docker
docker-compose run --rm marshab pytest

# Or with coverage
docker-compose run --rm marshab pytest --cov=marshab --cov-report=term
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_analysis_pipeline.py
│   ├── test_navigation_engine.py
│   ├── test_terrain.py
│   ├── test_mcdm.py
│   ├── test_pathfinding.py
│   ├── test_coordinates.py
│   ├── test_data_manager.py
│   └── ...
└── integration/            # Integration tests
    └── test_data_pipeline.py
```

## Common Test Commands

```bash
# Run all tests with verbose output
poetry run pytest -v

# Run tests and show print statements
poetry run pytest -v -s

# Run tests and stop on first failure
poetry run pytest -x

# Run tests matching a pattern
poetry run pytest -k "analysis" -v

# Run tests with detailed output
poetry run pytest -vv

# Run tests in parallel (if pytest-xdist is installed)
poetry run pytest -n auto
```

## Troubleshooting

### Poetry Not Found

If `poetry` is not recognized, add it to your PATH or use:

```bash
# Windows PowerShell
$env:Path += ";$env:APPDATA\Python\Scripts"
poetry run pytest

# Or use Python directly
python -m pytest
```

### Missing Dependencies

If tests fail due to missing dependencies:

```bash
poetry install --with dev
```

### Test Failures

If tests fail, check:
1. All dependencies are installed: `poetry install`
2. Test data files exist in `tests/data/`
3. Configuration is correct in `marshab_config.yaml`

## Continuous Integration

Tests are automatically run in CI/CD via GitHub Actions (`.github/workflows/ci.yml`) on:
- Every push to main branch
- Every pull request
- Includes: pytest, type checking (mypy), linting (ruff)




