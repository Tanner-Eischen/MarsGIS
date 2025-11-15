# Phase 0: Project Bootstrap

**Duration:** Day 1 – 4 hours  
**Goal:** Create fully functional development environment with all tooling, dependencies, and project scaffolding

---

## 0.1 Initialize Git Repository (15 minutes)

### Steps

```bash
# Create project root
mkdir marshab-site-selector
cd marshab-site-selector

# Initialize git
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Create initial commit hook
echo "# MarsHab Site Selector" > README.md
git add README.md
git commit -m "Initial commit"
```

### Expected Output
- `.git/` directory created
- `README.md` in root

---

## 0.2 Create Directory Structure (15 minutes)

### Steps

```bash
# Core package directories
mkdir -p marshab/core
mkdir -p marshab/processing
mkdir -p marshab/utils

# Testing directories
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/data

# Data directories (git-ignored)
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/cache
mkdir -p data/output

# Documentation
mkdir -p docs
mkdir -p scripts
mkdir -p .github/workflows

# Create __init__.py files to make directories importable
touch marshab/__init__.py
touch marshab/core/__init__.py
touch marshab/processing/__init__.py
touch marshab/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
```

### Expected Output
```
marshab-site-selector/
├── marshab/
│   ├── __init__.py
│   ├── core/
│   │   └── __init__.py
│   ├── processing/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── data/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   └── output/
├── docs/
├── scripts/
└── .github/
    └── workflows/
```

---

## 0.3 Install and Configure Poetry (45 minutes)

### Steps

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version

# Configure Poetry to create venv in project
poetry config virtualenvs.in-project true

# Initialize Poetry project
poetry init \
  --name marshab \
  --description "Mars Habitat Site Selection and Rover Navigation System" \
  --author "Your Name <your.email@example.com>" \
  --python "^3.11" \
  --no-interaction

# Add core geospatial dependencies
poetry add \
  rasterio@^1.3 \
  geopandas@^0.14 \
  shapely@^2.0 \
  fiona@^1.9 \
  pyproj@^3.6 \
  spiceypy@^6.0 \
  numpy@^1.26 \
  scipy@^1.11 \
  pandas@^2.1 \
  xarray@^2023.12

# Add visualization dependencies
poetry add \
  plotly@^5.18 \
  matplotlib@^3.8

# Add CLI and utility dependencies
poetry add \
  typer[all]@^0.9 \
  rich@^13.7 \
  pydantic@^2.5 \
  pydantic-settings@^2.1 \
  pyyaml@^6.0 \
  python-dotenv@^1.0

# Add logging dependency
poetry add structlog@^24.1

# Add development dependencies
poetry add --group dev \
  pytest@^7.4 \
  pytest-cov@^4.1 \
  pytest-mock@^3.12 \
  pytest-asyncio@^0.21

# Add linting and typing
poetry add --group dev \
  ruff@^0.1 \
  mypy@^1.7 \
  black@^23.12 \
  isort@^5.13

# Verify installation
poetry install
```

### Expected Output
```
Creating virtualenv marshab in /path/to/marshab-site-selector/.venv
Dependency resolution completed...
Installing collected packages...
Successfully created virtualenv
```

### Verify Poetry Works
```bash
# Activate environment
poetry shell

# Check Python version
python --version  # Should be 3.11+

# Check imports
python -c "import rasterio; print(f'Rasterio {rasterio.__version__}')"
python -c "import geopandas; print('✓ GeoPandas OK')"
python -c "import typer; print('✓ Typer OK')"

# Exit shell
exit
```

---

## 0.4 Create Docker Setup (45 minutes)

### 0.4.1 Create `environment.yml` (Conda environment file)

```bash
cat > environment.yml << 'EOF'
name: marshab
channels:
  - conda-forge
dependencies:
  - python=3.11
  - gdal=3.8
  - rasterio=1.3
  - geopandas=0.14
  - shapely=2.0
  - fiona=1.9
  - pyproj=3.6
  - numpy=1.26
  - scipy=1.11
  - pandas=2.1
  - xarray=2023.12
  - plotly=5.18
  - matplotlib=3.8
  - pip
  - pip:
    - spiceypy>=6.0
    - typer[all]>=0.9
    - rich>=13.7
    - pydantic>=2.5
    - pydantic-settings>=2.1
    - pyyaml>=6.0
    - python-dotenv>=1.0
    - structlog>=24.1
    - pytest>=7.4
    - pytest-cov>=4.1
    - pytest-mock>=3.12
    - ruff>=0.1
    - mypy>=1.7
    - black>=23.12
    - isort>=5.13
EOF
```

### 0.4.2 Create `Dockerfile`

```bash
cat > Dockerfile << 'EOF'
FROM condaforge/mambaforge:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy environment definition
COPY environment.yml .

# Create conda environment
RUN mamba env create -f environment.yml && \
    mamba clean --all -y

# Activate environment in shell
SHELL ["conda", "run", "-n", "marshab", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Install marshab package in development mode
RUN pip install -e .

# Set entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "marshab"]
CMD ["python", "-m", "marshab"]
EOF
```

### 0.4.3 Create `docker-compose.yml`

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Production/testing container
  marshab:
    build: .
    image: marshab:latest
    container_name: marshab-app
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./marshab_config.yaml:/app/marshab_config.yaml:ro
    environment:
      - MARSHAB_CONFIG_PATH=/app/marshab_config.yaml
      - MARSHAB_LOG_LEVEL=INFO
    command: --help

  # Development container with interactive shell
  dev:
    build: .
    image: marshab:latest
    container_name: marshab-dev
    volumes:
      - .:/app
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - MARSHAB_LOG_LEVEL=DEBUG
    stdin_open: true
    tty: true
    command: /bin/bash
    entrypoint: conda run --no-capture-output -n marshab /bin/bash
EOF
```

### 0.4.4 Build and Test Docker

```bash
# Build image
docker-compose build

# Verify image built successfully
docker images | grep marshab

# Test running container
docker-compose run marshab --help

# Launch development shell
# docker-compose run dev  # (Don't run yet, just verify it works)
```

### Expected Output
```
Building marshab
Successfully tagged marshab:latest
```

---

## 0.5 Configure Git Workflow & Pre-commit (30 minutes)

### 0.5.1 Create `.gitignore`

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/
.env

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Data files (git-ignored)
data/raw/**
data/processed/**
data/cache/**
data/output/**
*.tif
*.tiff
*.geojson
*.csv
*.json

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/

# Logs
logs/
*.log

# OS
Thumbs.db
.DS_Store

# IDE
.vscode/
.idea/
EOF
```

### 0.5.2 Create `.pre-commit-config.yaml`

```bash
cat > .pre-commit-config.yaml << 'EOF'
repos:
  # Code formatting
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.11

  # Import sorting
  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  # Linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies:
          - types-pyyaml
          - types-requests

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-merge-conflict
      - id: detect-private-key
EOF

# Install pre-commit hooks
pre-commit install
```

### 0.5.3 Create `.ruff.toml` (Ruff linter config)

```bash
cat > .ruff.toml << 'EOF'
line-length = 100
target-version = "py311"

[lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
]
ignore = [
    "E501",  # Line too long (handled by Black)
    "W503",  # Line break before binary operator
]

[lint.isort]
profile = "black"
EOF
```

### 0.5.4 Create `pyproject.toml` (Poetry config)

```bash
cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "marshab"
version = "0.1.0"
description = "Mars Habitat Site Selection and Rover Navigation System"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "marshab"}]

[tool.poetry.dependencies]
python = "^3.11"
rasterio = "^1.3"
geopandas = "^0.14"
shapely = "^2.0"
fiona = "^1.9"
pyproj = "^3.6"
spiceypy = "^6.0"
numpy = "^1.26"
scipy = "^1.11"
pandas = "^2.1"
xarray = "^2023.12"
plotly = "^5.18"
matplotlib = "^3.8"
typer = {extras = ["all"], version = "^0.9"}
rich = "^13.7"
pydantic = "^2.5"
pydantic-settings = "^2.1"
pyyaml = "^6.0"
python-dotenv = "^1.0"
structlog = "^24.1"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4"
pytest-cov = "^4.1"
pytest-mock = "^3.12"
pytest-asyncio = "^0.21"
ruff = "^0.1"
mypy = "^1.7"
black = "^23.12"
isort = "^5.13"

[tool.poetry.scripts]
marshab = "marshab.cli:app"

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "--cov=marshab --cov-report=html --cov-report=term-missing --strict-markers"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests",
]

[tool.isort]
profile = "black"
line_length = 100

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF
```

---

## 0.6 CI/CD Setup (30 minutes)

### 0.6.1 Create `.github/workflows/ci.yml`

```bash
mkdir -p .github/workflows

cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  # Linting and type checking
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Install dependencies
        run: poetry install
      
      - name: Lint with Ruff
        run: poetry run ruff check marshab tests
      
      - name: Format check with Black
        run: poetry run black --check marshab tests
      
      - name: Type check with mypy
        run: poetry run mypy marshab

  # Unit and integration tests
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Install dependencies
        run: poetry install
      
      - name: Run tests
        run: poetry run pytest -v --cov=marshab --cov-report=xml
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  # Docker build test
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker-compose build
      
      - name: Test Docker image
        run: docker-compose run marshab --help
EOF
```

---

## 0.7 Update README with Getting Started (15 minutes)

```bash
cat > README.md << 'EOF'
# MarsHab Site Selector

A geospatial analysis system for identifying optimal Mars habitat construction sites and generating autonomous rover navigation waypoints.

**Status:** In active development (Phase 0)

## Quick Start

### Option 1: Native Installation with Poetry

```bash
# Install dependencies
poetry install

# Run CLI
poetry run marshab --help

# Run tests
poetry run pytest -v
```

### Option 2: Docker (Recommended)

```bash
# Build image
docker-compose build

# Run pipeline
docker-compose run marshab pipeline --roi "40,41,180,181"

# Development shell
docker-compose run dev
```

## Project Structure

```
marshab-site-selector/
├── marshab/               # Main package
│   ├── core/             # Core services (DataManager, AnalysisPipeline, NavigationEngine)
│   ├── processing/       # Processing modules (DEM, Terrain, Coordinates, Pathfinding)
│   └── utils/            # Utilities (logging, config, validators)
├── tests/                # Test suite
├── data/                 # Data directory (git-ignored)
├── docs/                 # Documentation
└── scripts/              # Helper scripts
```

## Development

### Run Tests
```bash
poetry run pytest -v
poetry run pytest --cov=marshab
```

### Type Checking
```bash
poetry run mypy marshab
```

### Code Formatting
```bash
poetry run black marshab tests
poetry run isort marshab tests
poetry run ruff check marshab tests --fix
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Documentation

- [Architecture Document](docs/architecture.md)
- [Implementation Plan](docs/implementation-plan.md)
- [API Reference](docs/api-reference.md)

## License

MIT License

## Authors

MarsHab Development Team
EOF
```

---

## 0.8 Verification Checklist (15 minutes)

Run these commands to verify Phase 0 is complete:

```bash
# Verify Poetry environment
poetry --version
poetry env info

# Verify all dependencies installed
poetry show

# Verify Python version
poetry run python --version

# Verify key imports
poetry run python -c "import rasterio, geopandas, typer, structlog; print('✓ All imports OK')"

# Verify directory structure
find . -maxdepth 2 -type d | grep -E "^\./(marshab|tests|data|docs)" | head -20

# Verify git is initialized
git log --oneline | head -1

# Verify pre-commit hooks
pre-commit --version

# Build Docker image
docker-compose build

# Test Docker
docker-compose run marshab --help
```

### Expected Output
```
✓ All imports OK
Poetry version 1.7.x
Python 3.11.x
[app] marshab --help
Usage: marshab [OPTIONS] COMMAND [ARGS]...
```

---

## Phase 0 Summary

**Completed:**
- ✅ Git repository initialized
- ✅ Complete directory structure
- ✅ Poetry environment with all dependencies
- ✅ Docker + Docker Compose setup
- ✅ Pre-commit hooks and linting configured
- ✅ GitHub Actions CI/CD pipeline
- ✅ Initial README and documentation structure

**Next Steps:**
- Proceed to Phase 1: Core Infrastructure
- Follow Phase 1 tasks for types, config, logging

**Time Spent:** ~4 hours  
**Time Remaining:** ~56 hours for Phases 1-6

---

**Checkpoint:** All tooling and environment ready. Ready to begin Phase 1 implementation.
