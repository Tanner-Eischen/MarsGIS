#!/bin/bash
# Start MarsHab locally - Quick start script

echo "=========================================="
echo "MarsHab Local Development"
echo "=========================================="
echo ""

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed or not in PATH"
    echo "Please install Poetry: https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "✓ Poetry found: $(poetry --version)"
echo ""

# Install dependencies if needed
echo "Checking dependencies..."
poetry install --no-interaction

echo ""
echo "=========================================="
echo "MarsHab is ready!"
echo "=========================================="
echo ""
echo "Available commands:"
echo ""
echo "  # Show help"
echo "  poetry run marshab --help"
echo ""
echo "  # Run complete pipeline"
echo "  poetry run marshab pipeline --roi \"40,41,180,181\""
echo ""
echo "  # Download DEM data"
echo "  poetry run marshab download mola --roi \"40,41,180,181\""
echo ""
echo "  # Analyze terrain"
echo "  poetry run marshab analyze --roi \"40,41,180,181\""
echo ""
echo "  # Run tests"
echo "  poetry run pytest -v"
echo ""
echo "=========================================="




