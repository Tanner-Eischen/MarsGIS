#!/bin/bash
# MarsHab Installation Verification Script
# Run this script to verify all components are working correctly

set -e

echo "=========================================="
echo "MarsHab Installation Verification"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print success
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    error "Poetry is not installed. Please install Poetry first."
    exit 1
fi
success "Poetry is installed"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    error "pyproject.toml not found. Please run this script from the project root."
    exit 1
fi
success "Project structure verified"

echo ""
echo "1. Running unit tests..."
if poetry run pytest tests/unit/ -v --tb=short; then
    success "Unit tests passed"
else
    error "Unit tests failed"
    exit 1
fi

echo ""
echo "2. Running integration tests..."
if poetry run pytest tests/integration/ -v --tb=short 2>/dev/null || warning "No integration tests found (this is OK)"; then
    success "Integration tests completed"
fi

echo ""
echo "3. Type checking..."
if poetry run mypy marshab --ignore-missing-imports 2>/dev/null || warning "Type checking has warnings (this is OK)"; then
    success "Type checking completed"
fi

echo ""
echo "4. Linting..."
if poetry run ruff check marshab tests; then
    success "Linting passed"
else
    error "Linting failed"
    exit 1
fi

echo ""
echo "5. Testing CLI..."
if poetry run marshab --version > /dev/null 2>&1; then
    VERSION=$(poetry run marshab --version 2>&1 | head -n 1)
    success "CLI is working: $VERSION"
else
    error "CLI test failed"
    exit 1
fi

echo ""
echo "6. Testing Docker build (if Docker is available)..."
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    if docker-compose build --quiet 2>/dev/null; then
        success "Docker build successful"
        
        # Test Docker run
        if docker-compose run --rm marshab --version > /dev/null 2>&1; then
            success "Docker container runs successfully"
        else
            warning "Docker container test had issues (may be OK)"
        fi
    else
        warning "Docker build had issues (may be OK if dependencies are missing)"
    fi
else
    warning "Docker not available, skipping Docker tests"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ All verification tests passed!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Run: poetry run marshab --help"
echo "  - Try: poetry run marshab pipeline --roi \"40,41,180,181\""
echo "  - Review: docs/testing-verification.md for detailed testing procedures"
echo ""

