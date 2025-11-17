# MarsHab Docker Startup Script
# This script starts the MarsHab application using Docker

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MarsHab Site Selector - Docker Startup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    Write-Host "Docker is available" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not installed or not running" -ForegroundColor Red
    Write-Host "Please install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if docker-compose is available
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: docker-compose is not available" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker-compose build

Write-Host ""
Write-Host "Docker services are ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  docker-compose run marshab --help              - Show help"
Write-Host "  docker-compose run marshab --version           - Show version"
Write-Host "  docker-compose run marshab download            - Download DEM data"
Write-Host "  docker-compose run marshab analyze             - Analyze terrain"
Write-Host "  docker-compose run marshab navigate            - Generate waypoints"
Write-Host "  docker-compose run marshab pipeline            - Run full pipeline"
Write-Host "  docker-compose run dev                          - Development shell"
Write-Host ""
Write-Host "Example:" -ForegroundColor Cyan
Write-Host "  docker-compose run marshab pipeline --roi `"40,41,180,181`" --output /app/data/output"
Write-Host ""

# Optionally run a command if provided
if ($args.Count -gt 0) {
    Write-Host "Running: docker-compose run marshab $($args -join ' ')" -ForegroundColor Yellow
    Write-Host ""
    docker-compose run --rm marshab $args
} else {
    Write-Host "To run a command, use:" -ForegroundColor Yellow
    Write-Host "  .\scripts\start_docker.ps1 <command> <args>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or run directly:" -ForegroundColor Yellow
    Write-Host "  docker-compose run --rm marshab <command> <args>" -ForegroundColor Yellow
}

