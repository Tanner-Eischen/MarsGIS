# Start MarsHab locally - Quick start script for Windows PowerShell

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MarsHab Local Development" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Add Poetry to PATH if not already there
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    $env:Path += ";$env:APPDATA\Python\Scripts"
    if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
        Write-Host "Error: Poetry is not installed or not in PATH" -ForegroundColor Red
        Write-Host "Please install Poetry: https://python-poetry.org/docs/#installation" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "✓ Poetry found: " -NoNewline
poetry --version
Write-Host ""

# Install dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
poetry install --no-interaction

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MarsHab is ready!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Show help" -ForegroundColor Gray
Write-Host "  poetry run marshab --help" -ForegroundColor White
Write-Host ""
Write-Host "  # Run complete pipeline" -ForegroundColor Gray
Write-Host "  poetry run marshab pipeline --roi `"40,41,180,181`"" -ForegroundColor White
Write-Host ""
Write-Host "  # Download DEM data" -ForegroundColor Gray
Write-Host "  poetry run marshab download mola --roi `"40,41,180,181`"" -ForegroundColor White
Write-Host ""
Write-Host "  # Analyze terrain" -ForegroundColor Gray
Write-Host "  poetry run marshab analyze --roi `"40,41,180,181`"" -ForegroundColor White
Write-Host ""
Write-Host "  # Run tests" -ForegroundColor Gray
Write-Host "  poetry run pytest -v" -ForegroundColor White
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
