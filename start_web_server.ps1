# Start MarsHab Web API Server
Write-Host "Starting MarsHab Web API Server..." -ForegroundColor Cyan
Write-Host "Server will be available at http://localhost:5000" -ForegroundColor Green
Write-Host "API docs will be available at http://localhost:5000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

poetry run python -m marshab.web.server

