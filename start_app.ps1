# Start MarsHab Application - Backend and Frontend

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Starting MarsHab Application" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Start Backend Server
Write-Host "Starting Backend API Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; Write-Host '=== Backend API Server ===' -ForegroundColor Green; poetry run python -m marshab.web.server"

# Wait a moment
Start-Sleep -Seconds 2

# Start Frontend Server
Write-Host "Starting Frontend Dev Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\webui'; Write-Host '=== Frontend Dev Server ===' -ForegroundColor Green; npm run dev"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Servers Starting..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend UI: http://localhost:4000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Two PowerShell windows should open." -ForegroundColor Yellow
Write-Host "Wait ~10 seconds for servers to start, then open:" -ForegroundColor Yellow
Write-Host "http://localhost:4000" -ForegroundColor White -BackgroundColor DarkBlue
Write-Host ""
Write-Host "Press any key to exit this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")


