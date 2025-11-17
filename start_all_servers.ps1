# Start MarsHab Web Application (Backend + Frontend)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MarsHab Web Application Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if ports are available
$port5000 = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
$port4000 = Get-NetTCPConnection -LocalPort 4000 -ErrorAction SilentlyContinue

if ($port5000) {
    Write-Host "⚠️  Port 5000 is already in use. Backend server may already be running." -ForegroundColor Yellow
} else {
    Write-Host "Starting Backend API Server on port 5000..." -ForegroundColor Green
    $poetryPath = "$env:APPDATA\Python\Scripts"
    $poetryExe = Join-Path $poetryPath "poetry.exe"
    if (Test-Path $poetryExe) {
        $backendCmd = "cd '$PSScriptRoot'; & '$poetryExe' run python -m marshab.web.server"
    } else {
        $backendCmd = "cd '$PSScriptRoot'; poetry run python -m marshab.web.server"
    }
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Normal
    Start-Sleep -Seconds 3
}

if ($port4000) {
    Write-Host "⚠️  Port 4000 is already in use. Frontend server may already be running." -ForegroundColor Yellow
} else {
    Write-Host "Starting Frontend Dev Server on port 4000..." -ForegroundColor Green
    $frontendCmd = "cd '$PSScriptRoot\webui'; npm run dev"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -WindowStyle Normal
}


Write-Host ""
Write-Host "✓ Servers are starting in separate windows" -ForegroundColor Green
Write-Host ""
Write-Host "Backend API:  http://localhost:5000" -ForegroundColor Cyan
Write-Host "API Docs:     http://localhost:5000/docs" -ForegroundColor Cyan
Write-Host "Frontend UI:  http://localhost:4000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit this script (servers will continue running)..." -ForegroundColor Yellow
Read-Host



