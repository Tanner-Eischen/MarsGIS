# Smoke test script for MarsHab
# Runs a small end-to-end scenario to verify system functionality

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MarsHab Smoke Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

try {
    # Test 1: Health check
    Write-Host "Test 1: Health Check" -ForegroundColor Yellow
    $healthResponse = Invoke-WebRequest -Uri "http://localhost:5000/api/v1/health/live" -Method GET -UseBasicParsing
    if ($healthResponse.StatusCode -eq 200) {
        Write-Host "  ✓ Health check passed" -ForegroundColor Green
    } else {
        throw "Health check failed with status $($healthResponse.StatusCode)"
    }
    
    # Test 2: List presets
    Write-Host "Test 2: List Presets" -ForegroundColor Yellow
    $presetsResponse = Invoke-WebRequest -Uri "http://localhost:5000/api/v1/analysis/presets" -Method GET -UseBasicParsing
    if ($presetsResponse.StatusCode -eq 200) {
        $presets = $presetsResponse.Content | ConvertFrom-Json
        Write-Host "  ✓ Found $($presets.site_presets.Count) site presets" -ForegroundColor Green
    } else {
        throw "Presets check failed"
    }
    
    # Test 3: Example ROIs
    Write-Host "Test 3: Example ROIs" -ForegroundColor Yellow
    $examplesResponse = Invoke-WebRequest -Uri "http://localhost:5000/api/v1/examples/rois" -Method GET -UseBasicParsing
    if ($examplesResponse.StatusCode -eq 200) {
        $examples = $examplesResponse.Content | ConvertFrom-Json
        Write-Host "  ✓ Found $($examples.Count) example ROIs" -ForegroundColor Green
    } else {
        throw "Examples check failed"
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  All smoke tests passed!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Smoke test failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

