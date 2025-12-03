# Prepare Demo Data Script
# Pre-loads all necessary data for MarsHab demo workflow
# Region: Jezero Crater (18.0-18.6°N, 77.0-77.8°E)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MarsHab Demo Data Preparation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$roi = "18.0,18.6,77.0,77.8"
$dataset = "mola"
$outputDir = "data/output"

# Ensure output directory exists
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Write-Host "Created output directory: $outputDir" -ForegroundColor Green
}

Write-Host "Demo Region: Jezero Crater" -ForegroundColor Yellow
Write-Host "  Latitude: 18.0-18.6°N" -ForegroundColor Gray
Write-Host "  Longitude: 77.0-77.8°E" -ForegroundColor Gray
Write-Host "  Dataset: $dataset" -ForegroundColor Gray
Write-Host ""

# Step 1: Download DEM Data
Write-Host "Step 1: Downloading DEM Data..." -ForegroundColor Cyan
try {
    poetry run marshab download $dataset --roi $roi
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ DEM data downloaded successfully" -ForegroundColor Green
    } else {
        Write-Host "  ✗ DEM download failed" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ✗ Error downloading DEM: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Run Terrain Analysis
Write-Host "Step 2: Running Terrain Analysis..." -ForegroundColor Cyan
try {
    poetry run marshab analyze --roi $roi --dataset $dataset --output $outputDir --threshold 0.7
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Terrain analysis completed" -ForegroundColor Green
        
        # Check if sites.csv was created
        $sitesFile = Join-Path $outputDir "sites.csv"
        if (Test-Path $sitesFile) {
            $siteCount = (Import-Csv $sitesFile).Count
            Write-Host "  ✓ Found $siteCount suitable sites" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ Warning: sites.csv not found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ✗ Terrain analysis failed" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ✗ Error running analysis: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Generate Navigation Waypoints (if sites found)
$sitesFile = Join-Path $outputDir "sites.csv"
if (Test-Path $sitesFile) {
    $sites = Import-Csv $sitesFile
    if ($sites.Count -gt 0) {
        $firstSite = $sites[0]
        $siteId = $firstSite.site_id
        
        Write-Host "Step 3: Generating Navigation Waypoints..." -ForegroundColor Cyan
        Write-Host "  Target Site: $siteId" -ForegroundColor Gray
        Write-Host "  Start Position: 18.3°N, 77.4°E (center of ROI)" -ForegroundColor Gray
        
        try {
            poetry run marshab navigate $siteId --analysis $outputDir --start-lat 18.3 --start-lon 77.4 --output (Join-Path $outputDir "waypoints.csv")
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ Navigation waypoints generated" -ForegroundColor Green
                
                # Check if waypoints.csv was created
                $waypointsFile = Join-Path $outputDir "waypoints.csv"
                if (Test-Path $waypointsFile) {
                    $waypointCount = (Import-Csv $waypointsFile).Count
                    Write-Host "  ✓ Generated $waypointCount waypoints" -ForegroundColor Green
                }
            } else {
                Write-Host "  ⚠ Navigation planning failed (may be due to impassable terrain)" -ForegroundColor Yellow
                Write-Host "    This is OK for demo - you can plan routes manually in the UI" -ForegroundColor Gray
            }
        } catch {
            Write-Host "  ⚠ Navigation planning error: $_" -ForegroundColor Yellow
            Write-Host "    This is OK for demo - you can plan routes manually in the UI" -ForegroundColor Gray
        }
    } else {
        Write-Host "Step 3: Skipping navigation (no sites found)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Step 3: Skipping navigation (sites.csv not found)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Demo Data Preparation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Start the backend API: poetry run python -m marshab.web.server" -ForegroundColor White
Write-Host "  2. Start the frontend: cd webui; npm run dev" -ForegroundColor White
Write-Host "  3. Open http://localhost:4000 in your browser" -ForegroundColor White
Write-Host "  4. Follow DEMO_WORKFLOW.md for complete demo guide" -ForegroundColor White
Write-Host ""
Write-Host "Demo Data Location:" -ForegroundColor Yellow
Write-Host "  Sites: $(Join-Path $outputDir 'sites.csv')" -ForegroundColor White
if (Test-Path (Join-Path $outputDir "waypoints.csv")) {
    Write-Host "  Waypoints: $(Join-Path $outputDir 'waypoints.csv')" -ForegroundColor White
}
$cachePath = Join-Path "data" "cache"
Write-Host "  DEM Cache: $cachePath" -ForegroundColor White
Write-Host ""

