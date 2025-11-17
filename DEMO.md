# MarsHab Demo Guide

Quick guide to run a demonstration of the MarsHab application.

## 🚀 Quick Demo (Easiest Way)

### Option 1: Using Docker (Recommended - No Setup Needed)

```powershell
# 1. Show help and available commands
docker-compose run --rm marshab --help

# 2. Run a complete demo pipeline
docker-compose run --rm marshab pipeline --roi "40,41,180,181" --output /app/data/output

# 3. Or use the shorter 'mars' commands
docker-compose run --rm mars download mola --roi "40,41,180,181"
docker-compose run --rm mars terrain --roi "40,41,180,181" --output /app/data/output
```

### Option 2: Using Poetry (If Installed)

```powershell
# Add Poetry to PATH if needed
$env:Path += ";$env:APPDATA\Python\Scripts"

# Run demo
poetry run marshab pipeline --roi "40,41,180,181" --output data/output

# Or use shorter commands
poetry run mars download mola --roi "40,41,180,181"
poetry run mars terrain --roi "40,41,180,181" --output data/output
```

## 📋 Step-by-Step Demo

### Step 1: Check Installation

```powershell
# Docker
docker-compose run --rm marshab --version

# Poetry
poetry run marshab --version
```

### Step 2: Download DEM Data

```powershell
# Download MOLA DEM for a region on Mars
docker-compose run --rm marshab download mola --roi "40,41,180,181"

# Or with shorter command
docker-compose run --rm mars download mola --roi "40,41,180,181"
```

**What this does:**
- Downloads Mars elevation data (MOLA dataset)
- Caches it in `data/cache/` directory
- Region: Latitude 40-41°, Longitude 180-181° (near Olympus Mons area)

### Step 3: Analyze Terrain

```powershell
# Analyze terrain and find suitable sites
docker-compose run --rm marshab analyze --roi "40,41,180,181" --output /app/data/output

# Or shorter
docker-compose run --rm mars terrain --roi "40,41,180,181" --output /app/data/output
```

**What this does:**
- Loads DEM data
- Calculates slope, roughness, and terrain metrics
- Applies MCDM (Multi-Criteria Decision Making)
- Identifies suitable construction sites
- Saves results to `data/output/sites.csv`

### Step 4: Generate Navigation Waypoints

```powershell
# Generate waypoints to site #1
docker-compose run --rm marshab navigate 1 \
  --analysis /app/data/output \
  --start-lat 40.5 \
  --start-lon 180.5 \
  --output /app/data/output/waypoints.csv

# Or shorter
docker-compose run --rm mars navigation 1 \
  --analysis /app/data/output \
  --start-lat 40.5 \
  --start-lon 180.5 \
  --output /app/data/output/waypoints.csv
```

**What this does:**
- Loads site location from analysis results
- Transforms coordinates to rover SITE frame
- Generates A* pathfinding waypoints
- Saves waypoints to CSV file

### Step 5: Run Complete Pipeline (All-in-One)

```powershell
# Run everything in one command
docker-compose run --rm marshab pipeline \
  --roi "40,41,180,181" \
  --output /app/data/output
```

**What this does:**
1. Downloads DEM data
2. Analyzes terrain
3. Identifies sites
4. Generates navigation waypoints
5. Saves all results

## 🎯 Demo Examples

### Example 1: Small Region (Fast)

```powershell
# Small region for quick demo
docker-compose run --rm marshab pipeline --roi "40,40.5,180,180.5" --output /app/data/output
```

### Example 2: Different Mars Region

```powershell
# Valles Marineris area
docker-compose run --rm marshab pipeline --roi "-10,-9,280,281" --output /app/data/output
```

### Example 3: Using Shorter Commands

```powershell
# Download
docker-compose run --rm mars download mola --roi "40,41,180,181"

# Analyze
docker-compose run --rm mars terrain --roi "40,41,180,181" --output /app/data/output

# Navigate
docker-compose run --rm mars navigation 1 \
  --analysis /app/data/output \
  --start-lat 40.5 \
  --start-lon 180.5
```

## 📊 View Results

After running the demo, check the output:

```powershell
# View sites found
cat data/output/sites.csv

# View waypoints
cat data/output/waypoints.csv

# List all output files
ls data/output/
```

## 🔍 What to Expect

### Successful Run Output

```
✓ DEM downloaded
✓ Terrain analyzed
✓ Navigation planned
✓ Pipeline complete!
Results saved to: data/output
```

### Output Files

- `data/output/sites.csv` - Identified construction sites with suitability scores
- `data/output/waypoints.csv` - Navigation waypoints in SITE frame coordinates
- `data/cache/` - Cached DEM files (reused on subsequent runs)

## ⚠️ Troubleshooting

### Docker Not Running
```powershell
# Start Docker Desktop first, then:
docker-compose run --rm marshab --version
```

### Download Fails
- Check internet connection
- DEM URLs may change - see `docs/DATA_SOURCES.md` for manual download options

### No Sites Found
- Try a different ROI (region of interest)
- Lower the threshold: `--threshold 0.5`
- Some regions may not have suitable terrain

## 🎓 Next Steps

1. Try different regions of Mars
2. Experiment with different thresholds
3. Check the generated CSV files
4. Read the full documentation in `README.md`

---

**Tip**: Use `--help` with any command to see all options:
```powershell
docker-compose run --rm marshab --help
docker-compose run --rm marshab analyze --help
```




