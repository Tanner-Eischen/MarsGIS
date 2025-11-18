# MarsHab Mission Walkthrough

This guide walks you through a complete mission scenario using MarsHab, from selecting a region to planning a rover traverse.

## Scenario: Finding a Safe Landing Site Near Jezero Crater

### Step 1: Load Example Region

1. Open the **Decision Lab** page
2. Click **"Load Example"** button
3. Select **"Jezero Crater"** from the examples drawer
   - This automatically sets the ROI to the Perseverance rover landing site
   - Dataset is set to MOLA

### Step 2: Choose Analysis Preset

1. In the **Preset Selection** panel, choose **"Safe Landing"**
   - This preset prioritizes gentle slopes (40% weight) and smooth terrain (30% weight)
   - Ideal for risk-averse missions

2. Optionally, click **"Advanced Weights"** to see and customize the criterion weights
   - Adjust sliders to fine-tune priorities
   - Ensure weights sum to 100%

### Step 3: Run Analysis

1. Click **"Run Analysis"**
2. Wait for the analysis to complete (this may take a minute for large regions)
3. Review the ranked sites in the left panel
   - Sites are sorted by suitability score (highest first)
   - Each site shows: rank, score, area, slope, and location

### Step 4: Explore Results

1. **View on Map**: The terrain map shows:
   - DEM imagery with hillshade relief
   - Site polygons (or points) overlaid
   - Color-coded by suitability score

2. **Select a Site**: Click on a site in the list to:
   - Highlight it on the map
   - See detailed score breakdown in the **Explainability Panel**
   - Understand why it scored well (e.g., "very gentle slopes, smooth terrain")

3. **Explain This Map**: Toggle the **"Explain this map"** checkbox to see:
   - What the colors represent
   - What the current preset optimizes for
   - Plain-language description of the analysis

### Step 5: Compare Presets

1. Switch to **"Science-Focused"** preset
2. Click **"Run Analysis"** again
3. Observe how the site rankings change:
   - Science-focused preset accepts steeper slopes (max 12° vs 5°)
   - Rankings prioritize sites near features of scientific interest
   - Top sites may be different from the Safe Landing preset

### Step 6: Plan Rover Traverse

1. Navigate to **Mission Scenarios** page
2. Select **"Rover Traverse Wizard"** tab
3. **Step 1**: Select start and end sites from your analysis
   - Start: Site 1 (your landing site)
   - End: Site 2 (target exploration site)
4. **Step 2**: Configure rover capabilities:
   - Max slope: 25° (typical rover capability)
   - Route preset: Choose **"Safest Path"** for maximum safety
5. Click **"Plan Route"**
6. Review results:
   - Total distance
   - Estimated travel time
   - Risk score
   - Waypoint list

### Step 7: Export Data Products

1. In **Decision Lab**, after running analysis:
   - Click **"Export GeoTIFF"** to download suitability raster
   - Click **"Export Report"** to generate analysis summary (Markdown or HTML)

2. In **Mission Scenarios**, after planning a route:
   - Waypoints can be exported as CSV
   - Route visualization can be saved as image

### Step 8: Save as Project

1. Click **"Save as Project"** in Decision Lab or Mission Scenarios
2. Enter project name and description
3. Project saves:
   - Current ROI and dataset
   - Selected preset
   - Selected sites
   - All analysis parameters

4. Later, open the project from the **Projects** page to restore your work

## Advanced Features

### Custom Weights

Instead of using presets, you can:
1. Expand **"Advanced Weights"** panel
2. Adjust sliders for each criterion
3. System validates that weights sum to 100%
4. Run analysis with your custom configuration

### Shadow-Aware Routing

When planning routes:
1. Expand **"Sun Position Controls"**
2. Set sun azimuth (0-360°) and altitude (0-90°)
3. Route planning will avoid shadowed areas
4. Useful for solar-powered rovers

### 3D Visualization

1. Go to **Visualization** page
2. Set ROI and dataset
3. Toggle to **3D view**
4. Adjust relief/exaggeration slider
5. Rotate and zoom to explore terrain

## Tips

- **Start Small**: Begin with small ROIs (0.5° x 0.5°) for faster analysis
- **Use Examples**: Example ROIs are pre-configured with interesting regions
- **Compare Presets**: Run the same ROI with different presets to understand tradeoffs
- **Save Projects**: Save your work frequently to avoid re-running analyses
- **Export Early**: Export results as you go to preserve intermediate findings

## Next Steps

- Explore other example regions (Gale Crater, Olympus Mons, Valles Marineris)
- Try the **Landing Site Wizard** for guided mission planning
- Use **CLI commands** for batch processing and automation
- Check **Validation** page for system health and synthetic testing

