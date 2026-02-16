# Navigation Planning Performance Optimization Guide

## Current Performance Analysis

### Bottleneck Identification

Based on code analysis and test logs, the primary bottlenecks are:

1. **Terrain Analysis (Lines 312-322 in `navigation_engine.py`)**
   - Called for every navigation request when pickle file unavailable
   - Most expensive: `roughness` calculation (21 seconds in test logs)
   - Analyzed entire DEM instead of path corridor

2. **Large ROI Creation (Lines 302-308)**
   - Current: Creates ROI spanning start→goal + 0.1° buffer
   - Problem: For distant points (e.g., 18.2°→40.2°), creates massive ROI
   - Example: 22° lat span = ~2400 km = millions of pixels

3. **Cost Surface Generation (Lines 427-438)**
   - Computes cost for entire DEM, even for areas not on the path
   - Processes every pixel even though pathfinding only explores a fraction

---

## Optimization Strategies

### Strategy 1: Limit ROI Size ⭐ **Highest Impact**

**Current Code (Lines 302-308):**
```python
roi_size = 0.1
roi = BoundingBox(
    lat_min=min(start_lat, site_lat) - roi_size,
    lat_max=max(start_lat, site_lat) + roi_size,
    lon_min=min(start_lon, site_lon) - roi_size,
    lon_max=max(start_lon, site_lon) + roi_size
)
```

**Problem**: For distant points, this creates huge ROIs.

**Optimization A: Cap Maximum ROI Span**
```python
# Maximum ROI dimensions (degrees)
MAX_ROI_SPAN = 2.0  # ~200 km on Mars
roi_padding = 0.1

lat_span = abs(site_lat - start_lat)
lon_span = abs(site_lon - start_lon)

# If distance is too large, fail early with helpful message
if lat_span > MAX_ROI_SPAN or lon_span > MAX_ROI_SPAN:
    raise NavigationError(
        f"Navigation distance too large (lat: {lat_span:.1f}°, lon: {lon_span:.1f}°). "
        f"Maximum supported: {MAX_ROI_SPAN}°. Use intermediate waypoints or split into segments.",
        details={"start": (start_lat, start_lon), "goal": (site_lat, site_lon)}
    )

roi = BoundingBox(
    lat_min=min(start_lat, site_lat) - roi_padding,
    lat_max=max(start_lat, site_lat) + roi_padding,
    lon_min=min(start_lon, site_lon) - roi_padding,
    lon_max=max(start_lon, site_lon) + roi_padding
)
```

**Expected Impact**: Prevents massive ROI creation, fails fast with clear error message.

---

**Optimization B: Beeline ROI (Corridor Approach)**
```python
# Instead of bounding box, create a corridor along the beeline
# This reduces area by ~50% for diagonal paths

roi_padding = 0.15  # Slightly larger to allow detours

# Beeline vector
lat_diff = site_lat - start_lat
lon_diff = site_lon - start_lon
distance_deg = math.sqrt(lat_diff**2 + lon_diff**2)

# Perpendicular width (smaller for DIRECT strategy, larger for SAFEST)
corridor_width = {
    PathfindingStrategy.DIRECT: 0.05,
    PathfindingStrategy.BALANCED: 0.1,
    PathfindingStrategy.SAFEST: 0.2,
}.get(strategy, 0.1)

# Create ROI as rectangle rotated along beeline (simplified: use bounding box + margin)
# For production: implement proper rotated rectangle
roi = BoundingBox(
    lat_min=min(start_lat, site_lat) - corridor_width,
    lat_max=max(start_lat, site_lat) + corridor_width,
    lon_min=min(start_lon, site_lon) - corridor_width,
    lon_max=max(start_lon, site_lon) + corridor_width
)

# Cap total ROI size
lat_span = roi.lat_max - roi.lat_min
lon_span = roi.lon_max - roi.lon_min
if lat_span > MAX_ROI_SPAN or lon_span > MAX_ROI_SPAN:
    # Scale down proportionally
    scale = min(MAX_ROI_SPAN / lat_span, MAX_ROI_SPAN / lon_span)
    center_lat = (roi.lat_min + roi.lat_max) / 2
    center_lon = (roi.lon_min + roi.lon_max) / 2
    roi = BoundingBox(
        lat_min=center_lat - (lat_span * scale / 2),
        lat_max=center_lat + (lat_span * scale / 2),
        lon_min=center_lon - (lon_span * scale / 2),
        lon_max=center_lon + (lon_span * scale / 2)
    )
```

**Expected Impact**: 30-50% reduction in DEM size for diagonal paths.

---

### Strategy 2: Skip Redundant Terrain Analysis ⭐ **High Impact**

**Current Issue**: When pickle file doesn't exist, full terrain analysis runs (slope, roughness, TRI, aspect, hillshade).

**Optimization: Minimal Terrain Analysis for Navigation**

Navigation only needs:
- ✅ Elevation (already loaded with DEM)
- ✅ Slope (needed for cost map)
- ✅ Roughness (needed for cost map)
- ❌ TRI, aspect, hillshade (NOT needed)

**File to Modify**: `marshab/processing/terrain.py`

Add a `minimal=True` parameter to `TerrainAnalyzer.analyze()`:

```python
# In navigation_engine.py line 321:
metrics = terrain_analyzer.analyze(dem, minimal=True)  # Skip TRI, aspect, hillshade

# In terrain.py:
def analyze(self, dem: xr.DataArray, minimal: bool = False) -> TerrainMetrics:
    """Analyze terrain metrics from DEM.
    
    Args:
        dem: Digital elevation model
        minimal: If True, only compute slope and roughness (skip TRI, aspect, hillshade)
    """
    logger.info("Starting terrain analysis", shape=dem.shape)
    
    elevation = dem.values
    slope = self._calculate_slope(elevation)
    logger.info("Calculating roughness...")
    roughness = self._calculate_roughness(elevation)
    
    if minimal:
        # For navigation, we don't need full metrics
        logger.info("Minimal analysis mode - skipping TRI, aspect, hillshade")
        return TerrainMetrics(
            elevation=elevation,
            slope=slope,
            aspect=np.zeros_like(slope),  # Placeholder
            roughness=roughness,
            tri=np.zeros_like(slope),     # Placeholder
            hillshade=np.ones_like(slope) * 180.0  # Placeholder
        )
    
    # Full analysis
    logger.info("Calculating aspect...")
    aspect = self._calculate_aspect(slope)
    logger.info("Calculating TRI...")
    tri = self._calculate_tri(elevation)
    logger.info("Calculating hillshade...")
    hillshade = self._calculate_hillshade(elevation)
    
    return TerrainMetrics(...)
```

**Expected Impact**: ~40% reduction in terrain analysis time (skips TRI, aspect, hillshade).

---

### Strategy 3: Increase Downsampling for Large DEMs

**Current Code (Lines 596-598):**
```python
# If cost map is very large (> 1000x1000), downsample by 2x
if cost_map.shape[0] * cost_map.shape[1] > 1000000:
    downsample_factor = 2
```

**Optimization: Adaptive Downsampling**
```python
# Adaptive downsampling based on DEM size
pixels = cost_map.shape[0] * cost_map.shape[1]

if pixels > 5_000_000:      # > 5 megapixels
    downsample_factor = 4   # Downsample to ~300k pixels
elif pixels > 2_000_000:    # > 2 megapixels
    downsample_factor = 3
elif pixels > 1_000_000:    # > 1 megapixel
    downsample_factor = 2
else:
    downsample_factor = 1   # No downsampling

logger.info(
    "Pathfinding downsample factor determined",
    pixels=pixels,
    downsample_factor=downsample_factor,
    approx_pixels_after=pixels // (downsample_factor ** 2)
)
```

**Expected Impact**: 
- 4x downsample → 16x fewer pixels for A* to explore
- Path quality remains good (waypoints upsampled back to original resolution)

---

### Strategy 4: Cache DEM and Metrics per ROI

**Current Issue**: Every navigation request loads and analyzes the DEM from scratch.

**Optimization: In-Memory Cache with LRU Eviction**

```python
from functools import lru_cache
from typing import Tuple

# Add to NavigationEngine class:

@lru_cache(maxsize=4)
def _get_dem_and_metrics_cached(
    self,
    roi_key: Tuple[float, float, float, float],  # (lat_min, lat_max, lon_min, lon_max)
    dataset: str,
    max_slope_deg: float
) -> Tuple[xr.DataArray, Any]:
    """Load DEM and compute metrics with caching."""
    roi = BoundingBox(
        lat_min=roi_key[0],
        lat_max=roi_key[1],
        lon_min=roi_key[2],
        lon_max=roi_key[3]
    )
    
    dem = self.data_manager.get_dem_for_roi(roi, dataset=dataset, download=True, clip=True)
    cell_size_m = self.config.data_sources.get("mola", {}).get("resolution_m", 463.0)
    
    terrain_analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
    metrics = terrain_analyzer.analyze(dem, minimal=True)  # Use minimal analysis
    
    return dem, metrics

# In plan_to_site(), replace lines 310-322 with:
roi_key = (roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max)
dem, metrics = self._get_dem_and_metrics_cached(roi_key, "mola", max_slope_deg)
```

**Expected Impact**: Subsequent requests with same/similar ROI are instant (no re-analysis).

---

### Strategy 5: Frontend Validation

**Add to `NavigationPlanning.tsx`:**

```tsx
// Before submission, validate distance
const validateDistance = (startLat: number, startLon: number, siteId: number) => {
  const site = constructionSites.find(s => s.site_id === siteId)
  if (!site) return true  // Let backend handle missing site
  
  const latDiff = Math.abs(site.lat - startLat)
  const lonDiff = Math.abs(site.lon - startLon)
  
  const MAX_SPAN = 2.0  // degrees (~200 km)
  
  if (latDiff > MAX_SPAN || lonDiff > MAX_SPAN) {
    return {
      valid: false,
      message: `Distance too large (${Math.max(latDiff, lonDiff).toFixed(1)}° > ${MAX_SPAN}°). Use closer waypoints or split route into segments.`
    }
  }
  
  return { valid: true }
}

// In form submit:
const validation = validateDistance(startLat, startLon, siteId)
if (!validation.valid) {
  setError(validation.message)
  return
}
```

**Expected Impact**: Prevent expensive backend calls, instant feedback to user.

---

### Strategy 6: Progress Streaming (Already Implemented)

The code already supports progress tracking (lines 68-74, 268-273). Frontend just needs to connect to WebSocket.

---

## Recommended Implementation Order

| Priority | Strategy | Files to Modify | Estimated Time | Impact |
|----------|----------|-----------------|----------------|--------|
| 🔴 **P0** | Cap ROI size | `navigation_engine.py` (lines 302-308) | 5 min | Prevents catastrophic failures |
| 🔴 **P0** | Frontend validation | `NavigationPlanning.tsx` | 10 min | Instant user feedback |
| 🟡 **P1** | Minimal terrain analysis | `terrain.py`, `navigation_engine.py` | 15 min | ~40% speedup |
| 🟡 **P1** | Adaptive downsampling | `navigation_engine.py` (lines 596-598) | 10 min | 4-16x faster for large DEMs |
| 🟢 **P2** | DEM caching | `navigation_engine.py` | 20 min | Instant for repeated queries |
| 🟢 **P3** | Corridor ROI | `navigation_engine.py` | 30 min | 30-50% smaller ROI |

---

## Quick Win: Combined P0 Implementation

### File: `marshab/core/navigation_engine.py`

Replace lines 302-308:

```python
# Configuration
MAX_ROI_SPAN_DEG = 2.0  # ~200 km on Mars at equator
roi_padding = 0.1

# Calculate required ROI
lat_span = abs(site_lat - start_lat)
lon_span = abs(site_lon - start_lon)
distance_deg = math.sqrt(lat_span**2 + lon_span**2)

# Fail fast if distance is too large
if lat_span > MAX_ROI_SPAN_DEG or lon_span > MAX_ROI_SPAN_DEG:
    raise NavigationError(
        f"Navigation distance too large ({lat_span:.2f}° lat, {lon_span:.2f}° lon). "
        f"Maximum supported: {MAX_ROI_SPAN_DEG}° (~200 km). "
        f"Suggestion: Use intermediate waypoints or select a closer start position. "
        f"Distance: ~{distance_deg * 59:.0f} km (straight-line beeline).",
        details={
            "start_lat": start_lat,
            "start_lon": start_lon,
            "goal_lat": site_lat,
            "goal_lon": site_lon,
            "lat_span_deg": lat_span,
            "lon_span_deg": lon_span,
            "max_span_deg": MAX_ROI_SPAN_DEG
        }
    )

roi = BoundingBox(
    lat_min=min(start_lat, site_lat) - roi_padding,
    lat_max=max(start_lat, site_lat) + roi_padding,
    lon_min=min(start_lon, site_lon) - roi_padding,
    lon_max=max(start_lon, site_lon) + roi_padding
)

logger.info(
    "Navigation ROI created",
    roi=roi,
    span_lat=lat_span,
    span_lon=lon_span,
    approx_distance_km=distance_deg * 59
)
```

### File: `webui/src/pages/NavigationPlanning.tsx`

Add validation before submit (after line 57):

```tsx
const handleSubmit = (e: React.FormEvent) => {
  e.preventDefault()
  setTaskId(null)
  
  // Validate distance
  const site = constructionSites.find(s => s.site_id === siteId)
  if (site) {
    const latDiff = Math.abs(site.lat - startLat)
    const lonDiff = Math.abs(site.lon - startLon)
    const MAX_SPAN = 2.0
    
    if (latDiff > MAX_SPAN || lonDiff > MAX_SPAN) {
      // Show error in UI instead of submitting
      alert(
        `Navigation distance too large (${Math.max(latDiff, lonDiff).toFixed(1)}° > ${MAX_SPAN}°). ` +
        `Approximate distance: ${Math.sqrt(latDiff**2 + lonDiff**2) * 59:.0f} km. ` +
        `Please use a closer start position or select an intermediate waypoint.`
      )
      return
    }
  }
  
  mutation.mutate({
    site_id: siteId,
    start_lat: startLat,
    start_lon: startLon,
    strategy,
    analysis_dir: analysisDir,
  })
}
```

---

## Performance Benchmarks (Estimated)

| Optimization | ROI Size | Analysis Time | Pathfinding Time | Total Time |
|--------------|----------|---------------|------------------|------------|
| **Current (no limit)** | 22° × 103° | ~30s | ~60s+ | >90s ❌ |
| **With ROI cap (2°)** | 2° × 2° | ~3s | ~5s | ~8s ✅ |
| **+ Minimal analysis** | 2° × 2° | ~1.8s | ~5s | ~7s ✅ |
| **+ 4x downsample** | 2° × 2° | ~1.8s | ~1.2s | ~3s ✅✅ |
| **+ DEM cache (2nd call)** | 2° × 2° | ~0s (cached) | ~1.2s | ~1s ✅✅✅ |

---

## Implementation Plan

### Phase 1: Emergency Fix (5 minutes)
1. Add `MAX_ROI_SPAN_DEG = 2.0` check in `navigation_engine.py`
2. Add frontend distance validation in `NavigationPlanning.tsx`

### Phase 2: Core Optimizations (30 minutes)
1. Implement minimal terrain analysis mode
2. Increase adaptive downsampling thresholds
3. Add configuration for max ROI span

### Phase 3: Advanced Optimizations (1 hour)
1. Implement DEM/metrics caching
2. Add corridor-based ROI
3. Add progress updates to frontend

---

## Configuration Additions

Add to `marshab/config.py`:

```python
@dataclass
class NavigationConfig:
    # ... existing fields ...
    
    # Performance tuning
    max_roi_span_deg: float = 2.0  # Maximum navigation distance
    enable_dem_cache: bool = True  # Cache DEM/metrics
    adaptive_downsample: bool = True  # Use adaptive downsampling
    min_downsample_factor: int = 1  # Minimum downsample
    max_downsample_factor: int = 4  # Maximum downsample
```

---

## Testing Recommendations

After implementing optimizations, test with:

1. **Close range** (0.2° span): Should be < 5 seconds
2. **Medium range** (1.0° span): Should be < 15 seconds
3. **Max range** (2.0° span): Should be < 30 seconds
4. **Too far** (> 2.0° span): Should fail instantly with clear error

---

## Alternative: Async Background Jobs

For very long routes, consider:
- Submit navigation as background job
- Return task_id immediately
- Poll for completion via WebSocket or endpoint
- Show progress bar in UI

This would require changes to the API pattern but would support arbitrarily long computations.
