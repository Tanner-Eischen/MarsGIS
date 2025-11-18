# Phase 1: Core Analysis Pipeline - Architecture & Implementation Plan

**Duration:** 3-4 weeks  
**Priority:** HIGH  
**Goal:** Implement functional terrain analysis and site scoring algorithms

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Analysis Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ DataManager  │─────▶│   Terrain    │─────▶│   MCDM    │ │
│  │              │      │   Analyzer   │      │ Evaluator │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                     │                      │       │
│         │                     │                      │       │
│         ▼                     ▼                      ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  DEM Cache   │      │   Terrain    │      │   Sites   │ │
│  │              │      │   Metrics    │      │  GeoJSON  │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input:** ROI (BoundingBox) + Dataset selection + Criteria weights
2. **DEM Acquisition:** DataManager downloads/loads DEM for ROI
3. **Terrain Analysis:** TerrainAnalyzer computes slope, aspect, roughness, TRI
4. **Criteria Scoring:** MCDMEvaluator normalizes and weights criteria
5. **Site Identification:** Generate candidate site polygons from high-suitability regions
6. **Output:** Ranked sites with scores, GeoJSON, analysis summary

---

## Task 1: Terrain Analysis Implementation

### Objective
Complete the `TerrainAnalyzer.analyze()` method to produce real terrain metrics from DEM data.

### Current State
- `terrain.py` has method signatures but returns empty/stub results
- Basic gradient calculations exist but not integrated
- No hillshade calculation

### Implementation Details

#### 1.1 Slope Calculation Enhancement

**File:** `marshab/processing/terrain.py`

**Algorithm:**
```
slope = arctan(sqrt(dz/dx² + dz/dy²))
```

**Implementation:**
```python
def calculate_slope(self, dem: np.ndarray) -> np.ndarray:
    """Calculate slope magnitude in degrees using gradient method."""
    dy, dx = np.gradient(dem, self.cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Handle edge effects
    slope_deg = np.nan_to_num(slope_deg, nan=0.0)
    slope_deg = np.clip(slope_deg, 0, 90)
    
    return slope_deg.astype(np.float32)
```

**Validation:**
- Test with synthetic DEM (known gradient)
- Verify edge case handling (flat terrain, steep cliffs)
- Compare against GDAL slope calculation

#### 1.2 Aspect Calculation

**Algorithm:**
```
aspect = atan2(-dx, dy) converted to 0-360° from North
```

**Implementation:**
```python
def calculate_aspect(self, dem: np.ndarray) -> np.ndarray:
    """Calculate aspect (slope direction) in degrees from North."""
    dy, dx = np.gradient(dem, self.cell_size_m)
    
    # Calculate aspect from gradients
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad)
    
    # Convert to 0-360 range (0 = North, 90 = East)
    aspect_deg = (aspect_deg + 360) % 360
    
    # Mark flat areas as undefined (-1)
    flat_threshold = 0.1  # degrees
    slope_deg = self.calculate_slope(dem)
    aspect_deg = np.where(slope_deg < flat_threshold, -1, aspect_deg)
    
    return aspect_deg.astype(np.float32)
```

#### 1.3 Roughness Calculation

**Algorithm:** Standard deviation in moving window

**Implementation:**
```python
def calculate_roughness(self, dem: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Calculate terrain roughness using local standard deviation."""
    from scipy import ndimage
    
    def local_std(values):
        return np.std(values)
    
    roughness = ndimage.generic_filter(
        dem, 
        local_std, 
        size=window_size,
        mode='reflect'
    )
    
    return roughness.astype(np.float32)
```

#### 1.4 Terrain Ruggedness Index (TRI)

**Algorithm:** Mean absolute difference from neighbors

**Implementation:**
```python
def calculate_tri(self, dem: np.ndarray) -> np.ndarray:
    """Calculate Terrain Ruggedness Index."""
    from scipy import ndimage
    
    # 3x3 kernel (8 neighbors)
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=float) / 8.0
    
    # Mean of neighbors
    mean_neighbors = ndimage.convolve(dem, kernel, mode='reflect')
    
    # Absolute difference
    tri = np.abs(dem - mean_neighbors)
    
    return tri.astype(np.float32)
```

#### 1.5 Hillshade for Visualization

**New Method:**
```python
def calculate_hillshade(
    self, 
    dem: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0
) -> np.ndarray:
    """Calculate hillshade for visualization.
    
    Args:
        dem: Elevation array
        azimuth: Sun azimuth in degrees (0=North, 90=East)
        altitude: Sun altitude angle in degrees
    
    Returns:
        Hillshade values 0-255
    """
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # Calculate gradients
    dy, dx = np.gradient(dem, self.cell_size_m)
    
    # Slope and aspect
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dx, dy)
    
    # Hillshade calculation
    hillshade = (
        np.cos(altitude_rad) * np.cos(slope_rad) +
        np.sin(altitude_rad) * np.sin(slope_rad) *
        np.cos(azimuth_rad - aspect_rad)
    )
    
    # Scale to 0-255
    hillshade = np.clip(hillshade * 255, 0, 255)
    
    return hillshade.astype(np.uint8)
```

#### 1.6 Complete analyze() Method

**File:** `marshab/processing/terrain.py`

```python
def analyze(self, dem: xr.DataArray) -> TerrainMetrics:
    """Perform complete terrain analysis on DEM.
    
    Args:
        dem: Input DEM DataArray with lat/lon coordinates
        
    Returns:
        TerrainMetrics with all calculated products
    """
    logger.info("Starting terrain analysis", shape=dem.shape)
    
    # Extract elevation array
    elevation = dem.values.astype(np.float32)
    
    # Handle nodata values
    nodata = dem.attrs.get('nodata', -9999)
    elevation = np.where(elevation == nodata, np.nan, elevation)
    
    # Calculate all metrics
    logger.info("Calculating slope...")
    slope = self.calculate_slope(elevation)
    
    logger.info("Calculating aspect...")
    aspect = self.calculate_aspect(elevation)
    
    logger.info("Calculating roughness...")
    roughness = self.calculate_roughness(elevation, window_size=5)
    
    logger.info("Calculating TRI...")
    tri = self.calculate_tri(elevation)
    
    logger.info("Calculating hillshade...")
    hillshade = self.calculate_hillshade(elevation)
    
    # Log statistics
    logger.info(
        "Terrain analysis complete",
        slope_mean=float(np.nanmean(slope)),
        slope_max=float(np.nanmax(slope)),
        slope_std=float(np.nanstd(slope)),
        roughness_mean=float(np.nanmean(roughness)),
        tri_mean=float(np.nanmean(tri))
    )
    
    return TerrainMetrics(
        slope=slope,
        aspect=aspect,
        roughness=roughness,
        tri=tri,
        hillshade=hillshade,
        elevation=elevation
    )
```

### Testing Strategy

**File:** `tests/unit/test_terrain_complete.py`

```python
import pytest
import numpy as np
import xarray as xr
from marshab.processing.terrain import TerrainAnalyzer

class TestTerrainAnalyzerComplete:
    """Comprehensive tests for terrain analysis."""
    
    @pytest.fixture
    def flat_dem(self):
        """Flat terrain for baseline testing."""
        data = np.ones((100, 100)) * 1000.0
        return xr.DataArray(data, dims=['y', 'x'])
    
    @pytest.fixture
    def sloped_dem(self):
        """Uniformly sloped terrain."""
        x = np.arange(100)
        y = np.arange(100)
        data = np.outer(y, np.ones_like(x)) * 10.0  # 10m per pixel
        return xr.DataArray(data, dims=['y', 'x'])
    
    def test_flat_terrain_analysis(self, flat_dem):
        """Test analysis of flat terrain."""
        analyzer = TerrainAnalyzer(cell_size_m=100.0)
        metrics = analyzer.analyze(flat_dem)
        
        # Flat terrain should have near-zero slope
        assert np.nanmax(metrics.slope) < 1.0
        assert np.nanmean(metrics.roughness) < 0.1
        assert np.nanmean(metrics.tri) < 0.1
    
    def test_sloped_terrain_analysis(self, sloped_dem):
        """Test analysis of sloped terrain."""
        analyzer = TerrainAnalyzer(cell_size_m=100.0)
        metrics = analyzer.analyze(sloped_dem)
        
        # Should detect consistent slope
        expected_slope = np.degrees(np.arctan(10.0 / 100.0))
        assert np.abs(np.nanmean(metrics.slope) - expected_slope) < 0.5
        
        # Aspect should be consistent (northward)
        valid_aspect = metrics.aspect[metrics.aspect >= 0]
        assert np.abs(np.nanmean(valid_aspect) - 0.0) < 10.0
    
    def test_hillshade_calculation(self, sloped_dem):
        """Test hillshade generation."""
        analyzer = TerrainAnalyzer()
        elevation = sloped_dem.values
        
        hillshade = analyzer.calculate_hillshade(elevation, azimuth=315, altitude=45)
        
        assert hillshade.shape == elevation.shape
        assert hillshade.dtype == np.uint8
        assert np.min(hillshade) >= 0
        assert np.max(hillshade) <= 255
```

### Performance Targets
- Process 1° x 1° MOLA DEM (~200 x 200 km) in < 5 seconds
- Memory usage < 2GB for typical ROI
- Support DEMs up to 5000 x 5000 pixels

---

## Task 2: MCDM Integration

### Objective
Connect MCDM algorithms to the analysis pipeline for site scoring.

### Architecture

```
Terrain Metrics ──┐
                  │
Elevation Data ───┼──▶ Criteria Extraction ──▶ Normalization ──▶ Weighted Sum/TOPSIS ──▶ Suitability Score
                  │
Configuration ────┘
```

### Implementation Details

#### 2.1 Criteria Configuration System

**File:** `marshab/config/criteria_config.py` (new)

```python
"""Criteria configuration for MCDM analysis."""

from typing import Dict, Literal
from pydantic import BaseModel, Field

class Criterion(BaseModel):
    """Single criterion definition."""
    name: str
    display_name: str
    description: str
    beneficial: bool = True  # True if higher values are better
    weight: float = Field(ge=0.0, le=1.0)
    source: Literal["terrain", "derived", "external"]
    
class CriteriaConfig(BaseModel):
    """Complete criteria configuration."""
    criteria: Dict[str, Criterion]
    
    def validate_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(c.weight for c in self.criteria.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

# Default configuration
DEFAULT_CRITERIA = CriteriaConfig(
    criteria={
        "slope": Criterion(
            name="slope",
            display_name="Slope Safety",
            description="Lower slopes are safer for landing",
            beneficial=False,
            weight=0.30,
            source="terrain"
        ),
        "roughness": Criterion(
            name="roughness",
            display_name="Surface Roughness",
            description="Smoother surfaces are preferred",
            beneficial=False,
            weight=0.25,
            source="terrain"
        ),
        "elevation": Criterion(
            name="elevation",
            display_name="Elevation",
            description="Lower elevations have better atmospheric density",
            beneficial=False,
            weight=0.20,
            source="terrain"
        ),
        "solar_exposure": Criterion(
            name="solar_exposure",
            display_name="Solar Exposure",
            description="Higher solar exposure for power generation",
            beneficial=True,
            weight=0.15,
            source="derived"
        ),
        "science_value": Criterion(
            name="science_value",
            display_name="Science Value",
            description="Proximity to features of scientific interest",
            beneficial=True,
            weight=0.10,
            source="external"
        )
    }
)
```

#### 2.2 Criteria Extraction

**File:** `marshab/processing/criteria.py` (new)

```python
"""Extract and prepare criteria from terrain analysis."""

import numpy as np
import xarray as xr
from typing import Dict
from marshab.types import TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

class CriteriaExtractor:
    """Extracts criteria values from terrain analysis results."""
    
    def __init__(self, dem: xr.DataArray, metrics: TerrainMetrics):
        self.dem = dem
        self.metrics = metrics
    
    def extract_slope_criterion(self) -> np.ndarray:
        """Extract slope values as cost criterion."""
        return self.metrics.slope
    
    def extract_roughness_criterion(self) -> np.ndarray:
        """Extract roughness values as cost criterion."""
        return self.metrics.roughness
    
    def extract_elevation_criterion(self) -> np.ndarray:
        """Extract elevation values as cost criterion."""
        return self.metrics.elevation
    
    def calculate_solar_exposure(self) -> np.ndarray:
        """Calculate solar exposure based on slope and aspect.
        
        Assumes equatorial region with optimal sun from north.
        """
        slope_rad = np.radians(self.metrics.slope)
        aspect_rad = np.radians(self.metrics.aspect)
        
        # North-facing slopes get maximum exposure
        # Flat areas also good
        exposure = np.cos(slope_rad) + 0.5 * np.cos(aspect_rad) * np.sin(slope_rad)
        exposure = np.clip(exposure, 0, 1)
        
        return exposure
    
    def calculate_science_value(self) -> np.ndarray:
        """Placeholder for science value criterion.
        
        In real implementation, would use proximity to:
        - Crater features
        - Mineral deposits
        - Water ice signatures
        """
        # Uniform for now - to be replaced with actual science data
        return np.ones_like(self.metrics.slope) * 0.5
    
    def extract_all(self) -> Dict[str, np.ndarray]:
        """Extract all criteria into dictionary."""
        logger.info("Extracting criteria from terrain analysis")
        
        criteria = {
            "slope": self.extract_slope_criterion(),
            "roughness": self.extract_roughness_criterion(),
            "elevation": self.extract_elevation_criterion(),
            "solar_exposure": self.calculate_solar_exposure(),
            "science_value": self.calculate_science_value()
        }
        
        logger.info(
            "Criteria extracted",
            num_criteria=len(criteria),
            criteria_names=list(criteria.keys())
        )
        
        return criteria
```

#### 2.3 Updated MCDM Evaluator

**File:** `marshab/processing/mcdm.py` (modify)

Add method for complete evaluation:

```python
@staticmethod
def evaluate(
    criteria: Dict[str, np.ndarray],
    weights: Dict[str, float],
    beneficial: Dict[str, bool],
    method: Literal["weighted_sum", "topsis"] = "weighted_sum"
) -> np.ndarray:
    """Evaluate suitability using specified MCDM method.
    
    Args:
        criteria: Criterion name -> values array
        weights: Criterion name -> weight
        beneficial: Criterion name -> benefit direction
        method: MCDM method to use
        
    Returns:
        Suitability score array [0, 1]
    """
    logger.info(f"Evaluating suitability using {method}")
    
    if method == "weighted_sum":
        return MCDMEvaluator.weighted_sum(criteria, weights, beneficial)
    elif method == "topsis":
        return MCDMEvaluator.topsis(criteria, weights, beneficial)
    else:
        raise ValueError(f"Unknown MCDM method: {method}")
```

#### 2.4 Integration with AnalysisPipeline

**File:** `marshab/core/analysis_pipeline.py` (major modification)

```python
def run(
    self,
    roi: BoundingBox,
    dataset: Literal["mola", "hirise", "ctx"] = "mola",
    threshold: float = 0.7,
    criteria_weights: Optional[Dict[str, float]] = None,
    mcdm_method: Literal["weighted_sum", "topsis"] = "weighted_sum"
) -> AnalysisResults:
    """Run complete analysis pipeline.
    
    Args:
        roi: Region of interest
        dataset: DEM dataset to use
        threshold: Minimum suitability threshold
        criteria_weights: Optional custom weights (uses defaults if None)
        mcdm_method: MCDM method to use
        
    Returns:
        AnalysisResults with sites and suitability raster
    """
    logger.info(
        "Starting analysis pipeline",
        roi=roi.model_dump(),
        dataset=dataset,
        threshold=threshold
    )
    
    # Step 1: Load DEM
    logger.info("Loading DEM data")
    dem = self.data_manager.get_dem_for_roi(
        roi, 
        dataset=dataset,
        download=True,
        clip=True
    )
    
    # Step 2: Terrain analysis
    logger.info("Analyzing terrain")
    metrics = self.terrain_analyzer.analyze(dem)
    
    # Step 3: Extract criteria
    logger.info("Extracting criteria")
    from marshab.processing.criteria import CriteriaExtractor
    extractor = CriteriaExtractor(dem, metrics)
    criteria = extractor.extract_all()
    
    # Step 4: Configure weights
    from marshab.config.criteria_config import DEFAULT_CRITERIA
    config = DEFAULT_CRITERIA
    
    if criteria_weights:
        # Update weights with custom values
        for name, weight in criteria_weights.items():
            if name in config.criteria:
                config.criteria[name].weight = weight
    
    config.validate_weights()
    
    weights = {name: c.weight for name, c in config.criteria.items()}
    beneficial = {name: c.beneficial for name, c in config.criteria.items()}
    
    # Step 5: MCDM evaluation
    logger.info("Evaluating suitability")
    suitability = MCDMEvaluator.evaluate(
        criteria,
        weights,
        beneficial,
        method=mcdm_method
    )
    
    # Step 6: Identify candidate sites
    logger.info("Identifying candidate sites")
    sites = self._identify_sites(suitability, dem, threshold)
    
    # Step 7: Rank sites
    sites = self._rank_sites(sites)
    
    logger.info(
        "Analysis complete",
        num_sites=len(sites),
        mean_suitability=float(np.nanmean(suitability)),
        max_suitability=float(np.nanmax(suitability))
    )
    
    return AnalysisResults(
        sites=sites,
        suitability=suitability,
        dem=dem,
        metrics=metrics,
        criteria=criteria
    )
```

### Testing Strategy

**File:** `tests/integration/test_mcdm_integration.py`

```python
class TestMCDMIntegration:
    """Integration tests for MCDM pipeline."""
    
    def test_full_pipeline_with_real_data(self, sample_mars_dem):
        """Test complete pipeline from DEM to sites."""
        from marshab.core.analysis_pipeline import AnalysisPipeline
        from marshab.types import BoundingBox
        
        pipeline = AnalysisPipeline()
        roi = BoundingBox(lat_min=40.0, lat_max=41.0, 
                         lon_min=180.0, lon_max=181.0)
        
        results = pipeline.run(roi, dataset="mola", threshold=0.7)
        
        # Should have sites
        assert len(results.sites) > 0
        
        # Suitability should be in valid range
        assert np.nanmin(results.suitability) >= 0
        assert np.nanmax(results.suitability) <= 1
        
        # Sites should have required attributes
        for site in results.sites:
            assert site.suitability_score >= 0.7
            assert hasattr(site, 'geometry')
            assert hasattr(site, 'rank')
```

---

## Task 3: Site Identification

### Objective
Generate candidate site polygons from suitability raster.

### Algorithm

1. **Threshold**: Identify pixels above suitability threshold
2. **Connectivity**: Group connected high-suitability pixels
3. **Size Filter**: Eliminate regions below minimum area
4. **Polygon Generation**: Convert pixel regions to geospatial polygons
5. **Statistics**: Calculate site statistics and attributes

### Implementation

**File:** `marshab/core/analysis_pipeline.py`

```python
def _identify_sites(
    self,
    suitability: np.ndarray,
    dem: xr.DataArray,
    threshold: float
) -> List[Site]:
    """Identify candidate sites from suitability raster.
    
    Args:
        suitability: Suitability score array
        dem: Original DEM
        threshold: Minimum suitability threshold
        
    Returns:
        List of Site objects
    """
    from scipy import ndimage
    from shapely.geometry import shape
    from rasterio import features
    
    logger.info("Identifying candidate sites", threshold=threshold)
    
    # Create binary mask of suitable pixels
    suitable_mask = (suitability >= threshold) & ~np.isnan(suitability)
    
    # Label connected components
    labeled, num_features = ndimage.label(suitable_mask)
    
    logger.info(f"Found {num_features} candidate regions")
    
    # Get pixel size for area calculation
    resolution_m = float(dem.attrs.get('resolution_m', 200.0))
    min_pixels = int((0.5 * 1e6) / (resolution_m ** 2))  # 0.5 km²
    
    sites = []
    
    for region_id in range(1, num_features + 1):
        region_mask = (labeled == region_id)
        region_size = np.sum(region_mask)
        
        # Filter by minimum size
        if region_size < min_pixels:
            continue
        
        # Calculate site statistics
        region_suitability = suitability[region_mask]
        region_elevation = dem.values[region_mask]
        region_lat = dem.coords['lat'].values[region_mask]
        region_lon = dem.coords['lon'].values[region_mask]
        
        # Create site polygon
        # Transform to GeoJSON-compatible format
        shapes_gen = features.shapes(
            region_mask.astype(np.uint8),
            transform=rasterio.transform.from_bounds(
                float(np.min(region_lon)),
                float(np.min(region_lat)),
                float(np.max(region_lon)),
                float(np.max(region_lat)),
                region_mask.shape[1],
                region_mask.shape[0]
            )
        )
        
        # Get first polygon
        geom_dict, _ = next(shapes_gen)
        geometry = shape(geom_dict)
        
        # Create Site object
        site = Site(
            site_id=region_id,
            geometry=geometry,
            suitability_score=float(np.mean(region_suitability)),
            suitability_std=float(np.std(region_suitability)),
            mean_elevation_m=float(np.mean(region_elevation)),
            centroid_lat=float(np.mean(region_lat)),
            centroid_lon=float(np.mean(region_lon)),
            area_km2=region_size * (resolution_m ** 2) / 1e6,
            rank=0  # Will be set by ranking
        )
        
        sites.append(site)
    
    logger.info(f"Identified {len(sites)} sites above threshold")
    
    return sites

def _rank_sites(self, sites: List[Site]) -> List[Site]:
    """Rank sites by suitability score.
    
    Args:
        sites: List of Site objects
        
    Returns:
        Ranked list of sites
    """
    # Sort by suitability score (descending)
    sites.sort(key=lambda s: s.suitability_score, reverse=True)
    
    # Assign ranks
    for rank, site in enumerate(sites, start=1):
        site.rank = rank
    
    return sites
```

### Site Type Definition

**File:** `marshab/types.py` (add)

```python
from shapely.geometry import Polygon
from pydantic import BaseModel

class Site(BaseModel):
    """Candidate landing site."""
    
    site_id: int
    geometry: Polygon
    suitability_score: float
    suitability_std: float
    mean_elevation_m: float
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    rank: int
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_geojson_feature(self) -> dict:
        """Convert to GeoJSON feature."""
        return {
            "type": "Feature",
            "geometry": self.geometry.__geo_interface__,
            "properties": {
                "site_id": self.site_id,
                "rank": self.rank,
                "suitability_score": self.suitability_score,
                "suitability_std": self.suitability_std,
                "mean_elevation_m": self.mean_elevation_m,
                "centroid_lat": self.centroid_lat,
                "centroid_lon": self.centroid_lon,
                "area_km2": self.area_km2
            }
        }
```

---

## Task 4: Testing & Validation

### Test Coverage Requirements
- Unit tests: 80%+ coverage
- Integration tests: All major workflows
- Performance tests: Benchmark datasets

### Test Files

**Create: `tests/integration/test_analysis_pipeline_complete.py`**

```python
"""Complete integration tests for analysis pipeline."""

import pytest
import numpy as np
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.types import BoundingBox

class TestAnalysisPipelineComplete:
    """End-to-end pipeline tests."""
    
    @pytest.fixture
    def pipeline(self):
        return AnalysisPipeline()
    
    def test_complete_analysis_workflow(self, pipeline):
        """Test full workflow from ROI to ranked sites."""
        roi = BoundingBox(
            lat_min=40.0, lat_max=40.5,
            lon_min=180.0, lon_max=180.5
        )
        
        results = pipeline.run(
            roi,
            dataset="mola",
            threshold=0.6
        )
        
        # Validate results
        assert results.sites is not None
        assert len(results.sites) > 0
        assert results.suitability is not None
        
        # Check site properties
        for site in results.sites:
            assert site.suitability_score >= 0.6
            assert site.rank > 0
            assert site.area_km2 > 0
        
        # Check ranking
        scores = [s.suitability_score for s in results.sites]
        assert scores == sorted(scores, reverse=True)
    
    def test_custom_weights(self, pipeline):
        """Test analysis with custom criteria weights."""
        roi = BoundingBox(lat_min=40.0, lat_max=40.5,
                         lon_min=180.0, lon_max=180.5)
        
        custom_weights = {
            "slope": 0.5,  # Emphasize slope safety
            "roughness": 0.3,
            "elevation": 0.2
        }
        
        results = pipeline.run(
            roi,
            criteria_weights=custom_weights,
            threshold=0.5
        )
        
        assert len(results.sites) > 0
```

---

## Deliverables Checklist

- [ ] `TerrainAnalyzer.analyze()` fully implemented
- [ ] Hillshade calculation added
- [ ] `CriteriaExtractor` class created
- [ ] `CriteriaConfig` system implemented
- [ ] MCDM integration complete
- [ ] Site identification algorithm working
- [ ] Site ranking functional
- [ ] Unit tests with 80%+ coverage
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Code documentation complete
- [ ] PR submitted and reviewed

---

## Risk Mitigation

### Risk: Performance Issues with Large DEMs
**Mitigation:** Implement chunked processing with Dask if needed

### Risk: Memory Overflow
**Mitigation:** Add memory monitoring, implement lazy loading

### Risk: Edge Effects in Terrain Analysis
**Mitigation:** Use proper boundary handling (reflect, wrap), validate edge pixels

---

## Success Metrics

1. ✅ Pipeline produces valid site candidates
2. ✅ Suitability scores in [0, 1] range
3. ✅ Sites ranked correctly by score
4. ✅ Process 1° x 1° region in < 30 seconds
5. ✅ Memory usage < 2GB
6. ✅ 80%+ test coverage
7. ✅ All integration tests passing
