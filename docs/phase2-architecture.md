# Phase 2: Navigation & Pathfinding - Architecture & Implementation Plan

**Duration:** 2-3 weeks  
**Priority:** HIGH  
**Goal:** Implement rover navigation planning with A* pathfinding and coordinate transformations

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Navigation System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  Analysis    │─────▶│  Cost Map    │─────▶│    A*     │ │
│  │  Results     │      │  Generator   │      │ Pathfinder│ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                     │                      │       │
│         │                     │                      │       │
│         ▼                     ▼                      ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │    Sites     │      │ Traversability│      │ Waypoint  │ │
│  │   Database   │      │   Costs       │      │   Path    │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│                                │                      │       │
│                                │                      │       │
│                                ▼                      ▼       │
│                         ┌──────────────┐      ┌───────────┐ │
│                         │  Coordinate  │◀────│  SITE     │ │
│                         │ Transformer  │     │  Frame    │ │
│                         └──────────────┘      └───────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input:** Start site, goal site, terrain analysis results
2. **Cost Surface Generation:** Combine slope, roughness into traversability cost
3. **Pathfinding:** A* algorithm finds optimal path considering terrain costs
4. **Waypoint Sampling:** Downsample path to manageable waypoints
5. **Coordinate Transform:** Convert waypoints to SITE frame (North, East, Down)
6. **Output:** Waypoint CSV with SITE frame coordinates

---

## Task 1: Pathfinding Integration

### Objective
Complete the `NavigationEngine.plan_to_site()` method to generate rover navigation paths.

### Current State
- `pathfinding.py` has A* implementation but not integrated
- `navigation_engine.py` returns empty waypoint DataFrame
- Cost surface generation exists but incomplete

### Implementation Details

#### 1.1 Cost Surface Enhancement

**File:** `marshab/processing/terrain.py` (modify)

```python
def generate_cost_surface(
    slope: np.ndarray,
    roughness: np.ndarray,
    elevation: Optional[np.ndarray] = None,
    max_slope_deg: float = 25.0,
    max_roughness: float = 1.0,
    elevation_penalty_factor: float = 0.1
) -> np.ndarray:
    """Generate comprehensive traversability cost surface.
    
    Args:
        slope: Slope array in degrees
        roughness: Roughness array
        elevation: Optional elevation for altitude penalties
        max_slope_deg: Maximum traversable slope
        max_roughness: Maximum traversable roughness
        elevation_penalty_factor: Penalty factor for elevation gain
        
    Returns:
        Cost surface (1.0 = baseline, higher = more difficult, inf = impassable)
    """
    logger.info("Generating traversability cost surface")
    
    # Initialize base cost
    cost = np.ones_like(slope, dtype=np.float32)
    
    # Slope cost (exponential increase)
    # Cost doubles every 10 degrees
    slope_normalized = slope / max_slope_deg
    slope_cost = np.exp(slope_normalized * 2.0) - 1.0
    cost += slope_cost
    
    # Roughness cost (linear to quadratic)
    roughness_normalized = roughness / max_roughness
    roughness_cost = roughness_normalized ** 2 * 5.0
    cost += roughness_cost
    
    # Elevation cost (if provided)
    if elevation is not None:
        # Penalize uphill travel
        dy, dx = np.gradient(elevation)
        elevation_gradient = np.sqrt(dx**2 + dy**2)
        elevation_cost = elevation_gradient * elevation_penalty_factor
        cost += elevation_cost
    
    # Mark impassable areas
    cost[slope > max_slope_deg] = np.inf
    cost[roughness > max_roughness] = np.inf
    cost[np.isnan(slope) | np.isnan(roughness)] = np.inf
    
    # Statistics
    passable_fraction = np.sum(np.isfinite(cost)) / cost.size
    
    logger.info(
        "Cost surface generated",
        passable_fraction=float(passable_fraction),
        mean_cost=float(np.nanmean(cost[np.isfinite(cost)])),
        max_finite_cost=float(np.nanmax(cost[np.isfinite(cost)]))
    )
    
    return cost
```

#### 1.2 NavigationEngine Implementation

**File:** `marshab/core/navigation_engine.py` (major modification)

```python
"""Navigation planning engine for rover pathfinding."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import xarray as xr

from marshab.processing.pathfinding import AStarPathfinder
from marshab.processing.terrain import generate_cost_surface
from marshab.processing.coordinates import CoordinateTransformer
from marshab.types import SiteOrigin, Waypoint
from marshab.exceptions import NavigationError
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class NavigationEngine:
    """Plans rover navigation routes using terrain-aware pathfinding."""
    
    def __init__(self):
        """Initialize navigation engine."""
        self.transformer = CoordinateTransformer()
    
    def _load_analysis_results(self, analysis_dir: Path) -> dict:
        """Load terrain analysis results from directory.
        
        Args:
            analysis_dir: Directory containing analysis outputs
            
        Returns:
            Dictionary with dem, metrics, sites
        """
        import pickle
        
        results_file = analysis_dir / "analysis_results.pkl"
        if not results_file.exists():
            raise NavigationError(
                f"Analysis results not found: {results_file}"
            )
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def _get_site_coordinates(
        self, 
        site_id: int,
        sites: list
    ) -> Tuple[float, float]:
        """Get coordinates of site by ID.
        
        Args:
            site_id: Site identifier
            sites: List of Site objects
            
        Returns:
            (latitude, longitude) tuple
        """
        for site in sites:
            if site.site_id == site_id:
                return (site.centroid_lat, site.centroid_lon)
        
        raise NavigationError(f"Site {site_id} not found")
    
    def _latlon_to_pixel(
        self,
        lat: float,
        lon: float,
        dem: xr.DataArray
    ) -> Tuple[int, int]:
        """Convert lat/lon to pixel coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            dem: DEM DataArray with coordinate info
            
        Returns:
            (row, col) pixel coordinates
        """
        # Find nearest pixel
        lat_values = dem.coords['lat'].values
        lon_values = dem.coords['lon'].values
        
        # Handle 2D coordinate arrays
        if lat_values.ndim == 2:
            # Find minimum distance
            dist = (lat_values - lat)**2 + (lon_values - lon)**2
            row, col = np.unravel_index(np.argmin(dist), lat_values.shape)
        else:
            # 1D coordinate arrays
            row = np.argmin(np.abs(lat_values - lat))
            col = np.argmin(np.abs(lon_values - lon))
        
        return int(row), int(col)
    
    def _pixel_to_latlon(
        self,
        row: int,
        col: int,
        dem: xr.DataArray
    ) -> Tuple[float, float]:
        """Convert pixel coordinates to lat/lon.
        
        Args:
            row: Pixel row
            col: Pixel column
            dem: DEM DataArray
            
        Returns:
            (latitude, longitude) tuple
        """
        lat_values = dem.coords['lat'].values
        lon_values = dem.coords['lon'].values
        
        if lat_values.ndim == 2:
            lat = float(lat_values[row, col])
            lon = float(lon_values[row, col])
        else:
            lat = float(lat_values[row])
            lon = float(lon_values[col])
        
        return lat, lon
    
    def plan_to_site(
        self,
        site_id: int,
        analysis_dir: Path,
        start_lat: float,
        start_lon: float,
        max_waypoint_spacing_m: float = 100.0,
        max_slope_deg: float = 25.0
    ) -> pd.DataFrame:
        """Plan navigation route to target site.
        
        Args:
            site_id: Target site ID
            analysis_dir: Directory with analysis results
            start_lat: Starting latitude
            start_lon: Starting longitude
            max_waypoint_spacing_m: Maximum spacing between waypoints
            max_slope_deg: Maximum traversable slope
            
        Returns:
            DataFrame with waypoint columns: waypoint_id, x_site, y_site, tolerance_m
        """
        logger.info(
            "Planning route to site",
            site_id=site_id,
            start_lat=start_lat,
            start_lon=start_lon
        )
        
        # Load analysis results
        results = self._load_analysis_results(analysis_dir)
        dem = results['dem']
        metrics = results['metrics']
        sites = results['sites']
        
        # Get goal site coordinates
        goal_lat, goal_lon = self._get_site_coordinates(site_id, sites)
        
        logger.info(
            "Route parameters",
            goal_lat=goal_lat,
            goal_lon=goal_lon
        )
        
        # Generate cost surface
        cost_map = generate_cost_surface(
            metrics.slope,
            metrics.roughness,
            elevation=metrics.elevation,
            max_slope_deg=max_slope_deg
        )
        
        # Convert start/goal to pixel coordinates
        start_pixel = self._latlon_to_pixel(start_lat, start_lon, dem)
        goal_pixel = self._latlon_to_pixel(goal_lat, goal_lon, dem)
        
        logger.info(
            "Pixel coordinates",
            start_pixel=start_pixel,
            goal_pixel=goal_pixel
        )
        
        # Run A* pathfinding
        resolution_m = float(dem.attrs.get('resolution_m', 200.0))
        pathfinder = AStarPathfinder(cost_map, cell_size_m=resolution_m)
        
        # Calculate waypoint spacing in pixels
        max_spacing_pixels = int(max_waypoint_spacing_m / resolution_m)
        
        try:
            waypoint_pixels = pathfinder.find_path_with_waypoints(
                start_pixel,
                goal_pixel,
                max_waypoint_spacing=max_spacing_pixels
            )
        except NavigationError as e:
            logger.error("Pathfinding failed", error=str(e))
            raise
        
        logger.info(f"Found path with {len(waypoint_pixels)} waypoints")
        
        # Convert waypoints to lat/lon
        waypoints_latlon = []
        for row, col in waypoint_pixels:
            lat, lon = self._pixel_to_latlon(row, col, dem)
            elev = float(metrics.elevation[row, col])
            waypoints_latlon.append((lat, lon, elev))
        
        # Define SITE frame origin at start position
        start_elevation = float(
            metrics.elevation[start_pixel[0], start_pixel[1]]
        )
        site_origin = SiteOrigin(
            lat=start_lat,
            lon=start_lon,
            elevation_m=start_elevation
        )
        
        # Transform waypoints to SITE frame
        waypoints_site = []
        for i, (lat, lon, elev) in enumerate(waypoints_latlon):
            x, y, z = self.transformer.iau_mars_to_site_frame(
                lat, lon, elev, site_origin
            )
            
            waypoints_site.append({
                'waypoint_id': i + 1,
                'x_site': x,  # North (meters)
                'y_site': y,  # East (meters)
                'z_site': z,  # Down (meters)
                'latitude': lat,
                'longitude': lon,
                'elevation_m': elev,
                'tolerance_m': max_waypoint_spacing_m / 2.0
            })
        
        # Create DataFrame
        waypoints_df = pd.DataFrame(waypoints_site)
        
        logger.info(
            "Navigation plan complete",
            num_waypoints=len(waypoints_df),
            total_distance_m=float(
                np.sqrt(waypoints_df['x_site'].iloc[-1]**2 + 
                       waypoints_df['y_site'].iloc[-1]**2)
            )
        )
        
        return waypoints_df
    
    def save_waypoints(
        self,
        waypoints: pd.DataFrame,
        output_path: Path
    ) -> None:
        """Save waypoints to CSV file.
        
        Args:
            waypoints: Waypoint DataFrame
            output_path: Output file path
        """
        waypoints.to_csv(output_path, index=False)
        logger.info(f"Waypoints saved to {output_path}")
```

#### 1.3 Analysis Results Persistence

**File:** `marshab/core/analysis_pipeline.py` (add method)

```python
def save_results(self, results: AnalysisResults, output_dir: Path) -> None:
    """Save analysis results for later use.
    
    Args:
        results: AnalysisResults object
        output_dir: Output directory
    """
    import pickle
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete results as pickle
    results_file = output_dir / "analysis_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'dem': results.dem,
            'metrics': results.metrics,
            'sites': results.sites,
            'suitability': results.suitability,
            'criteria': results.criteria
        }, f)
    
    # Save sites as GeoJSON
    sites_geojson = {
        "type": "FeatureCollection",
        "features": [site.to_geojson_feature() for site in results.sites]
    }
    
    import json
    sites_file = output_dir / "sites.geojson"
    with open(sites_file, 'w') as f:
        json.dump(sites_geojson, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
```

### Testing Strategy

**File:** `tests/unit/test_navigation_engine.py`

```python
"""Unit tests for navigation engine."""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from marshab.core.navigation_engine import NavigationEngine
from marshab.types import SiteOrigin

class TestNavigationEngine:
    """Tests for NavigationEngine."""
    
    @pytest.fixture
    def engine(self):
        return NavigationEngine()
    
    @pytest.fixture
    def mock_dem(self):
        """Create mock DEM with coordinates."""
        data = np.random.randn(100, 100) * 100 + 1000
        lat = np.linspace(40.0, 41.0, 100)
        lon = np.linspace(180.0, 181.0, 100)
        
        dem = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={
                'lat': (['y'], lat),
                'lon': (['x'], lon)
            },
            attrs={'resolution_m': 200.0}
        )
        return dem
    
    def test_latlon_to_pixel(self, engine, mock_dem):
        """Test coordinate conversion."""
        lat, lon = 40.5, 180.5
        row, col = engine._latlon_to_pixel(lat, lon, mock_dem)
        
        assert 0 <= row < 100
        assert 0 <= col < 100
    
    def test_pixel_to_latlon(self, engine, mock_dem):
        """Test reverse coordinate conversion."""
        row, col = 50, 50
        lat, lon = engine._pixel_to_latlon(row, col, mock_dem)
        
        assert 40.0 <= lat <= 41.0
        assert 180.0 <= lon <= 181.0
```

---

## Task 2: Coordinate Transformations

### Objective
Complete IAU_MARS to SITE frame coordinate transformations for waypoint export.

### Current State
- `coordinates.py` has basic transformation framework
- Needs validation and error handling
- Missing batch transformation methods

### Implementation Details

#### 2.1 Enhanced Coordinate Transformer

**File:** `marshab/processing/coordinates.py` (modify)

```python
"""Coordinate transformations between Mars reference frames."""

from typing import Tuple, List, Optional
import numpy as np
from numpy.typing import NDArray

from marshab.exceptions import CoordinateError
from marshab.types import SiteOrigin
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class CoordinateTransformer:
    """Transforms between Mars coordinate reference frames."""
    
    def __init__(
        self,
        equatorial_radius: float = 3396190.0,
        polar_radius: float = 3376200.0,
    ):
        """Initialize coordinate transformer with Mars ellipsoid parameters.
        
        Args:
            equatorial_radius: Mars equatorial radius (meters)
            polar_radius: Mars polar radius (meters)
        """
        self.eq_radius = equatorial_radius
        self.pol_radius = polar_radius
        self.flattening = (equatorial_radius - polar_radius) / equatorial_radius
        self.eccentricity_sq = 2 * self.flattening - self.flattening ** 2
        
        logger.debug(
            "CoordinateTransformer initialized",
            eq_radius=equatorial_radius,
            pol_radius=polar_radius,
            flattening=self.flattening
        )
    
    def validate_coordinates(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float
    ) -> None:
        """Validate coordinate values.
        
        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            elevation_m: Elevation in meters
            
        Raises:
            CoordinateError: If coordinates are invalid
        """
        if not -90 <= lat_deg <= 90:
            raise CoordinateError(
                f"Latitude out of range: {lat_deg}",
                details={"valid_range": [-90, 90]}
            )
        
        if not 0 <= lon_deg <= 360:
            raise CoordinateError(
                f"Longitude out of range: {lon_deg}",
                details={"valid_range": [0, 360]}
            )
        
        # Check elevation is reasonable
        max_elevation = 30000.0  # Olympus Mons ~21km
        min_elevation = -10000.0  # Hellas Basin ~7km
        
        if not min_elevation <= elevation_m <= max_elevation:
            logger.warning(
                "Elevation outside typical range",
                elevation_m=elevation_m,
                typical_range=[min_elevation, max_elevation]
            )
    
    def planetocentric_to_cartesian(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
    ) -> Tuple[float, float, float]:
        """Convert planetocentric coordinates to Cartesian.
        
        Uses IAU_MARS reference ellipsoid with specified radii.
        
        Args:
            lat_deg: Planetocentric latitude (degrees)
            lon_deg: East-positive longitude (degrees, 0-360)
            elevation_m: Elevation above reference ellipsoid (meters)
            
        Returns:
            (x, y, z) in Mars body-fixed frame (meters)
        """
        self.validate_coordinates(lat_deg, lon_deg, elevation_m)
        
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)
        
        # Calculate radius at this latitude (ellipsoid)
        N = self.eq_radius / np.sqrt(
            1 - self.eccentricity_sq * sin_lat**2
        )
        
        # Add elevation
        radius = N + elevation_m
        
        # Convert to Cartesian
        x = radius * cos_lat * cos_lon
        y = radius * cos_lat * sin_lon
        z = (N * (1 - self.eccentricity_sq) + elevation_m) * sin_lat
        
        return float(x), float(y), float(z)
    
    def iau_mars_to_site_frame(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
        site_origin: SiteOrigin,
    ) -> Tuple[float, float, float]:
        """Transform IAU_MARS coordinates to rover SITE frame.
        
        SITE frame definition:
        - Origin: site_origin location
        - +X axis: North
        - +Y axis: East
        - +Z axis: Down
        
        Args:
            lat_deg: Target latitude (degrees)
            lon_deg: Target longitude (degrees, 0-360)
            elevation_m: Target elevation (meters)
            site_origin: SITE frame origin definition
            
        Returns:
            (x, y, z) in SITE frame (meters) - North, East, Down
            
        Raises:
            CoordinateError: If transformation fails
        """
        try:
            # Convert both points to Cartesian
            target_xyz = self.planetocentric_to_cartesian(
                lat_deg, lon_deg, elevation_m
            )
            origin_xyz = self.planetocentric_to_cartesian(
                site_origin.lat,
                site_origin.lon,
                site_origin.elevation_m
            )
            
            # Calculate offset vector in body-fixed frame
            dx = target_xyz[0] - origin_xyz[0]
            dy = target_xyz[1] - origin_xyz[1]
            dz = target_xyz[2] - origin_xyz[2]
            
            # Rotation matrix: Mars body-fixed → Local NED
            lat_rad = np.radians(site_origin.lat)
            lon_rad = np.radians(site_origin.lon)
            
            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            sin_lon = np.sin(lon_rad)
            cos_lon = np.cos(lon_rad)
            
            # Rotation matrix (3x3)
            R = np.array([
                [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
                [-sin_lon,            cos_lon,             0      ],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
            ])
            
            # Apply rotation
            offset_body = np.array([dx, dy, dz])
            offset_ned = R @ offset_body
            
            x_site = float(offset_ned[0])  # North
            y_site = float(offset_ned[1])  # East
            z_site = float(offset_ned[2])  # Down
            
            logger.debug(
                "Transformed to SITE frame",
                target_lat=lat_deg,
                target_lon=lon_deg,
                site_x=x_site,
                site_y=y_site,
                site_z=z_site,
                distance_m=float(np.linalg.norm(offset_ned))
            )
            
            return x_site, y_site, z_site
            
        except Exception as e:
            raise CoordinateError(
                "Failed to transform coordinates to SITE frame",
                details={
                    "lat": lat_deg,
                    "lon": lon_deg,
                    "elevation": elevation_m,
                    "origin": site_origin.model_dump(),
                    "error": str(e)
                }
            )
    
    def batch_transform_to_site(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
        elevations: NDArray[np.float64],
        site_origin: SiteOrigin
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Transform multiple points to SITE frame efficiently.
        
        Args:
            lats: Array of latitudes
            lons: Array of longitudes
            elevations: Array of elevations
            site_origin: SITE frame origin
            
        Returns:
            (x_array, y_array, z_array) in SITE frame
        """
        n = len(lats)
        x_site = np.zeros(n)
        y_site = np.zeros(n)
        z_site = np.zeros(n)
        
        for i in range(n):
            x_site[i], y_site[i], z_site[i] = self.iau_mars_to_site_frame(
                lats[i], lons[i], elevations[i], site_origin
            )
        
        return x_site, y_site, z_site
```

### Testing

**File:** `tests/unit/test_coordinates_enhanced.py`

```python
"""Enhanced tests for coordinate transformations."""

import pytest
import numpy as np
from marshab.processing.coordinates import CoordinateTransformer
from marshab.types import SiteOrigin
from marshab.exceptions import CoordinateError

class TestCoordinateTransformerEnhanced:
    """Enhanced coordinate transformation tests."""
    
    @pytest.fixture
    def transformer(self):
        return CoordinateTransformer()
    
    def test_coordinate_validation(self, transformer):
        """Test coordinate validation."""
        # Valid coordinates
        transformer.validate_coordinates(0, 0, 0)
        
        # Invalid latitude
        with pytest.raises(CoordinateError):
            transformer.validate_coordinates(91, 0, 0)
        
        # Invalid longitude
        with pytest.raises(CoordinateError):
            transformer.validate_coordinates(0, 361, 0)
    
    def test_round_trip_transformation(self, transformer):
        """Test transformation accuracy."""
        site = SiteOrigin(lat=0, lon=0, elevation_m=0)
        
        # Point 1km north
        lat = 0.009  # ~1km at equator
        lon = 0.0
        
        x, y, z = transformer.iau_mars_to_site_frame(
            lat, lon, 0, site
        )
        
        # Should be approximately 1000m north
        assert 900 < x < 1100
        assert abs(y) < 100
    
    def test_batch_transformation(self, transformer):
        """Test batch transformation."""
        site = SiteOrigin(lat=40, lon=180, elevation_m=-2500)
        
        lats = np.array([40.1, 40.2, 40.3])
        lons = np.array([180.1, 180.2, 180.3])
        elevs = np.array([-2500, -2500, -2500])
        
        x, y, z = transformer.batch_transform_to_site(
            lats, lons, elevs, site
        )
        
        assert len(x) == 3
        assert len(y) == 3
        assert len(z) == 3
        assert np.all(x > 0)  # All points north
        assert np.all(y > 0)  # All points east
```

---

## Task 3: Navigation API Endpoints

### Objective
Complete navigation API endpoints for route planning and waypoint visualization.

### Implementation

**File:** `marshab/web/routes/navigation.py` (modify)

```python
"""Navigation API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import pandas as pd

from marshab.core.navigation_engine import NavigationEngine
from marshab.exceptions import NavigationError
from marshab.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/navigation", tags=["navigation"])


class NavigationRequest(BaseModel):
    """Request for navigation planning."""
    site_id: int = Field(..., description="Target site ID")
    analysis_dir: str = Field(..., description="Analysis results directory")
    start_lat: float = Field(..., ge=-90, le=90)
    start_lon: float = Field(..., ge=0, le=360)
    max_waypoint_spacing_m: float = Field(100.0, gt=0)
    max_slope_deg: float = Field(25.0, gt=0, le=90)


class NavigationResponse(BaseModel):
    """Response with navigation waypoints."""
    waypoints: List[dict]
    num_waypoints: int
    total_distance_m: float
    site_id: int


@router.post("/plan-route", response_model=NavigationResponse)
async def plan_route(request: NavigationRequest):
    """Plan navigation route to target site.
    
    Returns waypoints in SITE frame (North, East, Down).
    """
    try:
        engine = NavigationEngine()
        
        waypoints_df = engine.plan_to_site(
            site_id=request.site_id,
            analysis_dir=Path(request.analysis_dir),
            start_lat=request.start_lat,
            start_lon=request.start_lon,
            max_waypoint_spacing_m=request.max_waypoint_spacing_m,
            max_slope_deg=request.max_slope_deg
        )
        
        # Calculate total distance
        if len(waypoints_df) > 0:
            last_waypoint = waypoints_df.iloc[-1]
            total_distance = float(
                (last_waypoint['x_site']**2 + last_waypoint['y_site']**2)**0.5
            )
        else:
            total_distance = 0.0
        
        return NavigationResponse(
            waypoints=waypoints_df.to_dict('records'),
            num_waypoints=len(waypoints_df),
            total_distance_m=total_distance,
            site_id=request.site_id
        )
        
    except NavigationError as e:
        logger.error("Navigation planning failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error in navigation planning", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/waypoints-geojson")
async def get_waypoints_geojson(
    analysis_dir: str = Query(..., description="Analysis directory"),
    site_id: int = Query(..., description="Site ID")
):
    """Get waypoints as GeoJSON for visualization."""
    try:
        waypoints_file = Path(analysis_dir) / f"waypoints_site_{site_id}.csv"
        
        if not waypoints_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Waypoints not found for site {site_id}"
            )
        
        waypoints_df = pd.read_csv(waypoints_file)
        
        # Convert to GeoJSON
        features = []
        coordinates = []
        
        for _, row in waypoints_df.iterrows():
            # Point feature for each waypoint
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['longitude'], row['latitude']]
                },
                "properties": {
                    "waypoint_id": int(row['waypoint_id']),
                    "x_site": float(row['x_site']),
                    "y_site": float(row['y_site']),
                    "tolerance_m": float(row['tolerance_m'])
                }
            })
            coordinates.append([row['longitude'], row['latitude']])
        
        # Add LineString for path
        if len(coordinates) > 1:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "path_type": "navigation_route",
                    "site_id": site_id
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
        
    except Exception as e:
        logger.error("Failed to generate waypoints GeoJSON", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Task 4: Testing & Integration

### Integration Tests

**File:** `tests/integration/test_navigation_pipeline.py`

```python
"""Integration tests for navigation pipeline."""

import pytest
from pathlib import Path
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.navigation_engine import NavigationEngine
from marshab.types import BoundingBox

class TestNavigationPipeline:
    """End-to-end navigation tests."""
    
    @pytest.fixture
    def analysis_results(self, tmp_path):
        """Run analysis and save results."""
        pipeline = AnalysisPipeline()
        roi = BoundingBox(
            lat_min=40.0, lat_max=40.5,
            lon_min=180.0, lon_max=180.5
        )
        
        results = pipeline.run(roi, dataset="mola", threshold=0.6)
        
        output_dir = tmp_path / "analysis"
        pipeline.save_results(results, output_dir)
        
        return output_dir, results
    
    def test_complete_navigation_workflow(self, analysis_results):
        """Test full workflow from analysis to waypoints."""
        output_dir, results = analysis_results
        
        # Get first site
        site = results.sites[0]
        
        # Plan navigation
        engine = NavigationEngine()
        waypoints = engine.plan_to_site(
            site_id=site.site_id,
            analysis_dir=output_dir,
            start_lat=40.1,
            start_lon=180.1,
            max_waypoint_spacing_m=100.0
        )
        
        # Validate waypoints
        assert len(waypoints) > 0
        assert 'waypoint_id' in waypoints.columns
        assert 'x_site' in waypoints.columns
        assert 'y_site' in waypoints.columns
        
        # First waypoint should be near origin
        assert abs(waypoints.iloc[0]['x_site']) < 50
        assert abs(waypoints.iloc[0]['y_site']) < 50
        
        # Last waypoint should be far from origin
        last = waypoints.iloc[-1]
        distance = (last['x_site']**2 + last['y_site']**2)**0.5
        assert distance > 1000  # At least 1km
```

---

## Deliverables Checklist

- [ ] Cost surface generation enhanced
- [ ] `NavigationEngine.plan_to_site()` fully implemented
- [ ] Coordinate transformer validated
- [ ] Batch transformation methods added
- [ ] Navigation API endpoints complete
- [ ] Waypoint CSV export working
- [ ] GeoJSON visualization endpoints functional
- [ ] Unit tests with 80%+ coverage
- [ ] Integration tests passing
- [ ] Documentation updated

---

## Success Metrics

1. ✅ Generate navigation routes successfully
2. ✅ Waypoints in valid SITE frame coordinates
3. ✅ Paths avoid impassable terrain
4. ✅ Coordinate transformations accurate to <1m
5. ✅ Process 1° x 1° region in < 60 seconds
6. ✅ API endpoints return valid responses
7. ✅ All tests passing
