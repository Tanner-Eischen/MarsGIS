# MarsHab Site Selector - Code Examples & Remaining Phases

## Phase 2: Data Management (Days 4-5 - 8 hours) [CONTINUED]

### 2.1 DEM Loader Module (2 hours)

**Create `marshab/processing/dem_loader.py`**:
```python
"""DEM loading and raster operations."""

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr
from shapely.geometry import box

from marshab.exceptions import DataError
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class DEMLoader:
    """Handles Mars DEM loading and preprocessing."""
    
    def __init__(self, mars_radius_eq: float = 3396190.0):
        """Initialize DEM loader.
        
        Args:
            mars_radius_eq: Mars equatorial radius in meters
        """
        self.mars_radius_eq = mars_radius_eq
    
    def load(self, path: Path) -> xr.DataArray:
        """Load DEM into xarray DataArray.
        
        Args:
            path: Path to GeoTIFF DEM file
        
        Returns:
            xarray DataArray with elevation data and coordinates
        
        Raises:
            DataError: If DEM cannot be loaded
        """
        try:
            with rasterio.open(path) as src:
                # Read elevation data
                elevation = src.read(1)
                
                # Create coordinate arrays
                height, width = elevation.shape
                transform = src.transform
                
                # Calculate lat/lon coordinates for each pixel
                rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                lons, lats = rasterio.transform.xy(transform, rows, cols)
                
                # Create xarray DataArray
                dem = xr.DataArray(
                    elevation,
                    dims=['y', 'x'],
                    coords={
                        'lat': (['y', 'x'], np.array(lats)),
                        'lon': (['y', 'x'], np.array(lons)),
                    },
                    attrs={
                        'crs': str(src.crs),
                        'transform': transform,
                        'nodata': src.nodata,
                        'resolution_m': abs(transform.a),
                    }
                )
                
                logger.info(
                    "Loaded DEM",
                    path=str(path),
                    shape=elevation.shape,
                    resolution_m=abs(transform.a),
                    crs=str(src.crs)
                )
                
                return dem
                
        except Exception as e:
            raise DataError(
                f"Failed to load DEM: {path}",
                details={"path": str(path), "error": str(e)}
            )
    
    def clip_to_roi(
        self,
        dem: xr.DataArray,
        roi: BoundingBox
    ) -> xr.DataArray:
        """Clip DEM to region of interest.
        
        Args:
            dem: Input DEM DataArray
            roi: Bounding box to clip to
        
        Returns:
            Clipped DEM DataArray
        """
        # Create mask based on lat/lon coordinates
        lat_mask = (dem.coords['lat'] >= roi.lat_min) & (dem.coords['lat'] <= roi.lat_max)
        lon_mask = (dem.coords['lon'] >= roi.lon_min) & (dem.coords['lon'] <= roi.lon_max)
        mask_combined = lat_mask & lon_mask
        
        # Apply mask
        clipped = dem.where(mask_combined, drop=True)
        
        logger.info(
            "Clipped DEM to ROI",
            original_shape=dem.shape,
            clipped_shape=clipped.shape,
            roi=roi.dict()
        )
        
        return clipped
    
    def resample(
        self,
        dem: xr.DataArray,
        target_resolution_m: float,
        method: Resampling = Resampling.bilinear
    ) -> xr.DataArray:
        """Resample DEM to different resolution.
        
        Args:
            dem: Input DEM DataArray
            target_resolution_m: Target resolution in meters
            method: Resampling method
        
        Returns:
            Resampled DEM DataArray
        """
        current_res = dem.attrs.get('resolution_m', 200.0)
        
        if abs(current_res - target_resolution_m) < 1.0:
            logger.info("DEM already at target resolution")
            return dem
        
        scale_factor = current_res / target_resolution_m
        new_height = int(dem.shape[0] * scale_factor)
        new_width = int(dem.shape[1] * scale_factor)
        
        logger.info(
            "Resampling DEM",
            from_resolution=current_res,
            to_resolution=target_resolution_m,
            from_shape=dem.shape,
            to_shape=(new_height, new_width)
        )
        
        # Use scipy for resampling (simplified approach)
        from scipy.ndimage import zoom
        
        resampled_data = zoom(dem.values, scale_factor, order=1)
        
        # Create new coordinate arrays
        # (Simplified - in production use proper affine transforms)
        resampled = xr.DataArray(
            resampled_data,
            dims=['y', 'x'],
            attrs={**dem.attrs, 'resolution_m': target_resolution_m}
        )
        
        return resampled


### 2.2 Data Manager Service (2 hours)

**Create `marshab/core/data_manager.py`**:
```python
"""Data acquisition and management service."""

import hashlib
import shutil
from pathlib import Path
from typing import Literal
import urllib.request

from marshab.config import get_config
from marshab.exceptions import DataError
from marshab.processing.dem_loader import DEMLoader
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class DataManager:
    """Manages Mars DEM data acquisition, caching, and loading."""
    
    def __init__(self):
        """Initialize data manager."""
        self.config = get_config()
        self.loader = DEMLoader(
            mars_radius_eq=self.config.mars.equatorial_radius_m
        )
        self.cache_dir = self.config.paths.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, dataset: str, roi: BoundingBox) -> Path:
        """Generate cache file path for dataset and ROI.
        
        Args:
            dataset: Dataset identifier
            roi: Region of interest
        
        Returns:
            Path to cached file
        """
        cache_key = hashlib.md5(
            f"{dataset}_{roi.lat_min}_{roi.lat_max}_{roi.lon_min}_{roi.lon_max}".encode()
        ).hexdigest()
        
        return self.cache_dir / f"{dataset}_{cache_key}.tif"
    
    def download_dem(
        self,
        dataset: Literal["mola", "hirise", "ctx"],
        roi: BoundingBox,
        force: bool = False
    ) -> Path:
        """Download Mars DEM covering specified ROI.
        
        Args:
            dataset: Dataset to download ("mola", "hirise", or "ctx")
            roi: Region of interest to cover
            force: Force re-download even if cached
        
        Returns:
            Path to downloaded/cached DEM file
        
        Raises:
            DataError: If download fails
        """
        cache_path = self._get_cache_path(dataset, roi)
        
        # Check cache
        if cache_path.exists() and not force:
            logger.info(
                "Using cached DEM",
                dataset=dataset,
                path=str(cache_path)
            )
            return cache_path
        
        # Get data source URL
        if dataset not in self.config.data_sources:
            raise DataError(
                f"Unknown dataset: {dataset}",
                details={"available": list(self.config.data_sources.keys())}
            )
        
        source = self.config.data_sources[dataset]
        url = source.url
        
        logger.info(
            "Downloading DEM",
            dataset=dataset,
            url=url,
            roi=roi.dict()
        )
        
        try:
            # Download to temporary file
            temp_path = cache_path.with_suffix('.tmp')
            
            # Use urllib for simple downloads
            # In production, use requests with progress bar
            urllib.request.urlretrieve(url, temp_path)
            
            # Move to final location
            shutil.move(str(temp_path), str(cache_path))
            
            logger.info(
                "Downloaded DEM successfully",
                dataset=dataset,
                path=str(cache_path)
            )
            
            return cache_path
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            
            raise DataError(
                f"Failed to download DEM: {dataset}",
                details={"url": url, "error": str(e)}
            )
    
    def load_dem(self, path: Path):
        """Load DEM from file path."""
        return self.loader.load(path)
    
    def get_dem_for_roi(
        self,
        roi: BoundingBox,
        dataset: Literal["mola", "hirise", "ctx"] = "mola",
        download: bool = True
    ):
        """Get DEM covering ROI, downloading if necessary.
        
        Args:
            roi: Region of interest
            dataset: Dataset to use
            download: Whether to download if not cached
        
        Returns:
            DEM DataArray clipped to ROI
        """
        if download:
            dem_path = self.download_dem(dataset, roi)
        else:
            dem_path = self._get_cache_path(dataset, roi)
            
            if not dem_path.exists():
                raise DataError(
                    f"DEM not found in cache: {dataset}",
                    details={"path": str(dem_path)}
                )
        
        # Load and clip to ROI
        dem = self.loader.load(dem_path)
        clipped = self.loader.clip_to_roi(dem, roi)
        
        return clipped
```

### 2.3 Unit Tests for Data Management (1 hour)

**Create `tests/unit/test_dem_loader.py`**:
```python
"""Unit tests for DEM loader."""

import numpy as np
import pytest

from marshab.processing.dem_loader import DEMLoader
from marshab.types import BoundingBox


def test_load_dem(synthetic_dem):
    """Test DEM loading."""
    loader = DEMLoader()
    dem = loader.load(synthetic_dem)
    
    assert dem is not None
    assert 'lat' in dem.coords
    assert 'lon' in dem.coords
    assert dem.shape == (100, 100)


def test_clip_to_roi(synthetic_dem):
    """Test ROI clipping."""
    loader = DEMLoader()
    dem = loader.load(synthetic_dem)
    
    roi = BoundingBox(
        lat_min=40.2,
        lat_max=40.8,
        lon_min=180.2,
        lon_max=180.8
    )
    
    clipped = loader.clip_to_roi(dem, roi)
    
    # Clipped should be smaller
    assert clipped.shape[0] < dem.shape[0]
    assert clipped.shape[1] < dem.shape[1]
    
    # All coordinates should be within ROI
    assert clipped.coords['lat'].min() >= roi.lat_min
    assert clipped.coords['lat'].max() <= roi.lat_max


def test_resample(synthetic_dem):
    """Test DEM resampling."""
    loader = DEMLoader()
    dem = loader.load(synthetic_dem)
    
    # Downsample to lower resolution
    resampled = loader.resample(dem, target_resolution_m=1000.0)
    
    # Should have fewer pixels
    assert resampled.shape[0] < dem.shape[0]
    assert resampled.shape[1] < dem.shape[1]
```

**Run tests**:
```bash
poetry run pytest tests/unit/test_dem_loader.py -v
```

---

## Phase 3: Terrain Analysis (Days 6-7 - 8 hours)

### 3.1 Terrain Analytics Module (3 hours)

**Create `marshab/processing/terrain.py`**:
```python
"""Terrain analysis functions."""

from typing import Tuple

import numpy as np
from scipy import ndimage
import xarray as xr

from marshab.types import TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class TerrainAnalyzer:
    """Performs terrain analysis on Mars DEMs."""
    
    def __init__(self, cell_size_m: float = 200.0):
        """Initialize terrain analyzer.
        
        Args:
            cell_size_m: DEM cell size in meters
        """
        self.cell_size_m = cell_size_m
    
    def calculate_slope(self, dem: np.ndarray) -> np.ndarray:
        """Calculate slope in degrees.
        
        Args:
            dem: Input elevation array
        
        Returns:
            Slope array in degrees
        """
        # Calculate gradients
        dy, dx = np.gradient(dem, self.cell_size_m)
        
        # Calculate slope magnitude
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        logger.debug(
            "Calculated slope",
            mean_slope=float(np.mean(slope)),
            max_slope=float(np.max(slope))
        )
        
        return slope
    
    def calculate_aspect(self, dem: np.ndarray) -> np.ndarray:
        """Calculate aspect (direction of slope) in degrees from North.
        
        Args:
            dem: Input elevation array
        
        Returns:
            Aspect array in degrees (0-360, 0=North)
        """
        # Calculate gradients
        dy, dx = np.gradient(dem, self.cell_size_m)
        
        # Calculate aspect (direction of maximum slope)
        # atan2 gives angle from east, convert to angle from north
        aspect = np.degrees(np.arctan2(-dx, dy))
        
        # Convert to 0-360 range
        aspect = (aspect + 360) % 360
        
        return aspect
    
    def calculate_roughness(
        self,
        dem: np.ndarray,
        window_size: int = 3
    ) -> np.ndarray:
        """Calculate terrain roughness (standard deviation of elevation).
        
        Args:
            dem: Input elevation array
            window_size: Window size for roughness calculation
        
        Returns:
            Roughness array
        """
        # Use generic filter to calculate local standard deviation
        roughness = ndimage.generic_filter(
            dem,
            np.std,
            size=window_size,
            mode='reflect'
        )
        
        logger.debug(
            "Calculated roughness",
            mean_roughness=float(np.mean(roughness)),
            window_size=window_size
        )
        
        return roughness
    
    def calculate_tri(self, dem: np.ndarray) -> np.ndarray:
        """Calculate Terrain Ruggedness Index.
        
        TRI is the mean absolute difference between center cell and neighbors.
        
        Args:
            dem: Input elevation array
        
        Returns:
            TRI array
        """
        # Create kernel for neighbors
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]) / 8.0
        
        # Convolve to get mean of neighbors
        mean_neighbors = ndimage.convolve(dem, kernel, mode='reflect')
        
        # TRI is absolute difference from neighbors
        tri = np.abs(dem - mean_neighbors)
        
        return tri
    
    def analyze(self, dem: xr.DataArray) -> TerrainMetrics:
        """Perform complete terrain analysis.
        
        Args:
            dem: Input DEM DataArray
        
        Returns:
            TerrainMetrics with all calculated products
        """
        logger.info("Starting terrain analysis", shape=dem.shape)
        
        elevation = dem.values
        
        # Calculate all metrics
        slope = self.calculate_slope(elevation)
        aspect = self.calculate_aspect(elevation)
        roughness = self.calculate_roughness(elevation)
        tri = self.calculate_tri(elevation)
        
        logger.info(
            "Terrain analysis complete",
            slope_mean=float(np.mean(slope)),
            roughness_mean=float(np.mean(roughness))
        )
        
        return TerrainMetrics(
            slope=slope,
            aspect=aspect,
            roughness=roughness,
            tri=tri
        )


def generate_cost_surface(
    slope: np.ndarray,
    roughness: np.ndarray,
    max_slope: float = 25.0
) -> np.ndarray:
    """Generate traversability cost surface for pathfinding.
    
    Args:
        slope: Slope array in degrees
        roughness: Roughness array
        max_slope: Maximum traversable slope (degrees)
    
    Returns:
        Cost surface (higher = more difficult/impossible)
    """
    # Base cost of 1.0
    cost = np.ones_like(slope)
    
    # Add slope cost (exponential)
    cost += (slope / 45.0) ** 2 * 10.0
    
    # Add roughness cost (normalized)
    roughness_norm = roughness / np.max(roughness)
    cost += roughness_norm * 5.0
    
    # Mark impassable areas
    cost[slope > max_slope] = np.inf
    
    return cost
```

### 3.2 MCDM Implementation (2 hours)

**Create `marshab/processing/mcdm.py`**:
```python
"""Multi-Criteria Decision Making (MCDM) implementation."""

from typing import Dict

import numpy as np

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class MCDMEvaluator:
    """Multi-criteria decision making using weighted sum and TOPSIS."""
    
    def normalize_criterion(
        self,
        data: np.ndarray,
        beneficial: bool = True
    ) -> np.ndarray:
        """Normalize criterion to 0-1 range.
        
        Args:
            data: Input criterion array
            beneficial: True if higher values are better, False if lower is better
        
        Returns:
            Normalized array (0-1 range)
        """
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        
        if data_max == data_min:
            return np.ones_like(data) * 0.5
        
        if beneficial:
            # Higher is better
            normalized = (data - data_min) / (data_max - data_min)
        else:
            # Lower is better (cost criterion)
            normalized = (data_max - data) / (data_max - data_min)
        
        return normalized
    
    def weighted_sum(
        self,
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool]
    ) -> np.ndarray:
        """Calculate weighted sum of normalized criteria.
        
        Args:
            criteria: Dictionary of criterion name -> array
            weights: Dictionary of criterion name -> weight (must sum to 1.0)
            beneficial: Dictionary of criterion name -> True if beneficial
        
        Returns:
            Suitability score array (0-1 range)
        """
        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Normalize all criteria
        normalized = {}
        for name, data in criteria.items():
            is_beneficial = beneficial.get(name, True)
            normalized[name] = self.normalize_criterion(data, is_beneficial)
        
        # Calculate weighted sum
        suitability = np.zeros_like(list(criteria.values())[0])
        
        for name, norm_data in normalized.items():
            weight = weights.get(name, 0.0)
            suitability += norm_data * weight
            
            logger.debug(
                f"Applied criterion: {name}",
                weight=weight,
                mean_value=float(np.mean(norm_data))
            )
        
        logger.info(
            "Weighted sum complete",
            mean_suitability=float(np.mean(suitability)),
            max_suitability=float(np.max(suitability))
        )
        
        return suitability
    
    def topsis(
        self,
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool]
    ) -> np.ndarray:
        """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
        
        Args:
            criteria: Dictionary of criterion name -> array
            weights: Dictionary of criterion name -> weight
            beneficial: Dictionary of criterion name -> True if beneficial
        
        Returns:
            TOPSIS score array (0-1 range, higher is better)
        """
        # Normalize criteria
        normalized = {}
        for name, data in criteria.items():
            # Vector normalization for TOPSIS
            norm = np.sqrt(np.sum(data**2))
            normalized[name] = data / norm if norm > 0 else data
        
        # Apply weights
        weighted = {}
        for name, norm_data in normalized.items():
            weight = weights.get(name, 0.0)
            weighted[name] = norm_data * weight
        
        # Determine ideal best and worst
        ideal_best = {}
        ideal_worst = {}
        
        for name, w_data in weighted.items():
            is_beneficial = beneficial.get(name, True)
            
            if is_beneficial:
                ideal_best[name] = np.max(w_data)
                ideal_worst[name] = np.min(w_data)
            else:
                ideal_best[name] = np.min(w_data)
                ideal_worst[name] = np.max(w_data)
        
        # Calculate distances to ideal best and worst
        dist_best = np.zeros_like(list(weighted.values())[0])
        dist_worst = np.zeros_like(list(weighted.values())[0])
        
        for name, w_data in weighted.items():
            dist_best += (w_data - ideal_best[name])**2
            dist_worst += (w_data - ideal_worst[name])**2
        
        dist_best = np.sqrt(dist_best)
        dist_worst = np.sqrt(dist_worst)
        
        # Calculate TOPSIS score
        # Avoid division by zero
        denominator = dist_best + dist_worst
        topsis_score = np.where(
            denominator > 0,
            dist_worst / denominator,
            0.5
        )
        
        logger.info(
            "TOPSIS complete",
            mean_score=float(np.mean(topsis_score)),
            max_score=float(np.max(topsis_score))
        )
        
        return topsis_score
```

### 3.3 Unit Tests for Terrain Analysis (1 hour)

**Create `tests/unit/test_terrain.py`**:
```python
"""Unit tests for terrain analysis."""

import numpy as np
import pytest

from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface


def test_calculate_slope():
    """Test slope calculation."""
    # Create simple sloped surface
    dem = np.tile(np.arange(50), (50, 1)).T  # Constant slope in one direction
    
    analyzer = TerrainAnalyzer(cell_size_m=100.0)
    slope = analyzer.calculate_slope(dem.astype(float))
    
    assert slope.shape == dem.shape
    assert np.all(slope >= 0)
    assert np.all(slope <= 90)
    
    # Slope should be roughly constant (not exactly due to edge effects)
    assert np.std(slope[5:-5, 5:-5]) < 1.0


def test_calculate_aspect():
    """Test aspect calculation."""
    dem = np.random.randn(50, 50) * 10
    
    analyzer = TerrainAnalyzer()
    aspect = analyzer.calculate_aspect(dem)
    
    assert aspect.shape == dem.shape
    assert np.all(aspect >= 0)
    assert np.all(aspect <= 360)


def test_calculate_roughness():
    """Test roughness calculation."""
    # Smooth surface
    smooth = np.ones((50, 50)) * 100.0
    
    # Rough surface
    rough = np.random.randn(50, 50) * 50 + 100.0
    
    analyzer = TerrainAnalyzer()
    
    roughness_smooth = analyzer.calculate_roughness(smooth)
    roughness_rough = analyzer.calculate_roughness(rough)
    
    # Rough surface should have higher roughness
    assert np.mean(roughness_rough) > np.mean(roughness_smooth)


def test_terrain_analyzer_full(sample_terrain_data):
    """Test complete terrain analysis."""
    analyzer = TerrainAnalyzer(cell_size_m=200.0)
    
    import xarray as xr
    dem = xr.DataArray(sample_terrain_data)
    
    metrics = analyzer.analyze(dem)
    
    assert metrics.slope.shape == sample_terrain_data.shape
    assert metrics.aspect.shape == sample_terrain_data.shape
    assert metrics.roughness.shape == sample_terrain_data.shape
    assert metrics.tri.shape == sample_terrain_data.shape


def test_generate_cost_surface():
    """Test cost surface generation."""
    slope = np.random.rand(50, 50) * 30  # 0-30 degrees
    roughness = np.random.rand(50, 50) * 0.5
    
    cost = generate_cost_surface(slope, roughness, max_slope=25.0)
    
    # High slopes should be impassable
    assert np.all(np.isinf(cost[slope > 25.0]))
    
    # Other areas should have finite cost
    assert np.all(np.isfinite(cost[slope <= 25.0]))
```

**Run tests**:
```bash
poetry run pytest tests/unit/test_terrain.py -v
```

---

## Phase 4: Coordinate Transformations & Navigation (Days 8-9 - 8 hours)

### 4.1 Coordinate Transform Module (3 hours)

**Create `marshab/processing/coordinates.py`**:
```python
"""Coordinate transformations for Mars reference frames."""

from typing import Tuple

import numpy as np
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    import warnings
    warnings.warn("SpiceyPy not available, using simplified transforms")

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
        use_spice: bool = True
    ):
        """Initialize coordinate transformer.
        
        Args:
            equatorial_radius: Mars equatorial radius (meters)
            polar_radius: Mars polar radius (meters)
            use_spice: Whether to use SPICE toolkit if available
        """
        self.eq_radius = equatorial_radius
        self.pol_radius = polar_radius
        self.use_spice = use_spice and SPICE_AVAILABLE
        
        if not self.use_spice:
            logger.warning("Using simplified coordinate transforms (SPICE not available)")
    
    def planetocentric_to_cartesian(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float
    ) -> Tuple[float, float, float]:
        """Convert planetocentric lat/lon to Cartesian coordinates.
        
        Args:
            lat_deg: Planetocentric latitude (degrees)
            lon_deg: East positive longitude (degrees)
            elevation_m: Elevation above datum (meters)
        
        Returns:
            (x, y, z) in Mars body-fixed frame (meters)
        """
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        
        # Calculate radius at this latitude (ellipsoid)
        # r = sqrt((a^2 * cos(lat))^2 + (b^2 * sin(lat))^2) / sqrt((a*cos(lat))^2 + (b*sin(lat))^2)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        
        numerator = np.sqrt(
            (self.eq_radius**2 * cos_lat)**2 + (self.pol_radius**2 * sin_lat)**2
        )
        denominator = np.sqrt(
            (self.eq_radius * cos_lat)**2 + (self.pol_radius * sin_lat)**2
        )
        
        radius = numerator / denominator + elevation_m
        
        # Convert to Cartesian
        x = radius * cos_lat * np.cos(lon_rad)
        y = radius * cos_lat * np.sin(lon_rad)
        z = radius * sin_lat
        
        return x, y, z
    
    def iau_mars_to_site_frame(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
        site_origin: SiteOrigin
    ) -> Tuple[float, float, float]:
        """Transform IAU_MARS coordinates to rover SITE frame.
        
        SITE frame definition:
        - Origin: site_origin location on Mars surface
        - +X axis: North
        - +Y axis: East
        - +Z axis: Down (nadir)
        
        Args:
            lat_deg: Target latitude (degrees, planetocentric)
            lon_deg: Target longitude (degrees, east positive)
            elevation_m: Target elevation (meters above datum)
            site_origin: SITE frame origin definition
        
        Returns:
            (x, y, z) in SITE frame (meters)
        """
        # Convert both points to Cartesian
        target_xyz = self.planetocentric_to_cartesian(lat_deg, lon_deg, elevation_m)
        origin_xyz = self.planetocentric_to_cartesian(
            site_origin.lat,
            site_origin.lon,
            site_origin.elevation_m
        )
        
        # Calculate offset vector
        dx = target_xyz[0] - origin_xyz[0]
        dy = target_xyz[1] - origin_xyz[1]
        dz = target_xyz[2] - origin_xyz[2]
        
        # Build rotation matrix from Mars-fixed to local NED frame
        # (Simplified approach - for production use proper DCM)
        lat_rad = np.radians(site_origin.lat)
        lon_rad = np.radians(site_origin.lon)
        
        # Rotation matrix: Mars-fixed (XYZ) -> Local NED (North-East-Down)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # DCM from ECEF to NED
        R = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],   # North
            [-sin_lon, cos_lon, 0],                               # East
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]   # Down
        ])
        
        # Apply rotation
        offset_ned = R @ np.array([dx, dy, dz])
        
        x_site = offset_ned[0]  # Northing
        y_site = offset_ned[1]  # Easting
        z_site = offset_ned[2]  # Down
        
        logger.debug(
            "Transformed to SITE frame",
            target_lat=lat_deg,
            target_lon=lon_deg,
            site_x=x_site,
            site_y=y_site
        )
        
        return float(x_site), float(y_site), float(z_site)
    
    def site_frame_to_iau_mars(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        site_origin: SiteOrigin
    ) -> Tuple[float, float, float]:
        """Transform SITE frame coordinates back to IAU_MARS.
        
        Args:
            x_m: X coordinate in SITE frame (North, meters)
            y_m: Y coordinate in SITE frame (East, meters)
            z_m: Z coordinate in SITE frame (Down, meters)
            site_origin: SITE frame origin
        
        Returns:
            (lat, lon, elevation) in IAU_MARS (degrees, degrees, meters)
        """
        # This is the inverse of iau_mars_to_site_frame
        # Implementation left as exercise - requires inverse rotation
        # and Cartesian to geodetic conversion
        
        raise NotImplementedError("Inverse transform not yet implemented")
```

### 4.2 Pathfinding Module (3 hours)

**Create `marshab/processing/pathfinding.py`**:
```python
"""Path planning algorithms for rover navigation."""

from heapq import heappush, heappop
from typing import List, Tuple, Optional

import numpy as np

from marshab.exceptions import NavigationError
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class AStarPathfinder:
    """A* pathfinding algorithm for rover navigation."""
    
    def __init__(self, cost_map: np.ndarray, cell_size_m: float = 1.0):
        """Initialize A* pathfinder.
        
        Args:
            cost_map: 2D array of traversability costs
            cell_size_m: Size of each cell in meters
        """
        self.cost_map = cost_map
        self.cell_size_m = cell_size_m
        self.height, self.width = cost_map.shape
    
    def heuristic(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int]
    ) -> float:
        """Calculate heuristic distance (Euclidean).
        
        Args:
            a: Start position (row, col)
            b: Goal position (row, col)
        
        Returns:
            Estimated distance
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) * self.cell_size_m
    
    def get_neighbors(
        self,
        pos: Tuple[int, int]
    ) -> List[Tuple[int, int, float]]:
        """Get valid neighbors of a position.
        
        Args:
            pos: Current position (row, col)
        
        Returns:
            List of (row, col, cost) tuples for valid neighbors
        """
        row, col = pos
        neighbors = []
        
        # 8-connected grid
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            
            # Check bounds
            if not (0 <= new_row < self.height and 0 <= new_col < self.width):
                continue
            
            # Check if passable
            cell_cost = self.cost_map[new_row, new_col]
            if np.isinf(cell_cost):
                continue
            
            # Calculate move cost (diagonal moves cost more)
            move_cost = self.cell_size_m * (1.414 if abs(dr + dc) == 2 else 1.0)
            total_cost = cell_cost * move_cost
            
            neighbors.append((new_row, new_col, total_cost))
        
        return neighbors
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path from start to goal using A*.
        
        Args:
            start: Start position (row, col)
            goal: Goal position (row, col)
        
        Returns:
            List of (row, col) positions forming path, or None if no path exists
        """
        logger.info(
            "Starting A* pathfinding",
            start=start,
            goal=goal,
            cost_map_shape=self.cost_map.shape
        )
        
        # Validate start and goal
        if not (0 <= start[0] < self.height and 0 <= start[1] < self.width):
            raise NavigationError("Start position out of bounds")
        
        if not (0 <= goal[0] < self.height and 0 <= goal[1] < self.width):
            raise NavigationError("Goal position out of bounds")
        
        if np.isinf(self.cost_map[start]):
            raise NavigationError("Start position is impassable")
        
        if np.isinf(self.cost_map[goal]):
            raise NavigationError("Goal position is impassable")
        
        # Initialize
        open_set = []
        heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        nodes_explored = 0
        
        while open_set:
            current = heappop(open_set)[1]
            nodes_explored += 1
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                logger.info(
                    "Path found",
                    path_length=len(path),
                    nodes_explored=nodes_explored,
                    total_cost=g_score[goal]
                )
                
                return path
            
            # Check all neighbors
            for neighbor_pos, _, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g
                    f_score[neighbor_pos] = tentative_g + self.heuristic(neighbor_pos, goal)
                    heappush(open_set, (f_score[neighbor_pos], neighbor_pos))
        
        logger.warning("No path found", nodes_explored=nodes_explored)
        return None
    
    def find_path_with_waypoints(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_waypoint_spacing: int = 50
    ) -> List[Tuple[int, int]]:
        """Find path and downsample to waypoints.
        
        Args:
            start: Start position
            goal: Goal position
            max_waypoint_spacing: Maximum spacing between waypoints (cells)
        
        Returns:
            List of waypoint positions
        """
        path = self.find_path(start, goal)
        
        if path is None:
            raise NavigationError("No path found between start and goal")
        
        # Downsample path to waypoints
        waypoints = [path[0]]  # Always include start
        
        for i in range(max_waypoint_spacing, len(path), max_waypoint_spacing):
            waypoints.append(path[i])
        
        # Always include goal
        if waypoints[-1] != path[-1]:
            waypoints.append(path[-1])
        
        logger.info(
            "Generated waypoints",
            path_length=len(path),
            num_waypoints=len(waypoints)
        )
        
        return waypoints
```

---

## Phase 5: CLI & Integration (Days 10-11 - 8 hours)

### 5.1 CLI Application (4 hours)

**Create `marshab/cli.py`**:
```python
"""Command-line interface for MarsHab."""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.progress import track
from rich.console import Console
from rich.table import Table

from marshab import __version__
from marshab.config import get_config
from marshab.core.data_manager import DataManager
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.navigation_engine import NavigationEngine
from marshab.types import BoundingBox, SiteOrigin
from marshab.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="marshab",
    help="Mars Habitat Site Selection and Rover Navigation System",
    add_completion=False
)

console = Console()
logger = get_logger(__name__)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        print(f"MarsHab Site Selector v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose logging"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """MarsHab Site Selector - Mars habitat site selection and rover navigation."""
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, format_type="console")
    
    # Set config path if provided
    if config_file:
        import os
        os.environ["MARSHAB_CONFIG_PATH"] = str(config_file)


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Dataset to download (mola/hirise/ctx)"),
    roi: str = typer.Option(
        ...,
        "--roi",
        help="Region of interest as 'lat_min,lat_max,lon_min,lon_max'"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-download")
):
    """Download Mars DEM data for specified region."""
    console.print(f"[bold blue]Downloading {dataset} DEM[/bold blue]")
    
    # Parse ROI
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    except Exception as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)
    
    # Download
    try:
        dm = DataManager()
        path = dm.download_dem(dataset, bbox, force=force)
        console.print(f"[green]✓ Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    roi: str = typer.Option(
        ...,
        "--roi",
        help="Region of interest as 'lat_min,lat_max,lon_min,lon_max'"
    ),
    dataset: str = typer.Option("mola", "--dataset", help="Dataset to use"),
    output: Path = typer.Option(Path("data/output"), "--output", "-o", help="Output directory"),
    threshold: float = typer.Option(0.7, "--threshold", help="Suitability threshold")
):
    """Analyze terrain and identify construction sites."""
    console.print("[bold blue]Starting terrain analysis[/bold blue]")
    
    # Parse ROI
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    except Exception as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)
    
    # Run analysis
    try:
        pipeline = AnalysisPipeline()
        results = pipeline.run(bbox, dataset=dataset, threshold=threshold)
        
        # Save results
        output.mkdir(parents=True, exist_ok=True)
        results.save(output)
        
        # Display summary
        console.print("\n[bold green]✓ Analysis complete[/bold green]\n")
        
        table = Table(title="Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Candidate Sites", str(len(results.sites)))
        table.add_row("Top Site Score", f"{results.top_site_score:.3f}")
        table.add_row("Output Directory", str(output))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Analysis failed: {e}[/red]")
        logger.exception("Analysis failed")
        raise typer.Exit(1)


@app.command()
def navigate(
    site_id: int = typer.Argument(..., help="Target site ID"),
    analysis_dir: Path = typer.Option(Path("data/output"), "--analysis", help="Analysis results directory"),
    start_lat: float = typer.Option(..., help="Start latitude"),
    start_lon: float = typer.Option(..., help="Start longitude"),
    output: Path = typer.Option(Path("waypoints.csv"), "--output", "-o", help="Waypoint output file")
):
    """Generate rover navigation waypoints to target site."""
    console.print("[bold blue]Generating navigation waypoints[/bold blue]")
    
    try:
        engine = NavigationEngine()
        waypoints = engine.plan_to_site(
            site_id=site_id,
            analysis_dir=analysis_dir,
            start_lat=start_lat,
            start_lon=start_lon
        )
        
        # Save waypoints
        waypoints.to_csv(output, index=False)
        
        console.print(f"\n[green]✓ Generated {len(waypoints)} waypoints[/green]")
        console.print(f"[green]✓ Saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Navigation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def pipeline(
    roi: str = typer.Option(..., "--roi", help="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    dataset: str = typer.Option("mola", "--dataset", help="Dataset to use"),
    output: Path = typer.Option(Path("data/output"), "--output", "-o", help="Output directory")
):
    """Run complete analysis and navigation pipeline."""
    console.print("[bold blue]Running full MarsHab pipeline[/bold blue]\n")
    
    # Parse ROI
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    except Exception as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)
    
    try:
        # 1. Download data
        with console.status("[bold green]Downloading DEM..."):
            dm = DataManager()
            dem_path = dm.download_dem(dataset, bbox)
        console.print("[green]✓ DEM downloaded[/green]")
        
        # 2. Run analysis
        with console.status("[bold green]Analyzing terrain..."):
            pipeline_obj = AnalysisPipeline()
            results = pipeline_obj.run(bbox, dataset=dataset)
        console.print("[green]✓ Terrain analyzed[/green]")
        
        # 3. Generate navigation
        with console.status("[bold green]Planning navigation..."):
            engine = NavigationEngine()
            # Navigate to top site
            waypoints = engine.plan_to_site(
                site_id=results.top_site_id,
                analysis_dir=output,
                start_lat=(bbox.lat_min + bbox.lat_max) / 2,
                start_lon=(bbox.lon_min + bbox.lon_max) / 2
            )
        console.print("[green]✓ Navigation planned[/green]")
        
        # 4. Save all outputs
        output.mkdir(parents=True, exist_ok=True)
        results.save(output)
        waypoints.to_csv(output / "waypoints.csv", index=False)
        
        console.print(f"\n[bold green]✓ Pipeline complete![/bold green]")
        console.print(f"[green]Results saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Pipeline failed: {e}[/red]")
        logger.exception("Pipeline failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

**Create `marshab/__main__.py`**:
```python
"""Entry point for python -m marshab."""

from marshab.cli import app

if __name__ == "__main__":
    app()
```

**Create `marshab/__init__.py`**:
```python
"""MarsHab Site Selector - Mars habitat site selection system."""

__version__ = "0.1.0"
__author__ = "MarsHab Development Team"
```

### 5.2 Test CLI (1 hour)

**Test commands**:
```bash
# Test help
poetry run marshab --help
poetry run marshab download --help

# Test with Docker
docker-compose run marshab --help
docker-compose run marshab pipeline --roi "40,41,180,181" --output /app/data/output

# Run full pipeline
poetry run marshab pipeline --roi "40,41,180,181" --output data/output --verbose
```

---

## Phase 6: Documentation & Polish (Day 12 - 4 hours)

### 6.1 Create README (1 hour)

Update `README.md` with full documentation, installation instructions, usage examples, and API reference.

### 6.2 CI/CD Setup (2 hours)

**Create `.github/workflows/ci.yml`**:
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH
    
    - name: Install dependencies
      run: |
        poetry install
    
    - name: Run tests
      run: |
        poetry run pytest --cov=marshab --cov-report=xml
    
    - name: Type check
      run: |
        poetry run mypy marshab
    
    - name: Lint
      run: |
        poetry run ruff check marshab
```

### 6.3 Final Testing (1 hour)

Run complete test suite and verify all components work end-to-end.

```bash
# Run all tests
poetry run pytest -v --cov=marshab --cov-report=html

# Type checking
poetry run mypy marshab

# Linting
poetry run ruff check marshab

# Build Docker image
docker-compose build

# Test full pipeline
docker-compose run marshab pipeline --roi "40,41,180,181"
```

---

## Completion Checklist

- [ ] Phase 0: Environment setup complete
- [ ] Phase 1: Core infrastructure implemented
- [ ] Phase 2: Data management working
- [ ] Phase 3: Terrain analysis functional
- [ ] Phase 4: Coordinate transforms and pathfinding done
- [ ] Phase 5: CLI fully operational
- [ ] Phase 6: Documentation and CI/CD configured
- [ ] All unit tests passing (>80% coverage)
- [ ] Integration tests passing
- [ ] Docker build successful
- [ ] Example pipeline runs end-to-end

---

## Quick Start Commands

### First Time Setup
```bash
# Clone and setup
git clone <your-repo>
cd marshab-site-selector
poetry install

# Download SPICE kernels
./scripts/setup_spice_kernels.sh

# Run tests
poetry run pytest
```

### Daily Development
```bash
# Activate environment
poetry shell

# Run tests
pytest -v

# Type check
mypy marshab

# Lint
ruff check marshab --fix

# Run CLI
marshab --help
```

### Docker Development
```bash
# Build
docker-compose build

# Run development shell
docker-compose run dev

# Run tests in container
docker-compose run marshab pytest
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-15  
**Estimated Completion**: 2-3 weeks (40-60 hours)
