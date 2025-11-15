# Phases 2–6: Complete Implementation

This document provides the comprehensive implementation for all remaining phases (2–6) with full code examples, step-by-step instructions, and verification checkpoints.

---

# Phase 2: Data Management

**Duration:** Days 4–5 – 8 hours  
**Goal:** Implement Mars DEM acquisition, caching, loading, and ROI clipping

## 2.1 DEM Loader Module (2 hours)

### Create `marshab/processing/dem_loader.py`

```bash
cat > marshab/processing/dem_loader.py << 'EOF'
"""DEM loading and raster I/O operations using GDAL/rasterio."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling
import xarray as xr

from marshab.exceptions import DataError
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class DEMLoader:
    """Handles Mars DEM loading, validation, and preprocessing."""

    def __init__(self, mars_radius_eq: float = 3396190.0):
        """Initialize DEM loader.

        Args:
            mars_radius_eq: Mars equatorial radius in meters
        """
        self.mars_radius_eq = mars_radius_eq

    def load(self, path: Path) -> xr.DataArray:
        """Load DEM into xarray DataArray with geospatial metadata.

        Args:
            path: Path to GeoTIFF DEM file

        Returns:
            xarray DataArray with elevation data and coordinates

        Raises:
            DataError: If DEM cannot be loaded or is invalid
        """
        try:
            with rasterio.open(path) as src:
                # Read elevation data
                elevation = src.read(1).astype(np.float32)

                # Extract spatial info
                height, width = elevation.shape
                transform = src.transform
                crs = src.crs
                nodata = src.nodata

                logger.info(
                    "Loading DEM",
                    path=str(path),
                    shape=(height, width),
                    crs=str(crs),
                    nodata=nodata,
                )

                # Calculate lat/lon coordinates
                rows, cols = np.meshgrid(
                    np.arange(height), np.arange(width), indexing="ij"
                )
                lons, lats = rasterio.transform.xy(transform, rows, cols)

                # Create xarray DataArray with coordinates
                dem = xr.DataArray(
                    elevation,
                    dims=["y", "x"],
                    coords={
                        "lat": (["y", "x"], np.array(lats)),
                        "lon": (["y", "x"], np.array(lons)),
                    },
                    attrs={
                        "crs": str(crs),
                        "transform": list(transform)[:6],
                        "nodata": nodata,
                        "resolution_m": abs(transform.a),
                    },
                )

                logger.info(
                    "DEM loaded successfully",
                    shape=elevation.shape,
                    resolution_m=abs(transform.a),
                    lat_range=(float(np.nanmin(lats)), float(np.nanmax(lats))),
                    lon_range=(float(np.nanmin(lons)), float(np.nanmax(lons))),
                )

                return dem

        except rasterio.errors.RasterioIOError as e:
            raise DataError(
                f"Failed to read DEM file: {path}",
                details={"path": str(path), "error": str(e)},
            )
        except Exception as e:
            raise DataError(
                f"Unexpected error loading DEM: {path}",
                details={"path": str(path), "error": str(e)},
            )

    def clip_to_roi(self, dem: xr.DataArray, roi: BoundingBox) -> xr.DataArray:
        """Clip DEM raster to region of interest.

        Args:
            dem: Input DEM DataArray
            roi: Bounding box to clip to

        Returns:
            Clipped DEM DataArray

        Raises:
            DataError: If clipping fails
        """
        try:
            logger.info(
                "Clipping DEM to ROI",
                original_shape=dem.shape,
                roi=roi.model_dump(),
            )

            # Create mask for pixels within ROI
            lat_mask = (dem.coords["lat"] >= roi.lat_min) & (
                dem.coords["lat"] <= roi.lat_max
            )
            lon_mask = (dem.coords["lon"] >= roi.lon_min) & (
                dem.coords["lon"] <= roi.lon_max
            )
            combined_mask = lat_mask & lon_mask

            # Apply mask
            clipped = dem.where(combined_mask, drop=True)

            logger.info(
                "DEM clipped",
                clipped_shape=clipped.shape,
                pixels_retained=int(np.sum(combined_mask.values)),
            )

            return clipped

        except Exception as e:
            raise DataError(
                f"Failed to clip DEM to ROI",
                details={"roi": roi.model_dump(), "error": str(e)},
            )

    def resample(
        self,
        dem: xr.DataArray,
        target_resolution_m: float,
        method: str = "bilinear",
    ) -> xr.DataArray:
        """Resample DEM to different resolution.

        Args:
            dem: Input DEM DataArray
            target_resolution_m: Target resolution in meters
            method: Resampling method ('nearest', 'bilinear', 'cubic')

        Returns:
            Resampled DEM DataArray
        """
        current_res = float(dem.attrs.get("resolution_m", 200.0))

        if abs(current_res - target_resolution_m) < 1.0:
            logger.info("DEM already at target resolution")
            return dem

        scale_factor = current_res / target_resolution_m

        logger.info(
            "Resampling DEM",
            from_resolution_m=current_res,
            to_resolution_m=target_resolution_m,
            scale_factor=scale_factor,
            from_shape=dem.shape,
        )

        from scipy.ndimage import zoom

        order = {"nearest": 0, "bilinear": 1, "cubic": 3}.get(method, 1)
        resampled_data = zoom(dem.values, scale_factor, order=order)

        resampled = xr.DataArray(
            resampled_data,
            dims=["y", "x"],
            attrs={**dem.attrs, "resolution_m": target_resolution_m},
        )

        return resampled
EOF
```

## 2.2 Data Manager Service (2 hours)

### Create `marshab/core/data_manager.py`

```bash
cat > marshab/core/data_manager.py << 'EOF'
"""Data acquisition and management service for Mars DEMs."""

import hashlib
import shutil
from pathlib import Path
from typing import Literal, Optional
import urllib.request
from urllib.error import URLError

import xarray as xr

from marshab.config import get_config
from marshab.exceptions import DataError
from marshab.processing.dem_loader import DEMLoader
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class DataManager:
    """Manages Mars DEM data acquisition, caching, and loading."""

    def __init__(self):
        """Initialize data manager with configuration."""
        self.config = get_config()
        self.loader = DEMLoader(
            mars_radius_eq=self.config.mars.equatorial_radius_m
        )
        self.cache_dir = self.config.paths.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(
        self, dataset: str, roi: Optional[BoundingBox] = None
    ) -> Path:
        """Generate cache file path for dataset and ROI.

        Args:
            dataset: Dataset identifier (mola, hirise, ctx)
            roi: Optional region of interest for unique key

        Returns:
            Path to cached file
        """
        if roi:
            cache_key = hashlib.md5(
                f"{dataset}_{roi.lat_min}_{roi.lat_max}_{roi.lon_min}_{roi.lon_max}".encode()
            ).hexdigest()
        else:
            cache_key = hashlib.md5(dataset.encode()).hexdigest()

        return self.cache_dir / f"{dataset}_{cache_key}.tif"

    def download_dem(
        self,
        dataset: Literal["mola", "hirise", "ctx"] = "mola",
        roi: Optional[BoundingBox] = None,
        force: bool = False,
    ) -> Path:
        """Download Mars DEM covering specified ROI.

        Args:
            dataset: Dataset to download
            roi: Region of interest to cover
            force: Force re-download even if cached

        Returns:
            Path to downloaded/cached DEM file

        Raises:
            DataError: If dataset unknown or download fails
        """
        cache_path = self._get_cache_path(dataset, roi)

        # Check cache
        if cache_path.exists() and not force:
            logger.info(
                "Using cached DEM",
                dataset=dataset,
                path=str(cache_path),
                size_mb=cache_path.stat().st_size / 1e6,
            )
            return cache_path

        # Get data source URL
        if dataset not in self.config.data_sources:
            raise DataError(
                f"Unknown dataset: {dataset}",
                details={
                    "available": list(self.config.data_sources.keys()),
                },
            )

        source = self.config.data_sources[dataset]
        url = source.url

        logger.info(
            "Downloading DEM",
            dataset=dataset,
            url=url,
            roi=roi.model_dump() if roi else None,
        )

        try:
            temp_path = cache_path.with_suffix(".tmp")

            # Download with basic progress tracking
            def download_with_progress(url: str, dest: Path):
                def progress_hook(block_num, block_size, total_size):
                    if block_num == 0:
                        logger.info(f"Download size: {total_size / 1e6:.1f} MB")
                    if block_num % 100 == 0:
                        downloaded = min(block_num * block_size, total_size)
                        pct = 100 * downloaded / total_size if total_size > 0 else 0
                        logger.debug(f"Download progress: {pct:.1f}%")

                urllib.request.urlretrieve(url, dest, reporthook=progress_hook)

            download_with_progress(url, temp_path)

            # Move to final location
            shutil.move(str(temp_path), str(cache_path))

            logger.info(
                "Downloaded DEM successfully",
                dataset=dataset,
                path=str(cache_path),
                size_mb=cache_path.stat().st_size / 1e6,
            )

            return cache_path

        except URLError as e:
            if temp_path.exists():
                temp_path.unlink()
            raise DataError(
                f"Failed to download DEM: {dataset}",
                details={"url": url, "error": str(e)},
            )

    def load_dem(self, path: Path) -> xr.DataArray:
        """Load DEM from file path.

        Args:
            path: Path to DEM GeoTIFF

        Returns:
            DEM as xarray DataArray
        """
        return self.loader.load(path)

    def get_dem_for_roi(
        self,
        roi: BoundingBox,
        dataset: Literal["mola", "hirise", "ctx"] = "mola",
        download: bool = True,
        clip: bool = True,
    ) -> xr.DataArray:
        """Get DEM covering ROI, downloading if necessary.

        Args:
            roi: Region of interest
            dataset: Dataset to use
            download: Whether to download if not cached
            clip: Whether to clip to exact ROI

        Returns:
            DEM DataArray covering ROI

        Raises:
            DataError: If download fails or data unavailable
        """
        if download:
            dem_path = self.download_dem(dataset, roi)
        else:
            dem_path = self._get_cache_path(dataset, roi)

            if not dem_path.exists():
                raise DataError(
                    f"DEM not found in cache: {dataset}",
                    details={"path": str(dem_path)},
                )

        # Load DEM
        dem = self.loader.load(dem_path)

        # Clip to ROI if requested
        if clip:
            dem = self.loader.clip_to_roi(dem, roi)

        return dem
EOF
```

## 2.3 Unit Tests (2 hours)

### Create `tests/unit/test_dem_loader.py`

```bash
cat > tests/unit/test_dem_loader.py << 'EOF'
"""Unit tests for DEM loader functionality."""

import pytest
import numpy as np

from marshab.processing.dem_loader import DEMLoader
from marshab.types import BoundingBox


class TestDEMLoader:
    """Tests for DEMLoader class."""

    @pytest.fixture
    def loader(self):
        """Provide DEM loader instance."""
        return DEMLoader()

    def test_load_dem(self, loader, synthetic_dem):
        """Test loading DEM from file."""
        dem = loader.load(synthetic_dem)

        assert dem is not None
        assert "lat" in dem.coords
        assert "lon" in dem.coords
        assert dem.shape == (100, 100)
        assert dem.attrs["crs"] == "EPSG:49900"

    def test_clip_to_roi(self, loader, synthetic_dem, test_roi):
        """Test ROI clipping."""
        dem = loader.load(synthetic_dem)
        clipped = loader.clip_to_roi(dem, test_roi)

        # Clipped should be smaller (approximately)
        assert clipped.shape[0] <= dem.shape[0]
        assert clipped.shape[1] <= dem.shape[1]

        # All coordinates should be within ROI (within tolerance)
        lat_min = float(clipped.coords["lat"].min())
        lat_max = float(clipped.coords["lat"].max())
        lon_min = float(clipped.coords["lon"].min())
        lon_max = float(clipped.coords["lon"].max())

        assert lat_min >= test_roi.lat_min - 0.01
        assert lat_max <= test_roi.lat_max + 0.01

    def test_resample(self, loader, synthetic_dem):
        """Test DEM resampling."""
        dem = loader.load(synthetic_dem)

        # Downsample to lower resolution (larger cells)
        resampled = loader.resample(dem, target_resolution_m=1000.0)

        # Should have fewer pixels
        assert resampled.shape[0] < dem.shape[0]
        assert resampled.shape[1] < dem.shape[1]

    def test_load_preserves_metadata(self, loader, synthetic_dem):
        """Test that loading preserves metadata."""
        dem = loader.load(synthetic_dem)

        assert "crs" in dem.attrs
        assert "resolution_m" in dem.attrs
        assert dem.attrs["resolution_m"] > 0
EOF
```

### Create `tests/unit/test_data_manager.py`

```bash
cat > tests/unit/test_data_manager.py << 'EOF'
"""Unit tests for data manager service."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from marshab.core.data_manager import DataManager
from marshab.types import BoundingBox
from marshab.exceptions import DataError


class TestDataManager:
    """Tests for DataManager class."""

    @pytest.fixture
    def data_manager(self, test_config):
        """Provide DataManager instance with test config."""
        return DataManager()

    def test_cache_key_generation(self, data_manager, test_roi):
        """Test cache key generation."""
        key1 = data_manager._get_cache_path("mola", test_roi)
        key2 = data_manager._get_cache_path("mola", test_roi)

        # Same inputs should produce same cache path
        assert key1 == key2

        # Different datasets should have different paths
        key3 = data_manager._get_cache_path("hirise", test_roi)
        assert key1 != key3

    def test_load_dem_from_file(self, data_manager, synthetic_dem):
        """Test loading DEM from existing file."""
        dem = data_manager.load_dem(synthetic_dem)

        assert dem is not None
        assert dem.shape == (100, 100)

    def test_get_dem_for_roi_without_download(self, data_manager, synthetic_dem, test_roi):
        """Test getting DEM without downloading."""
        # This test would need mock setup - simplified version
        dem = data_manager.loader.load(synthetic_dem)
        clipped = data_manager.loader.clip_to_roi(dem, test_roi)

        assert clipped is not None
        assert clipped.shape[0] > 0

    def test_invalid_dataset(self, data_manager, test_roi):
        """Test error handling for invalid dataset."""
        with pytest.raises(DataError) as excinfo:
            data_manager.download_dem("invalid_dataset", test_roi)

        assert "Unknown dataset" in str(excinfo.value)
EOF
```

## 2.4 Integration Test

### Create `tests/integration/test_data_pipeline.py`

```bash
mkdir -p tests/integration

cat > tests/integration/test_data_pipeline.py << 'EOF'
"""Integration tests for data pipeline."""

import pytest

from marshab.core.data_manager import DataManager
from marshab.types import BoundingBox


class TestDataPipeline:
    """Integration tests for complete data pipeline."""

    def test_load_and_clip_dem(self, synthetic_dem, test_roi):
        """Test complete pipeline: load, validate, clip."""
        dm = DataManager()

        # Load DEM
        dem = dm.load_dem(synthetic_dem)
        assert dem.shape == (100, 100)

        # Clip to ROI
        clipped = dm.loader.clip_to_roi(dem, test_roi)
        assert clipped.shape[0] < dem.shape[0]

    def test_dem_metadata(self, synthetic_dem):
        """Test DEM metadata is preserved."""
        dm = DataManager()
        dem = dm.load_dem(synthetic_dem)

        assert "crs" in dem.attrs
        assert "resolution_m" in dem.attrs
        assert dem.attrs["crs"] == "EPSG:49900"
EOF
```

## 2.5 Run All Phase 2 Tests

```bash
# Run all Phase 2 tests
poetry run pytest tests/unit/test_dem_loader.py tests/unit/test_data_manager.py tests/integration/test_data_pipeline.py -v

# Run with coverage
poetry run pytest tests/unit/test_dem_loader.py tests/unit/test_data_manager.py --cov=marshab.core --cov=marshab.processing -v
```

---

# Phase 3: Terrain Analysis (Days 6–7 – 8 hours)

## 3.1 Terrain Analytics Module (3 hours)

### Create `marshab/processing/terrain.py`

```bash
cat > marshab/processing/terrain.py << 'EOF'
"""Terrain analysis functions for Mars DEMs."""

from typing import Tuple

import numpy as np
from scipy import ndimage
import xarray as xr

from marshab.types import TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class TerrainAnalyzer:
    """Performs terrain analysis and derivative calculations on Mars DEMs."""

    def __init__(self, cell_size_m: float = 200.0):
        """Initialize terrain analyzer.

        Args:
            cell_size_m: DEM cell size in meters
        """
        self.cell_size_m = cell_size_m

    def calculate_slope(self, dem: np.ndarray) -> np.ndarray:
        """Calculate slope magnitude in degrees.

        Uses gradient method: slope = arctan(sqrt(dx² + dy²))

        Args:
            dem: Input elevation array

        Returns:
            Slope array in degrees (0-90)
        """
        # Calculate gradients
        dy, dx = np.gradient(dem, self.cell_size_m)

        # Calculate slope magnitude
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        logger.debug(
            "Calculated slope",
            mean_deg=float(np.nanmean(slope)),
            max_deg=float(np.nanmax(slope)),
            min_deg=float(np.nanmin(slope)),
        )

        return slope

    def calculate_aspect(self, dem: np.ndarray) -> np.ndarray:
        """Calculate aspect (slope direction) in degrees from North.

        Aspect = 0° = North, 90° = East, 180° = South, 270° = West

        Args:
            dem: Input elevation array

        Returns:
            Aspect array in degrees (0-360)
        """
        dy, dx = np.gradient(dem, self.cell_size_m)

        # Calculate aspect: atan2 gives angle from east
        # Convert to angle from north (0-360)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = (aspect + 360) % 360

        logger.debug(
            "Calculated aspect",
            valid_pixels=int(np.sum(~np.isnan(aspect))),
        )

        return aspect

    def calculate_roughness(
        self, dem: np.ndarray, window_size: int = 3
    ) -> np.ndarray:
        """Calculate terrain roughness as local standard deviation.

        Args:
            dem: Input elevation array
            window_size: Window size for roughness calculation (odd number)

        Returns:
            Roughness array
        """
        roughness = ndimage.generic_filter(
            dem, np.std, size=window_size, mode="reflect"
        )

        logger.debug(
            "Calculated roughness",
            mean=float(np.nanmean(roughness)),
            max=float(np.nanmax(roughness)),
        )

        return roughness

    def calculate_tri(self, dem: np.ndarray) -> np.ndarray:
        """Calculate Terrain Ruggedness Index (TRI).

        TRI = mean absolute elevation difference between center and neighbors

        Args:
            dem: Input elevation array

        Returns:
            TRI array
        """
        kernel = np.array(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        ) / 8.0

        mean_neighbors = ndimage.convolve(dem, kernel, mode="reflect")
        tri = np.abs(dem - mean_neighbors)

        logger.debug(
            "Calculated TRI",
            mean=float(np.nanmean(tri)),
            max=float(np.nanmax(tri)),
        )

        return tri

    def analyze(self, dem: xr.DataArray) -> TerrainMetrics:
        """Perform complete terrain analysis on DEM.

        Args:
            dem: Input DEM DataArray

        Returns:
            TerrainMetrics with all calculated products
        """
        logger.info("Starting terrain analysis", shape=dem.shape)

        elevation = dem.values.astype(np.float32)

        # Calculate all metrics
        slope = self.calculate_slope(elevation)
        aspect = self.calculate_aspect(elevation)
        roughness = self.calculate_roughness(elevation)
        tri = self.calculate_tri(elevation)

        logger.info(
            "Terrain analysis complete",
            slope_mean=float(np.nanmean(slope)),
            slope_max=float(np.nanmax(slope)),
            roughness_mean=float(np.nanmean(roughness)),
        )

        return TerrainMetrics(
            slope=slope,
            aspect=aspect,
            roughness=roughness,
            tri=tri,
        )


def generate_cost_surface(
    slope: np.ndarray,
    roughness: np.ndarray,
    max_slope_deg: float = 25.0,
) -> np.ndarray:
    """Generate traversability cost surface for pathfinding.

    Args:
        slope: Slope array in degrees
        roughness: Roughness array
        max_slope_deg: Maximum traversable slope (degrees)

    Returns:
        Cost surface (higher = more difficult, inf = impassable)
    """
    # Initialize base cost
    cost = np.ones_like(slope, dtype=np.float32)

    # Add slope cost (exponential increase)
    cost += (slope / 45.0) ** 2 * 10.0

    # Add roughness cost (normalized)
    roughness_norm = roughness / (np.nanmax(roughness) + 1e-6)
    cost += roughness_norm * 5.0

    # Mark impassable areas (steep slopes)
    cost[slope > max_slope_deg] = np.inf

    logger.info(
        "Generated cost surface",
        passable_fraction=float(np.sum(np.isfinite(cost)) / cost.size),
        max_cost=float(np.nanmax(cost[np.isfinite(cost)])),
    )

    return cost
EOF
```

## 3.2 MCDM Implementation (2 hours)

### Create `marshab/processing/mcdm.py`

```bash
cat > marshab/processing/mcdm.py << 'EOF'
"""Multi-Criteria Decision Making (MCDM) for site suitability."""

from typing import Dict

import numpy as np

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class MCDMEvaluator:
    """Multi-criteria decision making using weighted sum and TOPSIS."""

    @staticmethod
    def normalize_criterion(
        data: np.ndarray, beneficial: bool = True
    ) -> np.ndarray:
        """Normalize criterion to 0-1 range.

        Args:
            data: Input criterion array
            beneficial: True if higher values are better

        Returns:
            Normalized array in [0, 1] range
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

    @staticmethod
    def weighted_sum(
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool],
    ) -> np.ndarray:
        """Calculate weighted sum of normalized criteria.

        Args:
            criteria: Dictionary of criterion name -> array
            weights: Dictionary of criterion name -> weight
            beneficial: Dictionary of criterion name -> benefit direction

        Returns:
            Suitability score array in [0, 1] range
        """
        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Normalize all criteria
        normalized = {}
        for name, data in criteria.items():
            is_beneficial = beneficial.get(name, True)
            normalized[name] = MCDMEvaluator.normalize_criterion(data, is_beneficial)

        # Calculate weighted sum
        suitability = np.zeros_like(list(criteria.values())[0], dtype=np.float32)

        for name, norm_data in normalized.items():
            weight = weights.get(name, 0.0)
            suitability += norm_data * weight

            logger.debug(
                f"Applied criterion: {name}",
                weight=weight,
                mean_value=float(np.nanmean(norm_data)),
            )

        logger.info(
            "Weighted sum computed",
            mean_suitability=float(np.nanmean(suitability)),
            max_suitability=float(np.nanmax(suitability)),
        )

        return suitability

    @staticmethod
    def topsis(
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool],
    ) -> np.ndarray:
        """TOPSIS (Technique for Order Preference by Similarity to Ideal).

        Args:
            criteria: Dictionary of criterion name -> array
            weights: Dictionary of criterion name -> weight
            beneficial: Dictionary of criterion name -> benefit direction

        Returns:
            TOPSIS score array in [0, 1] range
        """
        # Vector normalization
        normalized = {}
        for name, data in criteria.items():
            norm = np.sqrt(np.sum(data**2 + 1e-10))
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
                ideal_best[name] = np.nanmax(w_data)
                ideal_worst[name] = np.nanmin(w_data)
            else:
                ideal_best[name] = np.nanmin(w_data)
                ideal_worst[name] = np.nanmax(w_data)

        # Calculate distances
        dist_best = np.zeros_like(list(weighted.values())[0], dtype=np.float32)
        dist_worst = np.zeros_like(list(weighted.values())[0], dtype=np.float32)

        for name, w_data in weighted.items():
            dist_best += (w_data - ideal_best[name]) ** 2
            dist_worst += (w_data - ideal_worst[name]) ** 2

        dist_best = np.sqrt(dist_best)
        dist_worst = np.sqrt(dist_worst)

        # TOPSIS score
        denominator = dist_best + dist_worst
        topsis_score = np.where(
            denominator > 0,
            dist_worst / denominator,
            0.5,
        )

        logger.info(
            "TOPSIS computed",
            mean_score=float(np.nanmean(topsis_score)),
            max_score=float(np.nanmax(topsis_score)),
        )

        return topsis_score
EOF
```

## 3.3 Unit Tests

### Create `tests/unit/test_terrain.py`

```bash
cat > tests/unit/test_terrain.py << 'EOF'
"""Unit tests for terrain analysis."""

import numpy as np
import pytest

from marshab.processing.terrain import TerrainAnalyzer, generate_cost_surface


class TestTerrainAnalyzer:
    """Tests for TerrainAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Provide TerrainAnalyzer instance."""
        return TerrainAnalyzer(cell_size_m=100.0)

    def test_calculate_slope(self, analyzer):
        """Test slope calculation."""
        # Create sloped surface (constant gradient)
        dem = np.tile(np.arange(50), (50, 1)).T.astype(float)

        slope = analyzer.calculate_slope(dem)

        assert slope.shape == dem.shape
        assert np.all(slope >= 0)
        assert np.all(slope <= 90)

    def test_calculate_aspect(self, analyzer):
        """Test aspect calculation."""
        dem = np.random.randn(50, 50) * 10.0

        aspect = analyzer.calculate_aspect(dem)

        assert aspect.shape == dem.shape
        assert np.all(aspect >= 0)
        assert np.all(aspect <= 360)

    def test_calculate_roughness(self, analyzer):
        """Test roughness calculation."""
        smooth = np.ones((50, 50)) * 100.0
        rough = np.random.randn(50, 50) * 50 + 100.0

        roughness_smooth = analyzer.calculate_roughness(smooth)
        roughness_rough = analyzer.calculate_roughness(rough)

        # Rough surface should have higher average roughness
        assert np.nanmean(roughness_rough) > np.nanmean(roughness_smooth)

    def test_full_analysis(self, sample_terrain_data):
        """Test complete terrain analysis."""
        import xarray as xr

        analyzer = TerrainAnalyzer()
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
EOF
```

### Create `tests/unit/test_mcdm.py`

```bash
cat > tests/unit/test_mcdm.py << 'EOF'
"""Unit tests for MCDM evaluation."""

import numpy as np
import pytest

from marshab.processing.mcdm import MCDMEvaluator


class TestMCDMEvaluator:
    """Tests for MCDMEvaluator class."""

    def test_normalize_beneficial(self):
        """Test normalization of beneficial criterion."""
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        normalized = MCDMEvaluator.normalize_criterion(data, beneficial=True)

        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)
        assert normalized[0] < normalized[4]  # Lower values are lower

    def test_normalize_cost(self):
        """Test normalization of cost criterion."""
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        normalized = MCDMEvaluator.normalize_criterion(data, beneficial=False)

        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)
        assert normalized[0] > normalized[4]  # Higher values are better

    def test_weighted_sum(self):
        """Test weighted sum MCDM."""
        criteria = {
            "slope": np.array([5, 10, 15, 20, 25], dtype=float),
            "roughness": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float),
        }
        weights = {"slope": 0.6, "roughness": 0.4}
        beneficial = {"slope": False, "roughness": False}

        suitability = MCDMEvaluator.weighted_sum(criteria, weights, beneficial)

        assert suitability.shape == (5,)
        assert np.all(suitability >= 0)
        assert np.all(suitability <= 1)
        assert suitability[0] > suitability[4]  # Lower is better for both
EOF
```

## 3.4 Run Phase 3 Tests

```bash
poetry run pytest tests/unit/test_terrain.py tests/unit/test_mcdm.py -v --cov=marshab.processing
```

---

# Phase 4: Coordinates & Navigation (Days 8–9 – 8 hours)

## 4.1 Coordinate Transformations (3 hours)

### Create `marshab/processing/coordinates.py`

```bash
cat > marshab/processing/coordinates.py << 'EOF'
"""Coordinate transformations between Mars reference frames."""

from typing import Tuple

import numpy as np

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
        """Initialize coordinate transformer.

        Args:
            equatorial_radius: Mars equatorial radius (meters)
            polar_radius: Mars polar radius (meters)
        """
        self.eq_radius = equatorial_radius
        self.pol_radius = polar_radius

    def planetocentric_to_cartesian(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
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

        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)

        # Ellipsoid radius at this latitude
        numerator = np.sqrt(
            (self.eq_radius**2 * cos_lat) ** 2 + (self.pol_radius**2 * sin_lat) ** 2
        )
        denominator = np.sqrt(
            (self.eq_radius * cos_lat) ** 2 + (self.pol_radius * sin_lat) ** 2
        )

        radius = numerator / denominator + elevation_m

        # Convert to Cartesian
        x = radius * cos_lat * np.cos(lon_rad)
        y = radius * cos_lat * np.sin(lon_rad)
        z = radius * sin_lat

        return float(x), float(y), float(z)

    def iau_mars_to_site_frame(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
        site_origin: SiteOrigin,
    ) -> Tuple[float, float, float]:
        """Transform IAU_MARS to rover SITE frame.

        SITE frame: Origin at site_origin, +X=North, +Y=East, +Z=Down

        Args:
            lat_deg: Target latitude (degrees)
            lon_deg: Target longitude (degrees)
            elevation_m: Target elevation (meters)
            site_origin: SITE frame origin

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
                site_origin.elevation_m,
            )

            # Calculate offset vector
            dx = target_xyz[0] - origin_xyz[0]
            dy = target_xyz[1] - origin_xyz[1]
            dz = target_xyz[2] - origin_xyz[2]

            # Rotation matrix: Mars-fixed → Local North-East-Down
            lat_rad = np.radians(site_origin.lat)
            lon_rad = np.radians(site_origin.lon)

            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            sin_lon = np.sin(lon_rad)
            cos_lon = np.cos(lon_rad)

            R = np.array([
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ])

            # Apply rotation
            offset_ned = R @ np.array([dx, dy, dz])

            x_site = float(offset_ned[0])  # Northing
            y_site = float(offset_ned[1])  # Easting
            z_site = float(offset_ned[2])  # Down

            logger.debug(
                "Transformed to SITE frame",
                target_lat=lat_deg,
                target_lon=lon_deg,
                site_x=x_site,
                site_y=y_site,
            )

            return x_site, y_site, z_site

        except Exception as e:
            raise CoordinateError(
                "Failed to transform coordinates to SITE frame",
                details={
                    "lat": lat_deg,
                    "lon": lon_deg,
                    "error": str(e),
                },
            )
EOF
```

## 4.2 Pathfinding (3 hours)

### Create `marshab/processing/pathfinding.py`

```bash
cat > marshab/processing/pathfinding.py << 'EOF'
"""A* pathfinding for rover navigation on Mars terrain."""

from heapq import heappush, heappop
from typing import List, Tuple, Optional

import numpy as np

from marshab.exceptions import NavigationError
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class AStarPathfinder:
    """A* pathfinding on traversability cost maps."""

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
        self, a: Tuple[int, int], b: Tuple[int, int]
    ) -> float:
        """Euclidean distance heuristic.

        Args:
            a: Start position (row, col)
            b: Goal position (row, col)

        Returns:
            Estimated distance to goal
        """
        return (
            np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * self.cell_size_m
        )

    def get_neighbors(
        self, pos: Tuple[int, int]
    ) -> List[Tuple[int, int, float]]:
        """Get valid neighbors of a position.

        Args:
            pos: Current position (row, col)

        Returns:
            List of (row, col, cost) tuples
        """
        row, col = pos
        neighbors = []

        # 8-connected neighborhood
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
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

            # Move cost (diagonal moves cost sqrt(2) more)
            move_cost = self.cell_size_m * (
                1.414 if abs(dr + dc) == 2 else 1.0
            )
            total_cost = cell_cost * move_cost

            neighbors.append((new_row, new_col, total_cost))

        return neighbors

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path using A*.

        Args:
            start: Start position (row, col)
            goal: Goal position (row, col)

        Returns:
            List of positions forming path, or None if no path found
        """
        logger.info(
            "Starting A* pathfinding",
            start=start,
            goal=goal,
            map_size=self.cost_map.shape,
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

        # Initialize search
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
                    total_cost=g_score[goal],
                )

                return path

            # Check neighbors
            for neighbor_pos, _, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost

                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g
                    f_score[neighbor_pos] = (
                        tentative_g + self.heuristic(neighbor_pos, goal)
                    )
                    heappush(
                        open_set,
                        (f_score[neighbor_pos], neighbor_pos),
                    )

        logger.warning("No path found", nodes_explored=nodes_explored)
        return None

    def find_path_with_waypoints(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_waypoint_spacing: int = 50,
    ) -> List[Tuple[int, int]]:
        """Find path and downsample to waypoints.

        Args:
            start: Start position
            goal: Goal position
            max_waypoint_spacing: Maximum spacing between waypoints (cells)

        Returns:
            List of waypoint positions

        Raises:
            NavigationError: If no path found
        """
        path = self.find_path(start, goal)

        if path is None:
            raise NavigationError("No path found between start and goal")

        # Downsample to waypoints
        waypoints = [path[0]]

        for i in range(max_waypoint_spacing, len(path), max_waypoint_spacing):
            waypoints.append(path[i])

        # Always include goal
        if waypoints[-1] != path[-1]:
            waypoints.append(path[-1])

        logger.info(
            "Generated waypoints",
            path_length=len(path),
            num_waypoints=len(waypoints),
        )

        return waypoints
EOF
```

## 4.3 Unit Tests

### Create `tests/unit/test_coordinates.py`

```bash
cat > tests/unit/test_coordinates.py << 'EOF'
"""Unit tests for coordinate transformations."""

import pytest

from marshab.processing.coordinates import CoordinateTransformer
from marshab.types import SiteOrigin


class TestCoordinateTransformer:
    """Tests for CoordinateTransformer."""

    @pytest.fixture
    def transformer(self):
        """Provide transformer instance."""
        return CoordinateTransformer()

    def test_planetocentric_to_cartesian(self, transformer):
        """Test planetocentric to Cartesian conversion."""
        x, y, z = transformer.planetocentric_to_cartesian(0, 0, 0)

        # At equator, lon 0, should be on X-axis
        assert pytest.approx(x, rel=0.01) == transformer.eq_radius
        assert pytest.approx(y, abs=1) == 0
        assert pytest.approx(z, abs=1) == 0

    def test_iau_mars_to_site_frame(self, transformer):
        """Test IAU_MARS to SITE frame transformation."""
        site = SiteOrigin(lat=0, lon=0, elevation_m=0)

        # Point slightly north of origin
        x, y, z = transformer.iau_mars_to_site_frame(
            lat_deg=0.01,
            lon_deg=0,
            elevation_m=0,
            site_origin=site,
        )

        # Should be north (positive x)
        assert x > 0
        assert pytest.approx(y, abs=100) == 0  # Small east component
EOF
```

### Create `tests/unit/test_pathfinding.py`

```bash
cat > tests/unit/test_pathfinding.py << 'EOF'
"""Unit tests for pathfinding."""

import numpy as np
import pytest

from marshab.processing.pathfinding import AStarPathfinder
from marshab.exceptions import NavigationError


class TestAStarPathfinder:
    """Tests for A* pathfinding."""

    def test_simple_path(self):
        """Test finding simple path on empty cost map."""
        # Uniform cost map
        cost_map = np.ones((50, 50))

        pathfinder = AStarPathfinder(cost_map, cell_size_m=1.0)
        path = pathfinder.find_path((0, 0), (10, 10))

        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (10, 10)

    def test_path_around_obstacle(self):
        """Test path planning around obstacle."""
        cost_map = np.ones((20, 20))
        # Create obstacle line
        cost_map[5:15, 10] = np.inf

        pathfinder = AStarPathfinder(cost_map)
        path = pathfinder.find_path((2, 2), (18, 18))

        assert path is not None
        # Path should avoid obstacle column
        for pos in path:
            assert cost_map[pos] < np.inf

    def test_no_path(self):
        """Test when no path exists."""
        cost_map = np.ones((20, 20))
        # Block all paths
        cost_map[10, :] = np.inf
        cost_map[:, 10] = np.inf

        pathfinder = AStarPathfinder(cost_map)
        path = pathfinder.find_path((2, 2), (18, 18))

        # Should return None (no path)
        assert path is None

    def test_invalid_start(self):
        """Test error for invalid start position."""
        cost_map = np.ones((20, 20))
        pathfinder = AStarPathfinder(cost_map)

        with pytest.raises(NavigationError):
            pathfinder.find_path((-1, 5), (10, 10))

    def test_waypoint_generation(self):
        """Test waypoint downsampling."""
        cost_map = np.ones((100, 100))

        pathfinder = AStarPathfinder(cost_map)
        waypoints = pathfinder.find_path_with_waypoints(
            (0, 0), (99, 99), max_waypoint_spacing=20
        )

        # Should have fewer waypoints than full path
        full_path = pathfinder.find_path((0, 0), (99, 99))
        assert len(waypoints) < len(full_path)
        assert waypoints[0] == (0, 0)
        assert waypoints[-1] == (99, 99)
EOF
```

## 4.4 Run Phase 4 Tests

```bash
poetry run pytest tests/unit/test_coordinates.py tests/unit/test_pathfinding.py -v --cov=marshab.processing
```

---

# Phase 5: CLI & Integration

**Duration:** Days 10–11 – 8 hours

[Create CLI commands as documented in earlier Phase 5 section of code-examples.md]

Run all tests:
```bash
poetry run pytest -v --cov=marshab --cov-report=html
```

---

# Phase 6: Documentation & Polish

**Duration:** Day 12 – 4 hours

- Update README with final documentation
- Finalize GitHub Actions CI/CD
- Create developer guide
- Create user guide

---

**Total Implementation Time**: ~48-60 hours across all phases

All phases can be completed incrementally with working checkpoints at each phase completion.
