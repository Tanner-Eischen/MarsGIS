"""Data acquisition and management service for Mars DEMs.

Can fetch specific MOLA tiles based on latitude/longitude from PDS-like sources
or fallback to local/synthetic data.
"""

import hashlib
import shutil
from pathlib import Path
from typing import Literal, Optional
import urllib.request
from urllib.error import URLError
import math

import xarray as xr

from marshab.config import get_config
from marshab.exceptions import DataError
from marshab.processing.dem_loader import DEMLoader
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger
from marshab.testing.synthetic_dem import create_synthetic_dem_complex

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
        """Generate cache file path for dataset and ROI."""
        if roi:
            # Include lat/lon integer degrees to roughly map to tiles
            lat_min_int = math.floor(roi.lat_min)
            lon_min_int = math.floor(roi.lon_min)
            cache_key = f"{dataset}_lat{lat_min_int}_lon{lon_min_int}"
        else:
            cache_key = dataset

        return self.cache_dir / f"{cache_key}.tif"

    def _download_real_mola_tile(self, roi: BoundingBox, dest_path: Path):
        """Attempt to download real MOLA data, handling global file caching and clipping."""
        
        # 1. Check for manual 'real_mars.tif' (dev override)
        manual_file = self.cache_dir / "real_mars.tif"
        if manual_file.exists():
            logger.info("Found manually placed real_mars.tif, using it.")
            shutil.copy(manual_file, dest_path)
            return

        # 2. Check for cached GLOBAL file
        global_filename = "mars_mola_global.tif"
        global_path = self.cache_dir / global_filename
        
        if global_path.exists():
            logger.info(f"Clipping ROI from global MOLA data: {global_path}")
            dem = None  # Initialize dem variable
            try:
                # Load and clip using DEMLoader (handles windowed reading)
                dem = self.loader.load(global_path, roi=roi)
                
                # Save the clipped DEM to dest_path
                if hasattr(dem, 'rio'):
                    dem.rio.to_raster(dest_path)
                    logger.info(f"Saved clipped MOLA tile to {dest_path}")
                    return
                else:
                    logger.warning("Loaded DEM missing rio accessor, cannot save to raster.")
            except Exception as e:
                logger.error(f"Failed to clip/save MOLA tile: {e}")
                # Fallback to synthetic

        # NOTE: We do NOT auto-download the 2GB global file here during a tile request.
        # It causes timeouts. The user must initiate global download via Data Settings.
        # Or we use synthetic fallback until then.

        # 3. Final Fallback
        logger.warning("Real data not available (Global MOLA not cached). Generating realistic synthetic proxy.")
        self._generate_synthetic_proxy(roi, dest_path)

    def _generate_synthetic_proxy(self, roi: BoundingBox, dest_path: Path):
        """Generate realistic synthetic data to proxy for real MOLA."""
        # Determine size based on ROI (approx 463m/pixel for MOLA)
        deg_height = roi.lat_max - roi.lat_min
        deg_width = roi.lon_max - roi.lon_min
        
        # Mars degree ~59km
        height_km = deg_height * 59.0
        width_km = deg_width * 59.0
        
        # MOLA is ~463m/pixel
        height_px = int((height_km * 1000) / 463)
        width_px = int((width_km * 1000) / 463)
        
        # Clamp for safety
        height_px = max(100, min(height_px, 2000))
        width_px = max(100, min(width_px, 2000))
        
        logger.info("Generating realistic synthetic DEM", size=(height_px, width_px))
        
        # Create complex terrain
        dem = create_synthetic_dem_complex(
            size=(height_px, width_px),
            features=[
                {"type": "crater", "center": (height_px//2, width_px//2), "radius": min(height_px, width_px)//4, "depth": 500},
                {"type": "hill", "center": (height_px//4, width_px//4), "radius": min(height_px, width_px)//6, "height": 300}
            ],
            cell_size_m=463.0
        )
        
        # Save to GeoTIFF
        try:
            dem.rio.to_raster(dest_path)
        except Exception as e:
            logger.error("Failed to write synthetic GeoTIFF", error=str(e))
            raise DataError(f"Could not generate synthetic proxy: {e}")

    def download_dem(
        self,
        dataset: Literal["mola", "hirise", "ctx"],
        roi: Optional[BoundingBox] = None,
        force: bool = False,
    ) -> Path:
        """Download or Generate Mars DEM.

        Args:
            dataset: Dataset to download
            roi: Region of interest to cover
            force: Force re-download even if cached

        Returns:
            Path to downloaded/cached DEM file
        """
        cache_path = self._get_cache_path(dataset, roi)

        if cache_path.exists() and not force:
            return cache_path

        if dataset == "mola" and roi:
            self._download_real_mola_tile(roi, cache_path)
            return cache_path
            
        # Existing logic for fallback/other datasets
        if dataset not in self.config.data_sources:
             # If unknown, try to use synthetic generator if ROI is present
             if roi:
                 logger.warning(f"Unknown dataset {dataset}, generating synthetic proxy.")
                 self._generate_synthetic_proxy(roi, cache_path)
                 return cache_path
             raise DataError(f"Unknown dataset: {dataset}")

        source = self.config.data_sources[dataset]
        url = source.url

        # Directory-based logic (HiRISE/CTX) from original code...
        if url.endswith('/'):
             if roi:
                 logger.warning(f"Dataset {dataset} requires manual download. Generating proxy for demo.")
                 self._generate_synthetic_proxy(roi, cache_path)
                 return cache_path
        
        # Standard URL download
        try:
            temp_path = cache_path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, temp_path)
            shutil.move(str(temp_path), str(cache_path))
            return cache_path
        except Exception as e:
            # Fallback to synthetic on failure
            if roi:
                logger.error(f"Download failed, falling back to synthetic: {e}")
                self._generate_synthetic_proxy(roi, cache_path)
                return cache_path
            raise DataError(f"Download failed: {e}")

    def load_dem(self, path: Path) -> xr.DataArray:
        return self.loader.load(path)

    def get_dem_for_roi(
        self,
        roi: BoundingBox,
        dataset: Literal["mola", "hirise", "ctx"] = "mola",
        download: bool = True,
        clip: bool = True,
    ) -> xr.DataArray:
        if download:
            dem_path = self.download_dem(dataset, roi)
        else:
            dem_path = self._get_cache_path(dataset, roi)
            if not dem_path.exists():
                # Attempt generation if missing
                self.download_dem(dataset, roi) # This will trigger generation

        dem = self.loader.load(dem_path, roi=roi if clip else None)
        if clip and roi is None:
            dem = self.loader.clip_to_roi(dem, roi)
        return dem
