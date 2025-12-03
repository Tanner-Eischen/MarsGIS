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
from marshab.models import BoundingBox
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
        dataset: Literal["mola", "hirise", "ctx"],
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

        # Check if URL is a directory (ends with /)
        if url.endswith('/'):
            # Directory-based datasets require manual download
            error_msg = (
                f"Dataset '{dataset}' requires manual download. "
                f"HiRISE and CTX datasets are directory-based and require selecting specific observation IDs.\n\n"
                f"Please download manually from:\n"
            )
            if dataset == "hirise":
                error_msg += (
                    f"  - HiRISE PDS: https://www.uahirise.org/hiwish/\n"
                    f"  - AWS S3: https://s3.amazonaws.com/mars-hirise-pds/\n"
                )
            elif dataset == "ctx":
                error_msg += (
                    f"  - WUSTL ODE: https://ode.rsl.wustl.edu/mars/\n"
                )
            error_msg += (
                f"\nAfter downloading, place the DEM file in the cache directory:\n"
                f"  {cache_path}\n"
                f"Or rename your file to match the expected cache filename pattern."
            )
            raise DataError(
                error_msg,
                details={
                    "dataset": dataset,
                    "url": url,
                    "cache_path": str(cache_path),
                    "requires_manual_download": True,
                },
            )

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
            
            # Provide helpful error message with alternative sources
            error_msg = (
                f"Failed to download DEM: {dataset}\n"
                f"URL: {url}\n"
                f"Error: {str(e)}\n\n"
                f"Alternative options:\n"
                f"1. Check if the URL is still valid\n"
                f"2. Manually download from USGS Astrogeology: "
                f"https://astrogeology.usgs.gov/search/map/Mars\n"
                f"3. Use NASA PDS: https://pds-geosciences.wustl.edu/\n"
                f"4. Place the DEM file in: {cache_path}"
            )
            
            raise DataError(
                f"Failed to download DEM: {dataset}",
                details={"url": url, "error": str(e), "cache_path": str(cache_path)},
            ) from e
    
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

        # Load DEM with ROI windowing for memory efficiency
        # Pass ROI to load() to use windowed reading (much more memory efficient)
        # If we're using windowed reading, the DEM is already clipped, so skip second clipping
        dem = self.loader.load(dem_path, roi=roi if clip else None)

        # Additional clipping only if we didn't use windowed reading
        # (windowed reading already clips to ROI, so second clip would remove everything)
        if clip and roi is None:
            # This shouldn't happen, but handle it anyway
            dem = self.loader.clip_to_roi(dem, roi)

        return dem

