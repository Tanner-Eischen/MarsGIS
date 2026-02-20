"""Data acquisition and management service for Mars DEMs.

Can fetch specific MOLA tiles based on latitude/longitude from PDS-like sources
or fallback to local/synthetic data.
"""

import math
import os
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import Affine, from_bounds

from marshab.config import get_config
from marshab.core.raster_contracts import REAL_DEM_UNAVAILABLE_ERROR_CODE
from marshab.exceptions import DataError
from marshab.models import BoundingBox
from marshab.processing.dem_loader import DEMLoader
from marshab.testing.synthetic_dem import create_synthetic_dem_complex
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_JEZERO_DEM_URL = (
    "https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/"
    "JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif"
)


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

    def _cache_looks_real(self, cache_path: Path) -> bool:
        """Heuristic check whether cached DEM is likely real (not tiny synthetic fallback)."""
        if not cache_path.exists():
            return False

        try:
            with rasterio.open(cache_path) as src:
                tags = src.tags()
                if tags.get("SOURCE_URL"):
                    return True
                # Synthetic fallback tiles are usually very small; real Jezero tile is much denser.
                if src.width * src.height >= 500_000:
                    return True
                # Last-resort heuristic: treat large files as likely real, but only if they open cleanly.
                try:
                    if cache_path.stat().st_size > 2 * 1024 * 1024:
                        return True
                except Exception:
                    return False
        except Exception:
            return False

        return False

    def _download_real_mola_tile(self, roi: BoundingBox, dest_path: Path):
        """Attempt to download real MOLA data, handling global file caching and clipping."""
        allow_synthetic = os.getenv("MARSHAB_ALLOW_SYNTHETIC_TILES", "false").lower() in {"1", "true", "yes"}

        # 1. Check for manual 'real_mars.tif' (dev override)
        manual_file = self.cache_dir / "real_mars.tif"
        if manual_file.exists():
            logger.info("Found manually placed real_mars.tif, using it.")
            tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.copy.tmp")
            shutil.copy(manual_file, tmp_path)
            if dest_path.exists():
                dest_path.unlink()
            tmp_path.replace(dest_path)
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
                    tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.clip.tmp")
                    dem.rio.to_raster(tmp_path)
                    if dest_path.exists():
                        dest_path.unlink()
                    tmp_path.replace(dest_path)
                    logger.info(f"Saved clipped MOLA tile to {dest_path}")
                    return
                else:
                    logger.warning("Loaded DEM missing rio accessor, cannot save to raster.")
            except Exception as e:
                logger.error(f"Failed to clip/save MOLA tile: {e}")
                # Fallback to synthetic

        # 3. If ROI is near Jezero, attempt curated high-resolution tile download.
        # This provides a much better map than synthetic fallback for portfolio demos.
        enable_curated_jezero = os.getenv("MARSHAB_ENABLE_CURATED_JEZERO_DEM", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        if enable_curated_jezero and self._roi_is_jezero_like(roi):
            try:
                logger.info("Attempting curated Jezero DEM download for MOLA request")
                self._download_curated_jezero_tile(dest_path)
                return
            except Exception as e:
                logger.warning("Curated Jezero DEM download failed", error=str(e))

        # NOTE: We do NOT auto-download the 2GB global file here during a tile request.
        # It causes timeouts. The user must initiate global download via Data Settings.
        # Or we use synthetic fallback until then.

        # 4. Final fallback
        if not allow_synthetic:
            raise DataError(
                REAL_DEM_UNAVAILABLE_ERROR_CODE,
                details={
                    "dataset": "mola",
                    "requires_manual_download": True,
                    "hint": "Cache global MOLA (463m) or provide real_mars.tif; synthetic fallback disabled.",
                },
            )
        logger.warning("Real data not available (Global MOLA not cached). Generating realistic synthetic proxy.")
        self._generate_synthetic_proxy(roi, dest_path)

    def _roi_is_jezero_like(self, roi: BoundingBox) -> bool:
        """Return True when ROI intersects the Jezero demo area."""
        # Broad bounds around Jezero crater and delta region.
        jezero_lat_min, jezero_lat_max = 17.8, 18.8
        jezero_lon_min, jezero_lon_max = 76.8, 77.8
        lat_span = roi.lat_max - roi.lat_min
        lon_span = roi.lon_max - roi.lon_min
        # Curated Jezero tile is only valid for small local windows.
        if lat_span > 2.0 or lon_span > 2.0:
            return False
        return not (
            roi.lat_max < jezero_lat_min
            or roi.lat_min > jezero_lat_max
            or roi.lon_max < jezero_lon_min
            or roi.lon_min > jezero_lon_max
        )

    def _looks_geographic(self, src: rasterio.DatasetReader) -> bool:
        """Return True when transform/bounds look like lon/lat in degrees."""
        b = src.bounds
        t = src.transform
        return (
            abs(t.a) < 1.0
            and abs(t.e) < 1.0
            and -90.0 <= b.bottom <= 90.0
            and -90.0 <= b.top <= 90.0
            and 0.0 <= b.left <= 360.0
            and 0.0 <= b.right <= 360.0
        )

    def _download_curated_jezero_tile(self, dest_path: Path) -> None:
        """Download and normalize a curated Jezero DEM tile into cache."""
        url = os.getenv("MARSHAB_JEZERO_DEM_URL", DEFAULT_JEZERO_DEM_URL)
        tmp_download = dest_path.with_suffix(".download.tmp.tif")
        tmp_normalized = dest_path.with_suffix(".normalized.tmp.tif")

        try:
            urllib.request.urlretrieve(url, tmp_download)
            with rasterio.open(tmp_download) as src:
                profile = src.profile.copy()
                transform = src.transform
                if not self._looks_geographic(src):
                    deg_per_m = 180.0 / (math.pi * self.config.mars.equatorial_radius_m)
                    transform = Affine(
                        transform.a * deg_per_m,
                        transform.b * deg_per_m,
                        transform.c * deg_per_m,
                        transform.d * deg_per_m,
                        transform.e * deg_per_m,
                        transform.f * deg_per_m,
                    )

                profile.update(
                    {
                        "transform": transform,
                        "crs": None,
                        "compress": "lzw",
                    }
                )

                with rasterio.open(tmp_normalized, "w", **profile) as dst:
                    for band_idx in range(1, src.count + 1):
                        dst.write(src.read(band_idx), band_idx)
                    tags = dict(src.tags())
                    tags["CRS_INFO"] = "EPSG:49900"
                    tags["SOURCE_URL"] = url
                    dst.update_tags(**tags)

            if dest_path.exists():
                dest_path.unlink()
            tmp_normalized.replace(dest_path)
            logger.info("Curated Jezero DEM cached", path=str(dest_path))
        finally:
            if tmp_download.exists():
                tmp_download.unlink()
            if tmp_normalized.exists():
                tmp_normalized.unlink()

    def _get_demo_seed(self) -> int | None:
        """Get deterministic demo seed from environment, if configured."""
        seed_raw = os.getenv("MARSHAB_DEMO_SEED")
        if not seed_raw:
            return None
        try:
            return int(seed_raw)
        except ValueError:
            logger.warning("Invalid MARSHAB_DEMO_SEED value; expected int", value=seed_raw)
            return None

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
        demo_seed = self._get_demo_seed()
        np_state = None
        py_state = None
        if demo_seed is not None:
            # Keep synthetic fallback deterministic for repeatable portfolio demos.
            np_state = np.random.get_state()
            py_state = random.getstate()
            np.random.seed(demo_seed)
            random.seed(demo_seed)

        try:
            # Create complex terrain
            dem = create_synthetic_dem_complex(
                size=(height_px, width_px),
                features=[
                    {"type": "crater", "center": (height_px//2, width_px//2), "radius": min(height_px, width_px)//4, "depth": 500},
                    {"type": "hill", "center": (height_px//4, width_px//4), "radius": min(height_px, width_px)//6, "height": 300}
                ],
                cell_size_m=463.0
            )
        finally:
            if demo_seed is not None and np_state is not None and py_state is not None:
                np.random.set_state(np_state)
                random.setstate(py_state)

        # Save to GeoTIFF using rasterio directly (works even without rioxarray).
        # Write to a temp file then atomically replace to avoid readers seeing partial files.
        tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.synthetic.tmp")
        try:
            transform = from_bounds(
                roi.lon_min,
                roi.lat_min,
                roi.lon_max,
                roi.lat_max,
                width_px,
                height_px,
            )
            dem_array = dem.values.astype("float32")
            with rasterio.open(
                tmp_path,
                "w",
                driver="GTiff",
                height=height_px,
                width=width_px,
                count=1,
                dtype=dem_array.dtype,
                crs=None,
                transform=transform,
                nodata=-9999.0,
            ) as dst:
                dst.write(dem_array, 1)
                # Preserve Mars CRS hint for loaders/tests when CRS registry support is absent.
                dst.update_tags(CRS_INFO="EPSG:49900")
            if dest_path.exists():
                dest_path.unlink()
            tmp_path.replace(dest_path)
        except Exception as e:
            logger.error("Failed to write synthetic GeoTIFF", error=str(e))
            raise DataError(f"Could not generate synthetic proxy: {e}")
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def download_dem(
        self,
        dataset: Literal["mola", "mola_200m", "hirise", "ctx"],
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

        allow_synthetic = os.getenv("MARSHAB_ALLOW_SYNTHETIC_TILES", "false").lower() in {"1", "true", "yes"}

        if cache_path.exists() and not force:
            # If synthetic is disallowed, never treat a non-real cache as acceptable.
            if not allow_synthetic and not self._cache_looks_real(cache_path):
                try:
                    cache_path.unlink()
                except Exception:
                    pass
            else:
                # Validate the cache opens cleanly (guards against partial/corrupt files).
                try:
                    with rasterio.open(cache_path) as src:
                        _ = src.width, src.height
                except Exception:
                    try:
                        cache_path.unlink()
                    except Exception:
                        pass
                else:
                    # Self-heal Jezero cache: if an old synthetic tile exists, upgrade it to curated real DEM.
                    if dataset == "mola" and roi is not None and self._roi_is_jezero_like(roi):
                        if not self._cache_looks_real(cache_path):
                            logger.info(
                                "Cached Jezero tile appears synthetic; refreshing curated DEM",
                                path=str(cache_path),
                            )
                            self._download_real_mola_tile(roi, cache_path)
                    return cache_path

        if dataset == "mola" and roi:
            self._download_real_mola_tile(roi, cache_path)
            return cache_path

        # Existing logic for fallback/other datasets
        if dataset not in self.config.data_sources:
            raise DataError(f"Unknown dataset: {dataset}")

        source = self.config.data_sources[dataset]
        url = source.url

        # Directory-based logic (HiRISE/CTX) from original code...
        if url.endswith('/'):
            if roi:
                if not allow_synthetic:
                    raise DataError(
                        REAL_DEM_UNAVAILABLE_ERROR_CODE,
                        details={
                            "dataset": dataset,
                            "requires_manual_download": True,
                            "hint": "Dataset requires manual download; synthetic fallback disabled.",
                        },
                    )
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
                if not allow_synthetic:
                    raise DataError(
                        REAL_DEM_UNAVAILABLE_ERROR_CODE,
                        details={
                            "dataset": dataset,
                            "requires_manual_download": True,
                            "hint": "Download failed and synthetic fallback is disabled.",
                            "error": str(e),
                        },
                    )
                logger.error(f"Download failed, falling back to synthetic: {e}")
                self._generate_synthetic_proxy(roi, cache_path)
                return cache_path
            raise DataError(f"Download failed: {e}")

    def load_dem(self, path: Path) -> xr.DataArray:
        return self.loader.load(path)

    def get_dem_for_roi(
        self,
        roi: BoundingBox,
        dataset: Literal["mola", "mola_200m", "hirise", "ctx"] = "mola",
        download: bool = True,
        clip: bool = True,
    ) -> xr.DataArray:
        if download:
            dem_path = self.download_dem(dataset, roi)
        else:
            dem_path = self._get_cache_path(dataset, roi)
            if not dem_path.exists():
                # Attempt generation if missing
                dem_path = self.download_dem(dataset, roi)  # Assign return value

        dem = self.loader.load(dem_path, roi=roi if clip else None)
        if clip and roi is not None:
            dem = self.loader.clip_to_roi(dem, roi)
        return dem
