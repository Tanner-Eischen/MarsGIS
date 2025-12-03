"""Overlay image caching system."""

import hashlib
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

from marshab.config import get_config
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum cache size in bytes (500MB default)
MAX_CACHE_SIZE = 500 * 1024 * 1024


class OverlayCache:
    """Manages caching of overlay images with LRU eviction."""

    def __init__(self, max_size_bytes: int = MAX_CACHE_SIZE):
        """Initialize overlay cache.

        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self.config = get_config()
        self.cache_dir = self.config.paths.cache_dir / "overlays"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata: Dict[str, Dict] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load cache metadata", error=str(e))
        return {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save cache metadata", error=str(e))

    def _get_cache_key(
        self,
        overlay_type: str,
        dataset: str,
        roi: BoundingBox,
        colormap: str = "terrain",
        relief: float = 0.0,
        sun_azimuth: float = 315.0,
        sun_altitude: float = 45.0,
        width: int = 800,
        height: int = 600,
    ) -> str:
        """Generate cache key for overlay parameters.

        Args:
            overlay_type: Type of overlay (elevation, solar, hillshade, etc.)
            dataset: Dataset identifier
            roi: Region of interest
            colormap: Colormap name
            relief: Relief/hillshade intensity
            sun_azimuth: Sun azimuth angle
            sun_altitude: Sun altitude angle
            width: Image width
            height: Image height

        Returns:
            Cache key string
        """
        roi_str = f"{roi.lat_min}_{roi.lat_max}_{roi.lon_min}_{roi.lon_max}"
        key_str = (
            f"{overlay_type}_{dataset}_{roi_str}_{colormap}_"
            f"{relief}_{sun_azimuth}_{sun_altitude}_{width}_{height}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, overlay_type: str) -> Path:
        """Get file path for cached overlay.

        Args:
            cache_key: Cache key
            overlay_type: Overlay type (for directory organization)

        Returns:
            Path to cached file
        """
        overlay_dir = self.cache_dir / overlay_type
        overlay_dir.mkdir(parents=True, exist_ok=True)
        return overlay_dir / f"{cache_key}.png"

    def get(
        self,
        overlay_type: str,
        dataset: str,
        roi: BoundingBox,
        colormap: str = "terrain",
        relief: float = 0.0,
        sun_azimuth: float = 315.0,
        sun_altitude: float = 45.0,
        width: int = 800,
        height: int = 600,
    ) -> Optional[Path]:
        """Get cached overlay if available.

        Args:
            overlay_type: Type of overlay
            dataset: Dataset identifier
            roi: Region of interest
            colormap: Colormap name
            relief: Relief/hillshade intensity
            sun_azimuth: Sun azimuth angle
            sun_altitude: Sun altitude angle
            width: Image width
            height: Image height

        Returns:
            Path to cached file if exists, None otherwise
        """
        cache_key = self._get_cache_key(
            overlay_type, dataset, roi, colormap, relief, sun_azimuth, sun_altitude, width, height
        )
        cache_path = self._get_cache_path(cache_key, overlay_type)

        if cache_path.exists():
            # Update access time in metadata
            if cache_key in self._metadata:
                self._metadata[cache_key]["access_count"] = (
                    self._metadata[cache_key].get("access_count", 0) + 1
                )
                self._save_metadata()
            logger.debug("Cache hit", cache_key=cache_key[:8], path=str(cache_path))
            return cache_path

        logger.debug("Cache miss", cache_key=cache_key[:8])
        return None

    def put(
        self,
        overlay_type: str,
        dataset: str,
        roi: BoundingBox,
        image_bytes: bytes,
        colormap: str = "terrain",
        relief: float = 0.0,
        sun_azimuth: float = 315.0,
        sun_altitude: float = 45.0,
        width: int = 800,
        height: int = 600,
    ) -> Path:
        """Store overlay in cache.

        Args:
            overlay_type: Type of overlay
            dataset: Dataset identifier
            roi: Region of interest
            image_bytes: PNG image bytes
            colormap: Colormap name
            relief: Relief/hillshade intensity
            sun_azimuth: Sun azimuth angle
            sun_altitude: Sun altitude angle
            width: Image width
            height: Image height

        Returns:
            Path to cached file
        """
        cache_key = self._get_cache_key(
            overlay_type, dataset, roi, colormap, relief, sun_azimuth, sun_altitude, width, height
        )
        cache_path = self._get_cache_path(cache_key, overlay_type)

        # Check cache size and evict if needed
        self._evict_if_needed(len(image_bytes))

        # Write image file
        cache_path.write_bytes(image_bytes)

        # Update metadata
        self._metadata[cache_key] = {
            "overlay_type": overlay_type,
            "dataset": dataset,
            "roi": roi.model_dump(),
            "colormap": colormap,
            "relief": relief,
            "sun_azimuth": sun_azimuth,
            "sun_altitude": sun_altitude,
            "width": width,
            "height": height,
            "size_bytes": len(image_bytes),
            "access_count": 1,
        }
        self._save_metadata()

        logger.info(
            "Cached overlay",
            cache_key=cache_key[:8],
            overlay_type=overlay_type,
            size_bytes=len(image_bytes),
        )
        return cache_path

    def _evict_if_needed(self, new_size: int):
        """Evict least recently used items if cache exceeds max size.

        Args:
            new_size: Size of new item to be added
        """
        current_size = self._get_cache_size()
        if current_size + new_size <= self.max_size_bytes:
            return

        # Sort by access count (LRU)
        items = []
        for cache_key, metadata in self._metadata.items():
            overlay_type = metadata.get("overlay_type", "unknown")
            cache_path = self._get_cache_path(cache_key, overlay_type)
            if cache_path.exists():
                items.append(
                    (
                        cache_key,
                        metadata.get("access_count", 0),
                        cache_path.stat().st_size,
                        cache_path,
                    )
                )

        # Sort by access count (ascending = least used first)
        items.sort(key=lambda x: x[1])

        # Evict until we have enough space
        freed = 0
        target_free = current_size + new_size - self.max_size_bytes
        evicted = []

        for cache_key, _, size, cache_path in items:
            if freed >= target_free:
                break
            try:
                cache_path.unlink()
                del self._metadata[cache_key]
                freed += size
                evicted.append(cache_key[:8])
            except Exception as e:
                logger.warning("Failed to evict cache item", cache_key=cache_key[:8], error=str(e))

        if evicted:
            self._save_metadata()
            logger.info(
                "Evicted cache items",
                count=len(evicted),
                freed_bytes=freed,
                evicted_keys=evicted,
            )

    def _get_cache_size(self) -> int:
        """Calculate total cache size in bytes."""
        total = 0
        for cache_key, metadata in self._metadata.items():
            overlay_type = metadata.get("overlay_type", "unknown")
            cache_path = self._get_cache_path(cache_key, overlay_type)
            if cache_path.exists():
                total += cache_path.stat().st_size
        return total

    def clear(self, overlay_type: Optional[str] = None):
        """Clear cache, optionally filtered by overlay type.

        Args:
            overlay_type: If provided, only clear this overlay type
        """
        if overlay_type:
            overlay_dir = self.cache_dir / overlay_type
            if overlay_dir.exists():
                shutil.rmtree(overlay_dir)
                # Remove from metadata
                self._metadata = {
                    k: v
                    for k, v in self._metadata.items()
                    if v.get("overlay_type") != overlay_type
                }
                self._save_metadata()
                logger.info("Cleared cache", overlay_type=overlay_type)
        else:
            # Clear all
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._metadata = {}
                self._save_metadata()
                logger.info("Cleared all overlay cache")

