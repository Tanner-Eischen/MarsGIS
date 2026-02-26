"""Cache helpers for DEM window GeoTIFF responses."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from marshab.config import get_config
from marshab.models import BoundingBox


@dataclass
class DemWindowCacheConfig:
    max_entries: int = 512
    ttl_seconds: int = 12 * 60 * 60


class DemWindowMemoryCache:
    def __init__(self, config: Optional[DemWindowCacheConfig] = None) -> None:
        self.config = config or DemWindowCacheConfig()
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    def get(self, key: str) -> Optional[bytes]:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def set(self, key: str, value: bytes) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self.config.max_entries:
            self._cache.popitem(last=False)


def _roi_signature(roi: BoundingBox) -> str:
    payload = {
        "lat_min": round(float(roi.lat_min), 6),
        "lat_max": round(float(roi.lat_max), 6),
        "lon_min": round(float(roi.lon_min), 6),
        "lon_max": round(float(roi.lon_max), 6),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def dem_window_cache_key(
    dataset: str,
    roi: BoundingBox,
    width: int,
    height: int,
    dem_signature: str,
) -> str:
    return f"{dataset}:{_roi_signature(roi)}:{width}x{height}:{dem_signature}"


def dem_window_cache_path(cache_key: str) -> Path:
    config = get_config()
    shard = cache_key[:2]
    return config.paths.cache_dir / "windows" / shard / f"{cache_key}.tif"


def read_dem_window_disk_cache(path: Path, ttl_seconds: int) -> Optional[bytes]:
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > ttl_seconds:
            return None
        return path.read_bytes()
    except Exception:
        return None


def write_dem_window_disk_cache(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_bytes(data)
    if path.exists():
        path.unlink()
    tmp_path.replace(path)
