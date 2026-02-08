"""Tile cache utilities for basemap/overlay tiles."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from marshab.config import get_config


@dataclass
class TileCacheConfig:
    max_entries: int = 4000
    ttl_seconds: int = 7 * 24 * 60 * 60


class TileCache:
    def __init__(self, config: Optional[TileCacheConfig] = None) -> None:
        self.config = config or TileCacheConfig()
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
        if len(self._cache) > self.config.max_entries:
            self._cache.popitem(last=False)


def tile_cache_path(kind: str, dataset: str, z: int, x: int, y: int, style_hash: str) -> Path:
    config = get_config()
    return (
        config.paths.cache_dir
        / "tiles"
        / kind
        / dataset
        / str(z)
        / str(x)
        / f"{y}_{style_hash}.png"
    )


def read_disk_cache(path: Path, ttl_seconds: int) -> Optional[bytes]:
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > ttl_seconds:
            return None
        return path.read_bytes()
    except Exception:
        return None


def write_disk_cache(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
