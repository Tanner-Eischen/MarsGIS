"""Visualization data export endpoints."""

from fastapi import APIRouter

from marshab.core.raster_service import (
    load_dem_window,  # noqa: F401 - re-exported for test patching
)
from marshab.core.tile_cache import read_disk_cache  # noqa: F401 - re-exported for test patching

from . import basemap, geojson, overlay_routes, terrain
from ._helpers import (
    TILE_CACHE,  # noqa: F401 - re-exported for test patching
    _load_sites_features,  # noqa: F401 - re-exported for test patching
    _load_waypoint_features,  # noqa: F401 - re-exported for test patching
)
from .basemap import get_basemap_tile  # noqa: F401 - re-exported for test patching

router = APIRouter()
router.include_router(basemap.router)
router.include_router(terrain.router)
router.include_router(geojson.router)
router.include_router(overlay_routes.router)

__all__ = [
    "router",
    "load_dem_window",
    "get_basemap_tile",
    "read_disk_cache",
    "TILE_CACHE",
    "_load_waypoint_features",
    "_load_sites_features",
]
