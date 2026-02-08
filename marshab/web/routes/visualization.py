"""Visualization data export endpoints."""

import asyncio
import io
import json
import os
import time
import urllib.request

import numpy as np
import xarray as xr
from rasterio.transform import Affine, rowcol
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from marshab.analysis.solar_potential import SolarPotentialAnalyzer
from marshab.config import get_config
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.data_manager import DataManager
from marshab.core.overlay_cache import OverlayCache
from marshab.core.raster_service import (
    compute_tile_bbox_epsg4326,
    load_dem_window,
    normalize_dataset,
    resolve_dataset_with_fallback,
    tile_style_hash,
    to_lon360,
)
from marshab.core.tile_cache import TileCache, TileCacheConfig, read_disk_cache, tile_cache_path, write_disk_cache
from marshab.models import BoundingBox
from marshab.processing.terrain import TerrainAnalyzer
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()
MARS_BASEMAP_TILE_URL = (
    "https://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/"
    "celestia_mars-shaded-16k_global/{z}/{x}/{y}.png"
)

TILE_CACHE = TileCache(
    TileCacheConfig(
        max_entries=int(os.getenv("MARSHAB_TILE_CACHE_MAX", "4000")),
        ttl_seconds=int(os.getenv("MARSHAB_TILE_CACHE_TTL", str(7 * 24 * 60 * 60))),
    )
)
ENABLE_TILE_BASEMAP = os.getenv("MARSHAB_ENABLE_TILE_BASEMAP", "true").lower() in {"1", "true", "yes"}
ENABLE_TILE_OVERLAYS = os.getenv("MARSHAB_ENABLE_TILE_OVERLAYS", "true").lower() in {"1", "true", "yes"}


def _build_dem_dataarray(raster_result) -> xr.DataArray:
    rows, cols = raster_result.array.shape
    lat_coords = np.linspace(raster_result.bbox_used.lat_max, raster_result.bbox_used.lat_min, rows)
    lon_coords = np.linspace(raster_result.bbox_used.lon_min, raster_result.bbox_used.lon_max, cols)
    data = xr.DataArray(
        raster_result.array,
        dims=("lat", "lon"),
        coords={"lat": lat_coords, "lon": lon_coords},
        attrs={
            "nodata": raster_result.nodata,
            "transform": raster_result.transform,
            "dataset": raster_result.dataset_used,
        },
    )
    return data


def _log_metric(name: str, value: float = 1.0, **fields) -> None:
    logger.info("metric", metric=name, value=value, **fields)


@router.get("/visualization/basemap/{z}/{x}/{y}.png")
async def get_mars_basemap_tile(z: int, x: int, y: int):
    """Proxy Mars basemap tiles through same-origin API to avoid browser CORS issues."""
    if z < 0 or x < 0 or y < 0:
        raise HTTPException(status_code=400, detail="Invalid tile coordinates")

    tile_url = MARS_BASEMAP_TILE_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(tile_url, headers={"User-Agent": "MarsHab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            tile_bytes = resp.read()
            if not tile_bytes:
                raise HTTPException(status_code=404, detail="Tile not found")
            response = Response(content=tile_bytes, media_type="image/png")
            # Allow light CDN/proxy caching while keeping refresh reasonably quick.
            response.headers["Cache-Control"] = "public, max-age=86400"
            return response
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Basemap tile proxy failed", z=z, x=x, y=y, error=str(e))
        raise HTTPException(status_code=502, detail="Failed to fetch basemap tile")


@router.get("/visualization/tiles/basemap/{dataset}/{z}/{x}/{y}.png")
async def get_basemap_tile(dataset: str, z: int, x: int, y: int):
    """Render DEM-backed basemap tiles."""
    if not ENABLE_TILE_BASEMAP:
        raise HTTPException(status_code=404, detail="Basemap tiling disabled")

    try:
        dataset_lower = normalize_dataset(dataset)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset. Use mola, mola_200m, or hirise")

    if dataset_lower not in {"mola", "mola_200m", "hirise"}:
        raise HTTPException(status_code=400, detail="Invalid dataset. Use mola, mola_200m, or hirise")

    _log_metric("tile.request.count", 1, kind="basemap")

    try:
        bbox = compute_tile_bbox_epsg4326(z, x, y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    resolution = resolve_dataset_with_fallback(dataset_lower, bbox)
    dataset_cache = resolution.dataset_used

    data_manager = DataManager()
    dem_cache_buster = "none"
    try:
        bbox_cache = BoundingBox(
            lat_min=bbox.lat_min,
            lat_max=bbox.lat_max,
            lon_min=to_lon360(bbox.lon_min),
            lon_max=to_lon360(bbox.lon_max),
        )
        cache_path = data_manager._get_cache_path(dataset_cache, bbox_cache)
        if cache_path.exists():
            dem_cache_buster = str(int(cache_path.stat().st_mtime))
    except Exception:
        pass

    style_params = {
        "colormap": "terrain",
        "relief": "0.3",
        "sun_azimuth": "315",
        "sun_altitude": "45",
    }
    style_hash = tile_style_hash(style_params)
    cache_key = f"basemap::{dataset_cache}::{z}:{x}:{y}:{style_hash}:{dem_cache_buster}"

    cached = TILE_CACHE.get(cache_key)
    if cached:
        _log_metric("tile.cache.hit.memory", 1, kind="basemap")
        response = Response(content=cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Dataset-Requested"] = resolution.dataset_requested
        response.headers["X-Dataset-Used"] = resolution.dataset_used
        response.headers["X-Fallback-Used"] = str(resolution.is_fallback).lower()
        if resolution.fallback_reason:
            response.headers["X-Fallback-Reason"] = resolution.fallback_reason
        return response

    disk_path = tile_cache_path("basemap", dataset_cache, z, x, y, style_hash)
    disk_cached = read_disk_cache(disk_path, TILE_CACHE.config.ttl_seconds)
    if disk_cached:
        _log_metric("tile.cache.hit.disk", 1, kind="basemap")
        TILE_CACHE.set(cache_key, disk_cached)
        response = Response(content=disk_cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Dataset-Requested"] = resolution.dataset_requested
        response.headers["X-Dataset-Used"] = resolution.dataset_used
        response.headers["X-Fallback-Used"] = str(resolution.is_fallback).lower()
        if resolution.fallback_reason:
            response.headers["X-Fallback-Reason"] = resolution.fallback_reason
        return response

    _log_metric("tile.cache.miss", 1, kind="basemap")
    start = time.time()
    try:
        raster_result = load_dem_window(
            dataset_lower,
            bbox,
            target_shape=(256, 256),
            allow_download=True,
        )
        dem = _build_dem_dataarray(raster_result)
        png_bytes, _bounds, _elev_min, _elev_max = await _generate_elevation_overlay(
            dem,
            bbox,
            "terrain",
            0.3,
            315.0,
            45.0,
            256,
            256,
            False,
        )
    except HTTPException:
        _log_metric("tile.error.5xx", 1, kind="basemap")
        raise
    except Exception as exc:
        _log_metric("tile.error.5xx", 1, kind="basemap")
        logger.exception("Basemap tile render failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to render basemap tile")

    duration_ms = (time.time() - start) * 1000
    _log_metric("tile.render.ms", duration_ms, kind="basemap")

    TILE_CACHE.set(cache_key, png_bytes)
    write_disk_cache(disk_path, png_bytes)

    response = Response(content=png_bytes, media_type="image/png")
    response.headers["Cache-Control"] = "public, max-age=86400"
    response.headers["X-Dataset-Requested"] = raster_result.dataset_requested
    response.headers["X-Dataset-Used"] = raster_result.dataset_used
    response.headers["X-Fallback-Used"] = str(raster_result.is_fallback).lower()
    if raster_result.is_fallback:
        _log_metric("tile.fallback.rate", 1, kind="basemap")
    if raster_result.fallback_reason:
        response.headers["X-Fallback-Reason"] = raster_result.fallback_reason
    return response


@router.get("/visualization/tiles/overlay/{overlay_type}/{dataset}/{z}/{x}/{y}.png")
async def get_overlay_tile(
    overlay_type: str,
    dataset: str,
    z: int,
    x: int,
    y: int,
    colormap: str = Query("terrain"),
    relief: float = Query(0.0, ge=0.0, le=3.0),
    sun_azimuth: float = Query(315.0, ge=0.0, le=360.0),
    sun_altitude: float = Query(45.0, ge=0.0, le=90.0),
    mars_sol: int | None = Query(None),
    season: str | None = Query(None),
    dust_storm_period: str | None = Query(None),
):
    """Render overlay tiles backed by DEM-derived layers."""
    if not ENABLE_TILE_OVERLAYS:
        raise HTTPException(status_code=404, detail="Overlay tiling disabled")

    valid_types = ["elevation", "hillshade", "slope", "aspect", "roughness", "tri", "solar", "dust"]
    overlay_type = overlay_type.lower()
    if overlay_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid overlay type: {overlay_type}")

    try:
        dataset_lower = normalize_dataset(dataset)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset. Use mola, mola_200m, hirise, or ctx")

    _log_metric("tile.request.count", 1, kind="overlay", overlay_type=overlay_type)

    try:
        bbox = compute_tile_bbox_epsg4326(z, x, y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    resolution = resolve_dataset_with_fallback(dataset_lower, bbox)
    dataset_cache = resolution.dataset_used

    data_manager = DataManager()
    dem_cache_buster = "none"
    try:
        bbox_cache = BoundingBox(
            lat_min=bbox.lat_min,
            lat_max=bbox.lat_max,
            lon_min=to_lon360(bbox.lon_min),
            lon_max=to_lon360(bbox.lon_max),
        )
        cache_path = data_manager._get_cache_path(dataset_cache, bbox_cache)
        if cache_path.exists():
            dem_cache_buster = str(int(cache_path.stat().st_mtime))
    except Exception:
        pass

    style_params = {
        "colormap": colormap,
        "relief": f"{relief:.2f}",
        "sun_azimuth": f"{sun_azimuth:.1f}",
        "sun_altitude": f"{sun_altitude:.1f}",
        "mars_sol": str(mars_sol) if mars_sol is not None else "",
        "season": season or "",
        "dust_storm_period": dust_storm_period or "",
    }
    style_hash = tile_style_hash(style_params)
    cache_key = f"overlay::{overlay_type}:{dataset_cache}:{z}:{x}:{y}:{style_hash}:{dem_cache_buster}"

    cached = TILE_CACHE.get(cache_key)
    if cached:
        _log_metric("tile.cache.hit.memory", 1, kind="overlay", overlay_type=overlay_type)
        response = Response(content=cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Dataset-Requested"] = resolution.dataset_requested
        response.headers["X-Dataset-Used"] = resolution.dataset_used
        response.headers["X-Fallback-Used"] = str(resolution.is_fallback).lower()
        if resolution.fallback_reason:
            response.headers["X-Fallback-Reason"] = resolution.fallback_reason
        return response

    disk_path = tile_cache_path("overlay", dataset_cache, z, x, y, style_hash)
    disk_cached = read_disk_cache(disk_path, TILE_CACHE.config.ttl_seconds)
    if disk_cached:
        _log_metric("tile.cache.hit.disk", 1, kind="overlay", overlay_type=overlay_type)
        TILE_CACHE.set(cache_key, disk_cached)
        response = Response(content=disk_cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Dataset-Requested"] = resolution.dataset_requested
        response.headers["X-Dataset-Used"] = resolution.dataset_used
        response.headers["X-Fallback-Used"] = str(resolution.is_fallback).lower()
        if resolution.fallback_reason:
            response.headers["X-Fallback-Reason"] = resolution.fallback_reason
        return response

    _log_metric("tile.cache.miss", 1, kind="overlay", overlay_type=overlay_type)
    start = time.time()
    try:
        raster_result = load_dem_window(
            dataset_lower,
            bbox,
            target_shape=(256, 256),
            allow_download=True,
        )
        dem = _build_dem_dataarray(raster_result)
        loop = asyncio.get_event_loop()
        if overlay_type == "elevation":
            png_bytes, _bounds, _elev_min, _elev_max = await _generate_elevation_overlay(
                dem,
                bbox,
                colormap,
                relief,
                sun_azimuth,
                sun_altitude,
                256,
                256,
                False,
            )
        elif overlay_type == "solar":
            png_bytes, _bounds = await _generate_solar_overlay(
                dem,
                bbox,
                raster_result.resolution_m or 200.0,
                sun_azimuth,
                sun_altitude,
                256,
                256,
                colormap,
                False,
                loop,
                None,
            )
        elif overlay_type == "dust":
            png_bytes, _bounds = await _generate_dust_overlay(
                bbox,
                256,
                256,
                colormap,
                False,
                loop,
            )
        elif overlay_type == "hillshade":
            png_bytes, _bounds = await _generate_hillshade_overlay(
                dem,
                bbox,
                raster_result.resolution_m or 200.0,
                sun_azimuth,
                sun_altitude,
                256,
                256,
                False,
                loop,
            )
        else:
            png_bytes, _bounds = await _generate_terrain_metric_overlay(
                overlay_type,
                dem,
                bbox,
                raster_result.resolution_m or 200.0,
                colormap,
                256,
                256,
                False,
                loop,
            )
    except Exception as exc:
        _log_metric("tile.error.5xx", 1, kind="overlay", overlay_type=overlay_type)
        logger.exception("Overlay tile render failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to render overlay tile")

    duration_ms = (time.time() - start) * 1000
    _log_metric("tile.render.ms", duration_ms, kind="overlay", overlay_type=overlay_type)

    TILE_CACHE.set(cache_key, png_bytes)
    write_disk_cache(disk_path, png_bytes)

    response = Response(content=png_bytes, media_type="image/png")
    response.headers["Cache-Control"] = "public, max-age=86400"
    response.headers["X-Dataset-Requested"] = raster_result.dataset_requested
    response.headers["X-Dataset-Used"] = raster_result.dataset_used
    response.headers["X-Fallback-Used"] = str(raster_result.is_fallback).lower()
    if raster_result.is_fallback:
        _log_metric("tile.fallback.rate", 1, kind="overlay", overlay_type=overlay_type)
    if raster_result.fallback_reason:
        response.headers["X-Fallback-Reason"] = raster_result.fallback_reason
    return response


@router.get("/visualization/elevation-at")
async def get_elevation_at(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude in degrees"),
    lon: float = Query(..., ge=-180.0, le=360.0, description="Longitude in degrees"),
    dataset: str = Query("mola", description="Dataset name (mola, mola_200m, hirise, ctx)"),
    window_deg: float = Query(
        0.05,
        gt=0.0,
        le=2.0,
        description="Half-window size in degrees used to load DEM around the point",
    ),
):
    """Sample numeric elevation from DEM at a clicked map coordinate."""
    try:
        dataset_lower = normalize_dataset(dataset)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset. Use one of: mola, mola_200m, hirise, ctx")

    # Keep internal lon domain consistent with DEM loaders (0..360).
    lon_360 = to_lon360(lon)

    bbox = BoundingBox(
        lat_min=max(-90.0, lat - window_deg),
        lat_max=min(90.0, lat + window_deg),
        lon_min=max(0.0, lon_360 - window_deg),
        lon_max=min(360.0, lon_360 + window_deg),
    )
    try:
        raster_result = load_dem_window(dataset_lower, bbox)
        elevation = raster_result.array.astype(np.float32)
        if elevation.size == 0:
            raise HTTPException(status_code=404, detail="No DEM data available at this location")

        rows, cols = elevation.shape
        sample_row = rows // 2
        sample_col = cols // 2

        transform_values = raster_result.transform
        if isinstance(transform_values, (list, tuple)) and len(transform_values) >= 6:
            try:
                transform = Affine(*transform_values[:6])
                sample_row, sample_col = rowcol(transform, lon_360, lat, op=round)
            except Exception:
                sample_row, sample_col = rows // 2, cols // 2

        sample_row = int(max(0, min(rows - 1, sample_row)))
        sample_col = int(max(0, min(cols - 1, sample_col)))

        nodata = raster_result.nodata
        valid_mask = np.isfinite(elevation)
        if nodata is not None:
            valid_mask &= elevation != nodata
        if not np.any(valid_mask):
            raise HTTPException(status_code=404, detail="DEM sample contains no valid elevation values")

        sampled_value = elevation[sample_row, sample_col]
        if (not np.isfinite(sampled_value)) or (nodata is not None and sampled_value == nodata):
            valid_indices = np.argwhere(valid_mask)
            distances = (valid_indices[:, 0] - sample_row) ** 2 + (valid_indices[:, 1] - sample_col) ** 2
            nearest_idx = int(np.argmin(distances))
            sample_row, sample_col = map(int, valid_indices[nearest_idx])
            sampled_value = elevation[sample_row, sample_col]

        valid_values = elevation[valid_mask]
        return {
            "dataset": raster_result.dataset_requested,
            "dataset_used": raster_result.dataset_used,
            "is_fallback": raster_result.is_fallback,
            "fallback_reason": raster_result.fallback_reason,
            "lat": float(lat),
            "lon": float(lon),
            "lon_360": float(lon_360),
            "elevation_m": float(sampled_value),
            "pixel_row": sample_row,
            "pixel_col": sample_col,
            "grid_shape": [int(rows), int(cols)],
            "window_deg": float(window_deg),
            "elevation_min_m": float(np.nanmin(valid_values)),
            "elevation_max_m": float(np.nanmax(valid_values)),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to sample DEM elevation", lat=lat, lon=lon, dataset=dataset_lower)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/sites/{analysis_id}")
async def get_sites_file(analysis_id: str):
    """Get sites CSV file for visualization."""
    try:
        config = get_config()
        output_dir = config.paths.output_dir

        # For now, use default sites.csv
        # In future, could support multiple analysis runs with IDs
        sites_file = output_dir / "sites.csv"

        if not sites_file.exists():
            raise HTTPException(status_code=404, detail="Sites file not found. Run analysis first.")

        return FileResponse(
            path=str(sites_file),
            filename="sites.csv",
            media_type="text/csv",
        )
    except Exception as e:
        logger.exception("Failed to get sites file")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/waypoints/{navigation_id}")
async def get_waypoints_file(navigation_id: str):
    """Get waypoints CSV file for visualization."""
    try:
        config = get_config()
        output_dir = config.paths.output_dir

        # For now, use default waypoints.csv
        waypoints_file = output_dir / "waypoints.csv"

        if not waypoints_file.exists():
            raise HTTPException(status_code=404, detail="Waypoints file not found. Plan navigation first.")

        return FileResponse(
            path=str(waypoints_file),
            filename="waypoints.csv",
            media_type="text/csv",
        )
    except Exception as e:
        logger.exception("Failed to get waypoints file")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/dem-image")
async def get_dem_image(
    dataset: str = Query(..., description="Dataset name (mola, mola_200m, hirise, ctx)"),
    roi: str = Query(..., description="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    colormap: str = Query("terrain", description="Colormap name (terrain, viridis, plasma, etc.)"),
    relief: float = Query(
        0.0,
        ge=0.0,
        le=3.0,
        description="Relief / hillshade intensity (0 = flat colors, higher = more 3D shading)",
    ),
    sun_azimuth: float = Query(
        315.0,
        ge=0.0,
        le=360.0,
        description="Sun azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)",
    ),
    sun_altitude: float = Query(
        45.0,
        ge=0.0,
        le=90.0,
        description="Sun altitude angle in degrees (0=horizon, 90=zenith)",
    ),
    width: int = Query(800, ge=100, le=4000, description="Image width in pixels"),
    height: int = Query(600, ge=100, le=4000, description="Image height in pixels"),
    buffer: float = Query(0.5, ge=0.0, le=5.0, description="Buffer factor to extend ROI (0.5 = 50% extension, 1.0 = 100% extension, 5.0 = 500% extension)"),
):
    """Get DEM as PNG image for visualization.

    Returns a PNG image of the DEM elevation data, cropped to the specified ROI.
    """
    start = time.time()
    logger.info(
        "DEM image request",
        dataset=dataset,
        roi=roi,
        width=width,
        height=height
    )
    try:
        # Parse ROI
        try:
            lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
            original_bbox = BoundingBox(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )
        except Exception as e:
            logger.error("Invalid ROI format", roi=roi, error=str(e))
            raise HTTPException(status_code=400, detail=f"Invalid ROI format: {e}")

        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        if lat_span <= 0 or lon_span <= 0:
            raise HTTPException(status_code=400, detail="Invalid ROI bounds: max must be greater than min")

        try:
            dataset_lower = normalize_dataset(dataset)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid dataset. Must be one of: mola, mola_200m, ctx, hirise",
            )

        # Prevent long-running overlay jobs that often surface in browser as CORS header errors
        # when the upstream platform returns 502/timeout.
        is_low_res = dataset_lower in {"mola", "mola_200m"}
        max_lat_span = 4.0 if is_low_res else 1.2
        max_lon_span = 4.0 if is_low_res else 1.2
        max_area_deg2 = 12.0 if is_low_res else 1.5
        if lat_span > max_lat_span or lon_span > max_lon_span or (lat_span * lon_span) > max_area_deg2:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"ROI too large for interactive overlay ({lat_span:.2f} x {lon_span:.2f} deg). "
                    "Zoom in or reduce ROI."
                ),
            )

        if width * height > 2_200_000:
            raise HTTPException(
                status_code=413,
                detail="Requested overlay image is too large. Reduce viewport size or zoom level.",
            )

        lon_min_norm = to_lon360(lon_min)
        lon_max_norm = to_lon360(lon_max)
        if lon_max_norm <= lon_min_norm:
            lon_max_norm = min(360.0, lon_min_norm + max(lon_span, 0.01))

        # Extend ROI bounds if buffer is specified
        if buffer > 0:
            delta_lat = (lat_max - lat_min) * buffer
            delta_lon = (lon_max_norm - lon_min_norm) * buffer
            extended_bbox = BoundingBox(
                lat_min=max(-90, lat_min - delta_lat),
                lat_max=min(90, lat_max + delta_lat),
                lon_min=max(0, lon_min_norm - delta_lon),
                lon_max=min(360, lon_max_norm + delta_lon),
            )
        else:
            extended_bbox = BoundingBox(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min_norm,
                lon_max=lon_max_norm,
            )

        logger.info("Loading DEM data...", bbox=extended_bbox.model_dump(), dataset=dataset)
        # Load DEM with extended bounds (allow download if not cached)
        # Run in thread pool to avoid blocking async handler
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        use_synthetic = False
        raster_result = None
        try:
            # Increase timeout for large DEMs, but add progress logging
            logger.info("Starting DEM load in thread pool...")
            with ThreadPoolExecutor() as executor:
                raster_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        lambda: load_dem_window(
                            dataset_lower,
                            extended_bbox,
                            target_shape=(height, width),
                            allow_download=True,
                        ),
                    ),
                    timeout=120.0
                )
            logger.info("DEM loaded successfully", shape=raster_result.array.shape, dtype=str(raster_result.array.dtype))
        except asyncio.TimeoutError:
            logger.warning("DEM loading timed out, generating synthetic visualization for demo mode", bbox=extended_bbox.model_dump())
            use_synthetic = True
        except Exception as e:
            logger.warning("DEM load failed, generating synthetic visualization for demo mode", error=str(e), bbox=extended_bbox.model_dump())
            use_synthetic = True

        logger.info("Processing elevation data...")
        # Get elevation data (or synthesize if DEM unavailable)
        if not use_synthetic and raster_result is not None:
            elevation = raster_result.array.astype(np.float32)
        else:
            # Create a synthetic elevation surface for visualization (fast demo fallback)
            synth_h = max(100, min(height, 600)) if 'height' in locals() else 300
            synth_w = max(100, min(width, 800)) if 'width' in locals() else 400
            y = np.linspace(0, 1, synth_h)
            x = np.linspace(0, 1, synth_w)
            xx, yy = np.meshgrid(x, y)
            elevation = (np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 1000.0
            elevation = elevation.astype(np.float32)

        # Handle nodata values
        if not use_synthetic and raster_result is not None and raster_result.nodata is not None:
            nodata = raster_result.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

        # Normalize elevation to 0-255 range for image
        valid_mask = np.isfinite(elevation)
        if not np.any(valid_mask):
            logger.error("No valid elevation data in ROI", roi=roi)
            raise HTTPException(status_code=400, detail="No valid elevation data in ROI")

        valid_elevation = elevation[valid_mask]
        elev_min = float(np.nanmin(valid_elevation))
        elev_max = float(np.nanmax(valid_elevation))

        logger.info("Elevation range", elev_min=elev_min, elev_max=elev_max)

        if elev_max == elev_min:
            # Constant elevation, create uniform image
            normalized = np.ones_like(elevation, dtype=np.uint8) * 128
        else:
            # Normalize to 0-255
            normalized = ((elevation - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)
            normalized[~valid_mask] = 0  # Set invalid pixels to black

        # Resize if needed - run in thread pool for large images
        if not use_synthetic and normalized.shape[0] != height or not use_synthetic and normalized.shape[1] != width:
            logger.info("Resizing image", from_shape=normalized.shape, to_size=(height, width))
            from PIL import Image as PILImage

            def resize_image(arr, target_size):
                img = PILImage.fromarray(arr, mode='L')
                img = img.resize((target_size[1], target_size[0]), PILImage.Resampling.LANCZOS)
                return np.array(img)

            with ThreadPoolExecutor() as executor:
                normalized = await loop.run_in_executor(
                    executor,
                    resize_image,
                    normalized,
                    (height, width)
                )

        logger.info("Applying colormap...")
        # Apply colormap - run in thread pool for large images
        def apply_colormap(arr, cmap_name, relief_val, sun_az, sun_alt):
            try:
                import matplotlib
                import matplotlib.cm as cm
                # Use the new API if available, fallback to deprecated get_cmap
                try:
                    colormap_func = cm.get_cmap(cmap_name)
                except (AttributeError, ValueError):
                    # Fallback for newer matplotlib versions
                    try:
                        colormap_func = matplotlib.colormaps.get_cmap(cmap_name)
                    except (AttributeError, ValueError):
                        # Final fallback: use terrain colormap
                        colormap_func = cm.get_cmap('terrain')

                normalized_float = arr.astype(np.float32) / 255.0
                colored = colormap_func(normalized_float)
                rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)

                # Optional relief / hillshade-style shading
                if relief_val > 0.0:
                    dy, dx = np.gradient(normalized_float)
                    azimuth_rad = np.deg2rad(sun_az)
                    altitude_rad = np.deg2rad(sun_alt)
                    slope = np.pi / 2.0 - np.arctan(relief_val * np.sqrt(dx * dx + dy * dy))
                    aspect = np.arctan2(-dx, dy)
                    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
                    shaded = np.clip(shaded, 0.0, 1.0)
                    relief_intensity = float(min(relief_val / 3.0, 1.0))
                    shade_factor = (1.0 - relief_intensity) + relief_intensity * shaded
                    rgb_array = np.clip(
                        rgb_array.astype(np.float32) * shade_factor[..., np.newaxis],
                        0,
                        255,
                    ).astype(np.uint8)
                return rgb_array
            except Exception as e:
                logger.warning("Colormap failed, using grayscale", error=str(e))
                return np.stack([arr, arr, arr], axis=2)

        with ThreadPoolExecutor() as executor:
            rgb_array = await loop.run_in_executor(
                executor,
                apply_colormap,
                normalized,
                colormap,
                relief,
                sun_azimuth,
                sun_altitude
            )

        logger.info("Creating PNG image...")
        # Create PIL Image and convert to PNG bytes
        def create_png(rgb_data):
            img = Image.fromarray(rgb_data, mode='RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            return img_bytes.getvalue()

        with ThreadPoolExecutor() as executor:
            png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

        logger.info("DEM image generated successfully", size_bytes=len(png_bytes))

        # Get bounds for frontend - use actual DEM bounds when available
        bounds_dict = None
        if use_synthetic or raster_result is None:
            bounds_dict = {
                "left": float(extended_bbox.lon_min),
                "right": float(extended_bbox.lon_max),
                "bottom": float(extended_bbox.lat_min),
                "top": float(extended_bbox.lat_max),
            }

        # If available, calculate bounds from transform
        if not use_synthetic and bounds_dict is None and raster_result and raster_result.transform:
            try:
                a, b, c, d, e, f = raster_result.transform[:6]
                rows, cols = elevation.shape
                lon_tl, lat_tl = c, f
                lon_tr, lat_tr = c + a * cols, f
                lon_bl, lat_bl = c, f + e * rows
                lon_br, lat_br = c + a * cols, f + e * rows
                lons = [lon_tl, lon_tr, lon_bl, lon_br]
                lats = [lat_tl, lat_tr, lat_bl, lat_br]
                bounds_dict = {
                    "left": float(min(lons)),
                    "right": float(max(lons)),
                    "bottom": float(min(lats)),
                    "top": float(max(lats)),
                }
                logger.debug("Calculated bounds from transform", bounds=bounds_dict)
            except Exception as e:
                logger.warning("Failed to calculate bounds from transform", error=str(e))

        # Final fallback: use requested ROI (not ideal, but better than nothing)
        if bounds_dict is None:
            logger.warning("Using requested ROI bounds as fallback - DEM bounds calculation failed")
            bounds_dict = {
                "left": float(extended_bbox.lon_min),
                "right": float(extended_bbox.lon_max),
                "bottom": float(extended_bbox.lat_min),
                "top": float(extended_bbox.lat_max),
            }

        logger.info(
            "Generated DEM image",
            size_bytes=len(png_bytes),
            bounds=bounds_dict,
            elevation_range=(elev_min, elev_max),
            duration_s=round(time.time() - start, 3)
        )

        # Return image with metadata in headers
        response = Response(content=png_bytes, media_type="image/png")
        if raster_result is not None:
            response.headers["X-Dataset-Requested"] = raster_result.dataset_requested
            response.headers["X-Dataset-Used"] = raster_result.dataset_used
            response.headers["X-Fallback-Used"] = str(raster_result.is_fallback).lower()
            if raster_result.fallback_reason:
                response.headers["X-Fallback-Reason"] = raster_result.fallback_reason
        response.headers["X-Bounds-Left"] = str(bounds_dict["left"])
        response.headers["X-Bounds-Right"] = str(bounds_dict["right"])
        response.headers["X-Bounds-Bottom"] = str(bounds_dict["bottom"])
        response.headers["X-Bounds-Top"] = str(bounds_dict["top"])
        response.headers["X-Elevation-Min"] = str(elev_min)
        response.headers["X-Elevation-Max"] = str(elev_max)
        response.headers["Content-Length"] = str(len(png_bytes))

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate DEM image")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/sites-geojson")
async def get_sites_geojson():
    """Get sites as GeoJSON for map visualization."""
    try:
        import json

        import pandas as pd

        config = get_config()
        output_dir = config.paths.output_dir
        sites_file = output_dir / "sites.csv"

        if not sites_file.exists():
            # Return empty GeoJSON instead of error for better UX
            logger.info("Sites file not found, returning empty GeoJSON", path=str(sites_file))
            return Response(
                content=json.dumps({
                    "type": "FeatureCollection",
                    "features": []
                }),
                media_type="application/json"
            )

        # Read sites CSV
        sites_df = pd.read_csv(sites_file)

        # Create GeoJSON FeatureCollection
        features = []
        for _, row in sites_df.iterrows():
            # Check if polygon coordinates are available
            polygon_coords = None
            if "polygon_coords" in row and pd.notna(row["polygon_coords"]):
                try:
                    if isinstance(row["polygon_coords"], str):
                        # Try JSON first (new format), then ast.literal_eval (fallback)
                        try:
                            polygon_coords = json.loads(row["polygon_coords"])
                        except (json.JSONDecodeError, ValueError):
                            import ast
                            polygon_coords = ast.literal_eval(row["polygon_coords"])
                    elif isinstance(row["polygon_coords"], list):
                        polygon_coords = row["polygon_coords"]
                except Exception as e:
                    logger.warning(f"Failed to parse polygon_coords for site {row['site_id']}", error=str(e))

            # Create geometry based on available data
            if polygon_coords and len(polygon_coords) >= 4:  # At least 4 points for a polygon (including closing point)
                geometry = {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]  # GeoJSON Polygon requires array of rings
                }
            else:
                # Fallback to point geometry
                lon = float(row.get("lon", 0))
                lat = float(row.get("lat", 0))

                # Validate coordinates
                if not (-180 <= lon <= 360) or not (-90 <= lat <= 90):
                    logger.warning(
                        f"Invalid coordinates for site {row.get('site_id', 'unknown')}: lon={lon}, lat={lat}"
                    )
                    # Skip invalid sites
                    continue

                geometry = {
                    "type": "Point",
                    "coordinates": [lon, lat]  # GeoJSON format: [longitude, latitude]
                }

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "site_id": int(row["site_id"]),
                    "rank": int(row["rank"]),
                    "area_km2": float(row["area_km2"]),
                    "suitability_score": float(row["suitability_score"]),
                    "mean_slope_deg": float(row["mean_slope_deg"]),
                    "mean_roughness": float(row["mean_roughness"]),
                    "mean_elevation_m": float(row["mean_elevation_m"]),
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # Log sample coordinates for debugging
        if len(features) > 0:
            sample = features[0]
            sample_coords = sample.get("geometry", {}).get("coordinates") if sample.get("geometry") else None
            logger.info(
                "Generated sites GeoJSON",
                num_sites=len(features),
                sample_coords=sample_coords,
                sample_id=sample.get("properties", {}).get("site_id"),
                sample_geometry_type=sample.get("geometry", {}).get("type")
            )

        return Response(
            content=json.dumps(geojson),
            media_type="application/json"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate sites GeoJSON")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/waypoints-geojson")
async def get_waypoints_geojson():
    """Get waypoints as GeoJSON LineString for path visualization."""
    try:
        import json

        import pandas as pd

        config = get_config()
        output_dir = config.paths.output_dir
        files = list(output_dir.glob("waypoints_*.csv"))
        features = []
        if not files:
            return Response(
                content=json.dumps({"type": "FeatureCollection", "features": []}),
                media_type="application/json"
            )
        colors = {"safest": "#00ff00", "balanced": "#1e90ff", "direct": "#ffa500"}
        for fpath in files:
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            if len(df) == 0:
                continue
            coords = []
            for _, row in df.iterrows():
                lon = None
                lat = None
                if "lon" in row and "lat" in row and pd.notna(row["lon"]) and pd.notna(row["lat"]):
                    lon = float(row["lon"])
                    lat = float(row["lat"])
                elif "longitude" in row and "latitude" in row and pd.notna(row["longitude"]) and pd.notna(row["latitude"]):
                    lon = float(row["longitude"])
                    lat = float(row["latitude"])
                else:
                    continue
                if not (-180 <= lon <= 360) or not (-90 <= lat <= 90):
                    continue
                point = {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "waypoint_id": int(row.get("waypoint_id", 0)),
                        "x_meters": float(row.get("x_meters", row.get("x_site", 0))),
                        "y_meters": float(row.get("y_meters", row.get("y_site", 0))),
                        "tolerance_meters": float(row.get("tolerance_meters", row.get("tolerance_m", 0))),
                    }
                }
                features.append(point)
                coords.append([lon, lat])
            if len(coords) > 1:
                name = fpath.name
                parts = name.split("_")
                route_type = parts[-1].replace(".csv", "") if len(parts) >= 2 else "balanced"
                color = colors.get(route_type, "#ff0000")
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"type": "navigation_path", "route_type": route_type, "line_color": color}
                })
        geojson = {"type": "FeatureCollection", "features": features}
        return Response(content=json.dumps(geojson), media_type="application/json")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate waypoints GeoJSON")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/terrain-3d")
async def get_terrain_3d(
    dataset: str = Query(..., description="Dataset name (mola, mola_200m, hirise, ctx)"),
    roi: str = Query(..., description="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    max_points: int = Query(50000, ge=1000, le=200000, description="Maximum number of points for 3D mesh"),
):
    """Get DEM data as 3D mesh for Plotly.js visualization.

    Returns elevation data as a JSON object with x, y, z arrays for 3D surface plotting.
    """
    try:
        # Parse ROI
        try:
            lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid ROI format: {e}")

        try:
            dataset_lower = normalize_dataset(dataset)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid dataset. Must be one of: mola, mola_200m, ctx, hirise")

        lon_min_norm = to_lon360(lon_min)
        lon_max_norm = to_lon360(lon_max)
        if lon_max_norm <= lon_min_norm:
            lon_max_norm = min(360.0, lon_min_norm + max(lon_max - lon_min, 0.01))

        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min_norm,
            lon_max=lon_max_norm,
        )

        # Load DEM (allow download if not cached)
        use_synthetic = False
        raster_result = None
        try:
            raster_result = load_dem_window(
                dataset_lower,
                bbox,
                allow_download=True,
            )
        except Exception as e:
            logger.warning("DEM load failed for 3D terrain, using synthetic surface for demo mode", error=str(e))
            use_synthetic = True

        # Get elevation data
        elevation = raster_result.array.astype(np.float32) if not use_synthetic and raster_result else None

        # Handle nodata values
        if not use_synthetic and raster_result and raster_result.nodata is not None:
            nodata = raster_result.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

        # Downsample if too large
        if not use_synthetic:
            height, width = elevation.shape
            total_points = height * width
        else:
            # Generate synthetic grid size based on max_points
            side = int(np.sqrt(max_points))
            height, width = side, side
            total_points = height * width

        downsample_factor = 1
        if not use_synthetic and total_points > max_points:
            downsample_factor = int(np.ceil(np.sqrt(total_points / max_points)))
            elevation = elevation[::downsample_factor, ::downsample_factor]
            height, width = elevation.shape

        # Replace NaN values with minimum valid elevation (or 0 if all NaN)
        if not use_synthetic:
            valid_mask = np.isfinite(elevation)
            if np.any(valid_mask):
                min_elevation = float(np.nanmin(elevation[valid_mask]))
                elevation[~valid_mask] = min_elevation
            else:
                elevation[:] = 0.0
        else:
            # Synthetic elevation surface
            yy = np.linspace(0, 1, height)
            xx = np.linspace(0, 1, width)
            X, Y = np.meshgrid(xx, yy)
            elevation = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)) * 1500.0

        # Create coordinate grids - use actual DEM bounds, not requested ROI
        lon_min_actual = lon_min
        lon_max_actual = lon_max
        lat_min_actual = lat_min
        lat_max_actual = lat_max

        # Try to get actual bounds from transform
        if not use_synthetic and raster_result and raster_result.transform:
            try:
                a, b, c, d, e, f = raster_result.transform[:6]
                height, width = elevation.shape
                lon_tl, lat_tl = c, f
                lon_tr, lat_tr = c + a * width, f
                lon_bl, lat_bl = c, f + e * height
                lon_br, lat_br = c + a * width, f + e * height
                lons = [lon_tl, lon_tr, lon_bl, lon_br]
                lats = [lat_tl, lat_tr, lat_bl, lat_br]
                lon_min_actual = float(min(lons))
                lon_max_actual = float(max(lons))
                lat_min_actual = float(min(lats))
                lat_max_actual = float(max(lats))
                logger.debug("Calculated 3D terrain bounds from transform", bounds=(lon_min_actual, lon_max_actual, lat_min_actual, lat_max_actual))
            except Exception as e:
                logger.warning("Failed to calculate 3D terrain bounds from transform", error=str(e))

        # Create coordinate arrays
        lons = np.linspace(lon_min_actual, lon_max_actual, width)
        lats = np.linspace(lat_max_actual, lat_min_actual, height)

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Prepare response data
        response_data = {
            "x": lon_grid.tolist(),
            "y": lat_grid.tolist(),
            "z": elevation.tolist(),
            "bounds": {
                "lon_min": float(lon_min_actual),
                "lon_max": float(lon_max_actual),
                "lat_min": float(lat_min_actual),
                "lat_max": float(lat_max_actual),
            },
            "dataset_requested": dataset_lower,
            "dataset_used": raster_result.dataset_used if raster_result else dataset_lower,
            "is_fallback": raster_result.is_fallback if raster_result else False,
            "fallback_reason": raster_result.fallback_reason if raster_result else None,
            "elevation_range": {
                "min": float(np.nanmin(elevation)),
                "max": float(np.nanmax(elevation)),
            },
        }

        return Response(
            content=json.dumps(response_data),
            media_type="application/json"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate 3D terrain data")
        raise HTTPException(status_code=500, detail={"error":"terrain3d_failed","detail":str(e)})


@router.get("/visualization/overlay")
async def get_overlay(
    overlay_type: str = Query(..., description="Overlay type: elevation, solar, hillshade, slope, aspect, roughness, tri"),
    dataset: str = Query(..., description="Dataset name (mola, mola_200m, hirise, ctx)"),
    roi: str = Query(..., description="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    colormap: str = Query("terrain", description="Colormap name (terrain, viridis, plasma, etc.)"),
    relief: float = Query(0.0, ge=0.0, le=3.0, description="Relief / hillshade intensity (for elevation overlay)"),
    sun_azimuth: float = Query(315.0, ge=0.0, le=360.0, description="Sun azimuth angle in degrees"),
    sun_altitude: float = Query(45.0, ge=0.0, le=90.0, description="Sun altitude angle in degrees"),
    width: int = Query(800, ge=100, le=4000, description="Image width in pixels"),
    height: int = Query(600, ge=100, le=4000, description="Image height in pixels"),
    buffer: float = Query(0.5, ge=0.0, le=5.0, description="Buffer factor to extend ROI"),
):
    """Get geospatial overlay as PNG image for visualization.

    Supports multiple overlay types: elevation, solar, hillshade, slope, aspect, roughness, tri.
    Images are cached for performance.
    """
    start = time.time()
    overlay_type = overlay_type.lower()

    valid_types = ["elevation", "solar", "dust", "hillshade", "slope", "aspect", "roughness", "tri"]
    if overlay_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid overlay_type. Must be one of: {', '.join(valid_types)}"
        )

    logger.info(
        "Overlay request",
        overlay_type=overlay_type,
        dataset=dataset,
        roi=roi,
        width=width,
        height=height
    )

    try:
        # Parse ROI
        try:
            lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
            original_bbox = BoundingBox(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )
        except Exception as e:
            logger.error("Invalid ROI format", roi=roi, error=str(e))
            raise HTTPException(status_code=400, detail=f"Invalid ROI format: {e}")

        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        if lat_span <= 0 or lon_span <= 0:
            raise HTTPException(status_code=400, detail="Invalid ROI bounds: max must be greater than min")

        try:
            dataset_lower = normalize_dataset(dataset)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid dataset. Must be one of: mola, mola_200m, ctx, hirise",
            )

        # Keep interactive overlays bounded to avoid platform 502/timeout responses
        # that browsers often surface as CORS header errors.
        is_low_res = dataset_lower in {"mola", "mola_200m"}
        max_lat_span = 4.0 if is_low_res else 1.2
        max_lon_span = 4.0 if is_low_res else 1.2
        max_area_deg2 = 12.0 if is_low_res else 1.5
        if lat_span > max_lat_span or lon_span > max_lon_span or (lat_span * lon_span) > max_area_deg2:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"ROI too large for interactive overlay ({lat_span:.2f} x {lon_span:.2f} deg). "
                    "Zoom in or reduce ROI."
                ),
            )

        if width * height > 2_200_000:
            raise HTTPException(
                status_code=413,
                detail="Requested overlay image is too large. Reduce viewport size or zoom level.",
            )

        lon_min_norm = to_lon360(lon_min)
        lon_max_norm = to_lon360(lon_max)
        if lon_max_norm <= lon_min_norm:
            lon_max_norm = min(360.0, lon_min_norm + max(lon_span, 0.01))

        # Extend ROI bounds if buffer is specified
        if buffer > 0:
            delta_lat = (lat_max - lat_min) * buffer
            delta_lon = (lon_max_norm - lon_min_norm) * buffer
            extended_bbox = BoundingBox(
                lat_min=max(-90, lat_min - delta_lat),
                lat_max=min(90, lat_max + delta_lat),
                lon_min=max(0, lon_min_norm - delta_lon),
                lon_max=min(360, lon_max_norm + delta_lon),
            )
        else:
            extended_bbox = BoundingBox(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min_norm,
                lon_max=lon_max_norm,
            )

        resolution = resolve_dataset_with_fallback(dataset_lower, extended_bbox)
        dataset_cache = resolution.dataset_used

        data_manager = DataManager()
        dem_cache_buster = "none"
        try:
            dem_cache_path = data_manager._get_cache_path(dataset_cache, extended_bbox)
            if dem_cache_path.exists():
                dem_cache_buster = str(int(dem_cache_path.stat().st_mtime))
        except Exception:
            # Best effort: cache buster stays as "none"
            pass

        # Check cache first
        overlay_cache = OverlayCache()
        cached_path = overlay_cache.get(
            overlay_type,
            dataset_cache,
            extended_bbox,
            colormap,
            relief,
            sun_azimuth,
            sun_altitude,
            width,
            height,
            dem_cache_buster,
        )

        if cached_path and cached_path.exists():
            logger.info("Cache hit", overlay_type=overlay_type, path=str(cached_path))
            png_bytes = cached_path.read_bytes()
            bounds_dict = {
                "left": float(extended_bbox.lon_min),
                "right": float(extended_bbox.lon_max),
                "bottom": float(extended_bbox.lat_min),
                "top": float(extended_bbox.lat_max),
            }

            response = Response(content=png_bytes, media_type="image/png")
            response.headers["X-Bounds-Left"] = str(bounds_dict["left"])
            response.headers["X-Bounds-Right"] = str(bounds_dict["right"])
            response.headers["X-Bounds-Bottom"] = str(bounds_dict["bottom"])
            response.headers["X-Bounds-Top"] = str(bounds_dict["top"])
            response.headers["X-Overlay-Type"] = overlay_type
            response.headers["X-Dataset-Requested"] = resolution.dataset_requested
            response.headers["X-Dataset-Used"] = resolution.dataset_used
            response.headers["X-Fallback-Used"] = str(resolution.is_fallback).lower()
            if resolution.fallback_reason:
                response.headers["X-Fallback-Reason"] = resolution.fallback_reason
            response.headers["Content-Length"] = str(len(png_bytes))
            return response

        # Generate overlay
        logger.info("Generating overlay", overlay_type=overlay_type)
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        use_synthetic = False
        raster_result = None
        dem = None

        # Load DEM
        try:
            with ThreadPoolExecutor() as executor:
                raster_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        lambda: load_dem_window(
                            dataset_lower,
                            extended_bbox,
                            target_shape=(height, width),
                            allow_download=True,
                        )
                    ),
                    timeout=120.0
                )
            dem = _build_dem_dataarray(raster_result) if raster_result else None
            logger.info("DEM loaded successfully", shape=dem.shape if dem is not None else None)
        except asyncio.TimeoutError:
            logger.warning("DEM loading timed out, using synthetic", bbox=extended_bbox.model_dump())
            use_synthetic = True
        except Exception as e:
            logger.warning("DEM load failed, using synthetic", error=str(e))
            use_synthetic = True

        # Get cell size
        config = get_config()
        if raster_result and raster_result.resolution_m:
            cell_size_m = raster_result.resolution_m
        elif dataset_lower in config.data_sources:
            cell_size_m = config.data_sources[dataset_lower].resolution_m
        else:
            cell_size_m = 200.0

        # Generate overlay based on type
        if overlay_type == "elevation":
            # Use existing DEM image logic
            png_bytes, bounds_dict, elev_min, elev_max = await _generate_elevation_overlay(
                dem if not use_synthetic else None,
                extended_bbox,
                colormap,
                relief,
                sun_azimuth,
                sun_altitude,
                width,
                height,
                use_synthetic
            )
        elif overlay_type == "solar":
            # Try to load DCI data for spatially variable dust degradation
            dust_cover_index = None
            if not use_synthetic:
                try:
                    # Note: DCI loading would need to be implemented in DataManager
                    # For now, we'll pass None and use global degradation factor
                    logger.info("DCI loading not yet implemented, using global dust factor")
                except Exception as e:
                    logger.warning("Could not load DCI data", error=str(e))

            png_bytes, bounds_dict = await _generate_solar_overlay(
                dem if not use_synthetic else None,
                extended_bbox,
                cell_size_m,
                sun_azimuth,
                sun_altitude,
                width,
                height,
                colormap,
                use_synthetic,
                loop,
                dust_cover_index
            )
        elif overlay_type == "dust":
            png_bytes, bounds_dict = await _generate_dust_overlay(
                extended_bbox,
                width,
                height,
                colormap,
                use_synthetic,
                loop
            )
        elif overlay_type == "hillshade":
            png_bytes, bounds_dict = await _generate_hillshade_overlay(
                dem if not use_synthetic else None,
                extended_bbox,
                cell_size_m,
                sun_azimuth,
                sun_altitude,
                width,
                height,
                use_synthetic,
                loop
            )
        elif overlay_type in ["slope", "aspect", "roughness", "tri"]:
            png_bytes, bounds_dict = await _generate_terrain_metric_overlay(
                overlay_type,
                dem if not use_synthetic else None,
                extended_bbox,
                cell_size_m,
                colormap,
                width,
                height,
                use_synthetic,
                loop
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported overlay type: {overlay_type}")

        # Cache the result
        overlay_cache.put(
            overlay_type,
            dataset_cache,
            extended_bbox,
            png_bytes,
            colormap,
            relief,
            sun_azimuth,
            sun_altitude,
            width,
            height,
            dem_cache_buster,
        )

        logger.info(
            "Generated overlay",
            overlay_type=overlay_type,
            size_bytes=len(png_bytes),
            duration_s=round(time.time() - start, 3)
        )

        # Return image with metadata
        response = Response(content=png_bytes, media_type="image/png")
        if raster_result is not None:
            response.headers["X-Dataset-Requested"] = raster_result.dataset_requested
            response.headers["X-Dataset-Used"] = raster_result.dataset_used
            response.headers["X-Fallback-Used"] = str(raster_result.is_fallback).lower()
            if raster_result.fallback_reason:
                response.headers["X-Fallback-Reason"] = raster_result.fallback_reason
        response.headers["X-Bounds-Left"] = str(bounds_dict["left"])
        response.headers["X-Bounds-Right"] = str(bounds_dict["right"])
        response.headers["X-Bounds-Bottom"] = str(bounds_dict["bottom"])
        response.headers["X-Bounds-Top"] = str(bounds_dict["top"])
        response.headers["X-Overlay-Type"] = overlay_type
        response.headers["Content-Length"] = str(len(png_bytes))
        if overlay_type == "elevation" and 'elev_min' in locals() and 'elev_max' in locals():
            response.headers["X-Elevation-Min"] = str(elev_min)
            response.headers["X-Elevation-Max"] = str(elev_max)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate overlay")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/dataset-coverage")
async def get_dataset_coverage(
    dataset: str = Query(..., description="Dataset name (mola, mola_200m, hirise, ctx)"),
    bbox: str = Query(..., description="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
):
    """Report coverage availability for a dataset within a bounding box."""
    try:
        dataset_lower = normalize_dataset(dataset)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset. Must be one of: mola, mola_200m, ctx, hirise")

    try:
        lat_min, lat_max, lon_min, lon_max = map(float, bbox.split(','))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid bbox format: {e}")

    lon_min_norm = to_lon360(lon_min)
    lon_max_norm = to_lon360(lon_max)
    if lon_max_norm <= lon_min_norm:
        lon_max_norm = min(360.0, lon_min_norm + max(lon_max - lon_min, 0.01))

    bbox_obj = BoundingBox(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min_norm,
        lon_max=lon_max_norm,
    )

    resolution = resolve_dataset_with_fallback(dataset_lower, bbox_obj)
    available = resolution.dataset_used == dataset_lower

    return {
        "dataset": dataset_lower,
        "available": available,
        "dataset_used": resolution.dataset_used,
        "fallback_reason": resolution.fallback_reason,
    }


async def _generate_elevation_overlay(
    dem, bbox, colormap, relief, sun_azimuth, sun_altitude, width, height, use_synthetic
):
    """Generate elevation overlay (reuse existing DEM image logic)."""
    # This reuses the logic from get_dem_image endpoint
    # For now, we'll call get_dem_image internally or refactor
    # For simplicity, let's extract the core logic

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()

    if use_synthetic:
        synth_h = max(100, min(height, 600))
        synth_w = max(100, min(width, 800))
        y = np.linspace(0, 1, synth_h)
        x = np.linspace(0, 1, synth_w)
        xx, yy = np.meshgrid(x, y)
        elevation = (np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 1000.0
        elevation = elevation.astype(np.float32)
    else:
        elevation = dem.values.astype(np.float32)
        if hasattr(dem, 'attrs') and 'nodata' in dem.attrs:
            nodata = dem.attrs['nodata']
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

    valid_mask = np.isfinite(elevation)
    if not np.any(valid_mask):
        raise HTTPException(status_code=400, detail="No valid elevation data in ROI")

    valid_elevation = elevation[valid_mask]
    elev_min = float(np.nanmin(valid_elevation))
    elev_max = float(np.nanmax(valid_elevation))

    if elev_max == elev_min:
        normalized = np.ones_like(elevation, dtype=np.uint8) * 128
    else:
        normalized = ((elevation - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)
        normalized[~valid_mask] = 0

    # Resize if needed
    if not use_synthetic and (normalized.shape[0] != height or normalized.shape[1] != width):
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode='L')
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
            )

    # Apply colormap
    def apply_colormap(arr, cmap_name, relief_val, sun_az, sun_alt):
        try:
            import matplotlib
            import matplotlib.cm as cm
            try:
                colormap_func = cm.get_cmap(cmap_name)
            except (AttributeError, ValueError):
                try:
                    colormap_func = matplotlib.colormaps.get_cmap(cmap_name)
                except (AttributeError, ValueError):
                    colormap_func = cm.get_cmap('terrain')

            normalized_float = arr.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)

            if relief_val > 0.0:
                dy, dx = np.gradient(normalized_float)
                azimuth_rad = np.deg2rad(sun_az)
                altitude_rad = np.deg2rad(sun_alt)
                slope = np.pi / 2.0 - np.arctan(relief_val * np.sqrt(dx * dx + dy * dy))
                aspect = np.arctan2(-dx, dy)
                shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
                shaded = np.clip(shaded, 0.0, 1.0)
                relief_intensity = float(min(relief_val / 3.0, 1.0))
                shade_factor = (1.0 - relief_intensity) + relief_intensity * shaded
                rgb_array = np.clip(
                    rgb_array.astype(np.float32) * shade_factor[..., np.newaxis],
                    0, 255
                ).astype(np.uint8)
            return rgb_array
        except Exception as e:
            logger.warning("Colormap failed, using grayscale", error=str(e))
            return np.stack([arr, arr, arr], axis=2)

    with ThreadPoolExecutor() as executor:
        rgb_array = await loop.run_in_executor(
            executor, apply_colormap, normalized, colormap, relief, sun_azimuth, sun_altitude
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode='RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    with ThreadPoolExecutor() as executor:
        png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

    # Get bounds
    if use_synthetic or not hasattr(dem, 'rio'):
        bounds_dict = {
            "left": float(bbox.lon_min),
            "right": float(bbox.lon_max),
            "bottom": float(bbox.lat_min),
            "top": float(bbox.lat_max),
        }
    else:
        try:
            bounds = dem.rio.bounds()
            bounds_dict = {
                "left": float(bounds.left),
                "right": float(bounds.right),
                "bottom": float(bounds.bottom),
                "top": float(bounds.top),
            }
        except Exception:
            bounds_dict = {
                "left": float(bbox.lon_min),
                "right": float(bbox.lon_max),
                "bottom": float(bbox.lat_min),
                "top": float(bbox.lat_max),
            }

    return png_bytes, bounds_dict, elev_min, elev_max


async def _generate_solar_overlay(
    dem, bbox, cell_size_m, sun_azimuth, sun_altitude, width, height, colormap, use_synthetic, loop, dust_cover_index=None
):
    """Generate solar potential overlay."""
    from concurrent.futures import ThreadPoolExecutor

    if use_synthetic:
        rows, cols = height, width
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        solar_potential = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) / 2.0
    else:
        # Run analysis pipeline to get terrain metrics
        pipeline = AnalysisPipeline()
        try:
            # Determine dataset from dem or use default
            dataset_name = 'mola'  # Default fallback
            if hasattr(dem, 'attrs') and 'dataset' in dem.attrs:
                dataset_name = dem.attrs['dataset']
            results = pipeline.run(bbox, dataset=dataset_name, threshold=0.5)
            if results.metrics is None:
                raise Exception("No metrics from pipeline")
        except Exception as e:
            logger.warning("Analysis pipeline failed, using synthetic", error=str(e))
            rows, cols = dem.shape[0], dem.shape[1]
            y = np.linspace(0, 1, rows)
            x = np.linspace(0, 1, cols)
            xx, yy = np.meshgrid(x, y)
            solar_potential = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) / 2.0
        else:
            # Calculate solar potential with optional DCI dust data
            analyzer = SolarPotentialAnalyzer(cell_size_m=cell_size_m)
            solar_result = analyzer.calculate_solar_potential(
                dem, results.metrics, sun_azimuth, sun_altitude, dust_cover_index
            )
            solar_potential = solar_result.solar_potential_map

    # Normalize to 0-255
    valid_mask = np.isfinite(solar_potential)
    if not np.any(valid_mask):
        solar_potential = np.zeros_like(solar_potential)
        valid_mask = np.ones_like(solar_potential, dtype=bool)

    valid_data = solar_potential[valid_mask]
    if len(valid_data) > 0 and np.nanmax(valid_data) > np.nanmin(valid_data):
        normalized = ((solar_potential - np.nanmin(valid_data)) /
                     (np.nanmax(valid_data) - np.nanmin(valid_data)) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(solar_potential, dtype=np.uint8)
    normalized[~valid_mask] = 0

    # Resize if needed
    if normalized.shape[0] != height or normalized.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode='L')
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
            )

    # Apply colormap (use plasma or viridis for solar)
    def apply_colormap(arr, cmap_name):
        try:
            import matplotlib
            import matplotlib.cm as cm
            try:
                colormap_func = cm.get_cmap(cmap_name)
            except (AttributeError, ValueError):
                try:
                    colormap_func = matplotlib.colormaps.get_cmap(cmap_name)
                except (AttributeError, ValueError):
                    colormap_func = cm.get_cmap('plasma')

            normalized_float = arr.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            return rgb_array
        except Exception as e:
            logger.warning("Colormap failed, using grayscale", error=str(e))
            return np.stack([arr, arr, arr], axis=2)

    with ThreadPoolExecutor() as executor:
        rgb_array = await loop.run_in_executor(
            executor, apply_colormap, normalized, colormap if colormap != 'terrain' else 'plasma'
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode='RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    with ThreadPoolExecutor() as executor:
        png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

    bounds_dict = {
        "left": float(bbox.lon_min),
        "right": float(bbox.lon_max),
        "bottom": float(bbox.lat_min),
        "top": float(bbox.lat_max),
    }

    return png_bytes, bounds_dict


async def _generate_dust_overlay(
    bbox, width, height, colormap, use_synthetic, loop
):
    """Generate TES Dust Cover Index overlay."""
    from concurrent.futures import ThreadPoolExecutor

    if use_synthetic:
        # Generate synthetic dust cover pattern
        rows, cols = height, width
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        # Synthetic pattern: higher dust in low areas, lower in high areas
        dust_cover = 0.85 + 0.1 * (np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) / 2.0
    else:
        # TODO: Load actual TES DCI data from DataManager
        # For now, generate synthetic data
        logger.warning("TES DCI data loading not yet implemented, using synthetic")
        rows, cols = height, width
        y = np.linspace(bbox.lat_min, bbox.lat_max, rows)
        x = np.linspace(bbox.lon_min, bbox.lon_max, cols)
        xx, yy = np.meshgrid(x, y)
        dust_cover = 0.85 + 0.1 * (np.sin(3 * np.pi * (xx - bbox.lon_min) / (bbox.lon_max - bbox.lon_min)) *
                                   np.cos(2 * np.pi * (yy - bbox.lat_min) / (bbox.lat_max - bbox.lat_min)) + 1.0) / 2.0

    # Normalize DCI to 0-255 (DCI range typically 0.85-1.0, lower = more dust)
    # Invert so higher values = more dust (for visualization)
    dust_normalized = 1.0 - dust_cover  # Higher dust = higher value
    valid_mask = np.isfinite(dust_normalized)
    if not np.any(valid_mask):
        dust_normalized = np.zeros_like(dust_normalized)
        valid_mask = np.ones_like(dust_normalized, dtype=bool)

    valid_data = dust_normalized[valid_mask]
    if len(valid_data) > 0 and np.nanmax(valid_data) > np.nanmin(valid_data):
        normalized = ((dust_normalized - np.nanmin(valid_data)) /
                     (np.nanmax(valid_data) - np.nanmin(valid_data)) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(dust_normalized, dtype=np.uint8)
    normalized[~valid_mask] = 0

    # Resize if needed
    if normalized.shape[0] != height or normalized.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode='L')
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
            )

    # Apply colormap (use brown/earth tones for dust)
    def apply_colormap(arr, cmap_name):
        try:
            import matplotlib
            import matplotlib.cm as cm
            try:
                colormap_func = cm.get_cmap(cmap_name)
            except (AttributeError, ValueError):
                try:
                    colormap_func = matplotlib.colormaps.get_cmap(cmap_name)
                except (AttributeError, ValueError):
                    colormap_func = cm.get_cmap('YlOrBr')  # Yellow-Orange-Brown for dust

            normalized_float = arr.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            return rgb_array
        except Exception as e:
            logger.warning("Colormap failed, using grayscale", error=str(e))
            return np.stack([arr, arr, arr], axis=2)

    with ThreadPoolExecutor() as executor:
        rgb_array = await loop.run_in_executor(
            executor, apply_colormap, normalized, colormap if colormap != 'terrain' else 'YlOrBr'
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode='RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    with ThreadPoolExecutor() as executor:
        png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

    bounds_dict = {
        "left": float(bbox.lon_min),
        "right": float(bbox.lon_max),
        "bottom": float(bbox.lat_min),
        "top": float(bbox.lat_max),
    }

    return png_bytes, bounds_dict


async def _generate_hillshade_overlay(
    dem, bbox, cell_size_m, sun_azimuth, sun_altitude, width, height, use_synthetic, loop
):
    """Generate hillshade overlay."""
    from concurrent.futures import ThreadPoolExecutor

    if use_synthetic:
        rows, cols = height, width
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        elevation = (np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 1000.0
    else:
        elevation = dem.values.astype(np.float32)
        if hasattr(dem, 'attrs') and 'nodata' in dem.attrs:
            nodata = dem.attrs['nodata']
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

    # Calculate hillshade
    def calc_hillshade(elev_arr, cell_size, az, alt):
        analyzer = TerrainAnalyzer(cell_size_m=cell_size)
        return analyzer.calculate_hillshade(elev_arr, az, alt)

    with ThreadPoolExecutor() as executor:
        hillshade = await loop.run_in_executor(
            executor, calc_hillshade, elevation, cell_size_m, sun_azimuth, sun_altitude
        )

    # Resize if needed
    if hillshade.shape[0] != height or hillshade.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode='L')
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            hillshade = await loop.run_in_executor(
                executor, resize_image, hillshade, (height, width)
            )

    # Convert to RGB (grayscale)
    rgb_array = np.stack([hillshade, hillshade, hillshade], axis=2)

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode='RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    with ThreadPoolExecutor() as executor:
        png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

    bounds_dict = {
        "left": float(bbox.lon_min),
        "right": float(bbox.lon_max),
        "bottom": float(bbox.lat_min),
        "top": float(bbox.lat_max),
    }

    return png_bytes, bounds_dict


async def _generate_terrain_metric_overlay(
    metric_type, dem, bbox, cell_size_m, colormap, width, height, use_synthetic, loop
):
    """Generate overlay for terrain metrics (slope, aspect, roughness, tri)."""
    from concurrent.futures import ThreadPoolExecutor

    if use_synthetic:
        rows, cols = height, width
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        if metric_type == "slope":
            metric_data = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 45.0
        elif metric_type == "aspect":
            metric_data = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 360.0
        else:
            metric_data = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 100.0
    else:
        # Calculate terrain metrics
        analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
        metrics = analyzer.analyze(dem)

        if metric_type == "slope":
            metric_data = metrics.slope
        elif metric_type == "aspect":
            metric_data = metrics.aspect
            # Handle -1 values (flat areas) - set to 0 for visualization
            metric_data = np.where(metric_data < 0, 0, metric_data)
        elif metric_type == "roughness":
            metric_data = metrics.roughness
        elif metric_type == "tri":
            metric_data = metrics.tri
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    # Normalize to 0-255
    valid_mask = np.isfinite(metric_data)
    if not np.any(valid_mask):
        metric_data = np.zeros_like(metric_data)
        valid_mask = np.ones_like(metric_data, dtype=bool)

    valid_data = metric_data[valid_mask]
    if len(valid_data) > 0 and np.nanmax(valid_data) > np.nanmin(valid_data):
        normalized = ((metric_data - np.nanmin(valid_data)) /
                     (np.nanmax(valid_data) - np.nanmin(valid_data)) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(metric_data, dtype=np.uint8)
    normalized[~valid_mask] = 0

    # Resize if needed
    if normalized.shape[0] != height or normalized.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode='L')
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
            )

    # Apply colormap (use cyclic colormap for aspect)
    def apply_colormap(arr, cmap_name, is_aspect=False):
        try:
            import matplotlib
            import matplotlib.cm as cm
            try:
                if is_aspect:
                    # Use cyclic colormap for aspect
                    try:
                        colormap_func = matplotlib.colormaps.get_cmap('hsv')
                    except Exception:
                        colormap_func = cm.get_cmap('hsv')
                else:
                    colormap_func = cm.get_cmap(cmap_name)
            except (AttributeError, ValueError):
                try:
                    colormap_func = matplotlib.colormaps.get_cmap(cmap_name if not is_aspect else 'viridis')
                except (AttributeError, ValueError):
                    colormap_func = cm.get_cmap('viridis')

            normalized_float = arr.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            return rgb_array
        except Exception as e:
            logger.warning("Colormap failed, using grayscale", error=str(e))
            return np.stack([arr, arr, arr], axis=2)

    is_aspect = metric_type == "aspect"
    cmap_to_use = 'hsv' if is_aspect else (colormap if colormap != 'terrain' else 'viridis')

    with ThreadPoolExecutor() as executor:
        rgb_array = await loop.run_in_executor(
            executor, apply_colormap, normalized, cmap_to_use, is_aspect
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode='RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    with ThreadPoolExecutor() as executor:
        png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

    bounds_dict = {
        "left": float(bbox.lon_min),
        "right": float(bbox.lon_max),
        "bottom": float(bbox.lat_min),
        "top": float(bbox.lat_max),
    }

    return png_bytes, bounds_dict
