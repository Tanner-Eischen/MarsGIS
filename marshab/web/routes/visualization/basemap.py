"""Basemap tile endpoints for Mars visualization."""

from __future__ import annotations

import time
import urllib.request

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from marshab.core.data_manager import DataManager
from marshab.core.raster_contracts import (
    REAL_DEM_UNAVAILABLE_ERROR_CODE,
    REAL_DEM_UNAVAILABLE_RESPONSE_DOC,
    REAL_DEM_UNAVAILABLE_STATUS_CODE,
    build_real_dem_unavailable_payload,
)
from marshab.core.raster_service import (
    compute_tile_bbox_epsg4326,
    normalize_dataset,
    resolve_dataset_with_fallback,
    tile_style_hash,
    to_lon360,
)
from marshab.core.tile_cache import read_disk_cache, tile_cache_path, write_disk_cache
from marshab.exceptions import DataError
from marshab.models import BoundingBox
from marshab.web.routes.visualization import load_dem_window

from ._helpers import (
    DEFAULT_GLOBAL_BASEMAP_STYLE,
    ENABLE_TILE_BASEMAP,
    MARS_BASEMAP_TILE_URL,
    TILE_CACHE,
    _build_dem_dataarray,
    _log_metric,
    _render_global_basemap_tile,
    _render_orthophoto_tile,
    _resolve_fallback_basemap_dataset,
    _resolve_global_basemap_source,
    _resolve_orthophoto_source_path,
    logger,
)
from .overlay_generators import generate_elevation_overlay

router = APIRouter()


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
            response.headers["Cache-Control"] = "public, max-age=86400"
            return response
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Basemap tile proxy failed", z=z, x=x, y=y, error=str(e))
        raise HTTPException(status_code=502, detail="Failed to fetch basemap tile")


@router.get("/visualization/tiles/basemap/global/{z}/{x}/{y}.png")
async def get_global_basemap_tile(
    z: int,
    x: int,
    y: int,
    style: str = Query(DEFAULT_GLOBAL_BASEMAP_STYLE, description="Global basemap style (blendshade, clrshade, shade)"),
    fallback_dataset: str = Query("mola_200m", description="Fallback DEM dataset when local global basemap is unavailable"),
    allow_fallback: bool = Query(True, description="Allow network/DEM fallback when local global basemap is unavailable"),
):
    """Render tiles from bundled global Mars basemap GeoTIFFs."""
    if not ENABLE_TILE_BASEMAP:
        raise HTTPException(status_code=404, detail="Basemap tiling disabled")

    try:
        bbox = compute_tile_bbox_epsg4326(z, x, y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    fallback_dataset_resolved = _resolve_fallback_basemap_dataset(fallback_dataset)
    source_info = _resolve_global_basemap_source(style)

    if source_info is None:
        if not allow_fallback:
            raise HTTPException(status_code=503, detail="Global basemap source unavailable")
        try:
            response = await get_mars_basemap_tile(z, x, y)
            response.headers["X-Global-Basemap-Used"] = "false"
            response.headers["X-Global-Basemap-Reason"] = "source_unavailable_network_fallback"
            return response
        except Exception:
            response = await get_basemap_tile(fallback_dataset_resolved, z, x, y)
            response.headers["X-Global-Basemap-Used"] = "false"
            response.headers["X-Global-Basemap-Reason"] = "source_unavailable_dem_fallback"
            return response

    style_used, source_path = source_info
    try:
        source_mtime = str(int(source_path.stat().st_mtime))
    except Exception:
        source_mtime = "0"
    style_hash = tile_style_hash(
        {
            "style": style_used,
            "source": source_path.name,
            "mtime": source_mtime,
        }
    )
    cache_key = f"globalbasemap::{style_used}::{z}:{x}:{y}:{style_hash}"

    cached = TILE_CACHE.get(cache_key)
    if cached:
        response = Response(content=cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Global-Basemap-Used"] = "true"
        response.headers["X-Global-Basemap-Style"] = style_used
        response.headers["X-Global-Basemap-Source"] = source_path.name
        response.headers["X-Global-Basemap-CRS"] = "cached"
        return response

    disk_path = tile_cache_path("globalbasemap", style_used, z, x, y, style_hash)
    disk_cached = read_disk_cache(disk_path, TILE_CACHE.config.ttl_seconds)
    if disk_cached:
        TILE_CACHE.set(cache_key, disk_cached)
        response = Response(content=disk_cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Global-Basemap-Used"] = "true"
        response.headers["X-Global-Basemap-Style"] = style_used
        response.headers["X-Global-Basemap-Source"] = source_path.name
        response.headers["X-Global-Basemap-CRS"] = "cached"
        return response

    try:
        png_bytes = _render_global_basemap_tile(source_path, bbox, tile_size=256)
    except Exception as exc:
        logger.exception("Global basemap tile render failed", error=str(exc), source=str(source_path))
        if not allow_fallback:
            raise HTTPException(status_code=503, detail="Global basemap render failed")
        try:
            response = await get_mars_basemap_tile(z, x, y)
            response.headers["X-Global-Basemap-Used"] = "false"
            response.headers["X-Global-Basemap-Reason"] = "render_failed_network_fallback"
            return response
        except Exception:
            response = await get_basemap_tile(fallback_dataset_resolved, z, x, y)
            response.headers["X-Global-Basemap-Used"] = "false"
            response.headers["X-Global-Basemap-Reason"] = "render_failed_dem_fallback"
            return response

    TILE_CACHE.set(cache_key, png_bytes)
    write_disk_cache(disk_path, png_bytes)

    response = Response(content=png_bytes, media_type="image/png")
    response.headers["Cache-Control"] = "public, max-age=86400"
    response.headers["X-Global-Basemap-Used"] = "true"
    response.headers["X-Global-Basemap-Style"] = style_used
    response.headers["X-Global-Basemap-Source"] = source_path.name
    response.headers["X-Global-Basemap-CRS"] = "source"
    return response


@router.get("/visualization/tiles/basemap/orthophoto/{z}/{x}/{y}.png")
async def get_orthophoto_basemap_tile(
    z: int,
    x: int,
    y: int,
    fallback_dataset: str = Query("hirise", description="Fallback dataset (mola, mola_200m, hirise)"),
    allow_dem_fallback: bool = Query(False, description="Allow DEM fallback when no orthophoto source is available"),
):
    """Render tiles from a single configured orthophoto source file."""
    try:
        bbox = compute_tile_bbox_epsg4326(z, x, y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    source_path = _resolve_orthophoto_source_path()
    fallback_dataset_resolved = _resolve_fallback_basemap_dataset(fallback_dataset)

    if source_path is None or not source_path.exists():
        if not allow_dem_fallback:
            raise HTTPException(status_code=503, detail="Orthophoto source unavailable; configure HiRISE orthophoto source")
        response = await get_basemap_tile(fallback_dataset_resolved, z, x, y)
        response.headers["X-Orthophoto-Used"] = "false"
        response.headers["X-Orthophoto-Reason"] = "path_not_configured"
        return response

    style_hash = tile_style_hash(
        {
            "source_name": source_path.name,
            "source_mtime": str(int(source_path.stat().st_mtime)),
            "stretch": "p2_p98",
        }
    )
    cache_key = f"orthophoto::{z}:{x}:{y}:{style_hash}"

    cached = TILE_CACHE.get(cache_key)
    if cached:
        response = Response(content=cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Orthophoto-Used"] = "true"
        response.headers["X-Orthophoto-Source"] = source_path.name
        response.headers["X-Orthophoto-CRS"] = "cached"
        return response

    disk_path = tile_cache_path("orthophoto", "global", z, x, y, style_hash)
    disk_cached = read_disk_cache(disk_path, TILE_CACHE.config.ttl_seconds)
    if disk_cached:
        TILE_CACHE.set(cache_key, disk_cached)
        response = Response(content=disk_cached, media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-Orthophoto-Used"] = "true"
        response.headers["X-Orthophoto-Source"] = source_path.name
        response.headers["X-Orthophoto-CRS"] = "cached"
        return response

    try:
        png_bytes = _render_orthophoto_tile(source_path, bbox, tile_size=256)
    except Exception as exc:
        logger.exception("Orthophoto tile render failed", error=str(exc), source=str(source_path))
        if not allow_dem_fallback:
            raise HTTPException(status_code=503, detail="Orthophoto render failed for HiRISE source")
        response = await get_basemap_tile(fallback_dataset_resolved, z, x, y)
        response.headers["X-Orthophoto-Used"] = "false"
        response.headers["X-Orthophoto-Reason"] = "render_failed"
        return response

    TILE_CACHE.set(cache_key, png_bytes)
    write_disk_cache(disk_path, png_bytes)

    response = Response(content=png_bytes, media_type="image/png")
    response.headers["Cache-Control"] = "public, max-age=86400"
    response.headers["X-Orthophoto-Used"] = "true"
    response.headers["X-Orthophoto-Source"] = source_path.name
    response.headers["X-Orthophoto-CRS"] = "source"
    return response


@router.get(
    "/visualization/tiles/basemap/{dataset}/{z}/{x}/{y}.png",
    responses=REAL_DEM_UNAVAILABLE_RESPONSE_DOC,
)
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
        png_bytes, _bounds, _elev_min, _elev_max = await generate_elevation_overlay(
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
    except DataError as exc:
        _log_metric("tile.error.5xx", 1, kind="basemap")
        if exc.message == REAL_DEM_UNAVAILABLE_ERROR_CODE:
            payload = build_real_dem_unavailable_payload(
                detail="Real DEM data is unavailable for the requested tile.",
                dataset_requested=dataset_lower,
                dataset_used=dataset_lower,
            )
            return JSONResponse(
                status_code=REAL_DEM_UNAVAILABLE_STATUS_CODE,
                content=payload,
            )
        logger.exception("Data error in basemap render", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
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
