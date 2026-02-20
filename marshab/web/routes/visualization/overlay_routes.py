"""Overlay tile and full overlay image endpoints."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from marshab.config import get_config
from marshab.core.data_manager import DataManager
from marshab.core.overlay_cache import OverlayCache
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
    ENABLE_TILE_OVERLAYS,
    TILE_CACHE,
    _build_dem_dataarray,
    _log_metric,
    logger,
)
from .overlay_generators import (
    generate_dust_overlay,
    generate_elevation_overlay,
    generate_hillshade_overlay,
    generate_solar_overlay,
    generate_terrain_metric_overlay,
)

router = APIRouter()


@router.get(
    "/visualization/tiles/overlay/{overlay_type}/{dataset}/{z}/{x}/{y}.png",
    responses=REAL_DEM_UNAVAILABLE_RESPONSE_DOC,
)
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
            png_bytes, _bounds, _elev_min, _elev_max = await generate_elevation_overlay(
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
            png_bytes, _bounds = await generate_solar_overlay(
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
            png_bytes, _bounds = await generate_dust_overlay(
                bbox,
                256,
                256,
                colormap,
                False,
                loop,
            )
        elif overlay_type == "hillshade":
            png_bytes, _bounds = await generate_hillshade_overlay(
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
            png_bytes, _bounds = await generate_terrain_metric_overlay(
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
    except DataError as exc:
        _log_metric("tile.error.5xx", 1, kind="overlay", overlay_type=overlay_type)
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
        logger.exception("Data error in overlay render", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
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
    """Get geospatial overlay as PNG image for visualization."""
    start = time.time()
    overlay_type = overlay_type.lower()

    valid_types = ["elevation", "solar", "dust", "hillshade", "slope", "aspect", "roughness", "tri"]
    if overlay_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid overlay_type. Must be one of: {', '.join(valid_types)}",
        )

    logger.info(
        "Overlay request",
        overlay_type=overlay_type,
        dataset=dataset,
        roi=roi,
        width=width,
        height=height,
    )

    try:
        try:
            lat_min, lat_max, lon_min, lon_max = map(float, roi.split(","))
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
            pass

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

        logger.info("Generating overlay", overlay_type=overlay_type)
        loop = asyncio.get_event_loop()
        allow_synthetic = os.getenv("MARSHAB_ALLOW_SYNTHETIC_TILES", "false").lower() in {"1", "true", "yes"}
        use_synthetic = False
        raster_result = None
        dem = None

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
                        ),
                    ),
                    timeout=120.0,
                )
            dem = _build_dem_dataarray(raster_result) if raster_result else None
            logger.info("DEM loaded successfully", shape=dem.shape if dem is not None else None)
        except asyncio.TimeoutError:
            if not allow_synthetic:
                raise DataError(
                    REAL_DEM_UNAVAILABLE_ERROR_CODE,
                    details={"dataset": dataset_lower, "hint": "DEM load timed out; synthetic fallback disabled."},
                )
            logger.warning("DEM loading timed out, using synthetic", bbox=extended_bbox.model_dump())
            use_synthetic = True
        except Exception as e:
            if not allow_synthetic:
                raise DataError(
                    REAL_DEM_UNAVAILABLE_ERROR_CODE,
                    details={"dataset": dataset_lower, "hint": "DEM load failed; synthetic fallback disabled.", "error": str(e)},
                )
            logger.warning("DEM load failed, using synthetic", error=str(e))
            use_synthetic = True

        config = get_config()
        if raster_result and raster_result.resolution_m:
            cell_size_m = raster_result.resolution_m
        elif dataset_lower in config.data_sources:
            cell_size_m = config.data_sources[dataset_lower].resolution_m
        else:
            cell_size_m = 200.0

        elev_min = None
        elev_max = None

        if overlay_type == "elevation":
            png_bytes, bounds_dict, elev_min, elev_max = await generate_elevation_overlay(
                dem if not use_synthetic else None,
                extended_bbox,
                colormap,
                relief,
                sun_azimuth,
                sun_altitude,
                width,
                height,
                use_synthetic,
            )
        elif overlay_type == "solar":
            dust_cover_index = None
            if not use_synthetic:
                logger.info("DCI loading not yet implemented, using global dust factor")

            png_bytes, bounds_dict = await generate_solar_overlay(
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
                dust_cover_index,
            )
        elif overlay_type == "dust":
            png_bytes, bounds_dict = await generate_dust_overlay(
                extended_bbox,
                width,
                height,
                colormap,
                use_synthetic,
                loop,
            )
        elif overlay_type == "hillshade":
            png_bytes, bounds_dict = await generate_hillshade_overlay(
                dem if not use_synthetic else None,
                extended_bbox,
                cell_size_m,
                sun_azimuth,
                sun_altitude,
                width,
                height,
                use_synthetic,
                loop,
            )
        elif overlay_type in ["slope", "aspect", "roughness", "tri"]:
            png_bytes, bounds_dict = await generate_terrain_metric_overlay(
                overlay_type,
                dem if not use_synthetic else None,
                extended_bbox,
                cell_size_m,
                colormap,
                width,
                height,
                use_synthetic,
                loop,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported overlay type: {overlay_type}")

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
            duration_s=round(time.time() - start, 3),
        )

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
        if overlay_type == "elevation" and elev_min is not None and elev_max is not None:
            response.headers["X-Elevation-Min"] = str(elev_min)
            response.headers["X-Elevation-Max"] = str(elev_max)

        return response

    except DataError as exc:
        if exc.message == REAL_DEM_UNAVAILABLE_ERROR_CODE:
            payload = build_real_dem_unavailable_payload(
                detail="Real DEM data is unavailable for the requested overlay.",
                dataset_requested=dataset_lower,
                dataset_used=dataset_lower,
            )
            return JSONResponse(
                status_code=REAL_DEM_UNAVAILABLE_STATUS_CODE,
                content=payload,
            )
        logger.exception("Data error while generating overlay", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate overlay")
        raise HTTPException(status_code=500, detail=str(e))
