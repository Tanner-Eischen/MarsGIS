"""Terrain, DEM, elevation, and 3D mesh endpoints."""

from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from rasterio.transform import Affine, rowcol

from marshab.config import get_config
from marshab.core.raster_contracts import (
    REAL_DEM_UNAVAILABLE_ERROR_CODE,
    REAL_DEM_UNAVAILABLE_RESPONSE_DOC,
    REAL_DEM_UNAVAILABLE_STATUS_CODE,
    build_real_dem_unavailable_payload,
)
from marshab.core.raster_service import normalize_dataset, to_lon360
from marshab.exceptions import DataError
from marshab.models import BoundingBox
from marshab.web.routes.visualization import load_dem_window

from ._helpers import (
    _load_sites_features,
    _load_waypoint_features,
    _project_features_to_mesh,
    logger,
)

router = APIRouter()


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
    relief: float = Query(0.0, ge=0.0, le=3.0, description="Relief / hillshade intensity"),
    sun_azimuth: float = Query(315.0, ge=0.0, le=360.0, description="Sun azimuth angle in degrees"),
    sun_altitude: float = Query(45.0, ge=0.0, le=90.0, description="Sun altitude angle in degrees"),
    width: int = Query(800, ge=100, le=4000, description="Image width in pixels"),
    height: int = Query(600, ge=100, le=4000, description="Image height in pixels"),
    buffer: float = Query(0.5, ge=0.0, le=5.0, description="Buffer factor to extend ROI"),
):
    """Get DEM as PNG image for visualization."""
    start = time.time()
    logger.info("DEM image request", dataset=dataset, roi=roi, width=width, height=height)
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

        logger.info("Loading DEM data...", bbox=extended_bbox.model_dump(), dataset=dataset)
        loop = asyncio.get_event_loop()
        use_synthetic = False
        raster_result = None
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
            logger.info("DEM loaded successfully", shape=raster_result.array.shape, dtype=str(raster_result.array.dtype))
        except asyncio.TimeoutError:
            logger.warning("DEM loading timed out, generating synthetic visualization for demo mode", bbox=extended_bbox.model_dump())
            use_synthetic = True
        except Exception as e:
            logger.warning("DEM load failed, generating synthetic visualization for demo mode", error=str(e), bbox=extended_bbox.model_dump())
            use_synthetic = True

        logger.info("Processing elevation data...")
        if not use_synthetic and raster_result is not None:
            elevation = raster_result.array.astype(np.float32)
        else:
            synth_h = max(100, min(height, 600))
            synth_w = max(100, min(width, 800))
            y = np.linspace(0, 1, synth_h)
            x = np.linspace(0, 1, synth_w)
            xx, yy = np.meshgrid(x, y)
            elevation = (np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 1000.0
            elevation = elevation.astype(np.float32)

        if not use_synthetic and raster_result is not None and raster_result.nodata is not None:
            nodata = raster_result.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

        valid_mask = np.isfinite(elevation)
        if not np.any(valid_mask):
            logger.error("No valid elevation data in ROI", roi=roi)
            raise HTTPException(status_code=400, detail="No valid elevation data in ROI")

        valid_elevation = elevation[valid_mask]
        elev_min = float(np.nanmin(valid_elevation))
        elev_max = float(np.nanmax(valid_elevation))

        logger.info("Elevation range", elev_min=elev_min, elev_max=elev_max)

        if elev_max == elev_min:
            normalized = np.ones_like(elevation, dtype=np.uint8) * 128
        else:
            normalized = ((elevation - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)
            normalized[~valid_mask] = 0

        if not use_synthetic and (normalized.shape[0] != height or normalized.shape[1] != width):
            def resize_image(arr, target_size):
                img = Image.fromarray(arr, mode="L")
                img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                return np.array(img)

            with ThreadPoolExecutor() as executor:
                normalized = await loop.run_in_executor(
                    executor,
                    resize_image,
                    normalized,
                    (height, width),
                )

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
                        colormap_func = cm.get_cmap("terrain")

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
                sun_altitude,
            )

        def create_png(rgb_data):
            img = Image.fromarray(rgb_data, mode="RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            return img_bytes.getvalue()

        with ThreadPoolExecutor() as executor:
            png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

        bounds_dict = None
        if use_synthetic or raster_result is None:
            bounds_dict = {
                "left": float(extended_bbox.lon_min),
                "right": float(extended_bbox.lon_max),
                "bottom": float(extended_bbox.lat_min),
                "top": float(extended_bbox.lat_max),
            }

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
            except Exception as e:
                logger.warning("Failed to calculate bounds from transform", error=str(e))

        if bounds_dict is None:
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
        response.headers["X-Elevation-Min"] = str(elev_min)
        response.headers["X-Elevation-Max"] = str(elev_max)
        response.headers["Content-Length"] = str(len(png_bytes))

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate DEM image")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/visualization/terrain-3d",
    responses=REAL_DEM_UNAVAILABLE_RESPONSE_DOC,
)
async def get_terrain_3d(
    dataset: str = Query(..., description="Dataset name (mola, mola_200m, hirise, ctx)"),
    roi: str = Query(..., description="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    max_points: int = Query(15000, ge=1000, le=120000, description="Maximum number of points for 3D mesh"),
):
    """Get DEM data as 3D scene payload for terrain + overlays."""
    try:
        from ._helpers import _parse_roi

        lat_min, lat_max, lon_min, lon_max = _parse_roi(roi)

        try:
            dataset_lower = normalize_dataset(dataset)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid dataset. Must be one of: mola, mola_200m, ctx, hirise")

        lat_span = lat_max - lat_min
        lon_span = abs(lon_max - lon_min)
        if lat_span <= 0 or lon_span <= 0:
            raise HTTPException(status_code=400, detail="Invalid ROI bounds: max must be greater than min")

        is_low_res = dataset_lower in {"mola", "mola_200m"}
        max_lat_span = 4.0 if is_low_res else 1.2
        max_lon_span = 4.0 if is_low_res else 1.2
        max_area_deg2 = 12.0 if is_low_res else 1.5
        if lat_span > max_lat_span or lon_span > max_lon_span or (lat_span * lon_span) > max_area_deg2:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"ROI too large for interactive 3D terrain ({lat_span:.2f} x {lon_span:.2f} deg). "
                    "Zoom in or reduce ROI."
                ),
            )

        max_points_effective = min(max_points, 35000 if is_low_res else 15000)

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

        use_synthetic = False
        raster_result = None
        try:
            raster_result = load_dem_window(
                dataset_lower,
                bbox,
                allow_download=False,
            )
        except DataError as e:
            if e.message == REAL_DEM_UNAVAILABLE_ERROR_CODE:
                payload = build_real_dem_unavailable_payload(
                    detail="Real DEM data is unavailable for the requested region.",
                    dataset_requested=dataset_lower,
                    dataset_used=None,
                )
                return JSONResponse(
                    status_code=REAL_DEM_UNAVAILABLE_STATUS_CODE,
                    content=payload,
                )
            logger.warning("DEM load failed for 3D terrain, using synthetic surface for demo mode", error=str(e))
            use_synthetic = True
        except Exception as e:
            logger.warning("DEM load failed for 3D terrain, using synthetic surface for demo mode", error=str(e))
            use_synthetic = True

        elevation = raster_result.array.astype(np.float32) if not use_synthetic and raster_result else None

        if not use_synthetic and raster_result and raster_result.nodata is not None:
            nodata = raster_result.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

        if not use_synthetic:
            height, width = elevation.shape
            total_points = height * width
        else:
            side = int(np.sqrt(max_points_effective))
            height, width = side, side
            total_points = height * width

        downsample_factor = 1
        if not use_synthetic and total_points > max_points_effective:
            downsample_factor = int(np.ceil(np.sqrt(total_points / max_points_effective)))
            elevation = elevation[::downsample_factor, ::downsample_factor]
            height, width = elevation.shape

        if not use_synthetic:
            valid_mask = np.isfinite(elevation)
            if np.any(valid_mask):
                min_elevation = float(np.nanmin(elevation[valid_mask]))
                elevation[~valid_mask] = min_elevation
            else:
                elevation[:] = 0.0
        else:
            yy = np.linspace(0, 1, height)
            xx = np.linspace(0, 1, width)
            X, Y = np.meshgrid(xx, yy)
            elevation = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)) * 1500.0

        lon_min_actual = lon_min
        lon_max_actual = lon_max
        lat_min_actual = lat_min
        lat_max_actual = lat_max

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
            except Exception as e:
                logger.warning("Failed to calculate 3D terrain bounds from transform", error=str(e))

        lons = np.linspace(lon_min_actual, lon_max_actual, width)
        lats = np.linspace(lat_max_actual, lat_min_actual, height)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        waypoints_features = _load_waypoint_features()
        sites_features = _load_sites_features()
        waypoints_projected = _project_features_to_mesh(waypoints_features, lons, lats, elevation)
        sites_projected = _project_features_to_mesh(sites_features, lons, lats, elevation)

        response_data = {
            "mesh": {
                "x": lon_grid.tolist(),
                "y": lat_grid.tolist(),
                "z": elevation.tolist(),
            },
            "bounds": {
                "lon_min": float(lon_min_actual),
                "lon_max": float(lon_max_actual),
                "lat_min": float(lat_min_actual),
                "lat_max": float(lat_max_actual),
            },
            "roi_requested": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "roi_effective": {
                "lat_min": float(lat_min_actual),
                "lat_max": float(lat_max_actual),
                "lon_min": float(lon_min_actual),
                "lon_max": float(lon_max_actual),
            },
            "dataset_requested": dataset_lower,
            "dataset_used": raster_result.dataset_used if raster_result else dataset_lower,
            "is_fallback": raster_result.is_fallback if raster_result else False,
            "fallback_reason": raster_result.fallback_reason if raster_result else None,
            "max_points_requested": int(max_points),
            "max_points_effective": int(max_points_effective),
            "mesh_shape": {"rows": int(height), "cols": int(width), "downsample_factor": int(downsample_factor)},
            "z_exaggeration_default": 1.0,
            "z_exaggeration_limits": {"min": 0.0, "max": 5.0},
            "basemap": {
                "global_tile_template": "/api/v1/visualization/tiles/basemap/global/{z}/{x}/{y}.png",
                "orthophoto_tile_template": "/api/v1/visualization/tiles/basemap/orthophoto/{z}/{x}/{y}.png",
                "fallback_tile_template": "/api/v1/visualization/tiles/basemap/mola_200m/{z}/{x}/{y}.png",
            },
            "overlays": {
                "paths": waypoints_projected["paths"],
                "waypoints": waypoints_projected["points"],
                "sites": sites_projected["points"],
            },
            "elevation_range": {
                "min": float(np.nanmin(elevation)),
                "max": float(np.nanmax(elevation)),
            },
        }

        import json

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate 3D terrain data")
        raise HTTPException(status_code=500, detail={"error": "terrain3d_failed", "detail": str(e)})
