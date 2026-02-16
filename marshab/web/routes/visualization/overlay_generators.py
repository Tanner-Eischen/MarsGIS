"""Overlay image generation for elevation, solar, dust, hillshade, and terrain metrics."""

from __future__ import annotations

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import HTTPException
from PIL import Image

from marshab.analysis.solar_potential import SolarPotentialAnalyzer
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.processing.terrain import TerrainAnalyzer

from ._helpers import logger


async def generate_elevation_overlay(
    dem, bbox, colormap, relief, sun_azimuth, sun_altitude, width, height, use_synthetic
):
    """Generate elevation overlay (reuse existing DEM image logic)."""
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
        if hasattr(dem, "attrs") and "nodata" in dem.attrs:
            nodata = dem.attrs["nodata"]
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

    if not use_synthetic and (normalized.shape[0] != height or normalized.shape[1] != width):
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode="L")
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
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
            executor, apply_colormap, normalized, colormap, relief, sun_azimuth, sun_altitude
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode="RGB")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return img_bytes.getvalue()

    with ThreadPoolExecutor() as executor:
        png_bytes = await loop.run_in_executor(executor, create_png, rgb_array)

    if use_synthetic or not hasattr(dem, "rio"):
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


async def generate_solar_overlay(
    dem, bbox, cell_size_m, sun_azimuth, sun_altitude, width, height, colormap, use_synthetic, loop, dust_cover_index=None
):
    """Generate solar potential overlay."""
    if use_synthetic:
        rows, cols = height, width
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        solar_potential = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) / 2.0
    else:
        pipeline = AnalysisPipeline()
        try:
            dataset_name = "mola"
            if hasattr(dem, "attrs") and "dataset" in dem.attrs:
                dataset_name = dem.attrs["dataset"]
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
            analyzer = SolarPotentialAnalyzer(cell_size_m=cell_size_m)
            solar_result = analyzer.calculate_solar_potential(
                dem, results.metrics, sun_azimuth, sun_altitude, dust_cover_index
            )
            solar_potential = solar_result.solar_potential_map

    valid_mask = np.isfinite(solar_potential)
    if not np.any(valid_mask):
        solar_potential = np.zeros_like(solar_potential)
        valid_mask = np.ones_like(solar_potential, dtype=bool)

    valid_data = solar_potential[valid_mask]
    if len(valid_data) > 0 and np.nanmax(valid_data) > np.nanmin(valid_data):
        normalized = (
            (solar_potential - np.nanmin(valid_data)) / (np.nanmax(valid_data) - np.nanmin(valid_data)) * 255
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(solar_potential, dtype=np.uint8)
    normalized[~valid_mask] = 0

    if normalized.shape[0] != height or normalized.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode="L")
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
            )

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
                    colormap_func = cm.get_cmap("plasma")

            normalized_float = arr.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            return rgb_array
        except Exception as e:
            logger.warning("Colormap failed, using grayscale", error=str(e))
            return np.stack([arr, arr, arr], axis=2)

    with ThreadPoolExecutor() as executor:
        rgb_array = await loop.run_in_executor(
            executor, apply_colormap, normalized, colormap if colormap != "terrain" else "plasma"
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode="RGB")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
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


async def generate_dust_overlay(bbox, width, height, colormap, use_synthetic, loop):
    """Generate TES Dust Cover Index overlay."""
    if use_synthetic:
        rows, cols = height, width
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        dust_cover = 0.85 + 0.1 * (np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) / 2.0
    else:
        logger.warning("TES DCI data loading not yet implemented, using synthetic")
        rows, cols = height, width
        y = np.linspace(bbox.lat_min, bbox.lat_max, rows)
        x = np.linspace(bbox.lon_min, bbox.lon_max, cols)
        xx, yy = np.meshgrid(x, y)
        dust_cover = (
            0.85
            + 0.1
            * (
                np.sin(3 * np.pi * (xx - bbox.lon_min) / (bbox.lon_max - bbox.lon_min))
                * np.cos(2 * np.pi * (yy - bbox.lat_min) / (bbox.lat_max - bbox.lat_min))
                + 1.0
            )
            / 2.0
        )

    dust_normalized = 1.0 - dust_cover
    valid_mask = np.isfinite(dust_normalized)
    if not np.any(valid_mask):
        dust_normalized = np.zeros_like(dust_normalized)
        valid_mask = np.ones_like(dust_normalized, dtype=bool)

    valid_data = dust_normalized[valid_mask]
    if len(valid_data) > 0 and np.nanmax(valid_data) > np.nanmin(valid_data):
        normalized = (
            (dust_normalized - np.nanmin(valid_data)) / (np.nanmax(valid_data) - np.nanmin(valid_data)) * 255
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(dust_normalized, dtype=np.uint8)
    normalized[~valid_mask] = 0

    if normalized.shape[0] != height or normalized.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode="L")
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
            )

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
                    colormap_func = cm.get_cmap("YlOrBr")

            normalized_float = arr.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            return rgb_array
        except Exception as e:
            logger.warning("Colormap failed, using grayscale", error=str(e))
            return np.stack([arr, arr, arr], axis=2)

    with ThreadPoolExecutor() as executor:
        rgb_array = await loop.run_in_executor(
            executor, apply_colormap, normalized, colormap if colormap != "terrain" else "YlOrBr"
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode="RGB")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
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


async def generate_hillshade_overlay(
    dem, bbox, cell_size_m, sun_azimuth, sun_altitude, width, height, use_synthetic, loop
):
    """Generate hillshade overlay."""
    if use_synthetic:
        rows, cols = height, width
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        elevation = (np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) * 1000.0
    else:
        elevation = dem.values.astype(np.float32)
        if hasattr(dem, "attrs") and "nodata" in dem.attrs:
            nodata = dem.attrs["nodata"]
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

    def calc_hillshade(elev_arr, cell_size, az, alt):
        analyzer = TerrainAnalyzer(cell_size_m=cell_size)
        return analyzer.calculate_hillshade(elev_arr, az, alt)

    with ThreadPoolExecutor() as executor:
        hillshade = await loop.run_in_executor(
            executor, calc_hillshade, elevation, cell_size_m, sun_azimuth, sun_altitude
        )

    if hillshade.shape[0] != height or hillshade.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode="L")
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            hillshade = await loop.run_in_executor(
                executor, resize_image, hillshade, (height, width)
            )

    rgb_array = np.stack([hillshade, hillshade, hillshade], axis=2)

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode="RGB")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
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


async def generate_terrain_metric_overlay(
    metric_type, dem, bbox, cell_size_m, colormap, width, height, use_synthetic, loop
):
    """Generate overlay for terrain metrics (slope, aspect, roughness, tri)."""
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
        analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
        metrics = analyzer.analyze(dem)

        if metric_type == "slope":
            metric_data = metrics.slope
        elif metric_type == "aspect":
            metric_data = metrics.aspect
            metric_data = np.where(metric_data < 0, 0, metric_data)
        elif metric_type == "roughness":
            metric_data = metrics.roughness
        elif metric_type == "tri":
            metric_data = metrics.tri
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    valid_mask = np.isfinite(metric_data)
    if not np.any(valid_mask):
        metric_data = np.zeros_like(metric_data)
        valid_mask = np.ones_like(metric_data, dtype=bool)

    valid_data = metric_data[valid_mask]
    if len(valid_data) > 0 and np.nanmax(valid_data) > np.nanmin(valid_data):
        normalized = (
            (metric_data - np.nanmin(valid_data)) / (np.nanmax(valid_data) - np.nanmin(valid_data)) * 255
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(metric_data, dtype=np.uint8)
    normalized[~valid_mask] = 0

    if normalized.shape[0] != height or normalized.shape[1] != width:
        def resize_image(arr, target_size):
            img = Image.fromarray(arr, mode="L")
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)

        with ThreadPoolExecutor() as executor:
            normalized = await loop.run_in_executor(
                executor, resize_image, normalized, (height, width)
            )

    def apply_colormap(arr, cmap_name, is_aspect=False):
        try:
            import matplotlib
            import matplotlib.cm as cm

            try:
                if is_aspect:
                    try:
                        colormap_func = matplotlib.colormaps.get_cmap("hsv")
                    except Exception:
                        colormap_func = cm.get_cmap("hsv")
                else:
                    colormap_func = cm.get_cmap(cmap_name)
            except (AttributeError, ValueError):
                try:
                    colormap_func = matplotlib.colormaps.get_cmap(cmap_name if not is_aspect else "viridis")
                except (AttributeError, ValueError):
                    colormap_func = cm.get_cmap("viridis")

            normalized_float = arr.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            return rgb_array
        except Exception as e:
            logger.warning("Colormap failed, using grayscale", error=str(e))
            return np.stack([arr, arr, arr], axis=2)

    is_aspect = metric_type == "aspect"
    cmap_to_use = "hsv" if is_aspect else (colormap if colormap != "terrain" else "viridis")

    with ThreadPoolExecutor() as executor:
        rgb_array = await loop.run_in_executor(
            executor, apply_colormap, normalized, cmap_to_use, is_aspect
        )

    def create_png(rgb_data):
        img = Image.fromarray(rgb_data, mode="RGB")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
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
