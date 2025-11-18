"""Visualization data export endpoints."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from marshab.config import get_config
from marshab.core.data_manager import DataManager
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


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
    dataset: str = Query(..., description="Dataset name (mola, hirise, ctx)"),
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
        
        # Extend ROI bounds if buffer is specified
        if buffer > 0:
            delta_lat = (lat_max - lat_min) * buffer
            delta_lon = (lon_max - lon_min) * buffer
            extended_bbox = BoundingBox(
                lat_min=max(-90, lat_min - delta_lat),
                lat_max=min(90, lat_max + delta_lat),
                lon_min=max(0, lon_min - delta_lon),
                lon_max=min(360, lon_max + delta_lon),
            )
        else:
            extended_bbox = original_bbox
        
        logger.info("Loading DEM data...", bbox=extended_bbox.model_dump(), dataset=dataset)
        # Load DEM with extended bounds (allow download if not cached)
        # Run in thread pool to avoid blocking async handler
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        data_manager = DataManager()
        loop = asyncio.get_event_loop()
        try:
            # Increase timeout for large DEMs, but add progress logging
            logger.info("Starting DEM load in thread pool...")
            with ThreadPoolExecutor() as executor:
                dem = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        lambda: data_manager.get_dem_for_roi(extended_bbox, dataset=dataset.lower(), download=True, clip=True)
                    ),
                    timeout=120.0  # Increased to 120 seconds for large DEMs
                )
            logger.info("DEM loaded successfully", shape=dem.shape, dtype=str(dem.dtype))
        except asyncio.TimeoutError:
            logger.error("DEM loading timed out after 120 seconds", bbox=extended_bbox.model_dump())
            raise HTTPException(
                status_code=504, 
                detail=f"DEM loading timed out after 120 seconds. The ROI may be too large. Try a smaller region or check if DEM is cached."
            )
        except Exception as e:
            logger.exception("Failed to load DEM", error=str(e), bbox=extended_bbox.model_dump())
            raise HTTPException(status_code=500, detail=f"Failed to load DEM: {str(e)}")
        
        logger.info("Processing elevation data...")
        # Get elevation data
        elevation = dem.values.astype(np.float32)
        
        # Handle nodata values
        if hasattr(dem, 'attrs') and 'nodata' in dem.attrs:
            nodata = dem.attrs['nodata']
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
        if normalized.shape[0] != height or normalized.shape[1] != width:
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
                import matplotlib.cm as cm
                import matplotlib
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
        
        # Get bounds for frontend - use actual DEM bounds, not requested ROI
        bounds_dict = None
        
        # Try to get bounds from rio accessor first
        if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
            try:
                bounds = dem.rio.bounds()
                bounds_dict = {
                    "left": float(bounds.left),
                    "right": float(bounds.right),
                    "bottom": float(bounds.bottom),
                    "top": float(bounds.top),
                }
                logger.debug("Using rio.bounds() for image bounds", bounds=bounds_dict)
            except Exception as e:
                logger.warning("Failed to get bounds from rio accessor", error=str(e))
        
        # If rio bounds failed, try to calculate from transform
        if bounds_dict is None and hasattr(dem, 'rio') and hasattr(dem.rio, 'transform'):
            try:
                import rasterio.transform
                transform = dem.rio.transform()
                height, width = dem.shape
                # Calculate bounds from transform: corners of the image
                # Top-left corner (row=0, col=0)
                lon_tl, lat_tl = rasterio.transform.xy(transform, 0, 0)
                # Bottom-right corner (row=height, col=width)
                lon_br, lat_br = rasterio.transform.xy(transform, height, width)
                # Top-right corner (row=0, col=width)
                lon_tr, lat_tr = rasterio.transform.xy(transform, 0, width)
                # Bottom-left corner (row=height, col=0)
                lon_bl, lat_bl = rasterio.transform.xy(transform, height, 0)
                
                # Get min/max from all corners
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
        
        # If transform calculation failed, try using coordinate arrays
        if bounds_dict is None and hasattr(dem, 'coords'):
            try:
                if 'lon' in dem.coords and 'lat' in dem.coords:
                    lon_coords = dem.coords['lon'].values
                    lat_coords = dem.coords['lat'].values
                    bounds_dict = {
                        "left": float(np.nanmin(lon_coords)),
                        "right": float(np.nanmax(lon_coords)),
                        "bottom": float(np.nanmin(lat_coords)),
                        "top": float(np.nanmax(lat_coords)),
                    }
                    logger.debug("Calculated bounds from coordinate arrays", bounds=bounds_dict)
            except Exception as e:
                logger.warning("Failed to calculate bounds from coordinates", error=str(e))
        
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
            elevation_range=(elev_min, elev_max)
        )
        
        # Return image with metadata in headers
        response = Response(content=png_bytes, media_type="image/png")
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
        import pandas as pd
        import json
        
        config = get_config()
        output_dir = config.paths.output_dir
        sites_file = output_dir / "sites.csv"
        
        if not sites_file.exists():
            raise HTTPException(status_code=404, detail="Sites file not found. Run analysis first.")
        
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
        import pandas as pd
        import json
        
        config = get_config()
        output_dir = config.paths.output_dir
        waypoints_file = output_dir / "waypoints.csv"
        
        if not waypoints_file.exists():
            raise HTTPException(status_code=404, detail="Waypoints file not found. Plan navigation first.")
        
        # Read waypoints CSV
        waypoints_df = pd.read_csv(waypoints_file)
        
        if len(waypoints_df) == 0:
            raise HTTPException(status_code=404, detail="No waypoints found")
        
        # For now, waypoints are in SITE frame (x_meters, y_meters)
        # We need to convert back to lat/lon for visualization
        # This is a simplified approach - in production, would use proper coordinate transformation
        
        # Create a simple LineString (in future, would transform coordinates properly)
        # For now, return waypoints as points
        features = []
        coordinates = []
        
        for _, row in waypoints_df.iterrows():
            # Check if lat/lon are available (new format)
            if "lat" in row and "lon" in row and pd.notna(row["lat"]) and pd.notna(row["lon"]):
                lon = float(row["lon"])
                lat = float(row["lat"])
                
                # Validate coordinates
                if not (-180 <= lon <= 360) or not (-90 <= lat <= 90):
                    logger.warning(
                        f"Invalid coordinates for waypoint {row.get('waypoint_id', 'unknown')}: lon={lon}, lat={lat}"
                    )
                    continue
            else:
                # Skip waypoints without lat/lon
                logger.warning(f"Waypoint {row.get('waypoint_id', 'unknown')} missing lat/lon, skipping")
                continue
            
            point = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]  # GeoJSON format: [longitude, latitude]
                },
                "properties": {
                    "waypoint_id": int(row["waypoint_id"]),
                    "x_meters": float(row.get("x_meters", 0)),
                    "y_meters": float(row.get("y_meters", 0)),
                    "tolerance_meters": float(row["tolerance_meters"]),
                }
            }
            features.append(point)
            coordinates.append([lon, lat])  # GeoJSON format: [longitude, latitude]
        
        # Create LineString if we have coordinates
        if len(coordinates) > 1:
            line_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "type": "navigation_path"
                }
            }
            features.insert(0, line_feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Log sample coordinates for debugging
        if len(features) > 0:
            point_feature = next((f for f in features if f.get("geometry", {}).get("type") == "Point"), None)
            line_feature = next((f for f in features if f.get("geometry", {}).get("type") == "LineString"), None)
            logger.info(
                "Generated waypoints GeoJSON",
                num_points=len([f for f in features if f.get("geometry", {}).get("type") == "Point"]),
                has_line=line_feature is not None,
                sample_point_coords=point_feature.get("geometry", {}).get("coordinates") if point_feature else None,
                line_coords_count=len(line_feature.get("geometry", {}).get("coordinates")) if line_feature else 0
            )
        
        return Response(
            content=json.dumps(geojson),
            media_type="application/json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate waypoints GeoJSON")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/terrain-3d")
async def get_terrain_3d(
    dataset: str = Query(..., description="Dataset name (mola, hirise, ctx)"),
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
            bbox = BoundingBox(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid ROI format: {e}")
        
        # Load DEM (allow download if not cached)
        data_manager = DataManager()
        dem = data_manager.get_dem_for_roi(bbox, dataset=dataset.lower(), download=True, clip=True)
        
        # Get elevation data
        elevation = dem.values.astype(np.float32)
        
        # Handle nodata values
        if hasattr(dem, 'attrs') and 'nodata' in dem.attrs:
            nodata = dem.attrs['nodata']
            if nodata is not None:
                elevation[elevation == nodata] = np.nan
        
        # Downsample if too large
        height, width = elevation.shape
        total_points = height * width
        
        downsample_factor = 1
        if total_points > max_points:
            downsample_factor = int(np.ceil(np.sqrt(total_points / max_points)))
            elevation = elevation[::downsample_factor, ::downsample_factor]
            height, width = elevation.shape
        
        # Replace NaN values with minimum valid elevation (or 0 if all NaN)
        valid_mask = np.isfinite(elevation)
        if np.any(valid_mask):
            min_elevation = float(np.nanmin(elevation[valid_mask]))
            elevation[~valid_mask] = min_elevation
        else:
            # All NaN - set to 0
            elevation[:] = 0.0
        
        # Create coordinate grids - use actual DEM bounds, not requested ROI
        lon_min_actual = lon_min
        lon_max_actual = lon_max
        lat_min_actual = lat_min
        lat_max_actual = lat_max
        
        # Try to get actual bounds from DEM
        if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
            try:
                bounds = dem.rio.bounds()
                lon_min_actual = float(bounds.left)
                lon_max_actual = float(bounds.right)
                lat_min_actual = float(bounds.bottom)
                lat_max_actual = float(bounds.top)
                logger.debug("Using rio.bounds() for 3D terrain", bounds=(lon_min_actual, lon_max_actual, lat_min_actual, lat_max_actual))
            except Exception as e:
                logger.warning("Failed to get bounds from rio accessor for 3D terrain", error=str(e))
        
        # If rio bounds failed, try to calculate from transform
        if (lon_min_actual == lon_min and lon_max_actual == lon_max and 
            hasattr(dem, 'rio') and hasattr(dem.rio, 'transform')):
            try:
                import rasterio.transform
                transform = dem.rio.transform()
                height, width = dem.shape
                # Calculate bounds from transform
                lon_tl, lat_tl = rasterio.transform.xy(transform, 0, 0)
                lon_br, lat_br = rasterio.transform.xy(transform, height, width)
                lon_tr, lat_tr = rasterio.transform.xy(transform, 0, width)
                lon_bl, lat_bl = rasterio.transform.xy(transform, height, 0)
                
                lons = [lon_tl, lon_tr, lon_bl, lon_br]
                lats = [lat_tl, lat_tr, lat_bl, lat_br]
                lon_min_actual = float(min(lons))
                lon_max_actual = float(max(lons))
                lat_min_actual = float(min(lats))
                lat_max_actual = float(max(lats))
                logger.debug("Calculated 3D terrain bounds from transform", bounds=(lon_min_actual, lon_max_actual, lat_min_actual, lat_max_actual))
            except Exception as e:
                logger.warning("Failed to calculate 3D terrain bounds from transform", error=str(e))
        
        # If transform calculation failed, try using coordinate arrays
        if (lon_min_actual == lon_min and lon_max_actual == lon_max and 
            hasattr(dem, 'coords') and 'lon' in dem.coords and 'lat' in dem.coords):
            try:
                lon_coords = dem.coords['lon'].values
                lat_coords = dem.coords['lat'].values
                lon_min_actual = float(np.nanmin(lon_coords))
                lon_max_actual = float(np.nanmax(lon_coords))
                lat_min_actual = float(np.nanmin(lat_coords))
                lat_max_actual = float(np.nanmax(lat_coords))
                logger.debug("Calculated 3D terrain bounds from coordinates", bounds=(lon_min_actual, lon_max_actual, lat_min_actual, lat_max_actual))
            except Exception as e:
                logger.warning("Failed to calculate 3D terrain bounds from coordinates", error=str(e))
        
        # Create coordinate arrays
        lons = np.linspace(lon_min_actual, lon_max_actual, width)
        lats = np.linspace(lat_max_actual, lat_min_actual, height)  # Note: reversed for lat
        
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
        raise HTTPException(status_code=500, detail=str(e))

