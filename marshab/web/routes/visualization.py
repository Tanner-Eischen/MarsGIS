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
    width: int = Query(800, ge=100, le=2000, description="Image width in pixels"),
    height: int = Query(600, ge=100, le=2000, description="Image height in pixels"),
    buffer: float = Query(0.5, ge=0.0, le=2.0, description="Buffer factor to extend ROI (0.5 = 50% extension, 1.0 = 100% extension)"),
):
    """Get DEM as PNG image for visualization.
    
    Returns a PNG image of the DEM elevation data, cropped to the specified ROI.
    """
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
        
        # Load DEM with extended bounds
        data_manager = DataManager()
        dem = data_manager.get_dem_for_roi(extended_bbox, dataset=dataset.lower(), download=False, clip=True)
        
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
            raise HTTPException(status_code=400, detail="No valid elevation data in ROI")
        
        valid_elevation = elevation[valid_mask]
        elev_min = float(np.nanmin(valid_elevation))
        elev_max = float(np.nanmax(valid_elevation))
        
        if elev_max == elev_min:
            # Constant elevation, create uniform image
            normalized = np.ones_like(elevation, dtype=np.uint8) * 128
        else:
            # Normalize to 0-255
            normalized = ((elevation - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)
            normalized[~valid_mask] = 0  # Set invalid pixels to black
        
        # Resize if needed
        if normalized.shape[0] != height or normalized.shape[1] != width:
            from PIL import Image as PILImage
            img = PILImage.fromarray(normalized, mode='L')
            img = img.resize((width, height), PILImage.Resampling.LANCZOS)
            normalized = np.array(img)
        
        # Apply colormap
        try:
            import matplotlib.cm as cm

            # Base RGB from colormap
            colormap_func = cm.get_cmap(colormap)
            normalized_float = normalized.astype(np.float32) / 255.0
            colored = colormap_func(normalized_float)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)

            # Optional relief / hillshade-style shading
            if relief > 0.0:
                # Simple hillshade using gradients of the normalized elevation.
                # Relief controls how strongly local slopes affect shading.
                dy, dx = np.gradient(normalized_float)

                azimuth_deg = 315.0
                altitude_deg = 45.0
                azimuth_rad = np.deg2rad(azimuth_deg)
                altitude_rad = np.deg2rad(altitude_deg)

                slope = np.pi / 2.0 - np.arctan(relief * np.sqrt(dx * dx + dy * dy))
                aspect = np.arctan2(-dx, dy)

                shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(
                    azimuth_rad - aspect
                )
                shaded = np.clip(shaded, 0.0, 1.0)

                # Blend shading with base colors: relief in [0, 3] -> intensity in [0, 1]
                relief_intensity = float(min(relief / 3.0, 1.0))
                shade_factor = (1.0 - relief_intensity) + relief_intensity * shaded
                rgb_array = np.clip(
                    rgb_array.astype(np.float32) * shade_factor[..., np.newaxis],
                    0,
                    255,
                ).astype(np.uint8)
        except Exception:
            # Fallback to grayscale if colormap fails
            rgb_array = np.stack([normalized, normalized, normalized], axis=2)
        
        # Create PIL Image
        img = Image.fromarray(rgb_array, mode='RGB')
        
        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Get bounds for frontend (use extended bounds)
        if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
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
                    "left": float(extended_bbox.lon_min),
                    "right": float(extended_bbox.lon_max),
                    "bottom": float(extended_bbox.lat_min),
                    "top": float(extended_bbox.lat_max),
                }
        else:
            bounds_dict = {
                "left": float(extended_bbox.lon_min),
                "right": float(extended_bbox.lon_max),
                "bottom": float(extended_bbox.lat_min),
                "top": float(extended_bbox.lat_max),
            }
        
        # Return image with metadata in headers
        response = Response(content=img_bytes.read(), media_type="image/png")
        response.headers["X-Bounds-Left"] = str(bounds_dict["left"])
        response.headers["X-Bounds-Right"] = str(bounds_dict["right"])
        response.headers["X-Bounds-Bottom"] = str(bounds_dict["bottom"])
        response.headers["X-Bounds-Top"] = str(bounds_dict["top"])
        response.headers["X-Elevation-Min"] = str(elev_min)
        response.headers["X-Elevation-Max"] = str(elev_max)
        
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
                geometry = {
                    "type": "Point",
                    "coordinates": [float(row["lon"]), float(row["lat"])]
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
            else:
                # Skip waypoints without lat/lon
                logger.warning("Waypoint missing lat/lon, skipping")
                continue
            
            point = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "waypoint_id": int(row["waypoint_id"]),
                    "x_meters": float(row.get("x_meters", 0)),
                    "y_meters": float(row.get("y_meters", 0)),
                    "tolerance_meters": float(row["tolerance_meters"]),
                }
            }
            features.append(point)
            coordinates.append([lon, lat])
        
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
        
        # Load DEM
        data_manager = DataManager()
        dem = data_manager.get_dem_for_roi(bbox, dataset=dataset.lower(), download=False, clip=True)
        
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
        
        # Create coordinate grids
        if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
            try:
                bounds = dem.rio.bounds()
                lon_min_actual = bounds.left
                lon_max_actual = bounds.right
                lat_min_actual = bounds.bottom
                lat_max_actual = bounds.top
            except Exception:
                lon_min_actual = lon_min
                lon_max_actual = lon_max
                lat_min_actual = lat_min
                lat_max_actual = lat_max
        else:
            lon_min_actual = lon_min
            lon_max_actual = lon_max
            lat_min_actual = lat_min
            lat_max_actual = lat_max
        
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

