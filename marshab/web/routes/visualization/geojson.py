"""GeoJSON and dataset coverage endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from marshab.config import get_config
from marshab.core.raster_service import normalize_dataset, resolve_dataset_with_fallback, to_lon360
from marshab.models import BoundingBox

from ._helpers import logger

router = APIRouter()


def _resolve_results_dir(analysis_dir: str | None) -> Path:
    if analysis_dir:
        return Path(analysis_dir).expanduser()
    return get_config().paths.output_dir


@router.get("/visualization/sites-geojson")
async def get_sites_geojson(
    analysis_dir: str | None = Query(None, description="Optional analysis directory containing sites.csv"),
):
    """Get sites as GeoJSON for map visualization."""
    try:
        output_dir = _resolve_results_dir(analysis_dir)
        sites_file = output_dir / "sites.csv"

        if not sites_file.exists():
            logger.info("Sites file not found, returning empty GeoJSON", path=str(sites_file))
            return Response(
                content=json.dumps({
                    "type": "FeatureCollection",
                    "features": [],
                }),
                media_type="application/json",
            )

        sites_df = pd.read_csv(sites_file)

        features = []
        for _, row in sites_df.iterrows():
            polygon_coords = None
            if "polygon_coords" in row and pd.notna(row["polygon_coords"]):
                try:
                    if isinstance(row["polygon_coords"], str):
                        try:
                            polygon_coords = json.loads(row["polygon_coords"])
                        except (json.JSONDecodeError, ValueError):
                            import ast

                            polygon_coords = ast.literal_eval(row["polygon_coords"])
                    elif isinstance(row["polygon_coords"], list):
                        polygon_coords = row["polygon_coords"]
                except Exception as e:
                    logger.warning(f"Failed to parse polygon_coords for site {row['site_id']}", error=str(e))

            if polygon_coords and len(polygon_coords) >= 4:
                geometry = {
                    "type": "Polygon",
                    "coordinates": [polygon_coords],
                }
            else:
                lon = float(row.get("lon", 0))
                lat = float(row.get("lat", 0))

                if not (-180 <= lon <= 360) or not (-90 <= lat <= 90):
                    logger.warning(
                        f"Invalid coordinates for site {row.get('site_id', 'unknown')}: lon={lon}, lat={lat}"
                    )
                    continue

                geometry = {
                    "type": "Point",
                    "coordinates": [lon, lat],
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
                },
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        if len(features) > 0:
            sample = features[0]
            sample_coords = sample.get("geometry", {}).get("coordinates") if sample.get("geometry") else None
            logger.info(
                "Generated sites GeoJSON",
                num_sites=len(features),
                sample_coords=sample_coords,
                sample_id=sample.get("properties", {}).get("site_id"),
                sample_geometry_type=sample.get("geometry", {}).get("type"),
            )

        return Response(
            content=json.dumps(geojson),
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate sites GeoJSON")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/waypoints-geojson")
async def get_waypoints_geojson(
    analysis_dir: str | None = Query(None, description="Optional analysis directory for waypoint CSVs"),
    site_id: int | None = Query(None, description="Optional site filter"),
    strategy: str | None = Query(None, description="Optional strategy filter (safest, balanced, direct)"),
):
    """Get waypoints as GeoJSON LineString for path visualization."""
    try:
        output_dir = _resolve_results_dir(analysis_dir)
        files: list[Path] = []
        if site_id is not None:
            site_specific = output_dir / f"waypoints_site_{site_id}.csv"
            if site_specific.exists():
                files.append(site_specific)
            files.extend(sorted(output_dir.glob(f"waypoints_{site_id}_*.csv")))
        else:
            files.extend(sorted(output_dir.glob("waypoints_*.csv")))
            default_file = output_dir / "waypoints.csv"
            if default_file.exists():
                files.append(default_file)

        if strategy:
            normalized_strategy = strategy.strip().lower()
            files = [
                fpath for fpath in files
                if fpath.stem.lower().endswith(f"_{normalized_strategy}") or fpath.stem.lower() == f"waypoints_{normalized_strategy}"
            ]

        features = []
        if not files:
            return Response(
                content=json.dumps({"type": "FeatureCollection", "features": []}),
                media_type="application/json",
            )
        colors = {"safest": "#00ff00", "balanced": "#1e90ff", "direct": "#ffa500", "site_route": "#ef4444"}
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
                    },
                }
                features.append(point)
                coords.append([lon, lat])
            if len(coords) > 1:
                stem = fpath.stem.lower()
                if stem.startswith("waypoints_site_"):
                    route_type = "site_route"
                elif stem.startswith("waypoints_"):
                    route_type = stem.split("_")[-1]
                else:
                    route_type = "route"
                color = colors.get(route_type, "#ff0000")
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"type": "navigation_path", "route_type": route_type, "line_color": color},
                })
        geojson = {"type": "FeatureCollection", "features": features}
        return Response(content=json.dumps(geojson), media_type="application/json")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate waypoints GeoJSON")
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
        lat_min, lat_max, lon_min, lon_max = map(float, bbox.split(","))
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
