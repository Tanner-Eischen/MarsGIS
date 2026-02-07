"""Run a deterministic portfolio demo across the 3 flagship flows.

Flows:
1) Site selection -> ranked sites + GeoJSON overlay
2) Route planning -> waypoint export from a fixed landing point to top site
3) Decision brief -> deterministic "why this site" summary
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Allow direct script execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROI = {
    "lat_min": 18.25,
    "lat_max": 18.45,
    "lon_min": 77.25,
    "lon_max": 77.45,
}
DEFAULT_OUTPUT_DIR = Path("data/output/portfolio_demo")
DEFAULT_WEIGHTS = {
    "slope": 0.30,
    "roughness": 0.25,
    "elevation": 0.20,
    "solar_exposure": 0.15,
    "science_value": 0.10,
}
DEFAULT_SEED = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _to_feature(site: Any) -> dict:
    if site.polygon_coords:
        geometry = {
            "type": "Polygon",
            "coordinates": [site.polygon_coords],
        }
    else:
        geometry = {
            "type": "Point",
            "coordinates": [site.lon, site.lat],
        }
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "site_id": site.site_id,
            "rank": site.rank,
            "suitability_score": round(site.suitability_score, 6),
            "area_km2": round(site.area_km2, 6),
            "mean_slope_deg": round(site.mean_slope_deg, 6),
            "mean_roughness": round(site.mean_roughness, 6),
            "mean_elevation_m": round(site.mean_elevation_m, 6),
        },
    }


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_waypoints_csv(path: Path, waypoints: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "waypoint_id",
        "pixel_row",
        "pixel_col",
        "lat",
        "lon",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(waypoints)


def run_demo(output_dir: Path, seed: int, threshold: float) -> None:
    from marshab.analysis.decision_brief import build_decision_brief
    from marshab.analysis.routing import plan_route
    from marshab.core.analysis_pipeline import AnalysisPipeline
    from marshab.models import BoundingBox

    _set_seed(seed)
    data_mode = "real_dem" if Path("data/cache/mola_lat18_lon77.tif").exists() else "synthetic"
    roi = BoundingBox(**DEFAULT_ROI)

    pipeline = AnalysisPipeline()
    results = pipeline.run(
        roi=roi,
        dataset="mola",
        threshold=threshold,
        criteria_weights=DEFAULT_WEIGHTS,
    )

    if not results.sites:
        raise RuntimeError("No candidate sites found for fixed demo ROI")
    if results.dem is None:
        raise RuntimeError("Analysis pipeline returned no DEM for route planning")

    dem = results.dem
    top_site = results.sites[0]

    # Convert top-site lat/lon into target pixel coordinates.
    target_row = int((roi.lat_max - top_site.lat) / (roi.lat_max - roi.lat_min) * (dem.shape[0] - 1))
    target_col = int((top_site.lon - roi.lon_min) / (roi.lon_max - roi.lon_min) * (dem.shape[1] - 1))
    target_row = max(0, min(dem.shape[0] - 1, target_row))
    target_col = max(0, min(dem.shape[1] - 1, target_col))

    # Fixed start point in opposite quadrant for repeatable route shape.
    start_row = 2 if target_row > dem.shape[0] // 2 else dem.shape[0] - 3
    start_col = 2 if target_col > dem.shape[1] // 2 else dem.shape[1] - 3

    route = plan_route(
        start=(start_row, start_col),
        end=(target_row, target_col),
        weights={
            "distance": 1.0,
            "slope_penalty": 10.0,
            "roughness_penalty": 5.0,
        },
        dem=dem,
        constraints={
            "max_slope_deg": 25.0,
            "max_roughness": 20.0,
            "enable_smoothing": False,
            "cliff_threshold_m": None,
        },
    )

    waypoint_rows = []
    lat_values = dem["y"].values
    lon_values = dem["x"].values
    for index, (row, col) in enumerate(route.waypoints, start=1):
        waypoint_rows.append(
            {
                "waypoint_id": index,
                "pixel_row": int(row),
                "pixel_col": int(col),
                "lat": float(lat_values[row]),
                "lon": float(lon_values[col]),
            }
        )

    site_payload = [site.model_dump() for site in results.sites]
    geojson_payload = {
        "type": "FeatureCollection",
        "features": [_to_feature(site) for site in results.sites],
    }
    brief_payload = build_decision_brief(
        top_site,
        start_lat=float(lat_values[start_row]),
        start_lon=float(lon_values[start_col]),
        seed=seed,
        data_mode=data_mode,
    )
    brief_payload["route_impacts"]["waypoint_count"] = len(waypoint_rows)
    brief_payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    manifest = {
        "demo_name": "marsgis_portfolio_week3",
        "seed": seed,
        "data_mode": data_mode,
        "roi": roi.model_dump(),
        "weights": DEFAULT_WEIGHTS,
        "threshold": threshold,
        "acceptance": {
            "site_selection": {
                "site_count": len(results.sites),
                "top_site_id": int(top_site.site_id),
                "top_site_score": round(float(top_site.suitability_score), 6),
                "has_geojson_features": len(geojson_payload["features"]) > 0,
            },
            "route_planning": {
                "waypoint_count": len(waypoint_rows),
                "distance_m": round(float(route.total_distance_m), 3),
                "has_export": len(waypoint_rows) > 2,
            },
            "decision_brief": {
                "summary_present": bool(brief_payload["summary"]),
                "reasons_count": len(brief_payload["reasons"]),
                "deterministic_fields": ["site_id", "rank", "suitability_score", "route_impacts.distance_m"],
            },
        },
    }

    _write_json(output_dir / "sites_ranked.json", site_payload)
    _write_json(output_dir / "sites_overlay.geojson", geojson_payload)
    _write_waypoints_csv(output_dir / "route_waypoints.csv", waypoint_rows)
    _write_json(output_dir / "decision_brief.json", brief_payload)
    _write_json(output_dir / "demo_manifest.json", manifest)

    print(f"Portfolio demo artifacts written to: {output_dir}")
    print(f"Site count: {len(results.sites)}")
    print(f"Top site: {top_site.site_id} (score {top_site.suitability_score:.3f})")
    print(f"Route waypoints: {len(waypoint_rows)}")
    print(f"Data mode: {data_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic MarsGIS portfolio demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated demo artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for deterministic synthetic fallback",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Suitability threshold for site selection",
    )
    args = parser.parse_args()

    run_demo(
        output_dir=args.output_dir,
        seed=args.seed,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
