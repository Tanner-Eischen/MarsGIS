"""Smoke check tile + 3D readiness with real-data-only enforcement."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_API_BASE = "http://localhost:5000/api/v1"
DEFAULT_ROI = "18.25,18.45,77.25,77.45"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parse_roi(roi_text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'lat_min,lat_max,lon_min,lon_max'")
    lat_min, lat_max, lon_min, lon_max = map(float, parts)
    if lat_min >= lat_max:
        raise ValueError("ROI lat_min must be < lat_max")
    if lon_min >= lon_max:
        raise ValueError("ROI lon_min must be < lon_max")
    return lat_min, lat_max, lon_min, lon_max


def _tile_indices_for_roi(roi: tuple[float, float, float, float], zoom: int) -> tuple[int, int]:
    lat_min, lat_max, lon_min, lon_max = roi
    lat = (lat_min + lat_max) / 2.0
    lon = (lon_min + lon_max) / 2.0
    x_cols = 2 ** (zoom + 1)
    y_rows = 2**zoom
    x = int(((lon + 180.0) / 360.0) * x_cols)
    y = int(((90.0 - lat) / 180.0) * y_rows)
    x = max(0, min(x_cols - 1, x))
    y = max(0, min(y_rows - 1, y))
    return x, y


def _http_request(
    method: str,
    url: str,
    timeout_s: float,
    payload: dict[str, Any] | None = None,
) -> tuple[bytes, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            body = response.read()
            return body, response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} - {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def _http_get_json(url: str, timeout_s: float) -> tuple[dict[str, Any], Any]:
    body, headers = _http_request("GET", url, timeout_s=timeout_s)
    try:
        return json.loads(body.decode("utf-8")), headers
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[dict[str, Any], Any]:
    body, headers = _http_request("POST", url, timeout_s=timeout_s, payload=payload)
    try:
        return json.loads(body.decode("utf-8")), headers
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


def run_smoke(args: argparse.Namespace) -> None:
    roi = _parse_roi(args.roi)
    x, y = _tile_indices_for_roi(roi, args.zoom)
    allow_fallback = args.allow_fallback

    print(f"[INFO] API base: {args.api_base}")
    print(f"[INFO] ROI: {args.roi}")
    print(f"[INFO] Tile z/x/y: {args.zoom}/{x}/{y}")
    print(f"[INFO] Require real data only: {str(not allow_fallback).lower()}")

    health_url = f"{args.api_base}/health/live"
    health, _headers = _http_get_json(health_url, timeout_s=args.timeout_s)
    _expect(isinstance(health, dict), "Health response must be a JSON object")
    print("[PASS] Health check")

    prewarm_url = f"{args.api_base}/prewarm/mola-tiles"
    prewarm_payload = {
        "roi": [roi[0], roi[1], roi[2], roi[3]],
        "tile_deg": args.tile_deg,
        "force": args.force_prewarm,
    }
    prewarm, _headers = _http_post_json(prewarm_url, prewarm_payload, timeout_s=args.timeout_s)
    _expect(prewarm.get("status") == "success", "Prewarm did not report success")
    _expect(prewarm.get("processed_count", 0) > 0, "Prewarm processed_count must be > 0")
    print(
        "[PASS] Prewarm complete "
        f"(tiles={prewarm.get('total_tiles')}, duration_s={prewarm.get('total_duration_s')})"
    )

    basemap_url = f"{args.api_base}/visualization/tiles/basemap/mola/{args.zoom}/{x}/{y}.png"
    basemap_bytes, basemap_headers = _http_request("GET", basemap_url, timeout_s=args.timeout_s)
    basemap_type = str(basemap_headers.get("Content-Type", "")).lower()
    _expect("image/png" in basemap_type, "Basemap tile content type is not image/png")
    _expect(len(basemap_bytes) > 100, "Basemap tile payload is unexpectedly small")
    if not allow_fallback:
        basemap_fallback = str(basemap_headers.get("X-Fallback-Used", "")).lower()
        _expect(basemap_fallback == "false", "Basemap tile used fallback data")
    print("[PASS] Basemap tile ready")

    overlay_url = (
        f"{args.api_base}/visualization/tiles/overlay/slope/mola/{args.zoom}/{x}/{y}.png"
        "?colormap=viridis"
    )
    overlay_bytes, overlay_headers = _http_request("GET", overlay_url, timeout_s=args.timeout_s)
    overlay_type = str(overlay_headers.get("Content-Type", "")).lower()
    _expect("image/png" in overlay_type, "Overlay tile content type is not image/png")
    _expect(len(overlay_bytes) > 100, "Overlay tile payload is unexpectedly small")
    if not allow_fallback:
        overlay_fallback = str(overlay_headers.get("X-Fallback-Used", "")).lower()
        _expect(overlay_fallback == "false", "Overlay tile used fallback data")
    print("[PASS] Overlay tile ready")

    terrain_qs = urllib.parse.urlencode(
        {
            "dataset": "mola",
            "roi": args.roi,
            "max_points": args.max_points,
        }
    )
    terrain_url = f"{args.api_base}/visualization/terrain-3d?{terrain_qs}"
    terrain, _headers = _http_get_json(terrain_url, timeout_s=args.timeout_s)
    for key in ("x", "y", "z", "bounds", "elevation_range"):
        _expect(key in terrain, f"terrain-3d missing key: {key}")

    x_grid = terrain["x"]
    y_grid = terrain["y"]
    z_grid = terrain["z"]
    _expect(isinstance(x_grid, list) and len(x_grid) > 0, "terrain-3d x grid is empty")
    _expect(isinstance(y_grid, list) and len(y_grid) > 0, "terrain-3d y grid is empty")
    _expect(isinstance(z_grid, list) and len(z_grid) > 0, "terrain-3d z grid is empty")
    _expect(len(y_grid) == len(z_grid), "terrain-3d y/z row counts do not match")
    _expect(
        isinstance(z_grid[0], list) and isinstance(x_grid[0], list) and len(x_grid[0]) == len(z_grid[0]),
        "terrain-3d x/z column counts do not match",
    )

    if not allow_fallback:
        _expect(not terrain.get("used_synthetic", False), "terrain-3d used synthetic surface")
        _expect(not terrain.get("is_fallback", False), "terrain-3d used dataset fallback")
    print("[PASS] 3D terrain ready")

    print("[PASS] Tile + 3D readiness smoke check completed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke check tile + 3D readiness.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--roi", default=DEFAULT_ROI, help="ROI as lat_min,lat_max,lon_min,lon_max")
    parser.add_argument("--zoom", type=int, default=6, help="Tile zoom level")
    parser.add_argument("--tile-deg", type=float, default=0.2, help="Prewarm tile size in degrees")
    parser.add_argument("--max-points", type=int, default=10000, help="3D max_points query value")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback/synthetic data instead of failing in real-data-only checks",
    )
    parser.add_argument(
        "--force-prewarm",
        action="store_true",
        help="Force re-download during prewarm (default uses cache)",
    )
    args = parser.parse_args()

    if args.zoom < 0:
        print("[FAIL] --zoom must be >= 0")
        return 2
    if args.tile_deg <= 0:
        print("[FAIL] --tile-deg must be > 0")
        return 2

    try:
        run_smoke(args)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
