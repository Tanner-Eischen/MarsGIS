"""Exercise the actual unified HTTP service without a browser or mocked API."""

import argparse
import hashlib
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def request(base, path, method="GET", data=None):
    encoded = json.dumps(data).encode() if data is not None else None
    headers = {"Content-Type": "application/json"} if encoded is not None else {}
    req = urllib.request.Request(
        base.rstrip("/") + path, method=method, data=encoded, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def payload(base, path, method="GET", expected=200, data=None):
    status, content_type, body = request(base, path, method, data)
    assert status == expected, (path, status, body[:300])
    assert "application/json" in content_type, (path, content_type)
    return json.loads(body)


def _spherical_distance(a, b, radius):
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dphi, dlon = p2 - p1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(max(0, min(1, h))))


def _wait_job(base, job, deadline, request_id):
    identifier = job["id"]
    while job["state"] == "running":
        assert job["request_id"] == request_id
        if time.monotonic() >= deadline:
            raise TimeoutError("The selected-point route did not finish")
        time.sleep(0.2)
        job = payload(base, f"/api/v1/explorer/routes/{identifier}")
        assert job["id"] == identifier and job["request_id"] == request_id
    assert job["state"] == "ready", job
    assert job["result"]["id"] == identifier and job["result"]["request_id"] == request_id
    return job


def _check_route(base, start, end, source, deadline):
    request_id = f"smoke_{uuid4().hex}"
    body = {"start": start, "end": end, "max_slope_deg": 25, "request_id": request_id}
    initial = payload(base, "/api/v1/explorer/routes", "POST", 202, body)
    job = _wait_job(base, initial, deadline, request_id)
    result = job["result"]
    assert result["requested"] == {"start": start, "end": end}
    assert result["source"]["id"] == source
    assert result["grid"]["slope_limit_deg"] == 25
    path, surface = result["path"], result["surface"]
    assert len(path) >= 2 and max(result["grid"]["shape"]) <= 256
    assert max(len(surface["elevation_m"]), len(surface["elevation_m"][0])) <= 96
    assert path[0] == result["snapped"]["start"] and path[-1] == result["snapped"]["end"]
    distance = sum(
        _spherical_distance(a, b, result["source"]["radius_m"])
        for a, b in zip(path[:-1], path[1:], strict=True)
    )
    assert math.isclose(distance, result["distance_m"], rel_tol=1e-9)
    assert result["distance_m"] >= result["straight_line_m"] - 1e-5
    rows, cols = surface["row_indices"], surface["col_indices"]
    for point in path:
        assert all(math.isfinite(point[key]) for key in ("lat", "lon", "x_m", "y_m", "elevation_m"))
        if point["row"] in rows and point["col"] in cols:
            r, c = rows.index(point["row"]), cols.index(point["col"])
            assert surface["x_m"][r][c] == point["x_m"]
            assert surface["y_m"][r][c] == point["y_m"]
            assert surface["elevation_m"][r][c] == point["elevation_m"]
    status, media, raw = request(base, f"/api/v1/explorer/routes/{job['id']}/geojson")
    assert status == 200 and "application/geo+json" in media
    assert json.loads(raw) == result["geojson"]
    assert result["geojson"]["run_id"] == job["id"]
    assert result["geojson"]["request_id"] == request_id
    print(
        f"PASS: {source} selected-point route, spherical length, matching 3D coordinates and GeoJSON ({len(path)} vertices)",
        flush=True,
    )
    return {
        "run_id": job["id"],
        "request_id": request_id,
        "source_id": source,
        "source_sha256": result["source"]["sha256"],
        "vertices": len(path),
        "distance_m": result["distance_m"],
        "terrain_cost_m_equivalent": result["terrain_cost_m_equivalent"],
        "analysis_spacing_m": result["source"]["analysis_resolution_m"],
        "export_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "passed",
    }


def _check_overload(base, deadline, previous):
    """One bounded three-request loopback burst; no synthetic delay or queue."""
    barrier = threading.Barrier(3)

    def submit(_):
        request_id = f"burst_{uuid4().hex}"
        barrier.wait(timeout=10)
        status, _, raw = request(
            base,
            "/api/v1/explorer/routes",
            "POST",
            {
                "start": {"lat": 18.380010, "lon": 77.385212},
                "end": {"lat": 18.384727, "lon": 77.411880},
                "max_slope_deg": 25,
                "request_id": request_id,
            },
        )
        return status, json.loads(raw), request_id

    with ThreadPoolExecutor(max_workers=3) as executor:
        outcomes = list(executor.map(submit, range(3)))
    codes = [status for status, _, _ in outcomes]
    assert set(codes) <= {202, 409} and 202 in codes and 409 in codes, codes
    for status, job, request_id in outcomes:
        if status == 202:
            _wait_job(base, job, deadline, request_id)
    _, _, previous_export = request(base, f"/api/v1/explorer/routes/{previous['run_id']}/geojson")
    assert hashlib.sha256(previous_export).hexdigest() == previous["export_sha256"]
    print(
        "PASS: actual loopback burst returns 409 while one scientific worker runs; prior export unchanged",
        flush=True,
    )
    return {
        "requests": 3,
        "http_statuses": codes,
        "queued_requests": 0,
        "previous_export_unchanged": True,
        "status": "passed",
    }


def smoke(base: str, timeout: int, skip_comparison: bool = False, contracts: bool = False) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if request(base, "/api/v1/health/live")[0] == 200:
                break
        except (OSError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Unified service did not start")
        time.sleep(1)
    live = payload(base, "/api/v1/health/live")
    ready = payload(base, "/api/v1/health/ready")
    assert ready["status"] == "ready" and ready["sources"]["status"] == "verified"
    assert ready["build"]["backend_sha256"] == live["build"]["backend_sha256"]
    assets = {}
    status, media_type, html = request(base, "/")
    assert status == 200 and "text/html" in media_type and b"MarsHab" in html
    for asset in re.findall(rb'(?:src|href)="(/assets/[^" ]+)"', html):
        code, _, content = request(base, asset.decode())
        assert code == 200
        assets[asset.decode()] = hashlib.sha256(content).hexdigest()
        if asset.endswith(b".js"):
            chunks = re.findall(rb"TerrainScene3D-[A-Za-z0-9_-]+\.js", content)
            assert chunks, "The default app must provide the 3D path viewer"
            for chunk in set(chunks):
                path = "/assets/" + chunk.decode()
                code, _, chunk_bytes = request(base, path)
                assert code == 200
                assets[path] = hashlib.sha256(chunk_bytes).hexdigest()
    metadata = payload(base, "/api/v1/explorer/metadata")
    assert (
        metadata["global"]["radius_m"] == 3396000
        and metadata["detail"]["native_resolution_m"] == 20
    )
    for key in ("global", "detail"):
        assert ready["sources"][key]["sha256"] == metadata[key]["sha256"]
    tiles = [request(base, f"/api/v1/explorer/tiles/mola/0/{x}/0.png") for x in (0, 1)]
    assert all(
        code == 200 and "image/png" in media and content.startswith(b"\x89PNG\r\n\x1a\n")
        for code, media, content in tiles
    )
    assert (
        tiles[0][2] != tiles[1][2]
    ), "The two real Mars hemispheres should not be a placeholder image"
    ctx_route = _check_route(
        base,
        {"lat": 18.380010, "lon": 77.385212},
        {"lat": 18.384727, "lon": 77.411880},
        "ctx",
        deadline,
    )
    mola_route = _check_route(
        base, {"lat": 10, "lon": 179}, {"lat": 12, "lon": -179}, "mola16", deadline
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "Actual HTTP transport and selected-point science; no browser or WebGL execution",
        "build": live["build"],
        "ready_sources": ready["sources"],
        "delivered_assets_sha256": assets,
        "routes": [ctx_route, mola_route],
        "fixed_reference_recomputed": not skip_comparison,
        "overload": _check_overload(base, deadline, ctx_route)
        if contracts
        else {"status": "not_run"},
    }
    print("PASS: real global map tiles and the lazy 3D viewer asset", flush=True)
    payload(base, "/api/v1/portfolio/status")
    payload(base, "/api/v1/missing-endpoint", expected=404)
    assert request(base, "/assets/missing-file.js")[0] == 404
    assert "text/html" in request(base, "/explore/jezero")[1]
    if skip_comparison:
        return report
    started = payload(base, "/api/v1/portfolio/run", "POST", expected=202)
    assert started["state"] in {"running", "ready"}
    while True:
        current = payload(base, "/api/v1/portfolio/status")
        if current["state"] == "failed":
            raise RuntimeError(current.get("error", "Scientific calculation failed"))
        if current["state"] == "ready":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Scientific calculation did not finish")
        time.sleep(1)
    scene = payload(base, "/api/v1/portfolio/scene")
    assert scene["data_mode"] == "observed_dem"
    assert scene["execution_mode"] == "server_computed_analysis"
    assert len(scene["candidates"]) == 3 and len(scene["layers"]) == 5
    base_path = scene["asset_base"]
    for path, expected in scene["artifacts_sha256"].items():
        assert not path.startswith("/") and ".." not in path.split("/")
        status, _, body = request(base, base_path + path)
        assert status == 200, (path, status)
        assert hashlib.sha256(body).hexdigest() == expected, path
    print(
        "PASS: same-origin React assets, real API computation, five terrain layers and all download checksums"
    )
    print(f"Run: {scene['run_id']}; grid: {scene['grid']['shape']}; candidates: 3")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--loopback-contracts",
        action="store_true",
        help="Check one actual three-request burst on loopback only",
    )
    parser.add_argument(
        "--report", type=Path, help="Write a new compact JSON result; never overwrite"
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Check the explorer without recalculating the reference candidate comparison",
    )
    args = parser.parse_args()
    if args.loopback_contracts and urllib.parse.urlsplit(args.base_url).hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        parser.error("Contract burst checks require a loopback host")
    if args.report and args.report.exists():
        parser.error("Report exists; retain prior evidence and use a new path")
    result = smoke(args.base_url, args.timeout, args.skip_comparison, args.loopback_contracts)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
