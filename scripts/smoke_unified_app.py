"""Exercise the actual unified HTTP service without a browser or mocked API."""

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request


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


def _check_route(base, start, end, source, deadline):
    body = {"start": start, "end": end, "max_slope_deg": 25}
    job = payload(base, "/api/v1/explorer/routes", "POST", 202, body)
    while job["state"] == "running":
        if time.monotonic() >= deadline:
            raise TimeoutError("The selected-point route did not finish")
        time.sleep(1)
        job = payload(base, f"/api/v1/explorer/routes/{job['id']}")
    assert job["state"] == "ready", job
    result = job["result"]
    assert result["requested"] == {"start": start, "end": end}
    assert result["source"]["id"] == source
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
    print(
        f"PASS: {source} selected-point route, spherical length, matching 3D coordinates and GeoJSON ({len(path)} vertices)",
        flush=True,
    )


def smoke(base: str, timeout: int, skip_comparison: bool = False) -> None:
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
    status, media_type, html = request(base, "/")
    assert status == 200 and "text/html" in media_type and b"MarsHab" in html
    for asset in re.findall(rb'(?:src|href)="(/assets/[^" ]+)"', html):
        code, _, content = request(base, asset.decode())
        assert code == 200
        if asset.endswith(b".js"):
            chunks = re.findall(rb"TerrainScene3D-[A-Za-z0-9_-]+\.js", content)
            assert chunks, "The default app must provide the 3D path viewer"
            assert all(
                request(base, "/assets/" + chunk.decode())[0] == 200 for chunk in set(chunks)
            )
    metadata = payload(base, "/api/v1/explorer/metadata")
    assert (
        metadata["global"]["radius_m"] == 3396000
        and metadata["detail"]["native_resolution_m"] == 20
    )
    tiles = [request(base, f"/api/v1/explorer/tiles/mola/0/{x}/0.png") for x in (0, 1)]
    assert all(
        code == 200 and "image/png" in media and content.startswith(b"\x89PNG\r\n\x1a\n")
        for code, media, content in tiles
    )
    assert (
        tiles[0][2] != tiles[1][2]
    ), "The two real Mars hemispheres should not be a placeholder image"
    _check_route(
        base,
        {"lat": 18.380010, "lon": 77.385212},
        {"lat": 18.384727, "lon": 77.411880},
        "ctx",
        deadline,
    )
    _check_route(base, {"lat": 10, "lon": 179}, {"lat": 12, "lon": -179}, "mola16", deadline)
    print("PASS: real global map tiles and the lazy 3D viewer asset", flush=True)
    payload(base, "/api/v1/portfolio/status")
    payload(base, "/api/v1/missing-endpoint", expected=404)
    assert request(base, "/assets/missing-file.js")[0] == 404
    assert "text/html" in request(base, "/explore/jezero")[1]
    if skip_comparison:
        return
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Check the explorer without recalculating the reference candidate comparison",
    )
    args = parser.parse_args()
    smoke(args.base_url, args.timeout, args.skip_comparison)
