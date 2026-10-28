"""Exercise the actual unified HTTP service without a browser or mocked API."""

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request


def request(base, path, method="GET"):
    req = urllib.request.Request(base.rstrip("/") + path, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def payload(base, path, method="GET", expected=200):
    status, content_type, body = request(base, path, method)
    assert status == expected, (path, status, body[:300])
    assert "application/json" in content_type, (path, content_type)
    return json.loads(body)


def smoke(base: str, timeout: int) -> None:
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
        assert request(base, asset.decode())[0] == 200
    payload(base, "/api/v1/portfolio/status")
    payload(base, "/api/v1/missing-endpoint", expected=404)
    assert request(base, "/assets/missing-file.js")[0] == 404
    assert "text/html" in request(base, "/explore/jezero")[1]
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
    args = parser.parse_args()
    smoke(args.base_url, args.timeout)
