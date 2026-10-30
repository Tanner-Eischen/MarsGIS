"""Bounded asynchronous route jobs over already provisioned planetary data."""

from __future__ import annotations

import copy
import json
import threading
from collections import OrderedDict
from uuid import uuid4

from marshab.analysis.planetary_routing import calculate_route
from marshab.core.workload import WorkloadBusyError, acquire_workload, release_workload


class ExplorerService:
    RETAINED_JOBS = 5
    MAX_RESULT_BYTES = 8 * 1024 * 1024
    MAX_RETAINED_BYTES = 24 * 1024 * 1024

    def __init__(self, data):
        self.data = data
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._running: str | None = None
        self._thread: threading.Thread | None = None

    def start(self, start: dict, end: dict, max_slope_deg: float) -> dict:
        with self._lock:
            if self._running is not None or not acquire_workload():
                raise WorkloadBusyError(
                    "Another scientific calculation is running; try again when it finishes"
                )
            identifier = uuid4().hex
            job = {
                "id": identifier,
                "state": "running",
                "poll_url": f"/api/v1/explorer/routes/{identifier}",
            }
            self._jobs[identifier] = job
            self._running = identifier
            try:
                self._thread = threading.Thread(
                    target=self._work,
                    args=(identifier, copy.deepcopy(start), copy.deepcopy(end), max_slope_deg),
                    daemon=True,
                    name=f"explorer-{identifier[:8]}",
                )
                self._thread.start()
            except Exception:
                self._jobs.pop(identifier)
                self._running = None
                release_workload()
                raise
            return dict(job)

    def _work(self, identifier: str, start: dict, end: dict, max_slope_deg: float) -> None:
        try:
            result = calculate_route(self.data, start, end, max_slope_deg, identifier)
            size = len(json.dumps(result, allow_nan=False, separators=(",", ":")).encode())
            if size > self.MAX_RESULT_BYTES:
                raise ValueError("Route exceeds the bounded result size; select a smaller region")
            with self._lock:
                self._jobs[identifier].update(state="ready", result=result)
                self._sizes[identifier] = size
        except Exception as exc:
            with self._lock:
                self._jobs[identifier].update(state="failed", error=str(exc)[:600])
                self._sizes[identifier] = 0
        finally:
            with self._lock:
                self._running = None
                while (
                    len(self._jobs) > self.RETAINED_JOBS
                    or sum(self._sizes.values()) > self.MAX_RETAINED_BYTES
                ):
                    oldest = next(iter(self._jobs))
                    self._jobs.pop(oldest)
                    self._sizes.pop(oldest, None)
            release_workload()

    def job(self, identifier: str) -> dict:
        with self._lock:
            if identifier not in self._jobs:
                raise FileNotFoundError("Unknown or expired route job")
            return copy.deepcopy(self._jobs[identifier])

    def geojson(self, identifier: str) -> dict:
        job = self.job(identifier)
        if job["state"] != "ready":
            raise ValueError("Route is not ready for download")
        return job["result"]["geojson"]
