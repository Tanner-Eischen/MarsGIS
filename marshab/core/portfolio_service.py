"""One-at-a-time observed-DEM analysis with immutable verified result snapshots."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from marshab.analysis.portfolio import SOURCE_CITATION, SOURCE_SHA256, SOURCE_URL, run_demo
from marshab.analysis.site_package import build_package, digest, verify_package
from marshab.config import Config


class AnalysisAlreadyRunningError(RuntimeError):
    """The process already has an active portfolio calculation."""


@dataclass(frozen=True)
class ResultSnapshot:
    folder: Path
    scene: dict
    hashes: dict[str, str]
    execution_mode: str
    computed_at_utc: str | None = None


class PortfolioService:
    """Compute fresh scientific results and publish only fully verified packages.

    Intended for one application process/worker. The latest five successful live
    runs and the reference stay available while a new run computes or fails. No request supplies arbitrary file paths,
    dataset URLs, scientific configurations, or unbounded job queues.
    """

    RETAINED_LIVE_RUNS = 5

    def __init__(self, config: Config, reference_dir: Path | None = None):
        self.source_path = Path(
            os.getenv("MARSHAB_REFERENCE_DEM_PATH")
            or os.getenv("MARSHAB_PORTFOLIO_DEM_PATH")
            or config.paths.cache_dir / "jezero_ctx_20m.tif"
        ).resolve()
        self.output_root = (config.paths.output_dir / "portfolio-runs").resolve()
        self._lock = threading.Lock()
        self._state = "idle"
        self._error: str | None = None
        self._current: str | None = None
        self._running: str | None = None
        self._runs: dict[str, ResultSnapshot] = {}
        self._thread: threading.Thread | None = None
        reference = self._find_reference(reference_dir)
        if reference is not None:
            try:
                verify_package(reference)
                scene = json.loads((reference / "scene.json").read_text(encoding="utf-8"))
                identifier = "reference-" + digest(reference / "scene.json")[:16]
                self._runs[identifier] = ResultSnapshot(
                    reference.resolve(),
                    scene,
                    scene["artifacts_sha256"].copy(),
                    "precomputed_reference_analysis",
                )
                self._current = identifier
                self._state = "ready"
            except Exception as exc:
                self._state = "failed"
                self._error = f"Reference package verification failed: {exc}"

    @staticmethod
    def _find_reference(explicit: Path | None) -> Path | None:
        configured = explicit or os.getenv("MARSHAB_PORTFOLIO_REFERENCE_DIR")
        if configured:
            return Path(configured)
        root = Path(__file__).resolve().parents[2]
        dist = Path(os.getenv("MARSHAB_WEBUI_DIST") or root / "webui" / "dist")
        for folder in (dist / "portfolio", root / "webui" / "public" / "portfolio"):
            if (folder / "scene.json").is_file():
                return folder
        return None

    def status(self) -> dict:
        with self._lock:
            result = {
                "state": self._state,
                "data_available": self._current is not None,
                "scene_url": "/api/v1/portfolio/scene" if self._current else None,
                "source": {"url": SOURCE_URL, "sha256": SOURCE_SHA256, "citation": SOURCE_CITATION},
                "current_run_id": self._current,
                "running_run_id": self._running,
                "retained_live_runs": sum(
                    run.execution_mode == "server_computed_analysis" for run in self._runs.values()
                ),
            }
            if self._error:
                result["error"] = self._error
            return result

    def start(self) -> dict:
        """Reserve the sole worker slot before starting the daemon thread."""
        with self._lock:
            if self._state == "running":
                raise AnalysisAlreadyRunningError("A portfolio analysis is already running")
            identifier = uuid4().hex
            self._state, self._running, self._error = "running", identifier, None
            self._thread = threading.Thread(
                target=self._work,
                args=(identifier,),
                name=f"portfolio-{identifier[:8]}",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception as exc:
                self._state, self._running = "failed", None
                self._error = f"Could not start analysis worker: {exc}"
                raise
        return self.status()

    def _work(self, identifier: str) -> None:
        try:
            folder = self.output_root / identifier
            analysis, package = folder / "analysis", folder / "package"
            # run_demo performs checksum-pinned acquisition; no synthetic fallback.
            run_demo(self.source_path, analysis)
            scene = build_package(analysis, package)
            verify_package(package)
            completed = datetime.now(UTC).isoformat()
            snapshot = ResultSnapshot(
                package,
                scene,
                scene["artifacts_sha256"].copy(),
                "server_computed_analysis",
                completed,
            )
            self._make_retention_room()
            with self._lock:
                self._runs[identifier] = snapshot
                self._current = identifier
                self._state, self._running, self._error = "ready", None, None
        except Exception as exc:
            # Keep the worker slot reserved until failed output cleanup finishes.
            with self._lock:
                self._running = None
            cleanup_error = ""
            try:
                self._remove_run_folder(identifier)
            except (OSError, ValueError) as cleanup_exc:
                cleanup_error = f" Partial output cleanup failed safely: {cleanup_exc}"
            with self._lock:
                self._state = "failed"
                self._error = (
                    f"Observed DEM analysis failed: {str(exc)[:500]}. "
                    "Check the pinned DEM, network access, and writable cache/output directories."
                    + cleanup_error
                )

    def _make_retention_room(self) -> None:
        """Remove oldest live snapshots before publishing the next verified run."""
        with self._lock:
            live = [
                identifier
                for identifier, run in self._runs.items()
                if run.execution_mode == "server_computed_analysis"
            ]
        for identifier in live[: max(0, len(live) - self.RETAINED_LIVE_RUNS + 1)]:
            self._remove_run_folder(identifier)
            with self._lock:
                del self._runs[identifier]

    def _remove_run_folder(self, identifier: str) -> None:
        """Delete only an internal UUID's checked direct child of the output root."""
        with self._lock:
            protected = identifier in {self._current, self._running}
            snapshot = self._runs.get(identifier)
            protected |= (
                snapshot is not None and snapshot.execution_mode != "server_computed_analysis"
            )
        if protected or re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
            raise ValueError("Refusing to remove a protected or non-UUID run")
        root = self.output_root.resolve()
        target = self.output_root / identifier
        resolved = target.resolve()
        if (
            root != self.output_root
            or target.is_symlink()
            or resolved != target
            or resolved.parent != root
            or not resolved.is_relative_to(root)
            or self.source_path.is_relative_to(resolved)
        ):
            raise ValueError("Refusing to remove an aliased run or a path outside the output root")
        if resolved.exists():
            shutil.rmtree(resolved)

    def scene(self) -> dict:
        with self._lock:
            if self._current is None:
                raise FileNotFoundError(
                    "No verified portfolio result is available; start an analysis"
                )
            identifier = self._current
            snapshot = self._runs[identifier]
            result = copy.deepcopy(snapshot.scene)
            result.update(
                asset_base=f"/api/v1/portfolio/files/{identifier}/",
                execution_mode=snapshot.execution_mode,
                analysis_id=identifier,
                run_id=identifier,
            )
            if snapshot.computed_at_utc:
                result["computed_at_utc"] = snapshot.computed_at_utc
            return result

    def file(self, identifier: str, relative: str) -> Path:
        """Resolve only known immutable artifacts, checking containment and bytes."""
        with self._lock:
            snapshot = self._runs.get(identifier)
            if snapshot is None or relative not in snapshot.hashes:
                raise FileNotFoundError("Unknown portfolio artifact")
            folder, expected = snapshot.folder.resolve(), snapshot.hashes[relative]
        # Reject Windows separators/drives as well as POSIX traversal, on every OS.
        if (
            not relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or "\\" in relative
            or ":" in relative
        ):
            raise FileNotFoundError("Invalid portfolio artifact path")
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise FileNotFoundError("Unknown portfolio artifact")
        if digest(path) != expected:
            raise ValueError("Portfolio artifact no longer matches its verified checksum")
        return path
