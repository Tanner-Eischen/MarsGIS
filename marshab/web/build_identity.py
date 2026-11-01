"""Cached identity of actual backend Python bytes, separate from declared ancestry."""

import copy
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from marshab import __version__


@lru_cache(maxsize=1)
def _identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Backend identity cannot include linked source files")
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    build = os.getenv("MARSHAB_BUILD_ID", "")
    revision = os.getenv("MARSHAB_SOURCE_COMMIT", "")
    return {
        "package_version": __version__,
        "build_id": build if re.fullmatch(r"[A-Za-z0-9._-]{1,80}", build) else None,
        "declared_source_commit": revision
        if re.fullmatch(r"[a-fA-F0-9]{40,64}", revision)
        else None,
        "declared_commit_scope": "Ancestry label only; actual backend byte identity is independent",
        "backend_sha256": fingerprint,
        "backend_python_files": len(files),
        "fingerprint_definition": "SHA-256 of sorted compact JSON mapping package-relative Python paths to byte SHA-256",
        "capture_scope": "Cached once per process; restart after source changes; frontend assets are separate",
    }


def build_identity() -> dict:
    return copy.deepcopy(_identity())
