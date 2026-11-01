"""Serve the built React client after API routes on the same origin."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


def register_webui(app: FastAPI) -> None:
    """Add safe file serving and a GET-only, extensionless SPA fallback.

    API/WebSocket namespaces and missing static resources always retain a JSON
    404. Resolving every candidate within the build directory also rejects
    traversal and symlinks pointing outside the published client.
    """
    default_dist = Path(__file__).resolve().parents[2] / "webui" / "dist"
    dist = Path(os.getenv("MARSHAB_WEBUI_DIST", str(default_dist))).resolve()

    async def unknown_service_path():
        raise HTTPException(status_code=404, detail="Not found")

    for path in ("/api", "/api/{unmatched:path}", "/ws", "/ws/{unmatched:path}"):
        app.add_api_route(
            path,
            unknown_service_path,
            methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            include_in_schema=False,
        )

    @app.get("/{resource:path}", include_in_schema=False)
    async def frontend(resource: str):
        parts = PurePosixPath(resource).parts
        if "\\" in resource or ":" in resource or any(part in {".", ".."} for part in parts):
            raise HTTPException(status_code=404, detail="Not found")
        try:
            candidate = (dist / resource).resolve()
            candidate.relative_to(dist)
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="Not found")
        if candidate.is_file():
            return FileResponse(candidate)

        # Resource URLs must never silently return index.html. The portfolio
        # directory contains scientific exports as well as images and JSON.
        if parts and (parts[0] in {"assets", "portfolio"} or PurePosixPath(resource).suffix):
            raise HTTPException(status_code=404, detail="Not found")
        index = (dist / "index.html").resolve()
        try:
            index.relative_to(dist)
        except ValueError:
            raise HTTPException(status_code=404, detail="Not found")
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Frontend build not available")
        return FileResponse(index, media_type="text/html")
