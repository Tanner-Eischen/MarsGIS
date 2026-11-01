"""Process liveness and pinned-source readiness, without source provisioning."""

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from marshab.config import get_config
from marshab.utils.logging import get_logger
from marshab.web.build_identity import build_identity
from marshab.web.routes.explorer import get_service

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    checks: dict
    build: dict
    sources: dict


@router.get("/live")
def health_live():
    """Liveness reads backend identity, never opens/provisions scientific sources."""
    return {"status": "alive", "build": build_identity()}


@router.get("/ready", response_model=HealthResponse)
def health_ready(request: Request, response: Response):
    """Validate existing pinned sources once; never download from a health request."""
    checks = {}
    sources = {"status": "unavailable"}
    try:
        config = get_config()
        checks["config"] = "ready"
        for label, directory in (
            ("cache_dir", config.paths.cache_dir),
            ("output_dir", config.paths.output_dir),
        ):
            checks[label] = "ready" if directory.is_dir() else "not_ready"
        # This store only validates existing pinned files. Provisioning is a separate CLI.
        metadata = get_service(request).data.metadata()
        sources = {
            "status": "verified",
            "verification_scope": "Pinned file checksums verified when the process source store initializes",
            "global": {key: metadata["global"][key] for key in ("id", "sha256")},
            "detail": {key: metadata["detail"][key] for key in ("id", "sha256")},
        }
        checks["planetary_sources"] = "ready"
    except Exception as error:
        logger.warning("Scientific readiness failed", error=str(error))
        checks["planetary_sources"] = "not_ready"
        sources["reason"] = "Existing pinned planetary sources are unavailable or invalid"
    ready = all(state == "ready" for state in checks.values())
    response.status_code = 200 if ready else 503
    return HealthResponse(
        status="ready" if ready else "not_ready",
        checks=checks,
        build=build_identity(),
        sources=sources,
    )
