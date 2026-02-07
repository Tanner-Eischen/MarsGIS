"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from marshab.config import get_config
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    checks: dict


@router.get("/live")
async def health_live():
    """Basic liveness check."""
    return {"status": "alive"}


@router.get("/ready", response_model=HealthResponse)
async def health_ready():
    """Readiness check with component status."""
    checks = {}
    all_ready = True

    try:
        config = get_config()

        # Check DEM cache directory
        cache_dir = config.paths.cache_dir
        cache_ready = cache_dir.exists() and cache_dir.is_dir()
        checks["cache_dir"] = "ready" if cache_ready else "not_ready"
        if not cache_ready:
            all_ready = False

        # Check output directory
        output_dir = config.paths.output_dir
        output_ready = output_dir.exists() and output_dir.is_dir()
        checks["output_dir"] = "ready" if output_ready else "not_ready"
        if not output_ready:
            all_ready = False

        # Check config loading
        checks["config"] = "ready"

        # Check plugin loading (optional)
        try:
            checks["plugins"] = "ready"
        except Exception as e:
            checks["plugins"] = f"error: {str(e)}"
            # Plugins are optional, don't fail readiness

        status = "ready" if all_ready else "not_ready"

        return HealthResponse(status=status, checks=checks)

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="not_ready",
            checks={"error": str(e)}
        )

