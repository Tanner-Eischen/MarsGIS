"""Same-origin access to fresh observed-DEM computations and verified artifacts."""

import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from marshab.config import get_config
from marshab.core.portfolio_service import AnalysisAlreadyRunningError, PortfolioService

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])
_initialization_lock = threading.Lock()


def get_service(request: Request) -> PortfolioService:
    """Lazy per-app initialization uses the application's current configuration."""
    with _initialization_lock:
        service = getattr(request.app.state, "portfolio_service", None)
        if service is None:
            service = PortfolioService(get_config())
            request.app.state.portfolio_service = service
        return service


@router.get("/status")
def portfolio_status(request: Request):
    return get_service(request).status()


@router.post("/run", status_code=202)
def run_portfolio(request: Request):
    service = get_service(request)
    try:
        return service.start()
    except AnalysisAlreadyRunningError:
        return JSONResponse(
            status_code=409,
            content={**service.status(), "error": "A portfolio analysis is already running"},
        )


@router.get("/scene")
def portfolio_scene(request: Request):
    try:
        return JSONResponse(get_service(request).scene(), headers={"Cache-Control": "no-store"})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/files/{run_id}/{path:path}")
def portfolio_file(run_id: str, path: str, request: Request):
    try:
        file = get_service(request).file(run_id, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return FileResponse(file, headers={"Cache-Control": "public, max-age=31536000, immutable"})
