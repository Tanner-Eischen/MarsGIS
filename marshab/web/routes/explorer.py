"""Observed planetary map tiles and bounded asynchronous route calculations."""

import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from marshab.config import get_config
from marshab.core.explorer_service import ExplorerService
from marshab.core.planetary_data import from_config
from marshab.core.workload import WorkloadBusyError

router = APIRouter(prefix="/api/v1/explorer", tags=["explorer"])
_initialization_lock = threading.Lock()


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    lat: float = Field(ge=-90, le=90, strict=True)
    lon: float = Field(ge=-180, le=180, strict=True)


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start: Point
    end: Point
    max_slope_deg: float = Field(default=25.0, gt=0, lt=90, strict=True)
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$", strict=True)


def get_service(request: Request) -> ExplorerService:
    with _initialization_lock:
        service = getattr(request.app.state, "explorer_service", None)
        if service is None:
            try:
                service = ExplorerService(from_config(get_config()))
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            request.app.state.explorer_service = service
        return service


@router.get("/metadata")
def metadata(request: Request):
    return get_service(request).data.metadata()


@router.get("/tiles/{layer}/{z}/{x}/{y}.png")
def tile(layer: str, z: int, x: int, y: int, request: Request):
    try:
        content = get_service(request).data.tile(layer, z, x, y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return Response(
        content=content, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
    )


@router.post("/routes", status_code=202)
def route(body: RouteRequest, request: Request):
    if body.start == body.end:
        raise HTTPException(status_code=400, detail="Select two different points")
    try:
        return get_service(request).start(
            body.start.model_dump(), body.end.model_dump(), body.max_slope_deg, body.request_id
        )
    except WorkloadBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/routes/{identifier}")
def route_status(identifier: str, request: Request):
    try:
        job = get_service(request).job(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JSONResponse(job, headers={"Cache-Control": "no-store"})


@router.get("/routes/{identifier}/geojson")
def route_geojson(identifier: str, request: Request):
    try:
        payload = get_service(request).geojson(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return JSONResponse(
        payload,
        media_type="application/geo+json",
        headers={
            "Content-Disposition": f'attachment; filename="marshab-route-{identifier}.geojson"'
        },
    )
