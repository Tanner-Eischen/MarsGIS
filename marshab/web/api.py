"""FastAPI application for MarsHab web interface."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from marshab.config import get_config
from marshab.utils.logging import configure_logging, get_logger

from .routes import analysis, download, navigation, status, visualization

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MarsHab web API")
    config = get_config()
    config.paths.create_directories()
    yield
    # Shutdown
    logger.info("Shutting down MarsHab web API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="MarsHab API",
        description="Mars Habitat Site Selection and Rover Navigation System API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:4000",
            "http://localhost:4001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:4000",
            "http://127.0.0.1:4001",
        ],  # Frontend ports - backend runs on 5000
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(status.router, prefix="/api/v1", tags=["status"])
    app.include_router(download.router, prefix="/api/v1", tags=["download"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
    app.include_router(navigation.router, prefix="/api/v1", tags=["navigation"])
    app.include_router(visualization.router, prefix="/api/v1", tags=["visualization"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.exception("Unhandled exception", exc=exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


# Create app instance
app = create_app()

