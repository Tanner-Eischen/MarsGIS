"""FastAPI application for MarsHab web interface."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from marshab.config import get_config
from marshab.utils.logging import configure_logging, get_logger

from .routes import analysis, download, navigation, status, visualization, site_analysis, route_analysis, mission_scenarios, export, projects, health, examples

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("=" * 60)
    print("MarsHab Web API - Starting up...")
    print("=" * 60)
    logger.info("Starting MarsHab web API")
    try:
        config = get_config()
        print("✓ Config loaded")
        config.paths.create_directories()
        print("✓ Directories created")
        
        # Load plugins
        try:
            from marshab.plugins import load_plugins_from_config
            load_plugins_from_config()
            print("✓ Plugins loaded")
        except Exception as e:
            logger.warning("Failed to load plugins", error=str(e))
            print(f"⚠ Plugins failed: {e}")
        
        print("=" * 60)
        print("MarsHab Web API - Ready!")
        print("Server: http://localhost:5000")
        print("API Docs: http://localhost:5000/docs")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        logger.exception("Startup failed")
        raise
    
    yield
    # Shutdown
    print("MarsHab Web API - Shutting down...")
    logger.info("Shutting down MarsHab web API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="MarsHab API",
        description="Mars Habitat Site Selection and Rover Navigation System API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS - allow common development ports
    # In production, this should be restricted to specific domains
    # Note: CORS doesn't support wildcards in origins when credentials are enabled
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:4000",
            "http://localhost:4001",
            "http://localhost:4002",
            "http://localhost:5173",  # Vite default
            "http://127.0.0.1:3000",
            "http://127.0.0.1:4000",
            "http://127.0.0.1:4001",
            "http://127.0.0.1:4002",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] {request.method} {request.url.path} - ERROR: {e} ({process_time:.3f}s)")
            raise
    
    # Include routers
    app.include_router(status.router, prefix="/api/v1", tags=["status"])
    app.include_router(download.router, prefix="/api/v1", tags=["download"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
    app.include_router(navigation.router, prefix="/api/v1/navigation", tags=["navigation"])
    app.include_router(visualization.router, prefix="/api/v1", tags=["visualization"])
    app.include_router(site_analysis.router, tags=["analysis"])
    app.include_router(route_analysis.router, tags=["analysis"])
    app.include_router(mission_scenarios.router, tags=["mission"])
    app.include_router(export.router, tags=["export"])
    app.include_router(projects.router, tags=["projects"])
    app.include_router(health.router, tags=["health"])
    app.include_router(examples.router, tags=["examples"])

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

