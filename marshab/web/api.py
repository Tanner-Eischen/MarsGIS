"""FastAPI application for MarsHab web interface."""

import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from marshab.config import get_config
from marshab.utils.logging import get_logger

from .routes import (
    ai_query,
    analysis,
    download,
    examples,
    export,
    health,
    mission_scenarios,
    ml_recommendation,
    navigation,
    progress,
    projects,
    route_analysis,
    site_analysis,
    solar_analysis,
    status,
    visualization,
)

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

    # Configure CORS.
    # Default is local dev origins. Production can extend with:
    # MARSHAB_CORS_ORIGINS=https://frontend.example.com,https://preview.example.com
    # MARSHAB_CORS_ALLOW_ALL=true (disables credentials)
    default_origins = [
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
    ]
    configured_origins = os.getenv("MARSHAB_CORS_ORIGINS", "")
    extra_origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    # Portfolio deployments span multiple preview domains; default to permissive CORS unless explicitly disabled.
    allow_all = os.getenv("MARSHAB_CORS_ALLOW_ALL", "true").lower() in {"1", "true", "yes"}
    cors_origins = ["*"] if allow_all else [*default_origins, *extra_origins]
    default_origin_regex = r"^https://([A-Za-z0-9-]+\.)*(vercel\.app|onrender\.com)$"
    configured_origin_regex = os.getenv("MARSHAB_CORS_ORIGIN_REGEX", "").strip()
    allow_origin_regex = configured_origin_regex or default_origin_regex

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=not allow_all,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.middleware("http")
    async def ensure_cors_headers(request: Request, call_next):
        """Fallback CORS headers so browser doesn't drop useful error responses."""
        origin = request.headers.get("origin")
        requested_headers = request.headers.get("access-control-request-headers", "*")
        requested_method = request.headers.get("access-control-request-method", "GET")

        def origin_allowed(value: str | None) -> bool:
            if not value:
                return False
            if allow_all:
                return True
            if value in cors_origins:
                return True
            if allow_origin_regex:
                try:
                    return re.match(allow_origin_regex, value) is not None
                except re.error:
                    return False
            return False

        if request.method == "OPTIONS" and origin_allowed(origin):
            response = Response(status_code=200)
            response.headers["Access-Control-Allow-Origin"] = "*" if allow_all else str(origin)
            response.headers["Access-Control-Allow-Methods"] = requested_method
            response.headers["Access-Control-Allow-Headers"] = requested_headers
            response.headers["Access-Control-Max-Age"] = "86400"
            response.headers["Vary"] = "Origin"
            if not allow_all:
                response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

        response = await call_next(request)
        if origin_allowed(origin):
            response.headers.setdefault("Access-Control-Allow-Origin", "*" if allow_all else str(origin))
            response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
            response.headers.setdefault("Access-Control-Allow-Headers", requested_headers)
            response.headers.setdefault("Vary", "Origin")
            if not allow_all:
                response.headers.setdefault("Access-Control-Allow-Credentials", "true")
        return response

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

    # Root route
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "MarsHab API",
            "version": "0.1.0",
            "description": "Mars Habitat Site Selection and Rover Navigation System API",
            "docs": "/docs",
            "api_base": "/api/v1",
            "endpoints": {
                "status": "/api/v1/status",
                "health": "/api/v1/health/live",
                "docs": "/docs"
            }
        }

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
    app.include_router(progress.router, tags=["progress"])
    app.include_router(solar_analysis.router, prefix="/api/v1", tags=["solar"])
    app.include_router(ml_recommendation.router, prefix="/api/v1", tags=["machine-learning"])
    app.include_router(ai_query.router, prefix="/api/v1", tags=["ai"])

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
