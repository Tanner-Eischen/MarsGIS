"""Server entry point for MarsHab web API."""

import os

import uvicorn


def main():
    """Main entry point for web server."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print("=" * 60)
    print("Starting MarsHab Web API Server")
    print("=" * 60)
    host = os.getenv("MARSHAB_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("MARSHAB_WEB_PORT", "5000")))
    reload_enabled = os.getenv("MARSHAB_WEB_RELOAD", "false").lower() in ("1", "true", "yes")
    workers = int(os.getenv("MARSHAB_WEB_WORKERS", os.getenv("WEB_CONCURRENCY", "1")))
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API docs available at: http://{host}:{port}/docs")
    print(f"Workers: {workers}")
    print("=" * 60)
    run_kwargs = {
        "app": "marshab.web.api:app",  # Use import string for reload support
        "host": host,
        "port": port,
        "reload": reload_enabled,
        "log_level": "info",
    }
    # Workers > 1 improve tile throughput for concurrent browser requests.
    # Uvicorn cannot combine workers with reload mode.
    if not reload_enabled and workers > 1:
        run_kwargs["workers"] = workers
    uvicorn.run(**run_kwargs)


if __name__ == "__main__":
    main()
