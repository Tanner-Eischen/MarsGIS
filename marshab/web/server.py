"""Server entry point for MarsHab web API."""

import uvicorn

from marshab.web.api import app


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
    print("Server will be available at: http://localhost:5000")
    print("API docs available at: http://localhost:5000/docs")
    print("=" * 60)
    uvicorn.run(
        "marshab.web.api:app",  # Use import string for reload support
        host="0.0.0.0",
        port=5000,
        reload=True,  # Enable auto-reload in development
        log_level="info",
    )


if __name__ == "__main__":
    main()

