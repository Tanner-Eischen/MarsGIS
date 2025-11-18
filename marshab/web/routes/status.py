"""Status and system information endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from marshab.config import get_config
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/status")
async def get_status():
    """Get system status and cache information."""
    logger.info("Status endpoint called")
    try:
        config = get_config()
        logger.info("Config loaded")
        
        # Check cache directory
        cache_dir = config.paths.cache_dir
        cache_size = 0
        cache_files = 0
        if cache_dir.exists():
            for file in cache_dir.glob("*.tif"):
                cache_size += file.stat().st_size
                cache_files += 1
        
        # Check output directory
        output_dir = config.paths.output_dir
        output_files = 0
        if output_dir.exists():
            output_files = len(list(output_dir.glob("*.csv")))
        
        return {
            "status": "operational",
            "cache": {
                "directory": str(cache_dir),
                "size_bytes": cache_size,
                "size_mb": round(cache_size / (1024 * 1024), 2),
                "file_count": cache_files,
            },
            "output": {
                "directory": str(output_dir),
                "file_count": output_files,
            },
            "config": {
                "data_sources": list(config.data_sources.keys()),
            },
        }
    except Exception as e:
        logger.exception("Failed to get status")
        raise HTTPException(status_code=500, detail=str(e))




