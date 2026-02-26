"""Bootstrap MOLA COG from one-time URL during startup.

Ensures MARSHAB_MOLA_COG_PATH exists. If missing, streams download from
MARSHAB_MOLA_COG_URL to *.partial, verifies SHA256 (optional), then atomically
renames. Designed for Render persistent disk deployments.

URL resolution order:
1. MARSHAB_MOLA_COG_URL env var
2. config.data_sources.mola.url from marshab_config.yaml
3. USGS public URL (fallback when primary fails, e.g. private GitHub 404)
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from marshab.utils.logging import get_logger

logger = get_logger(__name__)

# Public USGS fallback when env and config both fail
USGS_MOLA_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"

# Minimum expected size (bytes) for MOLA COG - guards HTML error pages
# MOLA 463m global is ~900MB; allow smaller but reject obviously wrong
MIN_EXPECTED_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def _compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(url: str, dest_path: Path) -> int:
    """Stream download from URL to dest_path. Returns bytes written."""
    total = 0
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "MarsGIS/1.0")
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(dest_path, "wb") as out:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
            try:
                out.flush()
                os.fsync(out.fileno())
            except (OSError, AttributeError):
                pass
    return total


def ensure_mola_cog_present() -> None:
    """Ensure MOLA COG exists locally; download if missing.

    - If MARSHAB_MOLA_COG_PATH exists: verify SHA256 if configured, return.
    - If missing and MARSHAB_MOLA_COG_URL set: stream download to *.partial,
      verify SHA256, atomic rename.
    - If missing and URL not set: raise with clear error.

    Raises:
        RuntimeError: If COG missing and URL not configured, or download fails.
    """
    cog_path_str = os.getenv("MARSHAB_MOLA_COG_PATH")
    if not cog_path_str:
        logger.debug("MARSHAB_MOLA_COG_PATH not set; skipping COG bootstrap")
        return

    cog_path = Path(cog_path_str)
    url = (os.getenv("MARSHAB_MOLA_COG_URL") or "").strip()
    if not url:
        try:
            from marshab.config import get_config
            cfg = get_config()
            if "mola" in cfg.data_sources:
                url = cfg.data_sources["mola"].url
                logger.info("Using MOLA URL from config", url=url)
        except Exception as e:
            logger.debug("Could not load config for MOLA URL fallback", error=str(e))
    expected_sha256 = (os.getenv("MARSHAB_MOLA_COG_SHA256") or "").strip().lower()

    if cog_path.exists():
        if expected_sha256:
            actual = _compute_sha256(cog_path)
            if actual != expected_sha256:
                logger.warning(
                    "MOLA COG SHA256 mismatch; removing and re-downloading",
                    expected=expected_sha256[:16] + "...",
                    actual=actual[:16] + "...",
                )
                cog_path.unlink()
            else:
                logger.info("MOLA COG present and verified", path=str(cog_path))
                return
        else:
            logger.info("MOLA COG present", path=str(cog_path))
            return

    if not url:
        raise RuntimeError(
            f"MOLA COG not found at {cog_path} and MARSHAB_MOLA_COG_URL is not set. "
            "Set MARSHAB_MOLA_COG_URL to a one-time download URL (e.g. GitHub Release asset)."
        )

    cog_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = cog_path.with_suffix(cog_path.suffix + ".partial")

    try:
        if partial_path.exists():
            partial_path.unlink()
    except OSError as e:
        logger.warning("Could not remove stale partial file", path=str(partial_path), error=str(e))

    urls_to_try = [url]
    if url != USGS_MOLA_URL:
        try:
            from marshab.config import get_config
            cfg = get_config()
            if "mola" in cfg.data_sources and cfg.data_sources["mola"].url not in urls_to_try:
                urls_to_try.append(cfg.data_sources["mola"].url)
        except Exception:
            pass
        if USGS_MOLA_URL not in urls_to_try:
            urls_to_try.append(USGS_MOLA_URL)

    last_error = None
    for attempt_url in urls_to_try:
        logger.info("Downloading MOLA COG", url=attempt_url, dest=str(cog_path))
        try:
            if partial_path.exists():
                partial_path.unlink()
            total = _stream_download(attempt_url, partial_path)
            break
        except Exception as e:
            last_error = e
            try:
                partial_path.unlink(missing_ok=True)
            except OSError:
                pass
            logger.warning("Download failed, trying fallback", url=attempt_url, error=str(e))
    else:
        raise RuntimeError(f"MOLA COG download failed: {last_error}") from last_error

    if total < MIN_EXPECTED_SIZE_BYTES:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"Downloaded file too small ({total} bytes) - likely HTML error page. "
            f"Expected at least {MIN_EXPECTED_SIZE_BYTES} bytes."
        )

    if expected_sha256:
        actual = _compute_sha256(partial_path)
        if actual != expected_sha256:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f"MOLA COG SHA256 mismatch after download. Expected {expected_sha256[:16]}..., got {actual[:16]}..."
            )
        logger.info("Downloaded file verified via SHA256")

    if cog_path.exists():
        cog_path.unlink()
    partial_path.rename(cog_path)
    logger.info("MOLA COG ready", path=str(cog_path), size_bytes=total)


def prewarm_cog() -> None:
    """Read a small window from COG to warm file cache. Non-fatal on failure."""
    cog_path_str = os.getenv("MARSHAB_MOLA_COG_PATH")
    if not cog_path_str:
        return
    cog_path = Path(cog_path_str)
    if not cog_path.exists():
        return

    try:
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(cog_path) as src:
            # Read 32x32 window from center
            h, w = src.height, src.width
            row_off = max(0, (h // 2) - 16)
            col_off = max(0, (w // 2) - 16)
            win = Window(col_off, row_off, 32, 32)
            _ = src.read(1, window=win)
        logger.debug("COG prewarm complete")
    except Exception as e:
        logger.warning("COG prewarm failed (non-fatal)", error=str(e))
        raise
