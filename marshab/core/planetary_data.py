"""Pinned planetary elevation sources and bounded geographic overview tiles.

MOLA native cells are 1/16 degree, with median measured topography and
interpolation in unsampled bins. Zero is a valid elevation, not NoData.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from marshab.analysis.portfolio import SOURCE_CITATION, SOURCE_SHA256, SOURCE_URL
from marshab.config import Config

GLOBAL_URL = (
    "https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/"
    "meg016/megt90n000eb.img"
)
GLOBAL_LABEL_URL = GLOBAL_URL.removesuffix(".img") + ".xml"
GLOBAL_SHA256 = "d18d9b9ab8c5516d02e157dd2cde0f1d0d160c21940e953ba22391269a545e7b"
GLOBAL_SHAPE = (2880, 5760)
GLOBAL_RADIUS_M = 3396000.0
GLOBAL_RESOLUTION_M = 2 * math.pi * GLOBAL_RADIUS_M / 5760
ELEVATION_RANGE = (-8177.0, 21171.0)
PALETTE = ((38, 30, 29), (104, 58, 45), (177, 98, 64), (222, 166, 110), (251, 239, 210))
TILE_SIZE = 256
MAX_ZOOM = 12
DISPLAY_Z_FACTOR = 3.0
DISPLAY_AZIMUTH_DEG = 315.0
DISPLAY_ALTITUDE_DEG = 45.0
DISPLAY_SHADE_RANGE = (0.35, 1.3)
MAX_CACHED_TILES = 128
MAX_CACHE_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file(path: Path, expected: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Planetary DEM is missing: {path}; run scripts/prepare_planetary_data.py"
        )
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Planetary DEM byte length mismatch: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"Planetary DEM checksum mismatch: {path}")


def ensure_global_dem(path: Path) -> Path:
    """Provision once through a streamed, checksum-pinned atomic download."""
    expected_bytes = math.prod(GLOBAL_SHAPE) * 2
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{uuid4().hex}.partial")
        try:
            request = urllib.request.Request(GLOBAL_URL, headers={"User-Agent": "MarsHab/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open(
                "wb"
            ) as output:
                digest = hashlib.sha256()
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > expected_bytes:
                        raise ValueError("Downloaded MOLA source exceeds its pinned byte length")
                    digest.update(chunk)
                    output.write(chunk)
            if count != expected_bytes or digest.hexdigest() != GLOBAL_SHA256:
                raise ValueError("Downloaded MOLA source differs from pinned bytes/checksum")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    _validate_file(path, GLOBAL_SHA256, expected_bytes)
    return path


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Geographic two-by-one root tiling, west/south/east/north degrees."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (z, x, y)):
        raise ValueError("Tile indices must be integers")
    if not 0 <= z <= MAX_ZOOM or not 0 <= x < 2 ** (z + 1) or not 0 <= y < 2**z:
        raise ValueError("Tile indices are outside the geographic tiling scheme")
    span = 180.0 / (2**z)
    west, north = -180.0 + x * span, 90.0 - y * span
    return west, north - span, west + span, north


def _png(values: np.ndarray, shade: np.ndarray | None = None) -> bytes:
    """Use a continuous fixed elevation palette, preserving invalid transparency."""
    valid = np.isfinite(values)
    low, high = ELEVATION_RANGE
    normalized = np.where(valid, np.clip((values.astype(float) - low) / (high - low), 0, 1), 0)
    colors = np.asarray(PALETTE)
    knots = np.linspace(0, 1, len(colors))
    channels = [np.interp(normalized, knots, colors[:, index]) for index in range(3)]
    rgb = np.stack(channels, axis=-1)
    if shade is not None:
        valid &= np.isfinite(shade)
        rgb = np.clip(rgb * np.where(np.isfinite(shade), shade, 0)[..., None], 0, 255)
    rgba = np.concatenate((rgb, (valid * 255)[..., None]), axis=-1)
    stream = io.BytesIO()
    Image.fromarray(np.rint(rgba).astype(np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


class PlanetaryDataStore:
    """Read-only source handles and a small bounded PNG cache; no request downloads."""

    def __init__(self, global_path: Path, ctx_path: Path):
        global_path, ctx_path = Path(global_path), Path(ctx_path)
        _validate_file(global_path, GLOBAL_SHA256, math.prod(GLOBAL_SHAPE) * 2)
        _validate_file(ctx_path, SOURCE_SHA256)
        self.global_array = np.memmap(global_path, dtype=">i2", mode="r", shape=GLOBAL_SHAPE)
        self.ctx_path = ctx_path
        self.global_provenance = {
            "id": "mola16",
            "label": "MOLA global topography · 16 pixels/degree",
            "url": GLOBAL_URL,
            "label_url": GLOBAL_LABEL_URL,
            "sha256": GLOBAL_SHA256,
            "bytes": global_path.stat().st_size,
            "radius_m": GLOBAL_RADIUS_M,
            "native_resolution_m": GLOBAL_RESOLUTION_M,
            "resolution_deg": 1 / 16,
            "shape": list(GLOBAL_SHAPE),
            "coordinates": "Planetocentric latitude; east-positive longitude 0–360 degrees",
            "vertical_reference": "Goddard Mars potential model GMM3 (mgm1025), degree/order 50 areoid",
            "elevation_units": "m",
            "nodata": None,
            "sampling": "Median observed topography; interpolated in bins without observations",
            "tile_resampling": "nearest native cell at each geographic tile pixel center",
            "tile_style": "colorized elevation with native-grid Lambertian hillshade",
            "display_relief": {
                "visual_only": True,
                "z_factor": DISPLAY_Z_FACTOR,
                "azimuth_deg": DISPLAY_AZIMUTH_DEG,
                "altitude_deg": DISPLAY_ALTITUDE_DEG,
                "intensity_range": list(DISPLAY_SHADE_RANGE),
                "flat_intensity": 1.0,
                "gradient_spacing": "native Mars-sphere latitude spacing and cos(latitude) longitude spacing",
                "edge_handling": "wrapped longitude; one-sided differences at first/last latitude rows",
            },
        }
        with rasterio.open(ctx_path) as src:
            if src.crs is None or src.count != 1:
                raise ValueError("CTX source must have one elevation band and an explicit Mars CRS")
            crs = CRS.from_user_input(src.crs)
            if not math.isclose(crs.ellipsoid.semi_major_metre, 3396190, rel_tol=0, abs_tol=1):
                raise ValueError("CTX source must retain its documented Mars sphere")
            self._ctx_geographic_crs = crs.geodetic_crs
            self.ctx_bounds = tuple(
                transform_bounds(src.crs, self._ctx_geographic_crs, *src.bounds, densify_pts=21)
            )
            self.ctx_provenance = {
                "id": "ctx",
                "label": "Jezero CTX stereo DEM · 20 m",
                "url": SOURCE_URL,
                "citation": SOURCE_CITATION,
                "sha256": SOURCE_SHA256,
                "bytes": ctx_path.stat().st_size,
                "radius_m": 3396190.0,
                "native_resolution_m": 20.0,
                "shape": [src.height, src.width],
                "bounds": list(self.ctx_bounds),
                "crs_wkt": src.crs.to_wkt(),
                "source_nodata": src.nodata,
                "elevation_units": "m above product's MOLA/DeltaGeoid reference",
                "tile_resampling": "bilinear display resampling; no change to native information content",
            }
        self._cache: OrderedDict[tuple[str, int, int, int], bytes] = OrderedDict()
        self._cache_bytes = 0
        self._cache_lock = threading.Lock()
        self._render_slots = threading.BoundedSemaphore(2)

    @classmethod
    def from_config(cls, config: Config) -> PlanetaryDataStore:
        return cls(
            Path(os.getenv("MARSHAB_GLOBAL_DEM_PATH") or config.paths.cache_dir / "mola16.img"),
            Path(
                os.getenv("MARSHAB_REFERENCE_DEM_PATH")
                or config.paths.cache_dir / "jezero_ctx20m.tif"
            ),
        )

    def metadata(self) -> dict:
        west, south, east, north = self.ctx_bounds
        return {
            "global": copy.deepcopy(self.global_provenance),
            "detail": copy.deepcopy(self.ctx_provenance),
            "tiles": {
                "mola": "/api/v1/explorer/tiles/mola/{z}/{x}/{y}.png",
                "ctx": "/api/v1/explorer/tiles/ctx/{z}/{x}/{y}.png",
            },
            "initial_view": {"lat": 0.0, "lon": 0.0, "zoom": 1},
            "detail_view": {"lat": (south + north) / 2, "lon": (west + east) / 2, "zoom": 9},
            "max_zoom": MAX_ZOOM,
            "tile_size": TILE_SIZE,
            "tiling": "geographic; 2x1 tiles at zoom 0, north-up, longitude -180..180",
            "elevation_palette": {
                "min": ELEVATION_RANGE[0],
                "max": ELEVATION_RANGE[1],
                "unit": "m",
                "colors": [list(color) for color in PALETTE],
            },
        }

    def tile(self, layer: str, z: int, x: int, y: int) -> bytes:
        bounds = tile_bounds(z, x, y)
        if layer not in {"mola", "ctx"}:
            raise ValueError("Unknown planetary elevation layer")
        key = (layer, z, x, y)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        with self._render_slots:
            # Another queued request may have rendered this tile while we waited.
            with self._cache_lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
            result = self._render_tile(layer, bounds)
            with self._cache_lock:
                previous = self._cache.pop(key, None)
                if previous:
                    self._cache_bytes -= len(previous)
                self._cache[key] = result
                self._cache_bytes += len(result)
                while len(self._cache) > MAX_CACHED_TILES or self._cache_bytes > MAX_CACHE_BYTES:
                    _, evicted = self._cache.popitem(last=False)
                    self._cache_bytes -= len(evicted)
            return result

    def _render_tile(self, layer: str, bounds: tuple[float, float, float, float]) -> bytes:
        """Render at most two tiles concurrently to bound temporary array memory."""
        if layer == "mola":
            west, south, east, north = bounds
            longitude = west + (np.arange(TILE_SIZE) + 0.5) * (east - west) / TILE_SIZE
            latitude = north - (np.arange(TILE_SIZE) + 0.5) * (north - south) / TILE_SIZE
            rows = np.clip(
                np.floor((90 - latitude) * self.global_array.shape[0] / 180).astype(int),
                0,
                self.global_array.shape[0] - 1,
            )
            cols = np.floor(np.mod(longitude, 360) * self.global_array.shape[1] / 360).astype(int)
            values = self.global_array[np.ix_(rows, cols)]
            shade = self._native_shade(rows, cols)
        else:
            values = self._ctx_tile(bounds)
            shade = None
        return _png(values, shade)

    def _native_shade(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Display-only relief from immediate native neighbors, never display-grid slopes.

        The normal uses east/south gradients in physical meters. Longitude wraps;
        polar edge rows use one-sided differences. No scientific source values,
        route costs or analysis derivatives are modified by display exaggeration.
        """
        height, width = self.global_array.shape
        north, south = np.maximum(rows - 1, 0), np.minimum(rows + 1, height - 1)
        east, west = (cols + 1) % width, (cols - 1) % width
        ze = np.asarray(self.global_array[np.ix_(rows, east)], dtype=float)
        zw = np.asarray(self.global_array[np.ix_(rows, west)], dtype=float)
        zn = np.asarray(self.global_array[np.ix_(north, cols)], dtype=float)
        zs = np.asarray(self.global_array[np.ix_(south, cols)], dtype=float)
        latitude = np.pi / 2 - (rows + 0.5) * np.pi / height
        dx = GLOBAL_RADIUS_M * (2 * np.pi / width) * np.cos(latitude)
        dy = GLOBAL_RADIUS_M * (np.pi / height) * (south - north)
        p = DISPLAY_Z_FACTOR * (ze - zw) / (2 * dx[:, None])
        q = DISPLAY_Z_FACTOR * (zs - zn) / dy[:, None]
        azimuth, altitude = np.radians([DISPLAY_AZIMUTH_DEG, DISPLAY_ALTITUDE_DEG])
        sun_east = np.sin(azimuth) * np.cos(altitude)
        sun_north = np.cos(azimuth) * np.cos(altitude)
        sun_up = np.sin(altitude)
        normal_dot_sun = (-p * sun_east + q * sun_north + sun_up) / np.sqrt(1 + p * p + q * q)
        return np.clip(normal_dot_sun / sun_up, *DISPLAY_SHADE_RANGE)

    def _ctx_tile(self, bounds: tuple[float, float, float, float]) -> np.ndarray:
        values = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
        west, south, east, north = bounds
        left, bottom, right, top = self.ctx_bounds
        if west >= right or east <= left or south >= top or north <= bottom:
            return values
        with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024), rasterio.open(self.ctx_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=values,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=from_bounds(*bounds, TILE_SIZE, TILE_SIZE),
                dst_crs=self._ctx_geographic_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                num_threads=1,
                warp_mem_limit=16,
            )
        return values

    def cache_info(self) -> dict:
        with self._cache_lock:
            return {
                "entries": len(self._cache),
                "bytes": self._cache_bytes,
                "max_entries": MAX_CACHED_TILES,
                "max_bytes": MAX_CACHE_BYTES,
            }


def from_config(config: Config) -> PlanetaryDataStore:
    """Construct from provisioned paths; missing files fail without network fallback."""
    return PlanetaryDataStore.from_config(config)
