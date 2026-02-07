"""Download and prepare a real Mars DEM tile for local portfolio demos.

This script installs a real Jezero-area DEM into the cache key expected by
`DataManager` for ROI tiles around lat 18 / lon 77:

    data/cache/mola_lat18_lon77.tif

It also normalizes metadata so the current DEM loader can clip by lat/lon
without relying on an external PROJ database at runtime.
"""

from __future__ import annotations

import argparse
import math
import urllib.request
from pathlib import Path

import rasterio
from rasterio.transform import Affine

DEFAULT_URL = (
    "https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/"
    "JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif"
)
DEFAULT_OUT = Path("data/cache/mola_lat18_lon77.tif")
MARS_RADIUS_M = 3396190.0
DEG_PER_M = 180.0 / (math.pi * MARS_RADIUS_M)


def _looks_geographic(src: rasterio.DatasetReader) -> bool:
    """Return True if dataset transform/bounds already look lon/lat in degrees."""
    b = src.bounds
    t = src.transform
    return (
        abs(t.a) < 1.0
        and abs(t.e) < 1.0
        and -90.0 <= b.bottom <= 90.0
        and -90.0 <= b.top <= 90.0
        and 0.0 <= b.left <= 360.0
        and 0.0 <= b.right <= 360.0
    )


def _normalize_to_geo_degrees(src_path: Path, out_path: Path, source_url: str) -> None:
    """Write a lon/lat-degree normalized copy to `out_path`."""
    tmp_out = out_path.with_name(out_path.stem + ".normalized.tmp.tif")

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        if not _looks_geographic(src):
            transform = Affine(
                transform.a * DEG_PER_M,
                transform.b * DEG_PER_M,
                transform.c * DEG_PER_M,
                transform.d * DEG_PER_M,
                transform.e * DEG_PER_M,
                transform.f * DEG_PER_M,
            )

        profile.update(
            {
                "transform": transform,
                "crs": None,
                "compress": "lzw",
            }
        )

        with rasterio.open(tmp_out, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst.write(src.read(band_idx), band_idx)
            merged_tags = dict(src.tags())
            merged_tags["CRS_INFO"] = "EPSG:49900"
            merged_tags["SOURCE_URL"] = source_url
            dst.update_tags(**merged_tags)

    if out_path.exists():
        out_path.unlink()
    tmp_out.replace(out_path)


def _print_summary(path: Path) -> None:
    with rasterio.open(path) as src:
        print(f"Prepared DEM: {path}")
        print(f"Shape: {src.height} x {src.width}")
        print(f"Bounds: {src.bounds}")
        print(f"Pixel size: ({src.transform.a}, {src.transform.e})")
        print(f"CRS: {src.crs}")
        print(f"Tags: {src.tags()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install real Jezero DEM into local cache tile path."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Source DEM URL")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output cache path (default: data/cache/mola_lat18_lon77.tif)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Always re-download even if output file already exists",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    download_tmp = args.out.with_name(args.out.stem + ".download.tmp.tif")
    source_path = args.out

    if args.force_download or not args.out.exists():
        print(f"Downloading DEM from: {args.url}")
        urllib.request.urlretrieve(args.url, download_tmp)
        source_path = download_tmp
    else:
        print(f"Using existing DEM file: {args.out}")

    try:
        _normalize_to_geo_degrees(source_path, args.out, args.url)
    finally:
        if download_tmp.exists():
            download_tmp.unlink()

    _print_summary(args.out)


if __name__ == "__main__":
    main()
