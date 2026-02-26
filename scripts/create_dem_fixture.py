#!/usr/bin/env python3
"""Create tests/fixtures/dem_small.cog.tif for CI testing.

Generates a 256x256 GeoTIFF with:
- Int16 dtype, nodata=-32768 (MOLA convention)
- Mars-like bounds (0-2 deg lon, 0-2 deg lat)
- Nodata region in one corner
- Compatible with cog_reader windowed reads
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUTPUT_PATH = FIXTURE_DIR / "dem_small.cog.tif"

# Mars-like bounds (lon_min, lat_min, lon_max, lat_max)
BOUNDS = (0.0, 0.0, 2.0, 2.0)
WIDTH, HEIGHT = 256, 256
NODATA = -32768


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Elevation: gentle slope + nodata region in top-right
    np.random.seed(42)
    elevation = np.random.randint(-2000, 3000, (HEIGHT, WIDTH), dtype=np.int16)
    elevation = elevation.astype(np.int16)
    # Add nodata region (top-right quadrant)
    elevation[0 : HEIGHT // 2, WIDTH // 2 :] = NODATA

    transform = from_bounds(*BOUNDS, WIDTH, HEIGHT)

    with rasterio.open(
        OUTPUT_PATH,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype=np.int16,
        crs=None,
        transform=transform,
        nodata=NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=64,
        blockysize=64,
    ) as dst:
        dst.write(elevation, 1)
        dst.update_tags(CRS_INFO="EPSG:49900")

    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
