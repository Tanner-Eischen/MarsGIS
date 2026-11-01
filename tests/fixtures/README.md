# Test Fixtures

## dem_small.cog.tif

A small 256×256 GeoTIFF used for CI/testing the COG reader and overlay pipeline. The session fixture `dem_small_cog` in `tests/conftest.py` generates it in pytest's temporary directory automatically; no checked-in binary or download is required. It is synthetic test data, not evidence about Mars observations.

The suite isolates configuration, data, cache, and output paths for each test and blocks external Python network connections. Tests of downloader behavior supply local mock response objects. Tests of synthetic fallback explicitly set `MARSHAB_ALLOW_SYNTHETIC_TILES=true`; it remains disabled by default.

### Properties

- **Size:** 256×256 pixels
- **Dtype:** Int16
- **Nodata:** -32768 (MOLA convention)
- **Bounds:** lon 0–2°, lat 0–2° (Mars-like)
- **Features:** Nodata region in top-right quadrant

### How to regenerate

```bash
poetry run python scripts/create_dem_fixture.py
```

Or from project root:

```bash
python scripts/create_dem_fixture.py
```

The optional script writes to ignored `tests/fixtures/dem_small.cog.tif` for manual experiments. Automated tests use their temporary fixture instead. This legacy reader fixture uses a geographic transform and `CRS_INFO` tag; separate georeferencing tests require a real Mars CRS definition.
