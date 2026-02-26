# Test Fixtures

## dem_small.cog.tif

A small 256×256 GeoTIFF used for CI/testing the COG reader and overlay pipeline. Avoids using the ~900MB real MOLA COG in tests.

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

The script writes to `tests/fixtures/dem_small.cog.tif`.
