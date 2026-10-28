# MarsHab / MarsGIS

MarsHab is a scientific software portfolio project for comparing Mars terrain and planning terrain-constrained routes. The validated example turns a measured NASA MRO CTX digital elevation model into terrain derivatives, a transparent suitability score, three candidate comparisons, and exported routes.

The emphasis is reproducible geospatial computation: explicit units and coordinate systems, invalid-data masks, analytical tests, independent shortest-path references, and inspectable outputs. Candidate scores illustrate terrain tradeoffs; they do not establish habitat safety.

![Jezero DEM, slope and terrain-constrained routes](docs/assets/jezero-comparison.png)

## Reproduce the example

Requirements: Python **3.11**, [uv](https://docs.astral.sh/uv/), internet for the first 9.7 MB DEM download. No credentials, paid APIs, or manual dataset acquisition. Subsequent runs use the checksum-verified local source.

From the repository root, on Windows PowerShell:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv/Scripts/python.exe requirements-dev.lock
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.venv/Scripts/python.exe scripts/run_portfolio_demo.py
.venv/Scripts/python.exe scripts/verify_portfolio_demo.py
```

On Linux/macOS, substitute `.venv/bin/python` for `.venv/Scripts/python.exe`. Linux CI is configured; this revision was verified locally on Windows. `requirements.lock` pins runtime dependencies; `requirements-dev.lock` adds the pinned test tools. `poetry.lock` and the Conda/Docker configuration are legacy environments and were not used for the results below.

The script downloads the source to `data/source/jezero_ctx_20m.tif`, refuses checksum mismatches, and writes to `data/output/portfolio_demo/`:

| Artifact | Contents |
| --- | --- |
| `suitability.tif` | Float32 score, native analysis-grid Mars CRS, transform and NoData |
| `terrain.tif` | Elevation, slope, aspect, roughness, TRI, hillshade and route-cost bands with units |
| `routes.geojson` | Three complete unsmoothed routes, candidate coordinates and cost comparisons |
| `report.md` | Candidate metrics, assumptions, limitations and figure |
| `comparison.png` | DEM, slope, suitability and route figure |
| `manifest.json` | Source URL/SHA-256, ROI, source/analysis CRS, resolution, parameters, software/source versions and output checksums |

Two local runs, including a fresh environment and a fresh source download, produced identical artifact checksums. To repeat that check, run with `--output-dir data/output/repeat`, then run `scripts/verify_portfolio_demo.py --compare data/output/repeat`.

**Planetary coordinates:** GeoJSON stores Mars east longitude and planetocentric latitude in degrees, declared in its metadata. It is not terrestrial RFC 7946/WGS84 GeoJSON. Use a planetary-aware tool; the GeoTIFFs contain a machine-readable Mars CRS.

## What the example demonstrates

1. Reproject the documented CTX 20 m DEM to a local spherical azimuthal equidistant grid at 20 m spacing.
2. Calculate slope, downslope aspect, 5x5 elevation standard deviation, mean absolute neighbor-difference TRI, and hillshade.
3. Score slope and roughness with weights 0.70/0.30, both lower-is-better, over jointly valid cells.
4. Compare the three highest-scoring candidate cells in the largest traversable component, separated by at least 1.5 km and excluded from the raster boundary.
5. Route from a common proposed start using eight-connected A*, exact diagonal distances and no corner cutting. Compare the optimized terrain cost with a distance-only route over the same passable cells.
6. Reopen and independently check exports, masks, coordinates, route connectivity and accumulated costs.

The source is [USGS's Jezero CTX stereo DEM](https://www.usgs.gov/media/images/m20jezerocraterctxdemmosaic20m100jpg), derived from NASA MRO observations and controlled to MOLA. Read [scientific methods and numerical tolerances](docs/scientific-methods.md), the [feature-verification matrix](docs/feature-verification.md), and the [interview walkthrough](docs/portfolio-walkthrough.md).

## Validation

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check marshab tests scripts
cd webui
npm ci
npm run lint
npm run build
```

Local result: **317 tests passed, one Windows-specific legacy cache test skipped; 54% overall coverage**. The full Python suite runs offline with temporary storage and generated fixtures. Numerical tests include analytical terrain surfaces, hand-calculated suitability scores, independent SciPy Dijkstra comparisons, Mars coordinate round trips, metadata-preserving exports, and repeated-run configuration isolation. Passing mocked API tests establish contracts, not real-data scientific correctness.

CI runs the complete offline suite on Windows and Linux, frontend lint/build, and a separate real-DEM demonstration with artifact verification/upload. Remote results are recorded in the PR checks and [GitHub Actions](https://github.com/Tanner-Eischen/MarsGIS/actions/workflows/ci.yml).

## Architecture and scope

- `marshab/analysis/portfolio.py`: strict real-data example orchestration.
- `marshab/processing/`: shared terrain, DEM ingestion, scoring and pathfinding kernels.
- `marshab/analysis/export.py`: metadata-preserving suitability export.
- `marshab/core/` and `marshab/web/`: legacy analysis orchestration, caching, FastAPI and navigation services.
- `webui/`: React, Leaflet and Plotly interface; local lint/build verified, interactive workflows unverified.
- `tests/`: scientific assertions, API contracts and integration checks.

The example intentionally uses only measured terrain criteria. Solar exposure and energy estimates, dust/science-value proxies, mission economics, ML recommendations, legacy cache/projection branches and complete UI workflows are **outside the validated portfolio scope**. Their remaining gaps are recorded in the matrix.

## API, frontend and deployment

The existing API can be started with `.venv/Scripts/python.exe -m marshab.web.server`; the frontend uses `npm run dev` from `webui/`. API schema is available locally at `/docs`. The flagship CLI does not require either service and does not populate the legacy API cache.

The previously advertised Render frontend and API health URLs returned **HTTP 404 on 2026-10-27**. There is no verified live deployment. `render.yaml` and Docker configuration remain for future deployment review; a liveness response alone would not establish DEM readiness.

Frontend installation reports dependency vulnerabilities; see the audit. Deployment hardening and dependency migration are separate follow-up work, and no production readiness is claimed.
