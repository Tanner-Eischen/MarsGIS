# Public deployment verification

Verified on **2026-10-28** using HTTP/API requests, without a browser or platform UI.

**Website:** [marshab.onrender.com](https://marshab.onrender.com)

## Deployed application

- Render service: `marshab`, free compute plan, Ohio, one Docker web process.
- Workspace: the user-confirmed **My Workspace**.
- Source: `Tanner-Eischen/MarsGIS`, branch `codex/portfolio-validation`.
- Scientific code commit: `af82bf8e5ced9b7eebc5246850c65dda2e1b0139`.
- Render deploy: `dep-dbgsf2rbc2fs73bhvhhg`, reported `live`.
- The service serves React, the scientific API and verified artifacts from one origin.
- Render uses its default TCP health probe; the public verification below separately checks HTTP liveness and the actual analysis workflow.

## Public workflow evidence

Executed:

```sh
python scripts/smoke_unified_app.py --base-url https://marshab.onrender.com --timeout 900
```

Result:

```text
PASS: same-origin React assets, real API computation, five terrain layers and all download checksums
Run: f38d9007831b4cfc854a8f0befe81c70; grid: [593, 563]; candidates: 3
```

The public API completed a fresh calculation and returned `data_mode=observed_dem` and `execution_mode=server_computed_analysis`. Its source SHA-256 was:

```text
16a1aa66c3f48c1bca252795827db7b417876217aff0f5e1aff18027fab2327b
```

This is the documented NASA MRO CTX Jezero DEM, reprojected to a 20 m analysis grid. The verification fetched and checked all 12 hashed artifact files, including the terrain/suitability GeoTIFFs, routes, report, manifest, figure, full-resolution layers and probe buffer. It also checked the React entry/assets, frontend deep-link response, JSON404 for an unknown API and404 for a missing static asset.

The public OpenAPI schema exposes only health and `/api/v1/portfolio/*` routes. Experimental global downloads and unvalidated legacy analysis endpoints are not exposed.

## Prior automated evidence

[CI run 40664863723](https://github.com/Tanner-Eischen/MarsGIS/actions/runs/40664863723) passed Linux tests (371), Windows tests (370 with one documented skip), frontend tests (75), lint/build/dependency audit, real-DEM repeatability and the Docker/HTTP workflow. The container was limited to512 MiB and its measured peak was approximately308 MiB.

## Limits and interview use

The free service can sleep after15 minutes without traffic and take about a minute to wake, according to [Render's free-service documentation](https://render.com/docs/free). Open the website before an interview. The verified reference is available after startup; fresh recomputation on this small free CPU can take a few minutes, while the previous results remain usable.

Live run IDs are temporary: only the latest five successful runs are retained, and a restart restores the bundled reference. Use the website root as the portfolio link, rather than an individual run's artifact URL.

Browser rendering and real-browser interaction were not inspected under the repository restriction. This record establishes public HTTP availability, actual server computation and artifact integrity; it does not claim browser visual QA, validated solar/mission predictions, or certified habitat safety.
