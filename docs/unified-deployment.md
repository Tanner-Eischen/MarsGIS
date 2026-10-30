# Unified website and analysis service

**Live URL:** https://marshab.onrender.com. Deployed to the confirmed **My Workspace** workspace in Ohio on Render's free plan. The public HTTP workflow passed on 2026-10-28; see [verification evidence](live-verification.md).

MarsHab now runs as one FastAPI application serving both the built React website and `/api/v1`. The browser uses relative API URLs, so there is one origin and no external Render API proxy or production CORS dependency.

## Run locally

Prerequisites: Python 3.11 and the pinned environment from the README; Node 22.12 or newer.

```powershell
.venv/Scripts/python.exe scripts/prepare_planetary_data.py
cd webui
npm ci
npm run build
cd ..
.venv/Scripts/python.exe -m marshab.web.server
```

Visit `http://localhost:5000` for the whole-planet map, selected-point routing and3D path viewer. Data preparation provisions33 MB global MOLA plus9.7 MB detailed CTX in the configured cache. See [map/routing methods](map-routing.md). The candidate/suitability example remains at `/comparison`.

For frontend development, run `npm run dev` in `webui/` alongside the Python API. Vite proxies `/api` to localhost:5000. The legacy research workspace is available only in development under `/experimental/analysis`; it is not part of the validated public path. The container sets `MARSHAB_ENABLE_EXPERIMENTAL_API=false` so legacy global downloads and unvalidated scientific endpoints are not exposed.

## One container

```powershell
docker compose up --build
```

The multi-stage Dockerfile builds React, installs the pinned Python environment, and verifies/bakes both sources into `/app/reference`. It starts one Uvicorn worker. The generated site-comparison reference package is included in the React build. Docker Compose exposes port5000 and optionally persists generated output in a named volume. Baked inputs are separate from writable runtime output.

`render.yaml` defines **one Docker web service**, with no required paid disk. The service honors Render's `PORT`. The current service was created directly through Render's API from the tested `codex/portfolio-validation` branch; PR #28 remains open for review. The previous Vercel static site and separate Render API are not this unified runtime.

## API contract

| Method and path | Behavior |
| --- | --- |
| GET `/api/v1/explorer/metadata` | Actual global/detail coverage, source checksums, sampling and map tile templates |
| GET `/api/v1/explorer/tiles/{layer}/{z}/{x}/{y}.png` | Bounded real-data Mars geographic tiles |
| POST `/api/v1/explorer/routes` | Selected start/destination and slope limit;202 accepted,409 when scientific slot is busy |
| GET `/api/v1/explorer/routes/{id}` | Asynchronous state and computed geographic/metric path, profile and3D mesh |
| GET `/api/v1/explorer/routes/{id}/geojson` | The actual computed planetary path |
| GET `/api/v1/portfolio/status` | idle/running/ready/failed, data availability, pinned source and run IDs |
| POST `/api/v1/portfolio/run` | Starts actual shared Python calculations; 202 accepted, 409 if already running |
| GET `/api/v1/portfolio/scene` | Verified current scene, execution mode and immutable run-specific asset base |
| GET `/api/v1/portfolio/files/{run_id}/{path}` | Allowlisted, contained, checksum-checked scientific artifacts |
| GET `/api/v1/health/live` | Process liveness |

The default map supports planetary pan/zoom, arbitrary start/destination selection, bounded source-aware routing and a metric 3D path view with display-only exaggeration. The site-comparison page separately supports five terrain layers, candidates, probes and original scientific downloads. Reference runs are explicitly precomputed; fresh jobs are identified as server-computed. Unsupported solar/mission models remain outside both workflows.

A new result is published only after computation, export verification and package verification all succeed. Failures preserve the old valid result. The latest five successful live runs and the bundled reference remain available. Older live runs expire and their UUID output folders are removed; failed partial outputs are cleaned up. **Use one worker/process.** A restart restores the bundled reference; automatic recovery of previous run IDs is not implemented. Runtime files on ephemeral hosting may be lost at restart.

Missing API endpoints return JSON404. Missing asset URLs return404 rather than index.html. Only safe extensionless frontend routes receive the SPA fallback.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check marshab tests scripts
cd webui
npm test
npm run lint
npm run build
npm audit --audit-level=moderate
cd ..
.venv/Scripts/python.exe scripts/smoke_unified_app.py
```

The final command talks to a running local service using ordinary HTTP requests. It loads React assets, starts an actual DEM calculation, waits for completion, checks every artifact checksum and tests API/asset404 behavior. It uses no browser. CI builds the Docker image and runs the same check in a container with a 512 MB memory limit.

Verification evidence: Linux CI passed 371 tests; Windows passed 370 with one existing skip; 75 frontend component/data tests passed. Frontend build/lint passed; npm audit reported zero known vulnerabilities. The Docker image and real HTTP workflow passed in CI under a 512 MiB limit. The public Render deployment also completed a fresh analysis and passed every artifact checksum. Browser visual/interaction testing was not performed because repository instructions restrict browser access.
