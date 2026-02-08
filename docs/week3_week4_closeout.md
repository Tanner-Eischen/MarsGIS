# Week 3 / Week 4 Closeout

## Week 3 (Release Hardening)

- Frontend API/WS base URL abstraction for deployment-safe backend connectivity.
- Deterministic demo artifact generation via `scripts/run_portfolio_demo.py`.
- CI smoke gates for backend smoke tests, critical Ruff checks, and frontend build.

## Week 4 (Portfolio Productionization)

- Added deterministic decision brief API endpoint:
  - `POST /api/v1/analysis/decision-brief`
- Persisted flow artifacts for cross-endpoint compatibility:
  - Site analysis now writes `sites.csv` to output dir.
  - Navigation planning now writes `waypoints_site_<id>.csv` and `waypoints.csv`.
- Added portfolio flow acceptance tests:
  - `tests/api/test_portfolio_flows.py`
- Added deployment blueprints:
  - `webui/vercel.json`
  - `render.yaml`
- Added configurable CORS for production:
  - `MARSHAB_CORS_ORIGINS` (comma-separated)
  - `MARSHAB_CORS_ALLOW_ALL=true` (no credentials)
- Added real-data-only tile/3D readiness operations:
  - API prewarm endpoint usage: `POST /api/v1/prewarm/mola-tiles`
  - One-command smoke check: `scripts/smoke_tile_3d_readiness.py`
  - 3D payload now reports `used_synthetic` for strict validation in runbooks.

## Validation Commands

```bash
poetry run pytest -q tests/test_pipeline_smoke.py tests/test_route_smoke.py tests/api/test_portfolio_flows.py
poetry run ruff check marshab tests scripts --select E9,F63,F7,F82
poetry run python scripts/smoke_tile_3d_readiness.py
cd webui && npm run build
```

## Portfolio Artifact Command

```bash
poetry run python scripts/run_portfolio_demo.py
```
