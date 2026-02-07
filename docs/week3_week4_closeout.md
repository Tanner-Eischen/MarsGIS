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

## Validation Commands

```bash
poetry run pytest -q tests/test_pipeline_smoke.py tests/test_route_smoke.py tests/api/test_portfolio_flows.py
poetry run ruff check marshab tests scripts --select E9,F63,F7,F82
cd webui && npm run build
```

## Portfolio Artifact Command

```bash
poetry run python scripts/run_portfolio_demo.py
```
