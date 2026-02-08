# MarsGIS Portfolio Release Checklist

## Scope Lock
- [x] Exactly three flagship flows remain in release scope.
- [x] Non-goals are documented and approved.

## Flow Acceptance
1. Site selection
- [x] ROI + weights request succeeds.
- [x] Ranked sites are returned.
- [x] GeoJSON overlay endpoint returns valid feature collection.
- [x] Empty-state behavior is clear when no candidates meet threshold.

2. Route planning
- [x] Route planning request succeeds for top-ranked site.
- [x] Waypoints output contains more than two points.
- [x] Waypoint export format is documented and downloadable.
- [x] Failure state is explicit for invalid site/start inputs.

3. Decision brief
- [x] Summary renders without external paid LLM services.
- [x] Explanation includes key factors (terrain, score, route/energy impacts).
- [x] Output fields are deterministic for same inputs and seed.

## Determinism & Demo
- [x] `MARSHAB_DEMO_SEED=42` is supported for synthetic mode.
- [x] Pipeline smoke test passes with deterministic synthetic data.
- [x] Route smoke test passes from top site selection to waypoints.
- [x] Demo script uses fixed ROI, presets, and expected outputs.

## Real-Data-Only Prewarm / Runbook
- [x] Real-data-only mode is documented (`MARSHAB_ALLOW_SYNTHETIC_TILES=false`).
- [x] MOLA tile prewarm endpoint is documented for target ROI.
- [x] One-command tile + 3D readiness smoke check is available.

### Runbook (Local API)
1. Prepare real Jezero DEM cache:
```bash
poetry run python scripts/setup_real_dem.py --force-download
```

2. Start API in real-data-only tile mode:
```bash
# bash/zsh
export MARSHAB_ALLOW_SYNTHETIC_TILES=false
poetry run python -m marshab.web.server
```

```powershell
# PowerShell
$env:MARSHAB_ALLOW_SYNTHETIC_TILES = "false"
poetry run python -m marshab.web.server
```

3. Optional direct prewarm call (same ROI used by smoke):
```bash
curl -X POST http://localhost:5000/api/v1/prewarm/mola-tiles \
  -H "Content-Type: application/json" \
  -d '{"roi":[18.25,18.45,77.25,77.45],"tile_deg":0.2,"force":false}'
```

4. Run one-command readiness smoke check:
```bash
poetry run python scripts/smoke_tile_3d_readiness.py
```

5. Pass criteria:
- Basemap + overlay tile requests return `200 image/png`.
- Tile headers include `X-Fallback-Used: false`.
- `terrain-3d` reports `used_synthetic=false` and `is_fallback=false`.

## Build & Quality Gates
- [x] Backend tests run in CI.
- [x] Ruff lint runs in CI.
- [x] Frontend `npm run build` passes in CI.
- [x] Frontend contains no hardcoded `http://localhost:5000/api/v1` calls.

## Deployability
- [x] `webui/.env.example` defines `VITE_API_URL`.
- [x] Docker image can run CLI and API entry commands.
- [x] Cloud start command for API is documented.
- [x] Portfolio README includes architecture, run steps, and known limits.
