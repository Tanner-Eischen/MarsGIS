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
