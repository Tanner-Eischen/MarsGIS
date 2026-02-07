# MarsGIS Portfolio Scope (Week 0)

## Objective
Ship a portfolio-ready MarsGIS demo focused on three reliable flows with deterministic outputs and clear engineering tradeoffs.

## In-Scope Flows
1. Site selection
- Input: ROI + criteria weights.
- Output: ranked candidate sites and GeoJSON overlay.

2. Route planning
- Input: landing/start point + selected top site.
- Output: route geometry and waypoint export.

3. Decision brief
- Input: selected site + computed metrics.
- Output: concise "why this site" summary using deterministic rules (no paid LLM dependency).

## Non-Goals (Portfolio v1)
- No additional feature surfaces outside these flows.
- No dependency on remote paid AI APIs for core demo.
- No requirement to run full real-DEM cloud pipeline on free hosting tiers.

## Demo Contract
- Fixed demo ROI and preset weights for recorded walkthrough.
- Deterministic synthetic fallback enabled with `MARSHAB_DEMO_SEED=42`.
- Expected outputs captured as baseline examples (rank ordering, route exists, summary fields present).

## Deployment Shape
- Frontend: Vercel static build with `VITE_API_URL`.
- Backend: Render/Fly Docker API service.
- If cloud resource limits apply, cloud demo runs in synthetic-only mode; real DEM workflow documented as local.

## Week 0 Exit Criteria
- Scope approved and frozen to 3 flows.
- Acceptance checklist defined and committed.
- P0 Week 1 tasks prioritized against this scope.
