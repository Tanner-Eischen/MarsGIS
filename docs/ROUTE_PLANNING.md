# Route Planning Flows

MarsHab has two distinct route-planning flows:

## 1. Navigation (waypoints for mission scenarios / nav UI)

- **API:** `POST /api/v1/navigation/plan-route`
- **Module:** `marshab.web.routes.navigation`
- **Engine:** `NavigationEngine.plan_to_site()`
- **Output:** Waypoints in SITE frame (North, East, Down), written as CSV
- **Use case:** Mission scenarios, rover navigation UI, CLI `marshab navigate`

## 2. Route cost analysis (cost breakdown between two sites)

- **API:** `POST /api/v1/analysis/route-plan`
- **Module:** `marshab.web.routes.route_analysis`
- **Engine:** `analysis.routing.plan_route()` + `compute_route_cost()`
- **Output:** GeoJSON route, waypoints, cost summary (slope, roughness, shadow)
- **Use case:** Route analysis UI, comparing route costs between sites

Both flows use A* pathfinding and terrain cost surfaces. The navigation flow is optimized for single-destination waypoint generation; the route analysis flow adds detailed cost breakdown and shadow penalties.
