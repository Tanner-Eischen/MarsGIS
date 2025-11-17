**PRD: Mars Landing Site & Route Decision Lab**

**1. Overview**

* Build a “Mars Terrain Decision Lab” focused on **landing site selection for a solar‑powered lander**, with **routes/waypoints** evaluated in the same criteria language as sites.
* Provide intuitive **presets** for tradeoffs (safety vs energy vs science) with **progressive disclosure** into fine‑tuning.
* Ground everything in **DEM‑based shaded relief** with **sun lighting controls**, so users can visually validate decisions (no heavy analytical overlays yet).

**2. Goals**

* Enable users to:
  * Rank candidate **sites** using multi‑criteria presets.
  * Evaluate **navigation routes/waypoints** using comparable cost criteria.
  * Visually inspect selected sites/routes on **DEM imagery with adjustable shading and sun angle**.
* Showcase:
  * Thoughtful UX (presets + progressive disclosure).
  * A clear domain story (mission‑style decision support).
  * Solid architecture (clean separation of data layer, scoring, and UI).

**3. Non‑Goals (for this phase)**

* No analytical overlays like slope heatmaps, roughness layers, or complex contour visualizations.
* No AI/chat UX yet (but design should be compatible with adding it later).
* No advanced export/import tooling (full data packages, CO‑GeoTIFF, etc.).
* No multi‑user collaboration, auth, or role management.

**4. Users & Personas**

* **Mission Designer / Systems Engineer**  
  Needs to justify site and route choices using clear criteria and visuals.
* **Planetary Scientist / Researcher**  
  Wants to explore terrain tradeoffs and understand implications of different priorities.
* **Advanced Student / Enthusiast**  
  Wants an educational, exploratory tool that feels like a “real” mission planning lab.

**5. Core Concepts**

* **DEM**: Base elevation raster for ROI; used for visualization, shading, and terrain‑based costs.
* **Site**: A candidate landing area with attributes (location, slope stats, elevation, solar potential, etc.).
* **Waypoint**: A point along a route (navigation plan), potentially with tolerance and other constraints.
* **Route**: An ordered sequence of waypoints between two or more sites; has an aggregate cost.
* **Criterion**: A dimension used for scoring (e.g., slope safety, roughness, solar, science proximity).
* **Preset**: Named combination of criterion weights for either sites or routes (e.g., _Safe Landing_, _Balanced_, _Science‑Focused_).
* **Advanced Weights**: Optional, finely tunable criterion weights accessible through progressive disclosure.

**6. Primary User Flows**

* **Flow A: Landing Site Selection**
  
  * User selects ROI + dataset, chooses a **site scoring preset** (e.g., Safe, Balanced, Science).
  * System:
    * Computes site scores from existing site dataset and DEM/MCDM logic.
    * Shows a **ranked list** of sites and highlights top sites on DEM map.
  * User can expand an “Advanced weights” panel to customize criteria if desired.
  * For a selected site, user can open a **details view** showing explanation and DEM with shaded relief and sun controls.

* **Flow B: Route & Waypoint Evaluation**
  
  * User selects start and end sites or an existing route (sequence of waypoints).
  * User picks a **route cost preset** (e.g., Shortest Path, Safest Path, Energy‑Optimal).
  * System:
    * Computes route cost using DEM‑dependent criteria (slope, roughness, elevation), plus distance.
    * Displays route overlay on DEM map and a simple cost breakdown (e.g., distance vs risk).
  * Advanced panel lets users adjust route criteria weights, with cost recalculated.

* **Flow C: Combined Scenario (Landing Site + Route)**
  
  * User chooses a landing site preset → identifies top sites.
  * User picks two or more top sites and chooses a route preset to plan a traverse between them.
  * System:
    * Shows selected sites and the planned route, with costs and DEM visualization.
  * User inspects both **site scores** and **route costs** and uses shaded relief and sun angle controls to validate.

**7. DEM Visualization & Shaded Relief (Priority P0)**

* **Requirements**
  
  * R1: Provide a DEM‑based imagery view of the ROI.
  * R2: Support shaded relief (hillshade) computed from DEM gradients.
  * R3: Expose **sun azimuth** and **sun altitude** parameters so users can control lighting.
  * R4: Provide a **relief exaggeration** control to amplify visual 3D effect.
  * R5: Keep this performant enough to update within ~1–2 seconds on typical ROIs.

* **Behavior**
  
  * Backend exposes DEM imagery endpoint that:
    * Accepts ROI, dataset, resolution, colormap.
    * Accepts relief, sun_azimuth, sun_altitude parameters.
    * Returns shaded PNG + bounds via headers.
  * Frontend:
    * Uses existing map page as base.
    * Adds UI to control relief, sun azimuth, sun altitude.
    * Updates DEM imagery when these values change (with debounce to avoid spamming backend).

**8. Preset‑Based Criteria Exploration (Sites & Routes) (Priority P1)**

* **Presets for Sites**
  
  * At least three site scoring presets:
    * **Safe Landing**: High weight on slope safety and roughness; low on science.
    * **Balanced**: Mixed weights on safety, solar, science.
    * **Science‑Focused**: Higher weight on science proximity, moderate safety.
  * For each preset, backend returns:
    * Name, description.
    * Criterion weights.

* **Presets for Routes**
  
  * At least three route cost presets:
    * **Shortest Path**: Minimize distance; safety constraints still enforced but less emphasized.
    * **Safest Path**: High penalty for steep slopes, hazardous terrain; distance less prioritized.
    * **Energy‑Optimal**: Balances distance and elevation/roughness to minimize energy use.
  * Each preset defined similarly via weights.

* **Progressive Disclosure**
  
  * Default UI shows:
    * A **preset dropdown**.
    * A short textual explanation of what that preset optimizes.
  * An “Advanced” or “Customize weights” toggle:
    * Reveals sliders for individual criteria (0–1 or 0–10).
    * Shows how the preset values map to those sliders initially.
  * Advanced panel is clearly optional; user is not required to touch it.

**9. Explainability (Sites & Routes) (Priority P1/P2)**

* **For Sites**
  
  * When a site is selected:
    * Show criteria contributions (e.g., simple bar chart or textual list: Slope 0.8, Roughness 0.7, Solar 0.6, Science 0.9).
    * Provide a plain‑language summary, derived from thresholds:
      * “This site is very safe (low slopes), with moderate solar exposure and high science value.”
  * Show how changing preset changes the rank and score for that site (even just “Rank changed from #3 to #1”).

* **For Routes**
  
  * When a route is selected:
    * Show total cost broken into components: distance, slope penalties, roughness penalties, maybe solar exposure.
    * Plain‑language summary:
      * “This route is longer than the shortest possible path but avoids steep slopes in these segments.”
    * Optionally highlight cost‑heavy segments in the future (but keep overlay details minimal in this first phase).

**10. Data & Inputs**

* DEM data for Mars (MOLA / HiRISE / CTX).
* Precomputed **sites** dataset (CSV + derived terrain attributes).
* Precomputed or generated **waypoints/routes** dataset.
* Preset and criteria configuration (likely a small YAML or JSON file).

**11. Milestones**

* **M0**: DEM + shaded relief + sun controls working in the visualization page.
* **M1**: Site scoring presets + basic site ranking and explanation UI.
* **M2**: Route cost presets + basic route cost breakdown UI.
* **M3**: Progressive disclosure with advanced sliders for both sites and routes.

* * *

**Architecture & Files**

**1. High‑Level Architecture**

* **Backend layers**
  
  * **Data access layer**: Uses existing DataManager / DEM loaders to fetch DEMs and site/waypoint data.
  * **Scoring/analysis layer**:
    * Site scoring: MCDM logic that converts DEM‑derived and site attributes + weights into site scores.
    * Route cost: cost function that aggregates terrain‑based costs over waypoints.
  * **Preset/config layer**:
    * Loads preset definitions from a configuration file (e.g., YAML) and exposes them via API.
  * **API layer** (FastAPI):
    * Visualization endpoints (DEM + shading).
    * Analysis endpoints (site scores, route costs, presets).

* **Frontend layers**
  
  * **State management**:
    * Selected ROI and dataset.
    * Selected site and route.
    * Active preset (site/route).
    * Advanced weights (if user opens advanced controls).
    * DEM visualization state: relief, sun azimuth, sun altitude.
  * **UI structure**:
    * Decision Lab page:
      * Left: Presets + advanced weights + list of sites/routes and their scores.
      * Right: DEM map with shaded relief and overlays (sites, routes).
      * Optional panel for explanation and 3D/lighting controls tied closely to DEM.

**2. Backend Files (Proposed)**

* **Docs**
  
  * docs/PRD_Landing_Site_Decision_Lab.md
    * Contains the PRD above.
  * docs/architecture/Decision_Lab_Architecture.md
    * Contains high‑level diagrams, data flow description, and module responsibilities.

* **Config**
  
  * marshab/config/criteria_presets.yaml
    * Defines site_presets and route_presets.
    * Each preset: { id, name, description, scope (site|route), weights: {slope, roughness, solar, science, distance, ...} }.

* **Analysis / Domain Modules**
  
  * marshab/analysis/criteria.py
    * Defines criterion names/enums and type definitions for CriterionWeights.
    * Helper to apply default values and validate weights.
  * marshab/analysis/site_scoring.py
    * Functions:
      * score_sites(sites_df, dem_metadata, weights) -> DataFrame with scores and component contributions.
      * Uses existing MCDM machinery if available (e.g., from what tests/unit/test_mcdm.py covers).
  * marshab/analysis/route_cost.py
    * Functions:
      * compute_route_cost(waypoints_df, dem, weights) -> RouteCostResult.
      * Breaks down cost into components (distance, slope_penalty, roughness_penalty, etc.).

* **API Modules**
  
  * marshab/web/routes/criteria_presets.py
    * Endpoint: GET /api/v1/analysis/presets
      * Returns list of site and route presets with metadata and criterion keys.
  * marshab/web/routes/site_analysis.py
    * Endpoint: POST /api/v1/analysis/site-scores
      * Input: ROI, dataset, preset ID or custom weights.
      * Output: list of sites with scores, component contributions, and simple textual explanation.
  * marshab/web/routes/route_analysis.py
    * Endpoint: POST /api/v1/analysis/route-cost
      * Input: route definition (start site, end site, or waypoints), preset ID or custom weights.
      * Output: total cost and component breakdown, plus basic explanation.
  * marshab/web/routes/visualization.py (existing)
    * Extend DEM endpoint to accept shading controls:
      * New query params: relief, sun_azimuth, sun_altitude.
      * Uses these to compute hillshade and adjust the DEM colormap for map/3D usage.

* **Tests**
  
  * tests/unit/test_criteria_presets.py
  * tests/unit/test_site_scoring.py
  * tests/unit/test_route_cost.py
  * tests/api/test_site_analysis_api.py
  * tests/api/test_route_analysis_api.py

**3. Frontend Files (Proposed)**

* **Docs**
  
  * webui/README_decision_lab.md
    * Briefly explains the Decision Lab page, presets, and how to demo it.

* **Pages**
  
  * webui/src/pages/DecisionLab.tsx
    * Main experience page:
      * ROI & dataset selection (can reuse patterns from Visualization.tsx).
      * Site & route preset selectors.
      * Advanced weights panel (hidden by default).
      * Site and route lists with scores.
      * Embedded DEM map + shading controls.

* **Components**
  
  * webui/src/components/PresetsSelector.tsx
    * Props: list of presets, current preset ID, onChange.
    * Scope: can be reused for both sites and routes (with a label).
  * webui/src/components/AdvancedWeightsPanel.tsx
    * Collapsible panel with sliders for each criterion.
    * Shows how they map to the active preset initially.
  * webui/src/components/SiteScoresList.tsx
    * List of sites, their total scores, and a small per‑criterion breakdown.
  * webui/src/components/RouteCostSummary.tsx
    * Displays route cost breakdown (distance, risk, energy).
  * webui/src/components/ExplainabilityPanel.tsx
    * Shared component that takes a scored site or route and renders:
      * Component contributions.
      * Plain‑language explanation string.
  * webui/src/components/TerrainMap.tsx (existing)
    * Extended to:
      * Accept relief, sunAzimuth, sunAltitude as props.
      * Call updated DEM endpoint with those parameters.
      * Display sites and routes as separate layers.
  * webui/src/components/SunLightingControls.tsx
    * Small control bar with sliders for sun azimuth & altitude, plus relief.
    * Sits near the map, feeds props into TerrainMap.

* **Services**
  
  * webui/src/services/analysisApi.ts
    * Functions:
      * getPresets()
      * getSiteScores(request)
      * getRouteCost(request)
    * Typed interfaces for requests and responses.

**4. Data Flow (High Level)**

* **Site scoring**
  
  * Frontend:
    * User selects ROI + dataset + site preset → calls getSiteScores.
  * Backend:
    * Loads preset weights; uses existing MCDM logic + DEM/site data to compute scores.
    * Returns scores and component contributions.
  * Frontend:
    * Renders ranked site list + explanation.
    * Highlights top sites on TerrainMap.

* **Route cost**
  
  * Frontend:
    * User selects sites or route → calls getRouteCost with selected route preset.
  * Backend:
    * Loads DEM for route corridor; computes cost using route_cost module.
    * Returns cost breakdown.
  * Frontend:
    * Displays route overlay on map + cost summary and explanation.

* **DEM + shading**
  
  * Frontend:
    * ROI/dataset/lighting state → fetch DEM image with relief, sunAzimuth, sunAltitude.
  * Backend:
    * Generates shaded DEM image and returns it.
  * Frontend:
    * Updates TerrainMap ImageOverlay and keeps it in sync with site/route overlays.
