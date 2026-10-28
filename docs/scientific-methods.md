# Scientific methods and limits

The validated workflow is `scripts/run_portfolio_demo.py`, backed by `marshab/analysis/portfolio.py`. It shares numerical terrain, scoring, routing and raster-export functions with the application. It bypasses legacy cache selection and placeholder criteria rather than treating those branches as verified.

## Source and spatial reference

- Product: `JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif`.
- [USGS product description](https://www.usgs.gov/media/images/m20jezerocraterctxdemmosaic20m100jpg); [direct GeoTIFF](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif).
- Observations: NASA Mars Reconnaissance Orbiter CTX stereo, MOLA controlled. This is a gridded, processed elevation product, not raw laser returns or an invented terrain surface.
- Verified source SHA-256: `16a1aa66c3f48c1bca252795827db7b417876217aff0f5e1aff18027fab2327b`; 9,662,387 bytes; 1,512 rows by 1,596 columns; 20 m native projected pixels.
- Source CRS: equirectangular Mars 2000 sphere, radius 3,396,190 m, standard parallel/central meridian 0 degrees; actual WKT and affine are captured in the manifest. Elevation is in meters in the source product's MOLA/DeltaGeoid reference; no vertical datum conversion is performed.
- Source NoData is approximately -3.402822655e38 and is passed explicitly to reprojection. Invalid target cells are NaN in memory and -9999 in exported float rasters.
- Requested ROI: 18.25–18.45 degrees N, 77.25–77.45 degrees E. Coordinates are east longitude and planetocentric latitude on the documented sphere.

Bilinear reprojection produces a 593 by 563 grid of 20 m cells in a spherical azimuthal equidistant projection centered at 18.35 N, 77.35 E. The pixel-aligned envelope extends slightly outside the requested ROI; centers outside the ROI are masked. The affine transform, not the requested ROI, controls all exports. Interpolation changes samples and does not add independent spatial information.

All distances in the example are planar grid distances. Local distortion is small over this approximately 12 km region, but off-center interpoint distances are not exact geodesics. General global/polar/antimeridian workflows are not validated. The DEM loader accepts only north-up unrotated grids with explicit CRS, transforms geographic ROIs into projected source space, and rejects nonoverlap. Geographic spacing uses a center-latitude Mars-sphere approximation; the flagship avoids that approximation by using a local metric grid.

## Terrain calculations

For north-up rows increasing southward, let `p=dz/dx` and `q=dz/drow_distance`, with elevation and horizontal distance both in meters. Gradients use central differences inside and one-sided differences at edges. Invalid centers and affected derivative stencils remain invalid.

- Slope: `atan(sqrt(p^2 + q^2))`, converted to degrees.
- Aspect: `atan2(-p,q) mod 360`, downslope clockwise from north; 0=N, 90=E, 180=S, 270=W. Slopes below 0.1 degrees have undefined aspect encoded as -1.
- Roughness: population standard deviation of elevations within a 5 by 5 window, in meters (100 m footprint). This includes regional slope; it is not detrended microtopographic roughness. Public helper default is 3 by 3.
- TRI: mean of the eight absolute center-to-neighbor elevation differences, in meters. This is explicitly the mean absolute difference definition, not the square-root sum-of-squares definition found in some GIS tools.
- Hillshade: `sin(h)*cos(s) + cos(h)*sin(s)*cos(A-aspect)`, clipped to [0,1] and scaled to uint8 [0,255], using altitude h=45 degrees and azimuth A=315 degrees. Internal aspect is the unthresholded gradient direction. No distant cast shadows are modeled. Invalid uint8 cells encode zero; the exported float band uses the slope validity mask to distinguish invalid cells from black shade.

Neighborhood filters reflect values at raster boundaries and propagate NaNs through their footprint. Candidate selection excludes a five-cell outer margin. A valid 20 m cell is not evidence that hazards smaller than that scale are absent.

## Suitability and candidate selection

Only slope (weight 0.70) and roughness (0.30) enter the flagship score; lower is preferred for both. For each active criterion, normalize as `(max-value)/(max-min)` over the common finite-cell population. A constant valid criterion contributes 0.5. Invalid cells remain NaN; zero-weight criteria do not change validity. Weights must be finite, nonnegative, and sum to one within 1e-8.

The weighted sum is an ROI-relative comparison, not an absolute safety probability. Changing the ROI changes the normalization population. Tests demonstrate expected ranking changes under alternative weights. TOPSIS has separate hand-calculated tests but is not used by the flagship.

Traversability requires slope <=15 degrees and 5x5 roughness <=20 m. These are illustrative thresholds, not validated rover or habitat requirements. Select the three highest-scoring traversable interior cells in the largest connected component, greedily enforcing 1.5 km separation and separation from the start. The proposed start is the connected cell nearest pixel (10,10). Ties use row-major ordering. Candidates are individual cells, not certified habitat footprints; disconnected areas are excluded from this demonstration.

## Routing objective

For valid traversable cells:

`cost = exp(2*slope/15) + 5*(roughness/(20+1e-6))^2`

Each directed edge costs the destination-cell multiplier times its step length: 20 m for orthogonal movement or `20*sqrt(2)` m for diagonal movement. The start cell incurs no initial cost. This convention makes the discrete graph directed. Obstacles are nonfinite cells or cells beyond either threshold. Diagonals require both adjacent orthogonal cells to be passable, preventing corner cutting.

A* uses a Euclidean-distance lower bound scaled by the minimum finite cell cost. Costs below one remain valid; zero minimum cost reduces the heuristic to Dijkstra. Negative costs and nonpositive/nonfinite spacing are rejected. Smoothing is disabled because it can change the optimized terrain objective.

The report lists route length, straight-line start-to-candidate distance, and accumulated terrain cost in meters equivalent. The manifest also records the terrain cost of the shortest geometric grid route under identical barriers. Optimized terrain cost must not exceed that baseline. This penalty is not energy, travel time, or a calibrated mobility model. The exported full grid path is the actual optimized route.

## Numerical verification

| Assertion | Reference | Tolerance / convention |
| --- | --- | --- |
| Plane slope and aspect | Analytical gradients on square and rectangular grids | Float64 assertions generally 1e-10 to 1e-12; see test-specific tolerances |
| Roughness and TRI | Explicit neighborhood standard deviation / absolute differences | NumPy allclose defaults on known small fixtures; edge checks use pytest approximate comparison |
| Hillshade | Independent surface-normal dot product; flat surfaces at 0, 30 and 90 degree altitude | At most one uint8 quantization level |
| Routing optimality | Independent SciPy sparse Dijkstra on small known grids | Relative/absolute tolerance 1e-12; barriers, low costs, zero costs and corners included |
| Suitability | Hand-computed normalized weighted sums and TOPSIS | Numerical tolerance 1e-7; exact validity masks; sensitivity and zero-weight cases |
| Ingestion/export | Known affine centers, projected ROI bounds, NoData and shape | Exact metadata/masks where possible; coordinate assertions in tests |
| Local projection | Forward/inverse Mars projection at ROI center and corners | 1e-10 degrees absolute; this is not the unfinished legacy SITE transform |
| Repeated requests | Default A, custom B, default A with real numerical kernels | Identical A outputs and unchanged global defaults |
| Real exports | Independent verifier reopens both GeoTIFFs, reconstructs route pixels and integrates costs | 1e-6 pixels, 1e-6 relative cost, 2e-7 score after float32 export |

Tests specify their tolerances, including NumPy defaults where not overridden. Generated analytical fixtures test correctness, not observational provenance. The strict real-data example separately verifies source bytes. `scripts/verify_portfolio_demo.py` does not call the terrain scorer or route solver: it recomputes the scoring and optimized route-cost formulas from exported bands and validates masks, connectivity and geometry. The distance-only baseline path is not exported; its reported scalar cost is compared with the optimized cost, while independent optimality evidence comes from the small-grid Dijkstra tests.

Output SHA-256s identify exact artifacts from one run. Repeated runs in the same pinned environment must reproduce numerical artifacts. Cross-platform GDAL, PROJ and plotting differences can change bytes; manifests record versions so cross-platform comparisons should use numerical tolerances rather than assuming all PNG/TIFF hashes match.

## Unvalidated scientific features

Solar code has local-incidence/altitude issues and an unlabelled synthetic API fallback; it has no seasonal ephemeris, terrain horizon, atmospheric or dust time-series validation. Mission-impact economics are heuristic. Science value and dust overlays contain placeholders. The legacy SITE coordinate conversion does not supply a validated inverse and mixes latitude conventions. None supplies evidence for the flagship results; see the feature matrix for required corrections before expanding claims.

## Website data fidelity

The website packages the existing exported rasters without recomputing scientific metrics in JavaScript. Five exact-size transparent PNGs use the full grid; a band-major little-endian Float32 buffer stores the same elevation/slope/roughness/hillshade/suitability cells as the exported TIFFs. The client verifies the buffer SHA-256 before probing. Routes are projected back to pixel centers using the exported affine and explicit Mars CRS. Candidate markers use `(column+0.5,row+0.5)`, matching those same grid cells.

The server's Run command invokes the shared validated Python workflow, independently verifies its exports, packages them, verifies the package and only then publishes a new immutable run. Existing displayed results and their run-specific URLs remain coherent during computation. The public API accepts no arbitrary source URL, region or output path.
