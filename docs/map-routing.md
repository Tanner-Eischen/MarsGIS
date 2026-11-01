# Mars map, selected-point routes and 3D

## User workflow

1. Drag the global map to pan. Scroll, pinch, or use its zoom controls to inspect another part of Mars.
2. Choose **A — start**, click the map, then choose **B — destination** and click again. Coordinate fields support keyboard entry and editing.
3. Press **Plan route**. The server selects the appropriate measured DEM and calculates a bounded terrain-weighted route. Moving the map never starts an analysis.
4. Inspect the path on the map and in the **Map + 3D** or **3D path** view. Drag the 3D scene to orbit; use its toolbar to pan and zoom.
5. Change **Vertical exaggeration** from1× to100×. Both controls update the same scene. **True scale —1×** restores physical vertical proportions; all stored elevation measurements remain unchanged.
6. Download the calculated planetary GeoJSON. Editing either point or the slope limit invalidates the old route; results for earlier selections are not silently reused.

**Jezero example** selects two known low-slope CTX cells about1.5 km apart. It is a convenient short demonstration while the whole-planet map remains available. The prior reproducible candidate/suitability workflow is preserved under **Site comparison**.

## Actual elevation sources

| Source | Coverage / native sampling | Reference system | Pinned SHA-256 |
| --- | --- | --- | --- |
| [MOLA MEGDR16 global topography](https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg016/) | Global;5,760×2,880;16 pixels/degree, about3,704 m at equator | Planetocentric, east-positive0–360°; sphere radius3,396,000 m; GMM3 degree/order50 areoid | `d18d9b9ab8c5516d02e157dd2cde0f1d0d160c21940e953ba22391269a545e7b` |
| [Jezero CTX stereo DEM](https://www.usgs.gov/media/images/m20jezerocraterctxdemmosaic20m100jpg) | Approximately77.1605–77.6990° E,18.2110–18.7212° N; native20 m | Mars2000 sphere radius3,396,190 m; source MOLA/DeltaGeoid vertical reference | `16a1aa66c3f48c1bca252795827db7b417876217aff0f5e1aff18027fab2327b` |

MOLA is observed-derived median topography, with interpolation where measurements are absent. Its authoritative PDS label does not declare a NoData sentinel. The supplied ENVI header's zero-as-NoData hint is **not** applied:2,128 source cells legitimately contain zero meters. The raw source is big-endian signed Int16; its MD5 also matches the official PDS inventory. See the [PDS label](https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg016/megt90n000eb.xml).

The two sources retain their own reference spheres and vertical datums. They are selected separately, never silently fused or relabeled. CTX is used only when the complete buffered routing window is covered. Otherwise the result identifies MOLA and its coarser resolution.

## Map display

Geographic tiles use two tiles across and one down at zoom zero, with northwest origin(-180°,90°). Longitudes wrap; rows run north to south. Leaflet uses a cloned geographic CRS with the MOLA Mars sphere for display distance calculations, not Earth's radius. CTX coverage tiles remain in canonical longitude bounds so they also appear in wrapped copies of the map.

The global basemap is colorized native MOLA elevation with shaded relief calculated from its immediate native neighbors. Relief uses north-up gradients, latitude-dependent east-west spacing, sun azimuth315°, altitude45°, and a visual-only3× gradient factor. This does not modify elevation or routing values. Poles use one-sided differences, longitude neighbors wrap, invalid data remains transparent, and flat terrain keeps its original color. CTX overlays use the actual source and preserve its NoData.

Zooming beyond a layer's native sampling enlarges the display, not its information content. Global tiles are cached only within a fixed entry/byte budget; the source stays in a read-only memory map. Request handlers never allocate full-world float grids or initiate dataset downloads.

## Routing method and limits

The server forms a regional window around the chosen points, using the shortest unwrapped longitude interval and a buffer. CTX has a100 m minimum margin; global MOLA uses a source-scale angular margin. Native rows/columns are sampled at a bounded stride so each analysis side has at most256 cells. MOLA is never upsampled and advertised as CTX-resolution data.

Slope uses the physical local spacing. MOLA derivatives convert angular gradients to meters using the source sphere and cosine of latitude. CTX derivatives use its metric-grid Jacobian. Roughness is5×5 elevation standard deviation in meters at the actual analysis spacing. The illustrative dimensionless cell penalty is:

`1 + (slope_degrees / 10)^2 + (roughness_meters / scale)^2`

The roughness scale is500 m for MOLA and20 m for CTX. The chosen slope limit and invalid cells create barriers. These are documented analysis settings, not calibrated rover engineering limits.

A* uses an eight-connected graph, exact source-sphere great-circle edge lengths times destination-cell penalties, and a great-circle lower bound scaled by the minimum finite cell cost. Diagonal moves cannot cut blocked corners. Reported path length is the sum of spherical edge distances; the direct comparison uses the same snapped endpoints and source radius.

The optimum applies to this sampled regional graph, not every possible route across the planet or a continuous physical surface. Endpoints snap to real analysis cell centers. Coincident physical points, identical snapped cells, invalid/blocked endpoints and unreachable requests return an error rather than a fabricated straight path. Coarse MOLA cannot resolve rover-scale hazards.

## 3D consistency

The route and display surface share a source-specific local azimuthal equidistant frame, with x/y/z in meters. Plotly displays all three axes in kilometers and reports actual elevation meters in hover details. The surface is sampled to at most96×96 cells; the full path retains its measured vertices and the mesh declares its row/column sampling.

Vertical exaggeration multiplies only the manual scene aspect ratio. It never changes path coordinates, elevation samples, costs, distances or exports. Null gaps remain gaps. Camera orbit and reset remain independent of the selected exaggeration. Large regional projections are visual previews; route distances still use the sphere rather than planar screen coordinates.

## Resource bounds and verification

The globe source is33 MB and the CTX source is9.7 MB. Both are checksum-verified and baked into the one-service image. The route service has no queue, keeps at most five completed jobs with an additional byte cap, and shares one process-wide scientific workload slot with site comparison. Tiles remain available while a route computes.

Tests cover authoritative source bytes, valid zero cells, tile alignment/poles/seams, analytical slope, independent Geod/Dijkstra costs, blocked corners, date-line/antipodal/polar routes, bounded grids, source selection, matching 3D coordinates, stale UI state, map controls and unchanged elevations under exaggeration. HTTP smoke verification additionally checks both hemispheres, a20 m CTX point pair, a global date-line pair, downloads and the lazy3D asset.

Actual local browser acceptance was authorized and executed on November 1, 2026: real WebGL orbit/zoom/reset, 1x–100x export invariance, keyboard/mobile, stale-response controls and unavailable-WebGL fallback passed. See [browser/free-tier receipt](implementation/MH03_BROWSER_ACCEPTANCE_2026-11-01.md). Component tests remain separately labeled virtual-DOM checks. Public deployment identity and hardware/other-browser coverage are separate from the local receipt.
