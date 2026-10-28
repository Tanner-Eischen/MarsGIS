# Interview walkthrough

Start with the problem: comparing candidate locations and routes is only useful if coordinates, units, masks and numerical methods are trustworthy. MarsHab demonstrates those engineering practices on a public planetary DEM.

## Five-minute demonstration

1. Run `scripts/run_portfolio_demo.py` in the pinned environment. Explain the small public CTX download, checksum check and absence of synthetic fallback or paid credentials.
2. Show `comparison.png`: observed terrain, slope, relative suitability and the three routes. Explain that scores are terrain-only and candidates are cells rather than construction recommendations.
3. Open `manifest.json`: source WKT differs from the local analysis CRS; all derivatives share the same 20 m grid. The source, code, package versions and outputs are identifiable.
4. Compare route length with straight distance and terrain cost. The cost-optimal route may be longer but cheaper under the explicitly stated terrain penalty. It does not predict rover energy.
5. Run `scripts/verify_portfolio_demo.py`. Show one analytical plane test and one independent Dijkstra test. Explain why shape/existence assertions would have missed the original formula and heuristic defects.

## Defensible technical discussion

- Why use a local metric projection? Gradients require compatible horizontal and vertical units; treating degrees or a nominal MOLA scale as actual CTX meters corrupts slopes.
- Why common validity masks? A missing observation must not receive a neutral score or change normalization inconsistently across criteria.
- Why preserve full route pixels? Smoothing/downsampling can change the discrete terrain objective or cut across obstacles.
- Why separate the CLI from legacy UI claims? The audit established a reproducible scientific path; other branches still need evidence. The UI build passes but its interactions and hosting are unverified.
- What remains? Hazard subpixel information, physical rover calibration, environmental criteria, full legacy workflow validation and deployment hardening. These are explicit limitations rather than implemented science claims.

## Portfolio wording supported by the evidence

“Developed and validated a reproducible Mars terrain-analysis workflow using Python, Rasterio, NumPy/SciPy and pyproj; added analytical numerical tests, independent shortest-path comparisons, provenance manifests and georeferenced raster/vector exports.”

Show the repository, [methods](scientific-methods.md), [audit matrix](feature-verification.md), figure and a locally regenerated report. Do not claim production deployment, a validated solar/mission model, or operational habitat safety.
