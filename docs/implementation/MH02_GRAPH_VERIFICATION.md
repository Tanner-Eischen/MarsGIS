# MH-02 independent archived graph verification

Date: November 1, 2026. This extends local scientific verification; it does not
complete release publication, application interaction or deployment acceptance.

## Design and affected files

`marshab/analysis/verification.py` now independently assembles the complete
eight-neighbor directed graph from the serialized raster cost and passability
mask. An edge costs the destination cell's serialized multiplier times 20 m
(orthogonal) or `20 * sqrt(2)` m (diagonal). The start incurs no initial cost.
Diagonals are excluded if either adjacent orthogonal cell is blocked. Grid
spacing comes from the checked affine/manifest, rather than a solver default.

SciPy sparse Dijkstra runs once from the declared start to all grid nodes. It is
an existing locked dependency. The checker imports neither the application
pathfinder nor its graph-construction/scoring helpers. Terrain formula and score
reconstruction remain local independent checks before graph construction.

For each archived route the checker verifies finite Mars coordinate pairs,
cell-center alignment, bounds, start/end cells, passability, connected steps,
corner constraints, length and accumulated destination-cell costs. It compares
both the serialized-path cost and the archived reported objective with the
Dijkstra optimum using relative tolerance `1e-6`, suitable for exported raster
precision. Different paths with equal objective remain valid. Optimality applies
to this finite graph, not physical mobility or hazards below the source scale.

The verifier returns a numerical summary and accepts `--report` through the
existing `scripts/verify_portfolio_demo.py` wrapper. Reports are created
exclusively; an existing report is never overwritten. Original reference files,
scientific defaults and the earlier compact handoff are unchanged.

## Reproduce and inspect evidence

From the repository root in the existing patched development environment:

```powershell
python -m ruff check marshab/analysis/verification.py tests/unit/test_independent_reference_verification.py
python -m pytest -q -o addopts= -p no:cacheprovider --disable-warnings tests/unit/test_independent_reference_verification.py tests/unit/test_portfolio_exports.py tests/unit/test_numerical_terrain_routing.py tests/unit/test_portfolio_site.py tests/api/test_portfolio_service.py
python -B scripts/verify_portfolio_demo.py webui/public/portfolio --report reports/mh02-graph-verification-2026-11-01/new-verification.json
```

Use a new report filename on every run. The actual recorded final run is
`reports/mh02-graph-verification-2026-11-01/verification-final.json`; adjacent
logs and `check-results.json` distinguish lint, targeted tests, archived artifact
verification and the report-overwrite rejection check. `SHA256SUMS` identifies
the captured evidence files. The report includes the exact checker and historical
manifest SHA-256s; it does not relabel the reference's old dirty source revision.

The subsequent versioned compact handoff is
`reports/portfolio-evidence-2026-11-01-graph-verified/portfolio-evidence.json`.
Its generator now records the full-graph check as passed, rather than carrying
forward the earlier packet's pending status. The original packet remains
unchanged. Export with an explicit `--release-id` and a new `--out` directory;
the exporter refuses to replace an existing packet. Git attributes preserve all
hash-bound report bytes across platform checkouts.

The verified graph has 333,859 raster nodes, 300,500 passable nodes and 2,359,894
directed edges. All three archived routes match their reconstructed optima:

| Rank | Route length (m) | Dijkstra cost (m equivalent) |
| --- | ---: | ---: |
| 1 | 11,510.924958 | 27,898.494775 |
| 2 | 18,000.672427 | 35,873.346822 |
| 3 | 10,036.366517 | 25,936.577287 |

The final targeted run passed all 77 tests (438 warnings; warning details
suppressed in the final compact log). Formatting, Ruff
lint, the actual archived verifier and report-overwrite protection also passed.
The initial broader run had five fixture-metadata failures (72 tests passed);
that log is retained separately. The existing generated packaging fixture now
declares its source CRS, graph conventions and synthetic software identity,
without representing the fixture as observational evidence.

## Meaningful failure checks

The new test module constructs a hand-calculated constant-cost 3-by-3 Mars export
without using the production analysis pipeline. It checks asymmetric destination
costs, explicit zero-cost edges and diagonal barrier disconnection. Corruption
tests include changed bytes, malformed/source-inconsistent CRS, a terrain raster
with changed CRS, invalid cost values/formulas, malformed or missing geometry,
disconnected steps and wrong endpoints.

Semantic corruption tests deliberately refresh artifact hashes so passing the
integrity layer cannot hide numerical/geometry defects. A valid connected detour
has matching updated length, costs, manifest and hashes; only the new shortest-path
check rejects it. Existing export, numerical routing, package and API orchestration
tests provide compatibility coverage for callers of the extended verifier.

## Remaining limits

This checks archived outputs in the current development environment. Fresh locked
workflow regeneration, new source acquisition, full corruption-source tests,
complete release-manifest contract, signed-out public access, asset permission
review and versioned publication remain separate work. No browser, WebGL,
visibility, merge, deployment or paid-call actions occurred.

The earlier Docker proof used a 1 GiB/two-CPU local container. Full-graph
reconstruction adds memory work; that container/resource test has not been rerun
for this verifier change. The prior 1 GiB measurement remains historical, and no
512 MiB headroom or new deployment-capacity claim is supported.

Rollback restores only the verifier source and removes the new test/documentation;
retain these versioned results as historical evidence. Earlier release archives
and the previous portfolio handoff remain recoverable and unchanged.
