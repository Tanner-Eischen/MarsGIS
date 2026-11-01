# MH-02 fresh locked Jezero reproduction

Date: November 1, 2026. One fixed historical scenario was regenerated locally in
a fresh Python 3.11.15 environment and checked against the original reference.

## Exact reconstruction profile

Scientific source came from a ZIP created with `git archive` at
`82e24bac7de34f1cb80d648f71fd8c794a16b012`. Its archive SHA-256 is
`791662cf6c50be11358677309851a6ba080189304b78d8de2d861f14b05c4fe8`.
All 102 Python files under `marshab/` and `scripts/` matched that archive, and all
93 installed package Python files matched their archived sources.

The committed package metadata still constrained several libraries to versions
older than the repaired locks permit. Only the extracted `pyproject.toml` was
overlaid with the already-existing patched metadata. Its SHA-256 is
`2fd4972761edbeb3b42af00aa895d73475c1ed3e4595956e8cb3adc46ca2521b`.
The source ZIP and owning checkout were not changed. The distribution metadata
version is consequently 0.1.1; this does not identify the complete earlier patched
release or change the committed scientific Python code.

This is a **committed-source reconstruction with an explicit metadata overlay**,
not an unmodified baseline installation. Bootstrap, runtime and development
lockfiles were separately copied byte-for-byte, hash-checked and installed with
`--require-hashes`. The fresh environment's 82 applicable locked packages matched
their pins, and `pip check` reported no broken requirements. The overlay and exact
locks are retained as compact inputs in the report directory.

The extracted source has no Git checkout. Its generated manifest correctly
records `git_revision: unavailable` and `dirty: true`. The external provenance
record binds it to the actual source commit, archive hash, overlay hash, locks and
verified Python bytes; those generated fields were not rewritten.

## Execution and numerical result

The existing fixed workflow used the already-cached 9,662,387-byte CTX DEM, freshly
verified against `16a1aa66c3f48c1bca252795827db7b417876217aff0f5e1aff18027fab2327b`.
No new source download, parameter matrix or defaults change occurred. It wrote a
new isolated output directory and finished in 45.56 seconds on this host.

The committed independent verifier then passed all three fresh routes on a graph
with 300,500 passable nodes and 2,359,894 directed edges:

| Rank | Route length (m) | Directed Dijkstra cost (m equivalent) |
| --- | ---: | ---: |
| 1 | 11,510.924958 | 27,898.494775 |
| 2 | 18,000.672427 | 35,873.346822 |
| 3 | 10,036.366517 | 25,936.577287 |

The separate numerical comparison passed all eight raster bands with identical
finite-data masks, candidate identities and scoped scalar metrics. Physical
tolerances were declared in the comparison tool before the fresh results were
inspected: elevation 0.001 m absolute, slope/aspect 1e-5 degrees, roughness/TRI
1e-4 m, hillshade exact, suitability 2e-7 absolute, and route cost 1e-6 relative
plus 1e-6 absolute. CRS, affine, source identity and scientific configuration also
had to agree. Candidate coordinate tolerance is 1e-10 degrees. Independent route
optimality retains its 1e-6 relative cost tolerance.

These are numerical agreement gates, not a claim of byte-identical PNG/TIFF
outputs across platforms. This observation is one Windows/Python profile and one
fixed scenario. Complete per-band differences and scalar checks are in the report.

## Artifacts and reproduction

`reports/mh02-locked-reproduction-2026-11-01/` contains sanitized provenance, fresh
manifest, independent graph report, numerical comparison, exact metadata/locks,
installation and workflow logs, a portfolio handoff and checksums. Large raster
outputs, source ZIP, environment and caches remain outside the repository.

The recorded commands use an isolated source directory and fresh environment:

```powershell
git archive --format=zip --output source.zip 82e24bac7de34f1cb80d648f71fd8c794a16b012
# Extract source.zip; copy the recorded metadata overlay into source/pyproject.toml.
uv venv --python 3.11 environment
uv pip install --python environment/Scripts/python.exe --require-hashes -r inputs/requirements-bootstrap.lock
uv pip install --python environment/Scripts/python.exe --require-hashes -r inputs/requirements.lock
uv pip install --python environment/Scripts/python.exe --require-hashes -r inputs/requirements-dev.lock
environment/Scripts/python.exe -I -m pip install --no-deps --no-build-isolation source
environment/Scripts/python.exe -I -m pip check
environment/Scripts/python.exe -I -B source/scripts/run_portfolio_demo.py --source cached-jezero-ctx.tif --output-dir new-output
environment/Scripts/python.exe -I -B source/scripts/verify_portfolio_demo.py new-output --report new-verification.json
environment/Scripts/python.exe -I -B scripts/compare_reference_numerics.py source/webui/public/portfolio new-output --report new-comparison.json
```

Use new directories/report names; do not overwrite the historical reference.
Copy the recorded `inputs/` files to the reproduction workspace and verify their
published hashes first. Use the pinned cached DEM path in place of the example
filename. Building the package disables both dependency resolution and isolated
backend installation; the backend comes from the hashed bootstrap profile.

## Scope still open

This closes the single-scenario fresh-environment numerical reproduction gap for
the explicitly described reconstruction profile. It does not establish that all
uncommitted release repairs are committed, or that a pristine metadata profile
passed. Current repository dirt remains preserved.

Container resource behavior after graph verification, repeated-run determinism,
browser/WebGL interactions, deployment, public access/publication, asset-rights
approval, and optional sensitivity/footprint studies remain separate gates. The
old 1 GiB/two-CPU Docker record remains historical; this run made no Docker or
512 MiB headroom claim. No source visibility, merge, push, cloud or paid-model
actions occurred.
