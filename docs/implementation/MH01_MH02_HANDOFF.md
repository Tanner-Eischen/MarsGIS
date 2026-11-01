# MH-01 / MH-02 local evidence handoff

Date: November 1, 2026. Status: local portfolio handoff prepared; neither release
package is complete. No visibility change, merge, publication or deployment occurred.

## Scope and preserved baseline

The checkout remains on `codex/portfolio-validation` at
`459973dba92dc5b82f9d24519fd333577646ae3a`, with existing uncommitted release and
dependency repairs preserved. GitHub API inspection found the repository private
and PR 28 open/unmerged. The reference retains its original dirty
`4eb93e615c985edca10643b3c0752587d9dc7a21` baseline and per-source hashes.

No scientific source, parameter, coordinate, mask, scoring or routing semantics
changed. The new exporter collects selected derived artifacts without regenerating
the historical reference or redistributing source DEMs.

## Reproduction and actual checks

Run from the repository root using the existing patched development environment:

```powershell
python -B scripts/export_portfolio_handoff.py --out reports/portfolio-evidence-2026-11-01
```

The exporter refuses an existing output directory, checks all five original output
hashes, executes the existing independent static reference checker, lints the new
exporter, checks exported text for local paths/credential patterns and verifies
the exported checksum manifest. Exact fresh check records and sanitized logs are
in `reports/portfolio-evidence-2026-11-01/validation.json` and adjacent log files.

The compact handoff supplies `portfolio-evidence.json`, a reference metadata
summary, report, comparison figure and route geometry. It omits the raster outputs;
complete private reference files are needed to rerun the static verifier. It is
a local publication candidate, not a publicly accessible evidence release.

The recorded runtime summary comes from the October 31 patched release proof;
its SHA-256 and original date remain explicit. Those tests and container checks
were not rerun for this handoff. The current source identity covers dirty files,
so HEAD alone is not the tested source identity.

## Requirement and test mapping

- MH-R01: private access verified through API; sanitized showcase publication pending.
- MH-R03 / MH-T04: reference summary retains source, radius/datum, CRS, affine,
  units, grid spacing, configuration and original output hashes. Fresh static
  verification checks geometry, masks, score formulas and accumulated costs.
- MH-R05: independent full-graph shortest-path optimality remains pending; the
  existing checker integrates serialized route costs without rerunning a solver.
- MH-R08 / MH-T10: prior local container evidence used 1 GiB/two CPUs and reached
  734,089,216 bytes peak. No 512 MiB headroom claim is supported.
- MH-T09: existing hash gate rejects altered reference artifact bytes. New explicit
  negative tests for changed source, parameters and CRS remain pending.

## Remaining gates and publication boundary

MH-01 still requires owner access/release decision, complete history and asset
rights review, signed-out accessible release links and rollback mapping. Existing
private source and archives must remain private unless separately authorized.

MH-02 still requires a fresh locked-environment numerical workflow, the stronger
independent graph solver, complete proposed release-manifest validation contract,
negative corruption tests and versioned public evidence publication. Browser and
WebGL interaction, live patched deployment, capacity headroom and scientific
research extensions were not run. Browser tools remain prohibited by the user's
current instructions.

Publication rights for the selected derived assets are explicitly pending review.
No private source links are presented as public. No machine paths, credentials,
raw source DEMs or source repository archive enter the compact export.

The handoff unlocks local portfolio drafting with correct private-access labels.
It does not unlock a verified live-demo or certified habitat/rover-safety claim.
Rollback consists of removing only this new handoff directory, this documentation
and its standalone exporter; historical references and earlier archives are untouched.
