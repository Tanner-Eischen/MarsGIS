# MH-03 coherent source/profile integration

November 1, 2026. The request/readiness changes were tested with existing repaired
dependencies and client fixes. Those directly used inputs are now included in
the same scoped source change, rather than leaving a clean checkout with
incompatible metadata or an incomplete frontend lint profile.

The reviewed subset comprises package version/metadata; exact bootstrap,
runtime/development inputs and locks; matching hash-required CI/Docker installs;
frontend package/lock and ESLint flat configuration; and two typed error-variable
declarations required by that lint profile. The latter preserve assignments in
both existing try/catch branches. Numerical terrain/routing methods, defaults,
source data and historical reference artifacts were not changed. NumPy, SciPy
and Rasterio versions remain the previously verified scientific profile.

A clean archive of staged source tree
`ca78a084ec328cf835e6eb8bc386725b6a1243f1` was tested using the existing matching
pinned development tools. It excludes unrelated untracked private packaging
scripts/tests. The frontend used the existing locked dependency installation via
a local junction; it was not described as a fresh npm installation.

- Backend lint: passed; full offline suite: 460 passed, one Windows skip.
- Frontend lint: zero errors, 15 existing hook warnings; all 115 tests passed.
- TypeScript `--noEmit`: passed.

Full logs and exact staged-source identity are retained under
`reports/mh03-coherent-profile-2026-11-01/`. Warning counts remain recorded in the
logs. The existing derivative-container packet remains unchanged and explicitly
describes its original dirty input snapshot; it is not relabeled as a pristine
commit result.

Legacy README/compose/other release documentation, the separate private release
builder/tests and generated output remain outside this scoped commit. An original
multistage-Dockerfile build and fresh image HTTP/resource gate follow separately
from clean committed source, with context exclusions audited before building.
No public deployment or browser interaction is established by this integration.
