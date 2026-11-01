# MH-03 non-browser request and HTTP acceptance

Date: November 1, 2026. This work verifies client state logic through CLI tests
and an actual local container HTTP workflow. It does not claim browser/WebGL
interaction or a public deployment.

## Bounded changes

The existing explorer now accepts an optional, bounded `request_id`. The server
generates one for older clients that omit it. Server job UUID, client request ID,
result and downloaded planetary GeoJSON remain bound together. A calculated
result with a different job identity fails before publication.

The React client uses an active request object, an AbortController and the current
selection to gate every POST and poll completion. Coordinate/slope changes and
unmounting abort network work. Late successes, late failures and A→B→A selection
changes cannot restore an abandoned request. Transport validation rejects wrong
job IDs, wrong request IDs and mismatched nested result identities.

This is client cancellation of transport/publication. An already accepted server
calculation continues to drain under the original single scientific worker slot;
it is not forcibly terminated or falsely reported as stopped. Historical jobs
retain their original UUIDs and bounded retention. A new request may receive 409
until that worker finishes. No queue or extra worker was introduced.

Liveness remains 200 independently of DEM availability and adds a cached digest
of all actual backend Python bytes. Configured Git ancestry is separately labeled
and cannot establish the digest. Readiness validates existing pinned MOLA/CTX
sources through the existing source store, never provisions them, and returns
503 when they are unavailable. It no longer treats the mere existence of folders
as scientific readiness or reports untested plugins as ready. Restart a process
after changing source bytes; the identity is deliberately cached per process.

Terrain kernels, source data, graph conventions and scientific defaults did not
change. The expensive fixed reference comparison was not recomputed for this gate.

## Verification actually executed

- Backend Ruff lint passed. The targeted explorer, health, unified-host and
  planetary-routing suite passed all 81 tests; the strengthened no-download health
  assertions subsequently passed all five health tests again.
- Frontend lint passed with zero errors and 15 existing hook warnings. The two
  transport/component test files passed all 26 tests, including delayed POST,
  delayed poll, A→B→A, stale failed POST and mismatched response cases.
- TypeScript/Vite build passed. The lazy 3D chunk remains larger than the build's
  advisory size threshold; no browser-loading performance was measured.
- Component tests use mocked map/3D components in JSDOM. They are not browser,
  visual, keyboard-accessibility or WebGL acceptance evidence.

## Actual derivative-container gate

Native Docker in Ubuntu WSL was used without launching Docker Desktop or a GUI.
The base was the exact cached patched runtime
`sha256:026b1fb772805cd2e554a277510e9012e3a226b2f8e5928708b42d71a07ba52d`.
A derivative Dockerfile removed and replaced its application source/script and
frontend trees with a frozen current snapshot. It used no dependency installation
or source downloads during build (`--pull=false`, `--network=none`).

The resulting image was
`sha256:3d6a00a1f0fe2a975a79bc41bd6a5332e57ab09178e585f13a2a6d2a20a18aa2`.
The actual 94-file backend digest was
`d3a76ec64c221ec453709bb4fce72399115f5f4332fcdd90198e89cce1a9a8d1`.
The HTTP response matched that frozen digest, and all 19 packaged frontend files
matched the newly built bytes delivered over HTTP.

This is an explicit cached-runtime derivative profile, not a clean build of the
repository's original Dockerfile. Its ancestry was `c9908c2`, while the exact
current bytes also included preserved, preexisting release/profile changes.
The source manifest, lock hashes and backend/frontend fingerprints identify that
dirty profile independently; a Git label alone does not identify it.

The container ran with one application worker, a 1 GiB memory limit and two CPUs.
Pinned cached MOLA/CTX files were mounted read-only. Actual loopback requests
verified source readiness, two distinct real global tiles, a real CTX route, a
real MOLA antimeridian route, spherical lengths, matching 3D coordinate payloads,
request/run identities and matching GeoJSON downloads. The 3D assets were fetched,
not executed. A genuine simultaneous three-request burst returned 202, 409, 409;
the earlier export remained unchanged.

The observed memory peak was 368,402,432 bytes. Docker reported no OOM and cgroup
`oom_kill` was zero. This is an observation for the specified journey/burst, not
a general headroom or load-capacity benchmark. A separate missing-CTX profile of
the same image returned liveness 200 and readiness 503 without leaking its source
path. Both application containers and the disposable dependency probe were
stopped/removed; unrelated containers were not modified.

The actual native interpreter was Python 3.11.18. All 72 applicable runtime and
bootstrap dependency pins matched the recorded locks, and `pip check` passed.
This environment is separate from the earlier fresh Windows/Python 3.11.15
reference reconstruction.

## Evidence and reproduction

`reports/mh03-loopback-container-2026-11-01/` contains the compact source manifest,
derivative Dockerfile, native gate, package-pin audit, selected HTTP results,
sanitized logs, provenance, portfolio handoff and SHA-256 manifest. The retained
private build context/image provide exact replay; large source assets, complete
source archive, caches and environments are excluded from the compact report.

After supplying the frozen source/compiled frontend context and pinned cache
mounts, the essential commands are:

```powershell
python -m ruff check marshab tests scripts
python -m pytest -q -o addopts= -p no:cacheprovider --disable-warnings tests/api/test_explorer_api.py tests/api/test_health_api.py tests/api/test_unified_host.py tests/unit/test_planetary_routing.py
# In webui/: npm run lint; npm test -- --maxWorkers=1 src/lib/explorer.test.ts src/pages/MarsExplorer.test.tsx; npm run build
# Native Docker: build the recorded derivative context; run with --memory=1g --memory-swap=1g --cpus=2, loopback-only port publication and read-only pinned DEM mounts.
python scripts/smoke_unified_app.py --base-url http://127.0.0.1:PORT --skip-comparison --loopback-contracts --report new-http-result.json
```

Use a new report path; the CLI refuses to overwrite an existing report. Burst
checks refuse non-loopback hosts. The original Dockerfile was not executed, and
no visibility change, push, cloud plan change or deployment occurred. Public
source/access, browser/keyboard/mobile/WebGL verification and release publication
remain open gates under the user's current no-browser instructions.
