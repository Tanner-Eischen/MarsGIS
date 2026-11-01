# Wave 2 browser acceptance

November 1, 2026. Actual browser/UI testing was explicitly authorized. This record
supersedes earlier local "browser not run" status, without rewriting historical
HTTP/numerical/container receipts or claiming a public deployment was verified.

## Tested profile and browser

The final test image used the verified original image `215303ce3216` and its baked,
pinned MOLA/CTX inputs, plus the exact worker-lifecycle file and rebuilt frontend
from staged source snapshot `45cdaa3c8b7b3e53f5ee7a0a01b5a8dfb7888c2f`.
The final backend digest was
`197047df690db53a94dbe14776919718d03708e362ec25ef41c87d22942ff5e4`.
This is a documented test overlay on the original runtime, not an assertion that
the earlier original image already contained the fixes. Corrected test coverage
was subsequently merged without changing these runtime bytes.

Playwright CLI 0.1.23 used its pinned Playwright 1.65.0-alpha-2026-10-30 and exact
Chromium revision 1252 / Chrome for Testing 157.0.8103.15. Actual WebGL 1.0 ran
through ANGLE/SwiftShader. Hardware GPU and other browser coverage are not inferred.

Windows-to-WSL transport required a transparent Windows loopback HTTP bridge to
the private WSL interface. It preserved request body/query/Host and response
status/content/download bytes. Dedicated ports, persistent terminal lifetimes and
backend fingerprint checks prevented cross-service/port-reuse mistakes. These
are test-harness conditions; no production proxy or Vite/mock API was substituted.

## Functional and visual observations

| Journey | Observed result |
| --- | --- |
| Initial map; A/B map clicks | Real pinned terrain loaded; clicks set distinct actual coordinate fields |
| CTX route | Ready real job, 20 m source/analysis label, matching request/job/result IDs |
| MOLA route and keyboard | Four fields reached by Tab, submitted by Enter; real dateline route and coarse 3.7 km label |
| Browser download | Completed GeoJSON download bound to the computed job/result |
| Actual 3D | Real surface/scatter3d WebGL canvas; orbit, wheel zoom and explicit camera reset worked |
| Exaggeration 1x to 100x | Aspect ratio changed 100-fold; scientific arrays/export bytes unchanged; no new scientific POST |
| Delayed/out-of-order/ABA | Real unmodified API responses held/reordered; abandoned results did not replace current selection, even when transport ignored abort |
| Actual cancellation | Browser emitted `net::ERR_ABORTED`; stale result stayed hidden; accepted server worker drained normally |
| Unavailable WebGL | Accessible fallback alert; 2D map, measured results and download remained available; no unhandled error |
| Narrow/mobile | 760/390 px layouts settled without horizontal overflow; keyboard/download controls remained usable; fresh mobile view inspected |
| Comparison page | Authentic full recomputation completed and changed UI to "Showing results computed by this analysis service" |

Screenshots were inspected for desktop, actual 3D, mobile and unsupported-graphics
fallback. Plotly resize is asynchronous; initial pre-settling screenshots were
diagnostic attempts, not promoted as final overflow results. Plotly's input-layout
camera is not its current interactive camera; assertions used the actual scene's
`getCamera()` after correcting that harness assumption.

Two actual UI defects were fixed: reset now explicitly calls `Plotly.relayout`
instead of relying only on `uirevision`, and unavailable/context-lost graphics now
produce an announced map/download fallback instead of a broken renderer pane.
Existing measured-elevation, NoData, toolbar, resize and renderer-error tests were
retained alongside the new regressions. No renderer changes alter scientific data.

## Free-tier resource and numerical acceptance

The initial 1 GiB journey including full comparison reached 656,302,080 bytes.
A genuine 512 MiB / 0.1 CPU browser journey also completed without OOM, but reached
its cap and triggered limit-pressure/cache reclaim events. That observation did
not establish headroom.

The worker now explicitly collects completed analysis cycles after `run_demo`
returns and before allocating independent package-verification buffers. A weakref
regression proves closed analysis cycles do not overlap the next verification
stage when automatic collection has not occurred. Collection stays in the worker;
terrain/scoring/routing methods, source bytes, defaults and numerical gates are
unchanged.

The final current-source browser journey, including real CTX/MOLA/WebGL actions
and full comparison recomputation, completed at **536,870,912 bytes / 0.1 CPU**
without OOM or OOM kills. Peak reached the configured cap; six limit-pressure
events were observed. Maximum sampled anonymous memory was 500,281,344 bytes and
maximum sampled file memory was 33,603,584 bytes. Cache ownership/reclaim and host
differences prevent a general Render headroom or production-capacity guarantee.
Full recomputation can take several minutes on this CPU tier; prior verified
results remain inspectable while it runs. No paid plan was requested or enabled.

All five serialized scientific files—both TIFFs, route GeoJSON, report and PNG—were
byte-identical to the pre-fix recomputation on this same runtime. New run IDs and
provenance differ as expected; no cross-platform byte-identity claim is made.

## Verification and handoff

Backend clean-source checks: 461 passed, one existing skip. Full frontend checks:
118 passed, including six renderer cases. Backend/frontend lint and TypeScript
checks passed; existing hook/deprecation warnings remain recorded.

The compact receipt is `reports/mh03-browser-acceptance-2026-11-01/`; larger traces,
network recordings and diagnostic attempts remain under versioned
`output/playwright/`. The separate publication audit covers all advertised heads,
tags and pull refs. Root owns push, auto-deploy and live-site verification.

An optional instruction-file improvement is drafted, not applied: read existing
target/test files before replacing them, and preserve baseline regression coverage
while adding fixes. No new AGENTS.md, resume edits or credential prompts were
required for this acceptance.
