# Authorized Wave 2 browser acceptance inventory

November 1, 2026. Browser/UI access is now explicitly authorized for this work.
This inventory precedes browser execution and is not a passing result.

The baseline is source `6482f99`, original image `215303ce3216`, and the subsequent
context/evidence commit `b5931e4`. Existing numerical, HTTP and container receipts
remain immutable. Use an isolated named Playwright CLI session and versioned
artifacts under `output/playwright/`; do not share another agent's browser state.

| ID | Journey / fault control | Required evidence |
| --- | --- | --- |
| B01 | Desktop initial map and pinned source loading | Screenshot, console/network, source metadata |
| B02 | A/B map clicks and keyboard coordinate entry | Actual field changes and selected points |
| B03 | Real CTX route | Matching request/job/result IDs, source and spacing labels |
| B04 | Real MOLA route including dateline selection | Correct coarse-source/spacing labels and result |
| B05 | Route download | Browser download matches computed job/result/GeoJSON |
| B06 | Actual WebGL terrain, orbit/zoom/reset | Canvas/renderer observation, camera changes, screenshot |
| B07 | Exaggeration 1x to 100x | Display changes; route/export/scientific arrays unchanged |
| B08 | Delayed real POST/poll, out-of-order/ABA and cancellation | Unmodified real API responses delayed only; stale publication suppressed |
| B09 | Unavailable WebGL | Useful 2D/results/download fallback, no unhandled failure |
| B10 | Keyboard, narrow desktop and mobile | Focus/action usability, no horizontal overflow, screenshots |
| B11 | Short exploratory pass and comparison-page controls | Observed defects or explicit pass, no invented live checks |
| B12 | Server resource tier | Native 1 GiB browser journey first, then bounded 512 MiB observation; exact peak/OOM |

Toolchain: Playwright CLI 0.1.23 with its pinned Playwright
1.65.0-alpha-2026-10-30 and matching Chromium revision 1252 / Chrome for Testing
157.0.8103.15. Browser rendering may use SwiftShader; hardware-GPU coverage must
not be inferred. No Playwright test-runner migration is needed for this CLI audit.

Publication audit: fetch all refs/tags; scan every reachable commit with the
official Gitleaks default rules and redacted output; classify detections against
actual source/artifact provenance; inventory history assets and restricted-material
indicators. Report concrete blockers without rewriting history. Root owns pushes,
visibility and Render mutations. The current Render blueprint specifies the free
plan and one worker; do not create paid resources or upgrade plans. No resume or
credential edits/questions are part of this work.
