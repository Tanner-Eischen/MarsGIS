# MH-03 original multistage container acceptance

November 1, 2026. This supplement used the original committed multistage
Dockerfile, separately from the earlier cached-runtime derivative gate.

## Source and context

The source was a clean archive of
`6482f998619673329e5b6a4d7a51709e20a5823d`, including the coherent package,
dependency, client and request/readiness changes. The archive SHA-256 is
`ab4f547577df37091e4d3b4fed85a6c49f878dc396b1c632018d3f58cae5745e`.
Git for Windows checkout line-ending filters apply to those archive bytes; the
same 94-file backend digest was tested by the clean-source QA and the image.
No cross-platform raw-source byte-identity claim is made.

The only overlay was the audited `.dockerignore`, excluding generated outputs,
reports, virtual environments and scratch/model/coverage directories while
preserving `webui/public/portfolio/**`. The Dockerfile itself remained byte-for-byte
equal to its archived copy. No Python, frontend or dependency metadata overlay
was needed. Unrelated working-tree changes were not used.

An actual scratch-image COPY/export audit proved the private-path sentinels absent
and the required terrain/suitability TIFFs, routes, manifest, comparison figure and
probe buffer present. Docker received a 23.56 MB build context. Sentinels were
context-exclusion tests, not scientific source data.

## Original build and genuine source acquisition

The original build completed in 191.13 seconds. Its
`RUN python scripts/prepare_planetary_data.py --output-dir /app/reference` step
actually executed, acquired and verified the real public source files:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| MOLA16 | 33,177,600 | `d18d9b9ab8c5516d02e157dd2cde0f1d0d160c21940e953ba22391269a545e7b` |
| Jezero CTX | 9,662,387 | `16a1aa66c3f48c1bca252795827db7b417876217aff0f5e1aff18027fab2327b` |

Both source-file timestamps were fresh for this build, and their bytes/checksums
were verified again inside the resulting image. They total 42,839,987 bytes.
There were no cached-input build arguments, source replacement files or runtime
source mounts in this gate. Normal dependency-layer caching was permitted.

The base image IDs/digests were resolved and retained in the report. Both frontend
and backend installation/build steps used the committed profile and exact locks.
`pip check` passed, and all 72 applicable Python bootstrap/runtime pins matched.
The actual runtime was Linux/WSL Python 3.11.18.

Result image:
`sha256:215303ce321634736b84d5654588fa993b6609a3167800b407a167d8716eb5ff`.

Actual backend digest:
`75a4d0685557f662f3958a8c393298c87022b7f509fb85ff6397cac1f83ad9ec`.

## Actual loopback acceptance

The new image ran as configured with one application worker, 1 GiB memory and two
CPUs. The host port was published only on 127.0.0.1. Actual HTTP checks passed for
health/source readiness, request/run identity, a real CTX route, a real MOLA
antimeridian route, matching spherical lengths/3D coordinate payloads, matching
GeoJSON exports and all 19 packaged static files. The lazy 3D assets were fetched,
not executed in a browser.

A genuine three-request burst returned 202, 409, 409, with no queue and the prior
export unchanged. The observed memory peak was 340,197,376 bytes. Docker OOM was
false, and all cgroup OOM/kill counters were zero. This observation includes the
recorded validation probes and requests; it is not a general production-capacity
or memory-headroom guarantee.

The application and never-started scratch-audit containers were removed. Unrelated
containers and the Docker daemon were not modified. Images and private source/build
context remain available locally; no registry push or public deployment occurred.

## Handoff and remaining gates

The compact packet is `reports/mh03-original-container-2026-11-01/`. It includes
source/build provenance, source hashes, base/image IDs, context audit, real build
and HTTP logs, installed pin/source inventory, a portfolio handoff and checksums.
Large context/source archives, downloaded DEMs and images are excluded from Git.
Earlier derivative and numerical-reproduction packets remain unchanged.

Browser/WebGL rendering, keyboard/mobile interaction, public source/access and
deployment approval remain open. The fixed historical numerical workflow was not
regenerated here; its separate fresh-environment reproduction remains the evidence
for that scenario. This gate adds an original-image HTTP/resource observation and
does not expand the scientific methods, defaults or safety claims.
