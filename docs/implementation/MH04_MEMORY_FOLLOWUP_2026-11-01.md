# 512 MiB warm-recompute memory correction

November 1, 2026. The public Render service was OOM-killed during a warm full
comparison at 12:14:59 UTC. That result overrides any inference of deployment
headroom from the earlier local browser receipt; earlier receipts stay unchanged.

The owning allocation was the independent graph verifier. For the real 593×563
grid and 2,359,894 directed edges, final CSR storage is 29,654,168 bytes, while
the previous triplet/list/concatenation construction peaked at 181,423,667 traced
bytes. It overlapped the warm application/analysis/package state.

Two-pass CSR construction now counts row edges, preallocates int32 indices and
float64 weights, and fills one directional mask at a time. Its measured peak is
42,647,461 bytes. The complete CSR data, indices and row-pointer byte hashes are
identical to the previous graph. Directionality, destination-cell weights,
corner barriers, explicit zero edges, sorted order and Dijkstra gates are retained.
No scientific defaults, grids, masks, costs or advertised features were disabled.

All 63 targeted graph/numerical/package tests and the archived real-route verifier
passed. A peak-allocation regression rejects the old duplicate-storage behavior;
an explicit directed reference checks all edges and corner/zero semantics.

A new native 512 MiB / two-CPU run first warmed real CTX/MOLA route history,
overload handling and 64 real map tiles, then repeated the ordinary complete HTTP
smoke and full comparison. Peak was 360,071,168 bytes, sampled anonymous maximum
323,276,800 bytes, and sampled file maximum 33,603,584 bytes. No OOM, kill or
limit-pressure event occurred. All five scientific outputs were byte-identical
to the frozen pre-fix run on the same runtime. The owned test container was removed.

This is bounded local evidence, not a general production/headroom guarantee.
The actual public 0.1-CPU service retest remains authoritative and is owned by root.
The compact receipt is `reports/mh04-memory-followup-2026-11-01/`.
