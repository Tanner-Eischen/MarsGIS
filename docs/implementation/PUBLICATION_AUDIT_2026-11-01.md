# Repository publication audit

November 1, 2026. Read-only audit before the browser-fix handoff; root owns remote
pushes, visibility and deployment. No history rewrite was performed.

All remote heads/tags were fetched, then every advertised pull-request head and
merge ref was fetched into local `refs/audit/`. The resulting inventory covered
37 local/audit refs, 189 reachable commits and 1,118 unique blobs. The later
acceptance commit is checked separately as a staged/local change.

Official Gitleaks 8.30.1 used its vendor-default rules across `--all`, with
100-percent redaction, inline suppressions ignored and an empty ignore file.
It reported 13 generic-api-key detections, all in scientific/source provenance
receipts. Every detected value was independently matched to the referenced
source-file SHA-256 (including recorded Git checkout line-ending forms).
These are noncredential digests: zero real credentials and zero unresolved
secret findings were identified. No secret values were printed or copied into
the public report.

The two historical `.env.example` blobs contain only credential-free API and
WebSocket endpoint settings. No actual `.env`, private-key/certificate container,
restricted benchmark/client-data path, or blob larger than 10 MB was identified
in the audited reachable history. This does not include ignored local files,
which are not candidates for publication.

Eight unique image blobs were inventoried. Two were inspected legacy app-only
screenshots with no account/browser chrome, credentials or personal data; the
other six are the generated NASA/USGS-derived comparison and terrain-layer
figures. Their observed-DEM provenance and source attribution remain in the
scientific manifests/methods. The legacy screenshots are not current browser
acceptance evidence. No restricted employer/client material was identified in
the audited paths or asset classes.

No repository LICENSE file was found and no new license grant is asserted by
making the repository visible. This audit records inspected content/provenance;
it is not a legal ownership certification. No ownership or credential questions
were required for these findings.

Machine-readable refs, redacted finding classifications and checksums are in the
new browser-acceptance evidence packet. Raw scanner output and full binary/source
inventory are retained privately. The audit does not change deployment, runtime
secrets, source methods or previously archived receipts.
