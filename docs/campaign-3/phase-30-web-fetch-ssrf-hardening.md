# Campaign 3 Phase 30 — Web-fetch connection contract

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

URL parsing and validation failures produce controlled fetch errors. Only
HTTP(S) public destinations are allowed; every redirect is validated again.
Resolve once, validate all returned addresses and connect to a vetted address
while retaining the original HTTP Host and TLS server name. HTTP libraries must
not silently resolve the hostname again after validation.

Body handling bounds decompression before large decoded buffers form. Overall
wall-clock deadlines cover redirects and reads; a concurrency cap limits resource
use. HTML extraction runs without an independent network request.

The hardening was historically deployed and checked. The optional multi-address
connection fallback is **closed as not pursued in this campaign**; no new
implementation or expanded host compatibility is claimed by this cleanup.
Internal review details belong in gitignored review records.
