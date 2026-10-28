# Campaign 3 Phase 29 — Readable web-page fetching

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

`web_fetch` opens one HTTP(S) URL and extracts readable HTML/plain text with
trafilatura. Extraction consumes already-fetched content; it must not perform
an independent unguarded fetch. Search remains a separate tool.

The client enforces public destinations, revalidates every redirect and bounds
bytes, time and concurrency. [Phase 30](phase-30-web-fetch-ssrf-hardening.md)
records the hardened connection contract. Model use and answer quality need live
evidence; making a tool real does not establish that every worker uses it well.
