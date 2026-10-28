# Campaign 3 Phase 37 — Media summaries

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

A bounded text call summarizes transcript and visual descriptions after ingest.
It runs in Audrey, stores readable file metadata and adds a searchable summary
artifact. It complements the transcript rather than replacing timestamped evidence.

Inputs are sampled across the source rather than taking only the opening text.
The prompt distinguishes spoken claims from visible content. Summary failure
does not discard usable transcript/visual results; requeue clears stale summary
text. Historical happy-path operation was observed; remaining edge-case checks
are closed without a new live-validation claim.
