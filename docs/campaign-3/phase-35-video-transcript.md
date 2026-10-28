# Campaign 3 Phase 35 — Timestamped transcripts

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Whisper runs in the media worker and returns timestamped segments. Transcripts
use an ingest path appropriate for speech chunks, retain timestamp provenance
and record the original media's metadata rather than the sidecar text's identity.

An empty/silent transcript can be a successful job. Completion posts the whole
result once, so a killed worker leaves no partial transcript searchable; lease
recovery retries it. Source bytes remain available to later visual processing
and the configured retention policy. Exact quotes also need
[lexical retrieval](phase-39-hybrid-retrieval.md).
