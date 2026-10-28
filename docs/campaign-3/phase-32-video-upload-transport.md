# Campaign 3 Phase 32 — Chunked media transport

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Authenticated upload sessions split large media into bounded parts, then
validate and assemble them server-side. Sessions, parts and completion are owned
by the caller. Quota and file-type checks apply to the assembled file too.

Transport and processing are separate: accepted video bytes enter an honest
`pending` state rather than holding the upload request through transcription.
Single-request uploads remain available for suitable files. Current limits come
from configuration and the file API, not this historical edge-cap measurement.
