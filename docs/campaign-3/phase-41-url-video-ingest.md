# Campaign 3 Phase 41 — URL media acquisition

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The authenticated file surface accepts supported video URLs. A dedicated fetcher
handles internet acquisition and captions, then hands downloaded bytes into the
same leased media pipeline as uploads. It is separate from the restricted media
worker. URL validation, host allowlists, quotas and bounded retries apply.

The fetcher writes to staging. Audrey validates and moves the result into owned
storage; workers do not write the metadata database. Download progress must emit
both the first update and the final completion update. Prefer usable captions
to repeating speech transcription, and preserve their source attribution.

Current files retain original downloads for playback. Previously reclaimed bytes
cannot be restored by changing retention; the viewer explains that state and
offers the source URL. Campaign 4 owns the current playback contract.
