# Campaign 3 Phase 04 — file and artifact downloads

**Status:** Complete and accepted.

## Contract

- `GET /api/files/{file_id}/download` streams the authenticated owner's original with its stored filename/media type, attachment disposition, byte-range support, private `no-store`, and `nosniff`.
- Missing and foreign ids receive the same 404. A retained owner row whose original was reclaimed or is missing receives 410.
- My Files hides original downloads before a fetched source lands and after reclamation. The route remains authoritative if storage changes after listing.
- `GET /api/files/{file_id}/artifacts/{artifact}/download` accepts `transcript`, `visual`, and `summary` for videos. Missing/empty sidecars return 404; non-video files return 422. Reclaiming an original does not remove its derived text.
- Derived names are `<video>.transcript.txt`, `<video>.visual-notes.txt`, and `<video>.summary.txt`. Transcript and Visual notes expose download actions; Summary stays copyable with no download button or empty action row.

## Summary behavior

New video summaries use a natural two or three sentence description with an 80-word safety cap. The accepted local primary/fallback generation path avoids repeating the writing instruction. A legacy row summary remains a viewer fallback when no summary sidecar exists.

No further work is scheduled for this phase.
