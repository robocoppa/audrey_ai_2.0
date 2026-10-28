# Campaign 3 Phase 22 — Explicit depth intent

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Adaptive routing considers the latest user turn's explicit depth cues as well
as token count. The configured phrase list is case-insensitive; an empty list
disables that signal. Generic words such as “compare” alone are not depth cues.

An explicit virtual-model choice and image handling take precedence. Streaming
and graph gates use the same helper and make the routing reason observable.
Current phrase values are in `config.yaml`.
