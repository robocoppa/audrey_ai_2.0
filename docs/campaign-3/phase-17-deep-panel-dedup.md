# Campaign 3 Phase 17 — Shared deep-panel setup

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

One preparation helper owns worker selection, bounded registry fallback,
subtask assignment, dispatch metrics and coroutine construction. Non-streaming
and streaming entry points differ in how they await results (`gather` versus
completion order), not in which workers they select or what they send.
