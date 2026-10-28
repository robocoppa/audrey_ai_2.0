# Campaign 2 Phase 02 — Prompt centralization and composition

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Pipeline prompt defaults live in
[`pipeline/prompts.py`](../../src/audrey/pipeline/prompts.py). One override loader
resolves `agentic.prompts` values; missing or blank overrides use defaults, and
oversized overrides produce a warning rather than silent truncation.

The shared composer orders incoming system messages, an optional task role,
memory context/hints, then tool-specific chat-history guidance. Guidance is
included only when its tool is available. Both graph and streaming entry points
use the same composition rules.

Phase 2a shipped the composer and prompt refactor. The speculative Phase 2b
fast-answerer/deep-worker prompt expansion is **closed without a scheduled
implementation**; a new prompt change requires a concrete behavior to improve.
