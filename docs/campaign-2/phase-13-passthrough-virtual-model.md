# Campaign 2 Phase 13 — Passthrough virtual model

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

`audrey_passthrough/<concrete>` sends a chat to an explicitly allowed Ollama
model while retaining authentication, per-user in-flight limits and fair local
GPU scheduling. It skips classifier, planner, deep synthesis and banners.
`config.yaml` owns the allowed concrete models.

Agent-provided tool schemas must round-trip. Ollama response calls are translated
to the OpenAI tool-call shape, including stringified JSON arguments and call IDs,
so clients receive structured calls rather than leaked tool syntax in prose.
Streaming accumulates tool-call deltas consistently with non-streaming replies.

Current identity and bot access are defined by Campaign 4; the original external
authentication dependency is retired.
