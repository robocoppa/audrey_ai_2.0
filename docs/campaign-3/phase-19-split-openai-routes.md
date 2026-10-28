# Campaign 3 Phase 19 — OpenAI route package

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The OpenAI route layer was split by responsibility into schemas, passthrough,
response formatting, streaming and endpoint orchestration. Package exports
preserve the public import surface.

The split preserves response shapes and SSE order: the first-token event precedes
answer deltas, tool-call accumulation stays intact, and progress stage ordering
is consistent across the extracted helpers. Later files may evolve independently
without reviving the monolithic route module.
