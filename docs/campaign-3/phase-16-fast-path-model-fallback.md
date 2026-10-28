# Campaign 3 Phase 16 — Fast-path fallback and acknowledgment

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The plain fast path tries a bounded number of healthy candidates after an
`OllamaError`, recording failures/cooldown and returning only a successful reply.
The tool-using branch remains single-attempt to avoid replaying side effects.
Streaming must not restart an answer after user-visible output has begun.

The fast `Thinking` acknowledgment precedes classifier work. A conservative
short-prompt gate can skip a router call; explicit task signals still take
precedence. Both graph and streaming routes preserve the same routing rules.
