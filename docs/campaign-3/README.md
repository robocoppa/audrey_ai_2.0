# Campaign 3 — correctness foundations, Audrey UI, and reusable skills

**Status:** In progress as of 2026-09-01. Waves 1A through 1D are complete and
Unraid-verified. Wave 1E.1 is implemented; its Unraid gate is next.

Campaign 3 first strengthens Audrey's platform boundaries and operational
contracts, then makes Audrey itself the application behind a native web client,
and finally adds the first general skills layer on that owned surface.


## Sequence

| Phase | Plan | Outcome | Entry gate | Status |
|---|---|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening-plan.md) | Strengthen data boundaries, request ownership, storage lifecycle, readiness, and runtime reproducibility | Campaign start | In progress (1E.1 Unraid gate next) |
| 02 | [Audrey application and web UI](phase-02-audrey-ui-plan.md) | Provider-neutral identity, Audrey-owned conversations and runs, a structured agent protocol, and a native browser client | Phase 01 completion gate | Planned |
| 03 | [Reusable skills](phase-03-skills-capability-plan.md) | Local versioned instruction/resource bundles, native explicit selection, enforced tool narrowing, and an evidence-gated automatic selector | Phase 02 completion gate | Planned |

Phase numbers repeat across campaigns. Refer to these as Campaign 3 Phase 1,
Campaign 3 Phase 2, and Campaign 3 Phase 3, or use the topic filenames.

## Campaign rules

- One deployable slice at a time. Each slice gets a laptop verification gate
  and a separate user-run Unraid smoke.
- Correctness tests land before refactors that change task, stream, or storage
  ownership.
- Audrey's internal application protocol is not the OpenAI compatibility
  protocol. Both adapt from the same typed run events.
- Browser authentication, Audrey authorization, and user-data ownership remain
  separate boundaries; no UI-supplied identity is trusted.
- Existing collection names and deployed source records remain recoverable
  through migrations and rollback windows.
- Skills do not execute code or grant permissions. Tools act; skills instruct;
  platform policy authorizes.
- A feature is not called verified on Unraid until the user confirms it.
- Any source/config edit runs the full hermetic suite, changed-file ruff, and
  the lesson-link checker required by `AGENTS.md`.

## What comes after these plans

The remaining product backlog—OCR, broader audio/media ingestion,
ordinary-answer provenance, artifact download, and `/v1/responses`—stays
available for later Campaign 3 phases. It is intentionally not interleaved with
the safety foundations or the first skills rollout.
