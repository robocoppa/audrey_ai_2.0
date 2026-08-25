# Campaign 3 — correctness foundations and reusable skills

**Status:** In progress as of 2026-08-25. Wave 1A is laptop-complete:
private-search isolation is Unraid-verified, and the request-ownership slice
awaits its user-run Unraid smoke.

Campaign 3 first strengthens Audrey's platform boundaries and operational
contracts, then uses those foundations to add the first general skills layer.


## Sequence

| Phase | Plan | Outcome | Entry gate | Status |
|---|---|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening-plan.md) | Strengthen data boundaries, request ownership, storage lifecycle, readiness, and runtime reproducibility | Campaign start | In progress (Wave 1A) |
| 02 | [Reusable skills](phase-02-skills-capability-plan.md) | Local versioned instruction/resource bundles, explicit selection, enforced tool narrowing, and an evidence-gated automatic selector | Phase 01 completion gate | Planned |

Phase numbers repeat across campaigns. Refer to these as Campaign 3 Phase 1
and Campaign 3 Phase 2, or use the topic filenames.

## Campaign rules

- One deployable slice at a time. Each slice gets a laptop verification gate
  and a separate user-run Unraid smoke.
- Correctness tests land before refactors that change task, stream, or storage
  ownership.
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
