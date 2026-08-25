# Campaign 3 — correctness foundations and reusable skills

**Status:** Planned on 2026-08-24. No Campaign 3 implementation has started.

Campaign 3 turns the 2026-08-24 whole-codebase review into shippable work, then
uses the repaired request, lifecycle, and capability boundaries to add Audrey's
first general skills layer.

## Source review

- [`../reviews/codebase-review-2026-08-24.md`](../reviews/codebase-review-2026-08-24.md)

## Sequence

| Phase | Plan | Outcome | Entry gate | Status |
|---|---|---|---|---|
| 01 | [Audit remediation](phase-01-audit-remediation-plan.md) | Close H1–H7 and M1–M9 in independently deployable waves; add safe data controls and component readiness | Review complete | Planned |
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

The audit's remaining product backlog—OCR, broader audio/media ingestion,
ordinary-answer provenance, artifact download, and `/v1/responses`—stays
available for later Campaign 3 phases. It is intentionally not interleaved with
the safety foundations or the first skills rollout.
