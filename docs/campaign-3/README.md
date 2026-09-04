# Campaign 3 — correctness foundations, Audrey UI, and reusable skills

**Status:** Campaign 3 Phase 1 is complete. Campaign 3 Phase 2 is in
progress; identity slices 2A.1 and 2A.2, including the account-purge lifecycle
corrective, and slice 2A.3 canonical application state are complete and
Unraid-verified. Slice 2A.4's optional Cloudflare Access boundary has also
passed its default-disabled Unraid smoke. Milestone 2A is complete. Milestone
2B is in progress: Its first owner-bound conversation/history API slice is
complete and Unraid-verified, as is its universal run-event/native-run slice.
The AG-UI boundary adapter and real pipeline tool/source observation wiring
are also complete and Unraid-verified. Canonical archive dual-write and its
rebuildable chat-search projection are laptop-complete; the 2B.5 Unraid gate
is next.

Campaign 3 first strengthens Audrey's platform boundaries and operational
contracts, then makes Audrey itself the application behind a native web client,
and finally adds the first general skills layer on that owned surface.


## Sequence

| Phase | Plan | Outcome | Entry gate | Status |
|---|---|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening-plan.md) | Strengthen data boundaries, request ownership, storage lifecycle, readiness, and runtime reproducibility | Campaign start | Complete |
| 02 | [Audrey application and web UI](phase-02-audrey-ui-plan.md) | Provider-neutral identity, Audrey-owned conversations and runs, a structured agent protocol, and a native browser client | Phase 01 completion gate | In progress (2B.5 laptop-complete; Unraid gate pending) |
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
