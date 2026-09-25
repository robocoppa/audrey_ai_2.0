# Campaign 3 — correctness foundations, Audrey UI, and reusable skills

**Status:** Campaign 3 Phase 1 is complete. Campaign 3 Phase 2 is in
progress. Milestones 2A and 2B, slices 2C.1–2C.3, and slices 2D.1–2D.4 are
complete and Unraid-verified. The standalone Audrey UI and public Cloudflare
route are live. Slice 2D.5 is deployed, while its full administration, role,
model-publication, exact-owner bootstrap, and provider-binding soak remains
open.

The Milestone 2E parity build is laptop-complete across native memory
settings, owner-bound file and video upload and inspection, source and answer
presentation, image and direct chat attachments, safe retry, reload recovery,
empty-draft cleanup, saved source and tool summaries, durable attachment
presentation, answer and code copying, attachment-picker dismissal, and
contextual startup and recovered-run presentation.
Selected corrections passed live checks, and on 2026-09-24 the user marked the
combined 2E regression gate and normal-use soak tested and settled. Milestone
2F has therefore started. Slice 2F.1's focused authentication smoke passed on
2026-09-24. Slice 2F.2 is deployed and makes the native application
operationally authoritative: the OWUI bearer adapter is disabled, its startup
state is live-confirmed, and a fresh auth-cutover smoke plus a direct laptop
fast eval passed. The standalone-proxy smoke did not execute because its old
env file was not Docker-compatible; the on-box fast eval completed with one
failed structural check whose diagnostic was hidden by the old wrapper. The
proxy, browser, on-box-eval, and confirmed stop-OWUI evidence remains open.
Historical chat import is optional and runs only after a new explicit request.

Campaign 3 first strengthens Audrey's platform boundaries and operational
contracts, then makes Audrey itself the application behind a native web client,
and finally adds the first general skills layer on that owned surface.


## Sequence

| Phase | Plan | Outcome | Entry gate | Status |
|---|---|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening-plan.md) | Strengthen data boundaries, request ownership, storage lifecycle, readiness, and runtime reproducibility | Campaign start | Complete |
| 02 | [Audrey application and web UI](phase-02-audrey-ui-plan.md) | Provider-neutral identity, Audrey-owned conversations and runs, a structured agent protocol, and a native browser client | Phase 01 completion gate | In progress (2A–2B, 2C.1–2C.3, and 2D.1–2D.4 verified; 2D.5 deployed; 2E settled; 2F.2 deployed with live proxy/browser gate partial) |
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
- Any source/config edit runs the full hermetic suite and changed-file ruff.
  Lesson-link sweeps run only when the user explicitly requests one.

## What comes after these plans

The remaining product backlog—OCR, broader audio/media ingestion,
ordinary-answer provenance, artifact download, and `/v1/responses`—stays
available for later Campaign 3 phases. It is intentionally not interleaved with
the safety foundations or the first skills rollout.
