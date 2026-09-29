# Campaign 3 — correctness foundations, Audrey UI, and reusable skills

**Status:** Campaign 3 Phases 1, 3, and 4 are complete. Phase 2's native
product cutover is live-settled, while its legacy 2D.5 administration, role,
model-publication, exact-owner bootstrap, and provider-binding soak remains
open. Phase 5 has started; Slice 5A's completed plain-text
`POST /v1/responses` adapter is live-settled after its targeted WARP smoke
passed.

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
fast eval passed. The standalone-proxy native UI smoke also passed, covering
the built assets and CSP, two-user isolation, a streamed Fast turn,
conversation management, cleanup, and repair readiness. The corrected on-box
`fast-capital` eval now passes 1/1 with its `Canberra` content contract.
Slice 2F.3 is live-passed: native runs no longer apply OWUI's `### Task:`
utility routing, `/v1` compatibility requests retain it, and the targeted
public native Deep turn passed while Open WebUI remained stopped. Slice 2F.4
is laptop-complete: the backend no longer builds, packages, configures, or
serves an embedded browser shell. `audrey-ui` is the sole browser surface.
The first live deploy exposed that Cloudflare still targeted backend port 8000;
after switching to 8090, the long-lived UI proxy retained the recreated
backend's old Docker address. Restarting `audrey-ui` restored authentication,
confirming the diagnosis. The request-time Docker DNS correction is
live-settled; its rendered resolver, variable upstream, and both variable proxy directives passed the deploy proof.
The already-passed auth, proxy, routing, and eval checks are not repeated.
Historical chat import is optional and runs only after a new explicit request.

Phase 3 completed on 2026-09-28. Milestones 3A.1, 3A.2, and 3B are
live-settled. The targeted 3B native Fast run selected and persisted
`video-analysis` v1 successfully and completed cleanup. For 3C, controls
passed 9/9; the repaired `grounded-document-analysis` skill then passed 9/9,
and human review accepted all answers. It ships as an explicit opt-in skill.
Automatic selection remains deferred because selector precision and false
activation were outside this evaluation.

Phase 4 completed on 2026-09-29. Slice 4A's owner-scoped, ranged
original-file downloads are live-settled. Slice 4B's read-only backend smoke
passed all three derived artifact types. The final native browser check passed
after Summary downloads were removed: summary text begins without the old
action row, while Transcript and Visual notes retain download actions.

Phase 5 started on 2026-09-29. Slice 5A adds completed plain-text
`POST /v1/responses` generation by adapting onto Audrey's existing
authenticated Chat Completions path. Streaming, background work, stored
response chaining, client tools, structured output, and multimodal input remain
explicit later slices.

Campaign 3 first strengthens Audrey's platform boundaries and operational
contracts, then makes Audrey itself the application behind a native web client,
and finally adds the first general skills layer on that owned surface.


## Sequence

| Phase | Plan | Outcome | Entry gate | Status |
|---|---|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening-plan.md) | Strengthen data boundaries, request ownership, storage lifecycle, readiness, and runtime reproducibility | Campaign start | Complete |
| 02 | [Audrey application and web UI](phase-02-audrey-ui-plan.md) | Provider-neutral identity, Audrey-owned conversations and runs, a structured agent protocol, and a native browser client | Phase 01 completion gate | In progress (2A–2B, 2C.1–2C.3, and 2D.1–2D.4 verified; 2D.5 deployed; 2E settled; 2F.3 live-passed; 2F.4 and permanent DNS correction live-settled) |
| 03 | [Reusable skills](phase-03-skills-capability-plan.md) | Local versioned instruction/resource bundles, native explicit selection, enforced tool narrowing, and an evidence-gated automatic selector | Phase 02 completion gate | Complete (3A–3C live-settled; explicit grounded-document pilot ships; automatic selection deferred) |
| 04 | [File and artifact downloads](phase-04-file-downloads.md) | Owner-scoped recovery of stored originals and useful derived artifacts from the native Files surface | Phase 03 explicit-skill ship decision | Complete |
| 05 | [Responses API compatibility](phase-05-responses-api.md) | OpenAI Responses clients use Audrey's existing authenticated generation and policy boundaries | Phase 04 completion | In progress (5A live-settled) |

Phase numbers repeat across campaigns. Refer to these as Campaign 3 Phase 1
through Campaign 3 Phase 5, or use the topic filenames.

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

The remaining product backlog - OCR, broader audio/media ingestion, and
ordinary-answer provenance - stays available for later Campaign 3 phases.
