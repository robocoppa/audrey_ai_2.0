# Campaign 3 — correctness foundations, Audrey UI, and reusable skills

**Status:** Campaign 3 Phases 1–11 are complete. Phase 12's revised My Files
placement and Phase 13 Slice 13A's Responses inline-image adapter are
laptop-complete and awaiting their targeted live gates.

The Milestone 2E parity build is laptop-complete across native memory
settings, owner-bound file and video upload and inspection, source and answer
presentation, image and direct chat attachments, safe retry, reload recovery,
empty-draft cleanup, saved source and tool summaries, durable attachment
presentation, answer and code copying, attachment-picker dismissal, and
contextual startup and recovered-run presentation. The follow-up model activity
disclosure passed live on 2026-10-01: new native answers show every concrete
generation model and call count immediately left of Tool calls, and the menu
survives refresh.
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
The initial standalone-UI soak is complete, and the stopped
`audrey-ai-retired` pre-cutover container has been removed.

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

Phase 5 completed on 2026-09-30. Slice 5A's completed plain-text
`POST /v1/responses` adapter and Slice 5B's typed Responses SSE both passed
their laptop and targeted live gates. Background work, stored response
chaining, client tools, and structured output remain explicit later slices.
Phase 13 Slice 13A now adds typed text and bounded inline image input through
the same completed and streaming generation paths.

Phase 14 started on 2026-10-02. Slice 14A adds a protected Bots access group,
Bot-targeted model policy, zero-day never-expiring personal tokens, and
uncached administration lists so reopening the panel shows new applicants.
The implementation and full backend gate pass; native acceptance is pending.

Phase 6 completed on 2026-09-30. Slice 6A's probe-only comparison finished
148 valid calls across Tev1 0.8B, Tev1 4B, Nimble, and the incumbent. On the 23
cases that reach Audrey's model router, `qwen3.5:4b` scored 23/23 and Nimble
22/23; both Tev1 sizes trailed further. Nimble also added a costly reasoning
route and escalation, loaded more slowly, and used over twice the resident
memory. No candidate cleared the gate, Slice 6B was not opened, and no
production classifier or config changed.

Phase 7 OCR completed on 2026-09-30. Slice 7A adds bounded English OCR for
image-only PDFs through the durable media-worker queue. The corrected live
fixture moved from Pending to Ready in 4.619 seconds, produced one indexed
chunk, returned page-marked recognized text, and cleaned up successfully. The
user also accepted real PDF upload, generated Summary and full Transcript
presentation, native reading, and grounded chat.

Phase 8 started on 2026-09-30. Slice 8A adds MP3 uploads as a first-class
audio kind while reusing the durable media queue, Whisper transcript path,
owner-scoped indexing, and fail-soft summaries. The native Files viewer offers
audio filtering plus Summary and Transcript tabs, and canonical chat
attachments persist audio through schema migration 16. The first synthetic
live attempt exposed a faster-whisper/PyAV decoder mismatch; the worker now
pins the compatible PyAV major and verifies WAV decoding during its image
build. The real MP3 browser flow passed. Slice 8B adds measured WAV, M4A,
and FLAC admission on the same pipeline. The user marked the WAV, M4A, and FLAC
native gate passed on 2026-10-01, including the M4A Files/chat flow. Phase 8 is
complete.

A 2026-10-01 model-maintenance slice removes `qwen3.5:397b-cloud` after its
confirmed HTTP 410 retirement. The tag is gone from the general registry,
tool-capable and passthrough lists, cloud-panel workers, and the pull script.
The already registered `kimi-k2.6:cloud` now fills its cloud-only reasoning,
general, and vision draft slots, preserving two distinct workers and the
separate `deepseek-v4-pro:cloud` synthesis fallback. A real-config invariant
prevents the retired tag from returning to either runtime configuration or the
pull list. The required full gate also exposed an uptime-dependent progress
reporter bug: a fresh host could suppress media-fetcher's first update because
`0.0` doubled as its never-sent timestamp. The reporter now uses an explicit
unset state, so its first update is unconditional and later updates remain
throttled. All 3,065 backend tests passed. The deployed native Cloud turn
then completed with `glm-5.3:cloud` and `kimi-k2.6:cloud`, without the retired
Qwen tag, and the on-box model inventory returned clean. This maintenance slice
is live-settled.

Campaign 3 first strengthens Audrey's platform boundaries and operational
contracts, then makes Audrey itself the application behind a native web client,
and finally adds the first general skills layer on that owned surface.


## Sequence

| Phase | Plan | Outcome | Entry gate | Status |
|---|---|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening-plan.md) | Strengthen data boundaries, request ownership, storage lifecycle, readiness, and runtime reproducibility | Campaign start | Complete |
| 02 | [Audrey application and web UI](phase-02-audrey-ui-plan.md) | Provider-neutral identity, Audrey-owned conversations and runs, a structured agent protocol, and a native browser client | Phase 01 completion gate | Complete (all native cutover, parity, administration, recovery, and DNS gates live-passed) |
| 03 | [Reusable skills](phase-03-skills-capability-plan.md) | Local versioned instruction/resource bundles, native explicit selection, enforced tool narrowing, and an evidence-gated automatic selector | Phase 02 completion gate | Complete (3A–3C live-settled; explicit grounded-document pilot ships; automatic selection deferred) |
| 04 | [File and artifact downloads](phase-04-file-downloads.md) | Owner-scoped recovery of stored originals and useful derived artifacts from the native Files surface | Phase 03 explicit-skill ship decision | Complete |
| 05 | [Responses API compatibility](phase-05-responses-api.md) | OpenAI Responses clients use Audrey's existing authenticated generation and policy boundaries | Phase 04 completion | Complete (5A and 5B live-settled) |
| 06 | [System One decision routing](phase-06-system-one-routing.md) | Measure purpose-built local decision models against Audrey's incumbent router and retain a one-setting rollback | Phase 05 Slice 5B gate | Complete (6A measured; incumbent retained; 6B not opened) |
| 07 | [Scanned PDF OCR](phase-07-scanned-pdf-ocr.md) | Queue image-only PDFs for bounded owner-scoped OCR, indexing, and native reading | Phase 06 decision | Complete (OCR and PDF presentation live-settled) |
| 08 | [Audio ingestion](phase-08-audio-ingestion.md) | Transcribe, summarize, search, inspect, and attach spoken audio as a first-class file kind | Phase 07 completion | Complete (8A and 8B live-passed) |
| 09 | [Broader spoken audio](phase-09-broader-audio.md) | Admit measured OGG, Opus, and raw AAC containers before separately scoped speaker or media analysis | Phase 08 and Phase 2D.5 completion | Complete (9A live-passed) |
| 10 | [Ordinary-answer provenance](phase-10-ordinary-answer-provenance.md) | Persist deterministic public URL and private file evidence on ordinary tool-backed answers | Phase 09 completion | Complete (10A live-passed) |
| 11 | [Native file explorer](phase-11-native-file-explorer.md) | Browse, attach, and manage private files through compact folder-style native views | Phase 10 completion | Complete (11A user-closed) |
| 12 | [Sidebar navigation](phase-12-sidebar-navigation.md) | Simplify primary conversation creation and move My Files into a distinct top-bar action | Phase 11 completion | Revised Slice 12A laptop-complete; native browser gate pending |
| 13 | [Responses multimodal input](phase-13-responses-multimodal-input.md) | Accept typed text and bounded inline image parts through the shared Responses adapter | Phase 12 implementation | Slice 13A laptop-complete; targeted live gate pending |
| 14 | [Bot accounts and token lifetimes](phase-14-bot-accounts-and-token-lifetimes.md) | Add limited automation accounts, intentional permanent PATs, and fresh admin data on every open | Phase 13 implementation | Slice 14A laptop-complete; native acceptance pending |

Phase numbers repeat across campaigns. Refer to these as Campaign 3 Phase 1
through Campaign 3 Phase 14, or use the topic filenames.

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

## Remaining work

Items 5–10, their execution order, and pass criteria are tracked in
[remaining-work-todo.md](remaining-work-todo.md). Items 5–6, legacy Phase
2D.5, and Phases 9–11 are closed. The user reprioritized Phase 12's sidebar
navigation slice ahead of Item 7. Its requested top-bar correction is ready
for browser acceptance, and Item 7's first multimodal slice is ready for its
targeted live gate. Phase 14's account administration slice is also
laptop-complete and awaits its native browser and PAT checks.
