# Maintainer course structure reference

This is the approved course design, retained after campaign closure on October
27, 2026. It is a reference for future user-requested lesson work, not an active
implementation queue. Unwritten lessons remain unwritten. The
[published course index](../lesson-ai/README.md) identifies the existing lessons;
its current filenames remain in use until the migrations below are performed.

Rewrite a lesson when the code replaced its subsystem. Add a lesson for a new
subsystem at the first point its prerequisites have been taught. Teach each
concept once: the opening system map orients the reader; later lessons supply
its detail. The Python course remains separate and resumes at the user's request.

## The course structure (revised 2026-10-23)

The approved design has 27 lessons in seven parts. Numbers here are final
positions, not a claim that corresponding files have been written or renamed.
“Current” numbers refer to the existing files listed in the migration table.

| Final | Lesson | Content source |
|---|---|---|
| | **Part 1 — Foundations** | |
| 0 | Introduction | Current 0 |
| 1 | Python language features | Current 1 |
| 2 | The libraries | Current 2 plus parts of 3 |
| | **Part 2 — The platform** | |
| 3 | The platform and the request lifecycle | Current 4 |
| 4 | Configuration and startup | Current 5 |
| 5 | The application database | New lesson |
| 6 | Identity and access | New lesson; model access from current 8 |
| | **Part 3 — The answer pipeline** | |
| 7 | Classification and routing | Current 8 |
| 8 | The model layer I: choosing a model | Current 6 |
| 9 | The model layer II: calling models | Current 7 |
| 10 | Fair scheduling | Current 15 |
| 11 | Deep mode and research mode | Current 9 |
| 12 | Tools: function calling and the ReAct loop | Current 10 plus parts of 11 |
| | **Part 4 — Knowledge, memory and the tool server** | |
| 13 | The knowledge base | Current 12 plus parts of 3 and 13 |
| 14 | Memory and history | Current 14 |
| 15 | The custom-tools sidecar | Current 17 |
| | **Part 5 — The native application** | |
| 16 | Streaming: run events and wire formats | New lesson; streaming from current 16 |
| 17 | The native run | New lesson, including the browser boundary |
| 18 | Durable side effects | New lesson |
| | **Part 6 — Files and media** | |
| 19 | Files as durable objects | New lesson; uploads from current 13 |
| 20 | Processing media: jobs, leases and the worker | New lesson |
| 21 | Reaching outside safely | New lesson |
| 22 | Asking about your files | New lesson |
| | **Part 7 — Shaping a turn, the bot API, and operating** | |
| 23 | Skills and Projects | New lesson |
| 24 | The bot API: Chat Completions and Responses | New lesson; routes from current 16 |
| 25 | Responses: caller tools, files and stored responses | New lesson; caller tools from current 11 |
| 26 | Operating Audrey | New lesson; Prometheus from current 3 |

Routing precedes model selection; identity and storage precede user-scoped
features. The bot API stays late because Responses depends on files and safe
outbound requests. The platform lesson introduces `/v1` without duplicating
its later protocol lessons.

## Existing-file migration

Paths below are relative to `docs/lesson-ai/`. Splits identify where existing
content belongs; they do not authorize new writing or immediate file removal.

| Existing file | Final destination and content boundary |
|---|---|
| `lesson-00-introduction.md` | 0: Introduction and seven-part map |
| `lesson-01-foundations.md` | 1: Python language features |
| `lesson-02-foundations-libraries.md` | 2: Main libraries |
| `lesson-03-foundations-satellites.md` | 2: httpx and brief test-suite context; 13: embeddings/Qdrant; 26: Prometheus; then retire source |
| `lesson-04-request-lifecycle.md` | 3: Platform map; shorten its detailed native trace when 17 exists |
| `lesson-05-configuration-and-startup.md` | 4: Configuration/startup; introduce store initialization alongside 5 |
| `lesson-06a-application-database.md` | 5: The application database; published early as 6a (2026-10-28), read between 5 and 6 |
| `lesson-06b-the-model-layer.md` | 8: Choosing a model |
| `lesson-07-calling-models.md` | 9: Calling models, including a high-level Private explanation |
| `lesson-08-classification-and-routing.md` | 7: Routing; move model-access detail to 6 |
| `lesson-09-deep-mode.md` | 11: Deep and research together |
| `lesson-10-tool-use-and-react.md` | 12: Tools and the ReAct loop |
| `lesson-11-how-function-calling-works.md` | 12: Protocol essentials; 25: Caller-executed tools; cut vendor tour; then retire source |
| `lesson-12-kb-ingest-and-search.md` | 13: Hybrid retrieval, ingest and search |
| `lesson-13-kb-lifecycle.md` | 13: Watcher/reconcile; 19: Uploads; then retire source |
| `lesson-14-memory-and-context-injection.md` | 14: Memory/history; identity becomes a recap of 6; delivery detail belongs in 18 |
| `lesson-15-fair-scheduling.md` | 10: Fair scheduling |
| `lesson-16-openai-routes.md` | 16: Streaming internals; 24: API routes; then retire source |
| `lesson-17-custom-tools-sidecar.md` | 15: Sidecar; archive delivery detail belongs in 18, scoped file tools in 22 |

Two scripted renumbers implement this design when lesson work resumes:

1. **Identity joins the database lesson.** The database lesson is already
   published as 6a. Review the identity draft, insert both after configuration
   as 6 and 7, and put the pipeline in request order. Keep the four
   source lessons awaiting merges beside their destination area. Interim order:
   0 introduction; 1 Python; 2 libraries; 3 satellite libraries; 4 lifecycle;
   5 configuration; 6 database; 7 identity; 8 routing; 9–10 model layer;
   11 scheduling; 12 deep/research; 13 tools; 14 function calling; 15 knowledge
   base; 16 KB lifecycle; 17 memory/history; 18 sidecar; 19 API routes.
2. **Final consolidation.** Retire current source lessons 3, 11, 13 and 16 only
   after their remaining content is taught elsewhere; then assign final 0–26.

Between renumbers, new files use the next free number. Move content out of its
source in the same lesson unit that absorbs it, avoiding duplicate teaching.
Each renumber updates footers, the course index, Lesson 0's map, this reference
and local continuity files. Add an old-to-new mapping to the local AUDIT;
historical campaign records keep their original numbering.

## Teaching contracts for future lesson work

These are content boundaries and prerequisites, not a writing schedule.
Numbers below use the final design. Recheck the current source before writing.

- **5 — Database** (after 3–4): Teach SQL transactions and WAL from scratch;
  canonical rows, typed repositories, additive migrations, ownership, backups
  and verified restores. Distinguish rebuildable projections from saved memories
  whose authority lives in Qdrant. Read `src/audrey/app_state/` and projection/CLI code.
- **6 — Identity** (after 5): Authentication versus authorization; signed Access
  assertions, stable principal/namespace, account roles/groups, hashed personal
  tokens/scopes/revocation, service callers, model-access enforcement and bootstrap.
- **10, 12–15 — Refreshes and merges:** Explain owned queue grants and cancellation
  after grant; tool schemas/messages, choice, parallel calls and failures; hybrid
  dense/BM25 retrieval, fusion, evidence rules, pinned embeddings and caller scope.
  Fold in watcher/reconcile; keep uploads in 19 and durable archive delivery in 18.
- **16 — Streaming** (after 11–12): Typed sequenced run events feed AG-UI,
  Chat Completions and Responses adapters. Teach SSE, shared terminal outcomes,
  owned-task cancellation and safe observations without parsing display text.
- **17 — Native run** (after 5–6 and 16): Persist before streaming; one active run,
  server ownership, Stop, restart settlement, bounded replay with durable fallback,
  and retry. Include a TypeScript reading primer and typed browser boundary;
  the browser renders server state. React internals stay outside the course.
- **18 — Durable effects** (after 5, 14–15 and 17): Transactional intent/outboxes,
  at-least-once delivery with idempotency, archive repair, immediate tombstones,
  export/correction/deletion, account purge and its process-local privacy gate.
- **19 — Files** (after 13 and 18): Chunk assembly, sniffed admission, atomic
  quota reservations, status ownership, operation locks and durable deletion;
  byte-range downloads, foreign/missing 404, reclaimed-original 410 and attachments.
- **20 — Media** (after 9–10, 13 and 19): Claims, leases, stale-worker fencing,
  expiry/retry accounting and service routes; probing, transcription, frame gates,
  brokered model calls, bounded OCR/summaries, searchable artifacts and measured cost.
- **21 — Safe outbound requests** (after 15 and 20): SSRF, redirects, DNS changes,
  vetted-address socket pinning, response bounds and isolated parsing; distinguish
  fetcher egress from worker isolation and prefer existing video captions.
- **22 — File questions** (after 12–13 and 19–20): Shared retrieval scope,
  filename notices, per-file coverage, deletion exclusion, file listing/text paging,
  declared user bindings and video specialization. Fusion itself stays in 13.
- **23 — Skills/Projects** (after 7, 12, 17 and 22): Bounded read-only bundles,
  digests, atomic registry/degradation and conservative English intent selection;
  instruction injection outside complexity decisions; intersect discovered tools,
  platform policy and skill allowance. Projects snapshot instructions/Ready references,
  preserve prompt order/ownership and treat filenames/passages as untrusted evidence.
  Direct models receive bounded text context without tool execution.
- **24 — Bot API** (after 6, 12 and 16, with files/outbound concepts available):
  Shared dispatch/generation, request admission before side effects, typed outcomes,
  passthrough, JSON Schema validation and sequenced Responses terminal events.
- **25 — Responses detail** (after 5, 19, 21 and 24): Caller tools on both APIs,
  stateless replay, owned files/bounded URLs, opt-in storage/continuation and supported
  reasoning controls. Stored continuation saves bookkeeping, not tokens.
- **26 — Operating** (after the preceding parts): Metrics and label cardinality,
  capability degradation, sanitized readiness, queues/workers/gate pressure, exactly
  one provider terminal outcome, reported usage versus unknown, dashboards and wrap-up.

Possible later additions remain outside the 27-lesson map: full Private processing
(policy/runtime, local tools, inherited restrictions and task admission; after 23),
and reference libraries/tax review (pinned indexing, cases, facts, worksheets and
verification; before the closing lesson). Neither is an active assignment.

## Lesson process

Follow [AGENTS.md — Lesson workflow](../../AGENTS.md#lesson-workflow) for one
user-requested lesson or refresh unit at a time: agree scope; audit; resolve that
unit's findings with approved code changes; obtain outline approval; write and
refresh paired sections; run scoped checks; update course links; hand over for review.
Findings and drain decisions belong only in the gitignored
[course AUDIT](../lesson-ai/AUDIT.md), never in tracked plans or lesson prose.

Match the latest lesson's teaching style, define concepts before using them,
include no labs/test walkthroughs, and keep comprehension questions to five.
Production behavior is the subject; campaign history, eval/probe tooling and
container operations beyond necessary trust boundaries stay outside the course.
