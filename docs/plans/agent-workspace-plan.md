# Audrey agent workspace, search, project knowledge and monitoring

**Status:** In progress, November 3, 2026. Saved task Agents, unified lexical
Search, Project Knowledge and public-page Watches with Project links are implemented locally. The owner confirms Search
returned expected matches; full acceptance remains pending. Campaigns 1–4 remain closed.

The owner selected unified search, saved background tasks, project working
memory and scheduled monitoring. Background tasks mean a visible agentic
architecture: Agents should be a prominent part of Audrey's native UI.
Backups and disaster recovery are outside this plan. Agent restart recovery is
part of the requested background execution capability.

## Current implementation slice

The native app has centered Chat, Agents and text-only Search buttons in the top
bar. Search toggles a right sidebar with its own
query, filters, results and source reader, while preserving the active workspace
and drafts. The panel shares Chat's 17rem (272px) sidebar width. At 1,280 pixels and above Search
is docked beside the workspace; smaller screens use an accessible overlay drawer.
Two small right-pointing chevrons collapse the panel. Collapse retains search
state, aborts hidden reads and revalidates the saved query
on reopening. Source text opens inside the panel with paging and Back to results;
Project Knowledge's separate source previews retain their modal behavior.
It has a saved task
workflow: Plan, Work and Review. Users can create tasks, inspect
saved step results/reports, pause/resume/cancel, rerun the same agent and delete
terminal history. Run again preserves completed checkpoints on unfinished tasks;
completed tasks repeat Work/Review using the saved Plan. The agent ID and
selection stay the same. Lists return compact summaries and load the selected report separately.
Agents has a left sidebar for creation, history and active controls; its workflow
and sidebar scroll independently. Small screens use a controls drawer. Terminal
Run again and the red two-click Delete remain at the workflow card's top right.
Top-bar hover uses the filled, soft glow of My Files, with distinct keyboard focus
retained. Agent headings/history use a saved
short title, and the unchanged original objective is available in an expandable
Original prompt section.

Application schema 26 introduced owner-bound tasks and operation reservations with
leases, fenced writes and cumulative limits. The lifecycle worker reuses committed
steps after interruption; uncommitted model/read work can repeat within the saved
allowance. Every provider/tool dispatch revalidates live task/account/model access.
Agents offer web search/fetch when needed and native downloadable file creation
during Work, as described below. Users can explicitly select ordinary Project
files as bounded local inputs and read further pages during Work when needed.
Automatic personal-memory recall and other write
tools remain pending. Every model step uses only verified local
`qwen3.8:latest`, with no classifier, cloud panel or fallback. The worker verifies
installed local weights before generation and pins their digest within each step;
missing or changed metadata stops execution. Public web tools still fetch source
material. Model calls share existing admission and generation limits. Account
purge removes the new task data and receipts.
Schema 28 adds persistent agent titles. One bounded local Qwen title call runs
before the first incomplete step, using the same verified client, model admission,
lease and cumulative operation allowance. A failed title leaves a short fallback
and does not block task execution. Rerun keeps a generated title; untitled legacy agents
receive a generation attempt on their next explicitly started run. The title call
has a 64-token allowance, thinking disabled and a ten-second timeout; it uses no
cloud fallback and does not alter the objective or saved Plan.

Model failures identify the actual stage and distinguish output limits from empty
report text. Per-call logs retain only stop/usage counts and text lengths;
report and reasoning bodies are excluded. Truncated tool calls stop before dispatch.
Work and Review allow up to 8,192 generated tokens per call, including model
reasoning. Plan has a separate 2,048-token cap, bounded by the overall call limit.
This raises the original 2,048-token report allowance after a reported Research
output-limit failure; live acceptance of the larger allowance remains pending.

This slice uses a fixed three-stage workflow and the existing bounded ReAct loop.
Adaptive multi-step plans, file-backed Knowledge context for agents, Private agent inputs,
semantic search,
owned-file/Private and multiple-source watches remain implementation work. Existing Private Tax tasks keep
their current behavior. The remaining sections specify the intended full feature.

The first unified Search slice searches native completed/incomplete user and
assistant message bodies, saved agent reports and locally readable file text.
Supported file sources include documents, generated output text, cached image OCR,
media transcripts, generated visual notes and summaries. Existing retained text
works when original bytes are missing; Search labels that state explicitly.
Private files use their saved local text without model or remote processing.

Search has source, Project, chat archive and inclusive UTC update-date filters.
Results contain canonical IDs/revisions, plain-text snippets and Unicode character
highlights. Opening a match reads a fresh bounded text window from the canonical
message, report or file artifact; it does not infer PDF pages or media timestamps.
The viewer pages through that source and rejects deleted or changed revisions.

A dedicated local SQLite FTS5 projection uses bounded owner-scoped backfill and
periodic reconciliation. Unchanged sources avoid index rewrites; ownership,
deletion, Project membership and source/stat pins are checked after awaited reads
and immediately before returning text. Account purge clears projected text, and
pagination is bound to owner, query, filters and index generation. Search uses no
models, embeddings, sidecar history/memory or cloud calls. Unsupported/oversized
sources and incomplete backfill produce an explicit partial-results state.
Coverage notices now count genuine unreadable/oversized sources within the selected
filters, exclude unfinished/failed uploads, and avoid duplicate generic warnings.
Indexing has its own status; multiple coverage details can be expanded.
Current bounds include 256 query characters/12 literal terms, 500,000 characters
per source, 5,000 sources/10 million source characters per owner, and the first
200 matching source candidates per query. File input/expanded parsing is bounded
to 8 MiB; existing cached text can cover larger originals. Approved Project notes
are now searchable, with source-review labels when references become stale. Semantic
candidates remain later work. Canonical source previews now link into their owned
conversation, file, agent or Project workspace, as described below.
Local tests and mocked responsive browser checks verify mechanics;
real deployed file/content acceptance remains manual.
The Search sidebar's affected tests, production build and mocked browser checks at
1,440, 1,100, 768, 390 and 320 pixels pass, including focus, independent scrolling,
source revisions and retained drafts. The Search layout alone needs only the UI rebuild.

Project Knowledge now stores user-authored Facts, Decisions, Questions and Actions
in application schema 27, with immutable revisions, pinning, open/completed/resolved
states and optimistic concurrency. Project home supports editing, history, explicit
approval/rejection and deletion. A completed Agent report can open a proposal editor:
the user writes a statement or explicitly generates local-model suggestions, then
chooses the statement and Project; saving creates a pending report-linked proposal.
Neither generation nor saving approves a proposal automatically. Replacement
approval checks the original revision and supersedes it atomically. Approval records
what the user accepts, rather than independent verification.

Native Project conversations select bounded approved notes (at most 12 entries and
1,000 content tokens within the existing 3,000-token Project source allowance), with
pinned/relevant/open-work priority. Notes are quoted data below platform guidance;
pending, rejected, superseded and stale-source entries cannot ground a new run.
Each native run saves only its selected entry IDs/revisions; generation rechecks the
frozen knowledge/source proofs and fails if they change. Private Project chat stays
inside its existing verified local policy. Public background Agents accept only
explicitly selected ordinary Project files, as described below; Private inputs
remain unsupported. Saved Private Tax tasks keep their established context/admission.

Source references contain canonical IDs/revisions only, with Project-scoped source
previews and local ownership/deletion/detachment checks. Explicit user notes and
report-linked user proposals retain their bodies with a source-review notice when
references disappear, while automatic grounding excludes them. Automatically extracted
source-derived notes, public-source snapshot references and automatic agent proposals
are later work. The source-reference editor is described below. Entry/Project deletion and account
purge erase the new rows/history/manifests and invalidate/remove search copies.
Limits are 200 entries per Project, 8,000 characters per entry, eight references and
100 revisions per entry. Context selection, source browsing and approval use no
models; the explicit report helper is described below. There are no global-memory writes or cloud
summaries. Local tests and mocked browser flows pass; real-model reuse
and deployed acceptance remain manual.

Public-page Watches now appear beside Tasks and Activity inside Agents. Creating a
watch explicitly saves one public HTTP(S) URL, a criterion and a previewed interval,
daily or weekly schedule in an IANA timezone. Query parameters, URL credentials,
fragments, nonstandard ports and local destinations are excluded. Saving schedules
the first future check; Run now is separate and can check a paused watch without
resuming its schedule. A fresh deployment has no watch definitions and starts no page checks.

Schema 29 stores watch definitions, immutable run inputs/source snapshots, baselines,
operation receipts and in-app notices. The scheduler coalesces missed occurrences,
runs repeated DST times once and moves nonexistent wall clocks to the first valid
minute. Leases, revision fences, cumulative allowances and capped retry/backoff cover
restarts; agents and watches share the one-active-task-per-owner boundary.
Pause/edit/delete revoke earlier work. Owner purge removes all new derivatives.

The guarded fetcher authorizes every network hop and rejects incomplete/oversized
responses. Versioned deterministic extraction establishes the first baseline and
skips model calls for unchanged text. Changed text has exact Unicode-offset evidence
and a separate bounded significance assessment by verified local Qwen, without tools
or cloud fallback. Failed fetches, extraction, comparison or model assessment retain
the last valid baseline. The current bounds are 256 KiB downloaded, 64,000 UTF-8 bytes
of normalized text, and 16,000 bytes of complete changed-span evidence; larger work
fails visibly rather than silently truncating a comparison. Compatible criterion and
cadence edits retain the baseline; URL/redirect identity or extraction changes rebaseline.
History exposes each run's saved URL, criterion and schedule separately from the
current definition. Significant changes and persistent failures create deduplicated in-app Activity
notices linked to the exact saved check. Unchanged checks remain in history. There
are no external notifications. Owned-file/Private watches,
multiple-source definitions remain later work.
Affected hermetic checks and mocked desktop/phone browser flows pass; real public
page changes, scheduled runtime and model-judgment acceptance remain manual.

Schema 30 unifies agent completion/failure notices and existing watch alerts in
Agents' Activity inbox. Terminal agent state and its deduplicated notice commit
together; retries, pauses and cancellations remain silent. Notices contain a
bounded display title and fixed status text, without copying reports or provider
errors. Agent targets bind the task ID and current terminal revision; watch targets
bind the owned watch and exact retained run. Resolving an item validates its target
before marking it read and opening the existing task/check, preserving drafts.
Run again removes the previous agent notice atomically because it replaces the saved
report; this adds no report history or backfill for already terminal tasks.

The inbox lists up to 50 current items, unread first, with a full count across both
origins. Older unread items surface as newer ones are read. Badge polling fetches
counts only; hidden reads abort and polling/read acknowledgments are fenced against
stale UI updates. Owner, target, revision and deletion checks also run after awaited
reads. Task/account deletion cascades notices; existing watch retention and legacy
watch-only notification routes remain intact. This slice adds no model calls or
external delivery. Affected storage/API/UI tests, lint, production build and mocked
desktop/phone browser flows pass; deployed acceptance remains manual. Rebuild both
backend and UI for this slice; the backend applies schema 30 automatically.

Schema 31 adds durable planning questions. The same verified local planning call
can return one bounded question when an essential detail cannot reasonably be
inferred. A complete, validated request saves the question, releases the lease and
compute, and exposes Waiting for input plus a deduplicated Activity notice. No
Work or Review starts while it waits. The stored task remains paused internally;
the public waiting state comes from its pending question, avoiding a task-table
rewrite. Waiting time does not consume active execution allowance.

The owner answers inside the existing agent card. The answer transaction checks
question identity, current task revision, owner access, remaining cumulative limits
and queue capacity before queuing that same task. Dispatch revalidates model/tool
authority. Questions and answers survive restart, are supplied to subsequent steps,
and remain visible as saved history. Summary polls omit their bodies while retaining
the waiting state; the selected task loads them separately. Drafts survive tab/task
changes, and a changed revision requires review before using a retained draft.
Activity opens the exact waiting task and cannot reopen an answered/cancelled request.

This slice supports planning questions only, with at most three per task lifetime,
2,000 UTF-8 bytes per question and 8,000 per answer. It preserves original objectives,
completed Plans and all operation/attempt/time usage. Pending questions require an
answer or Cancel; generic Resume cannot bypass them. Run again after cancellation
dismisses an unanswered question and retains answered history. Task/account deletion
erases these records. Strict output/answer validation, idempotent queued-answer retries,
post-await ownership/deletion/revision fences and plain-text rendering cover stale or
invalid requests. No extra classifier, cloud model, external delivery or additional
tool authority was added. Rebuild backend and UI; schema 31 applies automatically.

Schema 32 links agents to ordinary Projects. The composer offers
an optional Project, and Project home lists its saved agents plus an explicit
New agent in Project action. Prefill preserves the objective and requires Start
agent. Saved task and Project links resolve current owned targets before navigating;
hidden or superseded reads abort, while mounted conversation and agent drafts stay intact.

Creation freezes up to 2,000 tokens of Project instructions and 1,000 text tokens
across at most 12 approved notes. Selection prioritizes
pinned, relevant and open work; newly added notes are not silently adopted. Current
message/report-backed notes are supported as described below; file-backed notes
and Private Projects remain outside automatic context. File excerpts require
the separate explicit selection described below.
Every step receives the selected context as quoted working data under platform
guidance. Approved notes do not verify truth or authorize new tools or actions.
All model calls remain verified local Qwen; only Work offers the available tools.

The additive manifest stores instruction hashes and note IDs/revisions/prefix lengths,
without copying input bodies or Project names. Canonical rows supply text afresh.
Ownership, context and deletion fences cover claims, dispatch, saved steps, questions,
completion, answers, Resume and Run again. Changed selected instructions/notes block
further work and require a new explicit agent; they cannot silently replace inputs or
replan the existing agent. A rename or unrelated file/note addition does not invalidate
the task. Cancel, deletion and saved-report access remain available.

Project deletion removes canonical context and leaves saved reports with an unavailable
Project label; task/account deletion removes the manifest. Agent reports participate
in the existing Search Project filter, with canonical revision fences when their
Project disappears. Existing agents remain unscoped after migration. Affected local
storage/runtime/API/UI checks, lint, TypeScript, build and mocked desktop/phone flows
pass; real model and deployed acceptance remain manual. Rebuild backend and UI;
schema 32 applies automatically with no configuration change or required upload.

Project home now puts conversations and files in the first row, with Knowledge and
Agents side by side below. Scoped spacing, smaller controls and bounded scrolling
keep all four panels within the checked laptop viewports. The available pane width
controls stacking on narrow screens. Full instructions and long lists remain scrollable;
Knowledge editing and the existing Private Tax section remain available.

Search's saved-source reader now offers explicit Open conversation/file/agent/Project
actions after a successful canonical read. Navigation reads the owned target and then
rechecks the original source revision and exact target identity. Deleted, detached or
replaced sources clear the displayed source and leave the reader open with an error.
Transient navigation failures retain the source for retry. Collapse, Back, owner reset and
superseding navigation abort pending links and ignore late replies.

Conversation links select and focus the matching saved message, including archived
Private conversations, while preserving mounted drafts and running threads. Retained empty
conversation drafts remain reachable from a compact Draft conversations list. Agent links
open the existing task without creating or rerunning it. File links select the current
owned file and supported matching text/artifact tab in My Files. Image OCR and retained
document text remain readable in Search when their native preview/original is unavailable;
this slice does not add missing bytes or expand file-reading endpoints. Knowledge links
open their owned Project. Search queries/filters and agent objectives survive navigation.
Local UI tests, lint, TypeScript/build and mocked laptop/phone browser checks pass;
deployed acceptance remains manual. This slice needs only an audrey-ui rebuild, with
no schema/configuration change or upload.

Shared local-model admission now distinguishes interactive work from Agent/Watch
generations. Interactive requests precede unaged background work in both the
per-user cap and GPU queue. Eligible GPU waiters retain per-user round robin;
older background work joins the interactive tier after the configured aging
threshold. Running generations finish without interruption, and cloud GPU bypass
and existing total concurrency caps remain unchanged.

Agent titles, Plan, each Work model call, Review and Watch significance checks
use the background path. Public tool reads hold neither queue. A background call
waiting for GPU can give its per-user permit back to later chat, including when
the per-user cap is one. Handoffs retain the original queue age, preventing chat
traffic from continually restarting the aging clock. Admission remains per-user
then GPU; cancellation returns queued or just-granted resources, and provider work
and its authority/usage scope stay in the original task. Title operations are
reserved after admission, avoiding charges for titles cancelled while queued.

The shared `fairness.background_aging_seconds` setting defaults to 30 seconds and
accepts positive finite numbers up to 300. It is a priority threshold, not a
completion deadline: occupied generations and earlier eligible work still finish.
Workload/stage wait histograms and aggregate GPU workload/aging counts expose
queue pressure without owner IDs or content in metric labels. Dispatch still
revalidates account/model/Project/lease authority after waiting; only verified
local Qwen is available to agents, with no new model or cloud fallback.
Affected hermetic transport, queue, cancellation, lifecycle/configuration and
readiness checks pass; live concurrent-chat/model acceptance remains manual.
Rebuild and recreate audrey for the configuration addition. No schema migration
or upload is required.

The admission slice now has a short reusable installed-backend smoke. The Tower
wrapper streams it into the running container, imports the installed scheduling
code and reads the bound configuration. Five isolated cases check configured
capacity, cap-one chat priority, synthetic GPU-age promotion, running-call completion
and cancellation cleanup, with bounded waits and explicit exit codes. It needs no
credentials, accounts, model calls or network access. The runner and negative
failure detection pass locally; the deployed invocation is documented in
[live testing](../reference/live-smoke-testing.md#installed-admission-smoke).
The owner reported a Tower pass for all five cases with exit 0: configured GPU
capacity 1, per-user capacity 3 and background aging 30 seconds. Real GPU/HTTP
contention acceptance remains unrun.

### Project public-page watches

Schema 33 optionally associates a public-page Watch with an owned ordinary Project.
Project home lists its watches in a compact row below Knowledge/Agents. New watch
in Project opens the existing Watch composer with only the Project preselected;
the user still previews the schedule and presses Create watch. Existing links open
the same Watch even before its first check. Current definitions and saved checks
offer canonical Project navigation, while chat and unfinished Watch drafts remain
reachable. Private Projects are excluded.

This association organizes the public page; Project instructions, notes and files
are not check inputs. Each check records its original Project ID in its immutable
definition; names and availability come from fresh owned Project metadata. Edits
may reassign or explicitly unlink a Watch, cancelling old queued/running work.
An organizational change retains a URL-compatible baseline. Legacy unscoped
definitions, checks and creation retries remain compatible.

Project deletion or unavailability blocks dispatch, heartbeat and result commits
immediately. The next scheduler sweep pauses linked definitions and cancels
outstanding checks. Saved history stays readable; Pause/Edit/Delete and explicit
unlink remain available. Run now/Resume require an available Project. Synchronous
checks after awaited reads fence stale metadata and the final socket boundary.

Affected backend/UI tests, migration replay, model-response deletion boundaries
and responsive mocked browser checks verify this slice locally. The paused-watch
[native smoke](../reference/live-smoke-testing.md#project-watch-api-smoke) exercises
deployed Project/Watch API wiring without page fetches or model generation. The
owner reported a Tower pass for all six checks and disposable cleanup, exit 0:
paused creation/idempotency, Project filtering/canonical rename, unlink/reassignment,
deleted-Project Run/Resume rejection and unlink recovery. Actual scheduled page
comparisons and local-model acceptance remain unrun. Deployment uses backend/UI
rebuilds and automatic schema 33; no configuration change or upload is required.

### Knowledge source-reference editing

Knowledge's Add/Edit forms now let users browse saved Project messages (including
archived conversations), attached file text/artifacts and their owned Agent reports.
Canonical metadata listings are independent of Search backfill and never extract
file bodies. Signed keyset cursors bind the owner, namespace, Project version,
privacy and source type. Message labels include their role and sequence number.

The user previews local text before explicitly attaching its ID/revision. Preview
pages pin that revision; changed, detached or deleted sources cannot return stale
text. Missing originals can still offer retained text with the existing label.
Private sources remain inside an owned Private Project and use saved local text.
No models or external fetches are involved.

References can be removed or explicitly replaced with a reviewed current revision,
up to eight per entry. Text and reference drafts survive conflicts and rebasing;
text-only saves leave references alone. Revision history exposes saved reference
metadata without restoring historical source bodies. These remain user-authored
notes or user-reviewed report proposals, with existing stale-grounding restrictions.

Backend/UI tests and responsive mocked browser checks cover source editing and
its deletion/ownership boundaries. The reusable installed-backend
[source smoke](../reference/live-smoke-testing.md#installed-knowledge-source-smoke)
uses separate temporary SQLite/upload fixtures and an isolated ASGI application;
it needs no credentials, network, model or production-account writes. The local
smoke and its failure-detection harness pass. The owner reported all five installed
Tower checks passing with exit 0, including isolated fixture cleanup. Native-browser
acceptance with real sources remains pending. Deploy with
backend/UI rebuilds; no schema or configuration change is required.

### On-demand report suggestions

The report proposal dialog offers Generate suggestions for a completed owned
Agent report. One verified local `qwen3.8:latest` call returns up to three typed
suggestions, with no tools, classifier, cloud fallback or automatic retry. The
helper reads only that saved report; it does not send Project instructions,
notes, files or the browser's proposed text to the model. Report text is quoted
untrusted input, and suggestions preserve the report's uncertainty.

Suggestions are transient: generating them does not save Knowledge, change an
Agent/Plan/checkpoint, charge the completed task's saved allowance or create a
job. Use suggestion fills the editable proposal; replacing existing text requires
an explicit Replace draft action. Save proposal uses the existing report-revision
and idempotency guards, creates a pending entry, and still requires separate
approval before reuse. Generation and saving never start or complete an action.

The helper admits one request per account through shared background per-user/GPU
queues, with a 60-second total deadline. A complete prompt including schema must
fit 6,000 tokens; oversized reports remain available for manual proposals rather
than being silently truncated. Output is capped at 1,024 tokens (or the smaller
configured Agent call limit), three suggestions and 1,200 characters per statement.
Invalid, duplicate, incomplete, truncated or tool-bearing responses fail as a whole.

Live owner/model access and the exact completed report revision are checked after
waiting, before generation and before returning suggestions, including the route's
final synchronous response fence. Browser cancellation/disconnection releases
queued resources and closes the model client. Reopening or observing a report
rerun/deletion discards old suggestions. A deleted Project does not remove its
separately retained report or authorize new Project reads.

Hermetic transport/API/UI checks and mocked responsive browser acceptance verify
this slice. Its installed-backend [draft smoke](../reference/live-smoke-testing.md#installed-report-suggestion-smoke)
uses temporary canonical data and a mocked local provider, with no production
accounts, network or real model generation. The local six-case smoke and its
negative/cancellation harness pass. The owner reported all six installed Tower
checks and cleanup passing with exit 0. Real local-Qwen suggestion quality
remains pending. Deploy with backend/UI rebuilds; no schema
or configuration change is required.

### Message/report-backed Agent knowledge

Ordinary Project agents can now select approved notes whose references all point
to current ordinary messages in that Project or completed owned Agent reports.
This includes an explicitly saved report proposal after separate approval. Pending,
rejected, superseded, stale, foreign, Private, file-backed and mixed unsupported
references are excluded. Unattached user notes remain eligible. Report references
identify generated provenance; approval and a live reference do not verify truth.

Selection keeps the existing pinned/relevant/open-work priorities and limits of
12 notes and 1,000 note tokens, alongside 2,000 instruction tokens. It examines at
most 12 source-backed candidates, checking up to eight references each; unsupported
reference kinds are filtered before that work. Later unattached notes can still
fill the remaining allowance. Repeated references share a capture-local metadata
cache. The model receives only the selected note text, never linked message/report
bodies, reference metadata or file content.

The existing schema32 manifest adds one aggregate source-metadata hash per selected
source-backed entry, within its existing 4,000-character bound. Canonical source
revisions are verified during capture; every later dispatch/commit fence compares
current ownership, scope, privacy, status and metadata against the frozen proof.
Shared SQLite row/digest helpers preserve Search's canonical revision identities.
No source body, title or preview is saved in the manifest, and proofs work after
restart without a resolver cache. Existing sourceless manifests remain compatible;
old agents do not silently adopt newly approved knowledge.

A changed or deleted selected message, detached conversation, or rerun/deleted
report invalidates the context and blocks further dispatch, checkpoints, questions,
completion, Resume and Run again. The agent must be recreated to use current
knowledge. Project renames, new turns and unrelated notes/files remain safe. Saved
reports and user-authored/reviewed notes retain their existing read/deletion behavior.

The focused [installed smoke](../reference/live-smoke-testing.md#installed-agent-project-knowledge-smoke)
uses temporary canonical data with no credentials, network, models or production
account access. Affected storage/executor/Knowledge/Search tests, UI checks and the
local six-case smoke pass. The owner reported all six installed Tower checks and
cleanup passing with exit 0. Deployment requires backend/UI rebuilds, without
schema/configuration changes or an upload. Real-model and deployed browser
acceptance remain separate.

### Assigning Project actions and questions to Agents

An approved open Action or Question in an ordinary Project can explicitly open
a task draft through Assign to agent. The browser rereads the canonical
Project and entry before prefilling the exact entry text and Project. Existing draft text
requires an explicit choice before replacement. Opening a draft never starts a
model or changes Knowledge; Start agent remains a separate action. Text is editable,
and the composer exposes the selected entry revision with an explicit unlink action.

Start supplies the entry identity/revision with the normal Agent creation request.
The repository atomically requires an owned approved open Action/Question and
current supported source references, then pins that entry first within the existing
12-note/1,000-token context allowance. Facts, Decisions, closed or inactive entries,
Private Projects and unsupported/stale sources cannot start this flow. The manifest
adds only an origin flag to the entry's existing metadata pin, without copying
note bodies into the manifest. The reviewed objective remains normal saved task
text. No new schema, configuration or separate job is introduced. Creation idempotency
includes the origin identity and revision.

Saved Agents expose the originating entry revision. Knowledge cards can load their
Assigned agents and open the same saved task. Agent completion and Run again
preserve the entry's approval/work state; the user still resolves a question or
completes an action explicitly. Editing, closing or deleting the pinned entry, or
changing its selected sources, blocks further work through the existing context
fences. A Project rename or unrelated new note leaves the saved origin intact.
Execution retains the same verified local Qwen model and available tools.

Agents handle tasks the user assigns, including drafting, planning, analysis and
tool-assisted work. Research mode remains a separate chat workflow. Generic
Plan/Work/Review prompts choose tools when useful; work on supplied text does not
require browsing, citations or an external-source appendix. Missing web tools no
longer block creating or running a model-only task. Actual tool calls still check
the live allowlist, endpoint, owner, lease, model and context authority. The current
tool set now also includes native file creation during Work, described below.
External actions such as purchases or messages remain outside the implemented tools.
Retained `research` checkpoint IDs keep saved
Plans/results and Run again compatible while their displayed stage reads Work.

The focused [installed smoke](../reference/live-smoke-testing.md#installed-knowledge-agent-smoke)
uses a separate ASGI application and temporary canonical data, plus the real task
runtime/executor with mocked verified local-Qwen replies and no web tools. Its seven
checks require no credentials, live network/models or production-account writes.
Affected backend/UI checks and mocked browser acceptance at 1,366, 390 and 320 pixels
pass, including exact prefill, draft choices, explicit Start, origin payload, Work
labels and opening the same assigned Agent. Backend/UI rebuilds are required;
no schema/configuration change or upload was needed for that assignment slice.
The owner reports all seven installed Tower checks passed, exit 0. Real-model and
deployed browser acceptance remain separate.

### Agent file outputs

Work can use the existing native `create_file` tool to produce TXT, Markdown, CSV,
JSON, DOCX or XLSX deliverables. Files are ordinary owned generated outputs in
My Files, with a Project attachment when the task belongs to an ordinary Project
and its file limit permits it. A full Project leaves the file in My Files with
a tool notice. Plan and Review cannot write files. Selected ordinary file inputs
are described below; Private inputs, code execution and external write tools remain pending.

Each task has eight lifetime distinct output identities, counting pending attempts
and deleted files. Requests use the existing safe renderer, a 256 KiB argument
limit and its 4 MiB output limit, plus normal owner storage quotas. Rendered
spreadsheets do not introduce executable formulas. Owner, task and normalized
content determine the file identity: identical retries, restarts and Run again
reuse bytes, while changed content creates an immutable new file. Explicit file
deletion prevents recreation of that identity. Deleting task history removes
its receipts while retaining independently saved My Files outputs.

Application schema 34 stores reserved/ready output intents with owner/task identity,
filename and digest; receipts contain no file bodies or tool arguments. The final
publication holds application then upload locks, rechecks the live lease, owner,
model access, Project context and deletion tombstones, and commits the catalog
and ready receipt without awaiting. The upload database's shared commit helper
also preserves existing native chat behavior. These are separate databases: if
the catalog commits before a crash prevents the receipt commit, an authorized
retry adopts that exact file. Before starting the Agent worker, startup recovery
removes only exact unregistered pending source/sidecar paths and releases their
quota reservations; cataloged files are retained. Pending intents still count
toward the lifetime limit. Cancellation and revoked scope stop publication.

The task's Files panel exposes canonical authenticated downloads while working,
failed, cancelled, completed or rerunning. List summaries omit output metadata;
full task refreshes reconcile the panel. Work checkpoints and final reports retain
canonical file links even when Review omits them. Downloads use normal ownership
and deletion checks, with a recoverable missing-file message. Generated working
outputs do not become independent evidence.

Affected backend tests, frontend checks and mocked browser acceptance at 1,366,
390 and 320 pixels pass. The focused
[installed Agent file smoke](../reference/live-smoke-testing.md#installed-agent-file-smoke)
uses isolated temporary canonical data and mocked verified local replies, without
production accounts, credentials, a live network or models. The owner reports all
seven installed Tower file-output checks passed, exit 0. Live worker/proxy wiring
and actual Qwen file quality remain unrun. Backend/UI
rebuilds are required; schema 34 applies automatically. No settings change or
upload is needed.

### Explicit Project file inputs for Agents

The Agent composer offers an optional Project files chooser after an ordinary
Project is selected. It loads ready file metadata only when opened; up to three
files can be selected. Navigation and rejected starts preserve the draft and
selections. Changing Projects, Clear draft and successful creation clear selected
IDs. Hidden reads abort and reopening revalidates. Starting sends only file IDs,
with the reviewed objective and Project; it does not grant access to other files.

At Start, the server verifies the current owner, ordinary Project attachment,
ready catalog and deletion state, then reads canonical saved text locally.
Documents and images use extracted text or saved OCR; audio/video use saved
transcripts. Retained text can be used when original bytes were reclaimed.
Generated files are identified as working outputs. Empty, unreadable, protected
or oversized sources fail without starting a task or silently dropping a file.
The existing reader bounds disk/expanded inputs to 8 MiB and text to 500,000
characters. Each selected file initially supplies at most 1,500 tokens, up to 4,500 total;
the task detail shows the included/full character counts and shortened extracts.

Schema 35 persists only owner/task/Project identities and bounded metadata pins:
catalog and text digests, prefix length, chosen attachment timestamp/origin and
relative file stat proofs. It stores no source text or absolute paths. Each
Plan/Work/Review step rereads the pinned canonical text and verifies its digest.
Synchronous owner, catalog, attachment and disk checks guard model/tool admission,
checkpoint/final commits and generated-file publication. Publication uses a
shared locked catalog reader within the existing application→upload lock order,
so checking selected inputs does not reacquire the upload lock.

Restart and Run again preserve the same selected versions and completed Plan.
Changing, deleting, detaching or reattaching a selected source blocks new work;
the user starts a new task to select its current text. Project renaming, unrelated
file changes and membership in other Projects do not change the input selection.
Metadata remains visible with a stale-input notice; Resume, Run again and input
answers are blocked. Saved reports retain their normal history behavior. Task or
account deletion cascades input pins; this introduces no retained source-body copy.
Project Knowledge's automatic file-backed references and Private Agent inputs
remain later work. File excerpts are quoted data and confer no new tool authority.

Affected backend tests, frontend tests/typecheck/lint/build and mocked Chromium
checks at 1,366, 390 and 320 pixels pass. The
[installed file-input smoke](../reference/live-smoke-testing.md#installed-agent-file-input-smoke)
exercises real canonical storage, local extraction, task execution and generated
downloads with mocked verified-Qwen replies. The owner reports all seven installed
Tower checks passed, exit 0. Live worker/browser acceptance and actual model quality
remain pending. Rebuild backend/UI; schema 35 applies automatically.
No settings, credentials or upload are required for the isolated smoke.

### Observed Agent execution history

Each task now has Recent activity below Plan and progress. It shows actual local
model requests, allowed tool invocations and stage checkpoints, with attempt,
timestamp, outcome and measured duration/token counts when available. The compact
view shows six entries; users can expand its independently scrolling list and
load older pages. Active selected tasks refresh every three seconds. Collapsing,
hiding or changing tasks cancels reads, and reopening revalidates. New entries
preserve the visible history anchor, workflow scroll and keyboard focus.

Application schema 36 stores metadata-only events with per-task sequence numbers.
Only Plan/Work/Review/Title stages, model/stage/tool kinds and known allowlisted tool
names are accepted. There are no event fields for prompts, reasoning, search
queries, URLs, filenames, source/output bodies, raw exceptions or arbitrary
provider text. Model usage comes from accepted terminal measurements; missing or
unsafe counts remain unknown. These counts describe recent recorded calls,
not lifetime usage or the task's conservative reservation charges.

A model request starts after resource admission and live authorization, directly
before transport. Tool events wrap actual invocation after argument and policy
guards; a tool error is not reported as success. Started records are dispatch
intent, not proof of a completed effect. Successful stage events commit in the
same transaction as their saved checkpoint. Every callback rechecks the current
owner, lease, Project/file context and budget; inherited callbacks cannot outlive
their execution scope. Pause, Cancel, worker stop, failure and expired-lease
recovery mark unmatched starts Interrupted. This means no outcome/checkpoint was
recorded; it does not claim an external effect was undone. Known model/tool
failures retain their Failed outcome. A late callback cannot overwrite interruption.

History survives database reopening and Run again on the same task, with distinct
attempt labels and the existing saved Plan behavior. At most 200 recent entries
are retained per task; omitted older history is disclosed, and pruning cannot
prevent authorized work from completing. Legacy tasks have no invented historical
events. Event writes and reads leave task revisions, operation charges and
notification state unchanged. Task/account deletion cascades the rows. The
owner-only `/api/agents/{task_id}/events` endpoint returns newest-first pages of
at most 50 events with a before-sequence cursor and a final response fence;
events stay out of task responses and history-list summaries.

Affected backend tests, frontend tests/typecheck/lint/build and mocked responsive
browser checks at 1,366, 390 and 320 pixels pass. The
[installed event smoke](../reference/live-smoke-testing.md#installed-agent-event-smoke)
uses real temporary SQLite, owned API routes, task execution and file downloads,
with scripted local-model replies. Negative checks deliberately remove model
observations and disable interruption to confirm the probe fails. This is an
isolated integration check. The owner reports all seven installed Tower event
checks passed, exit 0; live worker/browser/Qwen acceptance remains unrun.
Backend/UI rebuilds are required; schema 36 applies automatically, without a
configuration change. No credentials or upload are required for the smoke.

### Reading further selected-file text

Work now offers a native `read_selected_file` tool when the task has explicit
ordinary Project file selections. The initial excerpts still guide Plan and
Review; Work can request additional parts of the same frozen saved text, OCR or
transcript. Selecting a Project alone grants no file access. The reader accepts
only a selected file ID, a Unicode character offset and a bounded character limit;
it accepts no caller-supplied user, Project, path, URL or revision. Private,
unselected, newly added and newly generated output files are not automatically
eligible. A generated file explicitly selected before task creation keeps its
working-output provenance label.

Pages report their actual start/end, full character count, whether more remains
and the exact next offset. The default request is 3,000 characters, with a maximum
of 6,000; the complete serialized JSON is limited to 5,000 characters. Escaping
and metadata can therefore shorten a page. The next offset follows the text
actually returned, so continuation cannot silently skip a shortened tail. EOF
has an empty page and null continuation. Invalid offsets and arguments return
fixed content-free errors. The existing 8 MiB/500,000-character source limits
and task call/round/time budgets still apply. Availability does not imply that
the model inspected the whole source; Work instructions require honest coverage.

Each read parses only the requested selected source, checks the full saved-text
digest and revalidates owner, lease, context, attachment, deletion, local-model
access and purge state after awaited disk reads. Changed or unavailable inputs
stop execution before source delivery or a new model request. Read callbacks are
bound to one Work attempt and expire on scope exit; escaped callbacks, unbound
chat/Private calls and forged remote specs have no HTTP fallback. The current
local Qwen weights are checked before dispatch, and reads use the existing
operation allowance. Page bodies are transient tool replies, with no new task
input copies or activity-body fields.

Schema 37 adds the read tool to the event allowlist by rebuilding the metadata
table transactionally, preserving existing sequences, outcomes, timestamps,
owners, constraints and task links. Upgrade and deliberate rollback tests verify
saved history remains intact. Recent activity labels reads as Read selected file;
input help distinguishes initial excerpts from later reads.

Affected service/executor/dispatch tests, migration/response tests and frontend
tests/typecheck/lint/build pass. The
[installed file-read smoke](../reference/live-smoke-testing.md#installed-agent-file-read-smoke)
uses real temporary files, canonical storage, task execution and native tool
dispatch with scripted local replies. It reads beyond an opening excerpt,
continues at the actual returned cursor and downloads a generated file containing
the later text. Negative probes detect omitted pages, broken cursors,
unselected-file reads and missing read events. This is isolated integration
evidence; Tower execution and real browser/worker/Qwen acceptance remain pending. Rebuild backend/UI;
schema 37 applies automatically, with no settings, credentials or upload needed
for the isolated smoke.

## Product outcome

A user can give Audrey an objective, choose its Project and sources, start an
agent, and return later to its plan, progress, questions and results. The agent
works on the assigned task within its available tools and context, checks progress
against the objective, and continues from committed steps after a backend restart.

The same workspace lets users find saved content, maintain approved Project
knowledge, and schedule recurring checks that produce meaningful change reports.
These are four connected capabilities with one execution foundation.

Example: start an agent to compare selected technical documents and produce a
cited report; find its evidence through Search; approve a proposed Project
decision; then schedule a weekly check of a selected public release page.

## Current foundations

| Area | Already present | Work required |
| --- | --- | --- |
| Native runs | Saved agent objectives, three-stage plans/checkpoints, recent observed activity, planning questions, owned downloadable files, leases and restart dispatch | Adaptive steps and finer-grained within-step recovery |
| Resource admission | Interactive preference, aged background fairness, cooperative queued handoffs and generation-boundary release | Live concurrent-chat/model acceptance and broader checkpoint-age monitoring |
| Private tasks | Leases, revisions, source/model pins, budgets and committed-result delivery for Private tax reviews | General agent contracts and adapters; existing Private recovery currently pauses work |
| Search | Local lexical API/sidebar, canonical reader and workspace links, exact message focus, bounded projection, approved notes and Project report filters | Semantic candidates and expanded native previews for retained text |
| Projects | Compact overview with Knowledge/Agents below and a Watch row; approved knowledge in native chat and ordinary agents; selected ordinary file inputs, fixed context, Project links and saved-work lists | File-backed Knowledge and Private agent context, automatic proposals |
| Scheduling | Explicit public-page schedules, Project organization/navigation, snapshots/diffs, baselines and Activity alerts | Owned-file/Private and multiple-source watches |

Code anchors:

- [Native run manager](../../src/audrey/routes/app/runs.py),
  [pipeline graph](../../src/audrey/pipeline/graph.py) and
  [startup](../../src/audrey/main.py).
- [Private task repository](../../src/audrey/app_state/private_tasks.py),
  [admission](../../src/audrey/private_task_admission.py),
  [step runner](../../src/audrey/private_task_runner.py) and
  [delivery](../../src/audrey/private_reviews.py).
- [Project repository](../../src/audrey/app_state/projects.py),
  [context snapshots](../../src/audrey/project_context.py),
  [history search](../../tools-server/chat_archive.py) and
  [guarded public fetch](../../tools-server/fetch.py).

Ordinary run history is durable, but its live execution/event buffer is not a
restartable agent. The existing pipeline compiles without a durable checkpointer;
interrupted ordinary runs become terminal failures on startup. Private tasks are
specialized around Tax cases and deliberately pause on restart. Exposing their
existing routes alone would not deliver the requested architecture.

## Main UI

Add **Agents** beside Chat and Projects in primary desktop/mobile navigation.
Keep an active-work count visible without requiring an open conversation.

The Agents home has New agent task and views for Running, Needs your input,
Scheduled and Finished. Paused/failed work remains accessible with its stop reason.
Cards identify the objective, Project, current stage, last real activity and result.

Each task opens a full workspace containing:

- Objective, success criteria, selected Project/sources and processing mode.
- A visible plan with pending/running/completed steps and bounded replanning.
- Observed activity, findings, source links and produced files.
- Questions that need the user's input, with answers saved against the task.
- Pause, Resume, Cancel and Run again. Run again keeps the same agent and its
  completed Plan, continuing unfinished work or refreshing a completed report.
- Results and proposed Project knowledge updates with links to their provenance.

The visible plan explains intended work; progress reflects observed actions.
Neither is a raw model reasoning transcript. Use stage/step status rather than
invented percentage completion.

Projects gain links to their agents, schedules and Knowledge panel. Search can
open matching messages, file passages, agent results and approved knowledge.
Chat and Project actions can prefill a new task; starting it remains explicit.

## Shared execution design

```mermaid
flowchart TD
    UI[Agents, Chat and Projects] --> API[Owner-bound native API]
    API --> DB[Canonical application SQLite]
    DB --> Worker[Agent dispatcher and coordinator]
    Worker --> Pipeline[Local Qwen 3.8 and permitted tools]
    Worker --> Context[Search and Project knowledge]
    Worker --> DB
    Schedule[Schedule dispatcher] --> DB
    DB --> Results[Saved results, events and notifications]
    Results --> UI
```

Start with a lifecycle-owned dispatcher inside Audrey. It must share the existing
GPU gate, per-user admission, discovery and processing policies. The current GPU
gate is process-local; moving work to a separate worker would first require
shared admission. No new orchestration service is required for the initial scope.

One coordinator owns each objective:

`admit -> plan -> execute bounded step -> commit result -> check objective`

The next action is another step, a bounded replan, a saved user question, or a
terminal result. Existing local generation and tool capabilities perform the actual
work. Trusted code validates the proposed steps, dependencies, tools and source
scope. Agent model work is local-only, currently restricted to `qwen3.8:latest`
for planning, execution and review. Cloud models and fallback are excluded.
All child work counts against the same allowance.

### Durable state and recovery

Keep task lifecycle and user-visible state in application SQLite. Proposed
entities, finalized during the first milestone:

| Entity | Purpose |
| --- | --- |
| Agent task | Owner, objective, accepted scope, Project, privacy, sources and limits |
| Agent run | One execution/retry, context manifest, status, usage and stop reason |
| Plan/step | Versioned work plan, dependencies, input identity and committed outcome |
| Attempt/operation | Lease, fencing generation, request identity and tool/output receipt |
| Event | Ordered public activity for reconnect and saved history |
| Input request | A specific question, expected revision and saved response |
| Delivery/notification | Durable result reference, deduplication key and read/delivery state |

Use separate task status and execution stage. Statuses include queued, running,
waiting for user, paused, completed, cancelled and failed; stages describe planning,
execution and checking. Waiting/paused tasks release compute and leases.

Persist explicit step inputs, selected source/context revisions, concise progress,
operation receipts and output references. Do not serialize live coroutine objects
or claim arbitrary instruction-level resumption. Resume at supported committed
step boundaries. A new attempt can repeat uncommitted read/model work within its
remaining allowance; committed outputs must be reused.

Use atomic claims, expiring leases and monotonically increasing fencing values.
Check the fence again before committing progress, attaching an output or delivering
a result. Commit checkpoint/result references and delivery intent together where
they share the application database; use durable receipts across storage boundaries.
Replay committed delivery after restart without duplicate files/messages/notices.

Record an operation identity before dispatching effects. An uncertain external
outcome without a safe idempotency/reconciliation path pauses for review instead
of being blindly repeated. Exactly-once external execution or model billing is
not promised.

The first implementation uses an explicit step controller around existing
pipelines. Evaluate a persistent LangGraph checkpointer where it adds useful
within-step recovery, after confirming installed-version compatibility. Task
state and Project knowledge remain different authorities: LangGraph likewise
distinguishes [thread checkpoints from cross-thread stores](https://docs.langchain.com/oss/python/langgraph/persistence).

### Scope, authority and resource use

Starting a task authorizes the selected read/research and generated-output work.
Routine work inside that scope proceeds without repeated confirmations. Missing
facts create saved input requests. Expanding the objective or performing a new
consequential external action requires a concrete user decision.

Revalidate the active owner, model/tool grants, Project membership, required
sources and processing mode on claim/resume, before each model/tool dispatch or
protected source read, and before committing outputs.
Never persist browser bearer credentials. Snapshot relevant inputs for each run;
material changes require revalidation/replanning, while a Project rename does not.

Bound queued/active work per user and globally, plan steps/replans, model/tool
calls, retries, active execution time, output bytes and event retention in
`config.yaml`. Start conservatively with one active agent task per user. Account
for all attempts and child calls; resume and Run again do not reset usage.
Reruns check current queue capacity and remaining saved allowances before changing
the task. Reserve resources
before dispatch and conservatively retain charges for unknown outcomes after a
crash until reconciled. User waiting time does not consume active execution allowance.

Extend existing admission to distinguish interactive and background calls, with
aging so neither is starved. Yield at model/step boundaries; do not interrupt an
active GPU generation to implement priority. Track queue wait, checkpoint age,
stops and retries through existing monitoring without private-content labels.

Private processing inherits canonical restrictions. Public web research/watches
are unavailable in Private mode; supported local source work can be enabled only
through verified local adapters. Keep the existing Private Tax task path intact
initially, then expose it through an Agents adapter once its behavior is preserved.

## Unified content search

Build a native Search view and an authenticated `/api/search` service. Derive
ownership from the principal; requests can filter source kind, Project, date and
archive state, but cannot select another user's namespace.

Return typed source IDs, source revision, title, matching excerpt/highlight ranges,
availability and an exact supported navigation target. Search messages and file
contents rather than only titles/filenames. Add agent results and approved Project
knowledge as those entities ship.

Use a dedicated, rebuildable local SQLite search projection for lexical content,
with bounded incremental indexing/backfill. Keep canonical application messages,
upload catalog records and retained artifacts authoritative. Changes and deletion
fences must be checked before backfill or returning any indexed snippet. Verify
FTS5 support in the runtime; explicit indexing/backfill and update/delete handling
are required for [SQLite full-text indexes](https://www.sqlite.org/fts5.html#external_content_tables).

Index completed/incomplete user and assistant messages directly by their native
IDs; exclude in-progress drafts and raw tool messages/results. The existing history archive
uses different derived IDs, so wrapping its search tool does not provide reliable
native message targets. File chunks retain file/artifact identity, content revision
and character offsets. Generated summaries/reports remain labeled as such.

Lexical search ships first. Add semantic candidates through scoped adapters once
canonical identity mapping and local processing are established. Preserve separate
document and history embedding profiles; merge retriever ranks rather than
comparing incompatible raw scores. Deduplicate chunks and limit domination by a
single long source. Exact search remains usable during semantic dependency failure,
with a clear partial-results state.

Results open a bounded native message window or the relevant file artifact and
highlight the match. Page numbers and media seek targets require real stored
locators; use text offsets when those locators do not exist. Do not infer them
from chunk order or generated prose.

All personal-content search is local. Private query text/snippets must not enter
ordinary sidecar history/memory or cloud paths. Native local lexical search can
cover authorized protected records; semantic Private search needs its verified
local adapter and isolated collection. A helper named `search_private_file_text`
currently means ordinary per-user files and excludes protected uploads; it cannot
stand in for the new Private path.

Enforce live Project membership, deletion and source revision after awaited
retrieval. Removed originals can still have valid retained transcripts, but an
index cannot manufacture the missing original. Bound query syntax, candidate
work, pages and snippets; bind pagination to the owner/query/filters. Render
untrusted excerpts as text, never executable HTML.

## Project working memory

Add a canonical Project Knowledge repository with Facts, Decisions, Questions and
Actions, separate from global key/value memory and Tax case records.

Entries have owner/Project IDs, kind, text, approval state, current revision,
creation/edit/approval dates and typed source references. Questions/actions also
have open/resolved/completed state; approving an action does not complete it.
Revisions support corrections, superseding and concurrent-edit conflicts.

Users can create/edit approved entries directly. Agents propose entries or
replacements; only a user can activate them. Task completion never self-approves
memory. Show a conflicting proposal beside the existing entry and its evidence.
Approval records accepted Project knowledge, not independently verified truth.

Support user-authored statements, native messages, inspected file passages and
versioned public-source snapshots. A generated report can identify where a
proposal originated, but its underlying inspected sources supply evidence.
Unsupported inferred suggestions remain explicitly labeled.

Extend Project context snapshots with bounded selected approved entries and their
revisions. Prefer relevant/pinned decisions and open work; set a combined Project
context allowance so knowledge does not crowd out file evidence. Treat entries
as quoted data below platform instructions. Exclude pending/rejected/superseded
and source-dependent stale entries from automatic grounding.

Each agent run records the knowledge revisions it used. Material changes require
explicit revalidation/replanning; do not silently replace the context under an
active run. Agents can propose updates from results and link actions to tasks,
but an action entry does not itself start work or create a schedule.

Project home exposes edit, approve/reject, history, source viewing and stale/conflict
review. Agent results link to proposals; entries link to their producing task and
accessible evidence. Search defaults to current approved knowledge; proposals and
history require explicit filters.

Private knowledge stays within canonical protected storage and verified local
processing. It never enters ordinary global-memory recall or cloud summarization.
Project deletion removes its knowledge/proposals, revision bodies, retained
excerpts and search projections while retaining the established conversation/file
behavior. Account purge includes every new derivative.

Source detachment removes grounding and source-preview links within that Project;
it does not remove the owner's My Files access or previews elsewhere. Source
deletion immediately fences all source reads and dependent automatic grounding.
Remove automatically extracted proposal/entry bodies, revision excerpts, saved
task input copies and owned-file watch snapshots derived solely from that source;
retain only unavailable-source metadata. These copies cannot become alternate
source reads. Explicit user-authored notes and separately saved reports can remain
with a missing-source notice, but are not substitutes for the deleted source.
Deleting those records and account purge remove their bodies and search projections.

## Scheduled monitoring

A watch is an owned, versioned definition that creates agent runs. It records
what to inspect, what changes matter, the authorized source scope, cadence,
IANA timezone, enabled state and notification policy.

Initial watch types:

- Selected public HTML/text URLs through the existing guarded fetcher.
- Selected owned file revisions or additions/revisions within an explicitly
  selected Project file scope. Private file checks use only the local path.

Each occurrence snapshots the sources currently authorized by the saved watch
scope. A Project-file watch can intentionally include later matching additions;
a single agent run still retains its own fixed input manifest.

The first successful check establishes a baseline. Save immutable source snapshots,
checksums, extraction version, check time and per-source coverage. Normalize known
template noise deterministically. Unchanged normalized content can skip model
comparison; changed content receives a bounded diff and evidence-linked explanation
against the user's criterion. Distinguish an observed content change from the
model's judgment about its significance.

Compare only compatible source identities and extraction/normalization versions.
Replacing a source or changing extraction rules establishes a fresh baseline or
an explicit rebaseline; it cannot manufacture a content-change alert. A criterion-only
edit can retain compatible snapshots, bound to the new definition revision.

Failed, blocked, empty or partially read sources never become "no change" and do
not replace the previous valid baseline. Advance each baseline only after its
successful snapshot/report commit. Distinguish normal changes, no change, partial
coverage and degraded checks in the UI.

Schedule defaults:

- Offer daily/weekly wall-clock schedules and bounded intervals; use the user's
  selected IANA zone, with upcoming occurrences previewed. Store execution UTC.
- A nonexistent DST time runs at the first valid instant after the gap; a repeated
  local time runs once. Test the exact library behavior before enabling schedules.
- Coalesce missed occurrences during downtime into one labeled catch-up check.
- Permit one active run per watch and at most one coalesced pending occurrence.
- Use a unique `(watch, definition revision, scheduled occurrence)` identity;
  enqueue and advance the schedule transactionally.
- Pausing/editing/deleting a watch fences affected queued work. Current-definition
  checks prevent an older run from replacing a newer definition's baseline.
- Apply capped retry/backoff and a deduplicated notice for persistent failures.

Saved recurring definitions require an explicit Create watch action. Natural
language can prefill a definition and preview its cadence; casual discussion does
not silently create one. Watches appear in Agents and their Project, with Run now,
Pause, Edit, History and Delete.

Notifications begin with a durable in-app inbox/activity badge. Agent completion,
input requests, relevant watch changes and persistent failures link to the exact
run. Unchanged checks remain in history without repeated notices. External/push
channels are a later explicit opt-in; they carry no Private source content.

Search-query watches, authenticated websites, mailbox/calendar integrations and
arbitrary external actions are outside the initial release. They can later use
the same task/schedule contracts after their source/permission behavior is defined.

## Delivery milestones

| Milestone | Deliverable | Exit condition |
| --- | --- | --- |
| 0 — Shared contracts | Source references, context manifests, lifecycle/revision rules, bounded task schema and Agents navigation design | One agreed contract connects all four features; initial limits and supported step types are explicit |
| 1 — Agents first | Working prominent Agents UI plus one useful research/document workflow, persistent plan/steps, input requests, controls, results and in-app notices | Close browser and restart backend; the task retains committed work and completes without duplicate outputs |
| 2 — Unified search | Local content search, bounded index backfill, typed targets, snippets, filters and deep navigation; semantic enhancement after identity mapping | Find content absent from titles/filenames and open the real matching message/artifact |
| 3 — Project knowledge | Manual approved entries/revisions, bounded cross-chat context, agent proposals and source/stale handling | A decision carries into a later Project chat; an agent proposal stays inactive until approved |
| 4 — Scheduled agents | Watches, timezone-aware occurrence dispatch, snapshots/diffs, baseline management and change/failure notices | Initial baseline, real change, no change, failure, downtime and duplicate wakeups produce distinct correct outcomes |
| 5 — Integrated acceptance | End-to-end Project/Agents/Search/Knowledge/Watch flows and focused recovery/load checks | The user's native-browser workflow succeeds with real sources and preserves all ownership/Private boundaries |

After milestone 0, search and knowledge storage/UI work can proceed independently
while the Agents vertical slice is built. Watches depend on reliable agent runs,
not on completion of semantic search. Deliver each milestone as a reviewable slice;
do not postpone all useful UI until the infrastructure is finished.

## Implementation touchpoints

New module and route names below are proposed, not existing interfaces.

| Layer | Planned changes |
| --- | --- |
| Application state | Add agent/knowledge/watch/notification repositories beside `app_state/projects.py`; register them in `ApplicationStore` with additive migrations |
| Execution | Add an agent coordinator/dispatcher in Audrey's lifespan; use existing pipeline/tool execution and extend shared admission for background priority |
| Source context | Add shared typed source/context manifests; extend `project_context.py` and source resolution without using model-generated paths |
| Search | Add `content_search.py`, a configured rebuildable SQLite projection/indexer, owner-scoped adapters and native message-window resolution |
| Native routes | Add `/api/agents`, `/api/search`, `/api/projects/{id}/knowledge`, `/api/watches` and unified `/api/activity`; retain watch-only `/api/notifications` compatibility. Actions use expected revisions and creation idempotency keys |
| Browser | Add Agents workspace and global Search navigation, task detail/input/result views, in-app notification state and Project Knowledge/agent/watch links |
| Configuration | Add bounded agent/search/knowledge/watch settings to canonical `config.yaml`; retain existing model selection and Private policy authority |

## Verification and rollout

For each slice, run affected hermetic tests and changed-file lint. Key failure
boundaries are:

- Two claims for one step, stale leases, crash before/after commit, lost delivery
  acknowledgement, partial file/output handover and cancellation during completion.
- Waiting for input across restart; stale answers and concurrent edits; bounded
  planner loops, retries and usage across resumed attempts.
- Owner/model revocation, account purge, source deletion or Project changes while
  retrieval/model work is in flight; delayed index writers cannot resurrect access.
- Detach a still-owned file without removing My Files access; delete a source after
  task/watch snapshots exist; delete a Project with knowledge history. Derived
  copies and delayed writers cannot bypass the relevant deletion or scope change.
- Lexical indexing backfill/update/delete correctness, hostile query syntax,
  duplicate filenames, native message anchors, missing artifacts and semantic outage.
- Conflicting knowledge revisions, inactive proposals, stale sources, Project
  isolation and bounded context with large collections.
- DST transitions, downtime catch-up, overlapping checks, schedule edits, partial
  fetches, unchanged content and duplicate alerts.
- Interactive chat responsiveness during agent work, eventual background progress,
  and Private requests making no ordinary memory/history/cloud/web calls.

Native acceptance is manual-first: start real work, leave/reconnect, answer an
input request, inspect results, find evidence, approve a knowledge proposal and
observe a controlled watch change. Hermetic pass counts do not establish model
answer quality; use task-specific success criteria and inspect actual outputs.

Implement and verify on the laptop checkout. The owner pulls and deploys on Tower
from `/mnt/user/appdata/audrey_ai_2.0`; backend/UI slices require the appropriate
`audrey`/`audrey-ui` rebuilds, and changed bind-mounted configuration requires service
recreation. Provide exact per-slice commands and required real inputs with each
handoff. The implemented Agents/Search slices require backend/UI rebuilds and backend
recreation; public research needs no upload. Existing saved content can verify Search;
upload a real document only if no suitable readable file already exists. Check a
word present in a message/file/report body but absent from its title, open its
highlighted source, then delete the source and confirm it cannot be opened again.

## Initial implementation boundaries

Use existing identity, model policies, tool declarations, file/output ownership,
Private enforcement and native application infrastructure. Additive migrations
must preserve canonical data. Existing Tax reviews, ordinary chat and Hermes'
separate tool/document workspace retain their contracts.

The first code slice is milestone 0 plus the smallest complete milestone-1 agent
workflow. Choosing an external notification channel or adding a separate worker
service does not block that slice. This plan authorizes no deployment, schedule
activation, external messages or runtime source changes by itself.
