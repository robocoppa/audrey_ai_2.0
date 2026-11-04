# Live testing — runner and deployment reference

This reference owns runner, authentication and rebuild instructions. Campaigns
are closed; no smoke or acceptance work is implied by these reusable commands.
English user input only; closed checks do not need another run.

## Where to run

| Work | Location and target |
|---|---|
| Hermetic tests/lint | Laptop checkout and its `.venv` |
| API-only smoke/eval | Laptop → `http://192.168.1.11:8000` (LAN/WARP) |
| Files/upload/chat acceptance | Browser → `https://ai.builtryte.xyz`; use a real file |
| Docker-only Ollama probe | Tower → `scripts/probes/probe-onbox.sh` |
| Native UI/proxy protocol smoke | Tower → `tests/smoke/smoke-native-onbox.sh` |
| Installed backend admission smoke, no models | Tower → `tests/smoke/smoke-agent-admission-onbox.sh` |
| Installed Knowledge source API smoke, isolated data | Tower → `tests/smoke/smoke-knowledge-sources-onbox.sh` |
| Installed report suggestion smoke, mocked model transport | Tower → `tests/smoke/smoke-agent-knowledge-drafts-onbox.sh` |
| Installed Agent Project Knowledge smoke, isolated data | Tower → `tests/smoke/smoke-agent-project-knowledge-onbox.sh` |
| Installed Knowledge Agent smoke, isolated data/mocked model | Tower → `tests/smoke/smoke-knowledge-agents-onbox.sh` |
| Installed Agent file smoke, isolated storage/mocked model | Tower → `tests/smoke/smoke-agent-files-onbox.sh` |
| Installed Agent file-input smoke, isolated sources/mocked model | Tower → `tests/smoke/smoke-agent-file-inputs-onbox.sh` |
| Installed Agent event smoke, isolated execution/scripted model | Tower → `tests/smoke/smoke-agent-events-onbox.sh` |
| Installed Agent file-read smoke, isolated paging/scripted model | Tower → `tests/smoke/smoke-agent-file-reads-onbox.sh` |
| Installed Agent file-backed Knowledge smoke, isolated sources/scripted model | Tower → `tests/smoke/smoke-agent-knowledge-files-onbox.sh` |

Laptop checkout: `/home/bart/Documents/github/audrey/audrey_ai_2.0`.
Tower checkout: `/mnt/user/appdata/audrey_ai_2.0`.
Tower has Docker, not a host Python/uv/repository `.venv`.
The backend is published on port 8000. The UI origin is Tower-loopback
`127.0.0.1:8090` or Docker DNS `http://audrey-ui:8080`; laptop access to that
origin needs an explicit tunnel. Tailscale `100.113.157.98:8000` remains
unreachable from the laptop. Repair is deferred until the user asks; use the
working LAN/WARP route for current checks.

## Credentials

Laptop `.env.test.local` is private, gitignored, mode 600. Inspect names/existence
without printing values before telling the user to source it.

| Variable | Use |
|---|---|
| `AUDREY_EVAL_BASE_URL` | API eval URL, normally `http://192.168.1.11:8000/v1` |
| `AUDREY_EVAL_API_KEY` | Audrey PAT with `compat:full` for compatibility calls |
| `AUDREY_USER_JWT` | Current Cloudflare Access application assertion for native user smokes |
| `AUDREY_ADMIN_JWT` | Distinct administrator's application assertion when required |

Tower `.env.smoke.local` contains only the two native JWT names. Use plain
`KEY=value`, no `export` or Markdown URLs. Application JWT values expire;
refresh them in place from the Audrey application's `CF_Authorization` cookie
or the existing cloudflared token workflow. A PAT is not a Cloudflare assertion.
Never print credentials, commit them, or paste them into chat.

## Choosing a check

Test the changed feature and its direct failure boundary. Inspect the real
script's environment names and arguments before handing off a command.
Prefer one targeted protocol case over a broad suite. Native upload/Files/chat
flows use manual browser acceptance; say whether an upload is required.
Create a focused reusable smoke when changed concurrency, deployment, restart or
service integration behavior benefits from it. Verify the runner locally, supply
its exact deployed command, and distinguish isolated checks from live-model evidence.

Every handoff states where commands run, what must rebuild/restart after pull,
and the visible pass condition. Stop at a failed step before a dependent one.
Laptop hermetic verification does not prove deployed behavior.

Short checks return results in the launching shell. For short direct Ollama
probes use `FOREGROUND=1` and a compact summary if supported; Telegram is skipped
unless `NOTIFY=1`. Detached runners are for long jobs needing disconnect safety.
Pings, exit codes and reports follow the [testing protocol](testing-protocol.md),
shared with ai-sec. API evals normally save to laptop `evals/results/`. If a necessary Tower report
is saved remotely, supply an explicit laptop copy step; do not rerun to retrieve
it. See [evals](../../evals/README.md) only for an actual evaluation task.

Delete temporary handoffs after acceptance. Keep a brief phase status and
regression code; retain model evaluation artifacts needed for decisions.

## Installed admission smoke

After deploying the interactive/background admission slice, run this on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
tests/smoke/smoke-agent-admission-onbox.sh
```

The wrapper streams the smoke into the running `audrey` container using its
installed package and `/opt/venv/bin/python`, and reads `/app/config.yaml`.
These smoke-only additions do not need an image rebuild. An older image without
admission support reports setup error 2; deploy the backend code/configuration
first using the service rebuild guidance below. No credentials, `.env.smoke.local`,
upload, model generation, network request or production account data is needed.

Expect five `PASS` lines and `PASS admission smoke (exit 0)`. Configured GPU/user
capacities and aging are printed separately from the deliberate cap-one fixture.
The checks cover configured capacity/cancel cleanup, chat passing queued background
work, GPU aging, running-call non-preemption and cancellation during a handoff.
Unaged-priority/handoff fixtures use a separately reported aging threshold of at
least 60 seconds so very short supported settings cannot age them prematurely.
The GPU-aging case uses the configured threshold and a backdated monotonic timestamp.
All overrides belong to isolated test queues; the running server's config is not changed.
The smoke normally takes a few seconds, with a ten-second deadline per check.
Short results return directly in the launching shell; no report transfer or
notifications are required. `--json` produces a compact machine-readable receipt.

This verifies installed scheduler/configuration behavior with isolated queues.
It does not inspect the server process's active queues, generate an answer, or
measure live GPU latency or HTTP wiring. Live contention acceptance remains separate.
Exit codes: 0 pass, 1 finding/timeout, 2 setup failure, 3 unexpected crash, 130 interrupted.
For a different checkout/container use `AUDREY_SMOKE_APPDATA` (or `APPDATA`) and
`AUDREY_SMOKE_CONTAINER`; forwarded Python options include `--config`, `--json`
and `--timeout-seconds` (positive, at most ten seconds).

## Project Watch API smoke

After pulling the Project Watch slice, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-native-onbox.sh smoke_native_project_watches.py
```

Backend startup applies schema 33 automatically. No config change, recreation
flag or upload is needed. The wrapper uses the `audrey:latest` image's Python
against `http://audrey-ui:8080`, mounting the smoke source read-only. It needs a
current `AUDREY_USER_JWT` in Tower `.env.smoke.local`; `AUDREY_ADMIN_JWT` is optional
for this smoke. Remote credential freshness cannot be checked from the laptop.
The account needs one free Watch slot and permission to create ordinary Projects
and Watches, with Watches/local model access enabled.

The smoke creates two disposable ordinary Projects and one **paused** public-page
Watch, checks idempotency, Project filtering/renaming, explicit unlink/reassignment,
and unavailable-Project handling, then deletes its fixtures in `finally`. Run and
Resume are tested only after Project deletion, where both must reject the request;
no check should be queued. The source is a reserved `.invalid` URL. The smoke
does not request successful page fetches, uploads or model generations and does
not alter existing Projects or Watches. Failure cleanup errors fail the run.

Expect `PASS` lines for the contracts and cleanup, followed by
`PASS Project Watches API smoke passed` and `(exit 0)`. Results return directly
in the launching shell, with no report transfer or notifications. Add `--json`
for a compact receipt. Each HTTP request has a ten-second timeout. Exit codes:
0 pass, 1 API contract/cleanup finding, 2 setup failure, 3 harness crash,
130 interrupted. This checks deployed native proxy/API integration; actual
scheduled public-page comparisons and local-model quality remain separate.

## Installed Knowledge source smoke

After pulling the source-reference editor, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-knowledge-sources-onbox.sh
```

No schema/configuration change, credentials, `.env.smoke.local` or upload is needed.
The wrapper streams the checked-out probe into the running backend's installed
Python package. It creates a separate ASGI application and temporary SQLite/upload
fixtures, then removes them. It never opens the server's account/upload databases,
contacts a network, calls a model or changes production data. An older image
without the source browser exits with setup error 2; rebuild the backend first.

Expect five `PASS` checks and `PASS Knowledge source smoke passed` with `(exit 0)`.
They cover Project-scoped canonical candidates/cursors, preview paging and exact
revisions, attach/refresh/remove, retained text after original loss, detachment,
Private/owner fencing and disposable cleanup. Private fixtures represent already
processed local sources; this does not test Private upload admission. Authentication
is overridden in the separate fixture application, so this does not prove live
Cloudflare/proxy wiring or browser behavior with real content.

The smoke normally takes seconds; each case has a ten-second deadline. Add `--json`
for a compact receipt or `--timeout-seconds` to lower the per-case deadline. Exit
codes are 0 pass, 1 finding/timeout, 2 setup failure, 3 harness crash, 130 interrupted.
Results print in the launching shell; no notifications or report transfer is needed.
`AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override the
checkout/container. Real-file browser acceptance remains manual: use an owned
Project's existing file/message, preview and attach it to a note, then explicitly
review/replace a changed reference. A new upload is optional.

## Installed report suggestion smoke

After pulling the report suggestion helper, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-agent-knowledge-drafts-onbox.sh
```

No schema/configuration change, credentials, `.env.smoke.local` or upload is needed.
The wrapper streams the checked-out probe into the running backend's installed
Python package. A separate ASGI application uses temporary canonical accounts,
Projects, notes and completed Agent reports, plus the real local-model client
with an in-memory mocked provider. It never opens production databases, contacts
a model/network or uses a live account. Missing installed helper support exits 2.

Expect six `PASS` checks and `PASS Report suggestion smoke passed` with `(exit 0)`.
The checks verify local metadata verification and report-only structured requests,
unchanged task/operation/notice/Knowledge state on generation, owner/report revision
and completion fences, invalid/truncated replies, an explicit pending proposal save,
discarding late output after rerun, and isolated cleanup. Authentication is overridden
inside the fixture application; this does not test live Cloudflare/proxy wiring,
real Qwen output quality, or the running server process's admission queues.

The smoke normally takes seconds, with a ten-second deadline per case. `--json`
prints a compact receipt; `--timeout-seconds` can lower the per-case deadline.
Exit codes: 0 pass, 1 finding/timeout, 2 setup failure, 3 harness crash, 130 interrupted.
Results print in the launching shell, with no notifications/report transfer.
`AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override the
checkout/container. Real-model browser acceptance uses an existing completed report:
Propose Project knowledge → Generate suggestions → Use suggestion → review/edit →
Save proposal. Verify it remains pending until separately approved in Project
Knowledge. Oversized reports are refused without truncation; manual proposals remain available.

## Installed Agent Project Knowledge smoke

After pulling message/report-backed Agent knowledge, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-agent-project-knowledge-onbox.sh
```

No schema/configuration change, credentials, `.env.smoke.local` or upload is needed.
The wrapper streams the checked-out probe into the running backend's installed
Python package. It creates separate temporary canonical accounts, Projects,
messages, approved notes and saved reports, and removes them afterwards. It never
opens production account databases or calls a model/network. An older image lacking
this support reports a failed contract or setup error; rebuild the backend first.

Expect six `PASS` checks and `PASS Agent Project Knowledge smoke passed` with
`(exit 0)`. They cover approved message/report notes, exclusion of unsupported/stale
sources, frozen pins after restart, source-change dispatch/commit fences, safe
unrelated changes and isolated cleanup. The smoke verifies installed repository
integration; it does not exercise the live HTTP proxy, server process, browser or
real model output quality.

The smoke normally takes seconds, with a ten-second deadline per case. `--json`
prints a compact receipt; `--timeout-seconds` can lower the per-case deadline.
Exit codes: 0 pass, 1 finding/timeout, 2 setup failure, 3 harness crash, 130 interrupted.
Results print in the launching shell, with no notifications/report transfer.
`AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override the
checkout/container. Manual browser acceptance uses an existing ordinary Project:
approve a current report-linked proposal, start a new agent in that Project, and
expand its Project context to confirm approved notes were selected. A source change
must block continuation while retaining the saved report. No upload is required.

## Installed Knowledge Agent smoke

After pulling the Knowledge action/question assignment flow, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-knowledge-agents-onbox.sh
```

No schema/configuration change, credentials, `.env.smoke.local` or upload is needed.
The wrapper streams the checked-out probe into the running backend's installed
Python package. It uses a separate ASGI application and temporary canonical accounts,
Projects, Knowledge entries and Agents, then removes them. Authentication is
overridden only in that isolated application. One case runs the real Agent runtime,
executor, ReAct loop and local-model client with mocked verified-Qwen replies and
no registered tools. It never opens production databases, contacts a live network
or starts a real model. Missing installed support exits 2; rebuild
the backend first.

Expect seven `PASS` checks and exit 0. They verify explicit creation without changing
the entry's open state, origin selection and metadata-only storage, input/access
boundaries, idempotency and canonical Project renaming, saved-origin restart/rerun
and change fences, a general task completing without web tools or forced sources,
and isolated cleanup. This verifies installed API/repository/executor
contracts; live Cloudflare/proxy wiring, the running worker and real model output
quality remain separate.

The smoke normally takes seconds, with a ten-second deadline per case. `--json`
prints a compact receipt; `--timeout-seconds` can lower the per-case deadline.
Exit codes: 0 pass, 1 finding/timeout, 2 setup failure, 3 harness crash, 130 interrupted.
Results print in the launching shell, with no notifications/report transfer.
`AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override the
checkout/container. Manual browser acceptance uses an existing ordinary Project:
open an approved open Action/Question, select Assign to agent, review the
draft and press Start agent. Confirm the saved agent's origin revision and that
Knowledge still shows the entry as Open after the task completes. Assigned agents
opens the same saved agent. No upload is required.

## Installed Agent file smoke

After pulling Agent file outputs, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-agent-files-onbox.sh
```

Application schema 34 applies automatically at backend startup. No settings change,
credentials, `.env.smoke.local` or upload is needed. The wrapper streams the
checked-out probe into the running backend's installed Python package. Each case
uses an isolated ASGI application and temporary canonical accounts, tasks, upload
catalog and source directory; no production databases are opened. Authentication
is overridden only inside the fixture. The first case runs the real task runtime,
executor, ReAct loop and local-model client with mocked verified-Qwen replies,
then downloads the actual rendered file through the owned route. Other cases use
the same native file service, storage quota and renderer. No live network or model
calls occur. Missing installed support exits 2; rebuild the backend first.

Expect seven `PASS` checks and `PASS Agent file smoke passed` with `(exit 0)`.
They verify an owned download and foreign-owner denial, identical retry/restart
reuse and immutable content changes, explicit file deletion, cancelled task
authority, quota and eight-file lifetime limits, task deletion retaining owned
files, and isolated cleanup. This proves installed API/executor/storage contracts;
the live worker, Cloudflare/proxy wiring and actual model output quality remain
separate acceptance checks.

The smoke normally takes seconds. `--json` prints a compact receipt;
`--timeout-seconds` lowers the ten-second per-case deadline. The Linux runner
uses a separate process per case so the deadline also bounds blocked synchronous
work; timed-out children are terminated before their temporary fixture is removed.
Exit codes: 0 pass, 1 finding/timeout, 2 setup failure, 3 harness crash, 130 interrupted.
Results print in the launching shell, with no notifications or report transfer.
`AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override the
checkout/container.

Manual browser acceptance needs no upload: ask an Agent to create a downloadable
checklist or CSV from text in its prompt. Download it from Files and confirm the
same output appears in My Files and the task's ordinary Project. Run again should
reuse identical content on the same task; changed content may create a new file.
Deleting task history should retain the saved file in My Files. Real model
acceptance checks whether it chooses the tool and creates the requested content.

## Installed Agent file-input smoke

After pulling selected Project file inputs, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-agent-file-inputs-onbox.sh
```

Application schema 35 applies automatically at backend startup. No settings change,
credentials, `.env.smoke.local` or upload is needed. The wrapper streams the
checked-out probe into the running backend's installed Python package. Temporary
accounts, ordinary/Private Projects, tasks, source bytes and upload catalogs are
isolated from production. Authentication is overridden only inside the fixture.
The execution case uses the real task runtime, executor, ReAct loop and renderer,
with mocked verified local-Qwen replies; it checks selected text in each stage,
creates a real output and downloads its bytes. No live network or models run.
Missing installed support exits 2; rebuild the backend first.

Expect seven `PASS` checks and `PASS Agent file input smoke passed` with `(exit 0)`.
They verify explicit selection/metadata, selected-only model context and a real
output, pinned restart and same-task rerun, owner/Private/attachment/readability
boundaries, changed/deleted/detached source fences, bounded retained/OCR/transcript
and generated text, and isolated cleanup. The running worker, proxy authentication
and actual model comprehension remain separate acceptance checks.

The smoke normally takes seconds. `--json` prints a compact receipt;
`--timeout-seconds` lowers the ten-second per-case deadline. Each Linux forked
case has a parent-enforced deadline, including blocked synchronous work; the child
is terminated before its temporary fixture is removed. Exit codes: 0 pass,
1 finding/timeout, 2 setup failure, 3 harness crash, 130 interrupted. Results print
in the launching shell; no notifications or report transfer are needed.
`AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override the
checkout/container.

For manual browser acceptance, use an existing ordinary Project with a readable
ready file. In Agents, choose the Project, open Project files and explicitly select
the file, then assign a task that uses its content. Confirm the task's Input files
details, including any shortened extract, and inspect the resulting work. No
upload is needed when a suitable file already exists; otherwise add one through
the Project's normal file flow first. Selecting a Project alone does not select
its files. Private Projects remain unavailable here.

## Installed Agent event smoke

After pulling per-task Recent activity, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-agent-events-onbox.sh
```

Application schema 36 applies automatically. No settings change, credentials,
`.env.smoke.local` or upload is needed. The wrapper streams the checked-out probe
into the backend container's installed Python package. It builds a separate ASGI
application with temporary accounts, SQLite databases, Projects and files;
authentication is overridden only in that fixture. It does not open production
databases or call the running HTTP server. Two cases use the real task runtime,
executor, model-client protocol, tool dispatcher and renderer, with scripted model
metadata/replies; one downloads a real generated file, the other forces a quota
failure. No real Qwen inference or live network requests occur.

Expect seven `PASS` checks and `PASS Agent event smoke passed` with `(exit 0)`.
Six scenario groups check actual event/usage recording without copied content,
tool failure classification, reopened-database/expired-lease recovery and rerun,
cancellation rejecting late success, owner/pagination/revision/deletion boundaries,
and 200-entry retention; the seventh confirms temporary fixture cleanup. Reopening
SQLite does not test a Docker restart. These checks prove installed integration
contracts, not the browser, Cloudflare authentication, live worker or model quality.

The smoke normally takes seconds because model replies are scripted and fixtures
are tiny and local. `--json` prints a compact receipt; `--timeout-seconds` lowers
the ten-second per-case deadline. Linux forked cases have parent-enforced hard
deadlines, including blocked synchronous work; children stop before fixture
cleanup. Exit codes: 0 pass, 1 finding/timeout, 2 setup failure, 3 harness crash,
130 interrupted. Results print in the launching shell; no notifications or report
transfer. `AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER`
override the checkout/container.

Manual live acceptance needs no upload: start an Agent with “Create a downloadable
Markdown checklist for preparing a vegetable garden.” While it runs, open Recent
activity and confirm model/tool entries populate, then download the created file.
Reload the page and confirm saved history remains; Run again should keep the same
task and append a new attempt. A completed task alone does not establish useful
content: inspect the file. This separately checks real local-model/tool choice,
the running worker and browser/proxy wiring.

## Installed Agent file-read smoke

After pulling selected-file paging, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-agent-file-reads-onbox.sh
```

Application schema 37 applies automatically, preserving existing event history.
No settings change, credentials, `.env.smoke.local` or upload is needed. The
wrapper streams the checked-out probe into the backend container's installed
Python package. It uses a separate ASGI application, synthetic accounts and
temporary canonical databases/files; it does not open production account data
or call the running HTTP server. Model metadata and replies are scripted.

Expect seven `PASS` checks and `PASS Agent file read smoke passed` with `(exit 0)`.
The execution case uses the real runtime, executor and native dispatcher to read
a marker beyond the 1,500-token opening excerpt, request another page at the
returned cursor, create a real file and download its exact bytes. Other cases
check fixed input versions through reopened storage/rerun, selected-owner/Private
boundaries, source changes and expired callbacks, escaped Unicode/JSON page
boundaries and EOF, metadata-only read events, and temporary cleanup. Real OCR
or transcription generation, Docker restart, browser authentication, live worker
and actual Qwen comprehension are separate acceptance checks.

The smoke normally takes seconds because the fixtures are local and no inference
runs. `--json` prints a compact receipt; `--timeout-seconds` lowers the ten-second
per-case deadline. Linux forked cases have parent-enforced deadlines, including
blocked synchronous work, and children stop before fixture cleanup. Exit codes:
0 pass, 1 finding/timeout, 2 setup failure, 3 harness crash, 130 interrupted.
Results print in the launching shell; no notifications or report transfer.
`AUDREY_SMOKE_APPDATA` (or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override
the checkout/container.

For manual live acceptance, select an existing ordinary Project file long enough
to show a shortened opening excerpt. Assign a task asking for specific information
near the end of that file, then inspect both its answer and Read selected file
entries in Recent activity. Confirm the requested information against the source;
the presence of a tool call alone does not prove useful reading. No upload is
needed when a suitable file already exists; otherwise add one through the normal
Project file flow before testing.

## Installed Agent file-backed Knowledge smoke

After pulling file-backed approved Agent Knowledge, run on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
tests/smoke/smoke-agent-knowledge-files-onbox.sh
```

No schema or settings change, credentials, `.env.smoke.local` or upload is needed.
The wrapper streams the checked-out probe into the backend container's installed
Python package. It uses temporary canonical application/upload databases and
files, an isolated ASGI application, synthetic accounts and scripted local model
replies. It does not open production accounts, call the live HTTP server or run
inference.

Expect seven `PASS` checks and `PASS Agent Knowledge file smoke passed` with
`(exit 0)`. Cases verify current and mixed file-backed notes, note-only model
context with no implicit file-reading tool, restart/rerun version pins,
edit/delete/detach/reattach fences, owner/Private/stale/unsupported exclusions,
and explicit Assign to agent with its origin and entry state preserved. The last
check verifies fixture cleanup. The local execution case creates and downloads
a real deliverable from the approved note, exercising source freshness checks
under the publication locks. Saved text includes retained transcripts and
generated artifacts; this probe does not generate new OCR or transcripts.

The smoke normally takes seconds because no inference runs. `--json` prints a
compact receipt; `--timeout-seconds` lowers the ten-second per-case deadline.
Linux forked cases use parent-enforced deadlines, including blocked synchronous
work, and children stop before temporary cleanup. Exit codes: 0 pass, 1 finding
or timeout, 2 setup failure, 3 harness crash, 130 interrupted. Results print in
the launching shell; no notifications or report transfer. `AUDREY_SMOKE_APPDATA`
(or `APPDATA`) and `AUDREY_SMOKE_CONTAINER` override the checkout/container.

Live browser/worker/Qwen acceptance is separate: in an ordinary Project, attach a
current saved file reference to an approved open Action or Question, choose Assign
to agent, review the draft and Start. Check that the task uses the approved note,
the entry stays open on completion and input files are added only through explicit
selection. No upload is needed if a suitable Project file already exists.

## Service rebuilds

The user runs these actions on Tower from /mnt/user/appdata/audrey_ai_2.0 after
the normal pull. They describe how to deploy a change, not an outstanding queue.

| Changed files | Tower action |
|---|---|
| Documentation only | None |
| Backend Python or dependencies | `docker compose up -d --build audrey` |
| Browser source or assets | `docker compose up -d --build audrey-ui` |
| Shared KB code used by backend and tools | Rebuild `audrey` and `custom-tools` |
| Tools sidecar source | Rebuild `custom-tools`; also rebuild `audrey` when its code/declarations change |
| Media worker or fetcher source | Rebuild the corresponding `media-worker` or `media-fetcher` service |
| config.yaml or service environment | `docker compose up -d --force-recreate audrey`, using the affected service and adding `--build` when code also changes |
| Evaluation image contents/dependencies | `docker compose --profile eval build audrey-eval` |
| Monitoring source/provisioning | Follow [monitoring operations](../../monitoring/README.md) for the changed artifact |

Tool discovery refreshes automatically. A service restart alone does not reload
an updated bind-mounted configuration. Secrets and host-specific settings remain
in .env; do not edit the repository configuration directly on Tower.
