# Audrey live smoke testing

Read this before giving or running a live smoke command. Audrey development
happens in the laptop checkout; Tower is the Docker-only deployment host.

## Choose the runner first

| Test | Runner | Target | Credentials |
|---|---|---|---|
| Hermetic pytest, lint, or build | Laptop | Local checkout | None |
| API-only live smoke, including `smoke_native_auth_cutover.py` | Laptop over LAN/WARP | Tower backend port `8000` | Credential required by that script |
| Eval harness | Laptop over LAN/WARP | Tower backend `/v1` on port `8000` | Audrey PAT with `compat:full` |
| Direct Ollama/model probe | `scripts/probes/probe-onbox.sh` on Tower | Ollama over `ollama-net` from the Audrey container | None |
| Full native UI/proxy smoke | Disposable container on Tower, or laptop through an explicit SSH tunnel | `http://audrey-ui:8080` inside `ollama-net`, or tunneled loopback port `8090` | Native smoke user/admin assertions |

Choose the smallest proof that exercises the changed functionality and its
direct dependency or failure boundary. Do not rerun broad live suites merely
because they exist or because an older phase checklist names them. Reuse
current evidence for unchanged surfaces. Escalate to a broader smoke only when
the targeted check fails, the change crosses additional boundaries, or the
user explicitly asks for broader coverage. Hermetic laptop verification is a
separate required gate for code changes; it does not substitute for live proof.

For user-facing upload, Files, and chat workflows, acceptance is manual-first:
use a real file in the native browser and inspect the visible state and result.
A script is appropriate for a protocol boundary or diagnosis, but it does not
replace the browser flow the user actually relies on.

Current backend addresses:

- Working laptop route — LAN/WARP: `http://192.168.1.11:8000`
- Currently unreachable from the laptop — Tailscale: `http://100.113.157.98:8000`

Use the working `192.168.1.11` route in handed-off laptop smoke commands until
Tailscale reachability is explicitly re-established.

Tower does not have host Python, `uv`, or a repository `.venv`. Never hand the
user any of those commands at a Tower prompt. Conversely, do not move an
API-only smoke to Tower merely because Tower runs Docker; the laptop `.venv`
and VPN route are the normal path.

The standalone UI bind `127.0.0.1:8090` is Tower-loopback-only. A laptop cannot
reach it at either private IP without an SSH tunnel. Do not confuse that limit
with the published backend port `8000`.

## Run direct Ollama probes on Tower

A probe that needs Docker-only Ollama DNS, model residency, or GPU observations
runs through `scripts/probes/probe-onbox.sh` on Tower. The wrapper copies the
selected probe and any `COPY=` fixtures into the running Audrey container,
self-detaches, keeps its log in `testing-out/probes`, and prints that log path.
It does not require host Python or a repository mount inside the container.

Campaign 3 Phase 6's System One comparison is this kind of probe. It needs no
browser, Audrey token, upload, or laptop network route. Because it deliberately
unloads models for cold samples, run it while Audrey is idle. Its exact version,
model-install, runner, and success criteria are maintained in
`docs/campaign-3/phase-06-system-one-routing.md`.

### Clef and Clef Flash measurement — settled 2026-10-03

The broad Clef and two-candidate router probes completed on Ollama 0.35.1. Full
Clef matched all 69 production-reached routing samples but failed the cold-load
and residency gates. Clef Flash returned two production-reached timeouts during
cold loading. Audrey retains `qwen3.5:4b`; do not repeat these probes unless a
later Ollama or Clef release changes load time, residency, or model behavior.

The commands are retained for reproducibility. Run the broad probe first so it
unloads Clef, then the router comparison so the incumbent is warm at the end:

```bash
cd /mnt/user/appdata/audrey_ai_2.0

scripts/probes/probe-onbox.sh systemone_decision_probe.py \
  COPY=systemone_decision_cases.json \
  MODELS=clef:latest \
  ROUNDS=3

scripts/probes/probe-onbox.sh systemone_router_probe.py \
  COPY=systemone_router_cases.json \
  CANDIDATES=clef:latest,clef-flash:latest \
  ROUNDS=3
```

These probes require no browser, upload, or Audrey credential and do not change
application data or configuration. They do load and unload Ollama models, so
any future repeat must run while Audrey is idle.

## Keep credentials private and runner-specific

The laptop's gitignored `.env.test.local` is the permanent local credential
file for evals and native API smokes. It contains the direct Audrey eval URL,
the first-party personal token, and the two Cloudflare Access application
assertions:

```text
AUDREY_EVAL_BASE_URL=http://192.168.1.11:8000/v1
AUDREY_EVAL_API_KEY=aud_pat_...
AUDREY_USER_JWT=eyJ...
AUDREY_ADMIN_JWT=eyJ...
```

Use the working `http://192.168.1.11:8000/v1` route for laptop evals. Create
the personal token in native Audrey Settings with `compat:full` scope. Native smoke
commands source this file; the eval harness reads only its `AUDREY_EVAL_*`
entries. Keep it mode `600`. The variable entries are permanent, but expired
Access JWT values still need to be refreshed in place.

The Docker-only Tower fallback keeps a separate, gitignored
`.env.smoke.local`, because Docker's `--env-file` would otherwise pass the
unrelated eval PAT into the disposable container. That root-owned file contains
only these same two native credential names:

```text
AUDREY_USER_JWT=eyJ...
AUDREY_ADMIN_JWT=eyJ...
```

Both files use exactly `KEY=value`: no `export`, no spaces around `=`, no
Markdown link syntax, and no obsolete `OWUI_API_KEY`, `TEST_OWUI_TOKEN`, or
`ADMIN_OWUI_TOKEN` entries. Before telling the user to source a file, confirm it
exists and inspect key names without printing values.

Every native smoke script ignores legacy OWUI credentials. It requires a
current Cloudflare Access **application** JWT for the user in `AUDREY_USER_JWT`.
Obtain it either with `cloudflared access token` for Audrey's public application,
or from the `CF_Authorization` cookie on the protected Audrey hostname in browser
developer tools. Do not use the separate team-domain global-session cookie.
Treat the value like a password: never print it, commit it, or paste it into chat.

The broader native UI smokes require both `AUDREY_USER_JWT` and a distinct
`AUDREY_ADMIN_JWT`. Tower's `.env.smoke.local` must be root-owned or otherwise
private and mode `600`.

## Run the 2F.1 authentication smoke from the laptop

Use the working LAN/WARP route. This consumes the credential already stored
in `.env.test.local`; do not prompt for it again:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_native_auth_cutover.py
)
```

The subshell keeps the token out of shell history and removes it from the
environment when the command exits. Success is exit code zero, JSON ending in
`"status": "passed"`, `"legacy_bearer_rejected_locally": true`, and no
`cleanup_error`.

## Run the current built-in skill-registry smoke from the laptop

This is the targeted API-only deploy proof for the 3A/3C built-in registry. It
checks that the tracked `video-analysis` and `grounded-document-analysis`
bundles are the two available catalog entries, skill readiness is healthy, and
admin rediscovery reloads both without diagnostics. It creates no user data,
conversations, tokens, files, or model calls. Use the working LAN/WARP route:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_skills_foundation.py
)
```

Success is exit code zero and JSON ending in `"status": "passed"`, with
catalog and readiness `ready`, capabilities `available`, both built-in skill
ids, and zero rediscovery diagnostics. The earlier all-`disabled` output
remains the recorded live proof for 3A.1; do not expect that result after
deploying the enabled registry.

## Run the 3C grounded-document product evaluation from the laptop

This gate is settled. The initial controls passed 9/9. After the
document-reading repair, the repeated skill arm also passed 9/9 and human review
accepted all nine answers. Do not repeat it unless a later change touches the
document reader, file tools, skill prompt, or selection path. If one does, keep
the settled controls as evidence and rerun only the skill arm below.

The account still needs both files from
`evals/fixtures/grounded-document-analysis/` in Ready state:

- `c3-grounded-operations.md`
- `c3-grounded-support.md`

If they were removed after the first run, upload them again in the native
browser. Three whole passes produce nine repaired skill samples:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
.venv/bin/python evals/eval_research.py \
  --base-url http://192.168.1.11:8000/v1 \
  --cases evals/cases/eval_prompts_grounded_documents.json \
  --only skill \
  --repeat 3 \
  --save-file evals/results/2026-09-28-grounded-documents-reader-fix-answers.md \
  --save-json evals/results/2026-09-28-grounded-documents-reader-fix-results.json
```

The structural gate is all nine cases completing without failed checks. Read
all nine answers and confirm they use the document contents, keep facts
attributed to the correct file, and do not treat a read failure as evidence of
absence. The JSON record must show `grounded-document-analysis` for every
sample. This evaluation creates compatibility chat-history entries and leaves
the two uploaded fixtures in the account; remove them manually after preserving
the artifacts. The accepted 2026-09-28 run is the baseline for later changes.

## Run the explicit skill-selection smoke from the laptop

This is the targeted 3B deploy proof. It makes one short Fast model call with
`video-analysis` selected explicitly through the native AG-UI route, verifies
the streamed answer and persisted id/version/digest/reason, then deletes the
temporary canonical conversation and archive projection. Use the working LAN/WARP route:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_skill_selection.py
)
```

Success is exit code zero, JSON ending in
`"status": "passed"`, selection showing `video-analysis` version 1 with reason
`request`, a 64-character digest, and cleanup showing repair status `ready`.
This smoke mutates live data only for its temporary conversation and removes it
before returning.

## Run the 4A original-file download smoke from the laptop

This targeted API proof uploads one small text file through the native endpoint,
downloads its exact bytes, exercises a byte range, verifies another Audrey
account receives the same 404 as an unknown file, and deletes the upload.
It requires the distinct user and admin assertions already stored in
`.env.test.local`:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_file_download.py
)
```

Success is exit code zero and JSON ending in `"status": "passed"`, with full
HTTP 200, range HTTP 206, cross-owner HTTP 404, and cleanup repair status
`ready`. This smoke mutates live data only for its temporary upload and removes
it before returning. After rebuilding `audrey-ui`, one manual Files-dialog
download confirms that the built browser action reaches the proven endpoint.

## Run the 4B artifact-download smoke from the laptop

This is a read-only backend proof. It needs one processed video with at least
one real derived artifact: a summary, transcript, or visual notes.

### Prepare the account

1. Open the native Audrey browser as the same user represented by
   `AUDREY_USER_JWT`.
2. Open **Files** and look for a video whose status is **Ready**.
3. Choose **View text**. If Summary, Transcript, or Visual notes contains text,
   the account is ready and no upload is needed.
4. If no Ready video has text, upload or fetch a short video with clear speech
   or visible scene changes. Wait until its status becomes **Ready**, then
   confirm that at least one **View text** tab contains text.

A video that is still processing, failed before producing text, or has three
empty artifact tabs cannot prove this slice. The original video file may already
be reclaimed; the derived text is what this smoke reads.

### Run from the laptop

The script scans the account's Ready videos, preferring reclaimed originals,
and selects the first one with actual derived text:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_artifact_download.py
)
```

It pages all three artifact readers, compares each available download
byte-for-byte, checks its stable filename and private headers, and expects HTTP
404 for missing artifacts. It creates or deletes nothing.

Success is exit code zero and JSON with:

- `"status": "passed"`;
- `"available_count"` of at least one;
- `"file"` naming the selected video;
- HTTP 200 for each artifact marked `"available": true`;
- HTTP 404 for each artifact marked `"available": false`.

Use the filename in the `"file"` block for the browser check. Open that video
under **Files → View text**. Summary must have no download action or blank
action row. A non-empty Transcript or Visual notes tab must show its download
action and save a `.transcript.txt` or `.visual-notes.txt` file. An
empty tab must not show a download action. This browser check passed on
2026-09-29.

Set `AUDREY_ARTIFACT_SMOKE_FILE_ID` only when a particular owned video must be
tested. Normally the automatic scan is sufficient.

This path is for scripts that must exercise the standalone proxy. Use the
checked wrapper so malformed or legacy env-file entries fail with a useful
message before Docker starts:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash tests/smoke/smoke-native-onbox.sh smoke_native_ui.py
```

Success is exit code zero, `"status": "passed"`, and no `cleanup_error`.

## Run the 5A/5B Responses API smoke from the laptop

**Result:** Completed and streaming contracts passed over LAN/WARP by
2026-09-30. Do not repeat unless a later change touches their adapter, event
renderer, or shared authenticated generation path.

This targeted backend proof makes one short streamed `audrey_fast` model call
through `POST /v1/responses`, validates typed events, stable ids, text deltas,
terminal output, and token usage, then proves `background: true` fails
explicitly before generation. The model prompt uses Audrey's compatibility
utility form, so the call is excluded from chat history. It uploads and deletes
nothing.

The laptop's `.env.test.local` already contains the required
`AUDREY_EVAL_API_KEY`. Use the working LAN/WARP route:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_responses_api.py
)
```

Success is exit code zero and JSON ending in `"status": "passed"`. The
`streamed` block must report HTTP 200, a `resp_` id, a `msg_` id,
`response.completed`, and `"sentinel": true`. The `unsupported` block
must report HTTP 400 and `responses_feature_unsupported`.

## Run the 13A Responses multimodal smoke from the laptop

This targeted proof sends an in-memory red PNG as an inline `input_image`
beside an `input_text` part. It requires a completed typed response and then
proves an unsupported `file://` image URL is rejected before generation. Its
output ceiling is 4,096 tokens to leave room for vision reasoning. It
uploads, stores, and deletes nothing.

After rebuilding Audrey, use the working LAN/WARP route and the existing
`AUDREY_EVAL_API_KEY`:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_responses_multimodal.py
)
```

Success is exit code zero and JSON ending in `"status": "passed"`. The
`multimodal` block must report HTTP 200, a `resp_` id, output type
`output_text`, and `"sentinel": true`. The `unsupported` block must
report HTTP 422 for the unsupported URL and `"validation_error": true`.

## Run the 13B Responses structured output smoke from the laptop

**Result:** Passed and settled on 2026-10-05. Completed and streamed
json_schema output both returned HTTP 200 with the sentinel object, typed
output, and valid usage. The stream produced 16 JSON deltas across 24 events,
hid Audrey progress text, and ended with response.completed. Legacy
json_object mode returned the expected HTTP 400. Do not repeat unless a later
change touches Responses text formats, final model calls, or stream rendering.

This targeted protocol proof makes two short Fast model calls: one completed
and one streamed. Both must return exactly the schema-constrained sentinel
object. The script validates schema echo, typed output, token usage, streaming
event order, matching deltas, and the absence of Audrey progress text inside
the JSON. It then proves legacy json_object mode remains an explicit HTTP 400.
It uploads, stores, and deletes nothing.

After rebuilding Audrey, use the working LAN/WARP route and the existing
AUDREY_EVAL_API_KEY:

~~~bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_responses_structured.py
)
~~~

Success is exit code zero and JSON ending in status passed. Both completed and
streamed blocks must report HTTP 200, format json_schema, and sentinel true.
The streamed block must report response.completed and progress_hidden true.
The unsupported block must report HTTP 400 and
responses_feature_unsupported. No file upload or browser action is needed.

## Run the 13C Responses file reference smoke from the laptop

**Result:** Passed and settled on 2026-10-05 over
`http://192.168.1.11:8000`, based on the user's reported output. Completed
and streamed answers both read the document code and recognized RED.
Ownership, missing/deleted, and wrong-kind guards passed; both uploads
were deleted without cleanup errors. Commands below are retained for
reproducibility; repeat only when a later change touches this contract.

This tests the Responses API's new references to files already stored in
Audrey. No browser action, video upload, or existing Ready file is required.
The script creates and uploads one tiny text file with a randomized code and
one red PNG under the smoke user. It waits for both files to become Ready,
then makes one completed and one streamed Fast model call. Each answer must
read the document's code and recognize the image's red color. Each call has
a 4,096-token output ceiling so vision reasoning can finish before the short
visible answer. This is a ceiling, not a required answer length.

Both Cloudflare Access application assertions are required:
`AUDREY_USER_JWT` and `AUDREY_ADMIN_JWT`, already stored in the laptop's
`.env.test.local`. They must resolve to distinct active Audrey accounts. The
second account is used only to prove it cannot read the first account's files;
no admin settings, repair operation, or personal-token change is performed.
This smoke uses those assertions rather than `AUDREY_EVAL_API_KEY`.

First deploy the code on Tower from the updated checkout:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey
```

Then run this on the laptop using the working LAN/WARP address:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_responses_file_inputs.py
)
```

Allow several minutes for image indexing and the two model calls. Success is
exit code zero and JSON ending in `"status": "passed"`, with:

- `uploads.ready: true` and `uploads.count: 2`;
- completed and streamed HTTP 200, `document_sentinel: true`, `image_color: red`,
  a `resp_` identifier, and integer token usage;
- streamed `terminal: response.completed` and matching non-empty deltas;
- guard results HTTP 404 for foreign, missing, and deleted references,
  `same_not_found: true`, and HTTP 422 for the wrong file kind;
- `cleanup.uploads_deleted: 2` and no `cleanup_errors`.

The script deletes only its temporary uploads, including when a model call
fails. This protocol check is the acceptance gate for Slice 13C. Existing
native browser upload and chat acceptance remains recorded as passed.

**First attempt, 2026-10-05:** Stopped at the initial `GET /api/me` with HTTP
401, before uploads or model calls. Both saved laptop Access assertions had
expired on 2026-10-04. `uploads_deleted: 0` is expected because no uploads were
created. Refresh both values in `.env.test.local` from the `CF_Authorization`
application cookies on `ai.builtryte.xyz`, using separate signed-in user and
admin browser profiles, then rerun this same targeted smoke. A rebuild is not
needed for a credential refresh.

**Second attempt, 2026-10-05:** Authentication worked, both uploads became
Ready, and cleanup deleted both files. The first completed answer was empty;
the smoke stopped before its streamed and denial checks. The old harness
hid response usage and status details, so the live stop cause is unconfirmed.
Its 64-token ceiling could be exhausted by vision-model thinking. Existing
vision measurements in `config.yaml` show even 2,048 tokens can produce no
visible answer; the corrected harness uses the established 4,096-token ceiling.

The follow-up preserves the completed Fast model's token-limit stop reason:
Responses returns `status: incomplete` with `reason: max_output_tokens`,
preserving partial text and usage. An empty answer without that stop reason
returns HTTP 502. On failure, the smoke now retains response id, status,
usage, answer length, and any incomplete reason. Cleanup still runs.

**Third attempt, 2026-10-05: Passed.** Completed generation returned HTTP
200, a 25-character answer using both inputs, 1,376 input tokens, and 205
output tokens with a 4,096-token ceiling. Streaming returned HTTP 200,
15 deltas across 23 events, 1,178 input tokens, 179 output tokens, and
`response.completed`. All four guard results matched their expected HTTP
statuses, foreign and missing responses matched, and cleanup deleted both
uploads. No browser action or existing library file was needed.

## Run the 13D Responses remote-input smoke from the laptop

**Result, 2026-10-05:** The normal answer and both private-URL guards passed.
Streaming reached the 4,096-token ceiling and returned only Audrey's progress
banner while incorrectly reporting `completed`. The corrected laptop code
reports token-limited streams as `incomplete`, suppresses progress in Responses
answer text, and refuses an empty normal completion. Live streaming acceptance
still needs the single retry below; retain the passed checks.

This API-only slice lets callers provide public image/document URLs directly in
`POST /v1/responses`. You do not need to upload a PDF, image, or video in the
browser. The script uses a small [W3C sample PDF](https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf)
containing "Dummy PDF file" and the [Python logo](https://www.python.org/static/community_logos/python-logo.png).
The normal answer read both correctly: 21 visible characters, 1,375 input tokens,
and 692 output tokens. Both loopback probes returned JSON HTTP 422 before SSE.
The failed stream reported 1,177 input tokens and 4,096 output tokens, with no
visible answer beyond the 36-character banner. No uploads were created.

The script uses the existing `AUDREY_USER_JWT` in laptop `.env.test.local`.
It does not need an admin token or the eval PAT. It creates no uploads or
library entries and uses `### Task:` so model calls do not enter compatibility
chat history. Temporary downloads and parsers are cleaned up by the backend;
there is no operator cleanup step.

First rebuild on Tower from the updated checkout:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey
```

Then retry **only streaming** on the laptop:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_responses_remote_inputs.py --case streamed --max-output-tokens 8192
)
```

This sends one generation request and does not repeat the successful normal
answer or guards. The explicit 8,192-token ceiling gives the vision model more
room for reasoning before its short answer. This changes only the smoke request;
Audrey still honors the caller's output limit. It is a ceiling, not a required
token spend, and does not guarantee every model draw finishes. Existing vision
measurements show that `think: false` does not remove qwen3-vl's reasoning cost.
We retain the current model and production thinking policy.

Allow a few minutes for the one vision call. Success is exit zero and:

- `status: passed`, `case: streamed`, and `uploads_created: 0`;
- `streamed.http: 200`, `pdf_text: true`, and `image_logo: python`;
- `streamed.terminal: response.completed`, `progress_hidden: true`, and
  matching non-empty answer deltas;
- a `resp_` id and integer usage, with `max_output_tokens: 8192`.

If the model reaches that ceiling too, the smoke must fail with terminal
`response.incomplete`, `status: incomplete`, and
`incomplete_details.reason: max_output_tokens`. Those fields prove honest
termination, but do not pass the answer gate; retain the output for diagnosis.
[Official OpenAI documentation](https://developers.openai.com/api/docs/guides/reasoning)
specifies that the output ceiling includes reasoning and that exhaustion can
occur before visible text.

The default smoke still runs both answers and the guards with a 4,096-token
ceiling. `--case completed` or `--case streamed` selects just one generation
case; `--max-output-tokens` makes the request budget explicit. Neither selector
reports skipped cases as passed. Fixture fetch failures remain separate from
model-answer failures. Slice 13C remains live-settled and is not repeated.

## Run the 15B Projects restart smoke

**Result:** Passed and settled on 2026-10-02. Project, conversation membership,
and Ready-file reference persistence all survived restart. Deletion retained
the conversation with a null project id and left the referenced file readable;
cleanup removed every temporary record. Do not repeat this smoke unless a later
change touches schema 19, project ownership, membership persistence, or project
deletion.

This is the targeted live gate for schema 19 and the owner-scoped Projects API.
It needs two distinct current Access assertions in the laptop's
`.env.test.local` and at least one file already showing **Ready** in the smoke
user's **My Files**. If none exists, upload one small document in the browser
and wait for Ready before starting. The script references that file but never
changes or deletes it.

First, rebuild Audrey on Tower. Then run the capture step from the laptop over
the working LAN/WARP route:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_native_projects.py capture
)
```

Capture creates two temporary projects and one project conversation for the
ordinary user plus one temporary project for the admin user. It checks the
server-owned limits, pagination, moving and ungrouping the conversation,
cross-owner HTTP 404 responses, one Ready file reference, and duplicate HTTP
409 behavior. Success ends with `"status": "captured"` and tells you to restart
Audrey. Leave the laptop snapshot in place.

On Tower, restart only the backend and wait for it to become healthy:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose restart audrey
```

Then return to the same laptop checkout and run verification:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_native_projects.py verify
)
```

Success is exit code zero and JSON ending in `"status": "passed"`. All three
`persistence` values must be `true`. Deletion must report project HTTP 204,
project-read HTTP 404, a retained conversation with null `project_id`, and
retained-file HTTP 200. Verification deletes every temporary project and the
temporary conversation, then removes its local snapshot. It leaves the
pre-existing Ready file untouched.

If capture succeeded but you decide not to restart and verify, run the same
laptop command with `cleanup` in place of `capture`; it removes the temporary
records and snapshot.

## Phase 15 direct project-upload browser check

**Result:** Passed and settled. The real file picker, upload progress, responsive
layout, processing transition, project grounding, My Files controls, and hard
refresh behavior were accepted. The steps remain below for regression use only.

1. Open a project home. Confirm **Upload to project** is the primary action and
   **Choose from My Files** remains beside it.
2. Use **Upload to project** for one small `.txt` file. Confirm the button shows
   the filename and real percentage, then the file appears in the project and
   in **My Files** without another selection step.
3. Upload one PDF, audio file, or video that requires processing. Confirm the
   project row says **Processing**. Hard refresh while it is processing and
   confirm the row remains assigned to the project. Wait for **Ready**; the
   processing label should disappear without reopening the project.
4. Start a project conversation and ask about the uploaded file without
   attaching it to the message. Confirm Audrey can use it after it is Ready.
5. Open **My Files**. Confirm **Add files** is a compact highlighted card that
   still expands and collapses normally. Open it and confirm the native
   **Choose files** action is visibly styled, keyboard focus is clear, drag and
   drop still works, and upload results remain readable.
6. Repeat the project actions at a narrow viewport. The two actions should
   stack without clipping or horizontal scrolling.

The upload is a real user file and remains in My Files until the user removes
it. Removing it from the project alone must leave the My Files copy intact.

## Retired Phase 16 document tooling

The Audrey document-approval and Project Brief smokes were removed on
2026-10-05. Document and spreadsheet authoring belongs to the existing Hermes
Bot Tools MCP plus Nextcloud and Collabora workspace. Audrey has no recurring
document-generation acceptance gate.

After deploying the retirement, perform only this regression check: My Files
has no **Create a document** panel, `/api/document-jobs` returns HTTP 404, and
normal upload, inspection, Project selection, and grounded chat still work.

## Phase 7 PDF acceptance and diagnostic

**Result:** Passed and settled on 2026-09-30. The synthetic OCR boundary moved
an image-only PDF from Pending to Ready and returned recognized text. The user
then accepted real PDF upload, generated Summary, full Transcript, and a
grounded chat answer. Do not repeat these checks unless a later change touches
OCR dispatch, PDF summary generation, document reading, or chat grounding.

`tests/smoke/smoke_scanned_pdf_ocr.py` remains an optional worker diagnostic;
it is not a recurring acceptance test.

## Phase 8 audio acceptance

**Slice 8A result:** Passed and settled on 2026-09-30. MP3 upload and processing,
Summary and Transcript presentation, grounded chat, and attachment persistence
after refresh all passed in the native browser. The earlier synthetic attempt
also exposed the faster-whisper/PyAV mismatch that is now fixed by `av<19` and
the worker image's real-WAV decoder gate. Do not rerun the MP3 script unless a
later failure needs that protocol diagnostic.

**Slice 8B result:** Pending after deploying the laptop-complete WAV, M4A, and
FLAC allowlist. This slice changes only the Audrey backend; rebuild `audrey`.
Use the native browser with short spoken recordings:

1. Upload one `.wav`, one `.m4a`, and one `.flac` in **Files**. Confirm all are
   accepted, labeled **Audio**, and move from **Transcribing** to **Ready**.
2. Open the M4A recording. Confirm Summary and Transcript exist, Visual notes
   is absent, and the transcript matches the speech.
3. Attach the M4A file to chat, ask about one distinctive spoken fact, and
   confirm Audrey answers from it.
4. Hard refresh and confirm the saved attachment remains.

WAV and FLAC need admission and Ready-state checks. M4A carries the full viewer
and chat acceptance because all three formats join the same queue after the
byte-sniff gate, and hermetic tests run real ffmpeg decoding for every one.

## Before handing over any smoke command

1. Name the machine: laptop or Tower.
2. Classify the test: backend API, eval, or standalone UI/proxy.
3. Confirm the target is reachable from that machine.
4. Confirm the credential source exists and has the keys the script reads.
5. Read the script's environment-variable names instead of recalling them.
6. Keep URLs in code blocks as plain URLs, never Markdown link syntax.
7. State the expected success evidence and whether the script mutates live data.
8. Never claim an Unraid smoke passed until the user provides its output.
9. Never ask for a credential interactively when the chosen procedure already
   stores it in a private env file.

If a command fails with missing `uv`, `.venv/bin/python`, or `python`, first
check whether it was mistakenly aimed at Tower. If it fails because an env file
is missing, inspect the actual checkout before proposing another filename.
