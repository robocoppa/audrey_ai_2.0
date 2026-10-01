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

## Run the 5A Responses API smoke from the laptop

**Result:** Passed over LAN/WARP on 2026-09-29. Do not repeat unless a later
change touches the Responses adapter or the shared authenticated generation
path.

This targeted backend proof makes one short `audrey_fast` model call through
`POST /v1/responses`, validates the completed typed output and token usage,
then proves `stream: true` fails explicitly before generation. The model
prompt uses Audrey's compatibility utility form, so the call is excluded from
chat history. It uploads and deletes nothing.

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
`completed` block must report HTTP 200, model `audrey_fast`, output
type `output_text`, and `"sentinel": true`. The `unsupported`
block must report HTTP 400 and `responses_feature_unsupported`.

## Phase 7 PDF acceptance and diagnostic

**Result:** The OCR boundary and the original PDF upload/chat experience passed
on 2026-09-30. Do not repeat them unless a later change touches OCR dispatch,
document indexing, the reader, or chat grounding.

The corrected synthetic proof moved an image-only PDF from Pending to Ready in
4.619 seconds, returned page-marked recognized text, and cleaned up. The user
then uploaded a real PDF in the native browser, confirmed the document was
readable, and received a grounded answer about an uploaded document in chat.

A later presentation follow-up adds a generated PDF Summary and a Transcript
tab containing all extracted or OCR text. After rebuilding `audrey` and
`audrey-ui`, its remaining check is manual:

1. Upload one new real PDF. Existing Ready PDFs are not backfilled.
2. Wait for Ready, open **View text**, and confirm **Summary** opens first with a
   natural two or three sentence description.
3. Open **Transcript** and confirm it contains the document's full text.
4. Ask one question answered by that document and confirm Audrey grounds its
   answer in the file.

The already-passed `tests/smoke/smoke_scanned_pdf_ocr.py` remains an optional
worker diagnostic. It should not be rerun for this presentation-only follow-up.

## Phase 8A MP3 manual acceptance and optional diagnostic

**Result:** Pending after deployment. The first scripted attempt correctly
reached the worker, where it exposed a faster-whisper/PyAV decoder mismatch.
The worker now pins `av<19` and decodes a real WAV during its image build.

Rebuild `audrey`, `audrey-ui`, `custom-tools`, and `media-worker`. Then perform
the primary acceptance check in the native browser with a short spoken MP3:

1. Upload the MP3 in **Files** and confirm **Audio** plus **Transcribing** become
   **Ready**.
2. Filter to **Audio** and open **View text**. Confirm Summary and Transcript
   exist, the transcript matches the recording, and Visual notes is absent.
3. Confirm the summary is a natural two or three sentence description.
4. Attach the Ready MP3 to chat, ask about one distinctive spoken fact, and
   confirm Audrey answers from it.
5. Hard refresh and confirm the saved message retains the attachment.

If that flow fails and the failing boundary is unclear, run the optional
synthetic diagnostic from the laptop checkout over LAN/WARP:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_audio_ingest.py
)
```

It creates and removes its own MP3. Success is exit code zero and JSON ending
in `"status": "passed"`: kind audio, Pending to Ready, positive chunks and
duration, Whisper transcript containing its known words, a nonempty summary,
HTTP 422 for a visual artifact, and clean deletion/repair.

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
