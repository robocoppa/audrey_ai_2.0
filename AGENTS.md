# Working In This Repo

This file is the tracked, tool-agnostic agent guide for Audrey. Read it before
editing, then use the laptop-local project-state file for the freshest state.

## First Read

At the start of a new session, read these in order when they exist:

1. `docs/PROJECT_STATE.md` - gitignored current priority, active work,
   verified stack state, recent decisions, and followups.
2. `docs/lessons/AUDIT.md` - gitignored lesson audit queue, only when writing
   lessons or acting on lesson audit findings.

`docs/campaign-1/HISTORY.md` and `docs/lessons/CONTINUITY.md` are archive
files. Consult them only when `PROJECT_STATE.md` points there or you need old
phase/lesson archaeology.

`CLAUDE.md` is a local Claude-compatibility shim and should point back here.
Keep durable workflow rules in this file, not in tool-specific side files.

## Project Shape

Audrey is a self-hosted, OpenAI-compatible FastAPI/LangGraph orchestrator. It
routes requests across local Ollama models and Ollama-cloud bridge models using:

```text
datetime -> memory_recall -> classify -> complexity
  -> fast_path (+ ReAct tools) | planner -> deep_panel -> synthesize -> reflect
```

Current public client is Open WebUI. Audrey exposes `/v1/chat/completions` and
five virtual models: `audrey_auto`, `audrey_fast`, `audrey_deep`,
`audrey_cloud`, and `audrey_local`. Treat `config.yaml` as the source of truth
for model registry, routing pools, agentic behavior, timeouts, KB paths, and
tool servers.

## Development Commands

Use the existing `uv` workflow:

```bash
uv sync --extra dev
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
uv run audrey
uv run audrey-ingest --source /path/to/docs --topic geology
```

The pytest suite is designed to be hermetic and offline; it should not need
Ollama, Qdrant, Open WebUI, or custom-tools.

## Deployment Boundaries

The deploy target is Bart's Unraid server. Agents work on the laptop repo; Bart
pulls and deploys on Unraid. Do not run write-side Docker or Unraid operations
unless Bart explicitly asks and the command is safe in the current environment.

Root `compose.yaml` manages only the services rebuilt often:

- `audrey-ai`
- `custom-tools`

`monitoring/compose.yaml` manages Prometheus and Grafana config, while their
persistent state lives under `/mnt/user/appdata/prometheus/...`.

Ollama, Qdrant, Open WebUI, and cloudflared are managed outside the root compose
flow. All containers use the external `ollama-net` network. Open WebUI is the
public surface through cloudflared; Audrey, custom-tools, Qdrant, Ollama,
Prometheus, and Grafana are internal or LAN-only unless HISTORY says otherwise.

On Unraid, the normal deploy commands are run from
`/mnt/user/appdata/audrey_ai_2.0`:

```bash
docker compose up -d --build audrey-ai
docker compose up -d --build custom-tools
docker compose logs -f audrey-ai
```

For monitoring:

```bash
cd /mnt/user/appdata/audrey_ai_2.0/monitoring
docker compose up -d
```

## Runtime Rules Worth Preserving

- `config.yaml` wins for `tools.servers` and `kb.dataset_paths` unless the
  matching env var is explicitly set. Do not reintroduce compose defaults that
  silently overwrite those YAML lists.
- Audrey discovers tools once at startup from custom-tools `/openapi.json`.
  Preserve the custom-tools healthcheck and `depends_on` ordering. If Audrey
  ever boots with `tools=0`, the admin rediscover route can rehydrate the live
  registry.
- Do not add `require_user` to KB query routes that custom-tools calls over
  docker DNS; internal ReAct tool dispatches do not carry browser auth headers.
- Keep user-scoped tools user-scoped. Audrey should override model-supplied
  `user` values with the authenticated pipeline user for memory/user data.
- Global KB reconcile covers global text/image collections only. Per-user
  upload collections are managed by the uploads SQLite/Qdrant reconciliation.
- CLIP text-to-image scores can look low and still be correct; do not treat
  normal low cosine values as a bug without an end-to-end check.
- Container DNS names such as `custom-tools` work inside `ollama-net`, not from
  the Unraid host shell. From the host, use mapped localhost ports or `docker
  exec` into a container on the network.

## Lesson Workflow

The lessons teach Bart the codebase so he can maintain it solo. Bart's Python
level is basic; explain async, FastAPI, LangGraph, Pydantic, and similar
concepts from concrete Audrey code instead of assuming them.

When working on lessons:

- Read `docs/PROJECT_STATE.md` and `docs/lessons/AUDIT.md` first.
- Audit the lesson's files before writing, but put findings in
  `docs/lessons/AUDIT.md`, not in the published lesson.
- Do not make code changes from a lesson audit without explicit approval.
- Lessons 4 and later use three sections: Context, Read-along, Comprehension
  questions.
- Use file/line markdown citations where helpful.
- Do not use real email addresses in lessons; use `alice@example.com`.
- Do not mention "Phase N" in public lesson prose; describe the feature or bug
  by substance instead.
- Do not bake exact codebase/KB counts or sizes into lessons; use ballpark
  wording unless the count is a file/line citation.

After editing any file under `src/audrey/`, `tools-server/`, or `config.yaml`,
run `scripts/check-lesson-links.py <changed file>` for each changed file. The
checker compares each lesson cite's line against the displayed code snippet
beneath it, and when they disagree, proposes the correct line number. Apply
the proposed fix. For cites without a nearby snippet, the checker falls back
to a landmark heuristic and emits a soft "DRIFT?" hint — eyeball those.

The script's `DRIFT` findings include a concrete `fix:` line you can apply
directly; `DRIFT?` findings are advisory and may be deliberate "into the
body" cites. Run without arguments to audit all lessons, or pass changed
source paths to scope the check to cites that target those files.

When a session changes current priorities, verified stack state, or active
followups, update `docs/PROJECT_STATE.md`. When a lesson concludes, also update
the lesson archive if it is still useful. When future phase work concludes,
archive the deep details in `docs/campaign-1/HISTORY.md` only if they are worth
preserving.

## Git And Change Hygiene

Bart runs git write operations. Do not run `git commit`, `git push`,
`git rebase`, or destructive history/worktree commands unless Bart explicitly
asks for that exact operation.

Before editing, check the worktree and preserve user changes. At the time this
guide was added, `docs/lessons/lesson-05-configuration-and-startup.md` had
local modifications; treat unrelated dirty files as user work.

Whenever you ship repo changes (code, docs, config, deployment notes, tests, or
agent instructions), always include a suggested git commit message in the final
response, even if Bart did not ask for one in that turn. Use a
conventional-commit style message in a fenced block:

```text
docs: add repository agent guide
```

Use one short message per concern. No `Commit:` prefix and no trailing period.
For pure Q&A or investigation with no file changes, no commit message is needed.
