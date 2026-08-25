```text
###########*:    ::.     :--:......:     :*###################################
###########:     ..    .---:.           ::-+##################################
##########=           .:..:.              . :=################################
##########-         ..:--==-    .. ...        -*##############################
##########:         .:-+==-.    . ....         =##############################
#########*.       ..:==++=:..  ..:::...:....:::+##############################
##########+        .:-==-::---:...  .=***=:......+############################
##########-          .:-==------::::-=*###+:-.:- .############################
##########-         .-*##*+*##***#***######*+=:--+############################
##########+        .=*###++++=====*######+=-::.:*#############################
##########-       :+###**###***#*++#####+--===:-*#############################
##########+--:.   +######+=:.:--+#####+-:.::.::*##############################
##########*+*#=..-*######*-::.=+=+###*=:-.:+::=###############################
###########*#+=***###########*########**##*+*++###############################
###########*+****######################**####++###############################
############*+-:-=+################*+*#+=####=################################
#############*-. .-###############*#**=-=##*++################################
##############*+=--*##################*+****+*################################
#################-.-*###########*=--:.::-=+=*#################################
#################*+==+*#########*+++===-..:-+***##############################
###################+*++=+#########*++++-..-#*-**=-*###########################
###################*+##*+-:=++*####***+:.:-*#=##*=-++*++*#####################
###################+.+###**=-:-:----:=*+-=##*=##*-+*#*=++==**#################
###################=..+#####**=-....-##*-=##++##+***#*---=++++==+*############
##############+++#*=.-=+#######*+-:-*##*+###*###+*##+.     .:+***=--=++#######
###########*=:  -*=  ...=#######*++################=.   . .:----*##*++=-==*###
######*==:.    .-::    .:*####*++#############**#*-  .:-+****+**+=*######+==-=
###*++*.       .-.--.   .:-+*+=*#########****+=-:. .=+*#######***==*########**
#*+*##*        :-.:-.      .  =########***+-:.   .=*###########***=-##########
*=*####:         - .==:   ..:.=######**+=-     .-*#############***+:=#########
+*#####=.           :.=--:-:=-*######*+=.     .=*#############****+-:*########
+*#####*:       . -:     .:..-#######*+.      =*#############***+==:.+########
++#####*+.     .: .. -:     .+#######*+.     .**###########****+===  +########
*=######*-         :..-=.:: -########*+.     :**##########****+=+-.  +########
```

# Audrey AI 2.0

Self-hosted multi-model LangGraph orchestrator that exposes an OpenAI-compatible
`/v1/chat/completions` endpoint, runs on Unraid with dual RTX 3090 Ti GPUs, and
routes requests across local Ollama models + cloud-bridge models through a
classify → route → tool-call → reflect pipeline.

## High-level architecture

```
Open WebUI ──[Cloudflare Tunnel]──> audrey-ai ─────> Ollama (local + cloud)
                                       │             │
                                       │             └─ both 3090 Ti GPUs
                                       │
                                       ├─> custom-tools (7 OpenAPI endpoints)
                                       │     web_search, kb_search,
                                       │     kb_image_search, memory_store,
                                       │     memory_recall, memory_search,
                                       │     chat_history_search
                                       │
                                       ├─> Qdrant (text vectors + CLIP images,
                                       │   per-user collections + global KB
                                       │   + per-user chat archive)
                                       │
                                       └─> KB watcher (event-driven re-ingest)
                                           + reconcile (periodic orphan sweep,
                                             startup + 30 min cadence)

                                  + Prometheus + Grafana (metrics + alerts,
                                    dashboards provisioned from monitoring/)
                                  + OWUI-backed auth on all routes
```

All containers sit on Docker network `ollama-net`. Only Open WebUI is exposed
publicly via the Cloudflare Tunnel; every other service is internal.

## Virtual models

Five virtual models, each a different routing mode over the same registry:

| Model            | Routing                                                                  |
|------------------|--------------------------------------------------------------------------|
| `audrey_auto`    | Adaptive: fast path on short prompts, deep panel on long ones.           |
| `audrey_fast`    | Always fast path; never escalates, even on long prompts.                 |
| `audrey_deep`    | Always deep panel, mixed cloud + local pool.                             |
| `audrey_cloud`   | Always deep panel, cloud-only pool (up to 3 parallel workers).           |
| `audrey_local`   | Always deep panel, local-only pool (serialized through GPU gate).        |

Streaming responses (when the client sends `stream: true`) emit progress
banners during each phase (Thinking / Planning / Dispatching / Synthesizing)
and a per-worker tools-used footer after the answer.

## Chat Completions compatibility

Audrey implements a deliberate subset of the OpenAI Chat Completions contract.
OpenAI-compatible here means that supported request and response shapes use the
same wire format; it does not mean every OpenAI generation option has identical
semantics.

Message compatibility:

| Role | Support |
|---|---|
| `system` | Text or content parts; forwarded as an instruction. |
| `developer` | Text or content parts; translated to an Ollama `system` instruction. Put instruction messages before conversation turns. |
| `user` | Text or multimodal content parts, including inline images. |
| `assistant` | Text may be null or omitted when function `tool_calls` are present. Call ids and JSON-string arguments are validated. |
| `tool` | Requires `tool_call_id`; Audrey resolves it to the matching Ollama `tool_name`. |

Unknown fields inside a message are rejected with HTTP 422. The OWUI
per-message `metadata` extension is accepted for archive identity but excluded
from model requests.

Top-level request compatibility:

| Field | Behavior |
|---|---|
| `model`, `messages`, `stream` | Supported. |
| `temperature`, `top_p`, `max_tokens` | Translated to Ollama generation options. |
| `tools` | Forwarded only for `audrey_passthrough/<concrete>`; pipeline models use Audrey-managed tools. |
| `user` | Accepted but never trusted for identity; the authenticated OWUI user wins. |
| `think` | Audrey passthrough extension; applied only when the concrete model declares thinking support. |
| OWUI `chat_id` / `metadata` | Read from the raw request for archive stitching, not forwarded to a model. |
| `tool_choice`, `parallel_tool_calls`, `response_format`, `seed`, `stop`, penalties, `n`, `logprobs`, `stream_options`, and other unmodelled fields | Accepted for client forward-compatibility but currently ignored. Do not rely on them. |

Function tools are the supported tool type. Assistant call/result relationships
round-trip through passthrough in streaming and non-streaming mode; custom and
legacy function-call message forms are not implemented.

## Pipeline shape

```
datetime injection
  → memory recall (per-user, semantic via custom-tools)
  → classify (keyword pre-filter + qwen3:4b router)
  → complexity gate (OWUI-task detect → virtual-model force → token threshold)
  → fast path (single model + ReAct if tool-capable)
    | planner (optional sub-question decomposition)
    → deep panel (N parallel workers, all tool-capable)
    → synth (cloud or local, streamed)
  → reflect (length + brevity-cue check)
  → retry on failure
```

Per-user fair scheduling (`FairLocalGate`) round-robins across users at the
local-GPU bottleneck. Per-user in-flight cap (default 3) prevents one user
from saturating the queue. Cloud calls bypass the gate. Every chat is captured
to the per-user chat archive (SQLite source-of-truth + Qdrant index) so the
model can search prior conversations via `chat_history_search`.

## Documentation

- **`AGENTS.md`** — start here. Tool-agnostic agent guide: project shape,
  runtime rules, deploy boundaries, git/change hygiene.
- **`docs/lesson-ai/`** — codebase walk-through (maintainer course). Lesson 4
  is the end-to-end request lifecycle; later lessons drill into specific
  subsystems (model layer, classify+route, deep mode, ReAct, KB
  ingest/lifecycle, fair scheduling, the OpenAI-compatible routes, the
  custom-tools sidecar). `docs/lesson-python/` is a separate beginner course
  that teaches Python by reading the same real codebase.
- **`docs/campaign-1/phase-N-deploy.md`** — original 31-phase build history.
- **`docs/campaign-2/phase-N-deploy.md`** — post-1.0 feature work
  (chat archive, prompt centralization, KB audit fixes, complexity-gate
  fixes, Thinking/Planning banners, Grafana dashboards).
- **`docs/unraid-ollama.md`** — clean Ollama container recreation on Unraid.
- **`monitoring/README.md`** — Prometheus + Grafana stack, including how to
  add a new dashboard via repo-managed provisioning.

## Dev workflow

```bash
# On the laptop
uv sync --extra dev          # install deps + pytest
.venv/bin/pytest tests/ -q   # run the hermetic test suite

# Manual run (rare — Unraid is the deploy target)
uv run audrey                # start FastAPI on :8000
uv run audrey-ingest --source /path/to/docs --topic geology
```

## Repo layout

```
src/audrey/         # orchestrator package (FastAPI + LangGraph)
tools-server/       # custom-tools FastAPI service (separate package)
tests/              # pytest suite — hermetic, no Ollama/Qdrant needed
docker/             # Dockerfiles (audrey-ai + custom-tools, base images pinned to digest)
monitoring/         # prometheus + grafana compose, scrape config, alert rules, provisioned dashboards
docs/               # campaign histories, lessons, deploy guides
images/             # screenshots used by docs
scripts/            # model-pull, smoke tests, lesson-cite link checker
config.yaml         # model registry, fast_path, deep_panel*, fairness, KB, reconcile
.env.example        # BRAVE_API_KEY, GRAFANA_ADMIN_PASSWORD, etc.
compose.yaml        # audrey-ai + custom-tools (ollama/qdrant/owui stay on Unraid UI)
AGENTS.md           # canonical agent guide
```

## Status

Active development. Campaign 1 (phases 1-31) shipped the core orchestrator;
Campaign 2 is post-1.0 feature work. See `docs/PROJECT_STATE.md`
(gitignored, laptop-local) for the current priority, verified stack state,
and per-phase behavioral facts.
