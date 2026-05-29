# Campaign 2 Phase 13 — Passthrough virtual model

Brings direct-Ollama LAN clients (OpenClaw on home-network desktops)
under Audrey's two fair-scheduling layers. Before this phase those
clients bypassed `FairLocalGate` and `UserInflightRegistry` entirely
and competed with Audrey users inside Ollama's FIFO queue with no
per-user fairness. After this phase they route through Audrey,
authenticate as a regular OWUI user, and contend for the local GPU
on the same terms as pipeline traffic.

## What it does

New virtual model selected by a prefix in the OpenAI `model` field:

  - `audrey_passthrough/<concrete>` — forward the chat directly to
    Ollama, no classifier, no complexity gate, no banners. Both
    fair-scheduling layers fire.

The prefix scheme lets a client pick the concrete model per
request without needing custom headers. `/v1/models` advertises one
entry per allowed concrete model under the prefix, so OpenAI-shaped
clients can present a model dropdown without out-of-band knowledge.

Two concrete models ship as allowed in `config.yaml`:

  - `qwen3.6:35b-64k` (23 GB) — general purpose
  - `qwen3-coder-next-64k:latest` (51 GB) — code-focused

Both are 64k-context variants of models the box already has loaded.
Adding more later is a one-line change in `config.yaml`'s
`passthrough.allowed_models` — no code change.

## Why this exists

`FairLocalGate` and `UserInflightRegistry` only fire when a request
goes through Audrey's pipeline. A LAN client that points directly
at Ollama's `:11434` skips both. Three of those clients running at
once would stall Audrey's deep requests at random, and neither side
could see why — they're all just FIFO-equal callers to Ollama.

Routing the LAN clients through `audrey_passthrough/<concrete>`
fixes this without paying for orchestration overhead. The classifier
adds ~50-200ms per request and would pick a model from the registry
based on task type — neither is what a passthrough caller wants.
Skipping those gives you "send this exact prompt to that exact
model" with fair scheduling layered on, and nothing else.

## Why `tools` must round-trip

Agent clients (Hermes, OpenClaw) advertise their own tools to the
model via the OpenAI `tools` array. If Audrey strips this on the
way through, the model sees a system prompt telling it "you have
access to read_file, write_file, …" but receives no schema for
how to call them. It can't issue structured `tool_calls`, so it
falls back to **emitting the tool syntax as plain text** like
`<read_file(path='…')>`. The agent client expects structured
calls, doesn't parse the text form, and forwards the raw model
output to the end user (Telegram, etc.). Result: tool syntax
leaking into chat output instead of actual tool execution.

The fix is to forward `tools` verbatim and reshape Ollama's
response `tool_calls` (which arrive as `{"function": {"name":
…, "arguments": {dict}}}`) into the OpenAI shape (`{"id": …,
"type": "function", "function": {"name": …, "arguments":
"<json-string>"}}`). The arguments-shape mismatch (dict vs
JSON-string) is the main thing — clients call `json.loads` on
the field and crash on a dict.

This applies to passthrough only. Audrey's pipeline modes
(`audrey_fast`, `audrey_deep`, …) use the server-side tool
registry from `tools/discovery.py` and ignore `payload.tools`
on purpose — they're using Audrey's tools, not the client's.

## What's in scope

  - **[`src/audrey/routes/openai.py`](../../src/audrey/routes/openai.py)** —
    prefix parser (`_is_passthrough`, `_passthrough_concrete`),
    config-driven validation (`_resolve_passthrough_model`), the
    route entry (`_handle_passthrough`), the streaming SSE emitter
    (`_passthrough_stream_sse`), the `/v1/models` listing expanded
    to include passthrough variants, and an
    `_ollama_to_openai_tool_calls` shape converter for the
    tool-forwarding path. `ChatCompletionRequest` gains a `tools`
    field — honored only on passthrough.
  - **[`src/audrey/pipeline/passthrough.py`](../../src/audrey/pipeline/passthrough.py)** —
    new module. Two thin helpers (`passthrough_chat`,
    `passthrough_stream`) that hold the GPU gate around the actual
    Ollama call, forward `tools` verbatim, and increment
    `audrey_dispatch_total` with `task_type="passthrough"`.
  - **[`src/audrey/models/ollama.py`](../../src/audrey/models/ollama.py)** —
    `chat_stream` gained an optional `tools` kwarg matching `chat`'s
    surface; needed so streaming passthrough can advertise tools.
  - **[`config.yaml`](../../config.yaml)** — new `passthrough` block
    next to `fairness`: `enabled`, `allowed_models`, `require_role`.
  - **[`tests/test_passthrough_route.py`](../../tests/test_passthrough_route.py)** —
    19 tests covering prefix parsing, the resolver's gating
    decisions (disabled, missing config block, allowlist enforcement,
    role gate), and the `/v1/models` prefix expansion.
  - **[`tests/test_passthrough_dispatch.py`](../../tests/test_passthrough_dispatch.py)** —
    13 tests covering the dispatch path end-to-end. Stubs the Ollama
    client so no network is involved; asserts the gate is acquired
    with the right concrete model + user, the inflight slot is held
    around the call, OpenAI-shaped non-streaming response is built
    correctly, streaming emits the expected SSE sequence, `OllamaError`
    surfaces as HTTP 502, the configuration gates fire before Ollama
    is reached, AND (the four tool-forwarding tests) `tools` flows
    through to Ollama on both non-streaming and streaming, Ollama's
    `tool_calls` get reshaped to OpenAI form (arguments as JSON
    string, synthetic call id, `finish_reason="tool_calls"`), and
    omitting `tools` produces a vanilla stop response with no
    `tool_calls` field on the assistant message.

No metric additions — the existing `pipeline_seconds`, `pipeline_total`,
and `dispatch_total` histograms gain `mode="passthrough"`,
`task_type="passthrough"`, and `path="passthrough"`/`path="passthrough_stream"`
labels through normal usage. Gate-wait and inflight-blocked metrics
fire automatically because the underlying layers haven't changed.

## What's not in scope

  - **OWUI user creation.** You create the OpenClaw user(s) in OWUI's
    Admin panel and mint the API key (`sk-…`) from
    Settings → Account → API Keys. Audrey's auth layer already
    accepts both OWUI JWTs and `sk-…` keys via the same
    `Authorization: Bearer` header — no auth.py changes needed.
  - **OpenClaw deploy.** This phase ships the Audrey-side surface
    only. Switching OpenClaw over to point at Audrey is a config
    edit on the OpenClaw side; see `passthrough-virtual-model-plan.md`
    for the YAML shape.
  - **Embeddings passthrough.** `/api/embeddings` is not exposed.
    Audrey has its own embed surface for KB ingest; if OpenClaw
    ever needs raw embeddings, that's a separate phase.
  - **Per-machine identity.** All LAN clients sharing one OWUI user
    share one inflight bucket (3 slots). The gate still round-robins
    across users so this is fine for the current scale; if you ever
    want per-machine bucketing, mint one user per machine.

## How it composes with the fair-scheduling layers

Per-request shape:

  1. `require_user` validates the bearer token (cached 30s).
  2. `_is_passthrough(model)` recognizes the prefix → branch to
     `_handle_passthrough`.
  3. `_resolve_passthrough_model` checks `passthrough.enabled`,
     `require_role`, and `allowed_models`. Returns the concrete
     model + its registry-declared location.
  4. `async with inflight.slot(user_id)` — same wrap pipeline
     traffic uses. Holds for the entire passthrough call (including
     the full stream for streaming requests).
  5. `passthrough_chat` / `passthrough_stream` acquires the GPU
     gate around the Ollama call. For streaming, the gate is held
     across the whole stream — releasing early would let another
     caller's request enter Ollama while ours is still generating,
     defeating the fairness goal.
  6. Response is shaped into the OpenAI chat-completion contract
     using the same helpers fast-mode uses (`_to_openai_response`).

A request from OpenClaw and a deep request from a human user now
round-robin at the gate. The 64k variants exceed VRAM if both were
loaded at once on the 24 GB 3090 Ti, so Ollama will swap between
them as requests alternate — that's a model-load latency cost, not
a fairness one. The first request after a swap sees the load
penalty; subsequent calls to the same model hit the warm cache.

## Verification

Hermetic:

```
.venv/bin/pytest tests/ -q          # 432 passing (28 new)
.venv/bin/ruff check .              # clean on touched files
.venv/bin/python scripts/check-lesson-links.py src/audrey/routes/openai.py
                                    # zero confident drift
```

Cross-lesson cite sweep: line shifts in `routes/openai.py` from the
new passthrough surface required four cite updates in
lessons 4, 7, and 13 (twice).

## Deploy on Unraid

Standard build + restart:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

Watch the boot line — `gpu_concurrency=1; max_inflight_per_user=3`
is unchanged; the passthrough surface is on by default per
`config.yaml`. Verify the listing:

```
curl -s https://audrey.your-domain/v1/models | jq '.data[].id'
```

Should include both `audrey_passthrough/qwen3.6:35b-64k` and
`audrey_passthrough/qwen3-coder-next-64k:latest` next to the
existing `audrey_*` virtual models.

## Followup wiring (separate from this phase)

### Shared prerequisites for every direct-Ollama client

  1. **Create an OWUI user for the client.** OWUI Admin → Users →
     add a regular-role user (e.g. `openclaw@local`,
     `hermes@local`, or one shared `passthrough@local`). Email is
     required by OWUI; the local-part doesn't have to resolve.
  2. **Mint an API key for that user.** Log in as the user once
     in a private window → Settings → Account → API Keys → Create
     new key. Copy the `sk-…` value.
  3. **Pick the base URL** OWUI's docs call "OpenAI-compatible
     endpoint" — this is Audrey's `/v1`. Three shapes depending
     on where the client runs:
       - **Public via cloudflared:** `https://audrey.your-domain/v1`
       - **LAN-direct to Unraid:** `http://192.168.1.11:8000/v1`
       - **Same host as Audrey:** `http://localhost:8000/v1`.
  4. **Pick the model string.** The full prefixed form is what
     Audrey expects in the OpenAI `model` field. From OWUI/curl:
     `audrey_passthrough/qwen3.6:35b-64k` or
     `audrey_passthrough/qwen3-coder-next-64k:latest`. Different
     clients differ in whether their "current model" UI maps to
     the literal `model` field or applies a prefix — see per-client
     notes below.

### OpenClaw

OpenClaw's config file lives at the path printed by
`openclaw config path`. On Linux that's typically
`~/.config/openclaw/config.yaml`. The `providers:` and `models:`
blocks are top-level siblings:

```yaml
providers:
  audrey:
    api: openai-completions
    base_url: https://audrey.your-domain/v1
    api_key: sk-<from shared prereqs step 2>
models:
  qwen3.6:35b-64k:
    provider: audrey
    model: audrey_passthrough/qwen3.6:35b-64k
  qwen3-coder-next-64k:latest:
    provider: audrey
    model: audrey_passthrough/qwen3-coder-next-64k:latest
```

The `model:` field under each entry is the literal string Audrey
receives — so the `audrey_passthrough/` prefix goes here, not in
the key. Restart OpenClaw (or `systemctl restart openclaw` if
running as a daemon) and confirm a chat completes. Grep Audrey
logs for `passthrough.chat` or `passthrough.stream` lines.

### Hermes

Hermes prompts interactively for its OpenAI-compatible endpoint
config. When you see:

```
Custom OpenAI-compatible endpoint configuration:

API base URL [e.g. https://api.example.com/v1]:
```

…enter Audrey's base URL (see shared prereqs step 3). At the
API-key prompt that follows, paste the `sk-…` value from step 2.

For the **model name**, Hermes shows `Current model: qwen3.6:35b-64k`
in your current state — but that's likely the *display* form. The
string sent in the API request needs the prefix, so when Hermes
prompts for the model name (or when you change it via `hermes
model`), enter the full form:

```
audrey_passthrough/qwen3.6:35b-64k
```

If Hermes auto-discovers models via `GET /v1/models`, the listing
already returns the prefixed IDs (verified by
`tests/test_passthrough_route.py::test_list_models_includes_passthrough_when_enabled`),
so the dropdown should show them correctly without manual entry.

For the **API compatibility mode**, pick **Chat Completions**.
Audrey only serves `POST /v1/chat/completions` — there's no
`/v1/responses` or `/v1/messages` endpoint. Auto-detect probably
lands there too based on the URL, but pinning explicitly avoids
heuristic surprises.

For the **context length**, both shipped passthrough models
(`qwen3.6:35b-64k`, `qwen3-coder-next-64k:latest`) are Modelfile
variants explicitly built with a 65536-token window. Hermes can
set anywhere from the model's default up to that ceiling. Guidance:

  - **Default 32768.** Plenty for chat-style prompting; uses about
    half the KV-cache VRAM of 65536. Leaves headroom for the model
    swap between the two passthrough models on the 24 GB 3090 Ti
    without thrashing.
  - **Bump to 65536** only if you hit context-length errors in
    real prompts — long-document RAG, long-conversation continuity,
    multi-file code review. Most interactive use doesn't approach
    32k tokens.
  - Asking for more than 65536 on these tags either gets clamped
    silently by Ollama or produces garbled output. Don't.
  - The KV cache for 65536 tokens on a 35B Q4 model is roughly
    4-8 GB on top of model weights. With `qwen3.6:35b-64k` (23 GB
    on disk) at full context the box is already tight; the coder
    variant (51 GB) won't fit on one GPU at any meaningful context
    and will offload layers to CPU, slowing generation
    significantly. First request after a model swap eats the load
    penalty (~30s for the coder); subsequent calls hit the warm
    cache.

After save, send a test prompt. Look for the same `passthrough.chat`
or `passthrough.stream` log lines on the Audrey side.

### Verifying any client end-to-end

Two checks confirm the wiring without trusting the client's UI:

```
# 1. Audrey logs the dispatch with the concrete model.
docker logs audrey-ai 2>&1 | grep -E "passthrough\.(chat|stream)"

# 2. Metrics show the passthrough mode + the right concrete.
curl -s http://192.168.1.11:9100/metrics | grep -E "audrey_dispatch_total.*passthrough"
```

If you see `path="passthrough"` (non-streaming) or
`path="passthrough_stream"` (streaming) with the concrete model
label, the round-trip succeeded and both fair-scheduling layers
fired on the way through.

## What this unblocks

The checkpoint-testing phase queued in
`docs/campaign-2/checkpoint-testing-plan.md` was always written
assuming Audrey-mediated traffic; with passthrough in place, the
fairness story actually covers everyone on the LAN. Streaming
cancel cleanup and the parallel context-injection drift items
both touch shared infrastructure (gate, inflight) — easier to
chase real production behavior with one source of truth for
GPU contention.
