# Campaign 2 Phase 2a - Prompt centralization and composer

Pure refactor + one new system-message constant.

Every pipeline prompt now lives in one place
(`src/audrey/pipeline/prompts.py`). The constants moved byte-for-byte
from their old homes (classify, planner, synthesize, react, memory),
so model behavior on existing requests does not change. The new bits:

- A single `compose_system_messages` helper that pins the canonical
  order of system messages (incoming → task-role → memory → chat-history
  guidance). Both memory-recall sites (the non-streaming graph node and
  the streaming deep-path thinking phase) now go through it.
- An override loader, `prompt_from_config`, for optional
  `agentic.prompts.*` config tuning. Defaults still live in code so
  Audrey boots from a normal checkout.
- A `CHAT_HISTORY_SEARCH_SYSTEM` system message that nudges tool-capable
  models toward deliberate archive lookup — the system-side complement
  to Phase 1's tool description. The composer injects it **only when
  `chat_history_search` is in the registry**; ordinary fast prompts that
  don't run memory recall never see it.

The 2b "fast-answerer / deep-worker role prompts" piece is **not**
shipped here. It's deferred until evidence shows a real failure mode
that a role prompt would fix.

What stays the same:

- Five virtual models, pipeline shape, fast/deep routing, durable
  memory behavior, chat archive behavior, tool registry, KB stack.
- Existing prompt text — every byte preserved. A regression test pins
  each constant against the historical text so any future reflow
  shows up as a focused diff.
- Token cost on requests that don't run memory recall (unauthenticated
  calls, in-flight slot empty, memory disabled in config): unchanged.

What changed:

- **`src/audrey/pipeline/prompts.py`** (new) -
  - Default constants: `CLASSIFIER_SYSTEM`, `PLANNER_SYSTEM`,
    `SYNTH_SYSTEM`, `REACT_FINAL_ANSWER_USER`, `MEMORY_STORE_HINT`,
    `CHAT_HISTORY_SEARCH_SYSTEM`.
  - `prompt_from_config(cfg, key, default)` — single resolver for
    optional overrides under `agentic.prompts.*`. Null / missing /
    empty / whitespace → default. Non-string → default with warning.
    Override > 4000 chars → still applied, with a one-time per-key
    warning so a runaway persona is visible without being silently
    truncated.
  - `compose_system_messages(...)` — single helper for the canonical
    order. Empty slots are skipped; blank chat-history text is
    treated as "disabled" so a wiped override never emits a hollow
    system message.
- **`src/audrey/pipeline/classify.py`** -
  - Local `_ROUTER_SYSTEM` is an alias for
    `prompts.CLASSIFIER_SYSTEM`. `router_classify` and `classify`
    take an optional `cfg` kwarg; when supplied,
    `agentic.prompts.classifier` overrides the system prompt.
- **`src/audrey/pipeline/planner.py`** -
  - Local `_PLANNER_SYSTEM` is an alias for `prompts.PLANNER_SYSTEM`.
    `plan` takes an optional `cfg` kwarg; `agentic.prompts.planner`
    overrides the system prompt.
- **`src/audrey/pipeline/synthesize.py`** -
  - Local `_SYNTH_SYSTEM` is an alias for `prompts.SYNTH_SYSTEM`.
    `_build_synth_messages` accepts `cfg=`; both call sites
    (`_try_synth` and `synthesize_stream`) pass it through.
    `agentic.prompts.synthesizer` overrides the synth system prompt.
- **`src/audrey/pipeline/react.py`** -
  - The "tool-budget reached, write the final answer now" user turn
    now reads from `prompts.REACT_FINAL_ANSWER_USER`. `run_react`
    takes an optional `cfg` kwarg; `agentic.prompts.react_final_answer`
    overrides the final-answer text.
- **`src/audrey/pipeline/memory.py`** -
  - Local `_MEMORY_STORE_HINT` is an alias for
    `prompts.MEMORY_STORE_HINT`. `memory_system_message` takes an
    optional `cfg` kwarg; `agentic.prompts.memory_store_hint`
    overrides the hint template (the `{user_id}` placeholder is still
    substituted at call time).
- **`src/audrey/pipeline/fast_path.py`** / **`deep_panel.py`** -
  - Both forward `cfg=` into `run_react` so the final-answer override
    works on the fast and deep paths uniformly.
- **`src/audrey/pipeline/graph.py`** -
  - `node_memory_recall` now goes through `compose_system_messages`,
    which merges the memory hint with `CHAT_HISTORY_SEARCH_SYSTEM`
    when `chat_history_search` is registered. A new log field
    (`chat_history_hint=on|off`) shows whether the guidance fired
    for this request.
- **`src/audrey/routes/openai.py`** -
  - `_phase_thinking` (streaming deep path) goes through the same
    composer. Same gating rule.
- **`tests/test_prompts.py`** (new) -
  - Byte-for-byte regression for every moved constant.
  - Call-site aliases match the central constants (one stray edit
    will break two tests, not one).
  - `prompt_from_config` resolution: default, null, empty, whitespace,
    non-string, valid override, oversize warning (once per key),
    unknown-key warning.
  - Composer ordering: each combination of incoming / task-role /
    memory / chat-history guidance, plus the "blank chat-history
    text = no message" rule and the "fresh list each call" rule.

Out of scope (still deferred):

- Per-task-type or per-virtual-model role prompts.
- New `FAST_ANSWER_SYSTEM` / `DEEP_WORKER_SYSTEM` (Phase 2b — gated
  on evidence).
- Hot-reloading overrides without a restart.
- Importing prompt overrides from anywhere except `agentic.prompts.*`
  in `config.yaml`.

## 1. Deploy

No data migration. Only `audrey-ai` changes; `custom-tools` is
untouched.

```bash
# Laptop:
git pull   # after the Phase 2a commit lands

# Unraid (from /mnt/user/appdata/audrey_ai_2.0):
git pull
docker compose up -d --build audrey-ai
docker compose logs --since 2m audrey-ai | grep -E "ready:|memory:"
```

Expected:

- A normal readiness line ending with `pipeline=compiled` and the
  same tool count as before Phase 2a (7, including
  `chat_history_search` from Phase 1).
- The first authenticated chat request emits a `memory:` log line
  with a new `chat_history_hint=on` field — confirmation that the
  composer is wired and gated correctly. If `chat_history_search`
  isn't registered for some reason, the field will be `chat_history_hint=off`.

## 2. Smoke tests

Set these once per shell:

```bash
AUDREY_URL="http://localhost:8000"
USER_TOKEN="<valid OWUI bearer token for a non-admin user>"
```

### 2.1 Verify the chat-history hint fires when the tool is registered

```bash
curl -sS -o /tmp/audrey-fast.json -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_deep","stream":false,
       "messages":[{"role":"user","content":"What is ZFS?"}]}'

docker compose logs --since 1m audrey-ai | grep "memory:" | tail -3
```

Expected: at least one `memory: user=… chat_history_hint=on` line. If
you see `chat_history_hint=off` on every line, the registry is missing
`chat_history_search` — fix via `/v1/tools/rediscover` (admin bearer
required) or restart audrey-ai.

### 2.2 Verify prompts are still byte-identical

Local check on the laptop, no Unraid call needed:

```bash
.venv/bin/python -m pytest tests/test_prompts.py -q
```

Expected: all 29 cases pass. If the byte-for-byte regression cases
fail, the move was not byte-faithful — back it out before deploying.

### 2.3 Verify existing model paths still behave

The classifier, planner, synth, and react prompts are unchanged
strings; pick any short conversation and confirm output quality is
indistinguishable from pre-2a. There's no Prometheus signal that
catches a "subtly different prompt" regression — only your eyes.
Recommended quick checks:

```bash
# Fast classification (no tools, no memory): should look identical.
curl -sS -o /dev/null -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,
       "messages":[{"role":"user","content":"What is the capital of France?"}]}'

# Deep panel: a multi-part question that exercises planner + synth.
curl -sS -o /tmp/deep.json -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_deep","stream":false,
       "messages":[{"role":"user","content":"Compare BTRFS and ZFS snapshots in a short table."}]}'
jq '.choices[0].message.content' /tmp/deep.json | head -c 300
```

Expected: `HTTP 200` on both; the table layout and tone match what
you saw pre-2a. Pre-2a answers are not preserved, so this is
eyeball-only.

### 2.4 Try an override

The override loader is wired into every call site that uses a
centralized prompt. Supported keys today:

- `classifier`
- `planner`
- `synthesizer`
- `react_final_answer`
- `memory_store_hint`
- `chat_history_search`

Pick one and add it under `agentic.prompts.*` in `config.yaml`. The
easiest visible test is `planner` because the override lands in every
deep-panel request that exceeds the planning token threshold:

Add the `prompts:` block as a sibling of the existing `react:` /
`planning:` / `memory:` entries under `agentic:`:

```yaml
agentic:
  # ... existing react/planning/reflection/memory entries stay as-is ...
  prompts:
    planner: |
      Decompose the user request into exactly two independent sub-questions.
      Return ONLY a JSON object of shape {"subtasks": ["<question 1>", "<question 2>"]}.
      Each sub-question must be a complete question (≤ 200 chars).
      No prose, no markdown.
```

The `|` is YAML's literal block scalar — the multi-line string is
passed to the model verbatim (one trailing newline, model ignores it).
The `<question 1>` / `<question 2>` markers read as placeholders, not
literal subtask names, which avoids the model producing
`{"subtasks": ["a", "b"]}` literally.

Restart audrey-ai:

```bash
docker compose up -d --build audrey-ai
```

Then issue a deep request whose prompt is long enough to trigger
planning (the default token threshold is 40):

```bash
curl -sS -o /tmp/override.json -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_deep","stream":false,
       "messages":[{"role":"user","content":"Compare BTRFS and ZFS in terms of snapshots, send/receive, and compression — be specific about modern Linux distros."}]}'

docker compose logs --since 1m audrey-ai | grep "planner:"
```

Expected: the planner log line shows exactly two subtasks (the
override forced 2). Removing the override (or setting it to `null` /
`""`) reverts to the default 2-or-3-subtasks behavior without a code
change.

Override rules:

- Empty string or `null` → default applies.
- Non-string value → default applies, with a startup warning.
- Override longer than 4000 chars → still applies, with a one-time
  per-key warning so a runaway persona is visible.
- An unknown override key (typo) → ignored, with a one-line warning.

### 2.5 Local verification

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src/audrey tests/
```

Expected:

- All tests pass.
- Ruff has no Phase-2a issues. The pre-existing ASYNC240 warnings
  in `src/audrey/kb/ingest.py` and `src/audrey/kb/watcher.py` are
  unchanged and accepted.

## 3. Rollback

Plain git rollback. No data, no config required.

```bash
git revert <phase-2a-commit>
docker compose up -d --build audrey-ai
```

The chat archive Qdrant collection and SQLite file stay; rollback
only undoes the prompt centralization and composer wiring.

## 4. Operational notes

- `chat_history_hint=on` in `memory:` logs proves the composer wired
  the chat-history guidance for that request. Off means the registry
  doesn't have `chat_history_search` — Phase 1's deploy doc covers
  the rediscover / restart procedure.
- An override longer than 4000 chars does not block startup or
  truncate the prompt — it just emits one warning per key per
  process. Use the warning as a smell test, not a rule.
- If a prompt key is misspelled in `agentic.prompts.*`, the loader
  emits a one-line warning and falls back to the default. Watch
  startup logs for `prompts: unknown override key`.
- The composer's canonical order is intentionally fixed in code.
  Anyone who wants to reorder system messages (e.g. move chat-history
  guidance above memory) should change `compose_system_messages` in
  one place rather than reordering at call sites.
- The 2a override surface only covers prompts that already existed
  in the codebase plus the new chat-history-search guidance. Fast-
  answerer and deep-worker prompts are 2b territory; their keys are
  not yet in `_PROMPT_KEYS` and adding them via config today will be
  rejected with a warning.

## 5. Followups

- Phase 2b (new role prompts) — gated on evidence. Open
  `docs/campaign-2/phase-02-prompt-centralization-plan.md` for the trigger
  conditions.
- If we ever add a per-turn timing signal for prompt-token totals,
  pin a regression test against `audrey_fast` token counts so a
  future composer change can't silently grow them.
