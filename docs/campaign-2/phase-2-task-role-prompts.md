# Campaign 2 Phase 2 - Task role prompts

## Goal

After Campaign 2 Phase 1 lands, give Audrey's model calls short,
task-specific role prompts so each model knows the job it is performing:
fast answerer, deep worker, planner, synthesizer, tool-using worker, and
chat-history search user.

This phase should improve consistency without turning Audrey into a pile of
long personas. Prompts should be compact job contracts.

## Why after Phase 1

Phase 1 adds the searchable chat archive and a new memory-adjacent tool path.
Once that tool exists, Audrey will have one more kind of model behavior to
shape: "search prior chats when useful, but do not dump archive history into
every prompt."

Phase 2 should then cleanly centralize prompt ownership before more prompts
spread across the codebase.

## Current state

Prompt text is currently embedded near each caller:

- `src/audrey/pipeline/classify.py` has the router/classifier system prompt.
- `src/audrey/pipeline/planner.py` has `_PLANNER_SYSTEM`.
- `src/audrey/pipeline/synthesize.py` has `_SYNTH_SYSTEM`.
- `src/audrey/pipeline/react.py` has the forced final-answer instruction.
- `src/audrey/pipeline/memory.py` has the durable `memory_store` usage hint.
- Fast path and deep workers mostly inherit the user's/system messages without
  a clear "your job in this pipeline" role prompt.

That is workable, but it makes prompt tuning harder and hides Audrey's
division of labor.

## Product rules

- Prompts must be short. Prefer 5-10 precise bullets over large personas.
- Prompts should describe the model's job, not a character or identity.
- Prompt text should not duplicate large system messages already supplied by
  Open WebUI or by the user.
- Config overrides should be optional. Audrey needs safe defaults in code.
- Keep token cost visible and small.
- Do not let prompt overrides bypass user scoping or tool safety rules.

## Proposed shape

Add one prompt home:

```text
src/audrey/pipeline/prompts.py
```

It should contain default prompt constants and helpers, for example:

```python
CLASSIFIER_SYSTEM = "..."
PLANNER_SYSTEM = "..."
FAST_ANSWER_SYSTEM = "..."
DEEP_WORKER_SYSTEM = "..."
SYNTH_SYSTEM = "..."
REACT_FINAL_ANSWER_USER = "..."
CHAT_HISTORY_SEARCH_SYSTEM = "..."

def prompt_from_config(cfg, key: str, default: str) -> str:
    ...
```

Then each pipeline module imports from `prompts.py` instead of owning its own
large constant.

## Config shape

Add optional overrides under `agentic.prompts`:

```yaml
agentic:
  prompts:
    classifier: null
    planner: null
    fast_answerer: null
    deep_worker: null
    synthesizer: null
    react_final_answer: null
    chat_history_search: null
```

Rules:

- `null` or missing means use the code default.
- Empty strings should be treated as missing.
- Keep all defaults in code so Audrey boots from a normal checkout.
- Config overrides are for small local tuning, not secrets.

## Prompt call sites

### Classifier

Move the current classifier prompt into `prompts.py`.

The classifier should keep its current contract:

```text
return code | reasoning | general | vl with a confidence/reason shape
```

No behavior change intended.

### Planner

Move `_PLANNER_SYSTEM` into `prompts.py`.

The planner should keep its current rules:

- return JSON only
- split into 2-3 independent subtasks
- return `[]` for atomic prompts

No behavior change intended beyond centralization.

### Fast answerer

Add a compact system prompt before the fast one-shot chat path and before
fast ReAct starts.

Intent:

```text
You are the fast answerer. Answer directly and efficiently. Use tools when
they are available and useful. Do not invent missing facts.
```

Keep this short because fast path is the most token-sensitive path.

### Deep worker

Add a compact worker prompt before each deep-panel worker call.

Intent:

```text
You are one independent deep-panel worker. Produce your strongest answer or
analysis for the assigned prompt/subtask. Do not mention other workers or
the panel.
```

The synthesizer already handles reconciliation, so workers should not write
"final synthesis" prose.

### Synthesizer

Move `_SYNTH_SYSTEM` into `prompts.py` first.

Then tune only if needed. The current synthesizer prompt is already a clear
role prompt, so Phase 2 should mostly centralize it and preserve behavior.

### ReAct final answer

Move the "tool-call budget reached" final-answer instruction into
`prompts.py`.

Keep it as a user message or make it a helper that returns the message object;
do not bury that final-mode switch inside a long loop body.

### Chat history search

After Phase 1 adds `chat_history_search`, give tool-capable models a short
instruction about when to use it.

Intent:

```text
Use chat_history_search only when the user asks about prior conversations or
when answering requires a specific prior decision. Do not call it for ordinary
personalization.
```

This should reinforce the Phase 1 token rule: archive lookup is deliberate,
not automatic prompt stuffing.

## Implementation steps

1. Add `src/audrey/pipeline/prompts.py` with current prompt constants moved
   over unchanged where possible.
2. Add a small helper for optional `agentic.prompts` overrides.
3. Update `classify.py`, `planner.py`, `synthesize.py`, `react.py`,
   `fast_path.py`, and `deep_panel.py` to consume prompt helpers.
4. Add fast-answerer and deep-worker system messages at the call boundaries.
5. After Phase 1 exists, add the chat-history-search guidance to the ReAct
   prompt path or tool-use context.
6. Keep prompt insertion order predictable:
   existing system messages first, Audrey task-role prompt next, then user
   content/tool messages.
7. Add tests that verify prompt messages are present without making real model
   calls.
8. Run the existing test suite.
9. Smoke test `audrey_fast`, `audrey_deep`, and a tool-using prompt.

## Tests

Minimum tests:

- `prompts.py` returns defaults when config has no override.
- Empty override strings fall back to defaults.
- Non-empty override strings are used.
- Fast path includes the fast-answerer prompt in one-shot and ReAct paths.
- Deep panel includes the deep-worker prompt for worker calls.
- Synthesizer still forwards prior system messages and includes the synth
  prompt.
- Planner still degrades to `[]` on bad model output.
- ReAct final-answer prompt is still appended when the tool budget is reached.

The tests should use fake `OllamaClient` calls or mock transports. No external
Ollama, Qdrant, or custom-tools dependency.

## Deployment notes

Expected deployment impact:

- Rebuild `audrey-ai`.
- No data migration.
- No custom-tools rebuild unless Phase 2 also adds prompt text to the
  `chat_history_search` tool description after Phase 1.
- Config change optional.

Suggested smoke tests:

1. `audrey_fast` short general prompt still answers directly.
2. `audrey_deep` multi-part prompt still runs workers and synthesizes.
3. Tool-using prompt still calls tools and produces a final answer.
4. After Phase 1, ask about a prior conversation and verify the model can use
   `chat_history_search`.
5. Confirm prompt token counts do not jump sharply for simple fast prompts.

## Out of scope

- Long persona prompts.
- Per-user custom prompt editing UI.
- Dynamic prompt generation by another model.
- Rewriting model selection or health behavior.
- Changing the durable memory database.
- Turning chat archive auto-recall on by default.

## Open questions

- Should prompt overrides be hot-reloaded someday, or remain restart-only like
  the rest of `config.yaml`?
- Should task prompts be globally configured, per virtual model, or per task
  type? Phase 2 should start global and only add dimensions if real examples
  require them.
- Should code tasks get a separate deep-worker prompt from general/reasoning
  tasks?
- How much should prompt text mention Audrey's internals, such as "deep panel"
  or "synthesizer," versus using user-facing language only?
