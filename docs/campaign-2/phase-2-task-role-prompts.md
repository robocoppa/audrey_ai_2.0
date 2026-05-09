# Campaign 2 Phase 2 - Prompt centralization and composition

## Goal

Centralize Audrey's existing pipeline prompts in one place, define the order
in which system messages are composed, and give Phase 1's chat-history tool
the small system-side guidance it needs to keep the model from over-calling
it.

This phase is deliberately split:

- **Phase 2a (do now)**: pure refactor + system-message composer +
  `CHAT_HISTORY_SEARCH_SYSTEM` guidance. No new role prompts on
  fast/deep paths. No behavior change for existing prompts.
- **Phase 2b (deferred until evidence)**: add new fast-answerer and
  deep-worker role prompts only after observing real outputs that a role
  prompt would actually fix.

The original Phase 2 plan added new role prompts on every fast request and
every deep worker call. Fast path is the most token-sensitive path in the
pipeline, and there is currently no documented failure mode that says fast
or deep outputs are inconsistent in a way a role prompt would fix. Adding
text on speculation is the prompt-engineering trap. 2b is gated on real
evidence.

## Why after Phase 1

Phase 1 adds `chat_history_search`. The model-facing tool description is
the only steering it ships with. Phase 2a adds the matching system-side
guidance and bakes it into the composer so it is delivered consistently
across fast/deep/ReAct paths.

## Current state

Prompt text is currently embedded near each caller:

- `src/audrey/pipeline/classify.py` has the router/classifier system prompt.
- `src/audrey/pipeline/planner.py` has `_PLANNER_SYSTEM`.
- `src/audrey/pipeline/synthesize.py` has `_SYNTH_SYSTEM`.
- `src/audrey/pipeline/react.py` has the forced final-answer instruction.
- `src/audrey/pipeline/memory.py` has the durable `memory_store` usage hint.
- Fast path and deep workers inherit the user's/system messages without a
  dedicated role prompt — and 2a leaves that alone.

That is workable but makes prompt tuning harder and hides Audrey's division
of labor. Centralizing also surfaces the message-ordering problem that gets
worse as soon as Phase 1's chat-history guidance lands.

---

# Phase 2a — Do now

## Product rules

- No behavior change for existing prompts. Move text byte-for-byte.
- One canonical system-message ordering, defined once and called from
  every node that adds a system message.
- Add `CHAT_HISTORY_SEARCH_SYSTEM` for tool-capable models because Phase 1
  needs it.
- Config overrides are optional; defaults live in code.
- A single override loader. No module reads `cfg.agentic.prompts.foo`
  directly.

## Proposed shape

Add one prompt home:

```text
src/audrey/pipeline/prompts.py
```

Contents:

```python
# Existing prompts moved unchanged from their current homes.
CLASSIFIER_SYSTEM = "..."         # from classify.py
PLANNER_SYSTEM = "..."            # from planner.py (_PLANNER_SYSTEM)
SYNTH_SYSTEM = "..."              # from synthesize.py (_SYNTH_SYSTEM)
REACT_FINAL_ANSWER_USER = "..."   # from react.py
MEMORY_STORE_HINT = "..."         # from memory.py (_MEMORY_STORE_HINT)

# New in 2a — needed because Phase 1 shipped chat_history_search.
CHAT_HISTORY_SEARCH_SYSTEM = "..."

def prompt_from_config(cfg, key: str, default: str) -> str: ...

def compose_system_messages(*, ...) -> list[dict]: ...
```

Each pipeline module imports from `prompts.py` instead of owning its own
constant.

## System-message composer

This is the load-bearing piece of 2a.

A fast request today can already accumulate up to three system messages:
OWUI's incoming system prompt, memory recall hits, and the
`memory_store` hint. Phase 1 adds chat-history-search guidance for
tool-capable models. Without a single composer the order is implicit and
will drift.

```python
def compose_system_messages(
    *,
    incoming: list[dict],          # OWUI / user-supplied system messages
    memory_hint: dict | None,      # from pipeline/memory.py
    task_role: str | None = None,  # 2b plug-in point, unused in 2a
    chat_history_guidance: bool = False,
) -> list[dict]:
    """Compose system messages in canonical order.

    Order:
      1. Incoming system messages (preserved as given).
      2. Task-role prompt (2b plug-in; 2a always passes None).
      3. Memory recall + memory_store hint.
      4. Chat-history-search guidance, when tools are available.

    Rationale: incoming first so user/OWUI persona wins on tone; task-role
    next because it shapes how the assistant performs that turn; memory
    third because it is content, not instruction; chat-history guidance
    last because it gates a single tool and should be the freshest
    instruction the model sees.
    """
```

Every node that wants to add a system message goes through this helper.
Direct list-mutation of `messages` to insert a system entry is removed.

## Config shape

Add optional overrides under `agentic.prompts`:

```yaml
agentic:
  prompts:
    classifier: null
    planner: null
    synthesizer: null
    react_final_answer: null
    chat_history_search: null
```

Rules:

- `null` or missing means use the code default.
- Empty strings should be treated as missing.
- Keep all defaults in code so Audrey boots from a normal checkout.
- Config overrides are for small local tuning, not secrets.
- Overrides that exceed a soft cap (default 4000 chars) emit a startup
  warning with the override length. The override still applies — this is
  a guardrail, not enforcement — but the user sees it.
- The override loader is the only path. No module reads
  `cfg.agentic.prompts.foo` directly.

Note: 2a does not add `fast_answerer` or `deep_worker` keys yet. 2b adds
them if and when their prompts get added.

## Prompt call sites in 2a

### Classifier

Move the current classifier prompt into `prompts.py`. Contract unchanged:

```text
return code | reasoning | general | vl with a confidence/reason shape
```

### Planner

Move `_PLANNER_SYSTEM` into `prompts.py`. Rules unchanged: return JSON
only, split into 2-3 independent subtasks, return `[]` for atomic prompts.

### Synthesizer

Move `_SYNTH_SYSTEM` into `prompts.py`. The current text is already a
clear role prompt; preserve it byte-for-byte.

### ReAct final answer

Move the "tool-call budget reached" final-answer instruction into
`prompts.py`. Keep it as a helper that returns the message object so the
final-mode switch is not buried inside a long loop body.

### memory_store hint

Move `_MEMORY_STORE_HINT` from `pipeline/memory.py` into `prompts.py`.
The `{user_id}` substitution stays in the call site.

### chat_history_search guidance (new)

Add `CHAT_HISTORY_SEARCH_SYSTEM`:

```text
Use chat_history_search only when the user asks about prior conversations
or when answering requires a specific prior decision. Do not call it for
ordinary personalization or to repeat back recent context.
```

The composer adds this system message only when the registry has
`chat_history_search` available — no point telling the model how to use a
tool it does not have.

## Implementation steps (2a)

1. Add `src/audrey/pipeline/prompts.py` with all existing prompt
   constants moved over byte-for-byte.
2. Add `prompt_from_config(cfg, key, default)` with the
   empty-string-as-missing rule and the length warning.
3. Add `compose_system_messages(...)` per the order above.
4. Update `classify.py`, `planner.py`, `synthesize.py`, `react.py`, and
   `memory.py` to import from `prompts.py`.
5. Replace direct system-message list mutation in fast/deep request
   prep with `compose_system_messages(...)` calls.
6. Add the chat-history-search guidance gated on
   `"chat_history_search" in registry.by_name`.
7. Add tests (see "Tests for 2a").
8. Smoke test fast/deep/tool-using requests.

## Tests for 2a

- Each existing prompt constant's text in `prompts.py` matches the
  pre-move source byte-for-byte. (Cheap regression insurance: any
  intentional change shows up in a separate diff.)
- `prompt_from_config` returns the default when the key is missing,
  null, or empty string.
- Override longer than the soft cap logs a warning at load time.
- `compose_system_messages` produces the documented order under every
  combination of `incoming/memory_hint/task_role/chat_history_guidance`.
- Composer is the only writer of system messages: grep test or unit
  test asserting no node directly inserts `{"role": "system", ...}`
  outside `prompts.py`.
- Chat-history-search guidance is included only when
  `chat_history_search` is in the registry.
- Planner still degrades to `[]` on bad model output.
- ReAct final-answer prompt is still appended when the tool budget is
  reached.

Tests should use fake `OllamaClient` calls or mock transports. No
external Ollama, Qdrant, or custom-tools dependency.

## Deployment notes (2a)

- Rebuild `audrey-ai`.
- No data migration.
- No custom-tools rebuild.
- Config change optional.
- Prompt token counts for fast/deep paths must not change. The smoke test
  is "fast prompt token count is identical to pre-2a for the same input
  and the same OWUI system prompt."

---

# Phase 2b — Deferred

Phase 2b adds the two new role prompts that the original plan included on
the fast and deep paths. It is gated on evidence, not scheduled.

## Trigger conditions

Open Phase 2b only when at least one of the following is true:

- A documented failure mode where fast-path output drifts from the
  expected behavior in a way a 5–10 line role prompt would plausibly fix
  (e.g., the fast model invents tool capabilities it does not have, or
  refuses to answer when it should).
- A documented failure mode where deep-panel workers write "final
  synthesis" prose that the synthesizer then has to fight, or workers
  reference each other.
- Bart asks for them after observing real chats for at least a couple of
  weeks post-Phase-1.

If none of these occur, do not add 2b. The default win is keeping the
fast path as cheap as it is today.

## Token budget for any 2b prompt

Pin a budget *before* writing text:

- Fast-answerer prompt: ≤ 60 tokens.
- Deep-worker prompt: ≤ 80 tokens.

Measured by the actual tokenizer of the model that will see it
(approximate is fine — `len(text) // 4` is good enough for a sanity
check). If a draft prompt exceeds the budget, cut it before merging, do
not raise the budget.

## Per-task-type stance

If 2b ships, start with a single global `DEEP_WORKER_SYSTEM`. Do not
preemptively split per task type (`code | reasoning | general | vl`) or
per virtual model. Add dimensions only after a real example shows the
single prompt is wrong for one task type.

This keeps the override surface small and the failure modes observable.

## What 2b would add

If trigger conditions are met:

1. Add `FAST_ANSWER_SYSTEM` and `DEEP_WORKER_SYSTEM` to `prompts.py`.
2. Add `fast_answerer` and `deep_worker` keys to the `agentic.prompts`
   config block.
3. Wire them in via `compose_system_messages(task_role=...)` — the
   plug-in point is already in 2a.
4. Add tests that the prompt is present in fast one-shot, fast ReAct,
   and deep-worker call paths.
5. Add a smoke test pinning token-count delta for typical fast prompts.

## What 2b will not add

- Long persona prompts.
- Per-user custom prompt editing UI.
- Dynamic prompt generation by another model.
- Per-virtual-model prompts.
- Per-task-type prompts (until evidence demands it).

---

## Open questions

- Should prompt overrides be hot-reloaded someday, or remain
  restart-only like the rest of `config.yaml`? Phase 2 keeps them
  restart-only.
- Should code tasks ever get a separate deep-worker prompt? Decided:
  not until 2b ships and only on evidence.
- How much should prompt text mention Audrey's internals (e.g., "deep
  panel," "synthesizer") versus user-facing language only? Default for
  any 2b prompt: internal-vocabulary fine in worker/synth/classifier
  prompts (the model never user-faces those directly); user-facing
  language only in fast-answerer (its output is what the user sees).
