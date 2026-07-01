# Specialist virtual model — prototype plan

Not a deploy doc yet. This is the design plan for a one-off
hand-coded specialist virtual model (the Option 2 path from the
2026-05-15 design discussion). The goal is to validate the pattern
on a single domain before generalizing to a config-driven
`specialists:` block (Option 4).

## What the user gets

A new entry in OWUI's model dropdown — e.g. `audrey_<topic>` —
that, when selected, behaves like `audrey_auto` but with:

- A domain-specific system prompt prepended to every turn.
- `kb_search` and `kb_image_search` calls scoped to a curated
  topic *plus* the calling user's own uploads collection.
- The full Audrey tool surface still available (web_search,
  memory_*, chat_history_search) — the specialist isn't
  KB-locked, just KB-biased.
- The same auto-routing complexity gate as `audrey_auto`
  (subject to Phase 6a's outcome).

OWUI sees this as one more virtual model. No OWUI-side
configuration needed beyond making sure the model is enabled
for the workspace.

## Client target: a Hermes agent (added 2026-06-03)

The first real consumer of this specialist is not OWUI — it's a
**Hermes agent** that needs to consult Audrey's RAG. OWUI still
works (it's just another virtual model), but the design below is
driven by the Hermes use case. Full Hermes wiring facts live in
`hermes-reference.md`; this section records the decisions specific
to the specialist.

### The constraint that shaped the design

Hermes's tool calling only works reliably when pointed **direct at
Ollama** (`:11434`). Routed **through Audrey's passthrough**, Hermes
struggles with tool calls (the whole saga in `hermes-reference.md` —
streaming tool_calls accumulation, `tool_choice`, message-shape
edge cases). So the naive "point Hermes's main model at the
specialist" approach would reintroduce that pain, because Hermes
sends its 47-tool `hermes-cli` bundle on every main-model call
(`agent.tool_use_enforcement: auto`) and would try to drive its own
agent loop *through* the specialist path.

### Why the specialist path sidesteps it entirely

Two different code paths, and they must not be confused:

- **Passthrough** (`audrey_passthrough/<model>`) — Audrey is a bare
  proxy; Hermes drives its own tool loop and Audrey relays the
  `tools` array and reshapes `tool_calls`. This is the path that
  struggles. **Not used for the specialist.**
- **Pipeline** (`audrey_<topic>`) — Audrey runs its *own* ReAct loop
  (kb_search etc.) and returns finished prose. On this path **Hermes
  sends no tools array** — it just asks a question and gets an
  answer. There is no tool-call relay to get wrong, because Hermes
  isn't calling tools through Audrey; **Audrey** is.

So the specialist must be reached as a **first-class virtual model**
(`audrey_<topic>`), never via the `audrey_passthrough/` prefix —
passthrough deliberately skips tool discovery and would leave the
specialist with no `kb_search` at all.

### Role split (locked in)

- **Audrey drives the tool loop.** The specialist is a Q&A endpoint:
  Hermes asks `audrey_<topic>` a question, Audrey runs classify →
  ReAct → kb_search → synthesize, returns prose. Hermes does **no**
  tool calling on this path.
- **Hermes keeps its own agent loop on direct Ollama** for its
  machine-local tools (filesystem, terminal, browser) — unchanged,
  where tool calls already work.

Clean separation, same one the Hermes reference doc landed on
independently: Hermes-side tools for client-machine-local stuff;
Audrey-side tools (KB, memory) behind the specialist.

### Model (locked in): `qwen3.6:35b`

Both paths use `qwen3.6:35b`:

- It is already the local pool's primary — fast-path local model
  (`config.yaml`) and the worker + synthesizer for the local
  deep_panel tasks. The specialist is "`audrey_auto` + a topic
  prompt," so it inherits this model with no new pin.
- It's the same tag Hermes runs direct-to-Ollama, so Hermes's main
  loop is unchanged.
- One model resident fits `GPU_CONCURRENCY=1`: the specialist call
  and Hermes's direct calls hit the *same* loaded weights — no
  eviction thrash. They contend only for the GPU **gate**, which the
  specialist path arbitrates via `FairLocalGate` + the inflight
  registry. Hermes's *direct* path bypasses those gates; if
  contention ever bites, route Hermes's direct calls through Audrey
  passthrough too. **Later-if-it-hurts, not a blocker.**

### How Hermes reaches the specialist (decision: auxiliary provider)

Hermes config uses named `custom_providers:` with model syntax
`<provider-name>:<model>`. The specialist is a named provider:

```yaml
custom_providers:
- name: audrey-kb
  base_url: http://192.168.1.11:8000/v1
  api_key: sk-<owui key for hermes's user>
  api_mode: chat_completions
  model: audrey_<topic>          # specialist — NOT audrey_passthrough/
```

The decision (2026-06-03) is to route **one of Hermes's
`auxiliary.*` blocks** at this provider, keeping Hermes's **main**
model on direct Ollama:

```yaml
model:
  default: "ollama:qwen3.6:35b"  # main loop stays direct, tools work

auxiliary:
  triage_specifier:              # example — pick the block that fits
    provider: audrey-kb
    model: audrey_<topic>
```

Why this works: auxiliary tasks (`triage_specifier`,
`kanban_decomposer`, `title_generation`, etc.) run as **separate
chat completions** with `tools=1` or none — *not* the 47-tool main
conversation (verified in `hermes-reference.md`). So the aux call to
the specialist is naturally tool-less, which is exactly the clean
pipeline path.

### Open caveat on the aux-block choice (resolve before building the Hermes side)

The auxiliary blocks fire on **Hermes's schedule** (their built-in
trigger semantics — triage, decomposition, titling), **not**
"whenever the agent wants to look something up." If the mental model
is "the agent should be able to consult the KB on demand mid-task,"
the aux-provider route may not fit, and the better option is
**Audrey-KB-as-a-Hermes-tool** (a single `ask_knowledge_base` tool
that POSTs to `audrey_<topic>` or directly to `/v1/kb/query`). That
*is* a Hermes tool call — but Hermes tool calls work fine
direct-to-Ollama, and it fires exactly when the model decides it
needs the KB.

**Action before wiring Hermes (not before building Audrey):** read
the aux-block trigger logic in the local Hermes checkout
(`~/.hermes/hermes-agent/`, the `agent/` and aux task source) to
confirm a block's trigger matches "consult KB." If none fit, switch
to the tool option. The **Audrey-side build is identical either
way**, so this decision does not block starting the build.

### Build order

The `audrey_<topic>` virtual model (Steps 2–5 below) is needed for
*any* client wiring and is wasted by neither outcome of the aux-vs-
tool question. Build the Audrey side first; settle the Hermes
integration shape against a working endpoint.

## Scope decisions (locked in)

- **Domain:** deferred. The prototype is written generically
  with `<topic>` placeholders; you pick the topic at
  implementation time.
- **KB scope:** curated topic + caller's per-user uploads. The
  specialist's `kb_search` retrieves from both, merged by score.
- **Tool surface:** kb_search, kb_image_search, web_search,
  memory_store, memory_recall, memory_search, chat_history_search.
  Same as `audrey_auto`.
- **Routing mode:** auto (gate decides per turn). Inherits the
  Phase 6a behavior — when Phase 6b ships, the specialist gets
  the fix automatically.

## Architecture

### What changes

Four touch points:

1. **`src/audrey/routes/openai.py`** — extend `VIRTUAL_MODELS`
   to include `audrey_<topic>`. Treat it like `audrey_auto` for
   routing (auto-gated), but flag the request as "specialist =
   <topic>" so the pipeline can branch.
2. **`src/audrey/pipeline/prompts.py`** — add a new specialist
   system prompt constant (`SPECIALIST_<TOPIC>_SYSTEM`) and a
   helper that returns it when the request is specialist-scoped.
   Plug it into `compose_system_messages` as a new layer that
   slots in after the task-role prompt and before memory recall.
3. **`src/audrey/pipeline/state.py`** (or wherever `PipelineState`
   lives) — add a `specialist: str | None` field so the
   identifier flows through classify → complexity → fast_path /
   deep_panel without re-deriving it.
4. **`src/audrey/tools/dispatch.py`** — when `state.specialist`
   is set and the tool being dispatched is `kb_search` or
   `kb_image_search`, inject a `topic=<topic>` argument before
   the call. The model doesn't need to know about the topic.

### What does NOT change

- `config.yaml` — no new schema. (Option 4 will add a
  `specialists:` block; the prototype hard-codes the one
  specialist.)
- `tools-server/` — the prototype reuses the existing
  `kb_search` endpoint. The topic argument is added at Audrey's
  dispatch layer, transparent to custom-tools. Custom-tools
  passes it through to `/v1/kb/query` which already supports
  per-topic filtering via the `kb` infrastructure.
- Memory / chat archive — same per-user scoping as today. Memory
  written under a specialist conversation is the same user's
  memory; it doesn't get sharded per-specialist. (Decide
  separately if that's the right call when generalizing.)

### Where the topic argument actually gets enforced

The specialist's KB scoping is **defense-in-depth at two layers**:

1. **System prompt** — tells the model to prefer the topic when
   it has a choice. Soft guard.
2. **Dispatch override** — Audrey rewrites the model's tool call
   to force `topic=<topic>` on `kb_search` / `kb_image_search`
   regardless of what the model emitted. Hard guard.

The hard guard is the one that matters. The model can mention
the topic in its `kb_search` argument or omit it — Audrey
overrides either way. This mirrors how `_USER_SCOPED_TOOLS`
already overrides the `user` argument with the authenticated
pipeline user for memory tools.

The "curated topic + user uploads" merge is implemented at the
KB query layer, not the dispatch layer. Audrey's
`routes/kb.py` `/v1/kb/query` already merges global KB hits with
per-user upload hits when a user is provided. The specialist
flow just hits this same endpoint with both `topic=<topic>` and
`user=<email>`; the existing score-merge logic handles the rest.

## Implementation steps

### Step 1 — pick the topic

Required before starting. Topic must already exist under
`/mnt/user/knowledge/<topic>/` and be visible to Audrey's
watcher (verify with `docker compose logs audrey-ai | grep
"kb watcher"`).

### Step 2 — wire the virtual model

In [`routes/openai.py:81`](../../src/audrey/routes/openai.py#L81),
extend `VIRTUAL_MODELS`:

```python
VIRTUAL_MODELS = (
    "audrey_deep",
    "audrey_cloud",
    "audrey_local",
    "audrey_auto",
    "audrey_fast",
    "audrey_<topic>",  # specialist prototype
)
```

Add a small helper that maps virtual model → specialist topic
(returns `None` for the generic models):

```python
SPECIALIST_TOPIC = {
    "audrey_<topic>": "<topic>",
}

def _specialist_topic_for(model: str) -> str | None:
    return SPECIALIST_TOPIC.get(model)
```

In the streaming gate (currently at the auto-routing decision in
`routes/openai.py` around the `forced_deep` / `forced_fast`
block), treat the specialist model like `audrey_auto`: gate
decides. Stash the specialist topic on the pipeline state /
options dict that flows downstream.

The non-streaming `_generate_via_pipeline` path needs the same
treatment — same gate behavior, same state-stashing.

### Step 3 — system prompt

In [`pipeline/prompts.py`](../../src/audrey/pipeline/prompts.py),
add:

```python
SPECIALIST_<TOPIC>_SYSTEM = (
    "You are a <topic> assistant. Prefer Audrey's knowledge "
    "base (kb_search / kb_image_search) for factual claims; "
    "fall back to web_search only when the KB doesn't cover "
    "the question. Cite specific sources when synthesizing."
)

SPECIALIST_PROMPTS = {
    "<topic>": SPECIALIST_<TOPIC>_SYSTEM,
}
```

In `compose_system_messages` (same file), accept a new
`specialist: str | None` parameter. When set and known, prepend
the matching prompt as the **second** system message, right
after the canonical incoming system message and before the
task-role prompt. The pin order becomes:

```text
incoming system  →  specialist  →  task-role  →  memory  →  chat-history
```

Update the existing `compose_system_messages` tests in
`tests/test_prompts.py` to cover specialist injection and
specialist-with-no-match.

### Step 4 — state propagation

`PipelineState` (TypedDict / dataclass) gains:

```python
specialist: NotRequired[str | None]
```

`build_pipeline_state` (the helper that builds state for graph
runs from the request) populates it from the virtual model. The
streaming path that bypasses the LangGraph build does the same
when invoking `run_fast_path` / `_stream_deep_with_banners`.

### Step 5 — dispatch override

In [`tools/dispatch.py`](../../src/audrey/tools/dispatch.py),
add an override mechanism analogous to `_USER_SCOPED_TOOLS`:

```python
_TOPIC_SCOPED_TOOLS = {"kb_search", "kb_image_search"}

def _apply_specialist_topic(tool_name: str, args: dict, specialist: str | None) -> dict:
    if specialist and tool_name in _TOPIC_SCOPED_TOOLS:
        return {**args, "topic": specialist}
    return args
```

Wire it into `dispatch_one` next to the existing user-scope
override. Order matters: user-scope first, then topic-scope —
they don't conflict (user is on a different argument).

Add unit tests in `tests/test_dispatch.py`:

- Specialist set → `kb_search` gets `topic=<topic>` regardless
  of what the model emitted.
- Specialist set → unrelated tools (e.g. `web_search`) are
  untouched.
- Specialist unset → no behavior change (matches current
  baseline).

### Step 6 — verify KB query path supports topic scoping

The `/v1/kb/query` route accepts `topic` already
([`routes/kb.py`](../../src/audrey/routes/kb.py)) for global
collection scoping. Quick check: when called with both `topic`
and `user`, it should query the topic's global collection AND
the user's upload collection, merge results by score. If it
doesn't, that's a small route fix; if it does, the prototype
just works.

Read `routes/kb.py` to confirm the merge behavior before
relying on it. If the route currently picks one source
exclusively when both are set, decide whether to:

- Adjust the route to merge (preferred, generalizes for future
  specialists).
- Have the dispatch override call `/v1/kb/query` twice and
  merge in custom-tools (workable but ugly).

### Step 7 — smoke tests on Unraid

After deploy, exercise from OWUI:

- **`audrey_<topic>` first turn, KB-shaped question.** Should
  route fast, dispatch `kb_search` with `topic=<topic>`
  injected. Verify in logs: `dispatch: kb_search ok ...`. The
  response should cite topic content.
- **`audrey_<topic>` first turn, off-topic question.** Specialist
  should still answer (system prompt doesn't refuse), but
  `kb_search` should still be topic-scoped. Confirm answer
  quality is reasonable for an off-topic prompt.
- **`audrey_<topic>` follow-up turn.** Phase 6a complexity-gate
  behavior applies. Note whether the follow-up routes deep
  (expected pre-6b) and matches the bug you've been
  characterizing.
- **`audrey_<topic>` with a user upload.** Upload a file via
  `/upload`, ask a question covering it. The specialist should
  surface the upload alongside topic content.
- **`audrey_auto` regression test.** Same topic-shaped question
  on `audrey_auto` — confirm the specialist hasn't changed
  generic behavior.

## Open decisions you'll want to make before implementing

- **Display name.** `audrey_<topic>` is the wire-level name.
  OWUI shows whatever you put in the `/v1/models` response.
  Decide whether OWUI should show "Audrey — Topic Helper" or
  just `audrey_<topic>`.
- **Should the specialist refuse off-topic?** The plan above
  has the system prompt prefer KB but not refuse. You may want
  a stricter persona that says "I'm scoped to <topic>, ask
  about that." Easy to swap.
- **Memory scoping.** As written, specialist conversations share
  memory with regular Audrey conversations under the same user.
  Alternative: per-specialist memory namespacing (memory_store
  / memory_recall keyed on `user + specialist`). The prototype
  defers this; revisit when generalizing.
- **Chat archive integration.** The specialist's conversations
  appear in `chat_history_search` results the same as any
  other audrey conversation. If you want specialists isolated
  from each other in chat history, that's a separate
  archiving-side change.
- **Multiple specialists in one process.** The prototype hard-
  codes one. When you add a second by hand, the structure
  (dict-based `SPECIALIST_TOPIC`, `SPECIALIST_PROMPTS`)
  naturally accommodates it. Adding the third or fourth is the
  signal to generalize to Option 4 (`config.yaml` block).

## What this prototype does NOT cover

- **Config-driven specialists.** Hard-coded only. Adding a new
  specialist requires a code change + deploy. That's
  intentional — Option 4 is the followup once the pattern is
  proven.
- **Specialist-specific tool restrictions.** All specialists
  see the full tool surface. If you want a "KB-only" specialist
  with web_search disabled, that's an Option 4 feature.
- **Specialist-specific routing mode.** All specialists are
  auto-gated. A "deep-only research" specialist would also be
  Option 4.
- **OWUI Workspace integration.** The prototype expects users
  to pick the specialist from OWUI's model dropdown directly.
  No OWUI workspace knowledge collections, no OWUI-side
  permissions. Treat OWUI as a passthrough.
- **Phase 6a complexity-gate fix.** The specialist inherits
  whatever the gate does. Don't try to patch the gate as part
  of this prototype; Phase 6b will handle it.

## When to come back to this

Three triggers, any one of which makes this worth picking up:

- **Phase 6b ships and the complexity gate behavior is settled.**
  Specialists become more useful when follow-up turns route
  predictably.
- **Phase 4 manual UI smoke tests surface a "wish I had a
  specialist for X" gap.** The plan is ready to drop in.
- **You hit two specialists by hand.** That's the signal to
  generalize to Option 4 — but Option 2 is the validation
  step, so build the second hand-coded one first, *then*
  generalize.

Estimated implementation cost from scratch: **1-2 focused
sessions** (half a day each). The largest unknown is the
`routes/kb.py` topic + user merge behavior in Step 6 — that's
the only step that could surprise.

## Followup work that this plan unblocks

- Option 4 generalization (config-driven specialists, ~1-2 days
  on top of the prototype).
- Per-specialist memory namespacing.
- Specialist-specific tool restrictions.
- "Refuse off-topic" persona variant.
- A per-specialist `/v1/models` description string surfaced in
  OWUI's model picker.
