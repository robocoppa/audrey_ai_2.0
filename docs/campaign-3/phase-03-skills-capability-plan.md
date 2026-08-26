# Campaign 3 Phase 3 — reusable skills

**Status:** Planned and gated on Campaign 3 Phase 2.

## Goal

Add a first-class, local skills capability to Audrey: versioned bundles of
instructions and optional read-only resources that can specialize a request
without creating another hard-coded virtual model or advertising every tool to
every workflow.

The first release is admin-installed, declarative, and non-executable. It
generalizes the existing `audrey_video` task role while preserving all normal
fast/deep/streaming behavior when no skill is selected.

## Decision

Skills are beneficial for Audrey when they are an **instruction-orchestration
layer**, not another tool/plugin runtime.

```text
memory / KB       = facts and user knowledge
tools             = actions the model may perform
pipeline mode     = how Audrey schedules model work
skill             = how to perform one reusable kind of task
```

OpenAI's current platform uses project-scoped skills with downloadable bundles,
immutable versions, and a default-version pointer. Containers can receive a
versioned skill reference or an inline skill bundle. Audrey should adopt those
useful contract ideas locally rather than depend on a hosted runtime:

- [OpenAI Skills API](https://developers.openai.com/api/reference/go/resources/skills)
- [Container skill attachment](https://developers.openai.com/api/reference/cli/resources/containers/methods/create)

## Phase 1 and Phase 2 prerequisites

Phase 3 starts only after the platform-hardening and native-UI completion
gates, with particular reliance on:

- private-data policy is enforced server-side rather than by prompt text;
- request tasks and GPU scheduling have explicit lifecycle ownership;
- streaming and request contracts have one owner and terminal outcome;
- model-visible capabilities have declarative policy and availability;
- configuration precedence and generated inventories are trustworthy;
- the native UI can discover skills and send an explicit skill id without
  owning selection, authorization, or prompt assembly;
- Audrey-owned conversation and run records retain immutable skill provenance.

A registry-only spike may be developed earlier behind an off-by-default flag,
but it does not ship or become an implicit dependency of `audrey_video` until
Phases 1 and 2 are closed.

## Product rules

- Zero behavior change when no skill is selected.
- Exactly one active skill per request in the MVP.
- Explicit selection wins. Automatic selection is a later, separately gated
  feature.
- A skill may restrict model-visible tools; it can never grant a capability or
  permission.
- Authenticated identity and user scoping remain platform policy.
- Skills contain no executable code in the MVP.
- Skills cannot fetch remote content during load.
- Passthrough remains skill-free unless the client explicitly opts into a
  future, separately documented contract.
- Invalid optional skills degrade individually; they do not prevent ordinary
  Audrey chat from starting.
- Skill version and activation reason travel with logs/eval/archive evidence.
- Skill instructions do not influence the fast-versus-deep complexity gate.

## Non-goals

- No arbitrary Python, shell, JavaScript, or binary execution from a skill.
- No user-uploaded or marketplace skills in the first release.
- No automatic installation from GitHub, URLs, or hosted skill registries.
- No multi-skill composition until one-skill ordering and evaluation are
  understood.
- No skill may override network policy, auth, user identity, quota, data
  retention, or server-side tool scoping.
- No new model-callable `skill_load` tool.
- No replacement of existing stage prompts such as classifier, planner,
  researcher, verifier, or synthesizer prompts. Those define Audrey's pipeline;
  a skill defines the user's task workflow.

## Why not implement skills as a tool

A `skill_load` OpenAPI tool looks simple but has the wrong lifecycle:

- only tool-capable models could use it;
- selection would cost a ReAct round and happen after routing;
- the model could skip it, repeat it, or load conflicting instructions;
- plain streaming and non-tool models would behave differently;
- tool dispatch is for actions, not for changing the instruction contract that
  governs the whole request.

Skill resolution belongs at the authenticated route boundary, before Audrey
splits into streaming/non-streaming and fast/deep paths.

## Bundle format

Use a directory bundle with one required `SKILL.md` and optional bounded
resources:

```text
skills/
  video-analysis/
    SKILL.md
    references/
      retrieval-rules.md
    templates/
      comparison.md
```

`SKILL.md` begins with YAML front matter followed by the instruction body:

```markdown
---
id: video-analysis
name: Video analysis
description: Read and compare uploaded transcripts, summaries, and frames.
version: 1
allowed_tools:
  - list_my_files
  - get_file_text
  - kb_search
  - kb_image_search
supported_modes:
  - auto
  - fast
  - deep
resources:
  - references/retrieval-rules.md
---

<bounded instruction text>
```

MVP validation rules:

- `id` is a stable lowercase slug and matches the directory name.
- `version` is an immutable positive integer for that content. Audrey also
  computes and logs a content digest.
- `description` is short enough to use in a selector/catalog.
- `allowed_tools` contains only known model-visible tools. Missing optional
  tools mark the skill degraded; they never expand the registry.
- Resource paths are relative, resolve inside the bundle, and reject absolute
  paths, `..`, symlinks that escape the root, device files, and unsupported
  types.
- Instruction, individual resource, total bundle, and active-context sizes
  have hard caps.
- `scripts/`, executable files, and unknown manifest keys are rejected in the
  MVP.

The loader reads bundles at startup and on an authenticated admin rediscovery
operation. It never watches writable directories continuously in the first
release.

## Core types

Recommended internal shape:

```python
@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    name: str
    description: str
    version: int
    digest: str
    instructions: str
    allowed_tools: frozenset[str]
    supported_modes: frozenset[str]
    resources: tuple[SkillResource, ...]

@dataclass(frozen=True, slots=True)
class ResolvedSkill:
    spec: SkillSpec
    reason: Literal["request", "virtual_model", "automatic"]
```

`SkillRegistry` owns validated specs, per-skill availability, and diagnostics.
It is distinct from `ToolRegistry`:

- `SkillRegistry` answers what instructions apply.
- `ToolRegistry` answers what model-callable actions exist.
- Phase 1's tool policy answers whether those actions are allowed and healthy.

## Request and selection contract

### Explicit API selection

Add one vendor extension to `ChatCompletionRequest`:

```json
{
  "model": "audrey_auto",
  "skill": "video-analysis",
  "messages": [...]
}
```

MVP rules:

- absent/null means no skill;
- unknown id returns a clear 400 with the available ids;
- known but unavailable/degraded skill returns a component-aware 503;
- a skill incompatible with the chosen virtual-model mode returns 400;
- passthrough plus `skill` is rejected explicitly rather than ignored;
- only one string is accepted, not a list.

Add an authenticated `GET /v1/skills` catalog containing safe metadata only:
id, name, description, version, supported modes, and availability. It does not
return instruction bodies or internal file paths.

### Virtual-model compatibility

Keep `audrey_video` in `/v1/models` during the migration. Resolve it internally
to `video-analysis` with reason `virtual_model`. The request field wins only
when it names the same skill; conflicting explicit and virtual selections are a
400 so behavior is never ambiguous.

Do not create a virtual model for every future skill. Virtual aliases are for
the small set Open WebUI needs to expose in its model picker while direct API
clients use `skill`.

### Automatic selection — deferred within the phase

Automatic selection remains off by default:

```yaml
skills:
  enabled: true
  auto_select: false
  max_active: 1
```

When evaluated later, selection order is:

1. explicit `skill` request;
2. virtual-model compatibility mapping;
3. deterministic high-confidence rules;
4. optional small-model selector for ambiguous cases;
5. no skill.

The selector sees skill ids/descriptions, not full instructions/resources.
It must be able to abstain. A low-confidence selection is no selection, not a
guess.

## Prompt composition and routing

The current video role is inserted at the route because that is the one point
shared by streaming and non-streaming and independent of memory. Preserve that
property.

Resolution flow:

```text
authenticate request
  -> validate virtual model + requested skill
  -> resolve skill/version/reason
  -> build model-visible tool subset
  -> inject selected skill instruction once
  -> existing stream/non-stream split
  -> existing classify + complexity + fast/deep behavior
```

Implementation rules:

- Generalize `task_role_for`, `with_task_role`, and `without_task_role` rather
  than layering a second injection mechanism next to them.
- The `audrey_video` migration must produce byte-identical prompt content and
  placement before any wording is changed.
- Preserve the existing position after incoming leading system messages at the
  route. Date/memory injection retains its current downstream behavior during
  the parity milestone.
- Generalize the complexity-gate exclusion so all Audrey-added skill
  instructions are removed from the gate's view in both graph and streaming
  paths.
- Count skill text in real prompt telemetry because it is sent to the model;
  exclude it only from the decision that estimates user-request complexity.
- Add parity tests that pin graph and streaming gate behavior together.

Do not rely on marker keys in messages sent to Ollama. Carry selected skill id,
version, digest, reason, and exact injected text in request/pipeline state, and
remove the exact Audrey-owned message for the gate just as the task role is
handled today.

## Tool restriction model

Add a non-mutating `ToolRegistry.restrict(names)` operation that returns a new
registry containing the intersection of:

```text
live discovered tools
  ∩ platform policy allowed tools
  ∩ selected skill allowed_tools
```

Important separation:

- Audrey-internal memory recall, archive capture, health, and lifecycle work
  use the platform registry/capabilities.
- Model-facing classify/ReAct/panel/research calls use the restricted registry.
- Dispatch receives that restricted registry too, so a hallucinated disallowed
  tool is `unknown_tool`; hiding the schema is not the only enforcement.
- `_USER_SCOPED_TOOLS`/the Phase 1 policy still overwrites identity after the
  skill restriction. A skill never supplies `user`.
- A missing declared tool degrades or rejects that skill according to whether
  it is marked required; it does not silently substitute a broader tool.

The no-skill path receives the same registry it receives before Phase 3.

## Resource delivery

MVP resources are small, declared, and included only for the selected skill.
They are assembled into one bounded Audrey-owned system message after bundle
validation. Do not expose a general filesystem-reading tool.

If real skills later need large references:

1. prefer ingesting stable material into the existing global KB with explicit
   provenance; or
2. add a dedicated read-only `skill_resource` capability with bundle-root path
   enforcement and per-request budgets.

That is a later measured decision. The MVP must not turn every skill bundle
into unbounded prompt context.

## Security model

First release:

- only the administrator installs bundles into a configured read-only root;
- roots are explicit in `config.yaml` and bind-mounted read-only in the
  container;
- no network download, package install, executable file, or subprocess;
- strict path containment and symlink handling;
- bounded parsed YAML and text sizes;
- unknown keys and duplicate ids fail that skill closed;
- a skill cannot mention/grant a tool absent from Phase 1 model-visible policy;
- instruction text is untrusted guidance, never an authorization control;
- logs never dump complete private skill instructions by default;
- catalog responses expose metadata, not source paths or content.

User-authored/imported skills are a later feature requiring separate storage
isolation, ownership, malware/supply-chain handling, permission review, export,
deletion, and UI. They are not an incremental checkbox on the admin loader.

## Configuration

Proposed shape:

```yaml
skills:
  enabled: false
  roots:
    - /app/skills
  auto_select: false
  max_active: 1
  max_instruction_chars: 12000
  max_resource_chars: 12000
  max_bundle_chars: 32000
  virtual_models:
    audrey_video: video-analysis
```

Rules:

- disabled/absent preserves current behavior;
- paths describe where this deployment stores bundles; bundle contents
  describe what the skills are;
- environment variables may override location/enablement only when documented
  and tested under the Phase 1 precedence contract;
- invalid limits or `max_active != 1` are startup configuration errors for the
  MVP;
- an invalid optional bundle is reported in skill readiness without blocking
  ordinary chat.

## Observability and evidence

Per request, log structured fields:

- skill id, version, digest prefix, and activation reason;
- selection latency and outcome;
- visible tool names after restriction;
- unavailable/missing declared tools;
- instruction/resource character counts;
- virtual model, final pipeline mode, and concrete model as today.

Add bounded-cardinality metrics such as:

- `audrey_skill_requests_total{skill,reason,outcome}`;
- `audrey_skill_selection_seconds{reason}`;
- `audrey_skill_registry_available` and invalid-bundle count.

Do not put a content digest or version with unbounded churn in metric labels;
record those in logs and archive metadata.

Archive/eval artifacts record the skill id/version/digest/reason so an answer
can be reproduced against the instruction bundle that generated it. Skill
instructions themselves are not copied into every chat record.

---

## Milestone 3A — registry and behavior-preserving video extraction

1. Add `src/audrey/skills/` models, loader, registry, and validation.
2. Package a built-in `video-analysis` bundle containing the exact existing
   video specialist instruction.
3. Add config loading and component readiness, disabled by default initially.
4. Resolve `audrey_video` through the registry while retaining a temporary
   fallback to the existing built-in constant for rollback.
5. Generalize task-role injection/exclusion helpers without changing final
   messages or routing.

Gate:

- snapshot proves byte-identical injected content and placement;
- `audrey_video` stays adaptive like `audrey_auto`;
- skill prompt tokens do not change the complexity decision;
- streaming and non-streaming produce the same resolved-skill metadata;
- missing/invalid bundle leaves general chat healthy and makes skill readiness
  precise;
- no-skill request payload, model choice, tools, and prompt remain unchanged.

## Milestone 3B — explicit selection and enforced tool restriction

1. Add the `skill` request extension and authenticated catalog route.
2. Carry `ResolvedSkill` metadata through graph and streaming state.
3. Add non-mutating model-visible registry restriction.
4. Enforce supported modes and passthrough rejection.
5. Record logs, metrics, archive, and eval metadata.

Gate:

- explicit selection works for auto, forced-fast, and forced-deep modes;
- fast-to-deep escalation retains the same selected skill and restriction;
- every deep worker, researcher, verifier/fact-check stage that accepts tools,
  and final model call sees the intended instruction/tool contract;
- disallowed/hallucinated tools cannot dispatch;
- user identity remains server-owned under a restricted registry;
- unknown, conflicting, unavailable, and passthrough selections return the
  documented errors;
- no-skill regression suite passes.

## Milestone 3C — first new skill and product evaluation

Pilot a `grounded-document-analysis` skill using only the existing file/KB
capabilities it needs, for example:

- `list_my_files`;
- `get_file_text`;
- `kb_search`;
- `kb_image_search` when explicitly relevant.

It should coordinate cross-tool behavior not already owned by one tool
description: selecting among files, separating evidence by file, stating
partial-read limits, and producing a comparison from retrieved content.

Do not duplicate instructions already present in tool descriptions. Campaign
2's video-specialist work repeatedly found that globally correct tool
descriptions were better than a specialist-only reminder.

Evaluation:

- build feature and regression cases before tuning the prompt;
- compare skill-selected requests with `audrey_auto` on identical cases;
- interleave arms and repeat to measure sampling noise;
- inspect saved answers, tool traces, selection metadata, latency, prompt
  tokens, tool calls, and cloud cost;
- require no regression on ordinary no-skill requests;
- separate structural harness checks from human answer-quality judgment.

The registry can ship even if one pilot prompt shows no gain, because it also
replaces hard-coded specialization plumbing. The pilot skill becomes a public
choice only if it is measurably useful or clearly improves workflow
consistency at acceptable cost.

## Milestone 3D — optional automatic selection

Open only after explicit selection has stable evidence.

1. Start with deterministic, high-precision rules and an abstain default.
2. Add a small-model selector only for cases rules cannot decide.
3. Load only skill catalog metadata for selection; full content loads after a
   choice.
4. Record false activation and missed activation in the eval report.
5. Keep `auto_select: false` as instant rollback.

Ship automatic selection only if repeated evaluation shows:

- useful-task selection precision at an agreed threshold;
- no material ordinary-chat regression;
- acceptable selector latency/token cost;
- skill prompt/context growth stays within budget;
- wrong selections are visible and reversible.

## Test matrix

### Loader and registry

- valid minimal and resource-bearing bundles;
- duplicate id/version, unknown keys, invalid YAML, empty body;
- absolute path, traversal, escaping symlink, device/unsupported file;
- per-file and total size caps;
- unknown/disallowed/missing tool declaration;
- one bad optional bundle does not block general chat;
- rediscovery swaps registry atomically and in-flight requests retain their
  resolved immutable spec.

### Request and routing

- no skill, explicit skill, virtual alias, conflicts, unknown, unavailable;
- auto/fast/deep/research compatibility rules;
- stream/non-stream parity;
- short and long prompt complexity parity with and without the same skill;
- image-turn and OWUI utility-task ordering remains unchanged;
- fast escalation carries skill state once, with no duplicate injection;
- passthrough rejects the extension.

### Prompt and tools

- exact insertion order with incoming system, datetime, memory, and chat-history
  guidance;
- prompt appears once in every model call that should receive it;
- internal memory/archive capabilities are not accidentally removed;
- model-visible registry is the required intersection;
- a skill cannot grant an absent tool or another user's identity;
- disallowed dispatch returns `unknown_tool` without touching a server;
- resource assembly is deterministic and capped.

### Lifecycle and operations

- disabled feature is byte-for-byte/no-call-path equivalent;
- malformed skill readiness without whole-service failure;
- admin rediscovery during in-flight requests;
- archive/eval metadata contains id/version/digest/reason;
- bounded-cardinality metrics;
- container bundle root is read-only and non-root.

## Likely files

New:

- `src/audrey/skills/models.py`
- `src/audrey/skills/loader.py`
- `src/audrey/skills/registry.py`
- `src/audrey/skills/selection.py` only when automatic selection opens
- `skills/video-analysis/SKILL.md`
- `skills/grounded-document-analysis/SKILL.md`
- focused skill tests under `tests/`

Changed:

- `src/audrey/main.py` for registry lifecycle/readiness;
- `src/audrey/config.py` and `config.yaml`;
- `src/audrey/routes/openai/schemas.py` and `routes.py`;
- `src/audrey/routes/openai/pipeline.py`;
- `src/audrey/pipeline/prompts.py`, `state.py`, and `graph.py`;
- `src/audrey/tools/discovery.py` for non-mutating restriction;
- chat/eval metadata and metrics;
- Compose/Dockerfile bundle mount;
- model/skill catalog documentation.

This list is a starting map, not permission to touch every file in one change.
Milestones stay independently reviewable.

## Deployment and rollback

1. Deploy 3A with `skills.enabled: false`; verify registry/readiness only.
2. Enable built-in video mapping and run the existing video comparison suite.
3. Enable explicit API selection for the pilot user/admin only.
4. Publish the first new skill only after its evaluation gate.
5. Leave automatic selection off until 3D passes.

Rollback order:

- disable `auto_select`;
- remove public exposure of the pilot skill;
- disable explicit skills while retaining `audrey_video` fallback;
- disable the registry and return to the existing built-in role mapping.

No rollback changes stored user data because MVP skills are read-only bundles
and request metadata.

## Phase 3 completion gate

Phase 3 is complete when:

- the registry loads versioned, validated, read-only bundles;
- `audrey_video` behavior and routing remain at parity through the registry;
- explicit skill selection works across supported fast/deep and stream/non-
  stream paths;
- tool restrictions are enforced at offer and dispatch time without weakening
  user scoping;
- no-skill behavior remains unchanged;
- skill version/reason is visible in logs, readiness, archive, and eval
  evidence;
- one new skill has completed a repeated control comparison and has an explicit
  ship, revise, or reject decision;
- automatic selection is either proven and shipped behind its switch or remains
  explicitly deferred;
- full hermetic tests, changed-file ruff, lesson-link checks, user-run Unraid
  smoke, and relevant live evals pass.
