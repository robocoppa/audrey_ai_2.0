# Campaign 3 Phase 3 — reusable skills

**Status:** Explicit `video-analysis` and `grounded-document-analysis` are
complete. Native automatic English selection (3D.6) is built; browser acceptance
is pending. `skills.auto_select: false` remains the default until acceptance.
English input only.

## Goal and decisions

Skills are local, versioned instruction/resource bundles for reusable task
workflows. Tools supply actions; skills coordinate how an authorized task uses
them. Skills do not create another execution runtime or virtual model per task.

- The first release is admin-installed, declarative, read-only, non-executable,
  and limited to exactly one active skill per request.
- No skill means unchanged ordinary routing, tools, prompts, and streaming.
- Explicit selection wins. A skill narrows capabilities and can never grant
  permissions, supply user identity, or override platform policy.
- No user/marketplace installation, remote fetching, executable bundles,
  `skill_load` tool, or multi-skill composition is in scope.

## Bundle and registry contract

A directory contains `SKILL.md` with YAML front matter and bounded instructions,
plus declared bounded read-only resources. The manifest defines stable id/name,
description, immutable positive version, allowed tools, supported modes, and
resource paths. Compute a content digest for provenance.

- The slug matches the directory. Reject duplicate ids, unknown keys, malformed
  metadata, unsupported files, executable content, and exceeded size budgets.
- Resources remain inside the bundle: reject absolute/traversal paths, escaping
  symlinks, device files, and undeclared or unsupported resources.
- Configured roots are mounted read-only. Load at startup and authenticated
  admin rediscovery; registry swaps are atomic and in-flight specs stay immutable.
- Invalid optional bundles degrade individually without blocking ordinary chat.
  Missing declared tools affect skill availability and never broaden tool access.
- Disabled skills preserve ordinary behavior. Invalid configuration limits,
  including `max_active != 1`, fail startup.

## Request, prompt, and tool contract

- Optional `skill` names one skill. Unknown/conflicting/incompatible choices
  return `400`; a known unavailable skill returns component-aware `503`.
  Passthrough plus a skill is rejected explicitly.
- Authenticated `GET /api/skills` returns safe catalog metadata and availability,
  never instruction bodies or private paths.
- `audrey_video` maps to `video-analysis`. A conflicting explicit selection is
  rejected. Preserve its existing task instruction and adaptive routing.
- Resolve before stream/non-stream and fast/deep branching. Inject the selected
  instruction once through the existing task-role mechanism.
- Skill context reaches the model and counts in real prompt usage, but is
  excluded from the user-request complexity decision. Escalation retains the
  same skill and restrictions.
- Model-visible and dispatchable tools are the intersection of discovered tools,
  platform policy, and the selected skill allowlist.
- Internal memory, archive, health, and lifecycle work keep platform capabilities.
  Server-owned user scoping still applies after restriction.
- Record id/version/digest/reason in run, log, archive, and evaluation provenance.
  Metrics use bounded labels; do not expose private instructions or content.

## Shipped explicit workflows

`video-analysis` packages the existing video workflow. The explicit
`grounded-document-analysis` v1 pilot coordinates existing file/KB tools,
separates facts by source file, states partial-read and missing-evidence limits,
and compares retrieved contents. Its ship decision is accepted; no unchanged
answer-comparison or native-selection gate remains.

## 3D.6 — native automatic selection

Native runs and AG-UI chat use the existing English evidence rules when
`skills.auto_select` is enabled. Explicit request/virtual-model choices run first.
Bot principals, direct models, and compatibility API handlers retain their
existing behavior. No classifier generation, router/model changes, or extra
skill bundles are introduced.

- Classify only the current user request and server-verified filename/kind
  metadata. File bodies, retrieved passages, project instructions, and prior
  assistant answers cannot select a skill.
- Evidence comes from current owned Ready attachments and the current Project
  snapshot. File followups revalidate the most recent canonical attachment set;
  deleted/unready evidence cannot authorize selection or silently fall back to
  an older attachment. Ordinary chat abstains before historical-file lookup.
- Affirmative document/video questions select one available compatible bundle.
  Quoted/code content, access denials, excluded files, file management, unresolved
  names, mixed requested kinds, and unsupported image/audio kinds abstain.
  An explicit supported target can exclude unrelated evidence of another kind.
- Selection is bounded to 20,000 request characters, 100 evidence entries, and
  300 filename characters. Larger/unclear requests use ordinary chat. Rule
  abstention changes no request permission or existing model behavior.
- The same immutable spec drives prompt instructions, tool restrictions,
  streaming, and persisted id/version/digest. `skill_reason` is `automatic`;
  existing run records, bounded metrics, and `skill.selected` logs expose it.

### Acceptance and activation

Use real Ready document and video files in native chat with **Tools and skills →
Automatic**. Confirm each answer uses its file evidence and a followup remains
grounded. Compare a document question with an explicit document-skill choice,
and confirm ordinary chat still answers normally. Server provenance must show
the expected skill with `reason=automatic`; explicit selection keeps
`reason=request`. This is a targeted workflow check, not a claim about population
precision or answer quality from synthetic labels. No broad live evaluation is
required for the rule-only opt-in slice.

The previous standalone selector study remains historical/regression evidence.
Its model-based measurement machinery does not activate production selection.
A classifier fallback remains parked until a real rules miss warrants it. The
failed Japanese branch is out of scope, not passed; do not tune/rerun it or add
runtime language rejection.

## Source map

Registry and resolution: `src/audrey/skills/`. Bundles: `skills/`.
Configuration: `config.yaml`. Prompt integration: `src/audrey/pipeline/`.
Tool policy/restriction: `src/audrey/tools/`. Standalone study:
`evals/eval_skill_selection.py`; preserved evidence: `evals/results/`.
