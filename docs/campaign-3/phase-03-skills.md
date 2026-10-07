# Campaign 3 Phase 3 — reusable skills

**Status:** Complete for explicit `video-analysis` and `grounded-document-analysis`. Automatic selection remains deferred and
`skills.auto_select: false`. No selector experiment is required for campaign
completion. User input is English-only.

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

## Automatic selection — parked

The standalone evaluator does not select runtime skills, execute tools, read
real uploads, or change production routing. Its preserved fixtures and reports
are historical/regression evidence, not untouched validation after tuning.

The Japanese 3D.5 check failed and is closed as out of scope, not passed.
Do not tune or rerun non-English cases. Preserve artifacts without adding
runtime language rejection.

Reopen only for a concrete English workflow and agreed benefit. Before activation:

1. Obtain human-reviewed fresh labels and agree precision, miss, ordinary false
   activation, error, latency, token/context, and call-budget limits.
2. Freeze evaluator source, cases, catalog, model tag, settings, and predicted
   call budget before measurement. Detect drift before HTTP; preserve private
   plans/reports without overwrites. Human attestations require actual review.
3. Explicit/virtual selection precedes deterministic rules, then an optional
   metadata-only router for undecided eligible cases, then abstention.
   Firm denials, unavailable evidence, and unsupported modes make no model call.
4. Distinguish quoted data from instructions and requested evidence from excluded
   files. Reject ineligible choices; keep guards and actual model decisions separate.
5. Report execution, label findings, and qualification separately. Small repeated
   synthetic samples do not establish production rates or final-answer benefit.
   Compare real answer/workflow benefit against existing explicit selection.
6. Keep automatic selection disabled until those gates are explicitly accepted.

## Source map

Registry and resolution: `src/audrey/skills/`. Bundles: `skills/`.
Configuration: `config.yaml`. Prompt integration: `src/audrey/pipeline/`.
Tool policy/restriction: `src/audrey/tools/`. Standalone study:
`evals/eval_skill_selection.py`; preserved evidence: `evals/results/`.
