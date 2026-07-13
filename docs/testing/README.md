# Audrey mode test suite

Live, repeatable accuracy/quality evals for each Audrey virtual model. The
hermetic suite (`tests/`) proves the plumbing; these put a real model on the
other end and let you read the actual answers. One harness
(`scripts/eval_research.py`) drives all of them — the cases differ, the script
doesn't.

See `../plans/PLAN-mode-test-suite.md` for the design rationale and case taxonomy.

## The protocols

| Protocol | Cases file | Default model | What it tests |
|----------|-----------|---------------|---------------|
| **Research** | `scripts/eval_prompts_protocol.json` | `audrey_research` | Staged research→verify→fact-check→write; grounding, citation, hedge-vs-drop discipline |
| **Deep** | `scripts/eval_prompts_deep.json` | `audrey_deep` | Panel synthesis quality across a wide topic range; flourish-leak failure mode; merge coherence; 2 coding anchors |
| **Fast** | `scripts/eval_prompts_fast.json` | `audrey_fast` / `audrey_auto` | Answer quality on simple turns; **routing correctness** (fast/deep gate); **latency/TTFT** |
| **Code** *(opt-in)* | `scripts/eval_prompts_code.json` | `audrey_deep` | Coding on the deep panel's code pool: implement/debug/refactor, with **executed** Python asserts (`code_runs`). The **easy/regression** tier — every pooled model tends to pass, so it's a plumbing gate, not a discriminator |
| **Code-hard** *(opt-in)* | `scripts/eval_prompts_code_hard.json` | `audrey_deep` | The **discriminating** coding tier for lineup optimization: edge-case-heavy algorithms (TTL+LRU cache, deterministic topo-sort with cycle detection), subtle async exception isolation, strict parsers/tokenizers. Asserts cover the corners easy prompts miss, so models actually diverge |
| **Topics** *(opt-in)* | `scripts/eval_prompts_topics.json` | `audrey_deep` | Reasoning/math (`reasoning-*`), explanation (`science-*`), instruction-following prose (`writing-*`), factual recall vs hedging (`gk-*`) |

Research/deep/fast are the default trio; **code, code-hard, and topics run on
demand** (`scripts/run_all_evals.sh code code-hard topics`) so the routine
full-suite runtime doesn't grow. Reach for **code-hard over code** when the
question is "which models earn a panel seat" — the easy tier can't separate them
(a real run had all pooled models pass every easy case; the only signal was
per-worker latency in the drafts). `code` and `code_models` also double as
**per-model sweep sets** — see "Per-model sweeps" below.

## What the harness checks

Per case, against the reassembled streamed answer:

- **reachable / no_error_marker / has_answer** — the turn completed with a real body.
- **banners** — the mode's progress banners appeared in order (deep:
  Planning→Dispatching→Synthesizing; research: Planning→Researching→Verifying→
  Writing; fast: a single Thinking banner). N/A when a case sets `expect_banners: false`.
- **sources / url_wellformed** — `## Sources` present with valid URLs (research
  grounding cases; opt out with `expect_sources: false`).
- **route** — *(opt-in, for `audrey_auto`)* the inferred path (fast/deep/research,
  read from the banner family) matches the case's `expect_route`. This is how the
  fast/deep gate (token threshold + deep-intent phrases) is tested.
- **code_block** — *(opt-in: `expect_code: true`, implied by `code_test`)* a
  language-tagged fenced code block exists in the answer.
- **code_runs** — *(opt-in: `code_test`)* the largest ` ```python ` block is
  extracted, the case's asserts are appended, and the file **actually runs** in
  a subprocess (scratch cwd, `code_timeout` seconds, default 15). Pass iff exit
  0 — the one objective correctness signal in the suite. Safety posture: the
  executed code is our own models' output answering our own stdlib-only
  prompts, on this laptop, with only the timeout for isolation — keep case
  prompts stdlib-only and don't point `code_test` at anything with side effects.
- **contains** — *(opt-in: `answer_contains: [..]`)* every listed string appears
  case-insensitively in the answer body. A weak objective signal for
  reasoning/knowledge cases whose right answer has a distinctive token
  ("82.8", "tungsten"). Pick tokens that survive formatting variance (avoid
  "8,611"-style numbers).
- **latency** — *(measured for every case, not pass/fail)* TTFT (first content
  delta) and total wall-clock. Printed per case and saved in the answers file.

Quality itself is **your read** of the printed answers — the checks are
guardrails, not a grader. Each report is one human evaluation pass, one sample
per case, on a non-deterministic system.

## First-time setup (once per laptop)

You need two things; the harness refuses to start without them.

1. **A venv** — `.venv/bin/python` must exist. From the repo root: `uv sync`.
2. **`.env.test.local`** at the repo root (gitignored via the `.env.*.local`
   rule — laptop-local, never committed). The harness auto-loads it:

   ```text
   # .env.test.local  (repo root)
   AUDREY_EVAL_BASE_URL=http://192.168.1.11:8080/api   # OWUI host:port + /api
   AUDREY_EVAL_API_KEY=sk-...                          # OWUI API key
   ```

   Mint the `sk-…` key in OWUI → Settings → Account → API Keys. (Why OWUI and
   not Audrey directly: Audrey's API needs an OWUI-validated bearer token, so
   the script hits OWUI's OpenAI-compatible API and OWUI forwards to Audrey with
   the session JWT — the repeatable path. Full rationale in the script docstring.)

You must also be **on the LAN/VPN** to reach the box — these run against the
live stack, never from off-network and never from inside a container.

## Running — the whole suite in one command

```bash
scripts/run_all_evals.sh
```

This chains the default trio (research + deep + fast) and writes today's dated
answers files into `docs/testing/` (`<date>-research-answers.md`, `-deep-`,
`-fast-`). It does the same preflight checks (venv, `.env.test.local`, valid
protocol) and **exits non-zero if any case in any protocol failed a check**, so
it can gate CI / a pre-deploy check. The trio is **slow (~60–90 min, ~40
cases)** — run it in the background and wait; don't foreground-block on it.

```bash
scripts/run_all_evals.sh research deep      # only the named protocols, in order
scripts/run_all_evals.sh code topics        # the opt-in protocols (not in the default trio)
DATE=2026-07-01 scripts/run_all_evals.sh    # override the date stamp
ONLY=euclid scripts/run_all_evals.sh research   # pass --only through to one case
```

## Running — one protocol at a time

The runner is just a wrapper; you can call the harness directly to control the
model, cases, and output file:

```bash
# Research (the original protocol)
.venv/bin/python scripts/eval_research.py \
  --cases scripts/eval_prompts_protocol.json \
  --save-file docs/testing/$(date +%F)-research-answers.md

# Deep
.venv/bin/python scripts/eval_research.py --model audrey_deep \
  --cases scripts/eval_prompts_deep.json \
  --save-file docs/testing/$(date +%F)-deep-answers.md

# Fast + auto (routing + latency)
.venv/bin/python scripts/eval_research.py --model audrey_fast \
  --cases scripts/eval_prompts_fast.json \
  --save-file docs/testing/$(date +%F)-fast-answers.md
```

Run one case with `--only <substring>`. `--no-answers` suppresses the per-case
answer dump (checks only); `--verbose` adds failure detail. Exit code is 0 iff
every case passed every applicable check (so a single protocol can gate a script
too). Full flag list: `.venv/bin/python scripts/eval_research.py --help`.

## The run/diff workflow

1. Run a protocol with `--save-file docs/testing/<date>-<mode>-answers.md`
   (the harness writes every answer + route + latency into that one file).
2. Hand-write a paired `docs/testing/<date>-<mode>-report.md` — the quality
   read: what's accurate, what regressed, the verdict. (See the existing
   `2026-06-26-factcheck-*` pair for the format.)
3. After any prompt/config change, re-run and **diff the new answers file
   against the prior baseline** — the report explains the diff.

The first run of each protocol establishes that mode's **baseline** — the file
every future run diffs against.

## Per-model sweeps (lineup optimization)

Deep-panel lineups are **fixed lists** in `config.yaml` (`deep_panel*.workers`)
— there is no per-request override. So optimizing a lineup means testing the
candidate models **in isolation**, comparing them, editing the workers list,
and confirming with a deep-protocol run. The isolation path is Audrey's
passthrough route: `model: "audrey_passthrough/<concrete>"` proxies straight to
Ollama — no classifier, no banners, no panel, **no tools** (so a `gk-*` case
becomes pure recall: exactly what you want to know about a candidate), and
latency is the model itself.

**Prerequisite (once per model set):** the concrete model name must be in
`passthrough.allowed_models` in `config.yaml` — the registry text-pool models
are already listed. After editing, redeploy on the box
(`docker compose up -d --force-recreate audrey-ai` — config-only, no rebuild)
and confirm the new `audrey_passthrough/<name>` ids show up in OWUI's model
list.

**Run a sweep** (`--models` runs every case once per model, grouped by model so
local models don't thrash GPU loads; ` [<model>]` is auto-suffixed onto case
names):

```bash
.venv/bin/python scripts/eval_research.py \
  --cases scripts/eval_prompts_code_models.json \
  --models 'audrey_passthrough/qwen3-coder-next:latest,audrey_passthrough/kimi-k2.7-code:cloud,audrey_passthrough/deepseek-v4-pro:cloud' \
  --save-file docs/testing/$(date +%F)-code-sweep-answers.md \
  --save-json docs/testing/$(date +%F)-code-sweep-results.json
```

**Build the comparison** (case × model matrix + per-model pass rate / mean
latency / answer length + failure list — the seed for the hand-written report):

```bash
.venv/bin/python scripts/eval_compare.py \
  docs/testing/$(date +%F)-code-sweep-results.json \
  --out docs/testing/$(date +%F)-code-compare.md
```

Feeding several results files merges them into one matrix (e.g. a sweep plus an
`audrey_deep` run of the same cases — the anchors line up by name). Naming
convention: `<date>-<desc>-sweep-answers.md` + `-results.json` + `-compare.md`,
plus the usual hand-written `-report.md` for the quality read.

**The lineup loop:** sweep the candidates (`eval_prompts_code_models.json` for
code; `eval_prompts_topics.json` sweeps too) → read the compare table + answers
→ propose a `deep_panel*.workers` edit in `config.yaml` → redeploy → re-run the
affected deep protocol (`run_all_evals.sh code-hard` or `deep`) and diff against
its baseline. Sweep the **`code-hard`** cases, not the easy ones, when the goal
is separating candidates — the easy set doesn't discriminate. And write
`code_test` asserts that cover the corners (edge cases, input non-mutation, the
raise-on-bad-input contract): a passing easy case can still hide a real defect,
which is exactly what the harder asserts surface. `agentic.debug_panel_drafts:
true` (live-toggle) is the complementary view: it appends every worker's draft
to deep answers, showing how panel members behave *inside* the panel rather than
solo — and the per-worker latency there is itself a lineup signal (a local
worker running 15–40× slower than the cloud ones gates the whole panel).

## Cross-mode anchors

Some prompts appear in more than one protocol on purpose, so a mode-vs-mode diff
is a direct prose comparison on an identical prompt:

- `library-alexandria`, `pythagoras`, `rust-async`, `2025-recent`,
  `birthday-toast` — in **research** and **deep** (quantifies what the
  fact-check stage adds over panel-anchor-only).
- `birthday-toast`, `recursion`, `ambiguous-mercury` — in **deep** and **fast**
  (same prompt, different latency/depth tradeoff).
- `code-lru-cache`, `code-debug-mutable-default` — in **deep** and **code**
  (the deep protocol's routine coding coverage); those two plus
  `code-merge-intervals`, `code-debug-binary-search` — in **code** and the
  **per-model sweep set** `eval_prompts_code_models.json` (panel-vs-solo on an
  identical prompt, and the rows that line up when merging results in
  `eval_compare.py`).

## Adding or editing a case

Cases are plain JSON — no code change needed. Each is one object in the
protocol's array:

```json
{
  "name": "auto-short-fast",          // unique label; --only matches a substring
  "model": "audrey_auto",             // which virtual model to hit
  "prompt": "What time zone is Tokyo in?",
  "expect_banners": false,            // optional; defaults to (model has banners)
  "expect_sources": false,            // optional; defaults to (model == audrey_research)
  "expect_route": "fast"              // optional; only for audrey_auto routing cases
}
```

Coding/knowledge cases add the objective-check fields:

```json
{
  "name": "code-rle",
  "prompt": "Write a Python function rle(s) that ... single complete Python code block ...",
  "code_test": "assert rle(\"aaabcc\") == \"a3b1c2\"\nprint(\"ok\")",
  "code_timeout": 15,                 // optional; seconds for the subprocess
  "expect_code": true,                // optional; implied by code_test
  "answer_contains": ["timsort"]      // optional; all must appear (case-insensitive)
}
```

Only `name` and `prompt` are required (`model` falls back to `--model`). The
`expect_*` fields default sensibly from the model, so a research case needs
none of them. Cases with `code_test` should instruct "a single complete Python
code block using only the standard library" and name the function/class exactly
— the harness extracts the largest ` ```python ` block and runs it against the
asserts. Add a case, re-run the protocol, and it's checked + saved like the
rest. Keep a cross-mode anchor's `name` and `prompt` byte-identical across
protocols so the diff stays clean. Sweep-set cases (`eval_prompts_code_models.json`)
omit `model` entirely — `--models` supplies it.

## Files / layout

```
scripts/
  eval_research.py              # the one harness — auth, streaming, checks, sweep, save
  eval_compare.py               # case × model comparison table from --save-json results
  run_all_evals.sh              # one-command runner: research + deep + fast (code/code-hard/topics opt-in), dated
  eval_prompts.json             # quick 6-case smoke set (the harness default)
  eval_prompts_protocol.json    # research protocol (~10 cases)
  eval_prompts_deep.json        # deep protocol (~18 cases, incl. 2 coding anchors)
  eval_prompts_fast.json        # fast + auto protocol (~12 cases)
  eval_prompts_code.json        # easy/regression coding tier on audrey_deep (~10 cases)
  eval_prompts_code_hard.json   # discriminating coding tier for lineup optimization (~5 cases)
  eval_prompts_code_models.json # compact all-objective sweep set (~6 cases, no model field)
  eval_prompts_topics.json      # reasoning/science/writing/gk domains (~13 cases)
docs/testing/
  README.md                     # this file — how to use the suite
  <date>-<protocol>-answers.md  # machine-written: every answer from one run
  <date>-<protocol>-report.md   # hand-written: the quality read for that run
  <date>-<desc>-results.json    # machine-written: per-case checks/latency (--save-json)
  <date>-<desc>-compare.md      # eval_compare.py: case × model matrix for a sweep
docs/plans/
  PLAN-mode-test-suite.md       # design rationale + case taxonomy
```

The hermetic unit suite is separate — `tests/` + `.venv/bin/pytest tests/ -q`.
It proves the plumbing offline (no live model); this live suite judges the
answers. Use both.
