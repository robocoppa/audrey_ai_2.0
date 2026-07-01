# Audrey mode test suite

Live, repeatable accuracy/quality evals for each Audrey virtual model. The
hermetic suite (`tests/`) proves the plumbing; these put a real model on the
other end and let you read the actual answers. One harness
(`scripts/eval_research.py`) drives all of them — the cases differ, the script
doesn't.

See `../plans/PLAN-mode-test-suite.md` for the design rationale and case taxonomy.

## The three protocols

| Protocol | Cases file | Default model | What it tests |
|----------|-----------|---------------|---------------|
| **Research** | `scripts/eval_prompts_protocol.json` | `audrey_research` | Staged research→verify→fact-check→write; grounding, citation, hedge-vs-drop discipline |
| **Deep** | `scripts/eval_prompts_deep.json` | `audrey_deep` | Panel synthesis quality across a wide topic range; flourish-leak failure mode; merge coherence |
| **Fast** | `scripts/eval_prompts_fast.json` | `audrey_fast` / `audrey_auto` | Answer quality on simple turns; **routing correctness** (fast/deep gate); **latency/TTFT** |

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

This chains all three protocols and writes today's dated answers files into
`docs/testing/` (`<date>-research-answers.md`, `-deep-`, `-fast-`). It does the
same preflight checks (venv, `.env.test.local`, valid protocol) and **exits
non-zero if any case in any protocol failed a check**, so it can gate CI / a
pre-deploy check. The full suite is **slow (~60–90 min, ~34 cases)** — run it in
the background and wait; don't foreground-block on it.

```bash
scripts/run_all_evals.sh research deep      # only the named protocols, in order
scripts/run_all_evals.sh fast               # just one
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

## Cross-mode anchors

Some prompts appear in more than one protocol on purpose, so a mode-vs-mode diff
is a direct prose comparison on an identical prompt:

- `library-alexandria`, `pythagoras`, `rust-async`, `2025-recent`,
  `birthday-toast` — in **research** and **deep** (quantifies what the
  fact-check stage adds over panel-anchor-only).
- `birthday-toast`, `recursion`, `ambiguous-mercury` — in **deep** and **fast**
  (same prompt, different latency/depth tradeoff).

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

Only `name`, `model`, and `prompt` are required. The `expect_*` fields default
sensibly from the model, so a research case needs none of them. Add a case,
re-run the protocol, and it's checked + saved like the rest. Keep a cross-mode
anchor's `name` and `prompt` byte-identical across protocols so the diff stays
clean.

## Files / layout

```
scripts/
  eval_research.py              # the one harness — auth, streaming, checks, save
  run_all_evals.sh              # one-command runner: research + deep + fast, dated
  eval_prompts.json             # quick 6-case smoke set (the harness default)
  eval_prompts_protocol.json    # research protocol (~10 cases)
  eval_prompts_deep.json        # deep protocol (~12 cases)
  eval_prompts_fast.json        # fast + auto protocol (~12 cases)
docs/testing/
  README.md                     # this file — how to use the suite
  <date>-<protocol>-answers.md  # machine-written: every answer from one run
  <date>-<protocol>-report.md   # hand-written: the quality read for that run
docs/plans/
  PLAN-mode-test-suite.md       # design rationale + case taxonomy
```

The hermetic unit suite is separate — `tests/` + `.venv/bin/pytest tests/ -q`.
It proves the plumbing offline (no live model); this live suite judges the
answers. Use both.
