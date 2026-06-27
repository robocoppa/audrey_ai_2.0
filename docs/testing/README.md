# Audrey mode test suite

Live, repeatable accuracy/quality evals for each Audrey virtual model. The
hermetic suite (`tests/`) proves the plumbing; these put a real model on the
other end and let you read the actual answers. One harness
(`scripts/eval_research.py`) drives all of them — the cases differ, the script
doesn't.

See `PLAN-mode-test-suite.md` for the design rationale and case taxonomy.

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

## Running

Credentials live in `.env.test.local` (gitignored, laptop-local — see the
script docstring). You must be on the LAN/VPN to reach the box.

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

Run one case with `--only <substring>`. Exit code is 0 iff every case passed
every applicable check (so it can gate a script).

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
