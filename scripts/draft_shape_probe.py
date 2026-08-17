#!/usr/bin/env python3
"""Does heavy thinking cost this model its output formatting? Two arms.

WHY

`nemotron-3.5-lightning` returned its `code-hard-lru-ttl` draft as bare,
unfenced source on four consecutive eval runs, while fencing every one of its
fourteen other drafts. The synthesizer repaired it each time, so every eval
check passed and the artifact showed only a slightly odd-looking draft.

⚠️ FOUR EXPLANATIONS ARE ALREADY DEAD, killed by the `deep_panel: draft` log
line on 2026-08-17 — do not re-open them:

  truncation      `done_reason=stop`. The model finished.
  `_strip_think`  `think_stripped=0`. The stripper removed nothing.
  the planner     `subtask=''`. The panel did not split, so the worker saw the
                  original prompt — "Reply with a single complete Python code
                  block" included — and disregarded it.
  the renderer    `_HR_LINE` matches hyphens only; it cannot touch a fence.

What is left is a correlation. On the anomalous call nemotron logged
`eval_count=6172` for `content_len=1053` — `chars_per_tok=0.17` against
0.35–0.48 on its four clean calls in the same run. Roughly 5,900 of those
6,172 tokens produced no text at all. They went to Ollama's separate
`thinking` field, which `OllamaClient.chat` never returns, so the only trace
is the ratio. **The one call that thought hardest is the one that lost its
fence.** That is n=1 on the mechanism, however solid the case is at 4-for-4 —
hence this probe.

THE ARMS

  think-default — exactly what the deep panel sends: no `think` field at all
  think-false   — the same prompt with thinking suppressed

If `think-default` loses the fence and `think-false` keeps it, the mechanism
is thinking, and the fix is a per-role knob rather than a prompt edit. If both
lose it, thinking is a bystander and it is plain model behaviour on this
prompt. If neither loses it, the panel is implicated after all and the next
step is the panel's own message list.

⚠️ THIS IS A PROBE, NOT A LICENCE. `think=false` is measured to be 2.7–7.4×
faster AND to have bought a fabrication — see the standing warning against
adding `think=` to deep-panel calls. Learning the mechanism here does not
authorise turning thinking off in the panel.

⚠️ ARM ORDER REVERSES HALFWAY (`FLIP`, on by default). A first arm eats the
model's cold load and gets it charged to the arm; this repo has credited that
to the wrong variable three times. One untimed warm-up runs before anything is
recorded.

HOW IT STAYS HONEST

`_strip_think` and `_fence_anomaly` are imported from `audrey.pipeline.
deep_panel` rather than reimplemented — a probe carrying its own copy of the
detector can pass while production fails.

USAGE

  scripts/probe-onbox.sh draft_shape_probe.py MODEL=nemotron-3.5-lightning:latest \
      N=5 COPY=eval_prompts_code_hard.json

⚠️ `COPY=` is not optional. The repo is not mounted into `audrey-ai`, so the
cases file is absent without it and the probe exits before calling anything.

  MODEL    required — the model to probe
  CASE     eval case name (default: code-hard-lru-ttl)
  CASES    cases file (default: eval_prompts_code_hard.json)
  N        runs per arm (default: 3)
  FLIP     1 to reverse arm order for the second half (default: 1)

⚠️ Ollama REJECTS `think` for a model without the `thinking` capability rather
than ignoring it, so the probe checks capabilities first and says so instead
of reporting a whole dead arm as a result.

Exit 1 when either arm produced an anomaly — a FINDING, not a failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app/src")

from audrey.models.ollama import OllamaClient, OllamaError
from audrey.pipeline.deep_panel import _fence_anomaly, _strip_think

MODEL = os.environ.get("MODEL", "")
CASE = os.environ.get("CASE", "code-hard-lru-ttl")
CASES = os.environ.get("CASES", "eval_prompts_code_hard.json")
N = int(os.environ.get("N", "3"))
FLIP = os.environ.get("FLIP", "1") == "1"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://ollama:11434")


def _load_prompt() -> str:
    """Find the case prompt, or say exactly how to supply it.

    ⚠️ `probe-onbox.sh` copies the PROBE into `audrey-ai` and nothing else —
    the repo is not bind-mounted there, so the cases file has to arrive under
    its own power. Rather than carry a stale copy of the prompt (which would
    drift from the suite silently and make every result a lie), this looks for
    the real file and, failing that, prints the one command that fixes it.
    """
    if os.environ.get("PROMPT"):
        return os.environ["PROMPT"]
    # `/tmp` first: that is where `probe-onbox.sh COPY=` lands it. Deduped
    # because the probe itself is copied to /tmp, so `__file__`'s parent IS
    # /tmp on the box and the failure message listed the same path twice.
    searched: list[Path] = []
    for p in (Path("/tmp") / CASES, Path("/eval") / CASES,  # noqa: S108
              Path("/app/scripts") / CASES, Path(__file__).resolve().parent / CASES):
        if p not in searched:
            searched.append(p)
    for p in searched:
        if p.exists():
            for case in json.loads(p.read_text()):
                if case.get("name") == CASE:
                    return str(case["prompt"])
            raise SystemExit(f"{p} has no case named {CASE!r}")
    raise SystemExit(
        f"{CASES} not found. Looked in: {', '.join(str(p) for p in searched)}\n"
        f"▶ The wrapper can bring it: add COPY={CASES} to the probe-onbox.sh line.\n"
        f"▶ Or pass the text directly:  PROMPT='...'"
    )


async def _one(client: OllamaClient, prompt: str, think: bool | None) -> dict:
    """One call. `think=None` sends NO `think` field — what the panel does.

    ⚠️ The tri-state matters: `None` is not `False`. `OllamaClient.chat` omits
    the field entirely on `None`, and that omission is the production arm.
    Sending `think=False` is a different request, which is the whole point of
    having two arms.
    """
    try:
        resp = await client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout_s=360.0,
            think=think,
        )
    except OllamaError as e:
        return {"error": str(e)[:200]}
    raw = (resp.get("message", {}) or {}).get("content", "") or ""
    stripped = _strip_think(raw)
    evals = int(resp.get("eval_count", 0) or 0)
    return {
        "done_reason": str(resp.get("done_reason") or "?"),
        "eval_count": evals,
        "raw_len": len(raw),
        "content_len": len(stripped),
        # The signal that diagnosed this in the first place. Crude and only
        # ever a comparator between arms of the SAME model — see the note on
        # `_log_draft_shape`.
        "chars_per_tok": round(len(stripped) / evals, 2) if evals else 0.0,
        "fences": stripped.count("```"),
        "anomaly": _fence_anomaly(stripped) or "none",
        "head": stripped.lstrip()[:60].replace("\n", "\\n"),
    }


async def _thinking_capable(client: OllamaClient) -> bool:
    """⚠️ Ollama REJECTS `think` for a model that lacks the capability.

    Rather than ignoring it — so an unguarded `think=False` arm errors on every
    call and reports as a finding when it is really a mis-run probe.
    """
    try:
        caps = await client.capabilities(MODEL)
    except Exception as e:  # noqa: BLE001 — a probe must not die on a capability read
        print(f"⚠️  could not read capabilities for {MODEL}: {e}")
        return False
    return "thinking" in {str(c).lower() for c in (caps or [])}


async def main() -> int:
    if not MODEL:
        raise SystemExit("MODEL is required")
    prompt = _load_prompt()
    client = OllamaClient(base_url=OLLAMA)

    # `None` = send no `think` field, which is exactly what the deep panel does.
    arms: dict[str, bool | None] = {"think-default": None}
    if await _thinking_capable(client):
        arms["think-false"] = False
    else:
        print(f"⚠️  {MODEL} reports no `thinking` capability — running the "
              "default arm ONLY. Ollama would reject `think=false` outright, so "
              "a second arm here would be all errors, not a result.")

    print(f"model={MODEL} case={CASE} n={N} flip={FLIP} arms={list(arms)}")
    print(f"prompt={len(prompt)} chars\n")

    print("warm-up (untimed, discarded)…")
    await _one(client, "Say OK.", None)

    results: dict[str, list[dict]] = {arm: [] for arm in arms}
    order = list(arms)
    for i in range(N):
        # Reverse for the back half so neither arm always runs first.
        run_order = list(reversed(order)) if (FLIP and i >= N / 2) else order
        for arm in run_order:
            r = await _one(client, prompt, arms[arm])
            results[arm].append(r)
            print(f"  [{i + 1}/{N}] {arm:14} {r}")

    print("\n─── summary ───")
    findings = False
    for arm, rows in results.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            print(f"{arm:14} all {len(rows)} calls errored: {rows[0].get('error', '')[:120]}")
            findings = True
            continue
        anomalies = [r["anomaly"] for r in ok if r["anomaly"] != "none"]
        lengths = [r for r in ok if r["done_reason"] == "length"]
        ratios = [r["chars_per_tok"] for r in ok]
        print(f"{arm:14} fenced {sum(1 for r in ok if r['fences'] >= 2)}/{len(ok)}  "
              f"anomalies {len(anomalies)}/{len(ok)} {set(anomalies) or ''}  "
              f"done=length {len(lengths)}/{len(ok)}  "
              f"chars_per_tok {min(ratios):.2f}–{max(ratios):.2f}")
        if anomalies or lengths:
            findings = True

    print(
        "\nRead it as:\n"
        "  default loses the fence, think-false keeps it → THINKING is the\n"
        "    mechanism. The fix is a per-role knob, not a prompt edit — and\n"
        "    note the standing warning: `think=false` bought a fabrication.\n"
        "  BOTH lose it → thinking is a bystander; plain model behaviour on\n"
        "    this prompt, and a lineup or prompt decision rather than a bug.\n"
        "  NEITHER loses it → the panel is implicated after all, and the next\n"
        "    place to look is the message list it actually sent.\n"
        "⚠️ Truncation, `_strip_think`, the planner subtask and the renderer are\n"
        "   ALL already eliminated (2026-08-17). Do not re-derive them."
    )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
