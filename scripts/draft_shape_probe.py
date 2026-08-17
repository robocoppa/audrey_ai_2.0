#!/usr/bin/env python3
"""Why does this model's draft come back malformed? Two arms, same model.

WHY

`nemotron-3.5-lightning` returned its `code-hard-lru-ttl` draft as bare,
unfenced source on three consecutive eval runs while fencing every other draft
it produced. The synthesizer repaired it each time, so every eval check passed
and the artifact showed nothing but a slightly odd-looking draft.

The panel could not answer why, because a deep-panel worker is handed a
PLANNER SUBTASK, not the user's question — `_messages_for_subtask` replaces the
last user message outright. The original prompt for that case ends with:

    Reply with a single complete Python code block using only the standard
    library, defining the class exactly as named.

If the planner's decomposition drops that sentence, the model has been told
nothing about fencing and an unfenced answer is CORRECT BEHAVIOUR. The rival
explanation is a token cap truncating the reply, which `done_reason` settles.
This probe separates the two without a panel in the way.

THE ARMS

  full    — the case prompt verbatim, formatting instruction included
  subtask — the same question with the trailing instruction sentence removed,
            standing in for what a planner decomposition looks like

If `full` fences and `subtask` does not, the fencing is a prompt-content
effect and the panel is losing the instruction. If NEITHER fences, it is the
model. If drafts come back `done_reason=length`, it is neither and you are
looking at a truncation.

⚠️ ARM ORDER IS REVERSED HALFWAY (`--flip`, on by default). A first arm eats
the model's cold load and gets its latency charged to the arm; this repo has
credited that to the wrong variable three times. There is also one untimed
warm-up call before anything is recorded.

HOW IT STAYS HONEST

`_strip_think` and `_fence_anomaly` are imported from `audrey.pipeline.
deep_panel` rather than reimplemented — a probe carrying its own copy of the
detector can pass while production fails.

USAGE

  scripts/probe-onbox.sh draft_shape_probe.py MODEL=nemotron-3.5-lightning:latest

  MODEL    required — the model to probe
  CASE     eval case name (default: code-hard-lru-ttl)
  CASES    cases file (default: eval_prompts_code_hard.json)
  N        runs per arm (default: 3)
  FLIP     1 to reverse arm order for the second half (default: 1)

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
    # `/tmp` first: that is where `docker cp` lands it, and this is a read of a
    # file the operator just placed there by hand, not a temp file we create.
    searched = [Path("/tmp") / CASES, Path("/eval") / CASES,  # noqa: S108
                Path("/app/scripts") / CASES, Path(__file__).resolve().parent / CASES]
    for p in searched:
        if p.exists():
            for case in json.loads(p.read_text()):
                if case.get("name") == CASE:
                    return str(case["prompt"])
            raise SystemExit(f"{p} has no case named {CASE!r}")
    raise SystemExit(
        f"{CASES} not found. Looked in: {', '.join(str(p) for p in searched)}\n"
        f"▶ Copy it in first, from the repo on the box:\n"
        f"    docker cp scripts/{CASES} audrey-ai:/tmp/{CASES}\n"
        f"▶ Or pass the text directly:  PROMPT='...'"
    )


def _strip_format_instruction(prompt: str) -> str:
    """Drop the trailing 'Reply with…' sentence — the planner's likely loss.

    Deliberately crude: it removes the LAST sentence beginning with "Reply
    with", which is how these case prompts carry their output-shape rule. If a
    prompt has no such sentence the arms are identical, and the probe says so
    rather than pretending it tested something.
    """
    marker = "Reply with"
    idx = prompt.rfind(marker)
    return prompt[:idx].rstrip() if idx != -1 else prompt


async def _one(client: OllamaClient, prompt: str) -> dict:
    try:
        resp = await client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout_s=360.0,
        )
    except OllamaError as e:
        return {"error": str(e)[:200]}
    raw = (resp.get("message", {}) or {}).get("content", "") or ""
    stripped = _strip_think(raw)
    return {
        "done_reason": str(resp.get("done_reason") or "?"),
        "eval_count": int(resp.get("eval_count", 0) or 0),
        "raw_len": len(raw),
        "content_len": len(stripped),
        "fences": stripped.count("```"),
        "anomaly": _fence_anomaly(stripped) or "none",
        "head": stripped.lstrip()[:60].replace("\n", "\\n"),
    }


async def main() -> int:
    if not MODEL:
        raise SystemExit("MODEL is required")
    full = _load_prompt()
    subtask = _strip_format_instruction(full)
    if subtask == full:
        print("⚠️  prompt has no 'Reply with…' sentence — both arms are IDENTICAL, "
              "so this run cannot separate prompt-content from model behaviour.")

    arms = {"full": full, "subtask": subtask}
    print(f"model={MODEL} case={CASE} n={N} flip={FLIP}")
    print(f"full={len(full)} chars  subtask={len(subtask)} chars  "
          f"(dropped {len(full) - len(subtask)})\n")

    client = OllamaClient(base_url=OLLAMA)
    print("warm-up (untimed, discarded)…")
    await _one(client, "Say OK.")

    results: dict[str, list[dict]] = {"full": [], "subtask": []}
    order = list(arms)
    for i in range(N):
        # Reverse for the back half so neither arm always runs first.
        run_order = list(reversed(order)) if (FLIP and i >= N / 2) else order
        for arm in run_order:
            r = await _one(client, arms[arm])
            results[arm].append(r)
            print(f"  [{i + 1}/{N}] {arm:8} {r}")

    print("\n─── summary ───")
    findings = False
    for arm, rows in results.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            print(f"{arm:8} all {len(rows)} calls errored")
            findings = True
            continue
        anomalies = [r["anomaly"] for r in ok if r["anomaly"] != "none"]
        lengths = [r["done_reason"] for r in ok if r["done_reason"] == "length"]
        print(f"{arm:8} fenced {sum(1 for r in ok if r['fences'] >= 2)}/{len(ok)}  "
              f"anomalies {len(anomalies)}/{len(ok)} {set(anomalies) or ''}  "
              f"done_reason=length {len(lengths)}/{len(ok)}")
        if anomalies or lengths:
            findings = True

    print("\nRead it as: full fences and subtask does not → the panel is losing the "
          "formatting instruction to `_messages_for_subtask`. Neither fences → the "
          "model. done_reason=length anywhere → truncation, and the other two "
          "readings do not apply.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
