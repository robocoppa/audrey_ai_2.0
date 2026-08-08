"""Does `think` actually change anything on a text model?

`scripts/thinking_audit.py` says which models *declare* the thinking
capability. That is a precondition, not an answer: **`qwen3-vl:32b` declares
it and ignores it** — measured 2026-08-04, neither `think: false` nor Qwen3's
`/no_think` prompt switch moved the reasoning at all (93-101% of baseline
either way). Deciding a per-role thinking policy from the capability list
alone would be deciding it from a flag that may do nothing.

This runs the three states side by side on one model and prints what came
back, so the question is settled by measurement per model:

    omitted    no `think` field in the request — today's behaviour everywhere
               except the vision path. This is the baseline: whatever the
               model's template does when nobody chooses.
    true       thinking explicitly on.
    false      thinking explicitly off.

## What to look at

**`thinking` chars** is the direct signal. Ollama returns reasoning separately
in `message.thinking`, so this is not inferred from a token ratio — a mistake
phase 38 made twice, at the cost of a deploy each time.

**`eval` tokens** is what you are billed, in wall clock locally and in credits
on a cloud model. Content and thinking together should account for it; if they
do not, there is a third thing happening and the accounting is worth chasing
before drawing conclusions.

**`content` chars** guards the trade. Thinking that is switched off and takes
the answer quality with it is not a saving, and no token column will say so —
the full replies are printed at the end for exactly that reason. Read them.

A verdict line reports whether `true` and `false` actually differ, and the
`omitted`-vs-`true` line compares the gap against the **within-state spread**
rather than against a percentage of the mean. That correction matters: on
2026-08-06 it called 14980c vs 19405c "they differ" while `omitted` alone
ranged 8544c to 18778c across three samples. At SAMPLES=3 the spread inside one
state routinely exceeds the gap between states.

## `TOOLS=1` — the mode that matches the fast path

⚠️ **The default prompt measures prose, and the fast path's job is tool
calling.** The 2026-08-06 runs concluded `think=false` was a 5.7x latency win
at equal quality, which was true for prose and said nothing about whether the
model still picks the right tool. `TOOLS=1` ships the real tool definitions
and reports **which tools each run called** — a state that stops calling them,
or swaps to a weaker one, is the regression that matters and it is invisible
in the content-length column.

## Running it

Needs Ollama, so it runs on the box. `audrey-ai` reaches it over `ollama-net`;
fed on stdin, so no rebuild is needed:

    # Unraid box, from /mnt/user/appdata/audrey_ai_2.0
    docker exec -i -e MODEL=qwen3.6:35b audrey-ai python3 - < scripts/thinking_probe.py

**Run it on a prompt like the role you are deciding about.** Reasoning scales
with how much work the question is, so a probe on "what is 2+2" says nothing
about a deep-panel worker. `PROMPT` overrides the default, which is a
deliberately reason-y question.

**A cloud model costs credits to probe.** Three states x three samples is nine
calls. That is cheap next to a wrong standing policy, but it is not free — and
`kimi-k3:cloud` is on the do-not-re-propose list for exactly this kind of
spend creep.

Environment:

    MODEL        required in practice (default qwen3.6:35b)
    OLLAMA_HOST  default http://ollama:11434
    PROMPT       the question to ask (default: a multi-step reasoning one)
    SAMPLES      runs per state (default 3) — reasoning length is noisy
    STATES       comma-separated subset of omitted,true,false
    NUM_PREDICT  default 2048
    TEMPERATURE  default 0.3
    TIMEOUT_S    default 240
    EXCERPT      chars of each reply to print (default 500, 0 = off)
    THINKING     1 to also print the reasoning text itself (long)
    TOOLS        1 to send real tool definitions and report tool choice
    EXPECTED_TOOL  tool a correct first move would use (TOOLS mode only;
                 default get_file_text, empty = just report the names)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("MODEL", "qwen3.6:35b")
HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
SAMPLES = int(os.environ.get("SAMPLES", "3"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "240"))
EXCERPT = int(os.environ.get("EXCERPT", "500"))
SHOW_THINKING = os.environ.get("THINKING", "") == "1"

# Reason-y on purpose. A trivial prompt produces no reasoning on any model, so
# every state would look identical and the probe would conclude "the flag does
# nothing" about a model where it does plenty.
DEFAULT_PROMPT = (
    "A team ships a feature behind a flag. Week one, 5% of users see it and "
    "conversion is up 8%. Week two they raise it to 50% and conversion is up "
    "1%. Nothing else changed. Give the two most likely explanations, say "
    "which is more likely and why, and name the one measurement that would "
    "tell them apart."
)
#: `TOOLS=1` swaps the analytical prompt for a tool-calling one and ships tool
#: definitions with the request.
#:
#: ⚠️ **This exists because the first two probe runs measured the wrong thing.**
#: 2026-08-06 measured `qwen3.6:35b` and `glm-5.2:cloud` on a plain analytical
#: prompt and concluded `think=false` was a 5.7x latency win at equal quality —
#: which it was, for prose. But the fast path's actual job is **tool calling**
#: (`fast_path` → `react`), and a prose prompt says nothing about whether a
#: model still picks the right tool with reasoning off. A policy set from those
#: runs would have been set from a benchmark that does not resemble the work.
#:
#: The metric here is therefore **which tools were called**, not how good the
#: prose is. A model that stops calling tools, or reaches for the wrong one, is
#: the failure that matters — and it is invisible in a content-length column.
WITH_TOOLS = os.environ.get("TOOLS", "") == "1"

#: Mirrors the three tools a fast-path turn is actually offered, trimmed to the
#: fields that affect selection. Names and descriptions are copied from
#: `tools-server/app.py` so the model faces the same choice it faces in
#: production; a paraphrase here would measure a tool set that does not exist.
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "list_my_files",
            "description": (
                "List the files this user has uploaded — filename, kind, upload "
                "time, processing status, and for a processed video its duration. "
                "This is a catalogue, not contents: it tells you what exists and "
                "what can be read, never what a file says."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_text",
            "description": (
                "Read a file's transcript, visual description or summary, by its "
                "exact filename as returned by list_my_files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Exact filename."},
                    "artifact": {
                        "type": "string",
                        "description": "'transcript', 'visual' or 'summary'.",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": (
                "Semantic search across this user's knowledge base. Optionally "
                "scoped to one file by filename."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filename": {"type": "string", "description": "Optional scope."},
                },
                "required": ["query"],
            },
        },
    },
]

#: Unambiguous on purpose. "What did they say about X in Y.mp4" can defensibly
#: start with either `kb_search` or `get_file_text`, and grading an ambiguous
#: choice is how a probe starts reporting its author's preference. This one has
#: a right answer: the contents of a named file cannot come from a catalogue.
DEFAULT_TOOL_PROMPT = (
    "Read the transcript of jasonRetirement.mp4 and tell me what the speaker "
    "said about his father."
)

PROMPT = os.environ.get(
    "PROMPT", DEFAULT_TOOL_PROMPT if WITH_TOOLS else DEFAULT_PROMPT,
)
#: Optional. Names the tool a correct first move would use, so the summary can
#: count matches instead of leaving every run to be eyeballed. Empty means
#: report the names and judge them yourself — the safer default.
EXPECTED_TOOL = os.environ.get("EXPECTED_TOOL", "get_file_text" if WITH_TOOLS else "")
STATES = [s.strip() for s in os.environ.get("STATES", "omitted,true,false").split(",") if s.strip()]


class Result:
    __slots__ = ("content", "error", "eval_count", "state", "thinking", "tools", "wall_s")

    def __init__(self, state: str) -> None:
        self.state = state
        self.content = ""
        self.thinking = ""
        self.eval_count = 0
        self.wall_s = 0.0
        self.error = ""
        #: Tool names this run asked for, in order. Empty means it answered
        #: without calling anything — which, for a question about the contents
        #: of a named file, is the failure this mode exists to catch.
        self.tools: list[str] = []


def _call(state: str) -> Result:
    out = Result(state)
    payload: dict[str, object] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "options": {"num_predict": NUM_PREDICT, "temperature": TEMPERATURE},
    }
    if WITH_TOOLS:
        payload["tools"] = TOOL_DEFS
    # `omitted` sends no field at all. That is the state every non-vision path
    # is in today, and it is not the same as `false` — Ollama rejects the field
    # outright for a model that cannot think, so "no field" is also the only
    # universally safe request.
    if state == "true":
        payload["think"] = True
    elif state == "false":
        payload["think"] = False

    req = urllib.request.Request(  # noqa: S310 - fixed http(s) host from env
        f"{HOST}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        out.error = f"HTTP {e.code}: {detail}"
        out.wall_s = time.perf_counter() - t0
        return out
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        out.error = f"{type(e).__name__}: {e}"
        out.wall_s = time.perf_counter() - t0
        return out

    out.wall_s = time.perf_counter() - t0
    message = body.get("message") or {}
    out.content = str(message.get("content") or "")
    out.thinking = str(message.get("thinking") or "")
    out.eval_count = int(body.get("eval_count") or 0)
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        out.tools.append(str(fn.get("name") or "?"))
    return out


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def main() -> int:
    print(f"model:   {MODEL}")
    print(f"ollama:  {HOST}")
    print(f"samples: {SAMPLES} per state   num_predict={NUM_PREDICT} temp={TEMPERATURE}")
    print(f"prompt:  {PROMPT[:100]}{'…' if len(PROMPT) > 100 else ''}\n")

    runs: dict[str, list[Result]] = {}
    for state in STATES:
        runs[state] = []
        for i in range(SAMPLES):
            res = _call(state)
            runs[state].append(res)
            flag = "!" if res.error else " "
            tools = ("  tools=" + (",".join(res.tools) or "NONE")) if WITH_TOOLS else ""
            print(f"  {flag} {state:<8} run {i + 1}/{SAMPLES}  "
                  f"{res.wall_s:6.1f}s  think={len(res.thinking):>6}c  "
                  f"content={len(res.content):>6}c  eval={res.eval_count:>6}"
                  + tools
                  + (f"  {res.error}" if res.error else ""))
    print()

    # A state that errored on every sample is a finding, not a gap: Ollama
    # rejecting `think` is exactly how a non-thinking model announces itself.
    print(f"{'state':<10}{'wall s':>9}{'think c':>10}{'content c':>11}{'eval tok':>10}   note")
    stats: dict[str, tuple[float, float, float, float]] = {}
    for state in STATES:
        ok = [r for r in runs[state] if not r.error]
        if not ok:
            first = runs[state][0].error if runs[state] else "no runs"
            print(f"{state:<10}{'—':>9}{'—':>10}{'—':>11}{'—':>10}   ALL FAILED: {first[:60]}")
            continue
        row = (
            _mean([r.wall_s for r in ok]),
            _mean([float(len(r.thinking)) for r in ok]),
            _mean([float(len(r.content)) for r in ok]),
            _mean([float(r.eval_count) for r in ok]),
        )
        stats[state] = row
        note = "" if len(ok) == len(runs[state]) else f"{len(runs[state]) - len(ok)} failed"
        print(f"{state:<10}{row[0]:>9.1f}{row[1]:>10.0f}{row[2]:>11.0f}{row[3]:>10.0f}   {note}")
    print()

    print("── verdict " + "─" * 58)
    if "true" in stats and "false" in stats:
        t_think, f_think = stats["true"][1], stats["false"][1]
        if t_think < 1 and f_think < 1:
            print("   No reasoning text in EITHER state — this model does not think on this")
            print("   prompt, so the flag is moot here. Re-run with a harder PROMPT before")
            print("   concluding anything about the model.")
        else:
            ratio = (f_think / t_think) if t_think else float("inf")
            print(f"   think=false produced {ratio:.0%} of the reasoning that think=true did.")
            if 0.8 <= ratio <= 1.2:
                print("   → WITHIN NOISE. The flag is NOT doing anything on this model.")
                print("     Same result as qwen3-vl:32b. Treat reasoning as a fixed cost")
                print("     here, and do not ship a config line that pretends otherwise.")
            elif ratio < 0.8:
                print("   → The flag IS honoured. A per-role policy is worth setting.")
                print("     Now check the content column: reasoning removed is only a saving")
                print("     if the answer survived it. Read the replies below.")
            else:
                print("   → false produced MORE reasoning than true. That is not a thing;")
                print("     re-run with more SAMPLES before believing it.")
    else:
        print("   Need both `true` and `false` to compare. Check STATES and the errors above.")

    if "omitted" in stats and "true" in stats:
        o_think, t_think = stats["omitted"][1], stats["true"][1]
        # ⚠️ Compare the gap against the SPREAD, not against a percentage of
        # the mean. On 2026-08-06 this line called 14980c vs 19405c "they
        # differ" — while `omitted` alone ranged 8544c to 18778c across its
        # three samples. The spread inside one state was wider than the gap
        # between states, so the distinction was noise being reported as a
        # finding. At SAMPLES=3 that is the norm, not the exception.
        spans = []
        for st in ("omitted", "true"):
            lens = [len(r.thinking) for r in runs.get(st, []) if not r.error]
            spans.append(max(lens) - min(lens) if len(lens) > 1 else 0)
        widest = max(spans) if spans else 0
        gap = abs(o_think - t_think)
        print(f"\n   Default (no field) reasoning is {o_think:.0f}c against {t_think:.0f}c for"
              f" think=true — a gap of {gap:.0f}c.")
        if gap <= widest:
            print(f"   Within-state spread is {widest:.0f}c, WIDER than the gap. Read these as")
            print("   the same: omitting the field thinks. Do not build on the difference.")
        else:
            print(f"   Within-state spread is {widest:.0f}c, narrower than the gap — so they")
            print("   do look different. Confirm with more SAMPLES before acting on it.")
        print("   Either way, omitting the field is NOT `false`: that is what every")
        print("   non-vision path in Audrey is doing right now.")

    if WITH_TOOLS:
        print()
        print("── tool choice " + "─" * 54)
        print("   The column that matters in this mode. Prose quality is not what the")
        print("   fast path buys with reasoning — tool selection is.")
        print()
        for state in STATES:
            rs = [r for r in runs.get(state, []) if not r.error]
            called = sum(1 for r in rs if r.tools)
            picks: dict[str, int] = {}
            for r in rs:
                picks[",".join(r.tools) or "NONE"] = picks.get(",".join(r.tools) or "NONE", 0) + 1
            detail = "  ".join(f"{k}×{v}" for k, v in sorted(picks.items()))
            line = f"   {state:<9} called a tool in {called}/{len(rs)} runs   {detail}"
            if EXPECTED_TOOL:
                hit = sum(1 for r in rs if EXPECTED_TOOL in r.tools)
                line += f"   [{EXPECTED_TOOL}: {hit}/{len(rs)}]"
            print(line)
        print()
        print("   A state that stops calling tools, or swaps to a weaker one, is the")
        print("   regression this mode exists to catch — and it is invisible in the")
        print("   content-length column the other mode reports.")
    print()

    if EXCERPT:
        print("── replies " + "─" * 58)
        for state in STATES:
            for i, r in enumerate(runs[state]):
                if r.error:
                    continue
                print(f"\n[{state} #{i + 1}] content ({len(r.content)}c):")
                print("   " + r.content[:EXCERPT].replace("\n", "\n   "))
                if SHOW_THINKING and r.thinking:
                    print(f"\n[{state} #{i + 1}] thinking ({len(r.thinking)}c):")
                    print("   " + r.thinking[:EXCERPT].replace("\n", "\n   "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
