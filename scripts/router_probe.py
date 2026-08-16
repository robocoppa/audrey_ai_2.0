#!/usr/bin/env python3
"""Can this model actually be the router? Parse rate, accuracy and latency.

WHY

`router.model` is the most constrained slot in the config and the least
forgiving:

  - It runs on the HOT PATH of every non-skipped turn, before the Thinking
    banner reaches the user.
  - ⚠️ **It is NOT GPU-gated.** `classify_with_registry` takes no gate
    argument and calls Ollama directly on both the graph and streaming paths,
    so `FairLocalGate` — which serialises deep workers IN-PROCESS — never sees
    it. Under `GPU_CONCURRENCY=1` a large router does not queue behind the
    worker, it EVICTS it. That is why `qwen3:4b` being tiny is load-bearing
    rather than incidental, and why "use the new big model" is the wrong move.
  - Its output must parse. `_parse_router_output` wants a `{...}` block with a
    `task` in {code, reasoning, general, vl} and a numeric `confidence`. A
    small model that chats instead of answering JSON fails EVERY call, and the
    failure is quiet: `classify` falls through to `("general",
    "fallback:general", 0.25)` and the box keeps serving, badly routed.

⚠️ **The confidence number is not cosmetic.** The escalation trigger is
`conf < 0.95` STRICTLY (`agentic.escalation.confidence_ceiling`), so a router
that habitually answers 0.8 sends nearly every turn into a deep panel — three
cloud calls each. A candidate that routes *correctly* but *timidly* is a
budget problem, which is why this probe reports the confidence distribution
and not just accuracy.

HOW IT STAYS HONEST

It imports `router_classify` and `_parse_router_output` from
`audrey.pipeline.classify` rather than reimplementing them. A probe with its
own copy of the prompt or the parser can pass while production fails — the
whole point is to exercise the real path.

⚠️ Accuracy here is against a small hand-labelled set, so treat it as a SMOKE
TEST, not a benchmark: it answers "can this model do the job at all", not
"which candidate is best". A model that parses 100 % and routes 9/12 is worth
trying; one that parses 40 % is disqualified regardless of its accuracy on the
rest.

USAGE

⚠️ The box has no `python3`, and the repo is not bind-mounted into `audrey-ai`
— but that container has Python, the `audrey` package, and a route to Ollama:

  docker cp scripts/router_probe.py audrey-ai:/tmp/rp.py
  docker exec -e MODEL=qwen3:4b audrey-ai python3 /tmp/rp.py

  # compare candidates in one go
  docker exec -e MODEL=qwen3:4b,some-small:3b audrey-ai python3 /tmp/rp.py

  # the comparison that decides whether a SMALLER model is viable at all
  docker exec -e MODEL=qwen3:4b -e BOTH=1 audrey-ai python3 /tmp/rp.py

Env:
  MODEL      comma-separated model tags to probe (required)
  OLLAMA     base URL (default http://ollama:11434)
  ROUNDS     samples per case (default 1; raise to see run-to-run variance)
  TIMEOUT    per-call seconds (default 20, matching `router.timeout_s`)
  BOTH=1     run each candidate twice: free prose AND `format=`-pinned
  FORMAT=1   run the schema-pinned arm only
  NOTHINK=1  ask the model not to reason first — the LATENCY arm

⚠️ **Arms are not order-independent.** The first arm eats the model cold-load,
so its first case can time out for a reason that has nothing to do with the arm.
`qwen3.5:4b` showed exactly that on 2026-08-15: two ReadTimeouts, both on case 1
of the FIRST arm. Re-run with the arms reversed before crediting a difference to
schema pinning.

⚠️ **Production does NOT pin the schema today** — `router_classify` defaults
`response_format=None`, so the pinned arm measures a change you have not made
yet. That is the point: the router's size floor is set by "will it emit clean
JSON unprompted", not by "can it classify", and pinning moves the floor.
⚠️ Pinning is not free — `Standing gotchas` records thinking breaking `format=`
JSON, and a pinned call in `deep_panel` has returned 200 OK with zero bytes for
what looks like that reason. Measure, do not assume it only helps.

Exit status is 1 if any candidate fails the parse-rate floor, so this can gate
a router change.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

# The cases carry their expected task. Kept deliberately small and obvious —
# this is a smoke test, and a big hand-labelled set would invite reading it as
# a benchmark. `vl` is absent on purpose: image turns are pinned to the vl pool
# before the router is consulted (`image_turn` short-circuits classify), so the
# router never has to produce it in practice.
CASES: list[tuple[str, str]] = [
    ("Refactor this function to avoid the nested loop and explain the tradeoff",
     "code"),
    ("Why does my Postgres query plan switch to a seq scan above ~10k rows?",
     "code"),
    ("Write a Python decorator that retries with exponential backoff", "code"),
    ("Walk me through whether it is cheaper to rent or buy given 6% rates",
     "reasoning"),
    ("Compare the tradeoffs between event sourcing and CRUD for an audit trail",
     "reasoning"),
    ("If a train leaves at 3pm going 60mph and another at 4pm going 80mph…",
     "reasoning"),
    ("What is the capital of Australia?", "general"),
    ("Summarise what happened with the Voyager 2 plasma instrument", "general"),
    ("Give me a packing list for three days of winter camping", "general"),
    ("Draft a polite email declining a meeting invitation", "general"),
]

_PARSE_FLOOR = 0.9  # below this the candidate is disqualified outright


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 2


async def probe_model(
    ollama, model: str, rounds: int, timeout_s: float, cfg, schema=None,
    no_thinking: bool = False,
) -> dict:
    from audrey.pipeline.classify import router_classify

    parsed = 0
    correct = 0
    total = 0
    confs: list[float] = []
    latencies: list[float] = []
    failures: list[tuple[str, str]] = []

    for prompt, expected in CASES:
        for _ in range(rounds):
            total += 1
            t0 = time.perf_counter()
            task, conf, body = await router_classify(
                ollama, router_model=model, user_text=prompt,
                timeout_s=timeout_s, cfg=cfg, response_format=schema,
                no_thinking=no_thinking,
            )
            latencies.append(time.perf_counter() - t0)
            if task is None:
                # `body` carries `ollama_error:…` or `parse_error:…` — keep the
                # head, since "it chatted at me" and "it timed out" need
                # opposite responses.
                failures.append((prompt[:40], body[:110]))
                continue
            parsed += 1
            confs.append(conf)
            if task == expected:
                correct += 1
            else:
                failures.append((prompt[:40], f"routed {task}, wanted {expected}"))

    return {
        "model": model + (" [schema-pinned]" if schema else "")
                       + (" [no-think]" if no_thinking else ""),
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "parse_rate": parsed / total if total else 0.0,
        "accuracy": correct / parsed if parsed else 0.0,
        "conf_median": statistics.median(confs) if confs else 0.0,
        "conf_at_or_above_ceiling": sum(1 for c in confs if c >= 0.95),
        "latency_median": statistics.median(latencies) if latencies else 0.0,
        "latency_max": max(latencies) if latencies else 0.0,
        "failures": failures,
    }


def render(r: dict, ceiling: float) -> str:
    out = [f"── {r['model']} ──"]
    verdict = "OK" if r["parse_rate"] >= _PARSE_FLOOR else "DISQUALIFIED"
    out.append(f"  parse rate : {r['parsed']}/{r['total']} "
               f"({100 * r['parse_rate']:.0f}%)  [{verdict}]")
    out.append(f"  accuracy   : {r['correct']}/{r['parsed']} "
               f"({100 * r['accuracy']:.0f}%) of the calls that parsed")
    out.append(f"  latency    : median {r['latency_median']:.2f}s  "
               f"max {r['latency_max']:.2f}s   (router.timeout_s bounds this)")
    out.append(f"  confidence : median {r['conf_median']:.2f}, "
               f"{r['conf_at_or_above_ceiling']}/{r['parsed']} at or above the "
               f"{ceiling} escalation ceiling")
    if r["parsed"] and r["conf_at_or_above_ceiling"] == 0:
        out.append("  ⚠️ NOTHING cleared the ceiling — every routed turn would be")
        out.append("     eligible to escalate into a deep panel. That is a COST")
        out.append("     problem even though routing may be perfectly accurate.")
    if r["parse_rate"] < _PARSE_FLOOR:
        out.append("  ⚠️ Below the parse floor. In production these fall through to")
        out.append("     `fallback:general` at conf 0.25 — served, silently misrouted.")
    for prompt, why in r["failures"][:6]:
        out.append(f"    ✗ {prompt!r} → {why}")
    if len(r["failures"]) > 6:
        out.append(f"    … and {len(r['failures']) - 6} more")
    return "\n".join(out)


async def amain() -> int:
    models = [m.strip() for m in os.environ.get("MODEL", "").split(",") if m.strip()]
    if not models:
        return _fail("set MODEL=<tag>[,<tag>…] — the candidate router model(s)")

    base = os.environ.get("OLLAMA", "http://ollama:11434")
    rounds = int(os.environ.get("ROUNDS", "1"))
    timeout_s = float(os.environ.get("TIMEOUT", "20"))

    try:
        from audrey.config import get_config
        from audrey.models.ollama import OllamaClient
    except ImportError as e:
        return _fail(
            f"cannot import the audrey package ({e}) — run this INSIDE the "
            "audrey-ai container, or from the repo root with the venv active"
        )

    # ⚠️ `get_config()` resolves `AUDREY_CONFIG` relative to CWD and runs the
    # pool validators, so it raises when run from a directory without a
    # config.yaml — which is exactly the case inside the container (`docker cp`
    # puts the script in /tmp). Degrade to the default ceiling rather than
    # refusing to probe: the ceiling only labels the confidence column, and a
    # probe that will not run because it could not find a config file is
    # useless precisely when it is most needed.
    try:
        cfg = get_config()
        ceiling = float(
            ((cfg.raw.get("agentic") or {}).get("escalation") or {})
            .get("confidence_ceiling", 0.95)
        )
    except Exception as e:  # noqa: BLE001 — probe must run without a loadable config
        print(f"note: no config loaded ({type(e).__name__}); using the default "
              f"escalation ceiling of 0.95 and the built-in classifier prompt\n",
              file=sys.stderr)
        cfg, ceiling = None, 0.95

    # BOTH=1 runs each candidate twice — free prose vs `format=`-pinned — which
    # is the comparison that decides whether a smaller model is viable at all.
    both = os.environ.get("BOTH", "").strip() not in ("", "0")
    schema_only = os.environ.get("FORMAT", "").strip() not in ("", "0")
    from audrey.pipeline.classify import ROUTER_SCHEMA
    arms: list = [None]
    if both:
        arms = [None, ROUTER_SCHEMA]
    elif schema_only:
        arms = [ROUTER_SCHEMA]

    # ⚠️ NOTHINK is the arm that matters for latency. `qwen3.5:4b` probed at a
    # 4.8s MEDIAN for a 4-way classification (2026-08-15) — enormous for the hot
    # path — and the qwen3.5 family declares `thinking`. Reasoning is not the
    # product here; the label is.
    nothink = os.environ.get("NOTHINK", "").strip() not in ("", "0")

    ollama = OllamaClient(base_url=base)
    print(f"probing {len(models)} candidate(s) × {len(arms)} arm(s) against "
          f"{len(CASES)} cases × {rounds} round(s) at {base}"
          f"{' [no-think]' if nothink else ''}\n")
    worst_ok = True
    for model in models:
        for schema in arms:
            result = await probe_model(ollama, model, rounds, timeout_s, cfg,
                                       schema, nothink)
            print(render(result, ceiling))
            print()
            worst_ok = worst_ok and result["parse_rate"] >= _PARSE_FLOOR

    print("⚠️ Accuracy here is a SMOKE TEST against ten hand-labelled prompts, "
          "not a benchmark.\n   It answers 'can this model do the job at all'.")
    print("⚠️ The router is NOT GPU-gated — a candidate that passes on quality "
          "can still\n   evict the deep worker under GPU_CONCURRENCY=1. Size is "
          "a separate check.")
    return 0 if worst_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
