#!/usr/bin/env python3
"""embed_contention_probe.py — sample KB-embed latency against Ollama residency.

Why this exists: a single hand-timed embed probe is a coin flip. Three identical
back-to-back calls on 2026-07-22 returned 1.01s, 23.82s and 0.08s — a 300x spread
decided entirely by what Ollama happened to be doing at that instant. You cannot
hit enter accurately enough to sample that, and the number you get is whichever
side of the eviction you landed on.

What it measures: `nomic-embed-text` costs ~0.07s when it is resident in VRAM and
tens of seconds when it must be swapped back in. Under `OLLAMA_MAX_LOADED_MODELS=1`
the embedder cannot co-reside with a deep-panel worker model (`qwen3.6:35b`, ~24 GB),
so every `kb_search` that follows a generation pays a full reload — which blows the
30s tool-dispatch ceiling in `tools/dispatch.py` and lands as `kb_search ✅0 ❌1`
with a bare `{"error": "timeout"}` body.

So this samples on a fixed interval and reports the DISTRIBUTION plus what was
resident at each sample. Start it, then run a local deep panel; the time series
covers the whole panel lifecycle instead of one arbitrary instant. The summary
counts how many samples would have exceeded the dispatch ceiling — that count,
not any single reading, is the number to judge a fix by.

USAGE (on the box — needs ollama-net, so exec it inside custom-tools):
    docker exec -i custom-tools python3 - < scripts/embed_contention_probe.py
    docker exec -i custom-tools python3 - 5 300 < scripts/embed_contention_probe.py
                                          ^interval ^duration (seconds)

CAVEAT: under `OLLAMA_MAX_LOADED_MODELS=1` the probe PERTURBS what it measures —
each sample drags the embedder back into VRAM and evicts the worker model, so a
short interval will actively slow the panel you are observing. That is tolerable
for confirming the fault, but read the post-fix run as the clean one.

Read-only: embeds a fixed short string and reads /api/ps. Changes nothing.
"""

from __future__ import annotations

import statistics
import sys
import time
from datetime import datetime

try:
    import httpx
except ImportError:  # pragma: no cover - runtime guard, mirrors kb_score_probe
    print("httpx not installed. Run: uv sync --extra dev", file=sys.stderr)
    raise SystemExit(2) from None

OLLAMA = "http://ollama:11434"
MODEL = "nomic-embed-text"
PROBE_TEXT = "top 10 guard positions in Brazilian jiu-jitsu"

# Matches `react.dispatch_timeout_s` (graph.py default, absent from config.yaml).
# A sample above this is a `kb_search` that would have failed in production.
DISPATCH_CEILING_S = 30.0


def _embed_seconds(client: httpx.Client) -> tuple[float, str | None]:
    """Return (elapsed, error_name). Elapsed is still meaningful on failure."""
    start = time.perf_counter()
    try:
        r = client.post(
            f"{OLLAMA}/api/embed",
            json={"model": MODEL, "input": PROBE_TEXT},
            timeout=180.0,
        )
        r.raise_for_status()
        return time.perf_counter() - start, None
    except httpx.HTTPError as e:
        return time.perf_counter() - start, type(e).__name__


def _resident(client: httpx.Client) -> list[str]:
    """Model names currently loaded, per Ollama's /api/ps (the `ollama ps` API)."""
    try:
        r = client.get(f"{OLLAMA}/api/ps", timeout=10.0)
        r.raise_for_status()
        return [m.get("name", "?") for m in r.json().get("models", [])]
    except httpx.HTTPError:
        return ["<ps unreachable>"]


def main() -> int:
    interval_s = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0

    print(f"sampling {MODEL} every {interval_s:g}s for {duration_s:g}s", flush=True)
    print(f"{'time':>8}  {'embed':>9}  resident", flush=True)

    samples: list[float] = []
    errors = 0
    deadline = time.monotonic() + duration_s
    client = httpx.Client()

    while time.monotonic() < deadline:
        elapsed, err = _embed_seconds(client)
        resident = _resident(client)
        if err is None:
            samples.append(elapsed)
            shown = f"{elapsed:.2f}s"
        else:
            errors += 1
            shown = f"{err}"
        # Flag the samples that would have failed a real dispatch.
        mark = " ⚠" if elapsed >= DISPATCH_CEILING_S else ""
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"{stamp:>8}  {shown:>9}  {', '.join(resident) or '-'}{mark}", flush=True)
        time.sleep(interval_s)

    if not samples:
        print(f"\nno successful samples ({errors} errors)", flush=True)
        return 1

    over = sum(1 for s in samples if s >= DISPATCH_CEILING_S)
    ordered = sorted(samples)
    print(
        f"\nn={len(samples)}  errors={errors}\n"
        f"min={ordered[0]:.2f}s  median={statistics.median(ordered):.2f}s  "
        f"max={ordered[-1]:.2f}s\n"
        f"over {DISPATCH_CEILING_S:g}s dispatch ceiling: {over}/{len(samples)}"
        f" ({100 * over / len(samples):.0f}%)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
