"""Streaming progress banners.

Renders one blockquote line per pipeline phase, growing in place as work
progresses:

    > _Thinking_
    > _Thinking._
    > _Thinking_ ✅
    > _Dispatching panel_
    > _Dispatching panel.._  ✅ kimi-k2.6:cloud
    > _Dispatching panel..._  ✅ kimi-k2.6:cloud  ❌ qwen3.6:35b
    > _Synthesizing_
    > _Synthesizing_ ✅

The `PhaseTicker` is the only public surface. Used as an async context
manager from the route handler:

    async with PhaseTicker("Thinking", emitter) as ticker:
        await do_work()

The ticker spawns one coroutine that pushes a "." into the emitter's
queue every 5s. On context exit, it cancels the tick task and pushes
the closing checkmark + newline. The emitter (route handler) drains
the queue and yields SSE frames in its own time — slow consumers
backpressure the ticker, not the model work.

Design constraints:
- The model side never waits on banner emission. The queue is bounded
  (maxsize=64); if the user's network is slow, dots stutter; workers
  never stall.
- Tick interval is fixed at 5s — short enough to feel alive, long
  enough to avoid frame spam on a 30s deep request.
- All formatting lives here so the user-visible phrasing is one-file.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

TICK_INTERVAL_S: float = 5.0
QUEUE_MAXSIZE: int = 64


# ─── Headers (one per phase) ──────────────────────────────────────────

BANNER_THINKING = "> _Thinking_"
BANNER_PLANNING = "> _Planning_"
BANNER_DISPATCHING = "> _Dispatching panel_"
BANNER_SYNTHESIZING = "> _Synthesizing_"

# Separator emitted between the last banner and the actual answer body.
# Two newlines on each side so the markdown renderer treats it as a
# horizontal rule rather than concatenating with the blockquote above.
BANNER_SEPARATOR = "\n\n---\n\n"


# ─── Inline tail fragments (no leading newline; appended in place) ────

def worker_ok(model: str) -> str:
    """Successful worker → '  ✅ qwen3.6:35b'."""
    return f"  ✅ {model}"


def worker_fail(model: str) -> str:
    """Failed worker → '  ❌ qwen3.6:35b'."""
    return f"  ❌ {model}"


# ─── Tool-usage footer ────────────────────────────────────────────────

def _format_calls(calls: list[dict]) -> str:
    """`[{name, is_error}, ...]` → `kb_search ×2, web_search ×1 ❌`.

    Counts duplicate names; appends ❌ to a tool that had any error in this
    worker's run. Order is first-seen so the footer stays stable across runs.
    """
    seen: list[str] = []
    counts: dict[str, int] = {}
    errors: dict[str, bool] = {}
    for c in calls:
        name = str(c.get("name") or "?")
        if name not in counts:
            seen.append(name)
            counts[name] = 0
            errors[name] = False
        counts[name] += 1
        if c.get("is_error"):
            errors[name] = True
    parts = []
    for name in seen:
        n = counts[name]
        suffix = " ❌" if errors[name] else ""
        if n == 1:
            parts.append(f"`{name}`{suffix}")
        else:
            parts.append(f"`{name}` ×{n}{suffix}")
    return ", ".join(parts)


def tool_summary_block(
    per_worker: list[tuple[str, list[dict]]],
) -> str:
    """Render a per-worker tools-used footer, or empty string if no tools fired.

    Input: list of `(model_name, tool_calls)` pairs — same shape across deep
    panel (one entry per WorkerDraft) and fast path (one entry total).

    Output (when at least one worker called at least one tool):

        \\n\\n---\\n> _Tools used:_\\n> - **qwen3.6:35b** — `kb_search` ×2\\n> - …

    The two leading newlines + horizontal rule give the markdown renderer a
    clean break from the answer body. Skipped silently when nothing to show.
    """
    rows = [
        (model, calls)
        for model, calls in per_worker
        if calls
    ]
    if not rows:
        return ""
    lines = ["", "", "---", "> _Tools used:_"]
    for model, calls in rows:
        formatted = _format_calls(calls)
        lines.append(f"> - **{model}** — {formatted}")
    return "\n".join(lines) + "\n"


# Type alias — async function that takes a string and yields it as an
# SSE frame. The route handler supplies this; the ticker doesn't know
# (or care) what the frame format is.
Emitter = Callable[[str], Awaitable[None]]


class PhaseTicker:
    """Owns the dot-tick lifecycle for one phase.

    Usage:
        async with PhaseTicker("Thinking", emit) as ticker:
            await do_phase_work()
            ticker.append_tail(" ✓ extra")  # optional inline result

    Lifecycle:
      __aenter__:  emit header                ('> _Thinking_')
      tick task:   emit '.' every TICK_INTERVAL_S
      append_tail: emit a fragment NOW (interleaves with dots)
      __aexit__:   cancel ticker, emit ' ✅\\n' (or ' ❌\\n' on error) —
                   the mark sits outside the header's italics
    """

    def __init__(
        self,
        header: str,
        emit: Emitter,
        *,
        tick_interval_s: float = TICK_INTERVAL_S,
    ) -> None:
        self._header = header
        self._emit = emit
        self._tick_interval_s = tick_interval_s
        self._tick_task: asyncio.Task[None] | None = None
        # The tail queue is for fragments produced mid-phase (e.g. per-worker
        # check/cross marks). Bounded so a slow consumer backpressures the
        # producer, never the model work.
        self._tail_q: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._tail_drainer: asyncio.Task[None] | None = None

    async def __aenter__(self) -> PhaseTicker:
        await self._emit(self._header)
        self._tick_task = asyncio.create_task(self._tick_loop(), name="banner-tick")
        self._tail_drainer = asyncio.create_task(self._drain_tail(), name="banner-tail")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Stop the tick first so we don't race a final dot in after the
        # checkmark.
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        # Drain whatever tail fragments are still queued before the
        # closing line. Sentinel None tells the drainer to stop.
        await self._tail_q.put(None)  # type: ignore[arg-type]
        if self._tail_drainer is not None:
            try:
                await self._tail_drainer
            except asyncio.CancelledError:
                pass
        # Closing mark. ❌ on exception so the user sees the phase failed.
        closing = " ❌\n" if exc_type is not None else " ✅\n"
        await self._emit(closing)
        # Falling off the end returns None (falsy), so __aexit__ does not
        # suppress — any in-flight exception propagates to the caller.

    def append_tail(self, fragment: str) -> None:
        """Push a fragment to be emitted between dots.

        Non-blocking. If the queue is full (slow consumer), drops the
        fragment with a warning rather than backpressuring the caller.
        For per-worker results, dropping is preferable to stalling the
        panel; the user just sees fewer checkmarks during transient
        congestion.
        """
        try:
            self._tail_q.put_nowait(fragment)
        except asyncio.QueueFull:
            log.warning("banner: tail queue full, dropping fragment: %r", fragment)

    async def _tick_loop(self) -> None:
        # Runs as its own task. __aexit__ cancels it; the resulting
        # CancelledError propagates and ends the loop — no handler needed.
        while True:
            await asyncio.sleep(self._tick_interval_s)
            # Awaiting the emitter blocks THIS tick task only on a slow
            # consumer, never the caller doing the model work — that's why
            # the model side never stalls on banner emission.
            await self._emit(".")

    async def _drain_tail(self) -> None:
        while True:
            item = await self._tail_q.get()
            if item is None:
                return
            await self._emit(item)


__all__ = [
    "BANNER_THINKING",
    "BANNER_PLANNING",
    "BANNER_DISPATCHING",
    "BANNER_SYNTHESIZING",
    "BANNER_SEPARATOR",
    "Emitter",
    "PhaseTicker",
    "TICK_INTERVAL_S",
    "tool_summary_block",
    "worker_fail",
    "worker_ok",
]
