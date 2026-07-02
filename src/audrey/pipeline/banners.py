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
import re
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

TICK_INTERVAL_S: float = 5.0
QUEUE_MAXSIZE: int = 64


# ─── Headers (one per phase) ──────────────────────────────────────────

BANNER_THINKING = "> _Thinking_"
BANNER_PLANNING = "> _Planning_"
BANNER_DISPATCHING = "> _Dispatching panel_"
BANNER_SYNTHESIZING = "> _Synthesizing_"

# audrey_research staged-pipeline phase banners
# (research → verify → fact-check → write).
BANNER_RESEARCHING = "> _Researching_"
BANNER_VERIFYING = "> _Verifying_"
BANNER_FACTCHECKING = "> _Fact-checking_"
BANNER_WRITING = "> _Writing_"

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
    """`[{name, is_error}, ...]` → `kb_search ✅2, web_search ✅2 ❌1`.

    Counts duplicate names and splits them into successes and failures:
      - ✅ always shows the number that succeeded;
      - ❌ shows the number that failed, and appears only when at least one did.
    So a row reads as plain counts — `✅10 ❌3` is "10 ok, 3 failed", `✅13` is
    "all 13 ok", `✅0 ❌13` is "all 13 failed". Nothing is inferred from the
    presence or absence of a number. An empty-but-OK result (e.g. a search
    returning 0 hits) is a success, not a failure, so it counts under ✅. Order
    is first-seen so the footer stays stable across runs.
    """
    seen: list[str] = []
    counts: dict[str, int] = {}
    errs: dict[str, int] = {}
    for c in calls:
        name = str(c.get("name") or "?")
        if name not in counts:
            seen.append(name)
            counts[name] = 0
            errs[name] = 0
        counts[name] += 1
        if c.get("is_error"):
            errs[name] += 1
    parts = []
    for name in seen:
        n_err = errs[name]
        n_ok = counts[name] - n_err
        mark = f" ✅{n_ok}" + (f" ❌{n_err}" if n_err else "")
        parts.append(f"`{name}`{mark}")
    return ", ".join(parts)


def tool_summary_block(
    per_worker: list[tuple[str, list[dict]]],
) -> str:
    """Render a per-worker tools-used footer, or empty string if no tools fired.

    Input: list of `(model_name, tool_calls)` pairs — same shape across deep
    panel (one entry per WorkerDraft) and fast path (one entry total).

    Output (when at least one worker called at least one tool):

        \\n\\n---\\n> _Tools used:_\\n> - **qwen3.6:35b** — `kb_search` ×2\\n> - …

    Notation in each row (see `_format_calls`): `✅N` = N calls succeeded;
    `❌k` = k calls failed, shown only when k > 0. So `web_search ✅10 ❌3` is
    "10 succeeded, 3 failed". When any failure appears anywhere in the footer,
    the header gains a one-line legend so a reader who has never seen this
    convention (or has forgotten it) can decode it from the artifact alone. The
    all-green common case stays uncluttered: no failures, no legend.

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
    formatted_rows = [(model, _format_calls(calls)) for model, calls in rows]
    # Only explain the failure notation if a failure actually shows. "❌" with
    # no following digit = total failure; "❌" + digits = partial (that many of
    # the call count failed). Detecting it from the rendered text keeps the
    # legend in lockstep with whatever _format_calls emits.
    any_failure = any("❌" in formatted for _, formatted in formatted_rows)
    header = "> _Tools used:_"
    if any_failure:
        header += "  _(✅ = calls succeeded, ❌ = calls failed)_"
    lines = ["", "", "---", header]
    for model, formatted in formatted_rows:
        lines.append(f"> - **{model}** — {formatted}")
    return "\n".join(lines) + "\n"


# ─── Panel-drafts debug block (opt-in via agentic.debug_panel_drafts) ─

# A standalone horizontal-rule line (---, ----, …) inside a draft. Replaced
# before rendering so a draft's own hr can't reproduce the banner/answer
# separator ("\n\n---\n\n"). Defence-in-depth: the eval now splits the answer
# body on the FIRST separator (see `_answer_body` in scripts/eval_research.py),
# which is already immune to in-prose rules — but neutralizing hr lines here
# keeps the debug block self-contained (a stray draft rule won't render as a
# page-wide divider) and holds even if a consumer splits differently. Table
# rows (|---|) don't match: they don't start at line-begin with hyphens only.
_HR_LINE = re.compile(r"(?m)^[ \t]*-{3,}[ \t]*$")


def _sanitize_draft_text(text: str) -> str:
    """Neutralize hr lines in a draft so the block can't fake a separator."""
    return _HR_LINE.sub("– – –", text).strip()


def panel_drafts_block(drafts: list[dict]) -> str:
    """Render every worker's full draft after the deep answer — debug/eval only.

    Input: the deep panel's `WorkerDraft` dicts (model, content, error,
    elapsed_s, tool_rounds). Output: a "## Panel drafts (debug)" section with
    one `### <model>` subsection per worker, so draft-vs-synth quality can be
    compared from the answer artifact alone (the eval's saved answers file
    then carries both). Empty input → "" (no header on a draft-less answer).

    Failed workers still get a subsection (naming who dropped is the point),
    with the error text stripped of square brackets so it can never collide
    with the eval's error markers ("[internal error]" / "[ollama error").

    Separator-proof by construction: the block contains NO standalone `---`
    line anywhere. The opener is `\\n\\n## Panel drafts (debug)` — a blank gap
    then a heading, with no horizontal rule at all — so it cannot reproduce the
    banner/answer separator (`\\n\\n---\\n\\n`) no matter how a consumer splits,
    and in-draft hr lines are neutralized by `_sanitize_draft_text`. (The eval
    also splits on the FIRST separator, which is already immune; this makes the
    block safe independent of that.)
    """
    if not drafts:
        return ""
    lines = ["", "", "## Panel drafts (debug)", ""]
    for d in drafts:
        model = str(d.get("model") or "?")
        content = (d.get("content") or "").strip()
        meta: list[str] = []
        elapsed = d.get("elapsed_s")
        if elapsed is not None:
            meta.append(f"{float(elapsed):.1f}s")
        rounds = int(d.get("tool_rounds") or 0)
        if rounds:
            meta.append(f"{rounds} tool round{'s' if rounds != 1 else ''}")
        head = f"### {model}" + (" — " + " · ".join(meta) if meta else "")
        lines.append(head)
        lines.append("")
        if content:
            lines.append(_sanitize_draft_text(content))
        else:
            err = str(d.get("error") or "").replace("[", "").replace("]", "")[:200]
            lines.append(f"_no usable draft{' — ' + err if err else ''}_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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

    `emit_header=False` skips the `__aenter__` header emit — for callers that
    already put the header on the wire (e.g. the fast path emits its Thinking
    ack before classifying, then opens a ticker to dot/close the same line).
    Ticking, tail fragments, and the closing mark are unaffected.
    """

    def __init__(
        self,
        header: str,
        emit: Emitter,
        *,
        tick_interval_s: float = TICK_INTERVAL_S,
        emit_header: bool = True,
    ) -> None:
        self._header = header
        self._emit = emit
        self._emit_header = emit_header
        self._tick_interval_s = tick_interval_s
        self._tick_task: asyncio.Task[None] | None = None
        # The tail queue is for fragments produced mid-phase (e.g. per-worker
        # check/cross marks). Bounded so a slow consumer backpressures the
        # producer, never the model work.
        self._tail_q: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._tail_drainer: asyncio.Task[None] | None = None

    async def __aenter__(self) -> PhaseTicker:
        if self._emit_header:
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
    "BANNER_RESEARCHING",
    "BANNER_VERIFYING",
    "BANNER_FACTCHECKING",
    "BANNER_WRITING",
    "BANNER_SEPARATOR",
    "Emitter",
    "PhaseTicker",
    "TICK_INTERVAL_S",
    "tool_summary_block",
    "worker_fail",
    "worker_ok",
]
