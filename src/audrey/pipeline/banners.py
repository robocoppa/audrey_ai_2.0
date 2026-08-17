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
    convention (or has forgotten it) can decode it from the artifact alone, and
    a plain-English disclosure line goes ABOVE the header — see
    `_failure_disclosure` for why the notation alone is not enough. The
    all-green common case stays uncluttered: no failures, no legend, no line.

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
    n_failed, failed_names = _failed_calls(rows)
    header = "> _Tools used:_"
    if n_failed:
        header += "  _(✅ = calls succeeded, ❌ = calls failed)_"
    lines = ["", "", "---"]
    if n_failed:
        lines.append(_failure_disclosure(n_failed, failed_names))
    lines.append(header)
    for model, formatted in formatted_rows:
        lines.append(f"> - **{model}** — {formatted}")
    return "\n".join(lines) + "\n"


def _failed_calls(rows: list[tuple[str, list[dict]]]) -> tuple[int, list[str]]:
    """Total failed calls and the distinct tools they belong to, first-seen.

    Read off the raw call dicts rather than by scanning the rendered "❌" out
    of `_format_calls` output, so the two can never disagree about what
    happened — the rendering is a view, not the record.
    """
    n = 0
    names: list[str] = []
    for _, calls in rows:
        for c in calls:
            if not c.get("is_error"):
                continue
            n += 1
            name = str(c.get("name") or "?")
            if name not in names:
                names.append(name)
    return n, names


def _failure_disclosure(n_failed: int, names: list[str]) -> str:
    """One plain-English line saying a tool failed, above the counts.

    ⚠️ The ❌ notation is not enough on its own, and the box proved it on
    2026-08-10: `kb_search` returned 500 for a whole turn (Qdrant was
    restarting), and the model answered anyway from `list_my_files` and four
    `get_file_text` pages — writing a confident section about a video it had
    never read. The ONLY signal anywhere on screen was `❌1` in a footer row.

    Deliberately mechanical rather than a prompt instruction. The renderer
    already knows a call failed; asking the model to disclose it depends on
    the model complying, and this repo has twice watched a well-meant
    instruction move behaviour somewhere unintended (the `kb_search`
    description's v1 stopped models searching at all). This reaches every
    model and every path, and no wording can talk it out of firing.
    """
    listed = ", ".join(f"`{n}`" for n in names)
    noun = "call" if n_failed == 1 else "calls"
    return (
        f"> ⚠️ **{n_failed} tool {noun} failed** ({listed}) — this answer was "
        f"written without what they would have returned, and may be incomplete."
    )


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


def _draft_section_lines(drafts: list[dict], *, heading: str) -> list[str]:
    """One `<heading> <model> — meta` subsection per worker draft.

    Shared by `panel_drafts_block` (### level) and `research_trace_block`
    (#### level). Failed workers still get a subsection — naming who dropped
    is the point — with the error text stripped of square brackets so it can
    never collide with the eval's error markers ("[internal error]" /
    "[ollama error").
    """
    lines: list[str] = []
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
            # How much web_search content actually reached this worker's context.
            # Only meaningful for a tool-using worker (a tool-free one never
            # searched), so it's gated on `rounds` alongside the round count.
            # The research-grounding diagnostic reads this to tell "never
            # retrieved" (≈0) from "retrieved but wrote thin grounding".
            ws_chars = int(d.get("web_search_chars") or 0)
            meta.append(f"web_search→ctx: {ws_chars} chars")
        # Draft-shape diagnostics. Only rendered when they have something to
        # say, so an ordinary draft's heading is unchanged — a field printed on
        # every draft is a field nobody reads.
        done_reason = str(d.get("done_reason") or "")
        if done_reason and done_reason != "stop":
            meta.append(f"done:{done_reason}")
        # The gap `_strip_think` removed. Worth showing only when it is large
        # enough to be an answer rather than trailing whitespace: a short draft
        # and a stripped-to-death one look identical in the body.
        raw_len = int(d.get("raw_content_len") or 0)
        if raw_len and raw_len - len(content) > 32:
            meta.append(f"raw:{raw_len}→{len(content)}")
        # ⚠️ The first cut of this sent the anomaly to the LOG ONLY, which left
        # the artifact showing an oddly-formatted draft and still unable to say
        # it was odd — the exact gap the diagnostic exists to close. Carried as
        # a field rather than recomputed here: `banners` is a leaf module with
        # no `audrey` imports, and reaching into `deep_panel` for one regex
        # would point the dependency the wrong way.
        anomaly = str(d.get("shape_anomaly") or "")
        if anomaly:
            meta.append(f"⚠ {anomaly}")
        head = f"{heading} {model}" + (" — " + " · ".join(meta) if meta else "")
        lines.append(head)
        lines.append("")
        # ⚠️ The subtask, not the user's question. A split panel replaces the
        # focal question per worker, so "why did this worker answer THAT?" is
        # unanswerable from the draft alone.
        subtask = _one_line(d.get("subtask") or "")
        if subtask:
            lines.append(f"_asked: {subtask[:300]}_")
            lines.append("")
        if content:
            lines.append(_sanitize_draft_text(content))
        else:
            err = str(d.get("error") or "").replace("[", "").replace("]", "")[:200]
            lines.append(f"_no usable draft{' — ' + err if err else ''}_")
        lines.append("")
    return lines


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
    lines += _draft_section_lines(drafts, heading="###")
    return "\n".join(lines).rstrip() + "\n"


# ─── Research trace block (opt-in via agentic.debug_research_trace) ───

def _one_line(text: object) -> str:
    """Collapse all whitespace so ledger/verdict list items stay on one line."""
    return " ".join(str(text or "").split())


def research_trace_block(
    *,
    drafts: list[dict],
    ledger: dict | None = None,
    factcheck: dict | None = None,
    critique: str = "",
    corrections: str = "",
    dispositions: str = "",
) -> str:
    """Render the research pipeline's intermediates — debug/eval only.

    The research-mode counterpart of `panel_drafts_block`, but staged: where
    deep workers write competing answers (draft-vs-synth is the comparison),
    research workers feed a structuring → ledger → fact-check → writer chain,
    so the interesting artifact is the whole chain. Sections, in stage order,
    each omitted when empty:

      Researcher notes   — every worker's raw notes (model, latency, rounds)
      Ledger             — structured claims + sources (`ResearchResult` dump)
      Verifier critique  — the verifier's prose flags
      Fact-check verdicts — per-claim verdicts (`FactCheckResult` dump)
      Corrections / Hedge dispositions — the blocks exactly as handed to the
                           writer, so writer behaviour can be judged against
                           its actual instructions

    `ledger`/`factcheck` are the plain-dict `.model_dump()` shapes carried on
    the pipeline's `done` event. Returns "" when every section is empty.

    Separator-proof like the drafts block: heading-only opener (no hr), and
    the assembled body has every standalone `---` line neutralized, so the
    block can never reproduce the banner/answer separator (`\\n\\n---\\n\\n`).
    One extra constraint: NO heading in here may start with "Sources" — the
    eval locates the real `## Sources` section by the substring `## sources`,
    and `### Sources…` would contain it. Source lists render under a bold
    `**Sources:**` label instead.
    """
    sections: list[str] = []

    if drafts:
        sections.append("### Researcher notes")
        sections.append("")
        sections.extend(_draft_section_lines(drafts, heading="####"))

    claims = list((ledger or {}).get("claims") or [])
    sources = list((ledger or {}).get("sources") or [])
    if claims or sources:
        sections.append(f"### Ledger — {len(claims)} claims, {len(sources)} sources")
        sections.append("")
        if claims:
            sections.append("**Claims:**")
            for c in claims:
                risk = _one_line(c.get("risk")) or "?"
                qual = f"risk: {risk}"
                if c.get("needs_hedge"):
                    reason = _one_line(c.get("hedge_reason"))
                    qual += ", needs hedge" + (f" — {reason}" if reason else "")
                ids = ", ".join(str(s) for s in (c.get("source_ids") or [])) or "none"
                sections.append(
                    f"- **{_one_line(c.get('id')) or '?'}** ({qual}) — "
                    f"{_one_line(c.get('text'))} _(sources: {ids})_"
                )
            sections.append("")
        if sources:
            sections.append("**Sources:**")
            for s in sources:
                stype = _one_line(s.get("source_type")) or "unknown"
                title = _one_line(s.get("title")) or "untitled"
                url = _one_line(s.get("url")) or "no url"
                backs = ", ".join(str(c) for c in (s.get("supports") or [])) or "none"
                sections.append(
                    f"- **{_one_line(s.get('id')) or '?'}** ({stype}) {title} — "
                    f"{url} _(supports: {backs})_"
                )
            sections.append("")
        unresolved = [q for q in ((ledger or {}).get("unresolved_questions") or []) if _one_line(q)]
        if unresolved:
            sections.append("**Unresolved questions:**")
            sections.extend(f"- {_one_line(q)}" for q in unresolved)
            sections.append("")

    if critique.strip():
        sections.append("### Verifier critique")
        sections.append("")
        sections.append(_sanitize_draft_text(critique))
        sections.append("")

    checks = list((factcheck or {}).get("checks") or [])
    fatal = [e for e in ((factcheck or {}).get("fatal_errors") or []) if _one_line(e)]
    if checks or fatal:
        n_drop = sum(1 for c in checks if c.get("verdict") == "unsupported")
        n_hedge = sum(1 for c in checks if c.get("verdict") in ("needs_hedge", "conflicting"))
        sections.append(
            f"### Fact-check verdicts — {len(checks)} checks "
            f"({n_drop} drop, {n_hedge} hedge)"
        )
        sections.append("")
        for c in checks:
            line = f"- **{_one_line(c.get('claim_id')) or '?'}** — {_one_line(c.get('verdict')) or '?'}"
            corrected = _one_line(c.get("corrected_text"))
            if corrected:
                line += f" — corrected: {corrected}"
            notes = _one_line(c.get("notes"))
            if notes:
                line += f" — {notes}"
            sections.append(line)
        if fatal:
            sections.append("")
            sections.append("**Fatal errors:**")
            sections.extend(
                f"- {_one_line(e).replace('[', '').replace(']', '')}" for e in fatal
            )
        sections.append("")

    if corrections.strip():
        sections.append("### Corrections handed to the writer")
        sections.append("")
        sections.append(_sanitize_draft_text(corrections))
        sections.append("")

    if dispositions.strip():
        sections.append("### Hedge dispositions handed to the writer")
        sections.append("")
        sections.append(_sanitize_draft_text(dispositions))
        sections.append("")

    if not sections:
        return ""
    body = "\n".join(["", "", "## Research trace (debug)", "", *sections])
    # Belt-and-braces: neutralize any hr line the assembly produced anyway.
    return _HR_LINE.sub("– – –", body).rstrip() + "\n"
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
    "panel_drafts_block",
    "research_trace_block",
    "tool_summary_block",
    "worker_fail",
    "worker_ok",
]
