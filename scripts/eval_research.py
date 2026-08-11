#!/usr/bin/env python3
"""Run prompt cases against the LIVE Audrey models and check the responses.

WHAT THIS DOES

The hermetic test suite (`tests/`) proves Audrey's plumbing — routing,
prompt strings, stage ordering — but it never calls a real model, so it
cannot judge *answer quality*: whether `audrey_research` actually grounds
facts, emits its stage banners, and renders a `## Sources` list. This
script fills that gap. It sends a set of prompt cases to the running stack
over the LAN (you must be on the VPN / same network) and runs structural,
deterministic-enough checks on each streamed response, then prints the full
answer so you can eyeball quality yourself.

It is the live counterpart to the box smoke test — repeatable, scriptable,
and diffable run-to-run.

HOW IT CONNECTS

Audrey's `/v1/chat/completions` requires an OWUI-validated bearer token
(there is no static API key). So by default this script talks to **OWUI's**
OpenAI-compatible API with an OWUI API key (`sk-…`); OWUI forwards to Audrey
with the session JWT — the same supported path OpenClaw/Hermes use.

    Settings → Account → API Keys in OWUI mints the `sk-…` key.

PERMANENT SETUP (laptop-local, gitignored)

Put the two values in `.env.test.local` at the repo root (gitignored via the
`.env.*.local` rule — separate from the app's `.env`, so the key never leaves
your laptop and can't be committed). The script auto-loads them:

    # .env.test.local  (repo root)
    AUDREY_EVAL_BASE_URL=http://192.168.1.11:8080/api    # OWUI host:port + /api
    AUDREY_EVAL_API_KEY=sk-...                           # OWUI API key

Then just run `.venv/bin/python scripts/eval_research.py` — no exports needed.
A one-off `export AUDREY_EVAL_*` or a `--flag` still overrides the file.

(`--base-url` / `--api-key` flags override env / .env. The OWUI path above is
the repeatable one and needs nothing else. Hitting Audrey directly is no longer
possible over the LAN — as of the 2026-07-18 security review Audrey's `:8000`
is not published to the host (ollama-net only). To debug against Audrey
directly, tunnel it first — `ssh -N -L 8000:audrey-ai:8000 <unraid>` — then
`--base-url http://localhost:8000/v1` with a valid (short-lived) OWUI JWT.)

WHAT IT CHECKS (structural / heuristic — no exact-match, models vary)

Per case, against the reassembled streamed answer:

  - reachable      : the request returned a 2xx SSE stream, not an auth/5xx error
  - no_error_marker: the answer carries no '[internal error]' / '[ollama error'
  - banners        : for deep/research models, the expected progress banners
                     appeared in order (research → Researching/Verifying/Writing)
  - has_answer     : non-empty answer body after the banner separator
  - sources        : for audrey_research on a grounding-type prompt, a
                     '## Sources' section is present with at least one URL
                     (skipped when the case sets "expect_sources": false)
  - url_wellformed : every URL in the Sources block parses as http(s)://…
  - route          : for adaptive (audrey_auto) cases, the inferred path
                     (fast | deep | research, from the banner family) matches
                     the case's "expect_route" — this is how we test the
                     fast/deep gate. Opt-in; skipped when no expect_route set.
  - code_block     : opt-in ("expect_code": true, implied by "code_test"): a
                     language-tagged fenced code block exists in the answer.
  - code_runs      : opt-in ("code_test": "<python asserts>"): the largest
                     ```python block is extracted, the case's asserts are
                     appended, and the file is run in a subprocess (scratch
                     cwd, "code_timeout" seconds, default 15). Pass iff exit
                     0 — the objective signal for the coding suites. The code
                     executed is our own models' output, on this laptop, with
                     only a timeout for isolation: keep case prompts to
                     stdlib-only tasks.
  - contains       : opt-in ("answer_contains": [..]): every listed string
                     appears (case-insensitively) in the answer body — a weak
                     objective signal for reasoning/knowledge cases.
  - not_contains   : opt-in ("answer_not_contains": [..]): NONE of the listed
                     strings appears. ⚠️ Exact substrings only — a model
                     paraphrases around them for free (proved 2026-08-10).
                     Use it for wordings that must never appear, not as the
                     only guard on a behaviour.
  - continuation   : opt-in ("expect_continuation_offer": true): the answer
                     does not refuse the full text while offering no way to
                     continue. Matches on shape, not phrasing.
  - disclaims      : opt-in ("expect_disclaims_absence": true): the answer
                     admits the thing asked for is missing, rather than
                     filling the gap from a summary, a filename, or world
                     knowledge. For cases where "I don't have that" is right.
  - not_truncated  : ALWAYS ON. The prose does not end on a colon (announcing
                     content that never arrived) or leave a code fence open.
                     `has_answer` counts characters and cannot see either.
  - names_files    : opt-in ("expect_names_files": [..]): every listed file is
                     named DISTINCTLY, matched longest-first so one filename
                     being a substring of another cannot satisfy both. The
                     only behavioural check here — for cases whose failure is
                     "answered from the wrong file", which every structural
                     check above scores as a pass.

Plus, for every case, latency is recorded: TTFT (first content delta) and
total wall-clock. The fast path's reason to exist is speed, so these make
"fast" falsifiable run-to-run. They print under each case and land in the
saved answers file; they are measurements, not pass/fail checks.

For research cases, a domain-based source breakdown is also reported —
`sources:N (official:.. academic:.. low_quality:.. other:..) quality:..` —
so a thin-grounding or junk-heavy run is visible at a glance. It re-classifies
the rendered URLs by host (the ledger's own source_type labels don't reach the
prose), so it's a coarse heuristic. Like latency, it is INFORMATIONAL, never a
pass/fail check — the harness proves liveness/structure, not answer quality.

Each case can pin `"model"`, the `"prompt"`, and optional expectations
(`expect_banners`, `expect_sources`). Cases live in a JSON file (default
`scripts/eval_prompts.json`) so you can add/edit prompts without touching
code.

WHAT IT DOES NOT DO

  - It does not assert a model said anything specific — quality is YOUR
    read of the printed answer. The checks are guardrails, not a grader.
  - It does not modify Audrey, OWUI, or any config. Send-and-read only.
  - It cannot run from the laptop without LAN/VPN reachability to the host.

USAGE

    # All cases, streaming, against OWUI:
    .venv/bin/python scripts/eval_research.py

    # One model, custom case file, show raw SSE on failure:
    .venv/bin/python scripts/eval_research.py --model audrey_research \\
        --cases scripts/eval_prompts.json --verbose

    # Compare two models on the same prompts (e.g. research vs deep):
    .venv/bin/python scripts/eval_research.py --model audrey_research
    .venv/bin/python scripts/eval_research.py --model audrey_deep

    # Per-model sweep (every case once per model; passthrough names need the
    # model in config.yaml passthrough.allowed_models on the box). --save-json
    # is the input for scripts/eval_compare.py's case-by-model matrix:
    .venv/bin/python scripts/eval_research.py \\
        --cases scripts/eval_prompts_code_models.json \\
        --models 'audrey_passthrough/qwen3.6:35b,audrey_passthrough/kimi-k2.7-code:cloud' \\
        --save-json docs/testing/2026-07-10-code-sweep-results.json

EXIT CODE

  0 if every case passed every check; 1 otherwise (so it can gate a script).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: uv sync --extra dev", file=sys.stderr)
    sys.exit(2)


DEFAULT_CASES = Path(__file__).parent / "eval_prompts.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_KEYS = ("AUDREY_EVAL_BASE_URL", "AUDREY_EVAL_API_KEY")


def load_dotenv(path: Path = REPO_ROOT / ".env.test.local") -> None:
    """Populate AUDREY_EVAL_* from the repo's `.env.test.local`, if present.

    `.env.test.local` is the eval harness's own secret file — kept separate
    from the app's `.env` (which `config.py` reads for AUDREY_* settings) so a
    test credential never mingles with app config. It's gitignored (matches the
    `.env.*.local` rule), laptop-local; the OWUI API key lives there so you
    don't re-enter it each run. Only fills a variable that is NOT already set in
    the real environment, so an explicit `export` or a `--flag` always wins.
    Tiny hand parser (KEY=VALUE, `#` comments, optional surrounding quotes) — no
    python-dotenv dependency, and it only ever reads our own `AUDREY_EVAL_*`
    keys so it can't disturb anything else.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in ENV_KEYS or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[key] = value
# Banners each mode emits, in order. Substrings (the live text is markdown
# blockquote, e.g. '> _Researching_'); we match the inner word.
_DEEP_BANNERS = ["Planning", "Dispatching panel", "Synthesizing"]
# "Fact-checking" is optional — the stage only runs when a factchecker is
# configured + tool-capable, and is skipped silently otherwise. Detected when
# present, but not required for the ordered-banner check (see _BANNER_SETS).
_RESEARCH_BANNERS = ["Planning", "Researching", "Verifying", "Writing"]
_FACTCHECK_BANNER = "Fact-checking"
# The fast path emits a single distinct banner ('> _Thinking_'); see
# routes/openai/pipeline.py. Its presence is how we tell a fast turn from a
# deep/research one — route inference (infer_route) keys off the banner FAMILY,
# not banner presence, so a fast turn (which does emit a banner) reads as
# "fast", not a false "deep".
_FAST_BANNER = "Thinking"
_BANNER_SETS = {
    "audrey_research": _RESEARCH_BANNERS,
    "audrey_deep": _DEEP_BANNERS,
    "audrey_cloud": _DEEP_BANNERS,
    "audrey_local": _DEEP_BANNERS,
}
_ERROR_MARKERS = ["[internal error]", "[ollama error", "[empty]",
                  "[deep panel produced no usable drafts"]
_SEPARATOR = "\n\n---\n\n"


@dataclass
class CaseResult:
    name: str
    model: str
    ok: bool
    checks: dict[str, bool | None]
    answer: str
    banners_seen: list[str] = field(default_factory=list)
    error: str = ""
    route: str = "unknown"            # inferred path: fast | deep | research
    code_detail: str = ""             # code_runs outcome detail (informational)
    ttft_s: float | None = None
    total_s: float | None = None
    # Informational domain-based source breakdown (never a pass/fail check).
    source_stats: SourceStats | None = None


@dataclass
class StreamTiming:
    """Wall-clock measurements for one streamed turn (seconds; None if N/A)."""
    ttft_s: float | None = None   # time to first content delta (banner or token)
    total_s: float | None = None  # request start → stream end


# One retry after this many seconds when the server refuses connections.
# Connection-refused means the endpoint is down NOW (observed: the Unraid
# scheduled stack restart mid-run, 2026-07-06 — OWUI was back ~51s later);
# without a retry every remaining case burns through in seconds. Only
# ConnectError gets this treatment: an in-flight stream error is not safely
# retryable (the turn may have partially executed server-side).
_CONNECT_RETRY_DELAY_S = 60.0


def _post_stream(base_url: str, api_key: str, model: str, prompt: str,
                 timeout_s: float) -> tuple[str, list[str], str, StreamTiming]:
    """Stream a chat completion. Returns (full_content, banner_words, error, timing).

    `full_content` is every delta concatenated (banners + answer, as the user
    would see). `banner_words` is the ordered list of banner phrases detected.
    `error` is non-empty on a transport/HTTP failure (then content is "").
    `timing` carries TTFT (first content delta) and total wall-clock — the fast
    path's whole reason to exist, so we make them falsifiable run-to-run. TTFT
    counts the first content delta, which for Audrey is the banner ack (the
    deliberate latency fix), so it reflects time-to-first-visible-output.

    A connection-refused (`httpx.ConnectError`) gets ONE retry after
    `_CONNECT_RETRY_DELAY_S` — enough to ride out a container restart
    without turning a genuinely-down stack into a hung run.
    """
    for attempt in (1, 2):
        out = _post_stream_once(base_url, api_key, model, prompt, timeout_s)
        if attempt == 1 and out[2].startswith("ConnectError"):
            print(f"    connection refused; retrying once in {_CONNECT_RETRY_DELAY_S:.0f}s...")
            time.sleep(_CONNECT_RETRY_DELAY_S)
            continue
        return out
    return out


def _post_stream_once(base_url: str, api_key: str, model: str, prompt: str,
                      timeout_s: float) -> tuple[str, list[str], str, StreamTiming]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    content_parts: list[str] = []
    timing = StreamTiming()
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client, \
             client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 300:
                resp.read()
                return "", [], f"HTTP {resp.status_code}: {resp.text[:300]}", timing
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    frame = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = frame.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if not text:
                    continue
                if timing.ttft_s is None:
                    timing.ttft_s = time.monotonic() - t0
                content_parts.append(text)
    except httpx.HTTPError as e:
        timing.total_s = time.monotonic() - t0
        return "", [], f"{type(e).__name__}: {e}", timing
    timing.total_s = time.monotonic() - t0
    full = "".join(content_parts)
    return full, _detect_banners(full), "", timing


# Fast emits 'Thinking'; include it so a fast turn's banner is captured and
# route inference can classify it (see infer_route).
_BANNER_PHRASES = [*_RESEARCH_BANNERS, _FACTCHECK_BANNER, _FAST_BANNER, *_DEEP_BANNERS]


def _detect_banners(full_content: str) -> list[str]:
    """Ordered list of progress-banner phrases the server emitted.

    Detected from the BANNER REGION only — the text BEFORE the first
    `\\n\\n---\\n\\n` separator — and only when a phrase appears in the server's
    blockquote-italic banner form (`> _Phrase_`), never as a bare word. Both
    guards matter now that the panel-drafts debug block streams worker PROSE
    after the answer: a draft that opens a sentence with "Writing in the 4th
    century…" or mentions "Thinking" would otherwise trip a bare substring scan
    and mislabel a plain deep turn as research/fast (observed on
    deep-pythagoras → 'Writing', deep-false-premise-einstein → 'Thinking').
    Scanning only the pre-separator region AND requiring the `> _` marker makes
    prose — wherever it sits — unable to register as a banner.

    Order follows first appearance in the region, so the ordered-subsequence
    banner check still sees Planning→Dispatching panel→Synthesizing as emitted.
    """
    region = full_content.split(_SEPARATOR, 1)[0] if _SEPARATOR in full_content else full_content
    hits: list[tuple[int, str]] = []
    for phrase in _BANNER_PHRASES:
        idx = region.find(f"> _{phrase}")
        if idx != -1:
            hits.append((idx, phrase))
    hits.sort(key=lambda h: h[0])
    ordered: list[str] = []
    for _, phrase in hits:
        if phrase not in ordered:
            ordered.append(phrase)
    return ordered


def _answer_body(full_content: str) -> str:
    """The text after the FIRST banner separator — the actual answer prose.

    The server emits the banner separator (`\\n\\n---\\n\\n`) exactly once,
    between the progress banners and the answer. Split on the FIRST occurrence,
    not the last: the synth/writer often puts its own `---` horizontal rules
    inside the prose (a section break), which reassemble as the identical
    `\\n\\n---\\n\\n`. Splitting on the last one there would discard the whole
    answer up to the final in-prose rule — the bug that made `deep-ssh-keys`
    look truncated and swallowed the panel-drafts debug block. The footer and
    drafts blocks deliberately open with `\\n\\n---\\n>` / `\\n\\n---\\n#` (no
    trailing blank line) so they never form this separator; only in-prose rules
    do, and the first-split is immune to them.
    """
    if _SEPARATOR in full_content:
        return full_content.split(_SEPARATOR, 1)[1].strip()
    return full_content.strip()


_SOURCES_HEADING = re.compile(r"(?m)^## sources")


def _sources_block(answer: str) -> str:
    """Return the '## Sources' section text, or '' if absent.

    The real block is the LAST line-anchored `## Sources` heading BEFORE the
    research trace, bounded at the next `## ` heading. Two hijacks this must
    dodge (both bit the 2026-07-06 trace-on run):

    - The trace renderer keeps its own headings clear of "Sources", but
      researcher NOTES are embedded verbatim and carry their own `## Sources`
      / `## SOURCES:` headings — after the real block, so a whole-answer
      search lands inside the trace (counted 25 "sources" on a block capped
      at 8). Cut at the trace opener before searching.
    - A bare substring search also matches INSIDE `### Sources…` (offset 1),
      so a note's H3 heading hijacks it too. Anchor to line-start `## `.

    Bounding at the next `## ` still matters for the panel-drafts block on
    deep answers (its opener is a level-2 heading).
    """
    low = answer.lower()
    cut = low.find("\n## research trace (debug)")
    if cut != -1:
        answer, low = answer[:cut], low[:cut]
    matches = list(_SOURCES_HEADING.finditer(low))
    if not matches:
        return ""
    block = answer[matches[-1].start():]
    nxt = block.find("\n## ", 1)
    return block[:nxt] if nxt != -1 else block


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for token in text.replace(")", " ").replace("(", " ").split():
        token = token.strip().strip(".,;]>")
        if token.startswith(("http://", "https://")):
            urls.append(token)
    return urls


def _ordered_subsequence(found: list[str], expected: list[str]) -> bool:
    """True if every expected phrase appears in `found` in the expected order."""
    it = iter(found)
    return all(any(f == e for f in it) for e in expected)


# --- Code checks (coding suites: code_block / code_runs) ----------------------

_CODE_FENCE = re.compile(r"```([^\s`]*)[ \t]*\n(.*?)```", re.DOTALL)
# ```py and ```python3 both appear in the wild; treat them as python.
_PY_TAGS = {"python", "py", "python3"}


def _pre_debug_region(answer: str) -> str:
    """The answer body BEFORE any appended debug block.

    With `debug_panel_drafts` / `debug_research_trace` on, the server appends
    every worker's full draft (or the research trace) after the answer — and a
    worker draft can carry its own fenced code. Cutting at the debug openers
    keeps code extraction (and answer_contains) reading the SYNTHESIZED answer,
    not a stray worker draft. Same hardening move as _sources_block's trace cut.
    """
    low = answer.lower()
    cut = len(answer)
    for marker in ("\n## panel drafts (debug)", "\n## research trace (debug)"):
        idx = low.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return answer[:cut]


def _has_tagged_code_block(answer: str) -> bool:
    """True if any fenced code block with a non-empty language tag exists."""
    return any(m.group(1) for m in _CODE_FENCE.finditer(_pre_debug_region(answer)))


def _extract_code_block(answer: str, lang: str = "python") -> str | None:
    """The LARGEST fenced block tagged `lang` (python accepts py/python3), or None.

    Largest, not first: coding answers often show a small usage/example snippet
    alongside the full implementation — the implementation is the big one. Case
    prompts also instruct "a single complete code block" to keep this
    unambiguous.
    """
    tags = _PY_TAGS if lang == "python" else {lang}
    blocks = [m.group(2) for m in _CODE_FENCE.finditer(_pre_debug_region(answer))
              if m.group(1).lower() in tags]
    return max(blocks, key=len) if blocks else None


def _run_code_check(code: str, test: str, timeout_s: float) -> tuple[bool, str]:
    """Run `code` + the case's `test` asserts in a fresh python subprocess.

    Scratch temp dir as cwd (auto-cleaned), `sys.executable` (the harness's own
    venv python), hard timeout. Pass iff exit 0. The detail string is the last
    stderr line on failure — for Python tracebacks that's the exception line,
    which is what you want in the report. No sandbox beyond the timeout: the
    code is our own models' output answering our own stdlib-only prompts.
    """
    with tempfile.TemporaryDirectory(prefix="audrey-eval-code-") as td:
        script = Path(td) / "case.py"
        script.write_text(code + "\n\n" + test + "\n")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)], cwd=td,
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s:.0f}s"
    if proc.returncode == 0:
        return True, "exit 0"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = tail[-1][:200] if tail else ""
    return False, f"exit {proc.returncode}: {detail}" if detail else f"exit {proc.returncode}"


def _contains_all(answer: str, needles: list[str]) -> bool:
    """True if every needle appears case-insensitively in the answer body."""
    low = _pre_debug_region(answer).lower()
    return all(n.lower() in low for n in needles)


def _contains_any(answer: str, needles: list[str]) -> bool:
    """True if ANY needle appears case-insensitively in the answer body."""
    low = _pre_debug_region(answer).lower()
    return any(n.lower() in low for n in needles)


# A refusal to hand over the whole thing, and an offer to keep going. Kept as
# families rather than exact phrases: a substring blacklist of the observed
# wordings ("output length constraints") was added 2026-08-10 and defeated by
# BOTH models on the very next run without either of them trying — "exceeds
# available context limits", "consult the file directly". A model paraphrases
# for free, so the check has to describe the shape, not the sentence.
_DECLINES = re.compile(
    r"\b(?:cannot|can'?t|can not|unable to)\s+"
    r"(?:provide|give|show|produce|display|share|output|retrieve|return)\b",
    re.I,
)
_OFFERS_MORE = re.compile(
    r"(?:would you like|shall i|want me to|let me know if|i can continue"
    r"|continue reading|keep reading|next page|read on|page by page"
    r"|specific sections)",
    re.I,
)


def _prose_region(answer: str) -> str:
    """The model's prose alone — no debug blocks, no tools footer.

    `_answer_body` splits on the FIRST banner separator, so the `_Tools used:_`
    footer (which opens `\\n\\n---\\n>`, deliberately not a separator) stays in
    the body. Fine for most checks; fatal for anything reading the END of the
    answer, which would see a footer row instead of the last thing said.
    """
    body = _pre_debug_region(answer)
    idx = body.find("\n\n---\n>")
    return (body[:idx] if idx >= 0 else body).strip()


def _looks_truncated(answer: str) -> bool:
    """True when the answer stops mid-promise or mid-block.

    ⚠️ `has_answer` only counts characters, and that is not enough. 2026-08-10:
    `video-unnamed-reference` returned "The most recent recording is X. Here's
    a summary of it:" and then nothing at all — a preamble comfortably over the
    20-char floor, announcing content that never arrived, scored PASS.

    Two signals, both unambiguous in prose: ending on a colon (announcing
    something that never came) and an odd number of code fences (a block that
    was opened and never closed).
    """
    prose = _prose_region(answer)
    if not prose:
        return False  # empty is `has_answer`'s job, not this one's
    return prose.rstrip().endswith(":") or prose.count("```") % 2 == 1


# Admitting that something is not there. Like `_DECLINES`, a family rather
# than a phrase list — the paging blacklist proved that models reword freely.
_DISCLAIMS_ABSENCE = re.compile(
    r"(?:no transcript|no summary|no artifacts|no content|no record|no file"
    r"|no information|no such|nothing (?:was|is|to)|does not (?:have|contain"
    r"|exist|cover|appear|include)|doesn'?t (?:have|contain|exist|cover|appear"
    r"|include)|do not (?:have|contain|cover)|don'?t (?:have|contain|cover)"
    r"|has no|there is no|there's no|was not|wasn'?t|is not available"
    r"|isn'?t available|not available|could not find|couldn'?t find"
    r"|unable to find|cannot find|can'?t find|not found|empty)",
    re.I,
)


def _disclaims_absence(answer: str) -> bool:
    """True when the answer admits the thing asked for is not there.

    For cases whose whole point is a gap: a video with no transcript, an
    upload that produced no artifacts, a filename that matches nothing, a
    topic the corpus does not cover. In every one, the failure is to answer
    anyway — from the summary, from the filename, or from world knowledge —
    and that failure reads as a perfectly good answer.
    """
    return bool(_DISCLAIMS_ABSENCE.search(_prose_region(answer)))


def _declines_without_offering(answer: str) -> bool:
    """True when the answer refuses the full text and offers no way forward.

    The `video-long-transcript-paging` failure, stated as a property instead
    of a phrase. `get_file_text` pages a document; the honest answer either
    keeps paging or says how much it has and offers to continue. The failure
    is to declare the whole thing unavailable and send the user elsewhere —
    to "video transcription software", on a file Audrey has already
    transcribed in full.

    Refusing is not itself wrong: an answer may legitimately decline to dump
    33,000 characters. What makes it a failure is refusing with no offer, so
    both halves are required before this fires.
    """
    body = _pre_debug_region(answer)
    return bool(_DECLINES.search(body)) and not _OFFERS_MORE.search(body)


def _names_all_files(answer: str, groups: list[str | list[str]]) -> bool:
    """True if the answer identifies every one of these files, DISTINCTLY.

    Each entry is one file, given either as a string or as a list of
    alternative ways to refer to it — models write "the Magnus Carlsen video"
    at least as often as the full filename, and a check that only accepts the
    exact name fails answers that were perfectly clear. A check that
    false-fails gets ignored, which is worse than not having it.

    ⚠️ Not `_contains_all`, and the difference is the whole point. One
    filename is often a substring of another — "Magnus Carlsen Teaches How to
    Win with the London System.mp4" contains "How to Win with the London
    System.mp4" — so a plain contains-check on both is satisfied by an answer
    that named only the longer file. That is exactly the failure being tested,
    and it would report a pass.

    So each match is CONSUMED, and files are resolved most-specific first (by
    their longest alternative). One mention can never satisfy two files.

    Written for `video-ambiguous-singular`, whose defining failure is silently
    picking one of two same-topic candidates and answering confidently — which
    reads perfectly sourced and, until 2026-08-10, scored PASS on every
    structural check in this file.
    """
    alts = [[g] if isinstance(g, str) else list(g) for g in groups]
    hay = _pre_debug_region(answer).lower()
    for group in sorted(alts, key=lambda g: max(len(a) for a in g), reverse=True):
        # Longest alternative first within the group too: prefer the most
        # specific mention so a vaguer one stays available for another file.
        for name in sorted(group, key=len, reverse=True):
            i = hay.find(name.lower())
            if i >= 0:
                hay = hay[:i] + hay[i + len(name):]
                break
        else:
            return False
    return True


# --- Source-quality reporting (informational, NOT a pass/fail check) ---------
#
# The eval reads the RENDERED answer, which is just `- [title](url)` lines — the
# ledger's own `source_type` labels never reach the prose. So we can only
# re-classify by DOMAIN here, and this is a coarse heuristic that will disagree
# with the pipeline's internal labels at the margins. These counts are printed
# for a human read (spot a thin-grounding or junk-heavy run at a glance); they
# are never gate checks — the harness proves liveness/structure, not quality.

# Host substrings that mark a URL as low-quality grounding. Matches the recurring
# SearXNG-surfaced junk (facebook groups / scribd / slideshare / content farms).
_LOW_QUALITY_HOST_MARKERS = (
    "facebook.com", "scribd.com", "slideshare.net", "quora.com", "pinterest.",
    "reddit.com", "medium.com", "blogspot.", "wordpress.com",
)
# Host substrings that mark academic / scholarly grounding.
_ACADEMIC_HOST_MARKERS = (
    "arxiv.org", ".edu", "scholar.google", "jstor.org", "springer.com",
    "sciencedirect.com", "nature.com", "ieee.org", "acm.org", "ncbi.nlm.nih.gov",
    "researchgate.net", "semanticscholar.org", "plato.stanford.edu",
)
# Host substrings that mark a recognized authoritative / reference domain. This
# is deliberately conservative — an unlisted host is "other", not "official".
_OFFICIAL_HOST_MARKERS = (
    ".gov", "wikipedia.org", "britannica.com", "mathworld.wolfram.com",
    "mactutor", "who.int", "nist.gov", "python.org", "rust-lang.org",
    "docs.", "developer.", "openai.com", "anthropic.com", "deepmind.com",
    "ai.meta.com", "mistral.ai", "deepseek.com",
)


def _classify_host(url: str) -> str:
    """Bucket a URL by host into official | academic | low_quality | other.

    Order matters: academic and low-quality markers are checked before the broad
    'official' set so, e.g., a university `.edu` lands academic and a wordpress
    blog lands low_quality even though neither is on the official list."""
    host = (urlparse(url).netloc or "").lower()
    if not host:
        return "other"
    if any(m in host for m in _ACADEMIC_HOST_MARKERS):
        return "academic"
    if any(m in host for m in _LOW_QUALITY_HOST_MARKERS):
        return "low_quality"
    if any(m in host for m in _OFFICIAL_HOST_MARKERS):
        return "official"
    return "other"


@dataclass
class SourceStats:
    """Informational domain-based breakdown of a case's `## Sources` list.

    None of these are pass/fail — they're reported numbers for a human read.
    `quality` is a one-word summary: GOOD (no low-quality hosts and at least one
    official/academic), PARTIAL (a mix), THIN (fewer than 2 usable URLs), or
    N/A (no Sources block at all)."""
    total: int = 0
    official: int = 0
    academic: int = 0
    low_quality: int = 0
    other: int = 0
    quality: str = "N/A"

    def line(self) -> str:
        """One compact reporting line."""
        return (
            f"sources:{self.total} "
            f"(official:{self.official} academic:{self.academic} "
            f"low_quality:{self.low_quality} other:{self.other}) "
            f"quality:{self.quality}"
        )


def source_stats(answer: str, *, expected: bool) -> SourceStats:
    """Compute the informational source breakdown from a case's answer.

    `expected` is whether a Sources block was expected for this case (creative
    controls opt out); when False and no block is present we report N/A rather
    than a spurious THIN."""
    block = _sources_block(answer)
    urls = _extract_urls(block)
    if not urls:
        return SourceStats(quality="N/A" if not expected else "THIN")
    buckets = {"official": 0, "academic": 0, "low_quality": 0, "other": 0}
    for u in urls:
        buckets[_classify_host(u)] += 1
    total = len(urls)
    if total < 2:
        quality = "THIN"
    elif buckets["low_quality"] == 0 and (buckets["official"] or buckets["academic"]):
        quality = "GOOD"
    else:
        quality = "PARTIAL"
    return SourceStats(
        total=total,
        official=buckets["official"],
        academic=buckets["academic"],
        low_quality=buckets["low_quality"],
        other=buckets["other"],
        quality=quality,
    )


def infer_route(banners: list[str]) -> str:
    """Infer which Audrey path served the turn, from the banner FAMILY seen.

    Audrey emits a different banner set per path (see routes/openai/pipeline.py):
      - fast     → '> _Thinking_'                       (single banner)
      - deep     → '> _Planning_ → _Dispatching panel_ → _Synthesizing_'
      - research → '> _Planning_ → _Researching_ → _Verifying_ → _Writing_'
    So route is observable, not just inferred-from-absence: a fast turn DOES emit
    a banner, it's just a different one. We classify by the most specific family
    present. Research and deep share 'Planning', so research's unique banners
    (Researching/Verifying/Writing) are checked first. Returns
    'fast' | 'deep' | 'research' | 'unknown' (no recognised banner — e.g. an
    error turn before any banner, or an OWUI-utility-task turn).
    """
    if any(b in banners for b in ("Researching", "Verifying", "Writing")):
        return "research"
    if any(b in banners for b in ("Dispatching panel", "Synthesizing", "Planning")):
        return "deep"
    if _FAST_BANNER in banners:
        return "fast"
    return "unknown"


def run_case(base_url: str, api_key: str, case: dict, default_model: str,
             timeout_s: float) -> CaseResult:
    model = case.get("model") or default_model
    name = case.get("name") or case["prompt"][:48]
    content, banners, err, timing = _post_stream(base_url, api_key, model, case["prompt"], timeout_s)
    route = infer_route(banners)

    checks: dict[str, bool | None] = {}
    if err:
        return CaseResult(name=name, model=model, ok=False, checks={"reachable": False},
                          answer="", banners_seen=banners, error=err, route=route,
                          ttft_s=timing.ttft_s, total_s=timing.total_s)

    checks["reachable"] = True
    answer = _answer_body(content)
    checks["no_error_marker"] = not any(m in content for m in _ERROR_MARKERS)
    checks["has_answer"] = len(answer) >= 20
    # Always on, like has_answer: no case ever wants an answer that stops
    # mid-promise, and the character floor cannot see one. See _looks_truncated.
    checks["not_truncated"] = not _looks_truncated(answer)

    # Banner expectation: explicit per-case, else inferred from the model.
    expect_banners = case.get("expect_banners")
    if expect_banners is None:
        expect_banners = model in _BANNER_SETS
    if expect_banners:
        wanted = _BANNER_SETS.get(model, [])
        checks["banners"] = bool(wanted) and _ordered_subsequence(banners, wanted)
    else:
        checks["banners"] = None  # not applicable

    # Sources expectation: default True for audrey_research, unless the case
    # opts out (e.g. a prompt where no grounding is expected).
    expect_sources = case.get("expect_sources")
    if expect_sources is None:
        expect_sources = model == "audrey_research"
    if expect_sources:
        block = _sources_block(answer)
        urls = _extract_urls(block)
        checks["sources"] = bool(block) and len(urls) >= 1
        checks["url_wellformed"] = all(
            urlparse(u).scheme in ("http", "https") and urlparse(u).netloc
            for u in urls
        ) if urls else False
    else:
        checks["sources"] = None
        checks["url_wellformed"] = None

    # Informational source breakdown — reported, never gated. Computed whenever a
    # Sources block was expected (or for any audrey_research case), so a run of
    # research answers carries its grounding-quality numbers for a human read.
    stats = (source_stats(answer, expected=expect_sources)
             if (expect_sources or model == "audrey_research") else None)

    # Route expectation (opt-in): for audrey_auto cases, assert the inferred
    # path matches the intended one — this is how we test the fast/deep gate
    # (token_threshold + deep_intent_phrases). Inferred from the banner family
    # (see infer_route); honest about being inference, not a server-truth signal.
    expect_route = case.get("expect_route")
    if expect_route:
        checks["route"] = route == expect_route
    else:
        checks["route"] = None

    # Code checks (opt-in, the coding suites): `expect_code` requires a
    # language-tagged fenced block; `code_test` (implies expect_code) extracts
    # the largest ```python block, appends the case's asserts, and runs it in a
    # subprocess — the objective pass/fail the other checks can't give.
    code_test = case.get("code_test") or ""
    expect_code = case.get("expect_code")
    if expect_code is None:
        expect_code = bool(code_test)
    checks["code_block"] = _has_tagged_code_block(answer) if expect_code else None
    code_detail = ""
    if code_test:
        code = _extract_code_block(answer, "python")
        if code is None:
            checks["code_runs"] = False
            code_detail = "no ```python block to run"
        else:
            passed, code_detail = _run_code_check(
                code, code_test, float(case.get("code_timeout", 15.0)))
            checks["code_runs"] = passed
    else:
        checks["code_runs"] = None

    # Contains check (opt-in): every `answer_contains` string must appear,
    # case-insensitively — a weak objective signal for reasoning/knowledge
    # cases where the right answer has a distinctive token ("82.8", "tungsten").
    needles = case.get("answer_contains") or []
    checks["contains"] = _contains_all(answer, needles) if needles else None

    # Names-files check (opt-in): every listed file must be named distinctly.
    # The one check here that is behavioural rather than structural — see
    # `_names_all_files` for why a plain contains-check cannot do this job.
    wanted_files = case.get("expect_names_files") or []
    checks["names_files"] = (
        _names_all_files(answer, wanted_files) if wanted_files else None
    )

    # Forbidden-phrase check (opt-in). Some failures have a signature wording
    # rather than a missing element — "system limitations", "output length
    # constraints" — and a model doing the right thing never reaches for them.
    # Cheaper and far more robust than trying to positively detect the good
    # behaviour, which has a hundred valid phrasings.
    banned = case.get("answer_not_contains") or []
    checks["not_contains"] = (not _contains_any(answer, banned)) if banned else None

    # Continuation check (opt-in): a paging case must not dead-end. Shape, not
    # wording — see `_declines_without_offering`.
    checks["continuation"] = (
        (not _declines_without_offering(answer))
        if case.get("expect_continuation_offer") else None
    )

    # Absence check (opt-in): the answer must admit the gap rather than fill
    # it. For the cases where the CORRECT answer is "I don't have that".
    checks["disclaims"] = (
        _disclaims_absence(answer)
        if case.get("expect_disclaims_absence") else None
    )

    ok = all(v for v in checks.values() if v is not None)
    return CaseResult(name=name, model=model, ok=ok, checks=checks,
                      answer=answer, banners_seen=banners, route=route,
                      ttft_s=timing.ttft_s, total_s=timing.total_s,
                      source_stats=stats, code_detail=code_detail)


def _fmt_check(v: bool | None) -> str:
    return "—" if v is None else ("✅" if v else "❌")


def _fmt_latency(r: CaseResult) -> str:
    """One-line latency summary: route + TTFT + total, when measured."""
    bits = [f"route:{r.route}"]
    if r.ttft_s is not None:
        bits.append(f"ttft:{r.ttft_s:.1f}s")
    if r.total_s is not None:
        bits.append(f"total:{r.total_s:.1f}s")
    return "  ".join(bits)


def render(results: list[CaseResult], *, show_answers: bool, verbose: bool) -> None:
    print("=" * 70)
    print("audrey research/deep/fast live eval")
    print("=" * 70)
    cols = ["reachable", "no_error_marker", "has_answer", "banners",
            "sources", "url_wellformed", "route", "code_block", "code_runs",
            "contains", "names_files", "not_contains", "continuation",
            "disclaims", "not_truncated"]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"\n[{status}] {r.name}   (model={r.model})")
        if r.error:
            print(f"   error: {r.error}")
        if r.code_detail:
            print(f"   code: {r.code_detail}")
        line = "   " + "  ".join(f"{c}:{_fmt_check(r.checks.get(c))}" for c in cols)
        print(line)
        print(f"   {_fmt_latency(r)}")
        if r.source_stats is not None:
            print(f"   {r.source_stats.line()}")
        if r.banners_seen:
            print(f"   banners: {' → '.join(r.banners_seen)}")
        if show_answers and r.answer:
            print("   ── answer ──")
            for ln in r.answer.splitlines():
                print(f"   | {ln}")
    passed = sum(1 for r in results if r.ok)
    print("\n" + "=" * 70)
    print(f"{passed}/{len(results)} cases passed all applicable checks")
    print("=" * 70)


def save_results(results: list[CaseResult], save_file: Path) -> None:
    """Write ALL case answers from one run into a single markdown file.

    One file per run (not per case): a short run header, then each case as an
    `## <name>` section (its structural checks + the reassembled answer body),
    so the whole run reads and diffs as a unit against a prior run's file.
    Pair it with a separate quality-evaluation report under the same name stem
    (e.g. `<date>-<desc>-answers.md` + `<date>-<desc>-report.md`).
    """
    passed = sum(1 for r in results if r.ok)
    parts: list[str] = [
        f"# eval run — {save_file.stem}\n",
        f"{len(results)} cases, {passed} passed all applicable checks. "
        "Each section is one case (structural header + answer body).\n",
    ]
    for r in results:
        checks = "  ".join(f"{k}:{_fmt_check(v)}" for k, v in r.checks.items())
        section = (
            f"---\n\n## {r.name}\n\n"
            f"- model: `{r.model}`\n"
            f"- status: {'PASS' if r.ok else 'FAIL'}\n"
            f"- route: {r.route}\n"
            f"- latency: {_fmt_latency(r)}\n"
            f"- banners: {' → '.join(r.banners_seen) or '(none)'}\n"
            f"- checks: {checks}\n"
        )
        if r.source_stats is not None:
            section += f"- {r.source_stats.line()}\n"
        if r.code_detail:
            section += f"- code: {r.code_detail}\n"
        if r.error:
            section += f"- error: {r.error}\n"
        section += f"\n{r.answer or '(no answer body)'}\n"
        parts.append(section)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    save_file.write_text("\n".join(parts))


def save_json(results: list[CaseResult], save_json_file: Path) -> None:
    """Machine-readable run record — the input for scripts/eval_compare.py.

    One flat JSON array, one object per case: the structural verdicts plus the
    informational measurements (latency, answer length). Deliberately dumb —
    no schema version, no nesting — so the compare tool stays a pure
    parse-and-format job. Answers are NOT included; they live in the paired
    `--save-file` markdown (the human-read artifact).
    """
    records = [
        {
            "name": r.name,
            "model": r.model,
            "ok": r.ok,
            "checks": r.checks,
            "route": r.route,
            "ttft_s": r.ttft_s,
            "total_s": r.total_s,
            "answer_len": len(r.answer),
            "banners": r.banners_seen,
            "error": r.error,
            "code_detail": r.code_detail,
            # Grounding-quality numbers (research/sourced cases). None when the
            # case computed no source stats (e.g. a code case) — kept as an
            # explicit null so the record shape is stable for eval_compare.py.
            "sources": (
                {
                    "total": r.source_stats.total,
                    "official": r.source_stats.official,
                    "academic": r.source_stats.academic,
                    "low_quality": r.source_stats.low_quality,
                    "other": r.source_stats.other,
                    "quality": r.source_stats.quality,
                }
                if r.source_stats is not None
                else None
            ),
        }
        for r in results
    ]
    save_json_file.parent.mkdir(parents=True, exist_ok=True)
    save_json_file.write_text(json.dumps(records, indent=2) + "\n")


def _expand_sweep(cases: list[dict], models: list[str]) -> list[dict]:
    """Cross every case with every sweep model: model overridden, name suffixed.

    Grouped BY MODEL (all cases for model 1, then model 2, …), not by case:
    consecutive requests to the same local model avoid Ollama reloading
    weights between every turn. The ` [<model>]` name suffix keeps save-file
    sections unique and diffable; eval_compare.py strips it to rebuild the
    case-by-model matrix.
    """
    return [
        {**c, "model": m, "name": f"{c.get('name') or c['prompt'][:48]} [{m}]"}
        for m in models
        for c in cases
    ]


def main() -> int:
    # Load the gitignored .env.test.local first, so the env-var defaults below
    # see AUDREY_EVAL_*. A real export or an explicit --flag still overrides it.
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("AUDREY_EVAL_BASE_URL", ""),
                   help="OWUI API base, e.g. http://192.168.1.11:3000/api "
                        "(or env AUDREY_EVAL_BASE_URL)")
    p.add_argument("--api-key", default=os.environ.get("AUDREY_EVAL_API_KEY", ""),
                   help="OWUI API key sk-… (or env AUDREY_EVAL_API_KEY)")
    p.add_argument("--model", default="audrey_research",
                   help="default model for cases that don't pin one (default audrey_research)")
    p.add_argument("--models", default="",
                   help="comma-separated model sweep: run EVERY case once per "
                        "model (overrides case/--model; result names get a "
                        "' [<model>]' suffix). Use audrey_passthrough/<name> "
                        "ids for per-model comparisons — the name must be in "
                        "config.yaml passthrough.allowed_models on the box")
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES,
                   help=f"prompt-case JSON (default {DEFAULT_CASES})")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-request timeout seconds (deep/research is slow; default 600)")
    p.add_argument("--no-answers", action="store_true",
                   help="don't print the full answer bodies (checks only)")
    p.add_argument("--verbose", action="store_true", help="extra detail on failures")
    p.add_argument("--only", default="",
                   help="run only cases whose name contains this substring")
    p.add_argument("--save-file", type=Path, default=None,
                   help="write ALL answers from this run into one markdown file "
                        "(one section per case). Name it with a date + test "
                        "description, e.g. docs/testing/2026-06-26-accuracy-stress-answers.md")
    p.add_argument("--save-json", type=Path, default=None,
                   help="write per-case results (checks + latency, no answers) "
                        "as JSON — the input for scripts/eval_compare.py, e.g. "
                        "docs/testing/2026-07-10-code-sweep-results.json")
    args = p.parse_args()

    if not args.base_url or not args.api_key:
        print("error: set --base-url and --api-key (or AUDREY_EVAL_BASE_URL / "
              "AUDREY_EVAL_API_KEY). See the module docstring for the OWUI setup.",
              file=sys.stderr)
        return 2

    if not args.cases.exists():
        print(f"error: case file not found: {args.cases}", file=sys.stderr)
        return 2
    cases = json.loads(args.cases.read_text())
    if args.only:
        cases = [c for c in cases if args.only.lower() in (c.get("name", "") + c["prompt"]).lower()]
    if not cases:
        print("error: no cases to run (check --only filter)", file=sys.stderr)
        return 2

    sweep = [m.strip() for m in args.models.split(",") if m.strip()]
    if sweep:
        cases = _expand_sweep(cases, sweep)

    results: list[CaseResult] = []
    for case in cases:
        print(f"running: {case.get('name') or case['prompt'][:48]} "
              f"(model={case.get('model') or args.model})…", file=sys.stderr)
        results.append(run_case(args.base_url, args.api_key, case, args.model, args.timeout))

    render(results, show_answers=not args.no_answers, verbose=args.verbose)
    if args.save_file is not None:
        save_results(results, args.save_file)
        print(f"saved {len(results)} answers to {args.save_file}", file=sys.stderr)
    if args.save_json is not None:
        save_json(results, args.save_json)
        print(f"saved {len(results)} results to {args.save_json}", file=sys.stderr)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
