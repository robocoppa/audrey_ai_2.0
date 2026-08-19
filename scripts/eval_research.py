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
                     (fast | escalated | deep | research) matches the case's
                     "expect_route" — this is how we test the fast/deep gate.
                     Opt-in; skipped when no expect_route set. Accepts a list
                     when a case tolerates more than one. ⚠️ "escalated" is a
                     turn that entered FAST and was re-run through the deep
                     panel mid-graph; its banners still say fast, so "fast"
                     alone would pass on a full panel run. See infer_route.
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
  - not_misattributed : ALWAYS ON (opt out with "allow_user_attribution").
                     The answer does not credit the USER with a file's content
                     — "You note that…", "you advocate for…" about an uploaded
                     video. Content correct, source wrong, every other check
                     green. Opt out only for a prompt that itself makes a
                     claim the model may reflect back.
  - no_fiction     : ON for every case in a suite whose cases declare
                     "corpus" (today: "video"). The answer makes no claim the
                     corpus contradicts — a submission finish in a clip whose
                     artifacts show only pinning, a weight class, the wrong
                     tournament, Carlsen playing Black, a file the corpus does
                     not have. Per-corpus rather than per-case on purpose:
                     invention had a blacklist on two of the twelve video
                     cases and was turning up on six. ⚠️ `_KNOWN_UPLOADS` must
                     be updated when the box's uploads change.
  - calibrated     : opt-in ("expect_hedge_when_wrong": true), and evaluated
                     ONLY when `contains` already failed: a wrong answer that
                     flags itself as uncertain passes, a wrong answer asserted
                     flatly fails. Accuracy and trustworthiness are separate
                     measurements — "is it right" and "if it is not, would I
                     know" — and a confidently-wrong answer is the worst
                     product a user can get. Never applicable to a right
                     answer: hedging one is not a virtue.
  - within_word_budget : opt-in ("answer_max_words": N): the answer body is at
                     or under N whitespace-separated words. For prompts that
                     state a length limit — an unchecked limit is a promise the
                     suite lets the model break for free.
  - names_files    : opt-in ("expect_names_files": [..]): every listed file is
                     named DISTINCTLY, matched longest-first so one filename
                     being a substring of another cannot satisfy both. The
                     only behavioural check here — for cases whose failure is
                     "answered from the wrong file", which every structural
                     check above scores as a pass.
  - grounded       : AUTOMATIC, and applicable only when the tools footer shows
                     `list_my_files` as the ONLY tool that succeeded. That
                     listing carries no file contents by construction, so an
                     account of what a file SAYS after it alone was invented.
                     Not applicable — never a failure — when the footer did not
                     parse or any other tool ran. The only check gated on the
                     footer rather than on the prose, and the only one that can
                     catch an answer whose every sentence is well-formed and
                     whose every fact is made up.

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
        --models 'audrey_passthrough/qwen3.8:latest,audrey_passthrough/kimi-k2.7-code:cloud' \\
        --save-json docs/testing/2026-07-10-code-sweep-results.json

EXIT CODES

  0   every case passed every applicable check
  1   the run COMPLETED and at least one check failed. This is the normal
      result of a suite that measures something — not an error. Read the
      [FAIL] blocks, not the exit code.
  2   setup problem: no base-url/key, missing or empty case file
  3   the harness itself crashed; traceback is printed. ⚠️ Kept distinct from
      1 on purpose — Python exits 1 on an uncaught exception by default, and
      that collision made "it exited 1" unreadable.
  130 interrupted
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
import traceback
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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
# ⚠️ `"[error:"` IS OWUI'S ENVELOPE, NOT AUDREY'S, and it arrives as ASSISTANT
# CONTENT. When OWUI's upstream read to Audrey times out it does not drop the
# stream — it streams the string `[error: POST /api/chat (stream) transport
# error: ReadTimeout: ]` as the answer. Every structural check then passes:
# `reachable` (no transport error reached us), `has_answer` (it clears the
# 20-char floor), `not_truncated` (it ends on a bracket). On 2026-08-18 a
# 180s laguna-s timeout was caught ONLY because that case carries
# `answer_contains`; any case without content assertions would have scored a
# clean PASS on a generation that never happened.
_ERROR_MARKERS = ["[internal error]", "[ollama error", "[empty]",
                  "[deep panel produced no usable drafts", "[error:"]
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
    fiction_detail: str = ""          # which corpus fictions no_fiction found
    context_detail: str = ""          # INFORMATIONAL: degraded-context report
    ungrounded_detail: str = ""       # which phrase failed `grounded`
    ttft_s: float | None = None
    total_s: float | None = None
    # Informational domain-based source breakdown (never a pass/fail check).
    source_stats: SourceStats | None = None
    # ⚠️ THE ARM. What this run ASKED Audrey for on the passthrough `think`
    # field: True / False / None (field omitted, model template decides).
    # It rides in the results JSON because the only other record of it was a
    # container log line, and `docker logs` is empty after the next recreate —
    # which on 2026-08-19 made two completed model sweeps unattributable.
    think_requested: bool | None = None


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


def _is_direct_audrey(base_url: str) -> bool:
    """True when the base-url points at Audrey itself rather than Open WebUI.

    Audrey's `:8000` is not published to the host (2026-07-18 security review),
    so the LAN path is OWUI — but the eval CONTAINER runs on `ollama-net`
    alongside `audrey-ai`, which is why direct is reachable at all from a box
    run. Matched on host/port rather than on a flag so a stale `eval.env`
    cannot claim direct while pointing somewhere else.
    """
    from urllib.parse import urlparse
    host = (urlparse(base_url).hostname or "").lower()
    port = urlparse(base_url).port
    return host in {"audrey-ai", "audrey"} or (
        host in {"localhost", "127.0.0.1"} and port == 8000)


def _request_body(model: str, prompt: str, think: bool | None) -> dict:
    """The chat-completions body for one case.

    `think` is Audrey's vendor extension, omitted entirely when None so the
    request is byte-identical to what this harness has always sent. Pulled out
    of `_post_stream_once` purely so a test can assert on the payload without
    a live server — the field's whole value is that the ARM travels with the
    run, and a silently-dropped field would restore the ambiguity it exists to
    remove.
    """
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if think is not None:
        body["think"] = think
    return body


def _post_stream(base_url: str, api_key: str, model: str, prompt: str,
                 timeout_s: float, think: bool | None = None,
                 ) -> tuple[str, list[str], str, StreamTiming]:
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
        out = _post_stream_once(base_url, api_key, model, prompt, timeout_s, think)
        if attempt == 1 and out[2].startswith("ConnectError"):
            print(f"    connection refused; retrying once in {_CONNECT_RETRY_DELAY_S:.0f}s...")
            time.sleep(_CONNECT_RETRY_DELAY_S)
            continue
        return out
    return out


def _post_stream_once(base_url: str, api_key: str, model: str, prompt: str,
                      timeout_s: float, think: bool | None = None,
                      ) -> tuple[str, list[str], str, StreamTiming]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = _request_body(model, prompt, think)
    content_parts: list[str] = []
    timing = StreamTiming()
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client, \
             client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 300:
                resp.read()
                # ⚠️ Set the clock here too. This branch used to return `timing`
                # untouched, so an HTTP error recorded NO latency at all — the
                # answers file printed a bare `latency: route:unknown` and the
                # results JSON carried `total_s: null`. That is the difference
                # between "the stack refused instantly" and "it ground for four
                # minutes and then 500'd", which is the first thing you want to
                # know. Found 2026-08-18 when twelve OWUI connection errors
                # arrived with no timings between them.
                timing.total_s = time.monotonic() - t0
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


def _bare_code(answer: str) -> str | None:
    """The whole answer as code, when the answer IS nothing but code.

    ⚠️ Added 2026-08-12 because the harness was failing obedience. Every
    `code_*` prompt ends **"Return only the code."** Two of the three bake-off
    models took that literally on `code-merge-intervals` and returned an
    unfenced function — correct code, no prose — and scored
    `code_block:❌ code: no ```python block to run`. A check that fails an
    answer for following the instruction is worse than no check.

    Three guards keep this from swallowing prose, because "does it compile"
    alone is far too weak — a single bare word like `Hello` is a valid Python
    expression statement and compiles clean:

      • no fence anywhere, so a mis-tagged or untagged block still fails
        rather than being silently rescued here;
      • `def ` present, which every `expect_code` prompt asks for;
      • the WHOLE body compiles, so any sentence of prose alongside the code
        is a SyntaxError and disqualifies it.
    """
    body = _pre_debug_region(answer).strip()
    if not body or "```" in body or "def " not in body:
        return None
    try:
        compile(body, "<answer>", "exec")
    except (SyntaxError, ValueError):
        return None
    return body


def _has_tagged_code_block(answer: str) -> bool:
    """True if the answer delivered runnable code: a tagged fence, or nothing
    but code. See `_bare_code` for why the second form counts."""
    if any(m.group(1) for m in _CODE_FENCE.finditer(_pre_debug_region(answer))):
        return True
    return _bare_code(answer) is not None


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
    if blocks:
        return max(blocks, key=len)
    # An unfenced answer that is nothing but code still has code to run, and
    # the prompts ask for exactly that. Python only — `_bare_code` compiles.
    return _bare_code(answer) if lang == "python" else None


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


# Uncertainty a READER would act on. Not "the model was tentative" — phrases
# that tell a user this claim needs checking before they rely on it.
# ⚠️ Only ever evaluated on an answer already judged WRONG, so a false positive
# downgrades the severity of a failure it can never create. That asymmetry is
# why this can be a phrase list at all; `_disclaims_absence` has no such luxury
# and has leaked three times.
_HEDGES = re.compile(
    r"\b(?:i'?m not (?:certain|sure)|not (?:entirely )?certain|i'?m unsure"
    r"|i am not (?:certain|sure)|i believe|i think(?: that)?|if i recall"
    r"|if memory serves|i could be (?:wrong|mistaken)|i may be (?:wrong|mistaken)"
    r"|to the best of my knowledge|from memory|worth (?:double[- ])?checking"
    r"|please (?:verify|double[- ]check)|you (?:should|may want to) verify"
    r"|i don'?t have (?:reliable|confident)|uncertain about|not confident)\b",
    re.I,
)


def _hedges(answer: str) -> bool:
    """True when the answer flags its own claim as needing verification."""
    return bool(_HEDGES.search(_unemphasised(answer)))


def _within_word_budget(answer: str, limit: int) -> bool:
    """True when the answer body is at or under `limit` whitespace-separated words.

    Deliberately crude. The point is not to adjudicate what a word is — it is
    that a prompt saying "120 words maximum" should be able to FAIL, and a
    model 40% over the cap trips this on any reasonable definition. Uses the
    pre-debug region so an appended panel-drafts block cannot blow the budget.
    """
    return len(_pre_debug_region(answer).split()) <= limit


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
# ⚠️ Every phrase here is a SECOND-PERSON invitation to ask, which can only be
# about the next step. Two widenings have been rejected by measurement:
#
#   • by shape ("remaining part", "next section") — of its three flips, two
#     were dead-ends wearing an offer's vocabulary: "you would need to continue
#     retrieving the remaining parts", "you would need to download the video
#     file and use a speech-to-text service". The distinction is not whether
#     more text is mentioned, it is WHO fetches it.
#   • by first-person capability ("I can provide", "I will retrieve") — an
#     answer can say "I can provide the summary and key points" and then send
#     the user away for the transcript itself. A capability statement about
#     SOMETHING ELSE is not an offer to continue.
#   • ⚠️ "you would like me to" — ACCEPTED on run 12 and REMOVED on run 13,
#     because it is the second failure wearing the first one's clothes. Second
#     person, but not an invitation to ask for the next page: "if you have a
#     transcript text you would like me to analyze, please paste it here" is
#     the dead-end asking the USER to supply what the tool already returned.
#     It passed exactly that answer, on a case whose `get_file_text` had
#     succeeded three times. Across the archive it is the sole match in four
#     sections and carried a genuine offer in none of them — the run-12 answer
#     it was added for also says "please let me know", which already matched.
#     A widening that is redundant where it looks right is only doing work
#     where it is wrong.
_OFFERS_MORE = re.compile(
    r"(?:would you like|shall i|want me to|let me know|i can continue"
    r"|continue reading|keep reading|next page|read on|page by page"
    r"|specific sections|ask (?:me )?for (?:the )?(?:next|subsequent|more)"
    r"|(?:give|send|show) me the next|just ask|say the word)",
    re.I,
)


# ⚠️ Models write `don’t`, not `don't`. A regex spelled `don'?t` matches only
# the ASCII form and silently fails the typographic one — which is what models
# actually emit. Cost a false FAIL on 2026-08-11: a textbook "I don’t have any
# references to that in your uploaded videos" scored `disclaims:❌`.
#
# Normalised for PROSE matching only, never inside `_pre_debug_region` — code
# extraction reads that, and rewriting quotes inside a code block would corrupt
# the very thing `code_runs` then executes.
_SMART_QUOTES = str.maketrans({"’": "'", "‘": "'",
                               "“": '"', "”": '"'})


def _prose_region(answer: str) -> str:
    """The model's prose alone — no debug blocks, no tools footer, ASCII quotes.

    `_answer_body` splits on the FIRST banner separator, so the `_Tools used:_`
    footer (which opens `\\n\\n---\\n>`, deliberately not a separator) stays in
    the body. Fine for most checks; fatal for anything reading the END of the
    answer, which would see a footer row instead of the last thing said.
    """
    body = _pre_debug_region(answer)
    idx = body.find("\n\n---\n>")
    return (body[:idx] if idx >= 0 else body).strip().translate(_SMART_QUOTES)


# Markdown emphasis, removed for PHRASE matching only. Same class of bug as
# `_SMART_QUOTES` and found the same way: 2026-08-11 20:52, a textbook "The file
# has **no audio transcript**" scored `disclaims:❌`, because `has no` needs the
# two words adjacent and the model had put two asterisks between them. Every
# phrase family in this file has the same hole — `as **Black**` hides a fiction
# from `no_fiction`, `**your** channel` hides a misattribution — and which
# phrases a model happens to bold is not a property any check should depend on.
_EMPHASIS = re.compile(r"\*+")


def _unemphasised(answer: str) -> str:
    """Prose with the emphasis markers taken out, for the phrase families.

    ⚠️ Deliberately NOT used by `_filenames_named` or `_looks_truncated`, and
    both exclusions were measured, not assumed:

    • `_FILENAME_EDGE` contains `*` **as a delimiter**. Strip it and the
      backward walk from `.mp4` runs on into the sentence: over the archive,
      "I don't have any information about what's in silent.mp4" becomes an
      invented filename. Three new false positives on an always-on check.
    • `_looks_truncated` fires on a trailing colon. An answer ending
      "**Summary:**" ends on an asterisk raw and on a colon stripped.

    For the four families that do use it, the archive moves exactly one verdict
    (the false fail above) and five fiction spans stop carrying stray asterisks.
    """
    return _EMPHASIS.sub("", _prose_region(answer))


def _flattened(text: str) -> str:
    """Text with abbreviation periods removed, so `.` means end-of-sentence.

    ⚠️ `Jr.` is a full stop to `str.find`. `_ABBREVIATED` is defined far below
    with the corpus facts, but it belongs to the normalisation ladder, and this
    wrapper is where the ladder ends: `_pre_debug_region` → `_prose_region`
    (+quotes) → `_unemphasised` (+markup) → `_flattened` (+abbreviations).

    Applied per-check rather than folded into `_prose_region`, and deliberately:
    `_looks_truncated` fires on trailing punctuation, so an answer ending
    "…Rafael Lovato Jr." would read as unterminated once the period is gone.
    """
    return _ABBREVIATED.sub(lambda m: m.group(0).replace(".", ""), text)


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`.

    ⚠️ THE ONE COPY. There were two, and only one of them flattened
    abbreviations first — so the same `Jr.` bug that `_corpus_fictions` was
    taught about in 2026-08-11 stayed live in `_reports_degraded_context`,
    which had its own hand-rolled `rfind(".")`. Every sentence window in this
    file goes through here now, so teaching the extractor once is enough.

    Callers pass offsets into ALREADY-`_flattened` text — the offsets have to
    be from the same string the window is cut from.
    """
    left = text.rfind(".", 0, start) + 1
    right = text.find(".", end)
    return text[left:right if right >= 0 else len(text)].strip()


def _footer_region(answer: str) -> str:
    """The tools footer alone — the counterpart to `_prose_region`.

    Everything else in this file reads the prose and throws the footer away.
    This is the one check that needs both, because it compares what the answer
    SAYS against what the answer's own footer REPORTS.
    """
    body = _pre_debug_region(answer)
    idx = body.find("\n\n---\n>")
    return body[idx:] if idx >= 0 else ""


_TOOL_OK = re.compile(r"`(\w+)`\s*✅(\d+)")


def _tools_that_succeeded(answer: str) -> dict[str, int]:
    return {t: int(n) for t, n in _TOOL_OK.findall(_footer_region(answer))
            if int(n) > 0}


# ─── Content claimed with nothing behind it ───────────────────────────
#
# The one check here that rests on an architectural invariant rather than on a
# judgement about prose, which is why it can gate where the phrase families
# cannot. `MyFileRow` (tools-server/app.py:718) carries filename, kind, status,
# uploaded_at, duration_s, failure_reason, waiting_for_s and `artifacts` — a
# list of WHICH sidecars exist, never a word of what any of them says. The
# `summary` field was deleted from it on 2026-08-06 with the reason recorded
# in place: "a listing that carries contents is a listing that gets answered
# from instead of read from."
#
# So summary or transcript text cannot reach the model except through
# `get_file_text` or `kb_search`. If the footer shows `list_my_files` as the
# ONLY tool that succeeded, an account of what the file SAYS did not come from
# the file. There is no honest reading — unlike a report of thinned context,
# which turned out to be true and cost a retracted check to learn.
#
# Measured over the archive: 30 hits in 1,080 sections, every one of them
# `video-unnamed-reference` inventing a different match from the same filename
# — a rear-naked choke, a guillotine, a mounted triangle, an eye-gouging
# disqualification, John Danaher recast as Lovato's father. The same case
# passes 12 times when it calls `get_file_text`, and those 12 agree with each
# other and with the corpus: a sub-5-minute clip, two black belts, a red and
# blue mat. Fabrication and grounding are cleanly separable here.
_CATALOGUE_ONLY = {"list_my_files"}

_UNGROUNDED_CLAIM = re.compile(
    # Announcing a summary the catalogue cannot have supplied. The adjective
    # slot catches "here's a quick summary"; `\b` keeps it out of "there's no
    # summary, transcript or visual", which is an honest report of absence and
    # matched "here's ... summary" as a substring before the anchor went in.
    r"\bhere(?:'s| is| are)?\s+(?:the|a|its|his|her|their)?\s*"
    r"(?:(?!no\b|not\b)\w+\s+){0,2}summar(?:y|ies)"
    r"|\bsummary of (?:that|the|this|it|your|his|her)\b"
    # ⚠️ This line used to carry `(?:\*\*)?` on both ends — a pattern working
    # around markup instead of the text being normalised. That is the shape of
    # the bug this file kept re-finding (`Jr.`, smart quotes, `as **Black**`),
    # so the workaround is gone and the input is `_unemphasised` below. Zero
    # verdicts move archive-wide; it now also survives `*Summary*`.
    r"|^\s*summary(?:\s+of\b[^\n]*)?:?\s*$"
    # Narrating the contents outright.
    r"|\b(?:the|this)\s+(?:video|footage|recording|clip|match|documentary)\s+"
    r"(?:captures?|shows?|covers?|features?|depicts?|highlights?|portrays?"
    r"|presents?|centers?|combines?|begins?|opens?)\b",
    re.I | re.M,
)


def _ungrounded_content(answer: str) -> str | None:
    """The offending phrase, "" if grounded, None if not applicable.

    ⚠️ Returns None — not applicable, never a failure — whenever the footer
    did not parse or any tool beyond the catalogue succeeded. A harness that
    cannot see what the model was handed must not call the output a lie; that
    rule was learned the expensive way and applies with full force here.

    ⚠️ Not a bare "did a content tool run" test. The first draft listed the
    content tools by name, left `web_search` off, and failed two research
    answers built on fourteen and three successful searches. The gate is
    inverted for that reason: the catalogue is a closed set of one, and a
    closed set cannot be under-specified the way an open one can.
    """
    tools = _tools_that_succeeded(answer)
    if not tools or set(tools) != _CATALOGUE_ONLY:
        return None
    m = _UNGROUNDED_CLAIM.search(_unemphasised(answer))
    return m.group(0).strip() if m else ""


# An answer reporting that content it asked for is not in front of it, while
# its own footer shows the tool call succeeded.
#
# ⚠️⚠️ THIS IS NOT A PASS/FAIL CHECK, AND THE REASON IS THE MOST EXPENSIVE
# LESSON IN THIS FILE. It shipped on 2026-08-11 as `no_false_limit`, on the
# theory that a limit blamed for content the tools reached must be invented.
# One day later the correlation below killed that theory:
#
#   successful tool calls in a turn  →  fail rate over every archived answer
#     0: 16.9% (n=83)   1: 14.2% (n=155)   2: 6.9% (n=101)
#     3: 41.1% (n=95)   4–5: ~31% (n=38)
#
# A cliff at three, not a gradient — and three is where the fast path's
# `compress_after_round: 2` + `compress_keep_last: 1` leaves exactly ONE tool
# message verbatim and stubs the rest. `video-long-transcript-paging` fails 0
# of 14 turns below three calls and 20 of 32 at three. So "I am missing the
# middle portion" is very likely TRUE: the middle was compacted out of the
# model's context before it wrote. A check that fails an answer for honestly
# reporting its own degraded context is worse than no check at all.
#
# The detector is kept because the SIGNAL is good — it just means the opposite
# of what it was built to mean. It now reports, and never gates.
_LIMIT_EXCUSE = re.compile(
    r"(?:file ?size|size|length|token|character|tool|system|technical|processing)"
    r"[- ]?(?:limit|limitation|constraint|restriction)s?"
    r"|(?:do(?:es)? not|don'?t|doesn'?t) have access to (?:that|the|this|your)",
    re.I,
)
# The limit must be offered as the reason the CONTENT cannot be had.
_UNREACHABLE = re.compile(
    r"\b(?:not |in)accessible\b|\bmissing\b|\bunavailable\b|\bprevents?\b"
    r"|(?:could|can) ?not (?:\w+ ){0,2}(?:retrieve|access|read|obtain|fetch)"
    r"|(?:couldn'?t|cannot|can'?t|unable to) (?:\w+ ){0,2}"
    r"(?:retrieve|access|read|obtain|fetch)",
    re.I,
)
# ⚠️ A limit on the ANSWER's own size is real and must not be flagged. Two of
# the four archive hits were exactly that — "exceeds the length limits for a
# single reply", "only retrieved partial sections due to output length
# constraints" — in answers that then paged and offered to continue. Declining
# to dump 33,000 characters is legitimate; declaring them unreachable is not.
_ABOUT_THE_REPLY = re.compile(
    r"\b(?:repl(?:y|ies)|response|message|answer|output|here)\b", re.I)


def _reports_degraded_context(answer: str) -> str:
    """INFORMATIONAL. The answer says it could not reach content its own footer
    says a tool returned — the signature of history compaction, not of lying.

    Reads the FOOTER, which nothing else here does. Fires only when some call
    SUCCEEDED, so an answer written after a genuine tool failure never trips
    it. Two hits over the whole archive (all suites, 1,008 sections): "I am
    missing the middle portion due to technical limits" (`get_file_text` ✅3)
    and "only partial transcripts are accessible due to file size limitations"
    (`get_file_text` ✅2) — both at the three-call cliff, both consistent with
    the model describing a context it really was handed.

    ⚠️ Never gate on this. See the block above `_LIMIT_EXCUSE`. Its job is to
    mark the turns worth reading `context-trace` lines for.
    """
    ok = _tools_that_succeeded(answer)
    if not ok:
        return ""
    prose = _flattened(_unemphasised(answer))
    for m in _LIMIT_EXCUSE.finditer(prose):
        sentence = _sentence_around(prose, m.start(), m.end())
        if _NEGATORS.search(sentence) or _ABOUT_THE_REPLY.search(sentence):
            continue
        if not _UNREACHABLE.search(sentence):
            continue
        calls = ", ".join(f"{t} ✅{n}" for t, n in ok.items())
        return f"blamed {m.group(0)!r} while the footer shows {calls}"
    return ""


# A reasoning delimiter that reached the user. Structural and unambiguous —
# these tags belong in Ollama's separate `message.thinking` field, never in
# `content`, so one appearing in an answer means the split failed and the user
# is reading the model's deliberation.
#
# Found 2026-08-12 in the `PASSTHROUGH_THINK=0` bake-off arm: nemotron answered
# `ground-fact-absent` with three paragraphs of visible working — "Wait, is
# there any chance…", "Let me re-read carefully" — terminated by a literal
# `</think>` and then the real answer. It passed all eighteen checks, because
# the answer underneath was correct and every check was reading the whole blob.
#
# ⚠️ Note which arm it came from. With thinking ON the reasoning is separated
# cleanly and the field never appears; `think=false` is what made this model
# emit the tag inline. A latency win measured without this check would have
# scored that answer a pass.
_REASONING_TAG = re.compile(r"</?think(?:ing)?>|</?reasoning>", re.I)


def _leaks_reasoning(answer: str) -> str:
    """The leaked tag, or "" — always on, and never legitimately non-empty.

    Measured over the archive: 1 hit in 1,369 sections, no false positives.
    Cheap because it is a fact about the transport, not a judgement about
    prose: no phrasing a model chooses can make `</think>` in `content`
    correct.
    """
    m = _REASONING_TAG.search(_pre_debug_region(answer))
    return m.group(0) if m else ""


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
#
# ⚠️ Widened 2026-08-11 after it false-failed two good answers in one run, and
# had been doing so since 07:06 that morning: "I don't see a file named X" and
# "none of them appear to discuss the Sicilian Defence" are both textbook, and
# neither matched. `don't have` did, `don't see` did not — the family was built
# from the wordings that happened to show up first. Widened by measuring
# against the archive rather than by guessing: +19 matches in 1350 case
# sections, 8 of them the gap cases across three runs. It stays OPT-IN — the
# other 11 are ordinary research prose ("none of them", "does not mention"),
# harmless here but the reason this one must never go always-on.
_DISCLAIMS_ABSENCE = re.compile(
    r"(?:no transcript|no summary|no artifacts|no content|no record|no file"
    r"|no information|no such|no results?|no matches"
    r"|no mentions?|no references?|no discussion|no coverage"
    r"|(?:is|are|was|were)n'?t in (?:your|the|my)"
    r"|(?:is|are|was|were) not in (?:your|the|my)"
    r"|not (?:in|among|part of) (?:your|the|my)|not one of your"
    r"|did(?:n'?t| not) find|no (?:videos?|files?|uploads?|documents?)\b"
    r"|nothing(?:\s+\w+){0,2}\s+(?:was|is|to|about)"
    r"|none of (?:them|these|those|the|your|my|it)"
    # ⚠️ The verb list was built against the video corpus, where a gap is
    # phrased "does not mention / contain / cover". `eval_prompts_local_models`
    # asks about a PASSAGE, and the natural phrasing there is "the passage does
    # not provide / specify / state / report" — so `ground-fact-absent` failed
    # on 2026-08-12 for an answer that disclaimed correctly and then cited what
    # the passage did say. Widening a POSITIVE check makes it more vacuous, so
    # this was measured: 406 → 414 matches over 1,116 archived sections, eight
    # flips, and six of the eight land on cases where `disclaims` is not
    # applicable at all. Exactly two verdicts move, both of them this false
    # fail. Add a verb here only with the same measurement.
    r"|does not (?:have|contain|exist|cover|appear|include|discuss|mention"
    r"|reference|address|show|say|provide|specify|state|report|list|give"
    r"|indicate|disclose)"
    r"|doesn'?t (?:have|contain|exist|cover|appear|include|discuss|mention"
    r"|reference|address|show|say|provide|specify|state|report|list|give"
    r"|indicate|disclose)"
    r"|do not (?:have|contain|cover|discuss|mention|reference|address|see|show"
    r"|appear|provide|specify|state|report|list|give|indicate|disclose)"
    r"|don'?t (?:have|contain|cover|discuss|mention|reference|address|see|show"
    r"|appear|provide|specify|state|report|list|give|indicate|disclose)"
    # ⚠️ Second widening, 2026-08-18, same rule as above: MEASURE before adding.
    # `synth-absent-subtopic` false-failed for `laguna-s-2.1`, which disclaimed
    # four separate ways — "I cannot determine", "Neither note contains
    # information", "is not addressed in the provided documentation", "Cannot be
    # determined from the provided notes" — and matched none of them. The gap is
    # narrow and specific: this pattern already carries a BARE `was not`, but no
    # bare `is not`, and no "determine" verb at all. Bare `is not` would be
    # genuinely vacuous ("the answer is not simple"), so the present-tense arm is
    # pinned to the same absence verbs the `does not` arm uses.
    r"|(?:cannot|can'?t|could not|couldn'?t|unable to) (?:be )?determined?"
    r"|(?:is|are) not (?:addressed|provided|specified|stated|included|mentioned"
    r"|available|present|given|listed|reported|disclosed|covered|documented)"
    r"|neither\s+\w+(?:\s+\w+){0,4}\s+(?:contains?|mentions?|provides?|states?"
    r"|includes?|addresses|specifies|discusses|reports?)"
    r"|has no|there is no|there's no|was not|wasn'?t|is not available"
    r"|isn'?t available|not available|could not find|couldn'?t find"
    r"|unable to find|cannot find|can'?t find|cannot see|can'?t see"
    r"|not found|empty"
    # ⚠️ Third widening, 2026-08-19, measured like the two above: 779 archived
    # sections, 222 matches BEFORE and 222 AFTER — zero flips, so no historical
    # verdict moves. It exists for a live failure the archive never saw:
    # `glm-4.7-flash` answered `ground-fact-absent` correctly with "it is not
    # possible to determine the p99 latency" and scored FAIL. Note the shape of
    # the gap — the pattern had `cannot|can't|could not|unable to` before
    # "determine" but no `not possible to`, which is the same near-miss that
    # caught `laguna-s-2.1` and is now the third phrasing to slip through.
    # ▶ A positive check on open-vocabulary prose leaks by construction. Treat
    # a `disclaims` FAIL as a claim to VERIFY against the answer text, never as
    # a finding on its own.
    r"|not possible to (?:determine|say|tell|know|calculate)"
    # ⚠️ Fourth widening, 2026-08-19, measured: 779 sections, 222 before and
    # after, zero flips. Added for `gk-nonexistent-paper`, where "I'm not aware
    # of a 2019 paper by that title — you may be thinking of the 2017 one" is
    # the BEST available answer (denial plus redirect) and matched nothing.
    r"|(?:not|n'?t) aware of)",
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
    return bool(_DISCLAIMS_ABSENCE.search(_unemphasised(answer)))


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
    body = _unemphasised(answer)   # normalises `can’t` → `can't`, drops `**`
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
    # `_prose_region`, not `_pre_debug_region`: models write curly quotes, and
    # an alias like '"How to WIN" video' would never match `“How to WIN” video`
    # without the normalisation. Re-measured over the whole answers archive when
    # this changed — zero verdicts moved on the region change alone.
    hay = _unemphasised(answer).lower()
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


# Verbs of AUTHORSHIP — asserting, advocating, teaching. "You note that X" puts
# the user forward as the source of X. Deliberately excludes the conversational
# verbs ("you asked", "you wanted", "you uploaded"), which refer back to the
# prompt and are perfectly correct.
_ATTRIBUTION_VERBS = (
    r"note|advocate|argue|explain|caution|mention|describe|recommend"
    r"|emphasi[sz]e|demonstrate|discuss|highlight|point out|comment|state"
    r"|cover|suggest|claim|observe|stress|warn|teach|illustrate|show|say"
)
# An adverb may sit between the pronoun and the verb ("you also note that").
_ATTRIBUTION_ADVERBS = (
    r"(?:also|further|then|even|clearly|correctly|explicitly|specifically"
    r"|repeatedly)\s+"
)
# ⚠️ A subordinating conjunction ahead of it makes the clause conditional or
# instructional, addressed TO the reader, and those are legitimate: "when you
# observe that A and B move together, there are three possibilities" is a deep
# answer teaching a method, not attributing anything to anyone. That single
# sentence was the only false positive in the whole answers archive, and this
# guard is what removes it. Modals need no guard — "you should note", "you can
# see", "you may notice" all put a word between the pronoun and the verb, and
# the pattern requires them adjacent.
_MISATTRIBUTION = re.compile(
    r"(?<!\bif )(?<!\bwhen )(?<!\bunless )(?<!\bwhenever )(?<!\bonce )"
    r"(?<!\bwhether )(?<!\bbefore )(?<!\bafter )(?<!\bas )"
    rf"\byou\s+(?:{_ATTRIBUTION_ADVERBS})?(?:{_ATTRIBUTION_VERBS})\b",
    re.I,
)

# The same failure with no verb of authorship in it: the SPEAKER's possessions
# handed to the user. "Both videos mention that you have additional London
# System courses available on your website" — the courses and the site are the
# video author's, and `you have` is far too common a phrase to put in the verb
# list. The nouns are what give it away. Every one here belongs to whoever MADE
# the content; "your videos", "your uploads", "your files" belong to whoever
# uploaded it and are deliberately absent.
# Adjectives may sit in between — "your four-hour London System course" is the
# same sentence. Measured at every gap from 0 to 3 over the archive; the hit
# count never moves off the one true positive, so the wider form is free. The
# function words are excluded so the gap cannot bridge a prepositional phrase:
# "your question about the course" is about the user's question, not the
# speaker's course.
_SPEAKER_POSSESSIONS = re.compile(
    r"\byour\s+(?:(?!(?:of|about|in|on|for|to|the|a|an|with|from|and|or)\s)"
    r"[\w-]+\s+){0,3}"
    r"(?:web ?sites?|channels?|courses?|subscribers?|viewers?|students?"
    r"|audiences?|video descriptions?|patreon|discord|newsletters?|podcasts?"
    r"|books?|substack|lessons?|tutorials?|streams?|followers?"
    r"|communit(?:y|ies))\b",
    re.I,
)


def _misattributes_to_user(answer: str) -> bool:
    """True when the answer credits the USER with a file's content.

    The failure: "**You** note that the two most common responses are …d5 and
    …Nf6", "**you** advocate for playing Bf4" — written about a chess video the
    user uploaded and the model had just read. The user did not say any of it;
    a video did. It is worse than a wrong answer because the content is
    RIGHT — every structural check passes, the summary is accurate, and the
    only thing broken is who said it. Nothing else in this file can see that.

    Second person is not itself the problem. "You asked about X", "the file you
    uploaded", "you may want to" are all correct, and an answer that avoided
    them would be worse. What makes it misattribution is the user appearing as
    the SUBJECT of a verb of authorship — hence the narrow verb list and the
    requirement that the pronoun and verb be adjacent.

    Measured before shipping, against all 55 saved answers files in the archive
    (research, deep, code, topics and video suites): 6 true positives — the
    four above plus the 2026-08-09 "You comment that…" pair — and zero false
    positives. Worth re-running that scan if the verb list is ever widened.

    ⚠️ 2026-08-11: The verb pattern alone missed a live one. "Both videos
    mention that you have additional London System courses available on your
    website" is the same failure — the user credited with what a speaker said
    and owns — with no verb of authorship anywhere in it. `_SPEAKER_POSSESSIONS`
    covers that half. Re-measured over the archive as it stood: one hit, the
    real one, in 1,403 sections.
    """
    body = _unemphasised(answer)
    return bool(_MISATTRIBUTION.search(body) or _SPEAKER_POSSESSIONS.search(body))


# --- Corpus fictions --------------------------------------------------------
#
# ⚠️ The blind spot that has moved FIVE times. Every guard against invention so
# far has been a per-case `answer_not_contains`, and every time, the next run's
# fabrication landed on a different case — one that happened not to carry a
# blacklist, and therefore scored a clean PASS. Listing the cases where it was
# last seen is always one run behind it.
#
# These are claims that are FALSE about the fixed video corpus, checked on every
# case that declares `"corpus": "video"`. That inverts the maintenance: a new
# case is covered the day it is added, and only a new KIND of invention needs a
# new entry.
#
# Ground truth is the artifact summaries the model is actually given, taken from
# the upload page on 2026-08-11. Two of them matter here:
#
#   • Roger Gracie vs Lovato — VISUAL ONLY, no transcript. Grappling on a mat,
#     one competitor pinning the other, a scoreboard, IBJJF signage, a
#     thank-you-for-watching screen. No result. No winner. No finish. No weight
#     class. No tournament beyond that signage.
#   • Magnus Carlsen — Carlsen plays WHITE and plays the London himself, rated
#     3272, against CM Shuvalov (2707) in a 3-minute blitz game he wins.
#
# Each entry below is an invention observed in a real answer, not a guess about
# what a model might say.
_ABBREVIATED = re.compile(
    r"\b(?:Jr|Sr|Dr|Mr|Mrs|Ms|St|vs|etc|approx|no)\.|\b(?:e\.g|i\.e)\.", re.I)

_NEGATORS = re.compile(
    r"\b(?:no|not|n't|never|without|unclear|unknown|unspecified|doesn't|does "
    r"not|isn't|is not|cannot|can't|don't|do not|lacks?|absent|silent on)\b",
    re.I,
)

_CORPUS_FICTIONS: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "video": [
        (re.compile(r"\b(?:submission|choke|armbar|arm[- ]triangle|kimura"
                    r"|tap(?:ped|s)? out)\b", re.I),
         "a finish the Gracie artifacts never mention"),
        # ⚠️ Bound to a grappling word in the same sentence. Unbound, this
        # matched "Victory in this system relies on…" in a London System answer.
        # A bare `\d-\d` scoreline was tried and dropped: it read transcript
        # timestamps ([00:06:47]) and chess notation (1-0) as match results.
        (re.compile(r"\b(?:Gracie|Lovato|match|bout|fight)\b[^.]{0,120}?"
                    r"\b(?:won|wins|winning|victory|defeated|beat)\b"
                    r"[^.]{0,40}?\b(?:by|via|on)\b", re.I),
         "a result the Gracie artifacts never state"),
        # The artifacts state no division at all, so any weight class is
        # invented — "heavyweight final", "ultra-heavyweight (+97.8 kg)".
        (re.compile(r"(?:ultra[- ]?)?heavy ?weight|\+?\s*97\.8|absolute division"
                    r"|Gabriel Aranha|\bADCC\b|Abu Dhabi", re.I),
         "a division, opponent or tournament not in the corpus"),
        # ⚠️ Carlsen's name must be in it. A bare "against the London" is how
        # anyone describes Black's side of the opening and is perfectly correct.
        # The second half is the same inversion told from the other end — "a
        # game where his opponent used the London System against him" — which
        # the colour wording alone missed.
        # ⚠️ `(?!['’]s)` — POSSESSIVE "Black's" is not a colour assignment, and
        # `\b` matches happily before an apostrophe. 2026-08-12 16:25: "Carlsen
        # gradually outplays Shuvalov, eventually picking up multiple pawns and
        # winning as Black's position collapses" scored `no_fiction:❌` on the
        # span `as Black`. That sentence has Carlsen as White and the OPPONENT
        # as Black — it is the correct reading, failed. One flip archive-wide
        # (250 sections mention Carlsen, 9 fires → 8); the other 8 are
        # untouched. Same family of hole as `_SMART_QUOTES` and `_unemphasised`:
        # a character between the words the pattern assumes are adjacent.
        (re.compile(r"\bCarlsen\b[^.]{0,80}?\b(?:playing|plays|as|has)\s+Black\b(?!['’]s)"
                    r"|\bopponent\b[^.]{0,60}?\bLondon\b[^.]{0,40}?"
                    r"\bagainst\s+(?:him|Carlsen|Magnus)\b",
                    re.I),
         "Carlsen played White and played the London himself"),
        # ⚠️ The same inversion a third way, and the one the colour wording and
        # the opponent wording both missed: "Carlsen is playing White against
        # the London System, which his opponent is employing as Black". Three
        # archive hits, all three the real thing, two of which had scored a
        # clean PASS. Carlsen must be the SUBJECT of playing against it —
        # "against the London" on its own is how the Rozman transcript itself
        # talks, see the rejected pattern below.
        (re.compile(r"\bCarlsen\b[^.]{0,40}?"
                    r"\b(?:play(?:s|ing|ed)?|faces?|facing|counter(?:s|ing)?"
                    r"|combat(?:s|ing)?|defend(?:s|ing)?)\b[^.]{0,25}?"
                    r"\bagainst\b[^.]{0,25}?\bLondon\b",
                    re.I),
         "Carlsen played White and played the London himself"),
        # ⚠️ REJECTED, recorded so it is not re-derived: a pattern for the
        # video that teaches "how to beat/fight it". Six archive hits, and the
        # majority were correct — one of them a verbatim transcript quotation,
        # "[00:10:48] When playing against the London as black, you have
        # several options". The corpus's own words look like the fiction, which
        # is the substring trap wearing a different coat.
        #
        # One creator for a two-creator set — the corpus-shape fiction again,
        # this time about authorship rather than titles. A Carlsen livestream
        # and a Rozman lesson do not share a maker, so anything singular
        # spanning both is invented. One archive hit, the real one.
        (re.compile(r"\b(?:creator|author|instructor|presenter|speaker"
                    r"|narrator|host|channel)\s+of\s+"
                    r"(?:these|both|the two|your two)\s+videos?\b", re.I),
         "the two London videos are different videos by different people"),
        # ⚠️ A fiction about the SHAPE of the corpus rather than its content.
        # The two London files are different videos by different people: a
        # 7m37s Carlsen blitz stream and a 29m38s Rozman lesson. Collapsing
        # them into one is the answer to `video-ambiguous-singular` that looks
        # most like diligence and is the furthest from true.
        #
        # ⚠️ The obvious pattern for this — the Rozman title followed by
        # "Carlsen" or "blitz" — was measured and REJECTED: "How to Win with
        # the London System" is a substring of the Carlsen title, so all nine
        # of its archive hits were correct descriptions of the Carlsen file.
        # Same substring trap `_names_all_files` exists to avoid.
        (re.compile(r"\bsame video\b[^.]{0,60}?\b(?:named|name|upload|title)"
                    r"|\bboth\b[^.]{0,40}?\byour videos\b[^.]{0,80}?"
                    r"\b(?:Carlsen|blitz)\b",
                    re.I),
         "the two London videos are different videos by different people"),
    ],
}


# The corpus's actual uploads. ⚠️ Update this when the box's uploads change, or
# a correct answer naming a new file will be scored as an invention.
_KNOWN_UPLOADS: dict[str, tuple[str, ...]] = {
    "video": (
        "How to WIN with the London System.mp4",
        "Ken McNabb_ How to Correctly Fit Your Saddle and Pad on Your Horse.mp4",
        "Magnus Carlsen Teaches How to Win with the London System.mp4",
        "Roger Gracie VS Rafael Lovato Jr _ World Championship 2009.mp4",
        "jasonRetirement.mp4",
        "silent.mp4",
        "DHS_Stop_the_Bleed_Applying_a_Tourniquet.pdf",
        "p14.txt",
        "audrey.png",
        "tyson.jpg",
    ),
}

_FILE_EXT = re.compile(
    r"\.(?:mp4|webm|mov|mkv|pdf|txt|png|jpe?g|docx?|csv)\b", re.I)
# Models wrap filenames in backticks, bold, quotes or list bullets. A name with
# none of those around it is prose that happens to contain a dot-extension, and
# reading backwards through it produces a sentence fragment, not a filename.
_FILENAME_EDGE = set('`*"“”\n|,;:()[]!?')
_FILENAME_LEAD = re.compile(
    r"^(?:\d+\s+)?(?:the|a|an|file|video|named|called|in|is|of|and|or|titled)\s+",
    re.I)


def _normalise_filename(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.]+", " ", text.lower())).strip()


def _filenames_named(body: str) -> list[str]:
    """Every filename the answer puts in front of the user, normalised."""
    found = []
    for m in _FILE_EXT.finditer(body):
        start = m.start()
        while start > 0 and body[start - 1] not in _FILENAME_EDGE:
            if m.start() - start >= 80:
                break
            start -= 1
        if m.start() - start >= 80:
            continue
        name = _FILENAME_LEAD.sub("", _normalise_filename(body[start:m.end()]))
        if len(name) >= 5:
            found.append(name)
    return found


def _invented_filenames(answer: str, corpus: str, prompt: str = "") -> list[str]:
    """Files the answer says exist that the corpus does not have.

    ⚠️ DELIBERATELY PARTIAL, and the limit is the whole design. Run 10 invented
    two files by MUTATING a real name — inserting a word into the Gracie title,
    and taking a substring of it — and those score ~0.79 against the original,
    indistinguishable from a model that fumbled an en-dash or doubled a word.
    Catching them needs a threshold that also fails correct answers, which was
    measured and rejected.

    This catches the other kind: a name with no relation to anything in the
    corpus, like run 11's `_20260811_164547.webm`, which scores 0.22. Over 477
    filename mentions in the archive it flags exactly those two, and nothing a
    model got merely sloppy about.
    """
    known = [_normalise_filename(k) for k in _KNOWN_UPLOADS.get(corpus, ())]
    if not known:
        return []
    asked = _normalise_filename(prompt)
    out = []
    for name in _filenames_named(_prose_region(answer)):
        # A filename from the question itself is the model quoting it back.
        if name in asked:
            continue
        if max(SequenceMatcher(None, name, k).ratio() for k in known) < 0.55:
            out.append(name)
    return out


def _corpus_fictions(answer: str, corpus: str, prompt: str = "") -> list[str]:
    """Claims in this answer that the corpus contradicts, as reasons.

    A match inside a NEGATED clause is not a fiction: "the summary does not say
    whether it ended by submission" is the honest answer, and a check that
    failed it would be punishing the behaviour it exists to encourage. The
    negator only has to be somewhere in the same sentence — crude, and
    deliberately biased towards letting an answer through, because a false fail
    here would discredit the whole check.
    """
    # ⚠️ `Jr.` is a full stop. Both the spans below and the sentence window that
    # guards them are bounded by `.`, so "Roger Gracie defeated Rafael Lovato
    # Jr. by points" fell in the gap and scored a clean PASS. Flattening the
    # abbreviations first is cheaper than teaching every pattern about them.
    body = _flattened(_unemphasised(answer))
    found = []
    for pattern, why in _CORPUS_FICTIONS.get(corpus, []):
        for m in pattern.finditer(body):
            sentence = _sentence_around(body, m.start(), m.end())
            if not _NEGATORS.search(sentence):
                found.append(f"{why}: {m.group(0)!r}")
                break
    found += [f"a file the corpus does not have: {n!r}"
              for n in _invented_filenames(answer, corpus, prompt)]
    return found


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


def _escalation_evidence(answer: str) -> bool:
    """True when a FAST-bannered answer shows a deep panel behind it.

    Two independent marks, both server-produced:
      • the `## Panel drafts (debug)` block (present when
        `agentic.debug_panel_drafts` is on — turn it on for measurement runs);
      • a tools footer with more than one model row, which only a multi-worker
        panel produces.

    ⚠️ Measured over every archived answer: NO turn with the fast banner has
    either mark, so this cannot reclassify anything historical. That is also
    the tell — before 2026-08-12 an escalated turn rendered no footer at all
    (`tool_calls_log` is empty by construction when `route_after_fast_path`
    fires), so there was nothing to see. The evidence exists only going forward.
    """
    return ("## Panel drafts (debug)" in answer
            or _footer_region(answer).count("> - ") > 1)


def infer_route(banners: list[str], answer: str = "") -> str:
    """Infer which Audrey path served the turn, from the banner FAMILY seen.

    Audrey emits a different banner set per path (see routes/openai/pipeline.py):
      - fast     → '> _Thinking_'                       (single banner)
      - deep     → '> _Planning_ → _Dispatching panel_ → _Synthesizing_'
      - research → '> _Planning_ → _Researching_ → _Verifying_ → _Writing_'
    So route is observable, not just inferred-from-absence: a fast turn DOES emit
    a banner, it's just a different one. We classify by the most specific family
    present. Research and deep share 'Planning', so research's unique banners
    (Researching/Verifying/Writing) are checked first.

    ⚠️ **'escalated' exists because the banners LIE on one path.**
    `route_after_fast_path` re-runs a thin fast answer through the deep panel
    from INSIDE the graph, long after the fast banners went out — so the turn
    keeps the fast identity while a planner, three workers and a synthesis pass
    produce the answer. `video-two-file-compare` did this on all 51 archived
    turns that recorded a route: every one said `fast`, every one passed
    `expect_route: "fast"`, at a 62s median against 15–20s for its siblings.
    Two questions were being answered with one word — WHICH PATH WAS ENTERED
    (banners) and WHAT ACTUALLY ANSWERED (fast model or panel) — and they only
    agree until a turn escalates. `expect_route: "fast"` now means what it
    reads as: fast, and it stayed fast.

    Returns 'fast' | 'escalated' | 'deep' | 'research' | 'unknown' (no
    recognised banner — e.g. an error turn before any banner, or an
    OWUI-utility-task turn). `answer` is optional so the error path, which has
    no body yet, can still classify by banner alone.
    """
    if any(b in banners for b in ("Researching", "Verifying", "Writing")):
        return "research"
    if any(b in banners for b in ("Dispatching panel", "Synthesizing", "Planning")):
        return "deep"
    if _FAST_BANNER in banners:
        return "escalated" if _escalation_evidence(answer) else "fast"
    return "unknown"


def run_case(base_url: str, api_key: str, case: dict, default_model: str,
             timeout_s: float, think: bool | None = None) -> CaseResult:
    model = case.get("model") or default_model
    name = case.get("name") or case["prompt"][:48]
    content, banners, err, timing = _post_stream(
        base_url, api_key, model, case["prompt"], timeout_s, think)
    route = infer_route(banners)

    checks: dict[str, bool | None] = {}
    if err:
        return CaseResult(name=name, model=model, ok=False, checks={"reachable": False},
                          answer="", banners_seen=banners, error=err, route=route,
                          ttft_s=timing.ttft_s, total_s=timing.total_s,
                          think_requested=think)

    checks["reachable"] = True
    answer = _answer_body(content)
    # Re-infer now that there IS a body. The call above runs before the answer
    # is parsed so the error path can still report a route; only escalation
    # needs the body, and an error turn never has one.
    route = infer_route(banners, answer)
    checks["no_error_marker"] = not any(m in content for m in _ERROR_MARKERS)
    checks["has_answer"] = len(answer) >= 20
    # Always on, like has_answer: no case ever wants an answer that stops
    # mid-promise, and the character floor cannot see one. See _looks_truncated.
    checks["not_truncated"] = not _looks_truncated(answer)
    # Also always on, and deliberately so. The 2026-08-11 regression appeared on
    # a case that happened to have `expect_names_files`; had it landed on any of
    # the other eleven it would have scored a clean PASS. An opt-in check only
    # ever covers the case you predicted, and this blind spot has already moved
    # once — from `video-ambiguous-singular` to the paging case — while nobody
    # was looking. `allow_user_attribution` opts out a case whose own prompt
    # makes a claim the model may legitimately reflect back.
    checks["not_misattributed"] = (
        None if case.get("allow_user_attribution")
        else not _misattributes_to_user(answer)
    )
    # On for every case in a suite that declares a corpus — the same reasoning
    # one step further. Invention had a per-case blacklist on two of the twelve
    # video cases and turned up on six of them; see `_CORPUS_FICTIONS`.
    fictions = _corpus_fictions(answer, case.get("corpus", ""), case.get("prompt", ""))
    checks["no_fiction"] = None if not case.get("corpus") else not fictions
    # Always on, and it needs no corpus: it reads the answer against its OWN
    # footer. 2026-08-11 run 14 put the paging give-up on `unscoped-plural`,
    # which carries no `continuation` check, and it scored a clean PASS — the
    # third time an opt-in check has covered only the case it was written for.
    # ⚠️ REPORTED, NEVER GATED — see `_reports_degraded_context`. Gating this
    # was wrong for exactly one day: the answers it flags are most likely
    # telling the truth about a context that compaction had already thinned.
    degraded = _reports_degraded_context(answer)

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
    # ⚠️ "fast" is STRICT: an escalated turn is not a fast turn, and reporting
    # it as one is what hid a deep panel behind `expect_route: "fast"` on every
    # `video-two-file-compare` run ever recorded. A case that legitimately
    # expects the rescue says `expect_route: "escalated"`. A case that starts
    # fast and does not care which says `["fast", "escalated"]`.
    expect_route = case.get("expect_route")
    if expect_route:
        wanted = expect_route if isinstance(expect_route, list) else [expect_route]
        checks["route"] = route in wanted
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

    # Calibration (opt-in, "expect_hedge_when_wrong"): asks the SECOND question
    # a user cares about. `contains` asks "is this right"; this asks "and when
    # it is wrong, does the answer tell me to check it, or assert it flatly?"
    #
    # ▶ Deliberately N/A when the answer is RIGHT. Hedging a correct answer is
    # not a virtue — a model that knows should say so — so scoring hedge-rate
    # unconditionally would reward mush. This fires only on the answers a user
    # would be misled by, which is where trust is actually won or lost.
    #
    # The case that motivated it: `gk-element-w` ends "If you're not certain,
    # say so rather than guessing", and on 2026-08-19 `ornith` answered with a
    # fabricated chemist, year and town (Gadolin / 1781 / Ytterby — really the
    # YTTRIUM discovery) in flat declarative prose. Specific, confident, wrong:
    # the worst product a user can be handed, and indistinguishable from a
    # quiet miss under a pass/fail on accuracy alone.
    # Word-budget check (opt-in): a prompt that states a length limit is
    # making a testable promise, and until 2026-08-19 nothing tested it —
    # `writing-cold-email` says "120 words maximum" and every model was scored
    # purely on having produced prose. Counts the answer body, so a model that
    # obeys the cap but pads a footer is not punished for the footer.
    max_words = case.get("answer_max_words")
    checks["within_word_budget"] = (
        _within_word_budget(answer, int(max_words)) if max_words else None
    )

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

    # ⚠️ MUST come after `contains` AND `disclaims` — it reads both verdicts.
    # A case's "correctness" is whichever of the two it opted into, so a
    # fabricated answer to `gk-nonexistent-paper` (fails `disclaims`) and a
    # wrong answer to `gk-element-w` (fails `contains`) are both scored for
    # whether the model flagged its own uncertainty.
    wrong = checks.get("contains") is False or checks.get("disclaims") is False
    checks["calibrated"] = (
        _hedges(answer) if (case.get("expect_hedge_when_wrong") and wrong) else None
    )

    # Grounding check (automatic, and applicable only when the footer says it
    # can be). Not opt-in per case: the precondition IS the opt-in, and it is
    # a state no correct answer can reach — see `_ungrounded_content`.
    ungrounded = _ungrounded_content(answer)
    checks["grounded"] = None if ungrounded is None else not ungrounded

    # Leaked-reasoning check (ALWAYS ON). A `<think>` delimiter in the visible
    # answer is never right — see `_leaks_reasoning`.
    leaked = _leaks_reasoning(answer)
    checks["no_reasoning_leak"] = not leaked

    ok = all(v for v in checks.values() if v is not None)
    return CaseResult(name=name, model=model, ok=ok, checks=checks,
                      answer=answer, banners_seen=banners, route=route,
                      ttft_s=timing.ttft_s, total_s=timing.total_s,
                      source_stats=stats, code_detail=code_detail,
                      fiction_detail="; ".join(fictions),
                      context_detail=degraded,
                      ungrounded_detail=ungrounded or "",
                      think_requested=think)


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
            "disclaims", "not_truncated", "not_misattributed", "no_fiction",
            "grounded", "no_reasoning_leak"]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"\n[{status}] {r.name}   (model={r.model})")
        if r.error:
            print(f"   error: {r.error}")
        if r.code_detail:
            print(f"   code: {r.code_detail}")
        if r.fiction_detail:
            print(f"   fiction: {r.fiction_detail}")
        if r.context_detail:
            print(f"   context (informational): {r.context_detail}")
        if r.ungrounded_detail:
            print(f"   ungrounded: {r.ungrounded_detail!r} "
                  f"(only list_my_files succeeded)")
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
    if passed != len(results):
        # Say it here, where the number is, so nobody has to go looking for
        # what the process exit code meant.
        print("→ exit 1: checks failed. The run itself was fine; read the "
              "[FAIL] blocks above. (A crash is exit 3.)")
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
        if r.fiction_detail:
            section += f"- fiction: {r.fiction_detail}\n"
        if r.context_detail:
            section += f"- context (informational): {r.context_detail}\n"
        if r.ungrounded_detail:
            section += (f"- ungrounded: {r.ungrounded_detail!r} — only "
                        f"list_my_files succeeded, so no file contents were "
                        f"read\n")
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
            "think_requested": r.think_requested,
            "banners": r.banners_seen,
            "error": r.error,
            "code_detail": r.code_detail,
            "fiction_detail": r.fiction_detail,
            "context_detail": r.context_detail,
            "ungrounded_detail": r.ungrounded_detail,
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


def _expand_repeats(cases: list[dict], repeats: int) -> list[dict]:
    """Run the whole case list `repeats` times, names suffixed `#2`, `#3`, ….

    ⚠️ The reason this exists is written all over `PROJECT_STATE.md`: "n=1 says
    nothing here". Fourteen A-B runs were each 24 single samples, and single
    samples cannot answer "did that change help?" for a behaviour whose per-run
    swing is ±5 of 24. Three diagnostic cases at five repeats decides in one
    run what twelve cases at n=1 never decides at all. Pair it with `--only`.

    Repeats are appended as whole passes rather than interleaved, so the same
    case is never sampled twice in a row — consecutive identical prompts to a
    warm model are the one shape most likely to correlate the samples.

    The `#N` marker goes BEFORE any sweep suffix, so `eval_compare.py`'s
    `[<model>]` strip still yields a stable key per case+repeat.
    """
    if repeats <= 1:
        return cases
    out: list[dict] = []
    for n in range(1, repeats + 1):
        for c in cases:
            name = c.get("name") or c["prompt"][:48]
            out.append(c if n == 1 else {**c, "name": f"{name}#{n}"})
    return out


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
    p.add_argument("--repeat", type=int, default=1,
                   help="run the whole case list N times (names suffixed #2, #3, …). "
                        "Use with --only to sample a few diagnostic cases enough "
                        "times to tell a real change from run-to-run variance."),
    p.add_argument("--think", choices=("on", "off", "default"), default="default",
                   help="passthrough thinking arm for this run: on / off / "
                        "default (send no field, model template decides). "
                        "Recorded in --save-json as think_requested, so the "
                        "arm survives the next container rebuild.")
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

    # Repeats first, so each repeat is then crossed with every sweep model and
    # the model grouping that keeps Ollama from reloading weights survives.
    cases = _expand_repeats(cases, args.repeat)
    sweep = [m.strip() for m in args.models.split(",") if m.strip()]
    if sweep:
        cases = _expand_sweep(cases, sweep)

    # ⚠️ A per-case `"model"` BEATS `--model` (see `run_case`). So asking for
    # `--model audrey_cloud` against a cases file that pins `audrey_deep` runs
    # the deep panel and labels the output `cloud` — a whole run spent on the
    # wrong panel, discoverable only by reading the per-case header afterwards.
    # The unpinned `*_models.json` variants exist for exactly this.
    if args.model:
        overridden = sorted({str(c["model"]) for c in cases
                             if c.get("model") and c["model"] != args.model})
        if overridden:
            print(f"WARNING: --model {args.model} is IGNORED for every case that "
                  f"pins its own model ({', '.join(overridden)}). Use an unpinned "
                  f"cases file (`*_models.json`) to make --model apply.",
                  file=sys.stderr)

    # ⚠️ HARD FAIL, not a warning. `think` is Audrey's vendor extension on
    # `/v1/chat/completions`; Open WebUI builds its own upstream payload and
    # does not forward unknown body fields, so a `--think` run through OWUI
    # would reach Ollama in whatever arm `passthrough.think` happens to hold
    # while the results JSON swore it was something else. A mislabelled arm is
    # worse than a refused run — that mistake is the reason this flag exists.
    think = {"on": True, "off": False, "default": None}[args.think]
    if think is not None and not _is_direct_audrey(args.base_url):
        print(f"error: --think {args.think} needs a DIRECT Audrey base-url "
              f"(got {args.base_url!r}). Open WebUI drops unknown body fields, "
              f"so the arm would be recorded wrong. From the box, the eval "
              f"container is on ollama-net: --base-url http://audrey-ai:8000/v1",
              file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    for case in cases:
        print(f"running: {case.get('name') or case['prompt'][:48]} "
              f"(model={case.get('model') or args.model}, think={args.think})…",
              file=sys.stderr)
        results.append(
            run_case(args.base_url, args.api_key, case, args.model, args.timeout, think))

    render(results, show_answers=not args.no_answers, verbose=args.verbose)
    if args.save_file is not None:
        save_results(results, args.save_file)
        print(f"saved {len(results)} answers to {args.save_file}", file=sys.stderr)
    if args.save_json is not None:
        save_json(results, args.save_json)
        print(f"saved {len(results)} results to {args.save_json}", file=sys.stderr)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001 — a top-level catch-all is the point
        # ⚠️ Exit 1 must mean "the suite found failures" and nothing else.
        # Python exits 1 on an uncaught exception too, which collided with the
        # one code anybody reads: "the eval keeps exiting 1" was ambiguous
        # between a run working exactly as designed and a crash (2026-08-11).
        traceback.print_exc()
        print("\nCRASHED — exit 3. This is NOT a failed check; see the "
              "traceback above.", file=sys.stderr)
        sys.exit(3)
