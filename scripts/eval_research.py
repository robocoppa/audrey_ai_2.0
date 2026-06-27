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

(`--base-url` / `--api-key` flags override env / .env. If you instead want
to hit Audrey directly, point `--base-url` at `http://192.168.1.11:8000/v1`
and pass a valid OWUI JWT as `--api-key` — but JWTs expire, so the OWUI path
is the repeatable one.)

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

Plus, for every case, latency is recorded: TTFT (first content delta) and
total wall-clock. The fast path's reason to exist is speed, so these make
"fast" falsifiable run-to-run. They print under each case and land in the
saved answers file; they are measurements, not pass/fail checks.

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

EXIT CODE

  0 if every case passed every check; 1 otherwise (so it can gate a script).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
    ttft_s: float | None = None
    total_s: float | None = None


@dataclass
class StreamTiming:
    """Wall-clock measurements for one streamed turn (seconds; None if N/A)."""
    ttft_s: float | None = None   # time to first content delta (banner or token)
    total_s: float | None = None  # request start → stream end


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
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    content_parts: list[str] = []
    banners: list[str] = []
    # Fast emits 'Thinking'; include it so a fast turn's banner is captured and
    # route inference can classify it (see infer_route).
    seen_banner_phrases = [*_RESEARCH_BANNERS, _FACTCHECK_BANNER, _FAST_BANNER, *_DEEP_BANNERS]
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
                for phrase in seen_banner_phrases:
                    if phrase in text and phrase not in banners:
                        banners.append(phrase)
    except httpx.HTTPError as e:
        timing.total_s = time.monotonic() - t0
        return "", [], f"{type(e).__name__}: {e}", timing
    timing.total_s = time.monotonic() - t0
    return "".join(content_parts), banners, "", timing


def _answer_body(full_content: str) -> str:
    """The text after the last banner separator — the actual answer prose."""
    if _SEPARATOR in full_content:
        return full_content.rsplit(_SEPARATOR, 1)[1].strip()
    return full_content.strip()


def _sources_block(answer: str) -> str:
    """Return the '## Sources' section text, or '' if absent."""
    low = answer.lower()
    idx = low.rfind("## sources")
    return answer[idx:] if idx != -1 else ""


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

    # Route expectation (opt-in): for audrey_auto cases, assert the inferred
    # path matches the intended one — this is how we test the fast/deep gate
    # (token_threshold + deep_intent_phrases). Inferred from the banner family
    # (see infer_route); honest about being inference, not a server-truth signal.
    expect_route = case.get("expect_route")
    if expect_route:
        checks["route"] = route == expect_route
    else:
        checks["route"] = None

    ok = all(v for v in checks.values() if v is not None)
    return CaseResult(name=name, model=model, ok=ok, checks=checks,
                      answer=answer, banners_seen=banners, route=route,
                      ttft_s=timing.ttft_s, total_s=timing.total_s)


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
            "sources", "url_wellformed", "route"]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"\n[{status}] {r.name}   (model={r.model})")
        if r.error:
            print(f"   error: {r.error}")
        line = "   " + "  ".join(f"{c}:{_fmt_check(r.checks.get(c))}" for c in cols)
        print(line)
        print(f"   {_fmt_latency(r)}")
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
        if r.error:
            section += f"- error: {r.error}\n"
        section += f"\n{r.answer or '(no answer body)'}\n"
        parts.append(section)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    save_file.write_text("\n".join(parts))


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

    results: list[CaseResult] = []
    for case in cases:
        print(f"running: {case.get('name') or case['prompt'][:48]} "
              f"(model={case.get('model') or args.model})…", file=sys.stderr)
        results.append(run_case(args.base_url, args.api_key, case, args.model, args.timeout))

    render(results, show_answers=not args.no_answers, verbose=args.verbose)
    if args.save_file is not None:
        save_results(results, args.save_file)
        print(f"saved {len(results)} answers to {args.save_file}", file=sys.stderr)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
