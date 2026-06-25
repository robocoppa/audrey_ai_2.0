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

    export AUDREY_EVAL_BASE_URL="http://192.168.1.11:3000/api"   # OWUI host:port + /api
    export AUDREY_EVAL_API_KEY="sk-..."                          # OWUI API key

(`--base-url` / `--api-key` flags override the env vars. If you instead want
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
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: uv sync --extra dev", file=sys.stderr)
    sys.exit(2)


DEFAULT_CASES = Path(__file__).parent / "eval_prompts.json"
# Banners each mode emits, in order. Substrings (the live text is markdown
# blockquote, e.g. '> _Researching_'); we match the inner word.
_DEEP_BANNERS = ["Planning", "Dispatching panel", "Synthesizing"]
_RESEARCH_BANNERS = ["Planning", "Researching", "Verifying", "Writing"]
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


def _post_stream(base_url: str, api_key: str, model: str, prompt: str,
                 timeout_s: float) -> tuple[str, list[str], str]:
    """Stream a chat completion. Returns (full_content, banner_words, error).

    `full_content` is every delta concatenated (banners + answer, as the user
    would see). `banner_words` is the ordered list of banner phrases detected.
    `error` is non-empty on a transport/HTTP failure (then content is "").
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
    seen_banner_phrases = (_RESEARCH_BANNERS + _DEEP_BANNERS)
    try:
        with httpx.Client(timeout=timeout_s) as client, \
             client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 300:
                resp.read()
                return "", [], f"HTTP {resp.status_code}: {resp.text[:300]}"
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
                content_parts.append(text)
                for phrase in seen_banner_phrases:
                    if phrase in text and phrase not in banners:
                        banners.append(phrase)
    except httpx.HTTPError as e:
        return "", [], f"{type(e).__name__}: {e}"
    return "".join(content_parts), banners, ""


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


def run_case(base_url: str, api_key: str, case: dict, default_model: str,
             timeout_s: float) -> CaseResult:
    model = case.get("model") or default_model
    name = case.get("name") or case["prompt"][:48]
    content, banners, err = _post_stream(base_url, api_key, model, case["prompt"], timeout_s)

    checks: dict[str, bool | None] = {}
    if err:
        return CaseResult(name=name, model=model, ok=False, checks={"reachable": False},
                          answer="", banners_seen=banners, error=err)

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

    ok = all(v for v in checks.values() if v is not None)
    return CaseResult(name=name, model=model, ok=ok, checks=checks,
                      answer=answer, banners_seen=banners)


def _fmt_check(v: bool | None) -> str:
    return "—" if v is None else ("✅" if v else "❌")


def render(results: list[CaseResult], *, show_answers: bool, verbose: bool) -> None:
    print("=" * 70)
    print("audrey research/deep live eval")
    print("=" * 70)
    cols = ["reachable", "no_error_marker", "has_answer", "banners",
            "sources", "url_wellformed"]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"\n[{status}] {r.name}   (model={r.model})")
        if r.error:
            print(f"   error: {r.error}")
        line = "   " + "  ".join(f"{c}:{_fmt_check(r.checks.get(c))}" for c in cols)
        print(line)
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


def main() -> int:
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
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
