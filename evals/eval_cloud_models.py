#!/usr/bin/env python3
"""Collect native Ollama replies, then score the saved replies offline."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import httpx
from eval_compare import build_table
from eval_research import (
    StreamTiming,
    _expand_repeats,
    _expand_sweep,
    evaluate_case,
    render,
    save_json,
    save_results,
)

SUITE = Path(__file__).with_name("cases") / "eval_prompts_cloud_models.json"
CANDIDATES = ("kimi-k3.1-pro:cloud", "hy4:cloud")
BASELINES = (
    "kimi-k3:cloud",
    "glm-5.3:cloud",
    "deepseek-v4-pro:cloud",
    "minimax-m3.1-flash:cloud",
)
MODELS = CANDIDATES + BASELINES


def inventory(client: httpx.Client) -> dict[str, str]:
    response = client.get("/api/tags", timeout=10)
    response.raise_for_status()
    tags = {entry["name"]: entry.get("digest") or "" for entry in response.json()["models"]}
    missing = [tag for tag in MODELS if tag not in tags]
    if missing:
        raise ValueError(f"missing Ollama tags: {', '.join(missing)}")
    return {tag: tags[tag] for tag in MODELS}


def fetch(client: httpx.Client, model: str, prompt: str) -> dict:
    """Record visible content and native stream termination without retaining thinking."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    timing = StreamTiming()
    tokens: dict = {}
    parts: list[str] = []
    error = ""
    start = time.monotonic()
    try:
        with client.stream("POST", "/api/chat", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                frame = json.loads(line)
                if frame.get("error"):
                    error = f"Ollama stream error: {str(frame['error'])[:240]}"
                    break
                content = (frame.get("message") or {}).get("content") or ""
                if content:
                    if not isinstance(content, str):
                        raise ValueError("assistant content is not text")
                    if timing.ttft_s is None:
                        timing.ttft_s = time.monotonic() - start
                    parts.append(content)
                if frame.get("done") is True:
                    timing.done_reason = frame.get("done_reason") or ""
                    tokens = {
                        "prompt_tokens": frame.get("prompt_eval_count"),
                        "output_tokens": frame.get("eval_count"),
                    }
                    break
            else:
                error = "Ollama stream ended before its done frame"
    except httpx.HTTPStatusError as exc:
        error = f"HTTP {exc.response.status_code} from Ollama"
    except (httpx.RequestError, ValueError, TypeError, AttributeError) as exc:
        error = f"Ollama stream {type(exc).__name__}: {str(exc)[:240]}"
    timing.total_s = time.monotonic() - start
    return {"answer": "".join(parts), "error": error, "timing": asdict(timing), **tokens}


def collect(args: argparse.Namespace) -> int:
    if args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--repeat must be positive and --timeout must be positive and finite")
    cases = json.loads(SUITE.read_text())
    filter_text = args.only.lower()
    cases = [case for case in cases if filter_text in case["name"].lower()]
    if not cases:
        raise ValueError(f"no cases match --only {args.only!r}")
    cases = _expand_repeats(cases, args.repeat)
    requests = _expand_sweep(cases, MODELS)
    with httpx.Client(base_url=args.ollama_url, timeout=args.timeout, trust_env=False) as client:
        tags = inventory(client)
        for tag, digest in tags.items():
            print(f">> {tag} ({digest or 'no digest reported'})")
        print(f">> inventory OK · {len(requests)} planned requests · provider defaults", flush=True)
        if args.check:
            return 0
        # Flush each response so an interrupted collection still leaves the raw evidence.
        with args.capture.open("x") as out:
            out.write(json.dumps({"cases": cases, "tags": tags}) + "\n")
            for case in requests:
                print(f"running: {case['name']}…", file=sys.stderr)
                response = fetch(client, case["model"], case["prompt"])
                out.write(json.dumps({"name": case["name"], **response}) + "\n")
                out.flush()
    return 0


def comparison(records: list[dict], tags: dict[str, str]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["model"]].append(record)
    lines = [
        "# Kimi K3.1 Pro and Hy4 cloud bake-off",
        "",
        "Each model ran the same text prompts through Ollama using its provider defaults.",
        "TTFT measures the first visible answer token. The matrix includes service errors.",
        "",
        "| Model | Role | Completed streams | Python assertion passes / checked | Ollama tag digest |",
        "|---|---|---:|---:|---|",
    ]
    for model, rows in groups.items():
        completed = [row for row in rows if not row["error"]]
        python_checks = [
            row["checks"]["code_runs"] for row in completed
            if row["checks"].get("code_runs") is not None
        ]
        role = "Candidate" if model in CANDIDATES else "Baseline"
        code = f"{sum(python_checks)}/{len(python_checks)}" if python_checks else "—"
        lines.append(
            f"| `{model}` | {role} | {len(completed)}/{len(rows)} | {code} "
            f"| `{tags[model] or 'unavailable'}` |"
        )
    lines += [
        "",
        "Read the saved answers for proof validity, evidence handling, factual accuracy, "
        "and writing quality. The automated checks measure only the criteria in each case.",
        "",
        build_table(records),
    ]
    return "\n".join(lines)


def score(args: argparse.Namespace) -> int:
    with args.capture.open() as capture:
        manifest = json.loads(capture.readline())
        replies = [json.loads(line) for line in capture]
    tags = manifest["tags"]
    cases = _expand_sweep(manifest["cases"], list(tags))
    if [case["name"] for case in cases] != [reply["name"] for reply in replies]:
        raise ValueError("capture is incomplete or does not match the run manifest")
    results = []
    for case, reply in zip(cases, replies, strict=True):
        timing = StreamTiming(**reply["timing"])
        result = evaluate_case(
            case, case["model"], reply["answer"], [], reply["error"], timing, direct_ollama=True,
        )
        result.ollama = {
            "tag_digest": tags[result.model] or None,
            "done_reason": timing.done_reason or None,
            "prompt_tokens": reply.get("prompt_tokens"),
            "output_tokens": reply.get("output_tokens"),
        }
        results.append(result)
    render(results, show_answers=False, verbose=False, title="Audrey cloud model bake-off")
    save_results(results, args.answers)
    save_json(results, args.results)
    args.compare.write_text(comparison(json.loads(args.results.read_text()), tags))
    return 0 if all(result.ok for result in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--ollama-url", default="http://ollama:11434")
    collector.add_argument("--capture", type=Path, required=True)
    collector.add_argument("--only", default="", help="case name substring")
    collector.add_argument("--repeat", type=int, default=1)
    collector.add_argument("--timeout", type=float, default=600)
    collector.add_argument("--check", action="store_true", help="check inventory without inference")
    scorer = commands.add_parser("score")
    for option in ("capture", "answers", "results", "compare"):
        scorer.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        try:
            return collect(args)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            print(f"error: cloud bake-off collection: {exc}", file=sys.stderr)
            return 2
    return score(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001 — reserve exit 3 for unexpected harness failures
        traceback.print_exc()
        sys.exit(3)
