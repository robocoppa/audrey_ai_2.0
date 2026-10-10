#!/usr/bin/env python3
"""Compare Ollama System One decision models with Audrey's current router.

This is Campaign 4 Phase 6A: measurement only. It calls the incumbent through
Audrey's real ``router_classify`` function and calls decision models through
Ollama's typed ``/v1/systemone`` endpoint. It does not edit ``config.yaml`` or
change production classification.

The probe deliberately unloads each model before its cold sample, so run it
while the box is otherwise idle. Candidates run first and the incumbent runs
last, leaving Audrey's current router warm when the probe finishes. Residency
snapshots show displacement of already-loaded models; they do not prove what
happens during concurrent generation.

On the Unraid checkout, after the candidate model is installed:

    scripts/probes/probe-onbox.sh systemone_router_probe.py \
      COPY=systemone_router_cases.json CANDIDATES=tev1:0.8b

No upload or browser action is involved. The default is one warm pass over 36
unique cases after one cold sample. Add ``ROUNDS=3`` to measure run-to-run
variance, or ``CANDIDATES=tev1:0.8b,tev1:4b`` to compare both candidates.

Environment:
  OLLAMA             Ollama base URL (default http://ollama:11434)
  CANDIDATES          comma-separated System One tags (default tev1:0.8b)
  INCUMBENT_MODEL     chat router tag (default from config, then qwen3.5:4b)
  ROUNDS              warm passes over every case (default 1)
  TIMEOUT             per-call seconds (default from config, then 20)
  KEEP_ALIVE          model residency after each call (default 10m)
  MIN_WINNER          proposed decision winner floor (default 0.55)
  MIN_MARGIN          proposed first-to-second margin floor (default 0.15)
  CASE_FILE           alternate labeled JSON fixture
  REPORT_PATH         optional path for the complete JSON report

Clef and Clef Flash require Ollama 0.35.1 or later; older System One models
retain the endpoint's 0.35.0 minimum. The proposed winner/margin floors are
reported, not shipped. System One's
``confidence`` measures distribution concentration; this probe keeps it
separate from Audrey's self-reported chat confidence and 0.95 escalation gate.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

TASKS = ("code", "reasoning", "general", "vl")
LIVE_EXPECTED_TASKS = frozenset({"code", "reasoning", "general"})
MIN_OLLAMA_VERSION = (0, 35, 0)
CLEF_MIN_OLLAMA_VERSION = (0, 35, 1)
CLEF_MODEL_FAMILIES = frozenset({"clef", "clef-flash"})

ROUTER_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": "Which Audrey task family should handle this user request?",
    "criteria": {
        "code": (
            "The user wants software code written, debugged, refactored, migrated, "
            "optimized, or explained line by line."
        ),
        "reasoning": (
            "The user wants analysis, comparison, review, multi-step logic, a proof, "
            "a calculation, or a detailed explanation."
        ),
        "general": (
            "The user wants conversation, facts, summaries, drafting, translation, "
            "or anything else, including questions about previously uploaded files."
        ),
        "vl": (
            "The user attached an image in this turn and wants its visible contents "
            "inspected. A named video or previously indexed file does not qualify."
        ),
    },
}


class ProbeError(RuntimeError):
    """The probe could not collect trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class RouterCase:
    id: str
    prompt: str
    expected: str
    source: str


@dataclass(frozen=True, slots=True)
class Decision:
    selected: str
    confidence: float
    probabilities: dict[str, float] | None
    winner_probability: float | None
    margin: float | None


@dataclass(frozen=True, slots=True)
class ProbeSettings:
    base_url: str
    candidates: tuple[str, ...]
    incumbent_model: str
    rounds: int
    timeout_s: float
    keep_alive: str
    escalation_ceiling: float
    min_winner: float
    min_margin: float


def _default_cases_path() -> Path:
    copied = Path(__file__).with_name("systemone_router_cases.json")
    if copied.exists():
        return copied
    return Path(__file__).resolve().parents[2] / "evals/cases/systemone_router_cases.json"


def load_cases(path: Path) -> list[RouterCase]:
    """Load and strictly validate the tracked decision fixture."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot load case file {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ProbeError("case file must be a schema-1 JSON object")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ProbeError("case file must contain a non-empty 'cases' list")

    cases: list[RouterCase] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProbeError(f"case {index} must be an object")
        case = RouterCase(
            id=str(row.get("id") or "").strip(),
            prompt=str(row.get("prompt") or "").strip(),
            expected=str(row.get("expected") or "").strip(),
            source=str(row.get("source") or "").strip(),
        )
        if not case.id or case.id in seen:
            raise ProbeError(f"case {index} has an empty or duplicate id: {case.id!r}")
        if not case.prompt or len(case.prompt) > 2_000:
            raise ProbeError(f"case {case.id!r} has an empty or overlong prompt")
        if case.expected not in LIVE_EXPECTED_TASKS:
            raise ProbeError(f"case {case.id!r} has unsupported expected task {case.expected!r}")
        if case.source not in {"legacy", "expanded"}:
            raise ProbeError(f"case {case.id!r} has unsupported source {case.source!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"System One {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ProbeError(f"System One {field} must be between 0 and 1")
    return number


def parse_systemone_response(body: dict[str, Any]) -> Decision:
    """Validate one typed choice answer and preserve its full distribution."""
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ProbeError("System One response is missing the 'answers' object")
    answer = answers.get("task")
    if not isinstance(answer, dict):
        raise ProbeError("System One response is missing the 'task' answer")

    selected = answer.get("choice")
    if selected not in TASKS:
        raise ProbeError(f"System One returned unknown choice {selected!r}")
    raw_probabilities = answer.get("probabilities")
    if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(TASKS):
        raise ProbeError("System One must return probabilities for all four task choices")
    probabilities = {
        task: _probability(raw_probabilities[task], field=f"probabilities.{task}")
        for task in TASKS
    }
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, abs_tol=0.02):
        raise ProbeError(f"System One probabilities sum to {total:.4f}, not 1")
    confidence = _probability(answer.get("confidence"), field="confidence")

    ordered = sorted(probabilities.values(), reverse=True)
    winner_probability = probabilities[selected]
    if winner_probability + 1e-9 < ordered[0]:
        raise ProbeError("System One choice does not match its highest probability")
    return Decision(
        selected=selected,
        confidence=confidence,
        probabilities=probabilities,
        winner_probability=winner_probability,
        margin=ordered[0] - ordered[1],
    )


class SystemOneProbeClient:
    """Probe-local Ollama HTTP surface; production keeps its current client."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            headers={"Accept": "application/json"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request_json(
        self, method: str, path: str, *, payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise ProbeError(
                f"{method} {path} transport error: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ProbeError(f"{method} {path} -> {response.status_code}: {response.text[:500]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise ProbeError(f"{method} {path} returned invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise ProbeError(f"{method} {path} returned {type(body).__name__}, expected object")
        return body

    async def version(self) -> str:
        body = await self._request_json("GET", "/api/version")
        version = body.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ProbeError("GET /api/version returned no version string")
        return version.strip()

    async def tags(self) -> list[dict[str, Any]]:
        body = await self._request_json("GET", "/api/tags")
        models = body.get("models")
        if not isinstance(models, list) or not all(isinstance(row, dict) for row in models):
            raise ProbeError("GET /api/tags returned no model list")
        return models

    async def ps(self) -> list[dict[str, Any]]:
        body = await self._request_json("GET", "/api/ps")
        models = body.get("models")
        if not isinstance(models, list) or not all(isinstance(row, dict) for row in models):
            raise ProbeError("GET /api/ps returned no model list")
        return models

    async def unload(self, model: str) -> None:
        await self._request_json(
            "POST",
            "/api/generate",
            payload={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )

    async def decide(self, *, model: str, state: str, keep_alive: str) -> Decision:
        body = await self._request_json(
            "POST",
            "/v1/systemone",
            payload={
                "model": model,
                "state": state,
                "questions": {"task": ROUTER_QUESTION},
                "keep_alive": keep_alive,
            },
        )
        return parse_systemone_response(body)


def _version_tuple(raw: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for piece in raw.removeprefix("v").split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
        if len(parts) == 3:
            break
    if len(parts) < 2:
        raise ProbeError(f"cannot parse Ollama version {raw!r}")
    return tuple([*parts, 0, 0][:3])  # type: ignore[return-value]


def _minimum_systemone_version(
    candidates: tuple[str, ...],
) -> tuple[tuple[int, int, int], str]:
    families = {model.partition(":")[0] for model in candidates}
    if families & CLEF_MODEL_FAMILIES:
        return CLEF_MIN_OLLAMA_VERSION, "0.35.1"
    return MIN_OLLAMA_VERSION, "0.35.0"


def _model_name(row: dict[str, Any]) -> str:
    return str(row.get("name") or row.get("model") or "")


def _model_metadata(tags: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    for row in tags:
        if _model_name(row) == model:
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            return {
                "name": _model_name(row),
                "size_bytes": row.get("size"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
            }
    return None


def _residency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": _model_name(row),
            "size_bytes": row.get("size"),
            "size_vram_bytes": row.get("size_vram"),
            "expires_at": row.get("expires_at"),
        }
        for row in rows
    ]


def _resident_names(rows: list[dict[str, Any]]) -> set[str]:
    return {_model_name(row) for row in rows if _model_name(row)}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


async def _sample(
    case: RouterCase,
    round_number: int,
    decide: Callable[[str], Awaitable[Decision]],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        decision = await decide(case.prompt)
    except Exception as exc:  # noqa: BLE001 - every failed decision is evidence
        return {
            "case_id": case.id,
            "expected": case.expected,
            "source": case.source,
            "round": round_number,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "case_id": case.id,
        "expected": case.expected,
        "source": case.source,
        "round": round_number,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "valid": True,
        "selected": decision.selected,
        "correct": decision.selected == case.expected,
        "confidence": round(decision.confidence, 6),
        "probabilities": decision.probabilities,
        "winner_probability": _rounded(decision.winner_probability),
        "margin": _rounded(decision.margin),
    }


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    backend: str,
    escalation_ceiling: float,
    min_winner: float,
    min_margin: float,
) -> dict[str, Any]:
    valid = [sample for sample in samples if sample["valid"]]
    latencies = [float(sample["latency_seconds"]) for sample in samples]
    confusion = Counter(
        f"{sample['expected']}->{sample['selected']}" for sample in valid
    )
    costly_false_reasoning = sum(
        sample["selected"] == "reasoning" and sample["expected"] != "reasoning"
        for sample in valid
    )
    missed_reasoning = sum(
        sample["expected"] == "reasoning" and sample["selected"] != "reasoning"
        for sample in valid
    )
    cheap_code_general_swaps = sum(
        sample["selected"] != sample["expected"]
        and {sample["selected"], sample["expected"]} <= {"code", "general"}
        for sample in valid
    )
    if backend == "systemone":
        projected = len(samples) - len(valid) + sum(
            float(sample["winner_probability"]) < min_winner
            or float(sample["margin"]) < min_margin
            for sample in valid
        )
        mapping = {
            "kind": "winner-and-margin",
            "minimum_winner_probability": min_winner,
            "minimum_margin": min_margin,
        }
    else:
        projected = len(samples) - len(valid) + sum(
            float(sample["confidence"]) < escalation_ceiling for sample in valid
        )
        mapping = {
            "kind": "incumbent-self-reported-confidence",
            "minimum_confidence": escalation_ceiling,
        }

    return {
        "samples": len(samples),
        "valid": len(valid),
        "response_failures": len(samples) - len(valid),
        "correct": sum(bool(sample["correct"]) for sample in valid),
        "accuracy": round(
            sum(bool(sample["correct"]) for sample in valid) / len(valid), 4
        ) if valid else None,
        "end_to_end_accuracy": round(
            sum(bool(sample["correct"]) for sample in valid) / len(samples), 4
        ) if samples else None,
        "latency_p50_seconds": _rounded(statistics.median(latencies) if latencies else None),
        "latency_p95_seconds": _rounded(_percentile(latencies, 0.95)),
        "costly_false_reasoning": costly_false_reasoning,
        "missed_reasoning": missed_reasoning,
        "cheap_code_general_swaps": cheap_code_general_swaps,
        "projected_fast_to_deep_escalations": projected,
        "uncertainty_mapping": mapping,
        "confusion": dict(sorted(confusion.items())),
    }


async def _safe_ps(client: SystemOneProbeClient) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return await client.ps(), None
    except ProbeError as exc:
        return [], str(exc)


async def _measure_model(
    *,
    client: SystemOneProbeClient,
    model: str,
    backend: str,
    cases: list[RouterCase],
    settings: ProbeSettings,
    metadata: dict[str, Any],
    decide: Callable[[str], Awaitable[Decision]],
) -> dict[str, Any]:
    observation_errors: list[str] = []
    try:
        await client.unload(model)
        cold_unload_succeeded = True
    except ProbeError as exc:
        cold_unload_succeeded = False
        observation_errors.append(f"cold unload failed: {exc}")

    before, before_error = await _safe_ps(client)
    if before_error:
        observation_errors.append(before_error)
    cold = await _sample(cases[0], 0, decide)

    warm: list[dict[str, Any]] = []
    for round_number in range(1, settings.rounds + 1):
        for case in cases:
            warm.append(await _sample(case, round_number, decide))

    after, after_error = await _safe_ps(client)
    if after_error:
        observation_errors.append(after_error)
    before_names = _resident_names(before)
    after_names = _resident_names(after)
    return {
        "status": "measured",
        "backend": backend,
        "model": model,
        "model_metadata": metadata,
        "confidence_kind": (
            "distribution_concentration" if backend == "systemone" else "self_reported"
        ),
        "cold": {
            "unload_succeeded": cold_unload_succeeded,
            "sample": cold,
        },
        "warm": summarize_samples(
            warm,
            backend=backend,
            escalation_ceiling=settings.escalation_ceiling,
            min_winner=settings.min_winner,
            min_margin=settings.min_margin,
        ),
        "samples": warm,
        "residency": {
            "before": _residency(before),
            "after": _residency(after),
            "evicted_models": sorted(before_names - after_names),
            "newly_loaded_models": sorted(after_names - before_names),
            "observation_errors": observation_errors,
        },
    }


def _read_config_defaults() -> tuple[Any, str, float, float, str | None]:
    try:
        from audrey.config import get_config

        cfg = get_config()
        raw = cfg.raw
        router = raw.get("router") or {}
        ceiling = float(
            ((raw.get("agentic") or {}).get("escalation") or {}).get(
                "confidence_ceiling", 0.95
            )
        )
        return (
            cfg,
            str(router.get("model") or "qwen3.5:4b"),
            float(router.get("timeout_s", 20)),
            ceiling,
            None,
        )
    except Exception as exc:  # noqa: BLE001 - copied probes may have no config
        return None, "qwen3.5:4b", 20.0, 0.95, f"{type(exc).__name__}: {exc}"


def _settings_from_env(default_model: str, default_timeout: float, ceiling: float) -> ProbeSettings:
    candidates = tuple(
        dict.fromkeys(
            model.strip()
            for model in os.environ.get("CANDIDATES", "tev1:0.8b").split(",")
            if model.strip()
        )
    )
    if not candidates:
        raise ProbeError("CANDIDATES must contain at least one model tag")
    rounds = int(os.environ.get("ROUNDS", "1"))
    if rounds < 1:
        raise ProbeError("ROUNDS must be at least 1")
    min_winner = float(os.environ.get("MIN_WINNER", "0.55"))
    min_margin = float(os.environ.get("MIN_MARGIN", "0.15"))
    if not 0 <= min_winner <= 1 or not 0 <= min_margin <= 1:
        raise ProbeError("MIN_WINNER and MIN_MARGIN must be between 0 and 1")
    return ProbeSettings(
        base_url=os.environ.get("OLLAMA", "http://ollama:11434").rstrip("/"),
        candidates=candidates,
        incumbent_model=os.environ.get("INCUMBENT_MODEL", default_model).strip(),
        rounds=rounds,
        timeout_s=float(os.environ.get("TIMEOUT", str(default_timeout))),
        keep_alive=os.environ.get("KEEP_ALIVE", "10m"),
        escalation_ceiling=ceiling,
        min_winner=min_winner,
        min_margin=min_margin,
    )


async def collect_report(
    settings: ProbeSettings,
    cases: list[RouterCase],
    *,
    cfg: Any = None,
    config_note: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Collect candidate and incumbent evidence without making a ship decision."""
    try:
        from audrey.models.ollama import OllamaClient
        from audrey.pipeline.classify import ROUTER_SCHEMA, router_classify
    except ImportError as exc:
        raise ProbeError(
            "cannot import audrey; run inside the Audrey container or from the project venv"
        ) from exc

    client = SystemOneProbeClient(
        settings.base_url, timeout_s=settings.timeout_s, transport=transport,
    )
    incumbent = OllamaClient(
        settings.base_url, default_timeout_s=settings.timeout_s, transport=transport,
    )
    try:
        version = await client.version()
        minimum_version, minimum_version_label = _minimum_systemone_version(
            settings.candidates
        )
        if _version_tuple(version) < minimum_version:
            raise ProbeError(
                f"Ollama {version} is too old; selected candidates require "
                f"{minimum_version_label}+"
            )
        tags = await client.tags()
        requested = [*settings.candidates, settings.incumbent_model]
        missing = [model for model in requested if _model_metadata(tags, model) is None]
        if missing:
            raise ProbeError(f"required model tags are not installed: {', '.join(missing)}")
        initial_residency, initial_ps_error = await _safe_ps(client)

        results: list[dict[str, Any]] = []
        for model in settings.candidates:
            async def decide_systemone(prompt: str, *, _model: str = model) -> Decision:
                return await client.decide(
                    model=_model, state=prompt, keep_alive=settings.keep_alive,
                )

            results.append(await _measure_model(
                client=client,
                model=model,
                backend="systemone",
                cases=cases,
                settings=settings,
                metadata=_model_metadata(tags, model) or {},
                decide=decide_systemone,
            ))

        async def decide_incumbent(prompt: str) -> Decision:
            task, confidence, body = await router_classify(
                incumbent,
                router_model=settings.incumbent_model,
                user_text=prompt,
                timeout_s=settings.timeout_s,
                cfg=cfg,
                response_format=ROUTER_SCHEMA,
                no_thinking=True,
            )
            if task is None:
                raise ProbeError(body)
            return Decision(
                selected=task,
                confidence=confidence,
                probabilities=None,
                winner_probability=None,
                margin=None,
            )

        results.append(await _measure_model(
            client=client,
            model=settings.incumbent_model,
            backend="chat-json",
            cases=cases,
            settings=settings,
            metadata=_model_metadata(tags, settings.incumbent_model) or {},
            decide=decide_incumbent,
        ))

        counts = Counter(case.expected for case in cases)
        return {
            "schema": 1,
            "status": "measured",
            "generated_at": datetime.now(UTC).isoformat(),
            "ollama": {
                "base_url": settings.base_url,
                "version": version,
                "minimum_systemone_version": minimum_version_label,
                "initial_residency": _residency(initial_residency),
                "initial_residency_error": initial_ps_error,
            },
            "cases": {
                "count": len(cases),
                "legacy_count": sum(case.source == "legacy" for case in cases),
                "expanded_count": sum(case.source == "expanded" for case in cases),
                "by_expected": dict(sorted(counts.items())),
            },
            "settings": asdict(settings),
            "config_note": config_note,
            "interpretation": {
                "ship_decision": "not_automatic",
                "systemone_confidence": "distribution_concentration_not_correctness",
                "residency": "observed_displacement_not_concurrent_contention",
            },
            "results": results,
        }
    finally:
        await incumbent.aclose()
        await client.aclose()


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"Ollama {report['ollama']['version']} | {report['cases']['count']} cases | "
        f"{report['settings']['rounds']} warm round(s)",
        file=sys.stderr,
    )
    for result in report["results"]:
        warm = result["warm"]
        cold = result["cold"]["sample"]
        print(
            f"{result['backend']:>10} {result['model']}: "
            f"{warm['correct']}/{warm['samples']} correct across all samples, "
            f"valid-only accuracy={warm['accuracy']:.4f}, "
            f"failures={warm['response_failures']}, "
            f"costly false reasoning={warm['costly_false_reasoning']}, "
            f"projected escalations={warm['projected_fast_to_deep_escalations']}, "
            f"cold={cold['latency_seconds']:.2f}s, "
            f"warm p50/p95={warm['latency_p50_seconds']:.2f}/"
            f"{warm['latency_p95_seconds']:.2f}s",
            file=sys.stderr,
        )
    print("Full distributions and per-case results follow as JSON.", file=sys.stderr)


def main() -> int:
    cfg, default_model, default_timeout, ceiling, config_note = _read_config_defaults()
    try:
        settings = _settings_from_env(default_model, default_timeout, ceiling)
        case_path = Path(os.environ.get("CASE_FILE", str(_default_cases_path())))
        cases = load_cases(case_path)
        report = asyncio.run(
            collect_report(settings, cases, cfg=cfg, config_note=config_note)
        )
    except (ProbeError, OSError, ValueError) as exc:
        report = {
            "schema": 1,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    _print_summary(report)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    report_path = os.environ.get("REPORT_PATH", "").strip()
    if report_path:
        Path(report_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"saved report to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
