#!/usr/bin/env python3
"""Measure Clef-style System One decision quality and runtime behavior.

This probe complements ``systemone_router_probe.py``. The router probe answers
one Audrey-specific question over 36 prompts; this probe exercises the broader
System One contract: choice, noul, score, structured JSON state, multiple
questions in one request, and Clef image input.

It is measurement-only. It does not edit Audrey's configuration or model
registry. Each model is unloaded before its cold request and again after its
warm samples, so a finished probe does not leave Clef occupying VRAM.

Run on Tower after the exact model tag is installed::

    scripts/probes/probe-onbox.sh systemone_decision_probe.py \
      COPY=systemone_decision_cases.json MODELS=clef:latest ROUNDS=3

Environment:
  OLLAMA      Ollama base URL (default http://ollama:11434)
  MODELS      comma-separated exact tags (default clef:latest)
  ROUNDS      warm passes over every case (default 1)
  TIMEOUT     per-request seconds (default 120)
  KEEP_ALIVE  residency after each decision request (default 10m)
  CASE_FILE   alternate schema-1 decision fixture
  REPORT_PATH optional path for the complete JSON report
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import statistics
import struct
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

MIN_OLLAMA_VERSION = (0, 35, 1)
MIN_OLLAMA_VERSION_LABEL = "0.35.1"
QUESTION_TYPES = frozenset({"choice", "noul", "score"})
INPUT_KINDS = frozenset({"text", "json", "image"})
IMAGE_FIXTURES = frozenset({"solid_red_png"})


class ProbeError(RuntimeError):
    """The probe could not collect trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class DecisionCase:
    id: str
    input_kind: str
    state: Any
    questions: dict[str, dict[str, Any]]
    expected: dict[str, dict[str, Any]]
    image_fixture: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeSettings:
    base_url: str
    models: tuple[str, ...]
    rounds: int
    timeout_s: float
    keep_alive: str


def _default_cases_path() -> Path:
    copied = Path(__file__).with_name("systemone_decision_cases.json")
    if copied.exists():
        return copied
    return Path(__file__).resolve().parents[2] / "evals/cases/systemone_decision_cases.json"


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProbeError(f"{field} must be finite")
    return number


def _probability(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ProbeError(f"{field} must be between 0 and 1")
    return number


def _validate_question(case_id: str, name: str, question: Any) -> str:
    if not isinstance(question, dict):
        raise ProbeError(f"case {case_id!r} question {name!r} must be an object")
    question_type = question.get("type")
    if question_type not in QUESTION_TYPES:
        raise ProbeError(
            f"case {case_id!r} question {name!r} has unsupported type {question_type!r}"
        )
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ProbeError(f"case {case_id!r} question {name!r} needs instructions")
    criteria = question.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
            raise ProbeError(
                f"case {case_id!r} choice {name!r} needs 2 to 26 criteria"
            )
        if any(not isinstance(key, str) or not key for key in criteria):
            raise ProbeError(f"case {case_id!r} choice {name!r} has an invalid option")
        if any(value is not None and not isinstance(value, str) for value in criteria.values()):
            raise ProbeError(
                f"case {case_id!r} choice {name!r} descriptions must be strings or null"
            )
    elif question_type == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict)
            or set(criteria) != {"false", "true"}
            or any(not isinstance(value, str) for value in criteria.values())
        ):
            raise ProbeError(
                f"case {case_id!r} noul {name!r} criteria must describe false and true"
            )
    elif not isinstance(criteria, list) or not 2 <= len(criteria) <= 26 or any(
        not isinstance(value, str) or not value.strip() for value in criteria
    ):
        raise ProbeError(f"case {case_id!r} score {name!r} needs 2 to 26 levels")
    return str(question_type)


def _validate_expected(
    case_id: str,
    name: str,
    question: dict[str, Any],
    expected: Any,
) -> None:
    if not isinstance(expected, dict) or expected.get("type") != question.get("type"):
        raise ProbeError(
            f"case {case_id!r} expected {name!r} must match its question type"
        )
    question_type = question["type"]
    if question_type == "choice":
        criteria = question["criteria"]
        if expected.get("value") not in criteria:
            raise ProbeError(
                f"case {case_id!r} expected choice {name!r} is not a criterion"
            )
    elif question_type == "noul":
        if not isinstance(expected.get("value"), bool):
            raise ProbeError(f"case {case_id!r} expected noul {name!r} must be boolean")
    else:
        minimum = _finite_number(expected.get("minimum"), field=f"{case_id}.{name}.minimum")
        maximum = _finite_number(expected.get("maximum"), field=f"{case_id}.{name}.maximum")
        highest = len(question["criteria"]) - 1
        if not 0 <= minimum <= maximum <= highest:
            raise ProbeError(
                f"case {case_id!r} expected score {name!r} is outside 0..{highest}"
            )


def load_cases(path: Path) -> list[DecisionCase]:
    """Load and strictly validate the tracked broad-decision fixture."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot load case file {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ProbeError("case file must be a schema-1 JSON object")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ProbeError("case file must contain a non-empty 'cases' list")

    cases: list[DecisionCase] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProbeError(f"case {index} must be an object")
        case_id = str(row.get("id") or "").strip()
        if not case_id or case_id in seen:
            raise ProbeError(f"case {index} has an empty or duplicate id: {case_id!r}")
        input_kind = str(row.get("input_kind") or "").strip()
        if input_kind not in INPUT_KINDS:
            raise ProbeError(f"case {case_id!r} has invalid input_kind {input_kind!r}")
        state = row.get("state")
        if not isinstance(state, (str, dict, list)) or not state:
            raise ProbeError(f"case {case_id!r} has an empty or invalid state")
        if len(json.dumps(state, ensure_ascii=False)) > 48_000:
            raise ProbeError(f"case {case_id!r} state exceeds the fixture budget")
        questions = row.get("questions")
        expected = row.get("expected")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
            raise ProbeError(f"case {case_id!r} needs 1 to 64 questions")
        if not isinstance(expected, dict) or set(expected) != set(questions):
            raise ProbeError(
                f"case {case_id!r} expected answers must match its question names"
            )
        for name, question in questions.items():
            if not isinstance(name, str) or not name:
                raise ProbeError(f"case {case_id!r} has an invalid question name")
            _validate_question(case_id, name, question)
            _validate_expected(case_id, name, question, expected[name])

        fixture = row.get("image_fixture")
        if input_kind == "image":
            if fixture not in IMAGE_FIXTURES:
                raise ProbeError(f"case {case_id!r} needs a supported image_fixture")
        elif fixture is not None:
            raise ProbeError(f"case {case_id!r} has an image fixture without image input")

        seen.add(case_id)
        cases.append(DecisionCase(
            id=case_id,
            input_kind=input_kind,
            state=state,
            questions=questions,
            expected=expected,
            image_fixture=fixture,
        ))
    return cases


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def solid_red_png_base64(width: int = 192, height: int = 192) -> str:
    """Create a deterministic RGB PNG without adding an image dependency."""
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes((220, 32, 32)) * width
    pixels = row * height
    png = (
        signature
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(pixels, level=9))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _images(case: DecisionCase) -> list[str]:
    if case.image_fixture == "solid_red_png":
        return [solid_red_png_base64()]
    return []


def _answer_type(answer: dict[str, Any], *, question: str, expected: str) -> None:
    if answer.get("type") != expected:
        raise ProbeError(
            f"answer {question!r} type is {answer.get('type')!r}, expected {expected!r}"
        )


def _probabilities(
    raw: Any,
    *,
    keys: set[str],
    field: str,
) -> dict[str, float]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ProbeError(f"{field} probabilities do not match the criteria")
    parsed = {
        key: _probability(value, field=f"{field}.probabilities.{key}")
        for key, value in raw.items()
    }
    total = sum(parsed.values())
    if not math.isclose(total, 1.0, abs_tol=0.02):
        raise ProbeError(f"{field} probabilities sum to {total:.4f}, not 1")
    return parsed


def evaluate_response(case: DecisionCase, body: dict[str, Any]) -> dict[str, Any]:
    """Validate a System One response and score it against one tracked case."""
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(case.questions):
        raise ProbeError("response answers do not match the requested question names")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ProbeError("response is missing usage")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        raise ProbeError("response usage token counts must be non-negative integers")

    results: list[dict[str, Any]] = []
    for name, question in case.questions.items():
        answer = answers[name]
        if not isinstance(answer, dict):
            raise ProbeError(f"answer {name!r} must be an object")
        question_type = question["type"]
        expected = case.expected[name]
        _answer_type(answer, question=name, expected=question_type)

        if question_type == "choice":
            options = set(question["criteria"])
            selected = answer.get("choice")
            if selected not in options:
                raise ProbeError(f"answer {name!r} returned unknown choice {selected!r}")
            probabilities = _probabilities(
                answer.get("probabilities"), keys=options, field=name,
            )
            if probabilities[selected] + 1e-9 < max(probabilities.values()):
                raise ProbeError(f"answer {name!r} choice is not its highest probability")
            confidence = _probability(answer.get("confidence"), field=f"{name}.confidence")
            expected_value = expected["value"]
            results.append({
                "name": name,
                "type": question_type,
                "correct": selected == expected_value,
                "selected": selected,
                "expected": expected_value,
                "probabilities": probabilities,
                "expected_probability": probabilities[expected_value],
                "confidence": confidence,
            })
        elif question_type == "noul":
            probability_true = _probability(answer.get("noul"), field=f"{name}.noul")
            expected_value = bool(expected["value"])
            predicted = probability_true >= 0.5
            target = 1.0 if expected_value else 0.0
            results.append({
                "name": name,
                "type": question_type,
                "correct": predicted == expected_value,
                "predicted": predicted,
                "expected": expected_value,
                "probability_true": probability_true,
                "brier": (probability_true - target) ** 2,
            })
        else:
            levels = len(question["criteria"])
            score = _finite_number(answer.get("score"), field=f"{name}.score")
            if not 0 <= score <= levels - 1:
                raise ProbeError(f"answer {name!r} score is outside 0..{levels - 1}")
            keys = {str(index) for index in range(levels)}
            probabilities = _probabilities(
                answer.get("probabilities"), keys=keys, field=name,
            )
            weighted = sum(int(index) * probability for index, probability in probabilities.items())
            if not math.isclose(score, weighted, abs_tol=0.03):
                raise ProbeError(
                    f"answer {name!r} score {score:.4f} does not match its distribution"
                )
            confidence = _probability(answer.get("confidence"), field=f"{name}.confidence")
            minimum = float(expected["minimum"])
            maximum = float(expected["maximum"])
            distance = minimum - score if score < minimum else score - maximum if score > maximum else 0.0
            results.append({
                "name": name,
                "type": question_type,
                "correct": minimum <= score <= maximum,
                "score": score,
                "expected_minimum": minimum,
                "expected_maximum": maximum,
                "out_of_range_distance": distance,
                "probabilities": probabilities,
                "confidence": confidence,
            })

    return {
        "model": body.get("model"),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "questions": results,
        "questions_correct": sum(bool(result["correct"]) for result in results),
        "questions_total": len(results),
        "case_exact": all(bool(result["correct"]) for result in results),
    }


class SystemOneDecisionClient:
    """Narrow probe-only client for Ollama's documented decision endpoint."""

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
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise ProbeError(
                f"{method} {path} transport error: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ProbeError(
                f"{method} {path} -> {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProbeError(f"{method} {path} returned invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise ProbeError(
                f"{method} {path} returned {type(body).__name__}, expected object"
            )
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

    async def decide(
        self,
        *,
        model: str,
        case: DecisionCase,
        keep_alive: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "state": case.state,
            "questions": case.questions,
            "keep_alive": keep_alive,
        }
        images = _images(case)
        if images:
            payload["images"] = images
        return await self._request_json("POST", "/v1/systemone", payload=payload)


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


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


async def _sample(
    client: SystemOneDecisionClient,
    *,
    model: str,
    case: DecisionCase,
    round_number: int,
    keep_alive: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        body = await client.decide(model=model, case=case, keep_alive=keep_alive)
        evaluated = evaluate_response(case, body)
    except Exception as exc:  # noqa: BLE001 - every failed decision is evidence
        return {
            "case_id": case.id,
            "input_kind": case.input_kind,
            "round": round_number,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "case_id": case.id,
        "input_kind": case.input_kind,
        "round": round_number,
        "latency_seconds": round(time.perf_counter() - started, 6),
        "valid": True,
        **evaluated,
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample["valid"]]
    latencies = [float(sample["latency_seconds"]) for sample in samples]
    questions = [question for sample in valid for question in sample["questions"]]
    type_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kind_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        type_rows[question["type"]].append(question)
    for sample in valid:
        kind_samples[sample["input_kind"]].append(sample)

    by_type = {
        question_type: {
            "questions": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": _rounded(
                sum(bool(row["correct"]) for row in rows) / len(rows)
            ),
        }
        for question_type, rows in sorted(type_rows.items())
    }
    by_input_kind = {
        input_kind: {
            "cases": len(rows),
            "exact": sum(bool(row["case_exact"]) for row in rows),
            "exact_rate": _rounded(
                sum(bool(row["case_exact"]) for row in rows) / len(rows)
            ),
        }
        for input_kind, rows in sorted(kind_samples.items())
    }
    noul_rows = type_rows.get("noul", [])
    choice_rows = type_rows.get("choice", [])
    score_rows = type_rows.get("score", [])
    total_latency = sum(float(sample["latency_seconds"]) for sample in valid)
    input_tokens = [int(sample["usage"]["input_tokens"]) for sample in valid]
    output_tokens = [int(sample["usage"]["output_tokens"]) for sample in valid]

    return {
        "requests": len(samples),
        "valid_requests": len(valid),
        "response_failures": len(samples) - len(valid),
        "questions": len(questions),
        "questions_correct": sum(bool(question["correct"]) for question in questions),
        "question_accuracy": _rounded(
            sum(bool(question["correct"]) for question in questions) / len(questions)
        ) if questions else None,
        "cases_exact": sum(bool(sample["case_exact"]) for sample in valid),
        "case_exact_rate": _rounded(
            sum(bool(sample["case_exact"]) for sample in valid) / len(valid)
        ) if valid else None,
        "latency_p50_seconds": _rounded(
            statistics.median(latencies) if latencies else None
        ),
        "latency_p95_seconds": _rounded(_percentile(latencies, 0.95)),
        "questions_per_second": _rounded(
            len(questions) / total_latency if total_latency else None
        ),
        "input_tokens_total": sum(input_tokens),
        "input_tokens_p50": _rounded(
            statistics.median(input_tokens) if input_tokens else None
        ),
        "output_tokens_total": sum(output_tokens),
        "by_type": by_type,
        "by_input_kind": by_input_kind,
        "noul_brier_mean": _rounded(
            statistics.fmean(float(row["brier"]) for row in noul_rows)
            if noul_rows else None
        ),
        "choice_expected_log_loss_mean": _rounded(
            statistics.fmean(
                -math.log(max(float(row["expected_probability"]), 1e-12))
                for row in choice_rows
            ) if choice_rows else None
        ),
        "score_out_of_range_distance_mean": _rounded(
            statistics.fmean(float(row["out_of_range_distance"]) for row in score_rows)
            if score_rows else None
        ),
    }


async def _safe_ps(
    client: SystemOneDecisionClient,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return await client.ps(), None
    except ProbeError as exc:
        return [], str(exc)


async def _measure_model(
    client: SystemOneDecisionClient,
    *,
    model: str,
    cases: list[DecisionCase],
    settings: ProbeSettings,
    metadata: dict[str, Any],
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
    cold = await _sample(
        client,
        model=model,
        case=cases[0],
        round_number=0,
        keep_alive=settings.keep_alive,
    )

    warm: list[dict[str, Any]] = []
    for round_number in range(1, settings.rounds + 1):
        for case in cases:
            warm.append(await _sample(
                client,
                model=model,
                case=case,
                round_number=round_number,
                keep_alive=settings.keep_alive,
            ))

    after, after_error = await _safe_ps(client)
    if after_error:
        observation_errors.append(after_error)
    try:
        await client.unload(model)
        cleanup_unload_succeeded = True
    except ProbeError as exc:
        cleanup_unload_succeeded = False
        observation_errors.append(f"cleanup unload failed: {exc}")

    before_names = {_model_name(row) for row in before if _model_name(row)}
    after_names = {_model_name(row) for row in after if _model_name(row)}
    return {
        "status": "measured",
        "model": model,
        "model_metadata": metadata,
        "cold": {
            "unload_succeeded": cold_unload_succeeded,
            "sample": cold,
        },
        "warm": summarize_samples(warm),
        "samples": warm,
        "residency": {
            "before": _residency(before),
            "after": _residency(after),
            "evicted_models": sorted(before_names - after_names),
            "newly_loaded_models": sorted(after_names - before_names),
            "cleanup_unload_succeeded": cleanup_unload_succeeded,
            "observation_errors": observation_errors,
        },
    }


def _settings_from_env() -> ProbeSettings:
    models = tuple(dict.fromkeys(
        model.strip()
        for model in os.environ.get("MODELS", "clef:latest").split(",")
        if model.strip()
    ))
    if not models:
        raise ProbeError("MODELS must contain at least one exact model tag")
    rounds = int(os.environ.get("ROUNDS", "1"))
    timeout_s = float(os.environ.get("TIMEOUT", "120"))
    if rounds < 1:
        raise ProbeError("ROUNDS must be at least 1")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ProbeError("TIMEOUT must be a positive number")
    return ProbeSettings(
        base_url=os.environ.get("OLLAMA", "http://ollama:11434").rstrip("/"),
        models=models,
        rounds=rounds,
        timeout_s=timeout_s,
        keep_alive=os.environ.get("KEEP_ALIVE", "10m"),
    )


async def collect_report(
    settings: ProbeSettings,
    cases: list[DecisionCase],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Collect broad System One evidence without changing Audrey configuration."""
    client = SystemOneDecisionClient(
        settings.base_url,
        timeout_s=settings.timeout_s,
        transport=transport,
    )
    try:
        version = await client.version()
        if _version_tuple(version) < MIN_OLLAMA_VERSION:
            raise ProbeError(
                f"Ollama {version} is too old; Clef requires {MIN_OLLAMA_VERSION_LABEL}+"
            )
        tags = await client.tags()
        missing = [
            model for model in settings.models
            if _model_metadata(tags, model) is None
        ]
        if missing:
            raise ProbeError(f"required model tags are not installed: {', '.join(missing)}")
        initial_residency, initial_ps_error = await _safe_ps(client)
        results = []
        for model in settings.models:
            results.append(await _measure_model(
                client,
                model=model,
                cases=cases,
                settings=settings,
                metadata=_model_metadata(tags, model) or {},
            ))
        final_residency, final_ps_error = await _safe_ps(client)
        question_counts = Counter(
            question["type"]
            for case in cases
            for question in case.questions.values()
        )
        input_counts = Counter(case.input_kind for case in cases)
        return {
            "schema": 1,
            "status": "measured",
            "generated_at": datetime.now(UTC).isoformat(),
            "ollama": {
                "base_url": settings.base_url,
                "version": version,
                "minimum_clef_version": MIN_OLLAMA_VERSION_LABEL,
                "initial_residency": _residency(initial_residency),
                "initial_residency_error": initial_ps_error,
                "final_residency": _residency(final_residency),
                "final_residency_error": final_ps_error,
            },
            "cases": {
                "count": len(cases),
                "questions": sum(question_counts.values()),
                "by_question_type": dict(sorted(question_counts.items())),
                "by_input_kind": dict(sorted(input_counts.items())),
            },
            "settings": asdict(settings),
            "interpretation": {
                "decision_only": True,
                "confidence": "distribution_concentration_not_correctness",
                "noul_threshold": 0.5,
                "score": "probability_weighted_zero_based_level",
                "residency": "sequential_loading_not_concurrent_contention",
                "ship_decision": "not_automatic",
            },
            "results": results,
        }
    finally:
        await client.aclose()


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"Ollama {report['ollama']['version']} | {report['cases']['count']} cases | "
        f"{report['cases']['questions']} questions | "
        f"{report['settings']['rounds']} warm round(s)",
        file=sys.stderr,
    )
    for result in report["results"]:
        warm = result["warm"]
        cold = result["cold"]["sample"]
        print(
            f"systemone {result['model']}: "
            f"questions={warm['questions_correct']}/{warm['questions']}, "
            f"exact cases={warm['cases_exact']}/{warm['valid_requests']}, "
            f"failures={warm['response_failures']}, "
            f"cold={cold['latency_seconds']:.3f}s, "
            f"warm p50/p95={warm['latency_p50_seconds']:.3f}/"
            f"{warm['latency_p95_seconds']:.3f}s, "
            f"qps={warm['questions_per_second']:.2f}",
            file=sys.stderr,
        )
    print("Full per-question evidence follows as JSON.", file=sys.stderr)


def main() -> int:
    try:
        settings = _settings_from_env()
        case_path = Path(os.environ.get("CASE_FILE", str(_default_cases_path())))
        cases = load_cases(case_path)
        report = asyncio.run(collect_report(settings, cases))
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
