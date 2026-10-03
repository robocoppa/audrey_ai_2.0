"""Contracts for the broad Clef/System One decision performance probe."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts/probes/systemone_decision_probe.py"
_SPEC = importlib.util.spec_from_file_location("systemone_decision_probe", _SCRIPT)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)
_CASES = _ROOT / "evals/cases/systemone_decision_cases.json"


def _choice_case() -> object:
    return probe.DecisionCase(
        id="team",
        input_kind="text",
        state="Checkout returns HTTP 500.",
        questions={
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {
                    "billing": "Payment issue",
                    "technical": "Software issue",
                },
            }
        },
        expected={"team": {"type": "choice", "value": "technical"}},
    )


def _choice_body() -> dict:
    return {
        "model": "clef:latest",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.02, "technical": 0.98},
                "confidence": 0.92,
            }
        },
        "usage": {"input_tokens": 80, "output_tokens": 1},
    }


class TestTrackedDecisionCases:
    def test_fixture_covers_every_decision_and_input_type(self):
        cases = probe.load_cases(_CASES)

        assert len(cases) == 12
        assert sum(len(case.questions) for case in cases) == 29
        assert {
            kind: sum(case.input_kind == kind for case in cases)
            for kind in probe.INPUT_KINDS
        } == {"text": 8, "json": 3, "image": 1}
        assert {
            kind: sum(
                question["type"] == kind
                for case in cases
                for question in case.questions.values()
            )
            for kind in probe.QUESTION_TYPES
        } == {"choice": 11, "noul": 12, "score": 6}

    def test_fixture_question_names_and_expectations_match(self):
        for case in probe.load_cases(_CASES):
            assert set(case.questions) == set(case.expected)
            for name, question in case.questions.items():
                assert question["type"] == case.expected[name]["type"]

    def test_duplicate_ids_fail_loudly(self, tmp_path):
        row = {
            "id": "same",
            "input_kind": "text",
            "state": "hello",
            "questions": {
                "greeting": {
                    "type": "noul",
                    "instructions": "Is this a greeting?",
                }
            },
            "expected": {
                "greeting": {"type": "noul", "value": True},
            },
        }
        path = tmp_path / "cases.json"
        path.write_text(json.dumps({"schema": 1, "cases": [row, row]}))

        with pytest.raises(probe.ProbeError, match="duplicate id"):
            probe.load_cases(path)

    def test_expected_choice_must_be_one_of_the_criteria(self, tmp_path):
        row = {
            "id": "bad-choice",
            "input_kind": "text",
            "state": "hello",
            "questions": {
                "kind": {
                    "type": "choice",
                    "instructions": "Which kind?",
                    "criteria": {"a": None, "b": None},
                }
            },
            "expected": {
                "kind": {"type": "choice", "value": "c"},
            },
        }
        path = tmp_path / "cases.json"
        path.write_text(json.dumps({"schema": 1, "cases": [row]}))

        with pytest.raises(probe.ProbeError, match="not a criterion"):
            probe.load_cases(path)


class TestResponseValidation:
    def test_scores_choice_noul_and_score_answers(self):
        case = probe.load_cases(_CASES)[0]
        body = {
            "model": "clef:latest",
            "answers": {
                "team": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {
                        "billing": 0.98,
                        "technical": 0.01,
                        "sales": 0.01,
                    },
                    "confidence": 0.94,
                },
                "refund_requested": {
                    "type": "noul",
                    "noul": 0.99,
                },
                "urgency": {
                    "type": "score",
                    "score": 0.8,
                    "probabilities": {"0": 0.3, "1": 0.6, "2": 0.1},
                    "confidence": 0.41,
                },
            },
            "usage": {"input_tokens": 150, "output_tokens": 3},
        }

        result = probe.evaluate_response(case, body)

        assert result["questions_correct"] == 3
        assert result["case_exact"] is True
        assert result["questions"][0]["expected_probability"] == 0.98
        assert result["questions"][1]["brier"] == pytest.approx(0.0001)
        assert result["questions"][2]["out_of_range_distance"] == 0.0

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (
                lambda body: body["answers"]["team"].update(choice="billing"),
                "highest probability",
            ),
            (
                lambda body: body["answers"]["team"]["probabilities"].pop("billing"),
                "do not match",
            ),
            (
                lambda body: body["usage"].update(input_tokens=-1),
                "non-negative integers",
            ),
        ],
    )
    def test_rejects_inconsistent_or_incomplete_responses(self, mutate, match):
        body = _choice_body()
        mutate(body)

        with pytest.raises(probe.ProbeError, match=match):
            probe.evaluate_response(_choice_case(), body)

    def test_summary_keeps_quality_calibration_latency_and_tokens_separate(self):
        first = {
            "valid": True,
            "input_kind": "text",
            "latency_seconds": 0.2,
            "case_exact": True,
            "usage": {"input_tokens": 100, "output_tokens": 2},
            "questions": [
                {
                    "type": "choice",
                    "correct": True,
                    "expected_probability": 0.8,
                },
                {"type": "noul", "correct": True, "brier": 0.01},
                {
                    "type": "score",
                    "correct": True,
                    "out_of_range_distance": 0.0,
                },
            ],
        }
        second = {
            "valid": True,
            "input_kind": "json",
            "latency_seconds": 0.4,
            "case_exact": False,
            "usage": {"input_tokens": 200, "output_tokens": 1},
            "questions": [
                {
                    "type": "choice",
                    "correct": False,
                    "expected_probability": 0.2,
                }
            ],
        }
        failed = {
            "valid": False,
            "input_kind": "image",
            "latency_seconds": 1.0,
            "error": "bad shape",
        }

        summary = probe.summarize_samples([first, second, failed])

        assert summary["valid_requests"] == 2
        assert summary["response_failures"] == 1
        assert summary["questions_correct"] == 3
        assert summary["question_accuracy"] == 0.75
        assert summary["case_exact_rate"] == 0.5
        assert summary["latency_p50_seconds"] == 0.4
        assert summary["latency_p95_seconds"] == 1.0
        assert summary["input_tokens_total"] == 300
        assert summary["by_input_kind"]["text"]["exact_rate"] == 1.0


class TestImageAndClientContract:
    def test_generated_red_fixture_is_a_plain_png_payload(self):
        encoded = probe.solid_red_png_base64()
        decoded = base64.b64decode(encoded)

        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
        assert not encoded.startswith("data:")
        assert len(decoded) > 100

    def test_client_sends_documented_structured_state_questions_and_images(self):
        seen: dict = {}
        image_case = probe.load_cases(_CASES)[-1]

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        async def run():
            client = probe.SystemOneDecisionClient(
                "http://ollama:11434",
                timeout_s=20,
                transport=httpx.MockTransport(handler),
            )
            try:
                await client.decide(
                    model="clef:latest",
                    case=image_case,
                    keep_alive="10m",
                )
            finally:
                await client.aclose()

        asyncio.run(run())

        assert seen["path"] == "/v1/systemone"
        assert seen["body"]["model"] == "clef:latest"
        assert seen["body"]["state"] == image_case.state
        assert seen["body"]["questions"] == image_case.questions
        assert seen["body"]["keep_alive"] == "10m"
        assert len(seen["body"]["images"]) == 1
        assert not seen["body"]["images"][0].startswith("data:")


class TestEndToEndCollection:
    def test_collects_cold_warm_usage_residency_and_cleanup(self):
        paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            paths.append(path)
            if path == "/api/version":
                return httpx.Response(200, json={"version": "0.35.1"})
            if path == "/api/tags":
                return httpx.Response(200, json={"models": [
                    {
                        "name": "clef:latest",
                        "size": 18_000_000_000,
                        "details": {
                            "parameter_size": "27B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]})
            if path == "/api/ps":
                return httpx.Response(200, json={"models": []})
            if path == "/api/generate":
                return httpx.Response(200, json={"done": True})
            if path == "/v1/systemone":
                return httpx.Response(200, json=_choice_body())
            raise AssertionError(f"unexpected request: {request.method} {path}")

        settings = probe.ProbeSettings(
            base_url="http://ollama:11434",
            models=("clef:latest",),
            rounds=1,
            timeout_s=20,
            keep_alive="10m",
        )
        report = asyncio.run(probe.collect_report(
            settings,
            [_choice_case()],
            transport=httpx.MockTransport(handler),
        ))

        assert report["status"] == "measured"
        assert report["ollama"]["minimum_clef_version"] == "0.35.1"
        assert report["results"][0]["warm"]["question_accuracy"] == 1.0
        assert report["results"][0]["warm"]["input_tokens_total"] == 80
        assert report["results"][0]["residency"]["cleanup_unload_succeeded"] is True
        assert paths.count("/v1/systemone") == 2  # cold + warm
        assert paths.count("/api/generate") == 2  # cold unload + cleanup unload

    def test_rejects_ollama_before_clef_minimum(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.35.0"})
            raise AssertionError("version rejection should stop before another request")

        settings = probe.ProbeSettings(
            base_url="http://ollama:11434",
            models=("clef:latest",),
            rounds=1,
            timeout_s=20,
            keep_alive="10m",
        )
        with pytest.raises(probe.ProbeError, match=r"requires 0\.35\.1\+"):
            asyncio.run(probe.collect_report(
                settings,
                [_choice_case()],
                transport=httpx.MockTransport(handler),
            ))
