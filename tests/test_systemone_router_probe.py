"""Contract tests for the probe-only System One router evaluation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts/probes/systemone_router_probe.py"
_SPEC = importlib.util.spec_from_file_location("systemone_router_probe", _SCRIPT)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

_ROUTER_SCRIPT = _ROOT / "scripts/probes/router_probe.py"
_ROUTER_SPEC = importlib.util.spec_from_file_location("legacy_router_probe", _ROUTER_SCRIPT)
assert _ROUTER_SPEC and _ROUTER_SPEC.loader
legacy_probe = importlib.util.module_from_spec(_ROUTER_SPEC)
sys.modules[_ROUTER_SPEC.name] = legacy_probe
_ROUTER_SPEC.loader.exec_module(legacy_probe)


def _valid_body(*, choice: str = "code") -> dict:
    probabilities = {
        "code": 0.7,
        "reasoning": 0.15,
        "general": 0.1,
        "vl": 0.05,
    }
    if choice != "code":
        probabilities["code"], probabilities[choice] = (
            probabilities[choice], probabilities["code"]
        )
    return {
        "answers": {
            "task": {
                "choice": choice,
                "probabilities": probabilities,
                "confidence": 0.62,
            }
        }
    }


class TestTrackedCases:
    def test_fixture_expands_the_original_set_evenly(self):
        cases = probe.load_cases(_ROOT / "evals/cases/systemone_router_cases.json")
        assert len(cases) == 36
        assert sum(case.source == "legacy" for case in cases) == 10
        assert sum(case.source == "expanded" for case in cases) == 26
        assert {task: sum(case.expected == task for case in cases) for task in probe.TASKS} == {
            "code": 12,
            "reasoning": 12,
            "general": 12,
            "vl": 0,
        }

    def test_legacy_subset_is_byte_for_byte_the_incumbent_probe(self):
        cases = probe.load_cases(_ROOT / "evals/cases/systemone_router_cases.json")
        legacy = [(case.prompt, case.expected) for case in cases if case.source == "legacy"]
        assert legacy == legacy_probe.CASES

    def test_vl_is_an_output_option_but_not_a_model_reached_expected_case(self):
        cases = probe.load_cases(_ROOT / "evals/cases/systemone_router_cases.json")
        assert "vl" in probe.ROUTER_QUESTION["criteria"]
        assert all(case.expected != "vl" for case in cases)

    def test_duplicate_ids_fail_loudly(self, tmp_path):
        case = {
            "id": "same",
            "prompt": "Write a function",
            "expected": "code",
            "source": "expanded",
        }
        path = tmp_path / "cases.json"
        path.write_text(json.dumps({"schema": 1, "cases": [case, case]}))
        with pytest.raises(probe.ProbeError, match="duplicate id"):
            probe.load_cases(path)


class TestTypedChoiceValidation:
    def test_preserves_distribution_and_derives_margin(self):
        decision = probe.parse_systemone_response(_valid_body())
        assert decision.selected == "code"
        assert decision.probabilities == {
            "code": 0.7,
            "reasoning": 0.15,
            "general": 0.1,
            "vl": 0.05,
        }
        assert decision.winner_probability == 0.7
        assert decision.margin == pytest.approx(0.55)
        assert decision.confidence == 0.62

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda answer: answer.update(choice="other"), "unknown choice"),
            (lambda answer: answer["probabilities"].pop("vl"), "all four"),
            (lambda answer: answer.update(confidence=1.2), "between 0 and 1"),
            (
                lambda answer: answer["probabilities"].update(
                    code=0.1, reasoning=0.75, general=0.1, vl=0.05
                ),
                "highest probability",
            ),
        ],
    )
    def test_rejects_incomplete_or_inconsistent_answers(self, mutate, match):
        body = _valid_body()
        mutate(body["answers"]["task"])
        with pytest.raises(probe.ProbeError, match=match):
            probe.parse_systemone_response(body)

    def test_client_sends_the_documented_systemone_shape(self):
        seen: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_valid_body(choice="general"))

        async def run():
            client = probe.SystemOneProbeClient(
                "http://ollama:11434",
                timeout_s=20,
                transport=httpx.MockTransport(handler),
            )
            try:
                return await client.decide(
                    model="tev1:0.8b", state="Draft a note", keep_alive="10m"
                )
            finally:
                await client.aclose()

        decision = asyncio.run(run())
        assert seen["path"] == "/v1/systemone"
        assert seen["body"] == {
            "model": "tev1:0.8b",
            "state": "Draft a note",
            "questions": {"task": probe.ROUTER_QUESTION},
            "keep_alive": "10m",
        }
        assert decision.selected == "general"


class TestEvidenceSummary:
    def test_systemone_counts_costly_routes_and_mapping_abstentions(self):
        samples = [
            {
                "valid": True,
                "expected": "code",
                "selected": "code",
                "correct": True,
                "confidence": 0.7,
                "winner_probability": 0.8,
                "margin": 0.6,
                "latency_seconds": 0.2,
            },
            {
                "valid": True,
                "expected": "general",
                "selected": "reasoning",
                "correct": False,
                "confidence": 0.2,
                "winner_probability": 0.52,
                "margin": 0.04,
                "latency_seconds": 0.4,
            },
            {
                "valid": False,
                "expected": "reasoning",
                "latency_seconds": 1.0,
                "error": "bad shape",
            },
        ]
        summary = probe.summarize_samples(
            samples,
            backend="systemone",
            escalation_ceiling=0.95,
            min_winner=0.55,
            min_margin=0.15,
        )
        assert summary["valid"] == 2
        assert summary["accuracy"] == 0.5
        assert summary["costly_false_reasoning"] == 1
        assert summary["projected_fast_to_deep_escalations"] == 2
        assert summary["latency_p50_seconds"] == 0.4
        assert summary["latency_p95_seconds"] == 1.0

    def test_incumbent_uses_its_existing_confidence_ceiling(self):
        samples = [
            {
                "valid": True,
                "expected": "general",
                "selected": "general",
                "correct": True,
                "confidence": 0.94,
                "latency_seconds": 0.2,
            },
            {
                "valid": True,
                "expected": "code",
                "selected": "code",
                "correct": True,
                "confidence": 0.95,
                "latency_seconds": 0.3,
            },
        ]
        summary = probe.summarize_samples(
            samples,
            backend="chat-json",
            escalation_ceiling=0.95,
            min_winner=0.55,
            min_margin=0.15,
        )
        assert summary["projected_fast_to_deep_escalations"] == 1
        assert summary["uncertainty_mapping"]["kind"] == (
            "incumbent-self-reported-confidence"
        )


class TestEndToEndCollection:
    def test_compares_typed_candidate_with_real_incumbent_adapter(self):
        paths: list[str] = []
        chat_payloads: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            paths.append(path)
            if path == "/api/version":
                return httpx.Response(200, json={"version": "0.35.0"})
            if path == "/api/tags":
                return httpx.Response(200, json={"models": [
                    {"name": "tev1:0.8b", "size": 812_000_000, "details": {}},
                    {"name": "qwen3.5:4b", "size": 4_500_000_000, "details": {}},
                ]})
            if path == "/api/ps":
                return httpx.Response(200, json={"models": []})
            if path == "/api/generate":
                return httpx.Response(200, json={"done": True})
            if path == "/v1/systemone":
                return httpx.Response(200, json=_valid_body())
            if path == "/api/show":
                return httpx.Response(200, json={"capabilities": ["thinking"]})
            if path == "/api/chat":
                payload = json.loads(request.content)
                chat_payloads.append(payload)
                return httpx.Response(200, json={
                    "message": {
                        "content": '{"task":"code","confidence":0.97}'
                    }
                })
            raise AssertionError(f"unexpected request: {request.method} {path}")

        settings = probe.ProbeSettings(
            base_url="http://ollama:11434",
            candidates=("tev1:0.8b",),
            incumbent_model="qwen3.5:4b",
            rounds=1,
            timeout_s=20,
            keep_alive="10m",
            escalation_ceiling=0.95,
            min_winner=0.55,
            min_margin=0.15,
        )
        cases = [
            probe.RouterCase(
                id="code-one",
                prompt="Write a Python function",
                expected="code",
                source="expanded",
            )
        ]
        report = asyncio.run(probe.collect_report(
            settings,
            cases,
            transport=httpx.MockTransport(handler),
        ))

        assert report["status"] == "measured"
        assert [result["backend"] for result in report["results"]] == [
            "systemone",
            "chat-json",
        ]
        assert report["results"][0]["samples"][0]["probabilities"]["code"] == 0.7
        assert report["results"][1]["samples"][0]["probabilities"] is None
        assert paths.count("/v1/systemone") == 2  # cold + warm
        assert paths.count("/api/chat") == 2      # cold + warm
        assert paths.count("/api/generate") == 2 # one cold unload per model
        assert all(payload["think"] is False for payload in chat_payloads)
        assert all(payload["format"]["required"] == ["task", "confidence"]
                   for payload in chat_payloads)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.35.0", (0, 35, 0)),
        ("v0.35.1-rc1", (0, 35, 1)),
        ("1.0", (1, 0, 0)),
    ],
)
def test_version_parser(raw, expected):
    assert probe._version_tuple(raw) == expected


@pytest.mark.parametrize(
    "candidates,expected",
    [
        (("tev1:0.8b",), ((0, 35, 0), "0.35.0")),
        (("nimble:latest",), ((0, 35, 0), "0.35.0")),
        (("clef:latest",), ((0, 35, 1), "0.35.1")),
        (("clef-flash:latest", "nimble:latest"), ((0, 35, 1), "0.35.1")),
    ],
)
def test_candidate_specific_minimum_ollama_version(candidates, expected):
    assert probe._minimum_systemone_version(candidates) == expected
