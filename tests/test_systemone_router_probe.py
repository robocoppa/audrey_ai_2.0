"""Contract tests for the probe-only System One router evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
        assert summary["end_to_end_accuracy"] == pytest.approx(1 / 3, abs=0.0001)
        assert summary["response_failures"] == 1
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
        assert report["cases"]["case_set_sha256"] == hashlib.sha256(json.dumps(
            [{"id": "code-one", "prompt": "Write a Python function",
              "expected": "code", "source": "expanded"}],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()).hexdigest()
        assert report["cases"]["case_set_digest_format"] == (
            "canonical_json_loaded_case_records_v1"
        )
        assert report["results"][0]["by_production_reachability"]["status"] == "unavailable"
        assert [result["backend"] for result in report["results"]] == [
            "systemone",
            "chat-json",
        ]
        assert report["results"][0]["samples"][0]["probabilities"]["code"] == 0.7
        assert report["results"][1]["samples"][0]["probabilities"] is None
        assert paths.count("/v1/systemone") == 2  # cold + warm
        assert paths.count("/api/chat") == 2      # cold + warm
        assert paths.count("/api/generate") == 3 # cold unloads + candidate cleanup
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
        (("nimble-v3:latest",), ((0, 40, 4), "0.40.4")),
        (("nimble-v3:9b-q4_K_M", "clef:latest"), ((0, 40, 4), "0.40.4")),
        (("clef:latest",), ((0, 35, 1), "0.35.1")),
        (("clef-flash:latest", "nimble:latest"), ((0, 35, 1), "0.35.1")),
    ],
)
def test_candidate_specific_minimum_ollama_version(candidates, expected):
    assert probe._minimum_systemone_version(candidates) == expected



def _settings(candidate="nimble-v3:latest"):
    return probe.ProbeSettings(
        base_url="http://ollama:11434", candidates=(candidate,),
        incumbent_model="qwen3.5:4b", rounds=1, timeout_s=20,
        keep_alive="10m", escalation_ceiling=0.95, min_winner=0.55,
        min_margin=0.15,
    )


@pytest.mark.parametrize("version", ["0.35.1", "0.40.3"])
def test_nimble_v3_old_version_fails_before_any_model_operation(version):
    requests = []

    async def handler(request):
        requests.append((request.method, request.url.path))
        assert request.url.path == "/api/version"
        return httpx.Response(200, json={"version": version})

    with pytest.raises(probe.ProbeError, match=r"require 0\.40\.4\+"):
        asyncio.run(probe.collect_report(
            _settings(), [probe.RouterCase("one", "Hello", "general", "expanded")],
            transport=httpx.MockTransport(handler),
        ))
    assert requests == [("GET", "/api/version")]


def test_model_metadata_retains_the_installed_digest_and_quantization():
    metadata = probe._model_metadata([{
        "name": "nimble-v3:latest", "digest": "sha256:immutable-model",
        "size": 9_500_000_000,
        "details": {"parameter_size": "9B", "quantization_level": "Q8_0"},
    }], "nimble-v3:latest")
    assert metadata == {
        "name": "nimble-v3:latest", "digest": "sha256:immutable-model",
        "size_bytes": 9_500_000_000, "parameter_size": "9B",
        "quantization_level": "Q8_0",
    }


def test_reachability_replays_real_guard_order_and_image_sanity_without_generation():
    cases = [
        probe.RouterCase("strong", "Compare the available approaches", "reasoning", "expanded"),
        probe.RouterCase("short", "Hello", "general", "expanded"),
        probe.RouterCase("weak", "Python", "code", "expanded"),
        probe.RouterCase("image", "What does this photo show?", "general", "expanded"),
        probe.RouterCase("model", (
            "Help me draft a thoughtful and professional message to announce "
            "a team celebration next week"
        ), "general", "expanded"),
    ]
    cfg = SimpleNamespace(raw={"router": {
        "skip_llm_under_tokens": 8, "no_thinking": True, "pin_schema": True,
    }})
    result = asyncio.run(probe.production_reachability(cases, cfg=cfg))
    routes = {row["case_id"]: row for row in result["cases"]}
    assert result["status"] == "available"
    assert result["model_reached_cases"] == 1
    assert result["guarded_cases"] == 4
    assert routes["strong"]["guard_reason"] == "keyword:reasoning_strong"
    assert routes["short"]["guard_reason"] == "short_skip:general"
    assert routes["weak"]["guard_reason"] == "short_skip_keyword:code_weak"
    assert routes["image"]["guarded_task"] == "general"
    assert routes["image"]["guard_reason"].startswith("vl_without_image:")
    assert routes["model"] == {"case_id": "model", "model_reached": True}
    assert result["assumptions"] == {
        "conversation_has_image": False, "registered_tool_names": [],
    }


def test_reachability_respects_disabled_short_prompt_skip():
    case = probe.RouterCase("short", "Hello", "general", "expanded")
    result = asyncio.run(probe.production_reachability(
        [case], cfg=SimpleNamespace(raw={"router": {"skip_llm_under_tokens": 0}}),
    ))
    assert result["model_reached_cases"] == 1


def test_reachability_does_not_guess_missing_production_configuration():
    result = asyncio.run(probe.production_reachability([], cfg=None))
    assert result == {
        "status": "unavailable", "reason": "production configuration unavailable",
    }


def test_tracked_fixture_groups_match_the_current_configured_rules():
    import yaml

    cases = probe.load_cases(_ROOT / "evals/cases/systemone_router_cases.json")
    cfg = SimpleNamespace(raw=yaml.safe_load((_ROOT / "config.yaml").read_text()))
    result = asyncio.run(probe.production_reachability(cases, cfg=cfg))
    assert result["model_reached_cases"] == 23
    assert result["guarded_cases"] == 13
    assert len(result["cases"]) == len(cases)


def test_reachability_subgroup_summaries_keep_failures_in_their_true_group():
    samples = [
        {"case_id": "model", "valid": False, "expected": "general",
         "latency_seconds": 20.0, "error": "timeout"},
        {"case_id": "guarded", "valid": True, "expected": "code",
         "selected": "code", "correct": True, "confidence": 0.99,
         "winner_probability": 0.9, "margin": 0.8, "latency_seconds": 0.2},
    ]
    summaries = probe._reachability_summaries(
        samples, backend="systemone", settings=_settings(),
        reachability={"status": "available", "cases": [
            {"case_id": "model", "model_reached": True},
            {"case_id": "guarded", "model_reached": False},
        ]},
    )
    assert summaries["model_reached"]["samples"] == 1
    assert summaries["model_reached"]["response_failures"] == 1
    assert summaries["model_reached"]["accuracy"] is None
    assert summaries["model_reached"]["end_to_end_accuracy"] == 0
    assert summaries["guarded"]["samples"] == 1
    assert summaries["guarded"]["correct"] == 1
    assert "not production latency" in summaries["guarded"]["interpretation"]


def _all_failed_report():
    samples = [{
        "case_id": "one", "round": 1, "valid": False, "expected": "code",
        "latency_seconds": 20.0, "error": "ProbeError: timeout",
    }]
    return {
        "schema": 1, "status": "measured", "ollama": {"version": "0.40.4"},
        "cases": {"count": 1}, "settings": {"rounds": 1},
        "results": [{
            "backend": "systemone", "model": "nimble-v3:latest",
            "cold": {"sample": {"latency_seconds": 20.0, "valid": False}},
            "by_production_reachability": {"status": "unavailable"},
            "warm": probe.summarize_samples(
                samples, backend="systemone", escalation_ceiling=0.95,
                min_winner=0.55, min_margin=0.15,
            ), "samples": samples,
        }],
    }


def test_summary_preserves_all_failed_measurement_instead_of_crashing(capsys):
    report = _all_failed_report()
    probe._print_summary(report)
    assert "valid-only accuracy=n/a" in capsys.readouterr().err
    compact = probe.compact_report(report)
    assert compact["results"][0]["warm"]["response_failures"] == 1
    assert compact["results"][0]["failure_examples"][0]["error"] == "ProbeError: timeout"
    assert "samples" not in compact["results"][0]
    assert report["results"][0]["samples"][0]["error"] == "ProbeError: timeout"


@pytest.mark.parametrize("argv", [[], ["--summary"]])
def test_cli_full_report_is_retained_when_compact_stdout_is_requested(
    argv, monkeypatch, tmp_path, capsys,
):
    report = _all_failed_report()
    report_path = tmp_path / "report.json"
    monkeypatch.setenv("REPORT_PATH", str(report_path))
    monkeypatch.setenv("CANDIDATES", "nimble-v3:latest")
    monkeypatch.setattr(probe, "_read_config_defaults", lambda: (None, "qwen3.5:4b", 20, .95, None))

    async def collect(*_args, **_kwargs):
        return report

    monkeypatch.setattr(probe, "collect_report", collect)
    assert probe.main(argv) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert json.loads(report_path.read_text()) == report
    assert ("samples" in stdout["results"][0]) == (not argv)
    assert stdout["results"][0]["warm"]["response_failures"] == 1



def test_compact_report_omits_guard_case_details_without_mutating_full_report():
    report = _all_failed_report()
    report["cases"]["production_reachability"] = {
        "status": "available", "model_reached_cases": 1, "guarded_cases": 0,
        "cases": [{"case_id": "one", "model_reached": True}],
    }
    compact = probe.compact_report(report)
    assert "cases" not in compact["cases"]["production_reachability"]
    assert compact["cases"]["production_reachability"]["model_reached_cases"] == 1
    assert report["cases"]["production_reachability"]["cases"] == [
        {"case_id": "one", "model_reached": True},
    ]



@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_candidate_cleanup_follows_its_snapshot_and_precedes_incumbent_generation(
    cleanup_fails,
):
    events = []
    loaded = {"untested:large"}
    candidate_unloads = 0

    async def handler(request):
        nonlocal candidate_unloads
        path = request.url.path
        payload = json.loads(request.content) if request.content else {}
        model = payload.get("model")
        events.append((path, model))
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.4"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "nimble-v3:latest", "digest": "sha256:candidate"},
                {"name": "qwen3.5:4b", "digest": "sha256:incumbent"},
            ]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [
                {"name": name} for name in sorted(loaded)
            ]})
        if path == "/api/generate":
            assert payload["keep_alive"] == 0
            assert model in {"nimble-v3:latest", "qwen3.5:4b"}
            if model == "nimble-v3:latest":
                candidate_unloads += 1
                if cleanup_fails and candidate_unloads == 2:
                    return httpx.Response(500, json={"error": "cleanup unavailable"})
            loaded.discard(model)
            return httpx.Response(200, json={"done": True})
        if path == "/v1/systemone":
            loaded.add(model)
            return httpx.Response(200, json=_valid_body())
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["thinking"]})
        if path == "/api/chat":
            loaded.add(model)
            return httpx.Response(200, json={"message": {
                "content": '{"task":"code","confidence":0.97}',
            }})
        raise AssertionError(f"unexpected request {path}")

    report = asyncio.run(probe.collect_report(
        _settings(), [probe.RouterCase("one", "Write Python code", "code", "expanded")],
        transport=httpx.MockTransport(handler),
    ))
    candidate, incumbent = report["results"]
    assert candidate["cleanup"]["succeeded"] is (not cleanup_fails)
    assert candidate["warm"]["correct"] == 1
    assert candidate["samples"][0]["valid"] is True
    assert {row["name"] for row in candidate["residency"]["after"]} == {
        "untested:large", "nimble-v3:latest",
    }
    candidate_unload_indices = [
        index for index, event in enumerate(events)
        if event == ("/api/generate", "nimble-v3:latest")
    ]
    assert len(candidate_unload_indices) == 2
    cleanup_index = candidate_unload_indices[-1]
    assert events[cleanup_index - 1] == ("/api/ps", None)
    assert cleanup_index < events.index(("/api/chat", "qwen3.5:4b"))
    assert events[-2:] == [("/api/chat", "qwen3.5:4b"), ("/api/ps", None)]
    assert incumbent["warm"]["correct"] == 1
    assert "cleanup" not in incumbent
    assert "qwen3.5:4b" in loaded
    assert "untested:large" in loaded
    assert ("nimble-v3:latest" in loaded) is cleanup_fails
    if cleanup_fails:
        assert "cleanup unavailable" in candidate["cleanup"]["error"]
