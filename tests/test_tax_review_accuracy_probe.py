"""The diagnostic stays paired, hides its labels and cannot grade itself correct."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tax_accuracy_probe", ROOT / "scripts/probes/tax_review_accuracy_probe.py")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)
FIXTURE = ROOT / "evals/cases/tax_review_accuracy.json"


def worksheet(case, **changes):
    gold = case["expected"]
    return probe.Worksheet.model_validate({
        **{key:value for key,value in gold.items() if key != "result_usd"},
        "source_ids": case["input"]["source_ids"] if gold["deduction_usd"] else [],
        "next_question": "What are your age, blindness and dependency statuses?" if gold["missing_fields"] else "",
        "explanation": "Limited worksheet from the supplied passage.", **changes,
    })


def test_fixture_arithmetic_is_consistent_and_gold_never_enters_the_model_packet():
    body = probe.load_cases(FIXTURE)
    assert len(body["cases"]) == 6
    for case in body["cases"]:
        packet = probe.input_packet(body, case)
        mutated = copy.deepcopy(body)
        mutated["cases"][0]["expected"] = {"leak": "DO_NOT_SEND_GOLD"}
        assert "expected" not in packet
        assert probe.input_packet(mutated, case) == packet
        assert probe.score(worksheet(case), case["expected"])["passed"]
        saved, candidate = probe.support_case(packet, worksheet(case))
        assert "expected" not in json.dumps(saved)
        assert "expected" not in json.dumps(candidate)
        assert all(source["tax_year_status"] == "proposed" for source in saved["evidence"])
    missing = next(case for case in body["cases"] if case["id"] == "missing-eligibility")
    assert not {"age", "blind", "dependent"} & missing["input"]["confirmed_profile"].keys()


@pytest.mark.parametrize("corruption", ["language", "source", "hash", "gold", "arithmetic", "unknown_source"])
def test_fixture_preflight_fails_before_any_model_request(tmp_path, corruption):
    body = json.loads(FIXTURE.read_text())
    if corruption == "language":
        body["language"] = "other"
    elif corruption == "source":
        body["sources"][0]["url"] = "https://example.com/not-irs"
    elif corruption == "hash":
        body["sources"][0]["passage"] += "changed"
    elif corruption == "gold":
        del body["cases"][0]["expected"]["wages_usd"]
    elif corruption == "arithmetic":
        body["cases"][0]["expected"]["result_usd"] = "999.00"
    else:
        body["cases"][0]["input"]["source_ids"] = ["not-supplied"]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(body))
    with pytest.raises(probe.SetupError):
        probe.load_cases(path)


def test_correct_ids_do_not_mask_wrong_amounts_or_unnecessary_questions():
    case = probe.load_cases(FIXTURE)["cases"][0]
    assert not probe.score(worksheet(case, wages_usd="59000.00"), case["expected"])["passed"]
    assert not probe.score(worksheet(case, deduction_usd="15000.00"), case["expected"])["passed"]
    assert not probe.score(worksheet(case, next_question="Continue?"), case["expected"])["passed"]
    assert not probe.score(worksheet(case, wage_record_ids=["w2-current", "w2-current"]), case["expected"])["passed"]
    assert not probe.score(worksheet(case, scope="complete_return"), case["expected"])["passed"]


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def chat(self, **kwargs):
        self.requests.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {"done": True, "done_reason": "stop", "prompt_eval_count": 100,
                "eval_count": 50, "message": {"content": json.dumps(reply)}}


def _assessment(case, candidate, attempt, status):
    return {"candidate_sha256": probe.digest(candidate), "case_revision": case["revision"],
            "attempt": attempt, "status": status, "stop_reason": "checks_complete" if status == "checked" else "evidence_gap",
            "findings": [{"finding_index": 0, "verdict": "supported" if status == "checked" else "unsupported",
                          "reason": "Check the exact supplied amount."}], "model": "local:latest"}


@pytest.mark.parametrize("final_correct", [True, False])
async def test_extended_arm_reuses_the_exact_initial_draft_and_repairs_at_most_once(monkeypatch, final_correct):
    body = probe.load_cases(FIXTURE)
    case = body["cases"][0]
    wrong = worksheet(case, deduction_usd="15000.00")
    repaired = worksheet(case) if final_correct else wrong
    fake = FakeModel([wrong.model_dump(), repaired.model_dump()])
    meter = probe.Meter(fake)
    app = SimpleNamespace(state=SimpleNamespace(ollama=meter))
    checked = []

    async def verify(_app, saved, candidate, *, reread, attempt, user_id):
        assert "expected" not in json.dumps((saved, candidate))
        assert user_id == "synthetic-tax-eval"
        assert await reread(saved["evidence"][0]) == saved["evidence"][0]
        checked.append(copy.deepcopy(candidate))
        return _assessment(saved, candidate, attempt, "checked" if attempt == 2 and final_correct else "needs_repair")

    monkeypatch.setattr(probe, "verify_review", verify)
    result = await probe.run_sample(app, body, case, model="local:latest", seed=100)
    assert len(fake.requests) == len(checked) == 2
    assert fake.requests[0]["messages"][:2] == fake.requests[1]["messages"][:2]
    assert json.loads(fake.requests[1]["messages"][2]["content"]) == wrong.model_dump()
    assert "expected" not in json.dumps(fake.requests)
    assert result["single_pass"]["worksheet"] == wrong.model_dump()
    assert result["single_pass"]["passed"] is False
    assert result["checked_repair"]["passed"] is final_correct
    assert result["single_pass"]["cost"]["generation_calls"] == 1
    assert result["checked_repair"]["cost"]["generation_calls"] == 2  # Checker mocked here.
    summary = probe.summarize([result])
    assert summary["improved"] == int(final_correct)
    assert summary["regressed"] == 0


async def test_checker_approval_cannot_override_independent_wrong_answer_score(monkeypatch):
    body = probe.load_cases(FIXTURE)
    entry = body["cases"][0]
    fake = FakeModel([worksheet(entry, deduction_usd="15000.00").model_dump()])
    async def approve(_app, saved, candidate, *, attempt, **kwargs):
        return _assessment(saved, candidate, attempt, "checked")
    monkeypatch.setattr(probe, "verify_review", approve)
    result = await probe.run_sample(SimpleNamespace(state=SimpleNamespace(ollama=probe.Meter(fake))),
                                    body, entry, model="local:latest", seed=100)
    assert not result["checked_repair"]["passed"]
    assert probe.summarize([result])["checker_false_accepts"] == 1
    assert len(fake.requests) == 1


async def test_repair_failure_retains_the_first_answer_without_inventing_a_pair(monkeypatch):
    body = probe.load_cases(FIXTURE)
    entry = body["cases"][0]
    fake = FakeModel([worksheet(entry).model_dump(), probe.OllamaError("failure")])
    async def reject(_app, saved, candidate, *, attempt, **kwargs):
        return _assessment(saved, candidate, attempt, "needs_repair")
    monkeypatch.setattr(probe, "verify_review", reject)
    result = await probe.run_sample(SimpleNamespace(state=SimpleNamespace(ollama=probe.Meter(fake))),
                                    body, entry, model="local:latest", seed=100)
    assert result["status"] == "failed"
    assert result["single_pass"]["passed"]
    assert "checked_repair" not in result
    assert probe.summarize([result])["matched_pairs"] == 0
    assert result["cost"]["output_tokens"]["unknown_calls"] == 1


async def test_live_production_checker_can_use_adapter_without_data_or_authority_mutation():
    from audrey.pipeline.fair_gate import FairLocalGate
    body = probe.load_cases(FIXTURE)
    entry = body["cases"][0]
    client = FakeModel([worksheet(entry).model_dump(), {"findings": [{"finding_index": 0,
        "verdict": "supported", "reason": "The supplied dated table and exact arithmetic support this worksheet."}]}])
    meter = probe.Meter(client)
    app = SimpleNamespace(state=SimpleNamespace(ollama=meter, gate=FairLocalGate(),
        cfg=SimpleNamespace(raw={"model_registry": {"general": [{"name": "local:latest", "priority": 100}]}})))
    before = copy.deepcopy(body)
    with probe.processing_scope(probe.ProcessingPolicy({"local:latest": "a" * 64}, "http://ollama:11434")):
        result = await probe.run_sample(app, body, entry, model="local:latest", seed=100)
    assert result["status"] == "completed" and result["checked_repair"]["passed"]
    assert result["assessments"][0]["status"] == "checked"
    assert result["source_reopens"] == 1
    assert result["cost"]["generation_calls"] == 2
    assert body == before


@pytest.mark.parametrize("raw", ["garbage", "{}", '{"wages_usd": "NaN"}'])
async def test_invalid_writer_output_is_a_sample_failure(raw):
    class Client:
        async def chat(self, **kwargs):
            return {"done": True, "message": {"content": raw}}
    body = probe.load_cases(FIXTURE)
    result = await probe.run_sample(SimpleNamespace(state=SimpleNamespace(ollama=probe.Meter(Client()))),
                                    body, body["cases"][0], model="local:latest", seed=100)
    assert result["status"] == "failed" and result["error"] == "writer_invalid"
    assert probe.summarize([result])["execution_failures"] == 1


def test_unknown_usage_is_not_a_known_zero():
    cost = probe.costs([{"seconds": 1, "input_tokens": None, "output_tokens": 4}])
    assert cost["input_tokens"] == {"known_total": 0, "unknown_calls": 1}
    assert cost["output_tokens"] == {"known_total": 4, "unknown_calls": 0}


@pytest.mark.parametrize("identity_changed,bad_checker", [(True, False), (False, False), (False, True)])
async def test_full_measurement_preflights_and_keeps_identity_drift_as_findings(tmp_path, monkeypatch, identity_changed, bad_checker):
    body = probe.load_cases(FIXTURE)
    calls = []
    class Client(FakeModel):
        def __init__(self, base_url, *, trust_env):
            assert trust_env is False
            super().__init__([worksheet(body["cases"][0]).model_dump(), {} if bad_checker else {
                "findings": [{"finding_index": 0, "verdict": "supported",
                              "reason": "The exact table and arithmetic support the scope."}]}])
            self.tag_reads = 0
            assert base_url == "http://ollama:11434"
        async def tags(self):
            self.tag_reads += 1
            return [{"name": "local:latest", "digest": ("b" if identity_changed and self.tag_reads > 1 else "a") * 64}]
        async def show(self, model):
            return {"model_info": {"general.architecture": "test", "general.parameter_count": 1},
                    "details": {"format": "gguf"}}
        async def chat(self, **kwargs):
            calls.append(kwargs)
            return await super().chat(**kwargs)
        async def aclose(self):
            pass
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model_registry:\n  general:\n    - {name: 'local:latest', priority: 1, location: local}\n")
    monkeypatch.setattr(probe, "OllamaClient", Client)
    result = await probe.measure(SimpleNamespace(cases=FIXTURE, config=cfg, repeats=1,
        only="basic-single", ollama_url="http://ollama:11434"))
    assert result["status"] == ("findings" if identity_changed or bad_checker else "passed")
    assert result["summary"]["matched_pairs"] == 1
    assert result["summary"]["unexpected_checking_stops"] == int(bad_checker)
    assert result["provenance"]["model_identity_stable"] is not identity_changed
    assert len(calls) == 2
    assert result["cost"]["output_tokens"] == {"known_total": 100, "unknown_calls": 0}


@pytest.mark.parametrize("reason", ["bad_case", "repeats", "cloud"])
async def test_measurement_setup_refuses_before_client_construction(tmp_path, monkeypatch, reason):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model_registry:\n  general:\n    - {name: 'remote:cloud', priority: 1, location: cloud}\n")
    monkeypatch.setattr(probe, "OllamaClient", lambda *args: pytest.fail("No client before preflight"))
    with pytest.raises(probe.SetupError):
        await probe.measure(SimpleNamespace(cases=FIXTURE, config=cfg,
            repeats=0 if reason == "repeats" else 1, only="bad" if reason == "bad_case" else "",
            ollama_url="http://ollama:11434"))


@pytest.mark.parametrize("force_unsafe_transport", [False, True])
async def test_measurement_crosses_real_private_client_boundary(tmp_path, monkeypatch, force_unsafe_transport):
    import httpx

    from audrey.models.ollama import OllamaClient

    body = probe.load_cases(FIXTURE)
    metadata = {"model_info": {"general.architecture": "test", "general.parameter_count": 1},
                "details": {"format": "gguf"}}
    replies = iter([worksheet(body["cases"][0]).model_dump(), {"findings": [{
        "finding_index": 0, "verdict": "supported",
        "reason": "The exact table and arithmetic support the limited worksheet."}]}])
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "local:latest", "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json=metadata)
        assert request.url.path == "/api/chat"
        assert request.url.host == "ollama"
        payload = json.loads(request.content)
        assert payload["model"] == "local:latest" and payload["think"] is False
        return httpx.Response(200, json={"done": True, "done_reason": "stop",
            "eval_count": 50, "prompt_eval_count": 100,
            "message": {"role": "assistant", "content": json.dumps(next(replies))}})

    def client_factory(base_url, **kwargs):
        assert kwargs == {"trust_env": False}
        if force_unsafe_transport:
            kwargs["trust_env"] = True
        return OllamaClient(base_url, transport=httpx.MockTransport(respond), **kwargs)

    # No real network traffic: only HTTP transport is replaced, including metadata.
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted-proxy.invalid:3128")
    monkeypatch.setattr(probe, "OllamaClient", client_factory)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model_registry:\n  general:\n    - {name: 'local:latest', priority: 1, location: local}\n")
    report = await probe.measure(SimpleNamespace(cases=FIXTURE, config=cfg, repeats=1,
        only="" if force_unsafe_transport else "basic-single", ollama_url="http://ollama:11434"))
    generations = [request for request in requests if request.url.path == "/api/chat"]
    if force_unsafe_transport:
        assert not generations
        assert report["status"] == "setup_error"
        assert report["measurement_status"] == "failed"
        assert report["error_code"] == "private_transport_not_allowed"
        assert report["accuracy_comparison_available"] is False
        assert len(report["samples"]) == 1  # Shared failure stops the remaining five cases.
        assert report["samples"][0]["error_code"] == "private_transport_not_allowed"
    else:
        assert len(generations) == 2
        assert [request.url.path for request in requests[:4]] == [
            "/api/tags", "/api/show", "/api/tags", "/api/show"]
        assert report["status"] == "passed"
        assert report["measurement_status"] == "completed"
        assert report["accuracy_comparison_available"] is True
        assert report["summary"]["matched_pairs"] == 1
        assert report["cost"]["output_tokens"] == {"known_total": 100, "unknown_calls": 0}


def test_main_uses_setup_exit_for_policy_blocked_measurement(monkeypatch, capsys):
    async def blocked(_args):
        return {"status": "setup_error", "measurement_status": "failed",
                "error_code": "private_transport_not_allowed"}
    monkeypatch.setattr(probe, "measure", blocked)
    assert probe.main([]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "private_transport_not_allowed"
