"""Hermetic checks for frozen skill-selection studies, using synthetic cases.

The prospective measurement fixture is never loaded by this module. A prepared
plan describes an evaluation and operator attestations, not production approval.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import stat
import sys
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "evals/eval_skill_selection.py"
_SPEC = importlib.util.spec_from_file_location("skill_selection_protocol_eval", _SCRIPT)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

DOCUMENT = "grounded-document-analysis"
VIDEO = "video-analysis"
MODEL = "qwen3.5:4b"


def _cases():
    document = probe.FileEvidence("protocol-brief.pdf", "document", "ready")
    video = probe.FileEvidence("protocol-clip.mp4", "video", "ready")
    return [
        probe.Case(
            "synthetic-document", "positive",
            "Resume protocol-brief.pdf en tres frases.", "auto", (document,),
            DOCUMENT, "PRIVATE_LABEL_REASON_D21A",
        ),
        probe.Case(
            "synthetic-video", "positive",
            "Resume protocol-clip.mp4 en tres frases.", "auto", (video,),
            VIDEO, "PRIVATE_LABEL_REASON_A46D",
        ),
        probe.Case(
            "synthetic-ordinary", "ordinary",
            "Delete protocol-brief.pdf from My Files.", "auto", (document,),
            "none", "This requests interface file management.",
        ),
        probe.Case(
            "synthetic-ambiguous", "ambiguous",
            "Summarize missing-protocol.pdf.", "auto", (document,),
            "none", "The named target is unavailable.",
        ),
    ]


def _prepare(cases=None, **overrides):
    values = {
        "backend": "hybrid", "model": MODEL, "repeats": 3,
        "timeout_s": 20, "catalog": probe.DEFAULT_CATALOG,
        "max_router_calls": 36, "labels_reviewed": False, "gates_agreed": False,
        **overrides,
    }
    return probe.prepare_plan(_cases() if cases is None else cases, **values)


def _validate(plan, cases=None, **overrides):
    return probe.validate_plan(
        plan, _cases() if cases is None else cases,
        **{"model": MODEL, "timeout_s": 20, "catalog": probe.DEFAULT_CATALOG, **overrides},
    )


def _write_cases(tmp_path, cases=None):
    path = tmp_path / "synthetic-cases.json"
    path.write_text(
        json.dumps({
            "schema": 1,
            "description": "Separate synthetic protocol tests, not a measurement fixture.",
            "cases": [asdict(case) for case in (_cases() if cases is None else cases)],
        }),
        encoding="utf-8",
    )
    return path


def _write_config(tmp_path):
    import yaml

    path = tmp_path / "synthetic-config.yaml"
    path.write_text(yaml.safe_dump({
        "skills": {"enabled": True, "auto_select": False, "roots": [str(_ROOT / "skills")]},
        "router": {"model": MODEL, "timeout_s": 20},
    }), encoding="utf-8")
    return path


def _reply(choice=DOCUMENT):
    return {
        "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps({"skill_id": choice})},
        "prompt_eval_count": 13, "eval_count": 4,
    }


class TestFrozenInputs:
    def test_fresh_plan_validates_against_its_exact_synthetic_inputs(self):
        plan = _prepare()
        assert _validate(plan) is None

    @pytest.mark.parametrize("field", ["prompt", "expected", "reason", "mode"])
    def test_changed_case_data_invalidates_prepared_plan(self, field):
        cases = _cases()
        plan = _prepare(cases)
        replacement = {
            "prompt": "Resume protocol-brief.pdf en una frase.",
            "expected": VIDEO,
            "reason": "A different proposed private judgment.",
            "mode": "deep",
        }[field]
        cases[0] = replace(cases[0], **{field: replacement})
        with pytest.raises(probe.EvalError):
            _validate(plan, cases)

    @pytest.mark.parametrize(
        "file",
        [
            probe.FileEvidence("renamed.pdf", "document", "ready"),
            probe.FileEvidence("protocol-brief.pdf", "document", "pending"),
            probe.FileEvidence("protocol-brief.pdf", "video", "ready"),
        ],
    )
    def test_changed_file_metadata_invalidates_prepared_plan(self, file):
        cases = _cases()
        plan = _prepare(cases)
        cases[0] = replace(cases[0], files=(file,))
        with pytest.raises(probe.EvalError):
            _validate(plan, cases)

    def test_changed_catalog_metadata_invalidates_prepared_plan(self):
        plan = _prepare()
        catalog = copy.deepcopy(probe.DEFAULT_CATALOG)
        catalog[0]["description"] = "A changed catalog meaning."
        with pytest.raises(probe.EvalError):
            _validate(plan, catalog=catalog)

    @pytest.mark.parametrize(
        "overrides", [{"model": "qwen3.5:other"}, {"timeout_s": 21}]
    )
    def test_changed_model_or_timeout_invalidates_prepared_plan(self, overrides):
        with pytest.raises(probe.EvalError):
            _validate(_prepare(), **overrides)

    def test_changed_source_invalidates_plan_without_modifying_real_evaluator(self, tmp_path, monkeypatch):
        plan = _prepare()
        changed = tmp_path / "changed-evaluator.py"
        changed.write_bytes(_SCRIPT.read_bytes() + b"\n# synthetic source drift\n")
        monkeypatch.setattr(probe, "__file__", str(changed))
        with pytest.raises(probe.EvalError):
            _validate(plan)

    def test_unknown_plan_field_is_not_ignored(self):
        plan = _prepare()
        plan["production_approval"] = True
        with pytest.raises(probe.EvalError):
            _validate(plan)

    @pytest.mark.parametrize(
        "override",
        [
            {"backend": "unknown"},
            {"repeats": True},
            {"repeats": 0},
            {"timeout_s": False},
            {"timeout_s": 61},
            {"max_router_calls": 0},
            {"max_router_calls": 769},
            {"max_router_calls": True},
            {"labels_reviewed": "yes"},
            {"gates_agreed": 1},
        ],
    )
    def test_plan_preparation_rejects_ambiguous_or_unbounded_types(self, override):
        with pytest.raises(probe.EvalError):
            _prepare(**override)

    def test_plan_budget_is_checked_before_starting_a_study(self):
        # Two genuine rule abstentions, each repeated three times, require six calls.
        with pytest.raises(probe.EvalError):
            _prepare(max_router_calls=5)


class TestPreparationCli:
    def test_prepare_creates_private_nonoverwritten_plan_without_http(self, tmp_path, monkeypatch, capsys):
        cases = _write_cases(tmp_path)
        config = _write_config(tmp_path)
        output = tmp_path / "frozen-plan.json"

        def forbidden_client(**_kwargs):
            pytest.fail("Preparing a plan must not construct an HTTP client")

        monkeypatch.setattr(probe.httpx, "AsyncClient", forbidden_client)
        args = [
            "--prepare-plan", str(output), "--cases", str(cases),
            "--config", str(config), "--backend", "hybrid", "--repeats", "3",
        ]
        assert probe.main(args) == 0
        captured = capsys.readouterr()
        summary = json.loads(captured.out)
        assert summary["status"] == "prepared"
        assert output.exists()
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
        original = output.read_bytes()
        assert probe.main(args) == 2
        assert output.read_bytes() == original
        assert "error" in json.loads(capsys.readouterr().err)

    def test_plan_file_symlink_cannot_overwrite_existing_artifact(self, tmp_path, monkeypatch, capsys):
        cases = _write_cases(tmp_path)
        config = _write_config(tmp_path)
        target = tmp_path / "retained.txt"
        target.write_text("original artifact", encoding="utf-8")
        link = tmp_path / "frozen-plan.json"
        link.symlink_to(target)
        monkeypatch.setattr(
            probe.httpx, "AsyncClient",
            lambda **_kwargs: pytest.fail("Prepare must not construct an HTTP client"),
        )
        assert probe.main([
            "--prepare-plan", str(link), "--cases", str(cases),
            "--config", str(config), "--backend", "hybrid",
        ]) == 2
        assert target.read_text(encoding="utf-8") == "original artifact"
        assert "error" in json.loads(capsys.readouterr().err)


class TestFrozenPlanShape:
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda plan: plan.update(schema=True),
            lambda plan: plan.update(production_activation=True),
            lambda plan: plan.update(repeats=True),
            lambda plan: plan.update(timeout_seconds="20"),
            lambda plan: plan["budget"].update(max_router_calls=True),
            lambda plan: plan["budget"].update(planned_router_calls=0),
            lambda plan: plan["review"].update(labels_reviewed="yes"),
            lambda plan: plan["review"].update(basis="automatic approval"),
            lambda plan: plan["generation"].update(max_output_tokens=8192),
            lambda plan: plan["criteria"].update(min_activation_precision=0.01),
            lambda plan: plan["fingerprints"].update(source_sha256="not-a-sha256"),
        ],
    )
    def test_tampered_plan_contract_is_rejected(self, mutate):
        plan = _prepare()
        mutate(plan)
        with pytest.raises(probe.EvalError):
            _validate(plan)

    def test_review_attestations_are_explicit_and_never_production_authority(self):
        plan = _prepare()
        assert plan["review"] == {
            "labels_reviewed": False,
            "gates_agreed": False,
            "basis": "operator_attestations_not_independent_proof",
        }
        assert plan["production_activation"] is False
        reviewed = _prepare(labels_reviewed=True, gates_agreed=True)
        assert reviewed["review"]["labels_reviewed"] is True
        assert reviewed["review"]["gates_agreed"] is True
        assert reviewed["production_activation"] is False
        assert reviewed["limitations"]["final_answer_and_workflow_proof_required"] is True

    def test_call_profile_uses_real_policy_guards_and_hybrid_rule_skip(self):
        cases = _cases()
        cases[0] = replace(cases[0], prompt="Summarize protocol-brief.pdf.")
        assert probe.select_rules(cases[0]).selected == DOCUMENT
        assert probe.select_rules(cases[1]).selected == "none"
        router = _prepare(cases, backend="router")
        hybrid = _prepare(cases, backend="hybrid")
        assert router["budget"]["planned_router_calls"] == 6
        assert hybrid["budget"]["planned_router_calls"] == 3
        assert hybrid["budget"]["max_router_calls"] == 36

    @pytest.mark.parametrize("content", ["not-json", "{}", "[]"])
    def test_missing_or_invalid_plan_has_a_setup_error(self, tmp_path, content):
        path = tmp_path / "invalid-plan.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(probe.EvalError):
            probe.load_plan(path)

    def test_nonexistent_plan_and_oversized_plan_are_rejected(self, tmp_path):
        path = tmp_path / "missing-plan.json"
        with pytest.raises(probe.EvalError):
            probe.load_plan(path)
        path.write_text(" " * 65_537, encoding="utf-8")
        with pytest.raises(probe.EvalError):
            probe.load_plan(path)


def _evaluate_synthetic(cases=None, *, backend="hybrid", repeats=3, decisions=None):
    cases = _cases() if cases is None else cases
    replies = iter(decisions) if decisions is not None else None
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if replies is not None:
            choice = next(replies)
        else:
            prompt = json.loads(requests[-1]["messages"][-1]["content"])["prompt"]
            choice = VIDEO if "protocol-clip.mp4" in prompt else DOCUMENT
        if isinstance(choice, httpx.Response):
            return choice
        return httpx.Response(200, json=_reply(choice))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                cases, backend=backend, client=client, model=MODEL,
                repeats=repeats, timeout_s=20,
            )

    return asyncio.run(run()), requests


class TestEvidenceQualification:
    @pytest.mark.parametrize(
        "review",
        [
            {"labels_reviewed": False, "gates_agreed": False},
            {"labels_reviewed": True, "gates_agreed": False},
            {"labels_reviewed": False, "gates_agreed": True},
        ],
    )
    def test_matching_labels_do_not_qualify_without_both_operator_attestations(self, review):
        plan = _prepare(**review)
        report, _requests = _evaluate_synthetic()
        qualification = probe.qualify_report(report, plan)
        assert qualification["status"] == "pending_review"
        assert qualification["production_activation"] is False

    def test_reviewed_adequate_synthetic_evidence_can_meet_frozen_criteria(self):
        plan = _prepare(labels_reviewed=True, gates_agreed=True)
        report, requests = _evaluate_synthetic()
        assert len(requests) == plan["budget"]["planned_router_calls"] == 6
        qualification = probe.qualify_report(report, plan)
        assert qualification["status"] == "criteria_met"
        assert qualification["production_activation"] is False
        wire = json.dumps(requests)
        assert "PRIVATE_LABEL_REASON" not in wire
        assert "expected" not in wire

    def test_all_guarded_matching_cases_cannot_qualify_as_model_evidence(self):
        cases = _cases()[2:]
        plan = _prepare(cases, labels_reviewed=True, gates_agreed=True)
        report, requests = _evaluate_synthetic(cases)
        assert not requests
        qualification = probe.qualify_report(report, plan)
        assert qualification["status"] == "insufficient_evidence"
        assert qualification["production_activation"] is False

    @pytest.mark.parametrize("missing_category", ["positive", "ordinary", "ambiguous"])
    def test_missing_case_population_cannot_qualify(self, missing_category):
        cases = [case for case in _cases() if case.category != missing_category]
        plan = _prepare(cases, labels_reviewed=True, gates_agreed=True)
        report, _requests = _evaluate_synthetic(cases)
        qualification = probe.qualify_report(report, plan)
        assert qualification["status"] == "insufficient_evidence"

    def test_single_repeat_is_not_sufficient_for_qualification(self):
        plan = _prepare(repeats=1, labels_reviewed=True, gates_agreed=True)
        report, _requests = _evaluate_synthetic(repeats=1)
        assert probe.qualify_report(report, plan)["status"] == "insufficient_evidence"

    def test_model_abstaining_on_every_positive_has_no_activation_evidence(self):
        plan = _prepare(labels_reviewed=True, gates_agreed=True)
        report, _requests = _evaluate_synthetic(decisions=["none"] * 6)
        assert probe.qualify_report(report, plan)["status"] == "insufficient_evidence"

    def test_one_miss_out_of_six_positives_exceeds_the_frozen_miss_gate(self):
        plan = _prepare(labels_reviewed=True, gates_agreed=True)
        report, _requests = _evaluate_synthetic(
            decisions=["none", DOCUMENT, DOCUMENT, VIDEO, VIDEO, VIDEO]
        )
        qualification = probe.qualify_report(report, plan)
        assert qualification["status"] == "findings"
        assert qualification["production_activation"] is False

    def test_transport_error_is_a_finding_in_a_reviewed_adequate_study(self):
        plan = _prepare(labels_reviewed=True, gates_agreed=True)
        report, _requests = _evaluate_synthetic(
            decisions=[httpx.Response(503), DOCUMENT, DOCUMENT, VIDEO, VIDEO, VIDEO]
        )
        assert probe.qualify_report(report, plan)["status"] == "findings"


def _prepare_cli(tmp_path, capsys):
    cases_path = _write_cases(tmp_path)
    config_path = _write_config(tmp_path)
    plan_path = tmp_path / "cli-plan.json"
    exit_code = probe.main([
        "--prepare-plan", str(plan_path), "--cases", str(cases_path),
        "--config", str(config_path), "--backend", "hybrid", "--repeats", "3",
        "--base-url", "http://ollama:11434",
    ])
    assert exit_code == 0
    capsys.readouterr()
    return plan_path, cases_path, config_path


def _forbid_client(monkeypatch):
    monkeypatch.setattr(
        probe.httpx, "AsyncClient",
        lambda **_kwargs: pytest.fail("Frozen-plan preflight must finish before constructing HTTP client"),
    )


class TestPlannedRunPreflight:
    @pytest.mark.parametrize(
        "override",
        [
            ["--only", "synthetic-document"],
            ["--backend", "router"],
            ["--repeats", "4"],
            ["--timeout", "21"],
            ["--model", "qwen3.5:other"],
            ["--base-url", "http://127.0.0.1:11434"],
            ["--max-router-calls", "35"],
            ["--labels-reviewed"],
            ["--gates-agreed"],
            ["--allow-nonretained-model"],
        ],
    )
    def test_forbidden_filter_and_conflicting_overrides_fail_before_http(
        self, tmp_path, monkeypatch, capsys, override
    ):
        _forbid_client(monkeypatch)
        plan, cases, config = _prepare_cli(tmp_path, capsys)
        assert probe.main([
            "--plan", str(plan), "--cases", str(cases), "--config", str(config),
            *override,
        ]) == 2
        assert "error" in json.loads(capsys.readouterr().err)

    @pytest.mark.parametrize("field", ["prompt", "expected", "files"])
    def test_changed_case_file_rejects_before_http(self, tmp_path, monkeypatch, capsys, field):
        _forbid_client(monkeypatch)
        plan, cases, config = _prepare_cli(tmp_path, capsys)
        payload = json.loads(cases.read_text(encoding="utf-8"))
        payload["cases"][0][field] = {
            "prompt": "Resume protocol-brief.pdf en una frase.",
            "expected": VIDEO,
            "files": [{"name": "protocol-brief.pdf", "kind": "document", "status": "pending"}],
        }[field]
        cases.write_text(json.dumps(payload), encoding="utf-8")
        assert probe.main([
            "--plan", str(plan), "--cases", str(cases), "--config", str(config),
        ]) == 2
        assert "error" in json.loads(capsys.readouterr().err)

    def test_changed_source_rejects_before_http(self, tmp_path, monkeypatch, capsys):
        _forbid_client(monkeypatch)
        plan, cases, config = _prepare_cli(tmp_path, capsys)
        changed_source = tmp_path / "changed-source.py"
        changed_source.write_bytes(_SCRIPT.read_bytes() + b"\n# frozen study drift\n")
        monkeypatch.setattr(probe, "__file__", str(changed_source))
        assert probe.main([
            "--plan", str(plan), "--cases", str(cases), "--config", str(config),
        ]) == 2
        assert "error" in json.loads(capsys.readouterr().err)

    def test_changed_catalog_rejects_before_http(self, tmp_path, monkeypatch, capsys):
        import yaml

        _forbid_client(monkeypatch)
        plan, cases, config = _prepare_cli(tmp_path, capsys)
        root = tmp_path / "changed-skills"
        catalog = list(probe.load_catalog(_ROOT / "skills"))
        catalog[0]["description"] = "A changed skill-selection description."
        for entry in catalog:
            directory = root / entry["id"]
            directory.mkdir(parents=True)
            (directory / "SKILL.md").write_text(
                "---\n" + yaml.safe_dump(entry) + "---\nSynthetic unused body.\n",
                encoding="utf-8",
            )
        settings = yaml.safe_load(config.read_text(encoding="utf-8"))
        settings["skills"]["roots"] = [str(root)]
        config.write_text(yaml.safe_dump(settings), encoding="utf-8")
        assert probe.main([
            "--plan", str(plan), "--cases", str(cases), "--config", str(config),
        ]) == 2
        assert "error" in json.loads(capsys.readouterr().err)

    @pytest.mark.parametrize("symlink", [False, True])
    def test_existing_output_cannot_spend_model_calls(self, tmp_path, monkeypatch, capsys, symlink):
        _forbid_client(monkeypatch)
        plan, cases, config = _prepare_cli(tmp_path, capsys)
        target = tmp_path / "retained-report.txt"
        target.write_text("original report", encoding="utf-8")
        output = tmp_path / "result.json"
        if symlink:
            output.symlink_to(target)
        else:
            output.write_text("original result", encoding="utf-8")
        original = output.read_bytes()
        assert probe.main([
            "--plan", str(plan), "--cases", str(cases), "--config", str(config),
            "--save-json", str(output),
        ]) == 2
        assert output.read_bytes() == original
        assert target.read_text(encoding="utf-8") == "original report"
        assert "error" in json.loads(capsys.readouterr().err)

    @pytest.mark.parametrize("content", [None, "not JSON", "{}"])
    def test_missing_or_invalid_plan_cannot_construct_http_client(
        self, tmp_path, monkeypatch, capsys, content
    ):
        _forbid_client(monkeypatch)
        cases = _write_cases(tmp_path)
        config = _write_config(tmp_path)
        plan = tmp_path / "absent-or-invalid.json"
        if content is not None:
            plan.write_text(content, encoding="utf-8")
        assert probe.main([
            "--plan", str(plan), "--cases", str(cases), "--config", str(config),
        ]) == 2
        assert "error" in json.loads(capsys.readouterr().err)


def test_changed_ollama_origin_invalidates_plan():
    plan = _prepare()
    with pytest.raises(probe.EvalError):
        _validate(plan, base_url="http://127.0.0.1:11434")


def test_planned_cli_uses_frozen_backend_repeats_timeout_and_keeps_qualification_separate(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("OLLAMA", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    plan_path, cases_path, config_path = _prepare_cli(tmp_path, capsys)
    output = tmp_path / "planned-report.json"
    requests = []
    real_client = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        requests.append((body, request.extensions["timeout"]))
        prompt = json.loads(body["messages"][-1]["content"])["prompt"]
        choice = VIDEO if "protocol-clip.mp4" in prompt else DOCUMENT
        return httpx.Response(200, json=_reply(choice))

    def offline_client(**kwargs):
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(probe.httpx, "AsyncClient", offline_client)
    exit_code = probe.main([
        "--plan", str(plan_path), "--cases", str(cases_path), "--config", str(config_path),
        "--save-json", str(output), "--summary",
    ])
    compact = json.loads(capsys.readouterr().out)
    full = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(requests) == 6
    assert full["provenance"]["backend"] == "hybrid"
    assert full["provenance"]["repeats"] == 3
    assert full["provenance"]["timeout_seconds"] == 20
    assert all(timeout["read"] == 20 for _body, timeout in requests)
    assert all(body["model"] == MODEL for body, _timeout in requests)
    assert full["qualification"]["status"] == "pending_review"
    assert compact["qualification"]["status"] == "pending_review"
    assert full["qualification"]["production_activation"] is False
    assert compact["qualification"]["production_activation"] is False
    assert full["qualification"]["criteria_status"] == "criteria_met"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.parametrize("review", [{"labels_reviewed": True}, {"gates_agreed": True}])
def test_pending_attestation_does_not_hide_quality_findings(review):
    plan = _prepare(**review)
    report, _requests = _evaluate_synthetic(
        decisions=["none", DOCUMENT, DOCUMENT, VIDEO, VIDEO, VIDEO]
    )
    qualification = probe.qualify_report(report, plan)
    assert qualification["status"] == "pending_review"
    assert qualification["criteria_status"] == "findings"
    assert qualification["findings"]
    assert qualification["production_activation"] is False


@pytest.mark.parametrize(
    ("status", "decisions"),
    [
        ("passed", None),
        ("findings", ["none", DOCUMENT, DOCUMENT, VIDEO, VIDEO, VIDEO]),
        ("failed", [httpx.Response(503), DOCUMENT, DOCUMENT, VIDEO, VIDEO, VIDEO]),
    ],
)
def test_compact_measurement_status_is_not_replaced_by_pending_qualification(status, decisions):
    plan = _prepare()
    report, _requests = _evaluate_synthetic(decisions=decisions)
    report["plan"] = plan
    report["qualification"] = probe.qualify_report(report, plan)
    compact = probe.terminal_summary(report)
    assert compact["status"] == status
    assert compact["qualification"]["status"] == "pending_review"
    assert compact["qualification"]["production_activation"] is False


def test_matching_all_guard_evidence_is_explicitly_insufficient_while_review_pending():
    cases = _cases()[2:]
    plan = _prepare(cases)
    report, requests = _evaluate_synthetic(cases)
    assert not requests
    report["plan"] = plan
    report["qualification"] = probe.qualify_report(report, plan)
    compact = probe.terminal_summary(report)
    assert compact["status"] == "passed"
    assert compact["qualification"]["status"] == "pending_review"
    assert compact["qualification"]["criteria_status"] == "insufficient_evidence"
    reasons = compact["qualification"]["insufficient_evidence_reasons"]
    assert "no_model_calls" in reasons
    assert "no_activations" in reasons
    assert compact["qualification"]["checks"]["activation_precision"] is None
    assert compact["qualification"]["production_activation"] is False


def test_compact_plan_is_selective_and_does_not_mutate_complete_full_report():
    plan = _prepare(labels_reviewed=True, gates_agreed=True)
    report, _requests = _evaluate_synthetic()
    report["plan"] = plan
    report["qualification"] = probe.qualify_report(report, plan)
    original = copy.deepcopy(report)
    compact = probe.terminal_summary(report)
    assert set(compact["plan"]) == {"created_at", "fingerprints", "budget"}
    assert set(compact["qualification"]) == {
        "status", "criteria_status", "review", "findings",
        "insufficient_evidence_reasons", "production_activation",
        "final_answer_and_workflow_proof_required", "checks",
    }
    assert compact["qualification"]["checks"] == {
        key: comparison["met"]
        for key, comparison in report["qualification"]["comparisons"].items()
    }
    assert "comparisons" not in compact["qualification"]
    assert "backends" not in compact
    assert "PRIVATE_LABEL_REASON" not in json.dumps(compact)
    assert "PRIVATE_LABEL_REASON" in json.dumps(report)
    assert len(report["backends"]["hybrid"]["samples"]) == 12
    assert report["plan"] == plan
    assert report == original


def test_planned_cli_returns_nonzero_for_insufficient_evidence_even_pending_review(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("OLLAMA", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cases_path = _write_cases(tmp_path, _cases()[2:])
    config_path = _write_config(tmp_path)
    plan_path = tmp_path / "guard-only-plan.json"
    output = tmp_path / "guard-only-report.json"
    assert probe.main([
        "--prepare-plan", str(plan_path), "--cases", str(cases_path),
        "--config", str(config_path), "--backend", "hybrid", "--repeats", "3",
        "--base-url", "http://ollama:11434",
    ]) == 0
    capsys.readouterr()
    requests = []
    real_client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_reply())

    def offline_client(**kwargs):
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(probe.httpx, "AsyncClient", offline_client)
    exit_code = probe.main([
        "--plan", str(plan_path), "--cases", str(cases_path), "--config", str(config_path),
        "--save-json", str(output), "--summary",
    ])
    compact = json.loads(capsys.readouterr().out)
    full = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert not requests
    assert compact["status"] == "passed"
    assert compact["qualification"]["status"] == "pending_review"
    assert compact["qualification"]["criteria_status"] == "insufficient_evidence"
    assert full["qualification"]["criteria_status"] == "insufficient_evidence"
    assert full["plan"]["review"]["labels_reviewed"] is False
    assert full["qualification"]["production_activation"] is False
