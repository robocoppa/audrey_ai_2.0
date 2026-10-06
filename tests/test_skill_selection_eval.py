"""Hermetic checks for the offline-only automatic skill selection study."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import stat
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "evals/eval_skill_selection.py"
_SPEC = importlib.util.spec_from_file_location("skill_selection_eval", _SCRIPT)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


DOCUMENT_SKILL = "grounded-document-analysis"
VIDEO_SKILL = "video-analysis"


def _case(
    *,
    case_id: str = "document-evidence",
    category: str = "positive",
    prompt: str = "Summarize the attached report.pdf and quote its main conclusion.",
    mode: str = "auto",
    files: tuple | None = None,
    expected: str = DOCUMENT_SKILL,
    reason: str = "The user explicitly requests analysis of ready document evidence.",
):
    return probe.Case(
        id=case_id,
        category=category,
        prompt=prompt,
        mode=mode,
        files=files
        if files is not None
        else (probe.FileEvidence(name="report.pdf", kind="document", status="ready"),),
        expected=expected,
        reason=reason,
    )


def _dataset_case(**changes):
    row = {
        "id": "document-evidence",
        "category": "positive",
        "prompt": "Summarize the attached report.pdf.",
        "mode": "auto",
        "files": [{"name": "report.pdf", "kind": "document", "status": "ready"}],
        "expected": DOCUMENT_SKILL,
        "reason": "Ready document evidence is explicitly requested.",
    }
    row.update(changes)
    return row


def _write_cases(tmp_path, rows):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema": 1, "description": "Hermetic test cases", "cases": rows}), encoding="utf-8")
    return path


class TestDatasetBoundaries:
    def test_loader_preserves_explicit_evidence_without_reading_any_files(self, tmp_path):
        path = _write_cases(tmp_path, [_dataset_case()])
        assert not (tmp_path / "report.pdf").exists()
        case = probe.load_cases(path)[0]
        assert case.expected == DOCUMENT_SKILL
        assert case.files == (
            probe.FileEvidence(name="report.pdf", kind="document", status="ready"),
        )

    def test_rejects_duplicate_case_ids(self, tmp_path):
        row = _dataset_case()
        with pytest.raises(probe.EvalError, match="duplicate"):
            probe.load_cases(_write_cases(tmp_path, [row, row]))

    def test_rejects_unknown_labels_instead_of_expanding_the_catalog(self, tmp_path):
        row = _dataset_case(expected="run-shell-commands")
        with pytest.raises(probe.EvalError):
            probe.load_cases(_write_cases(tmp_path, [row]))

    @pytest.mark.parametrize(
        "changes",
        [
            {"prompt": ""},
            {"reason": ""},
            {"category": "mystery"},
            {"category": []},
            {"expected": []},
            {"mode": []},
            {"files": [{"name": "report.pdf", "kind": {}, "status": "ready"}]},
            {"files": [{"name": "report.pdf", "kind": "document", "status": []}]},
            {"files": "report.pdf"},
            {"files": [{"name": "report.pdf", "kind": "document"}]},
            {"extra": "unreviewed fixture field"},
        ],
    )
    def test_rejects_malformed_or_unreviewed_cases(self, tmp_path, changes):
        with pytest.raises(probe.EvalError):
            probe.load_cases(_write_cases(tmp_path, [_dataset_case(**changes)]))

    @pytest.mark.parametrize("category", ["ordinary", "ambiguous"])
    def test_negative_categories_cannot_be_mislabeled_as_skill_positives(
        self, tmp_path, category
    ):
        row = _dataset_case(category=category, expected=DOCUMENT_SKILL)
        with pytest.raises(probe.EvalError):
            probe.load_cases(_write_cases(tmp_path, [row]))

    def test_positive_category_requires_a_skill_label(self, tmp_path):
        row = _dataset_case(expected="none")
        with pytest.raises(probe.EvalError):
            probe.load_cases(_write_cases(tmp_path, [row]))


class TestPromptPrivacy:
    def test_gold_labels_and_rationales_never_reach_the_router(self):
        case = _case(reason="GOLD_REASON_CANARY_4F0284")
        messages = probe.build_router_messages(case)
        body = json.dumps(messages)
        assert "GOLD_REASON_CANARY_4F0284" not in body
        assert "report.pdf" in body
        assert case.prompt in body
        assert all(message["role"] in {"system", "user"} for message in messages)

    def test_changing_gold_label_does_not_change_the_request(self):
        case = _case()
        other = replace(case, expected=VIDEO_SKILL, reason="A different private judgment.")
        assert probe.build_router_messages(case) == probe.build_router_messages(other)

    def test_skill_bodies_and_resources_are_excluded_from_metadata_prompt(self):
        catalog = [
            {
                **entry,
                "instructions": "SKILL_BODY_CANARY_D621",
                "resources": [{"content": "RESOURCE_BODY_CANARY_E726"}],
            }
            for entry in probe.DEFAULT_CATALOG
        ]
        body = json.dumps(probe.build_router_messages(_case(), catalog=catalog))
        assert "SKILL_BODY_CANARY_D621" not in body
        assert "RESOURCE_BODY_CANARY_E726" not in body


class TestReportPersistence:
    def test_results_are_private_and_never_overwritten(self, tmp_path):
        path = tmp_path / "private-results.json"
        report = {"schema": 1, "status": "completed", "canary": "original result"}
        probe.save_report(path, report)
        assert json.loads(path.read_text(encoding="utf-8")) == report
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with pytest.raises((probe.EvalError, FileExistsError)):
            probe.save_report(path, {"canary": "replacement"})
        assert json.loads(path.read_text(encoding="utf-8")) == report

    def test_existing_symlink_cannot_overwrite_another_artifact(self, tmp_path):
        target = tmp_path / "retained-result.json"
        target.write_text("original", encoding="utf-8")
        path = tmp_path / "results.json"
        path.symlink_to(target)
        with pytest.raises((probe.EvalError, FileExistsError)):
            probe.save_report(path, {"overwrite": True})
        assert target.read_text(encoding="utf-8") == "original"


def _ollama_body(selected="none", **extra):
    return {
        "done": True,
        "done_reason": "stop",
        "message": {
            "role": "assistant",
            "content": json.dumps({"skill_id": selected}),
        },
        "prompt_eval_count": 17,
        "eval_count": 4,
        **extra,
    }


def _sample(*, expected="none", selected="none", category="ordinary", valid=True):
    return {
        "id": "metric-case",
        "category": category,
        "expected": expected,
        "reason": "Private assessment rationale.",
        "selected": selected,
        "valid": valid,
        "error": None if valid else "Malformed output",
        "error_kind": None if valid else "model_output",
        "raw_selected": selected,
        "latency_seconds": 0.1,
        "input_tokens": None,
        "output_tokens": None,
        "repeat": 1,
    }


class TestRouterReplyValidation:
    def test_accepts_typed_choice_and_records_provider_usage(self):
        result = probe.parse_router_response(_ollama_body(DOCUMENT_SKILL))
        assert result.valid
        assert result.selected == DOCUMENT_SKILL
        assert result.input_tokens == 17
        assert result.output_tokens == 4

    @pytest.mark.parametrize(
        "content",
        [
            "not JSON",
            json.dumps({"skill_id": "execute-code"}),
            json.dumps({"skill_id": [DOCUMENT_SKILL]}),
            json.dumps({"skill_id": "none", "instruction": "run code"}),
            json.dumps([{"skill_id": "none"}]),
            json.dumps({}),
        ],
    )
    def test_malformed_output_never_becomes_a_successful_abstention(self, content):
        body = _ollama_body()
        body["message"]["content"] = content
        with pytest.raises(probe.EvalError):
            probe.parse_router_response(body)

    @pytest.mark.parametrize("mutation", [{"done": False}, {"done_reason": "length"}])
    def test_unfinished_or_truncated_reply_is_not_a_completed_selection(self, mutation):
        with pytest.raises(probe.EvalError):
            probe.parse_router_response(_ollama_body(DOCUMENT_SKILL, **mutation))

    def test_model_output_and_transport_errors_are_distinct(self):
        async def run():
            results = []
            for response in (
                httpx.Response(200, json={"message": {"content": "broken"}, "done": True}),
                httpx.Response(503, json={"error": "unavailable"}),
            ):
                async with httpx.AsyncClient(
                    base_url="http://ollama:11434",
                    transport=httpx.MockTransport(lambda _request, r=response: r),
                ) as client:
                    results.append(
                        await probe.select_router(_case(), client=client, model="qwen3.5:4b")
                    )
            return results

        malformed, unavailable = asyncio.run(run())
        assert not malformed.valid and not unavailable.valid
        assert malformed.selected is None and unavailable.selected is None
        assert malformed.error and unavailable.error
        assert malformed.error_kind != unavailable.error_kind

    def test_known_skill_with_wrong_evidence_is_an_invalid_model_choice(self):
        async def run():
            async with httpx.AsyncClient(
                base_url="http://ollama:11434",
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(200, json=_ollama_body(VIDEO_SKILL))
                ),
            ) as client:
                return await probe.select_router(_case(), client=client, model="qwen3.5:4b")

        result = asyncio.run(run())
        assert not result.valid
        assert result.selected is None
        assert result.raw_selected == VIDEO_SKILL
        assert result.error_kind == "ineligible_choice"


class TestSelectionAccounting:
    def test_wrong_skill_counts_as_both_false_activation_and_missed_correct_skill(self):
        summary = probe.summarize_samples(
            [_sample(expected=DOCUMENT_SKILL, selected=VIDEO_SKILL, category="positive")]
        )
        assert summary["activations"] == 1
        assert summary["correct_activations"] == 0
        assert summary["false_activations"] == 1
        assert summary["wrong_skill"] == 1
        assert summary["misses"] == 1
        assert summary["precision"] == 0

    def test_zero_activations_has_no_precision_denominator(self):
        summary = probe.summarize_samples(
            [_sample(expected=DOCUMENT_SKILL, selected="none", category="positive")]
        )
        assert summary["activations"] == 0
        assert summary["precision_denominator"] == 0
        assert summary["precision"] is None
        assert summary["misses"] == 1
        assert summary["activation_rate"] == 0
        assert summary["useful_selection_recall"] == 0

    def test_ordinary_false_activations_are_separate_from_ambiguous_controls(self):
        summary = probe.summarize_samples(
            [
                _sample(selected=DOCUMENT_SKILL),
                _sample(selected=VIDEO_SKILL, category="ambiguous"),
                _sample(),
            ]
        )
        assert summary["false_activations_nonpositive"] == 2
        assert summary["ordinary_false_activations"] == 1
        assert summary["ordinary_total"] == 2
        assert summary["ordinary_false_activation_rate"] == 0.5

    def test_failed_negative_response_is_not_counted_as_correct_none(self):
        summary = probe.summarize_samples(
            [_sample(selected=None, valid=False), _sample()]
        )
        assert summary["total"] == 2
        assert summary["errors"] == 1
        assert summary["valid"] == 1
        assert summary["correct"] == 1
        assert summary["abstentions"] == 1
        assert summary["error_rate"] == 0.5


class TestEvaluatorBehavior:
    def test_repeats_remain_serial_and_preserve_every_case_observation(self):
        calls = 0
        active = 0
        max_active = 0

        async def handler(_request):
            nonlocal calls, active, max_active
            calls += 1
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1
            return httpx.Response(200, json=_ollama_body(DOCUMENT_SKILL))

        async def run():
            async with httpx.AsyncClient(
                base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
            ) as client:
                return await probe.evaluate(
                    [_case()], backend="router", client=client, model="qwen3.5:4b", repeats=3
                )

        report = asyncio.run(run())
        assert calls == 3
        assert max_active == 1
        assert set(report["backends"]) >= {"rules", "router"}
        rows = report["backends"]["router"]["samples"]
        assert len(rows) == 3
        assert len({row["repeat"] for row in rows}) == 3
        assert report["backends"]["router"]["summary"]["total"] == 3
        assert report["backends"]["rules"]["summary"]["total"] == 3

    def test_partial_model_failure_marks_the_run_as_failed_and_keeps_good_results(self):
        calls = 0

        def handler(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, json=_ollama_body(DOCUMENT_SKILL))
            return httpx.Response(200, json=_ollama_body(message={"content": "broken"}))

        async def run():
            async with httpx.AsyncClient(
                base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
            ) as client:
                return await probe.evaluate(
                    [_case(), _case(case_id="second-document")],
                    backend="router",
                    client=client,
                    model="qwen3.5:4b",
                )

        report = asyncio.run(run())
        assert report["status"] == "failed"
        summary = report["backends"]["router"]["summary"]
        assert summary["total"] == 2
        assert summary["valid"] == 1
        assert summary["errors"] == 1
        assert summary["correct"] == 1

    @pytest.mark.parametrize(
        "case",
        [
            _case(mode="research", expected="none", category="ambiguous"),
            _case(files=(), expected="none", category="ambiguous"),
            _case(
                files=(probe.FileEvidence("report.pdf", "document", "pending"),),
                expected="none",
                category="ambiguous",
            ),
            _case(
                files=(probe.FileEvidence("report.pdf", "document", "failed"),),
                expected="none",
                category="ambiguous",
            ),
            _case(
                files=(probe.FileEvidence("other.pdf", "document", "ready"),),
                expected="none",
                category="ambiguous",
            ),
            _case(
                prompt="Summarize report.pdf and clip.mp4 using both files.",
                files=(
                    probe.FileEvidence("report.pdf", "document", "ready"),
                    probe.FileEvidence("clip.mp4", "video", "ready"),
                ),
                expected="none",
                category="ambiguous",
            ),
        ],
        ids=["unsupported-mode", "no-files", "pending", "failed", "unmatched", "mixed"],
    )
    @pytest.mark.parametrize("backend", ["router", "hybrid"])
    def test_hard_guards_do_not_call_model_or_claim_model_success(self, case, backend):
        calls = 0

        def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_ollama_body(DOCUMENT_SKILL))

        async def run():
            async with httpx.AsyncClient(
                base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
            ) as client:
                return await probe.evaluate(
                    [case], backend=backend, client=client, model="qwen3.5:4b"
                )

        report = asyncio.run(run())
        assert calls == 0
        row = report["backends"][backend]["samples"][0]
        assert row["selected"] == "none"
        assert row["source"] == "guard"
        assert row["model_called"] is False

    def test_hybrid_retains_rule_decision_without_calling_model(self):
        case = _case()
        assert probe.select_rules(case).selected == DOCUMENT_SKILL
        calls = 0

        def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_ollama_body(VIDEO_SKILL))

        async def run():
            async with httpx.AsyncClient(
                base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
            ) as client:
                return await probe.evaluate(
                    [case], backend="hybrid", client=client, model="qwen3.5:4b"
                )

        report = asyncio.run(run())
        assert calls == 0
        row = report["backends"]["hybrid"]["samples"][0]
        assert row["selected"] == DOCUMENT_SKILL
        assert row["source"] == "rules"
        assert row["model_called"] is False


class TestRuntimeIsolation:
    def test_runtime_auto_selection_must_remain_disabled(self, tmp_path):
        import yaml

        config = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))
        config["skills"]["auto_select"] = True
        path = tmp_path / "enabled-auto-select.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(probe.EvalError, match="auto_select"):
            probe.load_live_settings(
                path, model="qwen3.5:4b", base_url="http://ollama:11434"
            )

    def test_cloud_models_are_rejected_before_requests(self, tmp_path):
        import yaml

        config = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))
        config["skills"]["auto_select"] = False
        path = tmp_path / "offline-study.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(probe.EvalError):
            probe.load_live_settings(
                path,
                model="kimi-k3:cloud",
                base_url="http://ollama:11434",
                allow_nonretained_model=True,
            )


def _write_live_config(tmp_path, *, auto_select=False):
    import yaml

    config = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["skills"]["auto_select"] = auto_select
    config["skills"]["roots"] = [str(_ROOT / "skills")]
    path = tmp_path / "study-config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_false_activations_increase_activation_rate_but_never_useful_recall():
    positive = _sample(expected=DOCUMENT_SKILL, selected=DOCUMENT_SKILL, category="positive")
    baseline = probe.summarize_samples([positive, _sample()])
    wrong = probe.summarize_samples([positive, _sample(selected=VIDEO_SKILL)])
    assert baseline["activation_rate"] == 0.5
    assert wrong["activation_rate"] == 1
    assert baseline["useful_selection_recall"] == wrong["useful_selection_recall"] == 1
    assert baseline["precision"] == 1
    assert wrong["precision"] == 0.5


def test_hybrid_calls_model_only_for_eligible_rule_abstention_and_labels_subset():
    english = _case()
    spanish = _case(
        case_id="document-spanish",
        prompt="Resume el documento report.pdf en tres frases con sus principales conclusiones.",
    )
    assert probe.select_rules(english).selected == DOCUMENT_SKILL
    assert probe.select_rules(spanish).selected == "none"
    assert probe.eligible_skills(spanish) == {DOCUMENT_SKILL}
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_ollama_body(DOCUMENT_SKILL))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                [english, spanish], backend="hybrid", client=client, model="qwen3.5:4b"
            )

    report = asyncio.run(run())
    assert calls == 1
    assert set(report["backends"]) == {"rules", "hybrid"}
    rows = report["backends"]["hybrid"]["samples"]
    assert rows[0]["source"] == "rules" and rows[0]["model_called"] is False
    assert rows[1]["source"] == "router" and rows[1]["model_called"] is True
    assert report["conditional_router"]["summary"]["total"] == 1
    assert "not a full router comparison" in report["conditional_router"]["population"]
    assert report["provenance"]["automatic_selection_enabled_in_harness"] is False
    assert report["provenance"]["evaluation_only"] is True


def test_hybrid_invalid_model_reply_is_not_a_valid_rule_abstention():
    spanish = _case(prompt="Resume el documento report.pdf en tres frases.")
    assert probe.select_rules(spanish).selected == "none"

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=_ollama_body(message={"content": "broken"}))
            ),
        ) as client:
            return await probe.evaluate(
                [spanish], backend="hybrid", client=client, model="qwen3.5:4b"
            )

    report = asyncio.run(run())
    assert report["status"] == "failed"
    summary = report["backends"]["hybrid"]["summary"]
    assert summary["errors"] == 1
    assert summary["misses"] == 1
    assert summary["correct"] == 0
    assert summary["abstentions"] == 0


def test_cli_auto_select_guard_runs_before_constructing_http_client(
    tmp_path, monkeypatch, capsys
):
    path = _write_live_config(tmp_path, auto_select=True)

    def forbidden_client(**_kwargs):
        pytest.fail("Runtime isolation guard must reject before constructing an HTTP client")

    monkeypatch.setattr(probe.httpx, "AsyncClient", forbidden_client)
    exit_code = probe.main(["--backend", "router", "--config", str(path)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "auto_select" in json.loads(captured.err)["error"]


def test_cli_partial_failure_saves_report_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    config = _write_live_config(tmp_path)
    cases = _write_cases(
        tmp_path, [_dataset_case(), _dataset_case(id="second-document")]
    )
    output = tmp_path / "partial-results.json"
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=_ollama_body(DOCUMENT_SKILL))
        return httpx.Response(503, json={"error": "model unavailable"})

    real_client = httpx.AsyncClient

    def offline_client(**kwargs):
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(probe.httpx, "AsyncClient", offline_client)
    exit_code = probe.main(
        [
            "--backend", "router",
            "--config", str(config),
            "--cases", str(cases),
            "--repeats", "1",
            "--save-json", str(output),
        ]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 1
    assert report["status"] == "failed"
    assert calls == 2
    assert report["backends"]["router"]["summary"]["errors"] == 1
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
