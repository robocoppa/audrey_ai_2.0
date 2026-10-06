"""CLI results separate execution from proposed-label findings, without live calls."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import httpx
import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "skill_selection_reporting_eval", _ROOT / "evals/eval_skill_selection.py"
)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

DOCUMENT = "grounded-document-analysis"


def _inputs(tmp_path, *, expected=DOCUMENT):
    config = yaml.safe_load((_ROOT / "config.yaml").read_text())
    config["skills"]["auto_select"] = False
    config["skills"]["roots"] = [str(_ROOT / "skills")]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({
        "schema": 1, "description": "Proposed regression labels",
        "cases": [{
            "id": "spanish-content", "category": "positive" if expected != "none" else "ordinary",
            "prompt": "¿Qué obligaciones aparecen en brief.pdf?" if expected != "none" else "Please say hello. brief.pdf is available.", "mode": "fast",
            "files": [{"name": "brief.pdf", "kind": "document", "status": "ready"}],
            "expected": expected, "reason": "GOLD_REASON_MUST_NOT_ENTER_TERMINAL_SUMMARY",
        }],
    }))
    return config_path, cases_path


def _mock_client(monkeypatch, *, choice=DOCUMENT, failure=False):
    calls = []
    real_client = httpx.AsyncClient

    def handler(request):
        calls.append(json.loads(request.content))
        if failure:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={
            "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({"skill_id": choice})},
            "prompt_eval_count": 73, "eval_count": 11,
        })

    monkeypatch.setattr(probe.httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(handler)
    ))
    return calls


@pytest.mark.parametrize("compact", [False, True])
def test_hybrid_label_check_uses_selected_arm_and_never_counts_rules_miss_as_failure(
    tmp_path, monkeypatch, capsys, compact
):
    config, cases = _inputs(tmp_path)
    calls = _mock_client(monkeypatch)
    args = ["--backend", "hybrid", "--cases", str(cases), "--config", str(config), "--repeats", "1"]
    rc = probe.main(args + (["--summary"] if compact else []))
    result = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(calls) == 1
    if compact:
        assert result["status"] == "passed"
        assert result["measurement_status"] == "completed"
        assert result["selection"]["correct"] == 1
        assert result["findings"] == []
    else:
        assert result["status"] == "completed"
        assert result["backends"]["rules"]["summary"]["misses"] == 1
        assert result["selection_check"] == {
            "backend": "hybrid", "status": "matched_labels", "finding_count": 0,
            "scope": "proposed synthetic labels; not production acceptance",
        }


@pytest.mark.parametrize("compact", [False, True])
def test_valid_wrong_selection_exits_nonzero_even_when_execution_completed(
    tmp_path, monkeypatch, capsys, compact
):
    config, cases = _inputs(tmp_path, expected="none")
    calls = _mock_client(monkeypatch)
    rc = probe.main([
        "--backend", "router", "--cases", str(cases), "--config", str(config), "--repeats", "1",
        *(["--summary"] if compact else []),
    ])
    result = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert len(calls) == 1
    if compact:
        assert result["status"] == "findings"
        assert result["measurement_status"] == "completed"
        assert result["selection"]["ordinary_false_activations"] == 1
        assert result["execution"]["errors"] == 0
        assert result["findings"][0]["id"] == "spanish-content"
        assert result["findings"][0]["selected"] == DOCUMENT
    else:
        assert result["status"] == "completed"
        assert result["selection_check"]["status"] == "findings"
        assert result["selection_check"]["finding_count"] == 1


@pytest.mark.parametrize("failure", [False, True])
def test_compact_result_keeps_measured_cost_errors_and_full_private_artifact(
    tmp_path, monkeypatch, capsys, failure
):
    config, cases = _inputs(tmp_path)
    calls = _mock_client(monkeypatch, failure=failure)
    artifact = tmp_path / "full-results.json"
    rc = probe.main([
        "--backend", "hybrid", "--cases", str(cases), "--config", str(config), "--repeats", "1",
        "--summary", "--save-json", str(artifact),
    ])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    full = json.loads(artifact.read_text())
    assert rc == (1 if failure else 0)
    assert len(calls) == 1
    assert result["full_report"] == str(artifact)
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert result["provenance"]["case_set_sha256"] == full["provenance"]["case_set_sha256"]
    assert result["provenance"]["automatic_selection_enabled_in_harness"] is False
    assert "backends" not in result
    assert "GOLD_REASON_MUST_NOT_ENTER_TERMINAL_SUMMARY" not in captured.out
    assert "GOLD_REASON_MUST_NOT_ENTER_TERMINAL_SUMMARY" in artifact.read_text()
    assert result["execution"]["model_called"] == 1
    assert result["execution"]["model_latency_p50_seconds"] is not None
    if failure:
        assert result["status"] == "failed"
        assert result["measurement_status"] == "failed"
        assert result["execution"]["errors"] == 1
        assert result["execution"]["input_tokens"]["samples"] == 0
        assert result["findings"][0]["valid"] is False
        assert result["findings"][0]["error_kind"] == "transport"
    else:
        assert result["status"] == "passed"
        assert result["execution"]["input_tokens"]["total"] == 73
        assert result["execution"]["output_tokens"]["total"] == 11


def test_rules_only_positive_abstention_is_reported_as_finding_without_http(
    tmp_path, monkeypatch, capsys
):
    _, cases = _inputs(tmp_path)
    monkeypatch.setattr(probe.httpx, "AsyncClient", lambda **_: pytest.fail("Offline rules opened HTTP"))
    rc = probe.main(["--cases", str(cases), "--summary"])
    result = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert result["status"] == "findings"
    assert result["measurement_status"] == "completed"
    assert result["selection"]["misses"] == 1
    assert result["execution"]["model_called"] == 0
    assert result["execution"]["model_latency_p50_seconds"] is None
    assert result["findings"][0]["selected"] == "none"
