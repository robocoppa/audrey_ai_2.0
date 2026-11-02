"""Installed report-suggestion checks execute real code and erase every fixture."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_draft_smoke", Path(__file__).parent / "smoke" / "smoke_agent_knowledge_drafts.py",
)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


@pytest.fixture
def confined(tmp_path, monkeypatch):
    temporary = tempfile.TemporaryDirectory
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", lambda **kwargs: temporary(dir=tmp_path, **kwargs))
    return tmp_path


async def test_installed_smoke_uses_real_routes_service_and_client_without_network(confined, monkeypatch):
    runtime = smoke.load_runtime()

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("The isolated suggestion smoke attempted an external socket.")

    monkeypatch.setattr(runtime.httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    checks = await smoke.run(runtime, 10)
    assert checks == [name for name, _ in smoke.CASES] + ["isolated_fixture_cleanup"]
    assert list(confined.iterdir()) == []


async def test_smoke_detects_an_unrequested_persistent_knowledge_write(confined, monkeypatch):
    from audrey.agent_knowledge_drafts import ReportKnowledgeDrafts

    original = ReportKnowledgeDrafts.generate

    async def persisted(service, **scope):
        result = await original(service, **scope)
        store = service.app.state.application_store
        with store._lock:
            project_id = store._conn.execute("SELECT project_id FROM app_projects LIMIT 1").fetchone()[0]
        await store.knowledge.create(user_id=scope["user_id"], project_id=project_id,
                                     kind="fact", text="An unrequested saved suggestion.")
        return result

    monkeypatch.setattr(ReportKnowledgeDrafts, "generate", persisted)
    with pytest.raises(smoke.SmokeFailureError, match="mutated saved tasks"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []


async def test_smoke_rejects_a_parser_that_partially_accepts_an_invalid_batch(confined, monkeypatch):
    from audrey import agent_knowledge_drafts

    original = agent_knowledge_drafts._parse

    def partial(response):
        suggestions = json.loads(response["message"]["content"])["suggestions"]
        if any(item["kind"] == "unsupported" for item in suggestions):
            return [suggestions[0]]
        return original(response)

    monkeypatch.setattr(agent_knowledge_drafts, "_parse", partial)
    with pytest.raises(smoke.SmokeFailureError, match="expected HTTP 503, received 200"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []


async def test_case_deadline_fails_and_cleans_all_disposable_state(confined, monkeypatch):
    async def hang(_scenario):
        await asyncio.Event().wait()

    monkeypatch.setattr(smoke, "CASES", (("deadline", hang),))
    with pytest.raises(smoke.SmokeFailureError, match="deadline exceeded"):
        await smoke.run(smoke.load_runtime(), 0.02)
    assert list(confined.iterdir()) == []


@pytest.mark.parametrize("value", ["0", "nan", "inf", "11"])
def test_invalid_timeout_fails_before_installed_code_or_data_is_loaded(monkeypatch, capsys, value):
    monkeypatch.setattr(smoke, "load_runtime", lambda: pytest.fail("Invalid options loaded application state."))
    assert smoke.main(["--timeout-seconds", value, "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["checks"] == []


@pytest.mark.parametrize("error,code", [
    (smoke.SmokeSetupError("Installed support is missing."), 2),
    (smoke.SmokeFailureError("A fixed contract failed."), 1),
    (RuntimeError("private report and credential details"), 3),
    (KeyboardInterrupt(), 130),
])
def test_terminal_results_are_sanitized_and_use_documented_exit_codes(monkeypatch, capsys, error, code):
    def fail():
        raise error

    monkeypatch.setattr(smoke, "load_runtime", fail)
    assert smoke.main(["--json"]) == code
    output = capsys.readouterr().out
    assert json.loads(output)["exit_code"] == code
    assert "private report" not in output and "credential details" not in output


def test_plain_success_prints_each_check_and_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "load_runtime", lambda: None)

    def done(coroutine):
        coroutine.close()
        return ["isolated_fixture_cleanup"]

    monkeypatch.setattr(smoke.asyncio, "run", done)
    assert smoke.main([]) == 0
    output = capsys.readouterr().out
    assert "PASS isolated_fixture_cleanup" in output and "exit 0" in output
