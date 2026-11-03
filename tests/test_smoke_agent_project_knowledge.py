"""Installed source-pin smoke detects regressions and removes its fixtures."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "agent_project_knowledge_smoke", Path(__file__).parent / "smoke" / "smoke_agent_project_knowledge.py",
)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


@pytest.fixture
def confined(tmp_path, monkeypatch):
    temporary = tempfile.TemporaryDirectory
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", lambda **kwargs: temporary(dir=tmp_path, **kwargs))
    return tmp_path


async def test_installed_smoke_uses_canonical_state_without_network_and_removes_fixtures(confined, monkeypatch):
    import httpx

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("The isolated Agent source smoke attempted external HTTP.")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    checks = await smoke.run(smoke.load_runtime(), 10)
    assert checks == [name for name, _ in smoke.CASES] + ["isolated_fixture_cleanup"]
    assert list(confined.iterdir()) == []


async def test_smoke_detects_loss_of_source_backed_notes(confined, monkeypatch):
    from audrey.app_state.agent_projects import AgentProjectContexts

    original = AgentProjectContexts.capture

    def empty(contexts, **kwargs):
        original(contexts, **kwargs)
        contexts._conn.execute("UPDATE app_agent_project_contexts SET knowledge_json='[]' WHERE task_id=?", (kwargs["task_id"],))

    monkeypatch.setattr(AgentProjectContexts, "capture", empty)
    with pytest.raises(smoke.SmokeFailureError, match="were not selected"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []


async def test_smoke_detects_a_missing_dispatch_fence(confined, monkeypatch):
    from audrey.app_state.agents import AgentsRepository

    monkeypatch.setattr(smoke, "CASES", (("dispatch_fence", smoke.selected_source_changes_fence_work),))
    monkeypatch.setattr(AgentsRepository, "fence_execution", lambda *_args, **_kwargs: None)
    with pytest.raises(smoke.SmokeFailureError, match="passed the final dispatch fence"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []


async def test_case_deadline_fails_and_removes_all_disposable_state(confined, monkeypatch):
    async def hang(_scenario):
        await asyncio.Event().wait()

    monkeypatch.setattr(smoke, "CASES", (("deadline", hang),))
    with pytest.raises(smoke.SmokeFailureError, match="deadline exceeded"):
        await smoke.run(smoke.load_runtime(), 0.02)
    assert list(confined.iterdir()) == []


def test_old_installed_backend_is_a_setup_failure(monkeypatch):
    from audrey.app_state import agent_projects

    monkeypatch.delattr(agent_projects, "SUPPORTED_KNOWLEDGE_SOURCE_KINDS", raising=False)
    with pytest.raises(smoke.SmokeSetupError, match="rebuild audrey"):
        smoke.load_runtime()


@pytest.mark.parametrize("value", ["0", "nan", "inf", "11"])
def test_invalid_timeout_fails_before_installed_code_or_data_is_loaded(monkeypatch, capsys, value):
    monkeypatch.setattr(smoke, "load_runtime", lambda: pytest.fail("Invalid options loaded application state."))
    assert smoke.main(["--timeout-seconds", value, "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["checks"] == []


@pytest.mark.parametrize("error,code", [
    (smoke.SmokeSetupError("Installed support is missing."), 2),
    (smoke.SmokeFailureError("A fixed contract failed."), 1),
    (RuntimeError("private source and credential details"), 3),
    (KeyboardInterrupt(), 130),
])
def test_terminal_results_are_sanitized_and_use_documented_exit_codes(monkeypatch, capsys, error, code):
    def fail():
        raise error

    monkeypatch.setattr(smoke, "load_runtime", fail)
    assert smoke.main(["--json"]) == code
    output = capsys.readouterr().out
    assert json.loads(output)["exit_code"] == code
    assert "private source" not in output and "credential details" not in output


def test_plain_success_prints_each_check_and_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "load_runtime", lambda: None)

    def done(coroutine):
        coroutine.close()
        return ["isolated_fixture_cleanup"]

    monkeypatch.setattr(smoke.asyncio, "run", done)
    assert smoke.main([]) == 0
    output = capsys.readouterr().out
    assert "PASS isolated_fixture_cleanup" in output and "exit 0" in output
