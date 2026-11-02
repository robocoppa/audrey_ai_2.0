"""The installed source smoke detects failures and confines every write locally."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "source_smoke", Path(__file__).parent / "smoke" / "smoke_knowledge_sources.py",
)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


@pytest.fixture
def confined(tmp_path, monkeypatch):
    temporary = tempfile.TemporaryDirectory
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", lambda **kwargs: temporary(dir=tmp_path, **kwargs))
    return tmp_path


async def test_installed_smoke_uses_real_routes_without_network_or_retained_data(confined, monkeypatch):
    runtime = smoke.load_runtime()

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("A local source check attempted external HTTP.")

    monkeypatch.setattr(runtime.httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    checks = await smoke.run(runtime, 10)
    assert checks == [name for name, _ in smoke.CASES] + ["isolated_fixture_cleanup"]
    assert list(confined.iterdir()) == []


async def test_candidate_scope_smoke_rejects_an_outside_source_on_the_third_page(confined, monkeypatch):
    from audrey.knowledge_source_browser import KnowledgeSourceBrowser

    original = KnowledgeSourceBrowser.snapshot

    def missing_project_filter(browser, principal, project, kind, after, limit):
        values = original(browser, principal, project, kind, after, limit)
        if kind == "message" and project.name == "Source smoke":
            # The first two legitimate items still pass. A lost Project filter
            # leaks the already-created outside fixture only on a third page.
            values.append(("message:msg_outside", "Outside Project message", False, ("outside",)))
            values = sorted((value for value in values if value[0] > after), key=lambda value: value[0])[:limit + 1]
        return values

    monkeypatch.setattr(KnowledgeSourceBrowser, "snapshot", missing_project_filter)
    with pytest.raises(smoke.SmokeFailureError, match="crossed Project scope"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []


async def test_failed_contract_rejects_the_run_and_cleans_all_fixtures(confined, monkeypatch):
    async def fail(scenario):
        await scenario.message("msg_extra", "Disposable draft evidence.")
        raise smoke.SmokeFailureError("Injected contract failure.")

    monkeypatch.setattr(smoke, "CASES", (("injected", fail),))
    with pytest.raises(smoke.SmokeFailureError, match="Injected contract"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []


async def test_case_timeout_fails_and_removes_disposable_databases(confined, monkeypatch):
    async def hang(_scenario):
        await asyncio.Event().wait()

    monkeypatch.setattr(smoke, "CASES", (("deadline", hang),))
    with pytest.raises(smoke.SmokeFailureError, match="deadline exceeded"):
        await smoke.run(smoke.load_runtime(), 0.02)
    assert list(confined.iterdir()) == []


@pytest.mark.parametrize("value", ["0", "nan", "inf", "11"])
def test_invalid_timeout_is_setup_error_before_runtime_loading(monkeypatch, capsys, value):
    monkeypatch.setattr(smoke, "load_runtime", lambda: pytest.fail("Invalid options loaded application state."))
    assert smoke.main(["--timeout-seconds", value, "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["checks"] == []


@pytest.mark.parametrize("error,code", [
    (smoke.SmokeSetupError("Installed support is missing."), 2),
    (smoke.SmokeFailureError("A fixed contract failed."), 1),
    (RuntimeError("sensitive source path or credential"), 3),
    (KeyboardInterrupt(), 130),
])
def test_failure_exits_are_bounded_and_unexpected_details_are_sanitized(monkeypatch, capsys, error, code):
    def fail():
        raise error

    monkeypatch.setattr(smoke, "load_runtime", fail)
    assert smoke.main(["--json"]) == code
    output = capsys.readouterr().out
    assert json.loads(output)["exit_code"] == code
    assert "sensitive" not in output


def test_plain_success_prints_each_check_and_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "load_runtime", lambda: None)

    def done(coroutine):
        coroutine.close()
        return ["isolated_fixture_cleanup"]

    monkeypatch.setattr(smoke.asyncio, "run", done)
    assert smoke.main([]) == 0
    output = capsys.readouterr().out
    assert "PASS isolated_fixture_cleanup" in output and "exit 0" in output
