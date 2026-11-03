"""The installed selected-file read probe exercises canonical sources and detects regressions."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("agent_file_reads_smoke", Path(__file__).parent / "smoke" / "smoke_agent_file_reads.py")
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


@pytest.fixture
def confined(tmp_path, monkeypatch):
    temporary = tempfile.TemporaryDirectory
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", lambda **kwargs: temporary(dir=tmp_path, **kwargs))
    return tmp_path


async def test_installed_file_read_smoke_uses_real_canonical_storage_without_external_http_and_cleans_fixtures(confined, monkeypatch):
    import httpx

    async def forbidden(*_args, **_kwargs):
        pytest.fail("The isolated Agent file read smoke attempted external HTTP.")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    checks = await smoke.run(smoke.load_runtime(), 10)
    assert checks == [name for name, _ in smoke.CASES] + ["isolated_fixture_cleanup"]
    assert list(confined.iterdir()) == []


async def test_probe_detects_missing_native_read_results(confined, monkeypatch):
    runtime = smoke.load_runtime()

    async def missing(*_args, **_kwargs):
        return {"error": "missing_native_read"}

    monkeypatch.setattr(runtime.AgentFileInputs, "read_selected", missing)
    monkeypatch.setattr(smoke, "CASES", (("execution", smoke.local_agent_reads_later_pages_and_saves_output),))
    with pytest.raises(smoke.SmokeFailureError, match="did not complete its native read-and-save task"):
        await smoke.run(runtime, 10)
    assert list(confined.iterdir()) == []


async def test_probe_detects_untruthful_cursor_metadata(confined, monkeypatch):
    runtime = smoke.load_runtime()
    original = runtime.AgentFileInputs.read_selected

    async def bad_cursor(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        return {**result, "next_offset": result["end"] + 1} if result.get("has_more") else result

    monkeypatch.setattr(runtime.AgentFileInputs, "read_selected", bad_cursor)
    monkeypatch.setattr(smoke, "CASES", (("cursors", smoke.json_unicode_cursors_and_argument_bounds),))
    with pytest.raises(smoke.SmokeFailureError, match="untruthful next offset"):
        await smoke.run(runtime, 10)
    assert list(confined.iterdir()) == []


async def test_probe_detects_reading_a_file_outside_the_selection(confined, monkeypatch):
    runtime = smoke.load_runtime()
    original = runtime.AgentFileInputs.read_selected

    async def ignores_selection(self, principal, task, arguments, **kwargs):
        selected = task["file_inputs"]["files"][0]["file_id"]
        return await original(self, principal, task, {**arguments, "file_id": selected}, **kwargs)

    monkeypatch.setattr(runtime.AgentFileInputs, "read_selected", ignores_selection)
    monkeypatch.setattr(smoke, "CASES", (("selection", smoke.selected_owner_private_and_scope_fences),))
    with pytest.raises(smoke.SmokeFailureError, match="unauthorized or stale read"):
        await smoke.run(runtime, 10)
    assert list(confined.iterdir()) == []


async def test_probe_detects_missing_real_read_events(confined, monkeypatch):
    from audrey.app_state.agents import AgentsRepository

    original = AgentsRepository.events

    async def missing(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        return {**result, "items": []}

    monkeypatch.setattr(AgentsRepository, "events", missing)
    monkeypatch.setattr(smoke, "CASES", (("events", smoke.read_events_keep_metadata_without_bodies),))
    with pytest.raises(smoke.SmokeFailureError, match="missing or misreported in execution history"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []

async def test_case_timeout_removes_disposable_storage(confined, monkeypatch):
    async def hang(_scenario):
        await asyncio.Event().wait()

    monkeypatch.setattr(smoke, "CASES", (("timeout", hang),))
    with pytest.raises(smoke.SmokeFailureError, match="deadline exceeded"):
        await smoke.run(smoke.load_runtime(), 0.02)
    assert list(confined.iterdir()) == []


@pytest.mark.parametrize("blocking_thread", [False, True])
def test_cli_watchdog_stops_lock_deadlocks_and_cleans_parent_owned_fixtures(tmp_path, blocking_thread):
    command = textwrap.dedent("""\
        import asyncio
        import importlib.util
        import multiprocessing
        import sys
        import tempfile
        import threading
        import time
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("isolated_file_read_smoke", sys.argv[1])
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        root, mode = Path(sys.argv[2]), sys.argv[3]
        temporary = tempfile.TemporaryDirectory
        smoke.tempfile.TemporaryDirectory = lambda **kwargs: temporary(dir=root, **kwargs)
        runtime = smoke.load_runtime()
        smoke.load_runtime = lambda: runtime

        async def deadlock(scenario):
            def block():
                (root / "entered").write_text(str(scenario.root))
                (scenario.root / "unfinished-output.bin").write_bytes(b"fixture")
                lock = threading.Lock()
                lock.acquire()
                lock.acquire()
            if mode == "thread":
                await asyncio.to_thread(block)
            else:
                block()

        smoke.CASES = (("locked_case", deadlock),)
        started = time.monotonic()
        code = smoke.main(["--json", "--timeout-seconds", "1"])
        assert time.monotonic() - started < 4
        assert not multiprocessing.active_children()
        sys.exit(code)
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", command, str(_SPEC.origin), str(tmp_path),
         "thread" if blocking_thread else "event_loop"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("The CLI watchdog did not stop its isolated deadlocked case.")
    assert process.returncode == 1, stderr
    result = json.loads(stdout)
    assert result == {"exit_code": 1, "checks": [], "message": "locked_case deadline exceeded."}
    entered = tmp_path / "entered"
    assert entered.is_file(), "The case never reached the synchronous lock wait."
    assert not Path(entered.read_text()).parent.exists()
    assert str(tmp_path) not in stdout
    entered.unlink()
    assert list(tmp_path.iterdir()) == []


def test_cli_child_crashes_are_sanitized_and_disposable_data_is_removed(confined, monkeypatch, capsys):
    async def crash(_scenario):
        raise RuntimeError("private filename or credential value")

    monkeypatch.setattr(smoke, "CASES", (("crash", crash),))
    assert smoke.main(["--json"]) == 3
    output = capsys.readouterr().out
    assert json.loads(output) == {"exit_code": 3, "checks": [], "message": "Agent file read smoke harness crashed."}
    assert "private filename" not in output and "credential value" not in output
    assert list(confined.iterdir()) == []


def test_older_backend_is_rejected_before_creating_fixture_data(confined, monkeypatch):
    from audrey.agent_file_inputs import AgentFileInputs

    monkeypatch.delattr(AgentFileInputs, "read_selected")
    with pytest.raises(smoke.SmokeSetupError, match="rebuild audrey"):
        smoke.load_runtime()
    assert list(confined.iterdir()) == []


@pytest.mark.parametrize("value", ["0", "nan", "inf", "11"])
def test_invalid_deadline_does_not_load_runtime(monkeypatch, capsys, value):
    monkeypatch.setattr(smoke, "load_runtime", lambda: pytest.fail("Invalid options loaded the runtime."))
    assert smoke.main(["--json", "--timeout-seconds", value]) == 2
    assert json.loads(capsys.readouterr().out)["checks"] == []


@pytest.mark.parametrize("error,code", [
    (smoke.SmokeSetupError("Installed support is missing."), 2),
    (smoke.SmokeFailureError("A fixed contract failed."), 1),
    (RuntimeError("private filename or credential value"), 3),
    (KeyboardInterrupt(), 130),
])
def test_terminal_errors_are_sanitized_and_keep_exit_meanings(monkeypatch, capsys, error, code):
    def fail():
        raise error

    monkeypatch.setattr(smoke, "load_runtime", fail)
    assert smoke.main(["--json"]) == code
    output = capsys.readouterr().out
    assert json.loads(output)["exit_code"] == code and "private filename" not in output and "credential value" not in output
