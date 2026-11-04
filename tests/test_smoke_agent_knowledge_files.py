"""The installed file-backed Knowledge probe exercises canonical sources and detects regressions."""

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

_SPEC = importlib.util.spec_from_file_location("agent_knowledge_files_smoke", Path(__file__).parent / "smoke" / "smoke_agent_knowledge_files.py")
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


@pytest.fixture
def confined(tmp_path, monkeypatch):
    temporary = tempfile.TemporaryDirectory
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", lambda **kwargs: temporary(dir=tmp_path, **kwargs))
    return tmp_path


async def test_installed_knowledge_file_smoke_uses_real_canonical_storage_without_external_http_and_cleans_fixtures(confined, monkeypatch):
    import httpx

    async def forbidden(*_args, **_kwargs):
        pytest.fail("The isolated Agent Knowledge file smoke attempted external HTTP.")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    checks = await smoke.run(smoke.load_runtime(), 10)
    assert checks == [name for name, _ in smoke.CASES] + ["isolated_fixture_cleanup"]
    assert list(confined.iterdir()) == []


async def test_probe_detects_omitted_file_backed_knowledge_selection(confined, monkeypatch):
    from audrey.app_state.agent_projects import AgentProjectContexts

    original = AgentProjectContexts.capture

    def omit(contexts, **kwargs):
        original(contexts, **kwargs)
        contexts._conn.execute("UPDATE app_agent_project_contexts SET knowledge_json='[]' WHERE task_id=?", (kwargs["task_id"],))

    monkeypatch.setattr(AgentProjectContexts, "capture", omit)
    monkeypatch.setattr(smoke, "CASES", (("selection", smoke.current_file_and_mixed_source_notes),))
    with pytest.raises(smoke.SmokeFailureError, match="was not selected"):
        await smoke.run(smoke.load_runtime(), 10)
    assert list(confined.iterdir()) == []


async def test_probe_detects_file_proofs_that_ignore_source_changes(confined, monkeypatch):
    from audrey.agent_knowledge_files import AgentKnowledgeFiles

    original = AgentKnowledgeFiles.proof
    saved = {}

    def stale_proof(self, row, source, *, verify_revision, captured=None):
        key = (row["user_id"], row["project_id"], source["source_id"])
        if not verify_revision and key in saved:
            return saved[key]
        result = original(self, row, source, verify_revision=verify_revision, captured=captured)
        if result is not None:
            saved[key] = result
        return result

    monkeypatch.setattr(AgentKnowledgeFiles, "proof", stale_proof)
    monkeypatch.setattr(smoke, "CASES", (("file_fence", smoke.file_edit_delete_detach_and_reattach_fence_work),))
    with pytest.raises(smoke.SmokeFailureError, match="passed the final dispatch fence"):
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

        spec = importlib.util.spec_from_file_location("isolated_knowledge_file_smoke", sys.argv[1])
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
    assert json.loads(output) == {"exit_code": 3, "checks": [], "message": "Agent Knowledge file smoke harness crashed."}
    assert "private filename" not in output and "credential value" not in output
    assert list(confined.iterdir()) == []


def test_older_backend_is_rejected_before_creating_fixture_data(confined, monkeypatch):
    from audrey.app_state import agent_projects

    monkeypatch.setattr(agent_projects, "SUPPORTED_KNOWLEDGE_SOURCE_KINDS", frozenset({"message", "agent"}))
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
