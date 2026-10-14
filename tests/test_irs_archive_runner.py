"""Run the real IRS collector wrapper with local fake Docker and Telegram commands."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "reference" / "collect-irs-onbox.sh"
_FAKE_TOKEN = "only-a-test-token-never-print"  # noqa: S105 - synthetic fixture
_FAKE_CHAT = "only-a-test-chat"


@pytest.fixture
def runner(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "docker.jsonl"
    notifications = tmp_path / "telegram.jsonl"
    watchdog = tmp_path / "watchdog settings.env"
    watchdog.write_text(
        f"WATCHDOG_TOKEN={_FAKE_TOKEN}\nWATCHDOG_CHAT_ID={_FAKE_CHAT}\n"
    )
    commands = {
        "docker": (
            "import json, os, signal, sys, time\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['TEST_DOCKER_CALLS'], 'a') as log:\n"
            "    log.write(json.dumps(args) + '\\n')\n"
            "if args[:2] == ['image', 'inspect']:\n"
            "    sys.exit(int(os.environ.get('TEST_IMAGE_EXIT', '0')))\n"
            "if args[:2] == ['container', 'inspect']:\n"
            "    sys.exit(int(os.environ.get('TEST_CONTAINER_EXIT', '1')))\n"
            "if args[0] == 'stop':\n"
            "    marker = os.environ.get('TEST_RUN_STARTED')\n"
            "    if marker and Path(marker).exists():\n"
            "        os.kill(int(Path(marker).read_text()), signal.SIGTERM)\n"
            "    sys.exit(0)\n"
            "if args[0] == 'run':\n"
            "    marker = os.environ.get('TEST_RUN_STARTED')\n"
            "    if marker:\n"
            "        Path(marker).write_text(str(os.getpid()))\n"
            "    if os.environ.get('TEST_RUN_DELAY'):\n"
            "        time.sleep(float(os.environ['TEST_RUN_DELAY']))\n"
            "    mounts = [args[i+1] for i, value in enumerate(args) if value == '--mount']\n"
            "    reports = next(m for m in mounts if m.endswith(',dst=/reports'))\n"
            "    root = Path(reports.removeprefix('type=bind,src=').removesuffix(',dst=/reports'))\n"
            "    target = args[args.index('--report') + 1]\n"
            "    report = root / Path(target).name\n"
            "    if os.environ.get('TEST_WRITE_REPORT', '1') == '1':\n"
            "        report.write_text(json.dumps({'schema': 1, 'status': 'partial',"
            " 'downloads': 5, 'public_fixture': True}))\n"
            "    elif os.environ.get('TEST_WRITE_REPORT') == 'empty':\n"
            "        report.write_text('')\n"
            "    print('public IRS collection summary', flush=True)\n"
            "    print('public collector diagnostic', file=sys.stderr)\n"
            "    sys.exit(int(os.environ.get('TEST_RUN_EXIT', '0')))\n"
            "sys.exit(0)\n"
        ),
        "curl": (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "entry = {'arguments': args}\n"
            "for arg in args:\n"
            "    if arg.startswith('document=@'):\n"
            "        entry['document_content'] = Path(arg.removeprefix('document=@')).read_text()\n"
            "with open(os.environ['TEST_NOTIFICATIONS'], 'a') as log:\n"
            "    log.write(json.dumps(entry) + '\\n')\n"
            "mode = os.environ.get('TEST_CURL_MODE', 'ok')\n"
            "if mode == 'failed':\n"
            "    print('failed request: ' + next(a for a in args if 'api.telegram.org' in a),"
            " file=sys.stderr)\n"
            "    sys.exit(22)\n"
            "print(json.dumps({'ok': mode == 'ok'}))\n"
        ),
    }
    for name, source in commands.items():
        executable = binaries / name
        executable.write_text(f"#!{sys.executable}\n" + source)
        executable.chmod(0o700)
    environment = dict(
        os.environ,
        ARCHIVE_ROOT=str(tmp_path / "public reference archive"),
        OUT_DIR=str(tmp_path / "collection reports"),
        WATCHDOG_ENV=str(watchdog),
        IMAGE="test-audrey-image",
        COLLECTOR_CONTAINER="test-irs-collector",
        NOTIFY="1",
        TEST_DOCKER_CALLS=str(calls),
        TEST_NOTIFICATIONS=str(notifications),
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
    )

    def invoke(*arguments, **overrides):
        return subprocess.run(
            ["bash", str(_SCRIPT), *arguments],
            env=environment | overrides,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    return invoke, environment, calls, notifications


def _read_calls(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _message(notification):
    return next(
        arg.removeprefix("text=")
        for arg in notification["arguments"]
        if arg.startswith("text=")
    )


def test_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_container_mounts_only_public_archive_and_reports_and_preserves_arguments(runner):
    invoke, environment, calls, _ = runner
    result = invoke("--catalog", "current", "--max-downloads", "5",
                    "--user-agent", "Owner's reference collector contact@example.com",
                    NOTIFY="0")

    assert result.returncode == 0, result.stderr
    execution = next(call for call in _read_calls(calls) if call[0] == "run")
    mounts = [execution[i + 1] for i, value in enumerate(execution) if value == "--mount"]
    assert mounts == [
        f"type=bind,src={environment['ARCHIVE_ROOT']},dst=/archive",
        f"type=bind,src={environment['OUT_DIR']},dst=/reports",
    ]
    assert execution[execution.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert execution[execution.index("--entrypoint") + 1] == "/opt/venv/bin/python"
    assert "--read-only" in execution
    assert "--no-healthcheck" in execution
    assert execution[execution.index("--cap-drop") + 1] == "ALL"
    assert execution[execution.index("--security-opt") + 1] == "no-new-privileges"
    assert "--env-file" not in execution
    assert "--gpus" not in execution
    assert "--network" not in execution
    assert environment["WATCHDOG_ENV"] not in execution
    assert "/data" not in execution
    module = execution.index("-m")
    assert execution[module:module + 2] == ["-m", "audrey.references.irs_collect"]
    assert execution[-6:] == [
        "--catalog", "current", "--max-downloads", "5",
        "--user-agent", "Owner's reference collector contact@example.com",
    ]
    assert "public IRS collection summary" in result.stdout
    assert "public collector diagnostic" in result.stdout
    logs = list(Path(environment["OUT_DIR"]).glob("*-irs-archive.log"))
    assert len(logs) == 1
    assert logs[0].read_text() == (
        "public IRS collection summary\npublic collector diagnostic\n"
    )


@pytest.mark.parametrize(
    ("code", "verdict"),
    [
        (0, "collection run finished; see report for remaining coverage"),
        (1, "collection finished with findings"),
        (2, "SETUP ERROR"),
        (3, "CRASHED"),
        (130, "INTERRUPTED"),
        (143, "INTERRUPTED"),
        (125, "CRASHED"),
    ],
)
def test_start_and_completion_notifications_preserve_every_collector_exit(runner, code, verdict):
    invoke, environment, _, notifications = runner
    result = invoke("--max-downloads", "5", TEST_RUN_EXIT=str(code))

    assert result.returncode == code
    sends = _read_calls(notifications)
    assert len(sends) == 3
    assert "started. Public documents only; no model calls." in _message(sends[0])
    completion = _message(sends[1])
    assert verdict in completion
    assert f"finished (exit {code})" in completion
    report = next(Path(environment["OUT_DIR"]).glob("*-irs-archive-report.json"))
    assert str(report) in _message(sends[0])
    assert str(report) in completion
    assert "sendDocument" in " ".join(sends[2]["arguments"])
    assert f"document=@{report}" in sends[2]["arguments"]
    assert json.loads(sends[2]["document_content"]) == json.loads(report.read_text())
    assert json.loads(report.read_text())["status"] == "partial"
    assert _FAKE_TOKEN not in result.stdout + result.stderr
    assert _FAKE_CHAT not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"TEST_IMAGE_EXIT": "1"}, "Build the Audrey image first"),
        ({"TEST_CONTAINER_EXIT": "0"}, "collector container already exists"),
    ],
)
def test_setup_failure_notifies_and_attaches_public_failure_report(runner, overrides, message):
    invoke, environment, calls, notifications = runner
    result = invoke(**overrides)

    assert result.returncode == 2
    assert message in result.stderr
    assert not any(call[0] == "run" for call in _read_calls(calls))
    sends = _read_calls(notifications)
    assert len(sends) == 2
    assert "SETUP ERROR" in _message(sends[0])
    assert "finished (exit 2)" in _message(sends[0])
    report = next(Path(environment["OUT_DIR"]).glob("*-irs-archive-report.json"))
    assert json.loads(report.read_text()) == {
        "schema": 1, "status": "runner_failure", "exit_code": 2,
    }
    assert sends[1]["document_content"] == report.read_text()


@pytest.mark.parametrize("code", [0, 1, 3, 130, 143])
def test_missing_collector_report_cannot_claim_success(runner, code):
    invoke, environment, _, notifications = runner
    result = invoke(TEST_RUN_EXIT=str(code), TEST_WRITE_REPORT="0")

    expected_code = 3 if code == 0 else code
    assert result.returncode == expected_code
    report = next(Path(environment["OUT_DIR"]).glob("*-irs-archive-report.json"))
    assert json.loads(report.read_text()) == {
        "schema": 1, "status": "runner_failure", "exit_code": expected_code,
    }
    assert len(_read_calls(notifications)) == 3


@pytest.mark.parametrize("mode", ["failed", "rejected"])
@pytest.mark.parametrize("code", [0, 1, 3])
def test_telegram_failure_cannot_change_exit_or_print_credentials(runner, mode, code):
    invoke, _, _, notifications = runner
    result = invoke(TEST_RUN_EXIT=str(code), TEST_CURL_MODE=mode)

    assert result.returncode == code
    assert len(_read_calls(notifications)) == 3
    assert "WARN: Telegram" in result.stderr
    assert _FAKE_TOKEN not in result.stdout + result.stderr
    assert _FAKE_CHAT not in result.stdout + result.stderr
    assert "api.telegram.org/bot" not in result.stdout + result.stderr


def test_bounded_no_notify_run_returns_in_shell_without_telegram(runner):
    invoke, _, calls, notifications = runner
    result = invoke("--catalog", "current", "--max-downloads", "5", NOTIFY="0")

    assert result.returncode == 0
    assert "public IRS collection summary" in result.stdout
    assert any(call[0] == "run" for call in _read_calls(calls))
    assert not notifications.exists()


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_unavailable_notification_settings_do_not_stop_collection(runner, kind):
    invoke, environment, calls, notifications = runner
    watchdog = Path(environment["WATCHDOG_ENV"])
    if kind == "missing":
        watchdog.unlink()
    else:
        watchdog.write_text("")

    result = invoke()
    assert result.returncode == 0
    assert any(call[0] == "run" for call in _read_calls(calls))
    assert not notifications.exists()
    assert "WARN: notification" in result.stderr


@pytest.mark.parametrize("root_name", ["ARCHIVE_ROOT", "OUT_DIR"])
def test_relative_root_is_rejected_before_any_container_run(runner, root_name):
    invoke, _, calls, _ = runner
    result = invoke(**{root_name: "relative/path"})

    assert result.returncode == 2
    assert "must be absolute paths" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize("root_name", ["ARCHIVE_ROOT", "OUT_DIR"])
def test_symlink_root_is_rejected_before_any_container_run(runner, root_name, tmp_path):
    invoke, _, calls, _ = runner
    destination = tmp_path / "real folder"
    destination.mkdir()
    link = tmp_path / "linked folder"
    link.symlink_to(destination, target_is_directory=True)
    result = invoke(**{root_name: str(link)})

    assert result.returncode == 2
    assert "cannot be symlinks" in result.stderr
    assert not calls.exists()

@pytest.mark.parametrize(("signum", "exit_code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_signal_interrupts_active_collection_and_notifies(runner, tmp_path, signum, exit_code):
    _, environment, calls, notifications = runner
    started = tmp_path / "collector-started"
    process = subprocess.Popen(
        ["bash", str(_SCRIPT), "--max-downloads", "5"],
        env=environment | {"TEST_RUN_STARTED": str(started), "TEST_RUN_DELAY": "30"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists(), "fake collector did not start"
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == exit_code, stderr
        assert ["stop", "--time", "5", environment["COLLECTOR_CONTAINER"]] in _read_calls(calls)
        sends = _read_calls(notifications)
        assert len(sends) == 3
        assert "INTERRUPTED" in _message(sends[1])
        assert f"finished (exit {exit_code})" in _message(sends[1])
        assert _FAKE_TOKEN not in stdout + stderr
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()

def test_empty_collector_report_cannot_claim_success(runner):
    invoke, _, _, notifications = runner
    result = invoke(TEST_WRITE_REPORT="empty")

    assert result.returncode == 3
    assert "Collector did not write its report" in result.stderr
    assert "CRASHED" in _message(_read_calls(notifications)[1])
