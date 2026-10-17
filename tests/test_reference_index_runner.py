"""Run the isolated reference index wrapper with fake Docker and Telegram."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts/reference/index-irs-onbox.sh"


@pytest.fixture
def runner(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    knowledge = tmp_path / "public knowledge"
    archive = knowledge / "reference-libraries/tax/us/federal/irs"
    (archive / "catalogs").mkdir(parents=True)
    (archive / "catalogs/archive.sqlite").write_bytes(b"archive fixture")
    index = tmp_path / "public index state"
    index.mkdir()
    config = tmp_path / "server configuration.yaml"
    config.write_text("reference_libraries: {}\n")
    watchdog = tmp_path / "watchdog.env"
    watchdog.write_text(
        "WATCHDOG_TOKEN=synthetic-notification-marker\nWATCHDOG_CHAT_ID=synthetic-chat\n"
    )
    calls = tmp_path / "docker.jsonl"
    notifications = tmp_path / "telegram.jsonl"
    commands = {
        "docker": (
            "import json, os, signal, sys, time\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['TEST_DOCKER_CALLS'], 'a') as log:\n"
            "    log.write(json.dumps(args) + '\\n')\n"
            "if args[:2] == ['image', 'inspect']:\n"
            "    sys.exit(int(os.environ.get('TEST_IMAGE_EXIT', '0')))\n"
            "if args[:2] == ['network', 'inspect']:\n"
            "    sys.exit(int(os.environ.get('TEST_NETWORK_EXIT', '0')))\n"
            "if args[:2] == ['container', 'inspect']:\n"
            "    sys.exit(int(os.environ.get('TEST_CONTAINER_EXIT', '1')))\n"
            "if args[0] == 'stop':\n"
            "    started = os.environ.get('TEST_RUN_STARTED')\n"
            "    if started and Path(started).exists():\n"
            "        os.kill(int(Path(started).read_text()), signal.SIGTERM)\n"
            "    sys.exit(0)\n"
            "if args[0] == 'run' and '-c' in args:\n"
            "    sys.exit(int(os.environ.get('TEST_PREFLIGHT_EXIT', '0')))\n"
            "if args[0] == 'run' and '-m' in args:\n"
            "    started = os.environ.get('TEST_RUN_STARTED')\n"
            "    if started:\n"
            "        Path(started).write_text(str(os.getpid()))\n"
            "    if os.environ.get('TEST_RUN_DELAY'):\n"
            "        time.sleep(float(os.environ['TEST_RUN_DELAY']))\n"
            "    target = args[args.index('--report') + 1]\n"
            "    report = Path(os.environ['INDEX_HOST']) / 'reports' / Path(target).name\n"
            "    report.parent.mkdir(exist_ok=True)\n"
            "    if os.environ.get('TEST_WRITE_REPORT', '1') == '1':\n"
            "        report.write_text(json.dumps({'schema': 1, 'status': 'published',"
            " 'public_documents': 2, 'release_id': 'release-fixture'}))\n"
            "    print('public reference indexing summary', flush=True)\n"
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
            "if os.environ.get('TEST_CURL_FAIL'):\n"
            "    print('failed: ' + next(a for a in args if 'api.telegram.org' in a), file=sys.stderr)\n"
            "    sys.exit(22)\n"
            "print(json.dumps({'ok': True}))\n"
        ),
    }
    for name, source in commands.items():
        executable = binaries / name
        executable.write_text(f"#!{sys.executable}\n" + source)
        executable.chmod(0o700)
    environment = dict(
        os.environ,
        KNOWLEDGE_ROOT=str(knowledge),
        ARCHIVE_ROOT=str(archive),
        CONFIG_FILE=str(config),
        INDEX_HOST=str(index),
        OUT_DIR=str(tmp_path / "run reports"),
        IMAGE="test-image",
        NETWORK="test-ollama-net",
        INDEX_CONTAINER="test-reference-index",
        NOTIFY="1",
        WATCHDOG_ENV=str(watchdog),
        TEST_DOCKER_CALLS=str(calls),
        TEST_NOTIFICATIONS=str(notifications),
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
    )
    for setting in ("OLLAMA_HOST", "QDRANT_HOST", "QDRANT_PORT"):
        environment.pop(setting, None)

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


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _message(entry):
    return next(arg.removeprefix("text=") for arg in entry["arguments"] if arg.startswith("text="))


def test_wrapper_has_valid_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_only_public_sources_and_index_state_are_mounted_and_arguments_are_preserved(runner):
    invoke, environment, calls, notifications = runner
    result = invoke("--max-documents", "2", "--release-id", "resume identifier", NOTIFY="0")

    assert result.returncode == 0, result.stderr
    runs = [call for call in _read(calls) if call[0] == "run"]
    assert len(runs) == 2
    preflight, indexing = runs
    assert preflight[preflight.index("--network") + 1] == "none"
    preflight_mounts = [preflight[i + 1] for i, arg in enumerate(preflight) if arg == "--mount"]
    assert preflight_mounts == [
        f"type=bind,src={environment['INDEX_HOST']},dst=/data/reference-libraries",
    ]
    assert indexing[indexing.index("--network") + 1] == "test-ollama-net"
    assert "--user" not in indexing
    assert "--env-file" not in indexing
    assert "--gpus" not in indexing
    assert "--privileged" not in indexing
    assert "--read-only" in indexing and "--no-healthcheck" in indexing
    assert indexing[indexing.index("--memory") + 1] == "2g"
    assert indexing[indexing.index("--cap-drop") + 1] == "ALL"
    mounts = [indexing[i + 1] for i, arg in enumerate(indexing) if arg == "--mount"]
    assert mounts == [
        f"type=bind,src={environment['KNOWLEDGE_ROOT']},dst=/datasets,readonly",
        f"type=bind,src={environment['CONFIG_FILE']},dst=/app/config.yaml,readonly",
        f"type=bind,src={environment['INDEX_HOST']},dst=/data/reference-libraries",
    ]
    assert environment["WATCHDOG_ENV"] not in indexing
    assert indexing[indexing.index("--entrypoint") + 1] == "/opt/venv/bin/python"
    assert indexing[indexing.index("-m") + 1] == "audrey.references.index_cli"
    assert indexing[-4:] == ["--max-documents", "2", "--release-id", "resume identifier"]
    assert "public reference indexing summary" in result.stdout
    assert not notifications.exists()


def test_local_numerical_pools_cannot_exhaust_the_index_thread_budget(runner):
    invoke, _, calls, _ = runner
    result = invoke(OPENBLAS_NUM_THREADS="128", OMP_NUM_THREADS="128", NOTIFY="0")

    assert result.returncode == 0, result.stderr
    runs = [call for call in _read(calls) if call[0] == "run"]
    assert len(runs) == 2
    for call in runs:
        settings = [call[i + 1] for i, argument in enumerate(call) if argument == "-e"]
        assert "OPENBLAS_NUM_THREADS=1" in settings
        assert "OMP_NUM_THREADS=1" in settings
        assert "OPENBLAS_NUM_THREADS=128" not in settings
        assert "OMP_NUM_THREADS=128" not in settings
    assert runs[1][runs[1].index("--pids-limit") + 1] == "64"


@pytest.mark.parametrize("code", [0, 1, 2, 3, 130, 143])
def test_report_is_copied_from_owned_index_state_and_sent_on_every_exit(runner, code):
    invoke, environment, _, notifications = runner
    result = invoke(TEST_RUN_EXIT=str(code))

    assert result.returncode == code
    copied = next(Path(environment["OUT_DIR"]).glob("*-reference-index-report.json"))
    original = Path(environment["INDEX_HOST"]) / "reports" / copied.name
    assert copied.read_bytes() == original.read_bytes()
    assert json.loads(copied.read_text())["release_id"] == "release-fixture"
    sends = _read(notifications)
    assert len(sends) == 3
    assert "reference index started" in _message(sends[0])
    assert f"finished (exit {code})" in _message(sends[1])
    assert str(copied) in _message(sends[1])
    assert sends[2]["document_content"] == copied.read_text()


@pytest.mark.parametrize(
    "overrides",
    [
        {"TEST_IMAGE_EXIT": "1"},
        {"TEST_NETWORK_EXIT": "1"},
        {"TEST_CONTAINER_EXIT": "0"},
        {"TEST_PREFLIGHT_EXIT": "1"},
    ],
)
def test_setup_failures_notify_without_launching_indexing(runner, overrides):
    invoke, _, calls, notifications = runner
    result = invoke(**overrides)

    assert result.returncode == 2
    assert not any("-m" in call for call in _read(calls))
    sends = _read(notifications)
    assert len(sends) == 2
    assert "SETUP ERROR" in _message(sends[0])
    assert json.loads(sends[1]["document_content"])["exit_code"] == 2


@pytest.mark.parametrize("missing", ["CONFIG_FILE", "ARCHIVE_ROOT", "INDEX_HOST"])
def test_sources_and_backend_created_state_are_required_before_docker(runner, missing):
    invoke, environment, calls, notifications = runner
    result = invoke(**{missing: environment[missing] + "-missing"})

    assert result.returncode == 2
    assert not calls.exists()
    assert "SETUP ERROR" in _message(_read(notifications)[0])


def test_success_without_index_report_becomes_crash_and_public_failure_report(runner):
    invoke, environment, _, notifications = runner
    result = invoke(TEST_WRITE_REPORT="0")

    assert result.returncode == 3
    assert "Indexer did not write its report" in result.stderr
    report = next(Path(environment["OUT_DIR"]).glob("*-reference-index-report.json"))
    assert json.loads(report.read_text()) == {
        "schema": 1,
        "status": "runner_failure",
        "exit_code": 3,
    }
    assert "CRASHED" in _message(_read(notifications)[1])


@pytest.mark.parametrize("code", [0, 1, 3])
def test_notification_failure_cannot_change_run_status_or_reveal_credentials(runner, code):
    invoke, _, _, _ = runner
    result = invoke(TEST_CURL_FAIL="1", TEST_RUN_EXIT=str(code))

    assert result.returncode == code
    assert "WARN: Telegram" in result.stderr
    assert "synthetic-notification-marker" not in result.stdout + result.stderr
    assert "api.telegram.org/bot" not in result.stdout + result.stderr


@pytest.mark.parametrize(("signum", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_cancellation_stops_only_the_named_index_container_and_sends_report(
    runner,
    tmp_path,
    signum,
    code,
):
    _, environment, calls, notifications = runner
    started = tmp_path / "index-started"
    process = subprocess.Popen(
        ["bash", str(_SCRIPT)],
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
        assert started.exists()
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=3)
        assert process.returncode == code, stderr
        assert ["stop", "--time", "5", environment["INDEX_CONTAINER"]] in _read(calls)
        sends = _read(notifications)
        assert len(sends) == 3
        assert "INTERRUPTED" in _message(sends[1])
        assert f"finished (exit {code})" in _message(sends[1])
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
