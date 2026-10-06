"""`probe-onbox.sh` must keep the properties that make an unattended run survive.

Structural checks cover the wrapper conventions; local fake commands also
exercise its invocation and detached run identity without Docker or network.
Every property asserted here has a specific failure behind it:

- **Self-detaching.** Long runs previously died to laptop-sleep process
  orphaning and looked like broken probes for two days. An incantation you have
  to remember (`nohup … &`) is one you eventually forget, at the cost of the run.
- **`setsid`, not just `nohup`.** nohup only ignores SIGHUP; a closed terminal
  can still take the process group with it.
- **Logs on a bind mount.** A log written inside the container is destroyed by
  the `up -d --build` that a config change requires.
- **Container clock for the stamp.** The box runs three clocks (host PDT,
  container logs MDT, `docker inspect` UTC), so a host-stamped filename would
  not line up with the log lines inside the run it names.
- **The Telegram format is eval-onbox.sh's, unchanged.** It works and is
  trusted; it is not to be "improved".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "probes" / "probe-onbox.sh"


@pytest.fixture(scope="module")
def src() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_the_script_exists_and_is_executable():
    assert _SCRIPT.is_file()
    assert _SCRIPT.stat().st_mode & 0o111, "must be chmod +x to run on the box"


class TestItSurvivesTheSessionClosing:
    def test_it_self_detaches_rather_than_documenting_nohup(self, src):
        assert "_PROBE_DETACHED" in src, "no re-exec marker — it would loop forever"
        assert "setsid nohup" in src

    def test_setsid_not_just_nohup(self, src):
        # nohup alone only ignores SIGHUP. Leaving the session entirely is the
        # property that actually survives a dropped SSH connection.
        assert "setsid" in src

    def test_there_is_a_foreground_escape_hatch(self, src):
        # Debugging a wrapper that always detaches is miserable.
        assert "FOREGROUND" in src

    def test_it_tells_you_where_the_log_is_before_detaching(self, src):
        assert "Safe to close this session" in src
        assert "tail -f" in src


class TestItRunsWhereThePythonIs:
    def test_it_execs_inside_the_container(self, src):
        # Unraid has no python3; the probe cannot run on the host at all.
        assert "docker exec" in src
        assert "python3" in src

    def test_it_copies_the_probe_in(self, src):
        # The repo is NOT bind-mounted into audrey — only config.yaml,
        # /data and /datasets — so a `git pull` on the host is invisible
        # inside the container until the copy happens.
        assert "docker cp" in src

    def test_it_copies_every_run_not_once(self, src):
        assert "not mounted" in src or "not bind-mounted" in src, (
            "the reason for copying every run must stay written down"
        )

    def test_it_defaults_to_the_audrey_container(self, src):
        assert 'CONTAINER="${CONTAINER:-audrey}"' in src


class TestLogsSurviveARebuild:
    def test_the_log_dir_is_under_appdata_not_the_container(self, src):
        # `up -d --build` is required after a config.yaml change, and would
        # take a container-local log with it.
        assert 'OUT_DIR="${OUT_DIR:-${APPDATA}/testing-out/probes}"' in src

    def test_appdata_defaults_to_the_real_box_path(self, src):
        # ⚠️ WITH the _2.0 suffix. `/mnt/user/appdata/audrey` does not exist.
        assert "/mnt/user/appdata/audrey_ai_2.0" in src


class TestTheStampComesFromTheContainerClock:
    def test_stamp_is_read_from_the_container(self, src):
        assert 'docker exec "${CONTAINER}" date' in src

    def test_it_falls_back_to_the_host_clock(self, src):
        # A stopped container must not stop you naming a log file.
        assert "|| date +" in src


class TestTelegramFormatIsUnchanged:
    """The user confirmed this format works. It is not to be redesigned."""

    def test_it_matches_eval_onbox_s_message_shape(self, src):
        assert "finished (exit ${rc})" in src
        assert "（summary unavailable）" in src  # full-width parens, house style

    def test_the_full_log_goes_as_a_document_not_inline(self, src):
        # Telegram text messages cap at 4096 chars.
        assert "sendDocument" in src
        assert "sendMessage" in src

    def test_inlining_a_log_tail_is_recorded_as_reverted(self, src):
        assert "reverted" in src, (
            "the note explaining why the format is not to be changed is gone"
        )

    def test_notify_failures_are_non_fatal(self, src):
        # The probe already ran; a failed send must not mask its result.
        assert src.count("WARN: Telegram") >= 2


class TestExitCodes:
    def test_exit_one_is_described_as_a_finding_not_a_failure(self, src):
        # router_probe exits 1 for a DISQUALIFIED candidate;
        # check_model_inventory exits 1 when config names a missing model.
        # Both are the point of running them.
        assert "FINDINGS" in src

    def test_the_probe_s_exit_code_is_propagated(self, src):
        assert 'exit "${rc}"' in src

    def test_usage_and_missing_probe_exit_two(self, src):
        assert src.count("exit 2") >= 2


class TestArgumentHandling:
    def test_key_value_pairs_become_container_env(self, src):
        assert 'ENV_FLAGS+=("-e" "${kv}")' in src

    def test_args_is_special_cased_for_flags(self, src):
        assert "ARGS=*)" in src

    def test_an_unrecognised_argument_warns_rather_than_being_swallowed(self, src):
        assert "WARN: ignoring" in src


def test_evaluation_harness_is_found_copied_and_invoked(tmp_path):
    """Exercise the real wrapper with Docker replaced, without network calls."""
    appdata = tmp_path / "checkout"
    harness = appdata / "evals" / "eval_skill_selection.py"
    fixture = appdata / "evals" / "cases" / "skill_selection_cases.json"
    fixture.parent.mkdir(parents=True)
    harness.write_text("print('evaluation')\n")
    fixture.write_text('{"schema":1,"cases":[]}')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "docker-calls.jsonl"
    docker = binaries / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_DOCKER_CALLS'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if 'date' in sys.argv:\n"
        "    print('2026-10-06-120000')\n"
        "elif 'python3' in sys.argv:\n"
        "    print('evaluation completed')\n"
    )
    docker.chmod(0o700)
    environment = dict(os.environ, APPDATA=str(appdata), FOREGROUND="1",
                       OUT_DIR=str(tmp_path / "results"),
                       WATCHDOG_ENV=str(tmp_path / "missing.env"),
                       TEST_DOCKER_CALLS=str(calls),
                       PATH=str(binaries) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        ["bash", str(_SCRIPT), "eval_skill_selection.py",
         "COPY=skill_selection_cases.json",
         "ARGS=--backend hybrid --config /app/config.yaml"],
        env=environment, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    operations = [json.loads(line) for line in calls.read_text().splitlines()]
    copies = [call for call in operations if call[0] == "cp"]
    assert copies == [
        ["cp", str(harness), "audrey:/tmp/probe-2026-10-06-120000.py"],
        ["cp", str(fixture), "audrey:/tmp/skill_selection_cases.json"],
    ]
    execution = next(call for call in operations if "python3" in call)
    assert execution == [
        "exec", "audrey", "python3", copies[0][2].split(":", 1)[1],
        "--backend", "hybrid", "--config", "/app/config.yaml",
    ]
    assert "evaluation completed" in result.stdout


@pytest.fixture
def identity_runner(tmp_path):
    """Run the actual wrapper with deterministic clocks and local-only commands."""
    appdata = tmp_path / "checkout"
    harness = appdata / "scripts" / "probes" / "identity_probe.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("print('identity probe output')\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "docker-calls.jsonl"
    notifications = tmp_path / "notifications.jsonl"
    completed = tmp_path / "detached-completed"
    clock = tmp_path / "clock-count"
    watchdog = tmp_path / "watchdog.env"
    watchdog.write_text("WATCHDOG_TOKEN=test-only-token\nWATCHDOG_CHAT_ID=test-only-chat\n")
    commands = {
        "docker": (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with open(os.environ['TEST_DOCKER_CALLS'], 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if 'date' in sys.argv:\n"
            "    clock = Path(os.environ['TEST_CLOCK_COUNT'])\n"
            "    count = int(clock.read_text()) if clock.exists() else 0\n"
            "    clock.write_text(str(count + 1))\n"
            "    print(f'2026-10-06-1200{count:02d}')\n"
            "elif 'python3' in sys.argv:\n"
            "    print('identity probe output')\n"
        ),
        "setsid": (
            "import os, subprocess, sys\n"
            "from pathlib import Path\n"
            "result = subprocess.run(sys.argv[1:], check=False)\n"
            "Path(os.environ['TEST_DETACHED_COMPLETED']).touch()\n"
            "sys.exit(result.returncode)\n"
        ),
        "nohup": "import os, sys\nos.execvp(sys.argv[1], sys.argv[1:])\n",
        "curl": (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "record = {'arguments': sys.argv[1:]}\n"
            "for arg in sys.argv[1:]:\n"
            "    if arg.startswith('document=@'):\n"
            "        record['document_content'] = Path(arg[10:]).read_text()\n"
            "with open(os.environ['TEST_NOTIFICATIONS'], 'a') as log:\n"
            "    log.write(json.dumps(record) + '\\n')\n"
        ),
    }
    for name, source in commands.items():
        executable = binaries / name
        executable.write_text(f"#!{sys.executable}\n" + source)
        executable.chmod(0o700)
    environment = dict(
        os.environ, APPDATA=str(appdata), OUT_DIR=str(tmp_path / "results"),
        WATCHDOG_ENV=str(watchdog), TEST_DOCKER_CALLS=str(calls),
        TEST_CLOCK_COUNT=str(clock), TEST_NOTIFICATIONS=str(notifications),
        TEST_DETACHED_COMPLETED=str(completed),
        STAMP="inherited-old-stamp", LOG=str(tmp_path / "inherited-old.log"),
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
    )
    environment.pop("_PROBE_DETACHED", None)
    environment.pop("FOREGROUND", None)
    return environment, calls, notifications, completed


def test_detached_clock_rollover_preserves_parent_run_identity(identity_runner):
    environment, calls, notifications, completed = identity_runner
    result = subprocess.run(
        ["bash", str(_SCRIPT), "identity_probe.py"], env=environment,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    deadline = time.monotonic() + 5
    while not completed.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert completed.exists(), "fake detached wrapper did not finish"
    announced = Path(next(
        line.removeprefix(">> log: ") for line in result.stdout.splitlines()
        if line.startswith(">> log: ")
    ))
    assert announced.name == "2026-10-06-120000-identity_probe.log"
    captured = announced.read_text()
    assert "identity probe output" in captured
    assert ">> exit    : 0" in captured
    operations = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len([call for call in operations if "date" in call]) == 1
    assert [call for call in operations if call[0] == "cp"] == [[
        "cp", str(Path(environment["APPDATA"]) / "scripts/probes/identity_probe.py"),
        "audrey:/tmp/probe-2026-10-06-120000.py",
    ]]
    sends = [json.loads(line) for line in notifications.read_text().splitlines()]
    assert len(sends) == 2
    message = next(arg for arg in sends[0]["arguments"] if arg.startswith("text="))
    assert f"→ {announced}\n" in message
    assert "finished (exit 0)" in message
    assert f"document=@{announced}" in sends[1]["arguments"]
    assert sends[1]["document_content"] == captured


def test_fresh_foreground_invocation_replaces_inherited_run_identity(identity_runner):
    environment, calls, notifications, _ = identity_runner
    environment["FOREGROUND"] = "1"
    environment["_PROBE_DETACHED"] = "1"  # A foreground call still starts afresh.
    result = subprocess.run(
        ["bash", str(_SCRIPT), "identity_probe.py"], env=environment,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    operations = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len([call for call in operations if "date" in call]) == 1
    assert "audrey:/tmp/probe-2026-10-06-120000.py" in next(
        call for call in operations if call[0] == "cp"
    )
    sends = [json.loads(line) for line in notifications.read_text().splitlines()]
    assert len(sends) == 1  # Foreground output goes to the caller, without a log file.
    message = next(arg for arg in sends[0]["arguments"] if arg.startswith("text="))
    expected_log = Path(environment["OUT_DIR"]) / "2026-10-06-120000-identity_probe.log"
    assert f"→ {expected_log}\n" in message
    assert environment["LOG"] not in message
    assert "identity probe output" in result.stdout
