"""Host-side contracts for the installed-image admission smoke wrapper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "smoke" / "smoke-agent-admission-onbox.sh"
SMOKE_BODY = "# standalone smoke fixture\nprint('literal $payload; `payload`')\n"


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    appdata = tmp_path / "appdata with spaces"
    smoke_dir = appdata / "tests" / "smoke"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "smoke_agent_admission.py").write_text(SMOKE_BODY)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        f"#!{shutil.which('python3')}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with Path(os.environ['DOCKER_CALLS']).open('a') as log:\n"
        "    log.write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'inspect':\n"
        "    print(os.environ.get('DOCKER_RUNNING', 'true'))\n"
        "    sys.exit(int(os.environ.get('DOCKER_INSPECT_EXIT', '0')))\n"
        "if args[0] == 'exec':\n"
        "    Path(os.environ['DOCKER_STDIN']).write_text(sys.stdin.read())\n"
        "    print('PASS: installed admission checks')\n"
        "    sys.exit(int(os.environ.get('DOCKER_EXEC_EXIT', '0')))\n"
        "sys.exit(99)\n"
    )
    fake_docker.chmod(0o700)
    env = os.environ.copy()
    for key in ("AUDREY_SMOKE_APPDATA", "APPDATA", "AUDREY_SMOKE_CONTAINER"):
        env.pop(key, None)
    env.update(
        {
            "APPDATA": str(appdata),
            "PATH": str(bin_dir),
            "DOCKER_CALLS": str(tmp_path / "docker-calls.jsonl"),
            "DOCKER_STDIN": str(tmp_path / "docker-stdin.txt"),
        }
    )
    return appdata, bin_dir, env


def _run(tmp_path: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("bash"), str(SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _calls(env: dict[str, str]) -> list[list[str]]:
    path = Path(env["DOCKER_CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_streams_host_smoke_into_running_image_with_literal_arguments(tmp_path):
    appdata, _bin_dir, env = _fixture(tmp_path)
    env["AUDREY_SMOKE_CONTAINER"] = "audrey fixture"
    literal_arg = "value with spaces; $(touch should-not-exist) `echo unsafe`"

    result = _run(tmp_path, env, "--json", "--timeout-seconds", "15", literal_arg)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "PASS: installed admission checks\n"
    assert _calls(env) == [
        [
            "inspect",
            "--type",
            "container",
            "--format",
            "{{.State.Running}}",
            "--",
            "audrey fixture",
        ],
        [
            "exec",
            "-i",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "--",
            "audrey fixture",
            "/opt/venv/bin/python",
            "-",
            "--config",
            "/app/config.yaml",
            "--json",
            "--timeout-seconds",
            "15",
            literal_arg,
        ],
    ]
    assert Path(env["DOCKER_STDIN"]).read_text() == SMOKE_BODY
    assert sorted(
        path.relative_to(appdata).as_posix() for path in appdata.rglob("*") if path.is_file()
    ) == ["tests/smoke/smoke_agent_admission.py"]
    assert not (tmp_path / "should-not-exist").exists()


def test_explicit_appdata_takes_precedence_and_needs_no_auth_file(tmp_path):
    appdata, _bin_dir, env = _fixture(tmp_path)
    env.update({"AUDREY_SMOKE_APPDATA": str(appdata), "APPDATA": str(tmp_path / "wrong")})

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stderr
    assert _calls(env)[1][5] == "audrey"
    assert Path(env["DOCKER_STDIN"]).read_text() == SMOKE_BODY
    assert not (appdata / ".env.smoke.local").exists()


def test_missing_host_smoke_fails_before_docker(tmp_path):
    appdata, _bin_dir, env = _fixture(tmp_path)
    (appdata / "tests" / "smoke" / "smoke_agent_admission.py").unlink()

    result = _run(tmp_path, env)

    assert result.returncode == 2
    assert "smoke script missing or unreadable" in result.stderr
    assert _calls(env) == []


def test_missing_docker_is_a_setup_error(tmp_path):
    _appdata, bin_dir, env = _fixture(tmp_path)
    (bin_dir / "docker").unlink()

    result = _run(tmp_path, env)

    assert result.returncode == 2
    assert "docker not found" in result.stderr
    assert _calls(env) == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"DOCKER_RUNNING": "false"}, "is not running"),
        ({"DOCKER_RUNNING": "true", "DOCKER_INSPECT_EXIT": "1"}, "cannot inspect container"),
    ],
)
def test_unavailable_container_fails_without_exec(tmp_path, overrides, message):
    _appdata, _bin_dir, env = _fixture(tmp_path)
    env.update(overrides)

    result = _run(tmp_path, env)

    assert result.returncode == 2
    assert message in result.stderr
    assert len(_calls(env)) == 1
    assert not Path(env["DOCKER_STDIN"]).exists()


@pytest.mark.parametrize("exit_code", [1, 2, 3, 130])
def test_preserves_smoke_exit_status(tmp_path, exit_code):
    _appdata, _bin_dir, env = _fixture(tmp_path)
    env["DOCKER_EXEC_EXIT"] = str(exit_code)

    result = _run(tmp_path, env)

    assert result.returncode == exit_code
    assert len(_calls(env)) == 2
