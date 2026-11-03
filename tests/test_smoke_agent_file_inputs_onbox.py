"""The Tower file-input wrapper streams source/options literally without credentials."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "smoke" / "smoke-agent-file-inputs-onbox.sh"


def fixture(tmp_path):
    appdata = tmp_path / "appdata with spaces"
    source = appdata / "tests" / "smoke" / "smoke_agent_file_inputs.py"
    source.parent.mkdir(parents=True)
    source.write_text("# literal $source `source`\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        f"#!{shutil.which('python3')}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "with Path(os.environ['DOCKER_CALLS']).open('a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'inspect':\n"
        "    print(os.environ.get('DOCKER_RUNNING', 'true'))\n    sys.exit(0)\n"
        "Path(os.environ['DOCKER_STDIN']).write_text(sys.stdin.read())\n"
        "sys.exit(int(os.environ.get('DOCKER_EXIT', '0')))\n"
    )
    docker.chmod(0o700)
    env = os.environ.copy()
    env.update({"AUDREY_SMOKE_APPDATA": str(appdata), "AUDREY_SMOKE_CONTAINER": "audrey fixture",
                "PATH": str(binaries), "DOCKER_CALLS": str(tmp_path / "calls"), "DOCKER_STDIN": str(tmp_path / "input")})
    return source, env


def run(tmp_path, env, *args):
    return subprocess.run([shutil.which("bash"), str(SCRIPT.resolve()), *args], env=env, cwd=tmp_path,
                          capture_output=True, text=True, check=False, timeout=10)


def test_literal_stdin_options_and_isolated_no_credentials(tmp_path):
    source, env = fixture(tmp_path)
    literal = "$(touch do-not-execute) `unsafe`"
    result = run(tmp_path, env, "--json", literal)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["DOCKER_CALLS"]).read_text().splitlines()]
    assert calls[-1] == ["exec", "-i", "-e", "PYTHONDONTWRITEBYTECODE=1", "--", "audrey fixture", "/opt/venv/bin/python", "-", "--json", literal]
    assert Path(env["DOCKER_STDIN"]).read_text() == source.read_text()
    assert not (tmp_path / "do-not-execute").exists() and not (source.parents[2] / ".env.smoke.local").exists()


@pytest.mark.parametrize("code", [1, 2, 3, 130])
def test_preserves_smoke_exit_code(tmp_path, code):
    _, env = fixture(tmp_path)
    env["DOCKER_EXIT"] = str(code)
    assert run(tmp_path, env).returncode == code


def test_missing_source_never_enters_container(tmp_path):
    source, env = fixture(tmp_path)
    source.unlink()
    result = run(tmp_path, env)
    assert result.returncode == 2 and not Path(env["DOCKER_CALLS"]).exists()


def test_stopped_container_gets_no_source(tmp_path):
    _, env = fixture(tmp_path)
    env["DOCKER_RUNNING"] = "false"
    result = run(tmp_path, env)
    assert result.returncode == 2 and not Path(env["DOCKER_STDIN"]).exists()
