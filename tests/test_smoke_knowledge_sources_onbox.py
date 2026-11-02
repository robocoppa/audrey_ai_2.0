"""Tower source-smoke wrapper streams literal arguments and preserves exits."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "smoke" / "smoke-knowledge-sources-onbox.sh"


def fixture(tmp_path):
    appdata = tmp_path / "appdata with spaces"
    source = appdata / "tests" / "smoke" / "smoke_knowledge_sources.py"
    source.parent.mkdir(parents=True)
    source.write_text("# literal $source `source`\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        f"#!{shutil.which('python3')}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['DOCKER_CALLS']).open('a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'inspect':\n"
        "    print(os.environ.get('DOCKER_RUNNING', 'true'))\n"
        "    sys.exit(0)\n"
        "Path(os.environ['DOCKER_STDIN']).write_text(sys.stdin.read())\n"
        "sys.exit(int(os.environ.get('DOCKER_EXIT', '0')))\n"
    )
    docker.chmod(0o700)
    env = os.environ.copy()
    env.update({"AUDREY_SMOKE_APPDATA": str(appdata), "AUDREY_SMOKE_CONTAINER": "audrey fixture",
                "PATH": str(binaries), "DOCKER_CALLS": str(tmp_path / "calls"),
                "DOCKER_STDIN": str(tmp_path / "input")})
    return source, env


def run(tmp_path, env, *args):
    return subprocess.run([shutil.which("bash"), str(SCRIPT.resolve()), *args],
                          env=env, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=10)


def test_wrapper_streams_literal_script_and_options_without_credentials(tmp_path):
    source, env = fixture(tmp_path)
    literal = "literal $(touch no-execute) `unsafe`"
    result = run(tmp_path, env, "--json", literal)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["DOCKER_CALLS"]).read_text().splitlines()]
    assert calls[-1] == ["exec", "-i", "-e", "PYTHONDONTWRITEBYTECODE=1", "--", "audrey fixture",
                         "/opt/venv/bin/python", "-", "--json", literal]
    assert Path(env["DOCKER_STDIN"]).read_text() == source.read_text()
    assert not (tmp_path / "no-execute").exists()
    assert not (source.parents[2] / ".env.smoke.local").exists()


@pytest.mark.parametrize("exit_code", [1, 2, 3, 130])
def test_wrapper_preserves_runner_exit_code(tmp_path, exit_code):
    _, env = fixture(tmp_path)
    env["DOCKER_EXIT"] = str(exit_code)
    assert run(tmp_path, env).returncode == exit_code


def test_missing_script_stops_before_any_container_operation(tmp_path):
    source, env = fixture(tmp_path)
    source.unlink()
    result = run(tmp_path, env)
    assert result.returncode == 2 and "missing or unreadable" in result.stderr
    assert not Path(env["DOCKER_CALLS"]).exists()


def test_stopped_container_does_not_receive_the_smoke(tmp_path):
    _, env = fixture(tmp_path)
    env["DOCKER_RUNNING"] = "false"
    result = run(tmp_path, env)
    assert result.returncode == 2 and "not running" in result.stderr
    assert not Path(env["DOCKER_STDIN"]).exists()
