"""Contracts for the Docker-only Tower native-smoke wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "smoke" / "smoke-native-onbox.sh"


def _run(tmp_path: Path, env_text: str) -> subprocess.CompletedProcess[str]:
    appdata = tmp_path / "appdata"
    scripts = appdata / "tests" / "smoke"
    scripts.mkdir(parents=True)
    (scripts / "smoke_native_ui.py").write_text("# fixture\n")
    env_file = appdata / ".env.smoke.local"
    env_file.write_text(env_text)
    env = os.environ.copy()
    env.update({"APPDATA": str(appdata), "ENV_FILE": str(env_file)})
    return subprocess.run(
        ["bash", str(SCRIPT), "smoke_native_ui.py"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_rejects_shell_export_syntax_before_docker(tmp_path):
    result = _run(
        tmp_path,
        "export UNSUPPORTED_API_KEY = legacy\n"
        "AUDREY_USER_JWT=user-jwt\n"
        "AUDREY_ADMIN_JWT=admin-jwt\n",
    )

    assert result.returncode == 2
    assert "starts with 'export'" in result.stderr
    assert "legacy" not in result.stderr


def test_rejects_unknown_key_without_printing_its_value(tmp_path):
    result = _run(
        tmp_path,
        "UNSUPPORTED_API_KEY=legacy-secret\n"
        "AUDREY_USER_JWT=user-jwt\n"
        "AUDREY_ADMIN_JWT=admin-jwt\n",
    )

    assert result.returncode == 2
    assert "unsupported key UNSUPPORTED_API_KEY" in result.stderr
    assert "legacy-secret" not in result.stderr


def test_requires_both_native_accounts_for_ui_smoke(tmp_path):
    result = _run(
        tmp_path,
        "AUDREY_USER_JWT=user-jwt\n",
    )

    assert result.returncode == 2
    assert "missing AUDREY_ADMIN_JWT" in result.stderr
    assert "user-jwt" not in result.stderr

def test_wrapper_pins_the_tower_runtime_contract():
    text = SCRIPT.read_text()
    assert '--network "${NETWORK}"' in text
    assert '--env-file "${ENV_FILE}"' in text
    assert "AUDREY_SMOKE_BASE_URL=${BASE_URL}" in text
    assert "${APPDATA}/tests/smoke:/smoke:ro" in text
    assert "/opt/venv/bin/python" in text

@pytest.mark.parametrize("script,admin_required", [
    ("smoke_native_ui.py", True),
    ("smoke_native_project_watches.py", False),
])
def test_valid_native_env_invokes_the_expected_container_contract(tmp_path, script, admin_required):
    appdata = tmp_path / "appdata"
    scripts = appdata / "tests" / "smoke"
    scripts.mkdir(parents=True)
    (scripts / script).write_text("# fixture\n")
    env_file = appdata / ".env.smoke.local"
    env_file.write_text(
        "AUDREY_USER_JWT=user-jwt\n"
        + ("AUDREY_ADMIN_JWT=admin-jwt\n" if admin_required else "")
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_args = tmp_path / "docker-args"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"run\" ]]; then\n"
        "  printf '%s\\n' \"$@\" > \"${DOCKER_ARGS}\"\n"
        "fi\n"
    )
    fake_docker.chmod(0o700)

    env = os.environ.copy()
    env.update(
        {
            "APPDATA": str(appdata),
            "ENV_FILE": str(env_file),
            "IMAGE": "test-image",
            "NETWORK": "test-network",
            "AUDREY_DIRECT_SMOKE_MODEL_ID": "direct/example-model:latest",
            "DOCKER_ARGS": str(docker_args),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    args = docker_args.read_text().splitlines()
    assert args[:4] == ["run", "--rm", "--network", "test-network"]
    assert ["--env-file", str(env_file)] == args[4:6]
    assert "AUDREY_DIRECT_SMOKE_MODEL_ID=direct/example-model:latest" in args
    assert "AUDREY_SMOKE_BASE_URL=http://audrey-ui:8080" in args
    assert f"{appdata}/tests/smoke:/smoke:ro" in args
    assert args[-3:] == [
        "test-image",
        "/opt/venv/bin/python",
        f"/smoke/{script}",
    ]
