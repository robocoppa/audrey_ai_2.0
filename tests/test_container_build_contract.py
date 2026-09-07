"""Container reproducibility and non-root ownership contracts."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
AUDREY_DOCKERFILE = ROOT / "docker" / "audrey.Dockerfile"
TOOLS_DOCKERFILE = ROOT / "docker" / "custom-tools.Dockerfile"
FETCHER_DOCKERFILE = ROOT / "docker" / "media-fetcher.Dockerfile"
COMPOSE = ROOT / "compose.yaml"
CONFIG = ROOT / "config.yaml"


def _text(path: Path) -> str:
    return path.read_text()


def _arg_default(text: str, name: str) -> str:
    match = re.search(rf"^ARG {name}=(\d+)$", text, re.MULTILINE)
    assert match, f"{name} must have an explicit numeric default"
    return match.group(1)


def test_python_services_install_from_the_workspace_lock():
    expected_packages = {
        AUDREY_DOCKERFILE: "audrey",
        TOOLS_DOCKERFILE: "audrey-custom-tools",
    }
    for path, package in expected_packages.items():
        text = _text(path)
        assert "COPY pyproject.toml uv.lock /app/" in text
        assert "COPY tools-server/pyproject.toml /app/tools-server/pyproject.toml" in text
        assert f"uv sync --locked --no-dev --package {package}" in text
        assert "--no-install-workspace --no-cache" in text
        assert "uv pip compile" not in text


def test_native_ui_build_includes_repository_portraits():
    text = _text(AUDREY_DOCKERFILE)
    assert "COPY images /workspace/images" in text
    assert text.index("COPY images /workspace/images") < text.index(
        "RUN npm run build --prefix /workspace/web"
    )


def test_every_shared_writer_uses_unraids_numeric_identity():
    texts = [
        _text(path)
        for path in (AUDREY_DOCKERFILE, TOOLS_DOCKERFILE, FETCHER_DOCKERFILE)
    ]
    assert {_arg_default(text, "APP_UID") for text in texts} == {"99"}
    assert {_arg_default(text, "APP_GID") for text in texts} == {"100"}
    assert "USER audrey" in texts[0]
    assert "USER tools" in texts[1]
    assert "USER fetcher" in texts[2]


def test_bind_mounts_match_the_non_root_cache_and_read_only_dataset_contract():
    compose = yaml.safe_load(COMPOSE.read_text())
    volumes = compose["services"]["audrey-ai"]["volumes"]
    assert "/mnt/user/appdata/clip-cache:/home/audrey/.cache/clip" in volumes
    assert "/mnt/user/knowledge:/datasets:ro" in volumes

    config = yaml.safe_load(CONFIG.read_text())
    assert config["kb"]["image_cache_folder"] == "/home/audrey/.cache/clip"


def test_root_only_clip_cache_path_is_gone_from_runtime_files():
    paths = (
        AUDREY_DOCKERFILE,
        COMPOSE,
        CONFIG,
        ROOT / "src/audrey/main.py",
        ROOT / "src/audrey/kb/cli.py",
    )
    offenders = [
        str(path.relative_to(ROOT))
        for path in paths
        if "/root/.cache/clip" in _text(path)
    ]
    assert not offenders, f"root-only CLIP cache path remains in: {offenders}"
