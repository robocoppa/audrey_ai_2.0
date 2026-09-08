"""Container reproducibility and non-root ownership contracts."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
AUDREY_DOCKERFILE = ROOT / "docker" / "audrey.Dockerfile"
UI_DOCKERFILE = ROOT / "web" / "Dockerfile"
UI_NGINX_TEMPLATE = ROOT / "web" / "docker" / "default.conf.template"
UI_VITE_CONFIG = ROOT / "web" / "vite.config.ts"
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


def test_native_ui_build_is_self_contained_and_has_a_transitional_fallback():
    audrey = _text(AUDREY_DOCKERFILE)
    ui = _text(UI_DOCKERFILE)
    vite = _text(UI_VITE_CONFIG)
    compose = yaml.safe_load(COMPOSE.read_text())

    assert "COPY images /workspace/images" not in audrey
    assert "COPY --from=web-build /workspace/web/dist" in audrey
    assert "COPY --from=build --chown=101:101 /workspace/dist" in ui
    assert (
        "nginxinc/nginx-unprivileged:1.30.4-alpine3.24@"
        "sha256:442753882674b49ae2c1de83ed67896131c0777f56df5005e356e62bc3f7e7ce"
        in ui
    )
    assert "USER 101" in ui
    assert 'outDir: "dist"' in vite
    assert "../src/audrey/static/app" not in vite
    for asset in (
        "audrey2.png",
        "audrey3.png",
        "audrey7.png",
        "audrey8.png",
        "cloudModel.png",
        "localModel.png",
        "search.png",
    ):
        assert (ROOT / "web" / "src" / "assets" / "models" / asset).is_file()

    service = compose["services"]["audrey-ui"]
    backend = compose["services"]["audrey"]
    assert "audrey-ai" not in compose["services"]
    assert backend["container_name"] == "audrey"
    assert backend["image"] == "audrey:latest"
    assert set(backend["networks"]) == {"ollama-net", "media-net", "fetch-net"}
    for network in backend["networks"].values():
        assert network["aliases"] == ["audrey-ai"]
    assert service["build"] == {"context": "./web", "dockerfile": "Dockerfile"}
    assert service["depends_on"]["audrey"]["condition"] == "service_healthy"
    assert service["networks"] == ["ollama-net"]
    assert service["ports"] == ["127.0.0.1:${AUDREY_UI_PORT:-8090}:8080"]
    assert service["environment"]["AUDREY_UPSTREAM"] == (
        "${AUDREY_UI_UPSTREAM:-http://audrey:8000}"
    )
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert compose["services"]["custom-tools"]["environment"]["AUDREY_URL"] == (
        "${AUDREY_URL:-http://audrey:8000}"
    )
    for sidecar in ("media-worker", "media-fetcher"):
        assert compose["services"][sidecar]["depends_on"]["audrey"] == {
            "condition": "service_healthy"
        }
        assert (
            compose["services"][sidecar]["environment"]["AUDREY_ENDPOINT"]
            == "http://audrey:8000"
        )
    assert compose["services"]["audrey"]["labels"] == {
        "net.unraid.docker.icon": (
            "${AUDREY_REPO_DIR:-/mnt/user/appdata/audrey_ai_2.0}/"
            "web/src/assets/models/audrey2.png"
        )
    }


def test_native_ui_proxy_preserves_auth_streams_uploads_and_static_boundaries():
    template = _text(UI_NGINX_TEMPLATE)

    assert "location ~ ^/(api|v1)(/|$)" in template
    assert "proxy_set_header Cf-Access-Jwt-Assertion" in template
    assert "proxy_buffering off;" in template
    assert "proxy_request_buffering off;" in template
    assert "proxy_read_timeout 3600s;" in template
    assert "proxy_pass ${AUDREY_UPSTREAM};" in template
    assert "resolver 127.0.0.11" not in template
    assert "client_max_body_size ${AUDREY_UI_MAX_BODY_SIZE};" in template
    assert "location ^~ /assets/" in template
    assert "try_files $uri =404;" in template
    assert "try_files $uri $uri/ /index.html;" in template
    assert "default-src 'self'" in template


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
    volumes = compose["services"]["audrey"]["volumes"]
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
