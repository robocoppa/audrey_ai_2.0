"""Config loader.

Merges two sources (env wins):
  1. `config.yaml` at the repo root (or `AUDREY_CONFIG` env var)
  2. Environment variables listed in `EnvOverrides` (12-factor friendly)

`config.yaml` is the source of truth for the model registry and pipeline
knobs — see the top of that file for the authoritative schema. Env vars
are only for deployment-specific things (ports, URLs, secrets).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvOverrides(BaseSettings):
    """Env-driven settings. These override `config.yaml` where they overlap."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    audrey_config: Path = Field(default=Path("config.yaml"), alias="AUDREY_CONFIG")

    # Ollama
    ollama_host: str = Field(default="http://ollama:11434", alias="OLLAMA_HOST")

    # Qdrant
    qdrant_host: str = Field(default="qdrant", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")

    # Tools (comma-separated URLs). Default None so YAML's `tools.servers`
    # wins when the env var is unset; setting TOOL_SERVERS overrides YAML.
    tool_servers: str | None = Field(default=None, alias="TOOL_SERVERS")

    # KB dataset paths (comma-separated). Default None so YAML's
    # `kb.dataset_paths` wins when KB_DATASET_PATHS is unset.
    kb_dataset_paths: str | None = Field(default=None, alias="KB_DATASET_PATHS")

    # Open WebUI — `require_user` proxies the browser's bearer token here
    # to validate identity. Same-origin via cloudflared, so this is an
    # internal ollama-net URL.
    owui_url: str = Field(default="http://open-webui:8080", alias="OWUI_URL")

    # The same Open WebUI, addressed the way a BROWSER can reach it — which is
    # a different string and must not be conflated with `owui_url` above.
    # `http://open-webui:8080` resolves inside the docker network and nowhere
    # else, so emitting it into a page produces a link that fails for everyone.
    #
    # Used only for the upload page's Home link. Empty means no link is
    # rendered: the upload page is opened in a new tab from an OWUI banner
    # (`target="_blank"`), so there is no history to fall back on, and a Home
    # button that goes nowhere is worse than none. Set it to whatever you type
    # in the address bar to reach OWUI, e.g. `https://chat.example.com`.
    owui_public_url: str = Field(default="", alias="OWUI_PUBLIC_URL")

    kb_service_token: str = Field(default="", alias="KB_SERVICE_TOKEN")

    # Search
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")

    # Pipeline tunables (override YAML)
    complexity_token_threshold: int | None = Field(default=None, alias="COMPLEXITY_TOKEN_THRESHOLD")
    gpu_concurrency: int | None = Field(default=None, alias="GPU_CONCURRENCY")
    tool_max_rounds: int | None = Field(default=None, alias="TOOL_MAX_ROUNDS")
    planning_min_tokens: int | None = Field(default=None, alias="PLANNING_MIN_TOKENS")
    max_deep_workers_cloud: int | None = Field(default=None, alias="MAX_DEEP_WORKERS_CLOUD")
    max_inflight_per_user: int | None = Field(default=None, alias="MAX_INFLIGHT_PER_USER")

    # Video job lease knobs (override `kb.video` in YAML). Env-overridable
    # specifically so the Phase 33 lease verification doesn't require editing
    # `config.yaml` on the box — that edit dirties the deployed working tree
    # and has to be reverted by hand afterwards, which is the kind of step
    # that gets forgotten and silently leaves production on a 1-minute lease.
    # Put VIDEO_LEASE_MINUTES=1 in `.env` (gitignored) instead, and delete the
    # line when the run is over.
    video_lease_minutes: int | None = Field(default=None, alias="VIDEO_LEASE_MINUTES")
    video_max_attempts: int | None = Field(default=None, alias="VIDEO_MAX_ATTEMPTS")

    # KB watcher toggle (Pydantic Settings parses 1/true/yes as True)
    kb_watcher_enabled: bool = Field(default=False, alias="KB_WATCHER_ENABLED")


class Config:
    """Merged YAML + env view. Access via `get_config()`."""

    def __init__(self, yaml_cfg: dict[str, Any], env: EnvOverrides) -> None:
        self._merged = yaml_cfg
        self.env = env
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        if (v := self.env.complexity_token_threshold) is not None:
            self._merged.setdefault("complexity", {})["token_threshold"] = v
        if (v := self.env.gpu_concurrency) is not None:
            self._merged.setdefault("gpu", {})["concurrency"] = v
        if (v := self.env.tool_max_rounds) is not None:
            self._merged.setdefault("tools", {})["max_rounds"] = v
        if (v := self.env.planning_min_tokens) is not None:
            self._merged.setdefault("agentic", {}).setdefault("planning", {})["min_prompt_tokens"] = v
        if (v := self.env.max_deep_workers_cloud) is not None:
            self._merged.setdefault("agentic", {})["max_deep_workers_cloud"] = v
        if (v := self.env.max_inflight_per_user) is not None:
            self._merged.setdefault("fairness", {})["max_inflight_per_user"] = v
        if (v := self.env.tool_servers) is not None:
            self._merged.setdefault("tools", {})["servers"] = [
                s.strip() for s in v.split(",") if s.strip()
            ]
        if (v := self.env.kb_dataset_paths) is not None:
            self._merged.setdefault("kb", {})["dataset_paths"] = [
                p.strip() for p in v.split(",") if p.strip()
            ]
        if (v := self.env.video_lease_minutes) is not None:
            self._merged.setdefault("kb", {}).setdefault("video", {})["lease_minutes"] = v
        if (v := self.env.video_max_attempts) is not None:
            self._merged.setdefault("kb", {}).setdefault("video", {})["max_attempts"] = v

    # Convenient typed accessors — add more as needed in later phases.
    @property
    def version(self) -> str:
        return self._merged.get("version", "0.0.0")

    @property
    def router(self) -> dict[str, Any]:
        return self._merged.get("router", {})

    @property
    def model_registry(self) -> dict[str, list[dict[str, Any]]]:
        return self._merged.get("model_registry", {})

    @property
    def timeouts(self) -> dict[str, int]:
        return self._merged.get("timeouts", {})

    @property
    def tools(self) -> dict[str, Any]:
        return self._merged.get("tools", {})

    @property
    def raw(self) -> dict[str, Any]:
        return self._merged


def _load_yaml(path: Path) -> dict[str, Any]:
    # Intentional fast-fail: a missing or malformed config has no sensible
    # default, so we let the exception bubble out of lifespan and crash the
    # process. Asymmetric with the qdrant boot path, which degrades.
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. Set AUDREY_CONFIG or run from repo root."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level of {path}, got {type(data).__name__}")
    return data


def _validate_deep_panel_pools(merged: dict[str, Any]) -> None:
    """Reject deep-panel configs with a missing required role or an unknown
    model name in any model slot.

    Two pool shapes are validated:

    - **Flat pools** (`deep_panel`, `deep_panel_cloud`, `deep_panel_local`):
      require a `synthesizer`; name models in `workers` / `synthesizer` /
      `fallback_synth`.
    - **Staged pool** (`deep_panel_research`): a body carrying a
      `researchers` list is the staged `audrey_research` shape. It requires
      `researchers` (non-empty), `verifier`, and `writer`; names models in
      `researchers[*]` / `verifier` / `writer` / `fallback_synth`. It has no
      `synthesizer` — the writer produces the answer.

    Two failure modes this catches at boot instead of at request time:

    1. A pool/task missing its required role — the pipeline would `KeyError`
       (flat: `pick_synthesizer`) or have nothing to dispatch (staged) on the
       first request.
    2. A model name absent from `model_registry`. At request time an unknown
       model passes `HealthTracker.is_healthy` (unknown → True), defaults to
       `location="local"` in `ModelRegistry.location_of`, then fails the
       Ollama call — wasting a GPU-gate slot on a model that can't load before
       degrading. Catching it here turns a silent dud-fallback into a loud
       boot failure.

    Same fast-fail posture as `_load_yaml`.
    """
    registry_names: set[str] = set()
    for specs in (merged.get("model_registry") or {}).values():
        if isinstance(specs, list):
            for spec in specs:
                if isinstance(spec, dict) and spec.get("name"):
                    registry_names.add(str(spec["name"]))

    errors: list[str] = []
    for pool_key in (k for k in merged if k.startswith("deep_panel")):
        pool = merged.get(pool_key) or {}
        if not isinstance(pool, dict):
            errors.append(f"{pool_key}: expected dict, got {type(pool).__name__}")
            continue
        for task, body in pool.items():
            if not isinstance(body, dict):
                errors.append(f"{pool_key}/{task}: expected dict, got {type(body).__name__}")
                continue
            # A `researchers` key marks the staged `audrey_research` shape;
            # everything else is a flat panel. The two have different required
            # roles and different model-bearing slots.
            staged = "researchers" in body
            named: list[tuple[str, str]] = []
            if staged:
                researchers = body.get("researchers") or []
                if not researchers:
                    errors.append(f"{pool_key}/{task}: `researchers` must be non-empty")
                for slot in ("verifier", "writer"):
                    if not body.get(slot):
                        errors.append(f"{pool_key}/{task}: missing required `{slot}` key")
                for r in researchers:
                    named.append(("researcher", str(r)))
                # `factchecker` is optional (omit → stage skipped); validated
                # only when present.
                for slot in ("verifier", "factchecker", "writer", "fallback_synth"):
                    if body.get(slot):
                        named.append((slot, str(body[slot])))
            else:
                if not body.get("synthesizer"):
                    errors.append(f"{pool_key}/{task}: missing required `synthesizer` key")
                for w in (body.get("workers") or []):
                    named.append(("worker", str(w)))
                for slot in ("synthesizer", "fallback_synth"):
                    if body.get(slot):
                        named.append((slot, str(body[slot])))
            # Every named model must exist in the registry, or scheduling
            # (local vs cloud) and health checks operate on a phantom. Skip
            # this check entirely when no registry is present (stripped-down
            # configs have nothing to validate against).
            if not registry_names:
                continue
            for slot, name in named:
                if name not in registry_names:
                    errors.append(
                        f"{pool_key}/{task}: {slot} {name!r} is not in model_registry"
                    )
    if errors:
        bullets = "\n  - " + "\n  - ".join(errors)
        raise ValueError(f"Invalid deep-panel configuration:{bullets}")


def _validate_upload_limits(merged: dict[str, Any]) -> None:
    """Reject a per-user quota smaller than the largest permitted single upload.

    These are two independent numbers in two separate config blocks that have
    to agree, and nothing about editing either one prompts you to check the
    other. `max_user_bytes` was 1 GiB while `chunked.max_upload_mb` was 2048 —
    so the transport advertised a 2 GiB ceiling that the quota could never
    accept, and any upload over 1 GiB was refused no matter how empty the
    user's storage was.

    That failure is quiet in the worst way: the refusal is a correct-looking
    413 naming the quota, so it reads as "you are out of space" rather than
    "these limits contradict each other". A user with an empty account would
    be told to free some up.

    Checked at boot, alongside the deep-panel pools, for the same reason —
    both are configuration that cannot be wrong at request time without
    someone having already been told something false.
    """
    kb = merged.get("kb") or {}
    if not isinstance(kb, dict):
        return
    quota = kb.get("max_user_bytes")
    chunked = kb.get("chunked") or {}
    per_file_mb = chunked.get("max_upload_mb") if isinstance(chunked, dict) else None
    if not isinstance(quota, int) or not isinstance(per_file_mb, int):
        return

    per_file = per_file_mb * 1024 * 1024
    if quota < per_file:
        raise ValueError(
            "Invalid upload configuration:\n"
            f"  - kb.max_user_bytes ({quota} bytes) is below "
            f"kb.chunked.max_upload_mb ({per_file_mb} MB = {per_file} bytes), "
            "so the largest upload the transport accepts can never fit the "
            "quota. Raise max_user_bytes or lower chunked.max_upload_mb."
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load config once per process. Tests can call `get_config.cache_clear()`."""
    env = EnvOverrides()
    # Resolve config path relative to CWD if not absolute
    cfg_path = env.audrey_config if env.audrey_config.is_absolute() else Path.cwd() / env.audrey_config
    yaml_cfg = _load_yaml(cfg_path)
    cfg = Config(yaml_cfg, env)
    _validate_deep_panel_pools(cfg.raw)
    _validate_upload_limits(cfg.raw)
    return cfg


# Convenience for tests/REPL
def reload_config() -> Config:
    get_config.cache_clear()
    return get_config()


__all__ = ["Config", "EnvOverrides", "get_config", "reload_config"]
