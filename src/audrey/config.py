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

    # Search
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")

    # Pipeline tunables (override YAML)
    complexity_token_threshold: int | None = Field(default=None, alias="COMPLEXITY_TOKEN_THRESHOLD")
    gpu_concurrency: int | None = Field(default=None, alias="GPU_CONCURRENCY")
    tool_max_rounds: int | None = Field(default=None, alias="TOOL_MAX_ROUNDS")
    planning_min_tokens: int | None = Field(default=None, alias="PLANNING_MIN_TOKENS")
    max_deep_workers_cloud: int | None = Field(default=None, alias="MAX_DEEP_WORKERS_CLOUD")
    max_inflight_per_user: int | None = Field(default=None, alias="MAX_INFLIGHT_PER_USER")

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


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load config once per process. Tests can call `get_config.cache_clear()`."""
    env = EnvOverrides()
    # Resolve config path relative to CWD if not absolute
    cfg_path = env.audrey_config if env.audrey_config.is_absolute() else Path.cwd() / env.audrey_config
    yaml_cfg = _load_yaml(cfg_path)
    return Config(yaml_cfg, env)


# Convenience for tests/REPL
def reload_config() -> Config:
    get_config.cache_clear()
    return get_config()


__all__ = ["Config", "EnvOverrides", "get_config", "reload_config"]
