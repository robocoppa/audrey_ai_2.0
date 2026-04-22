"""Config loader.

Merges three sources (last wins):
  1. `config.yaml` at the repo root (or `AUDREY_CONFIG` env var)
  2. Environment variables listed in `EnvOverrides` (12-factor friendly)
  3. Runtime patches (not used in Phase 4)

`config.yaml` is the source of truth for the model registry and pipeline
knobs — see the top of that file for the authoritative schema. Env vars
are only for deployment-specific things (ports, URLs, secrets).
"""

from __future__ import annotations

import os
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

    # Tools (comma-separated URLs)
    tool_servers: str = Field(default="http://custom-tools:8001", alias="TOOL_SERVERS")

    # KB
    kb_dataset_paths: str = Field(default="/datasets/geology", alias="KB_DATASET_PATHS")

    # Search
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")

    # Pipeline tunables (override YAML)
    complexity_token_threshold: int | None = Field(default=None, alias="COMPLEXITY_TOKEN_THRESHOLD")
    gpu_concurrency: int | None = Field(default=None, alias="GPU_CONCURRENCY")
    tool_max_rounds: int | None = Field(default=None, alias="TOOL_MAX_ROUNDS")
    planning_min_tokens: int | None = Field(default=None, alias="PLANNING_MIN_TOKENS")
    max_deep_workers_cloud: int | None = Field(default=None, alias="MAX_DEEP_WORKERS_CLOUD")

    # Data dir (for any local sqlite/caches Audrey itself owns)
    data_dir: Path = Field(default=Path("/data"), alias="AUDREY_DATA_DIR")


class Config:
    """Merged YAML + env view. Access via `get_config()`."""

    def __init__(self, yaml_cfg: dict[str, Any], env: EnvOverrides) -> None:
        self._yaml = yaml_cfg
        self.env = env
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        if (v := self.env.complexity_token_threshold) is not None:
            self._yaml.setdefault("complexity", {})["token_threshold"] = v
        if (v := self.env.gpu_concurrency) is not None:
            self._yaml.setdefault("gpu", {})["concurrency"] = v
        if (v := self.env.tool_max_rounds) is not None:
            self._yaml.setdefault("tools", {})["max_rounds"] = v
        if (v := self.env.planning_min_tokens) is not None:
            self._yaml.setdefault("agentic", {}).setdefault("planning", {})["min_prompt_tokens"] = v
        if (v := self.env.max_deep_workers_cloud) is not None:
            self._yaml.setdefault("agentic", {})["max_deep_workers_cloud"] = v
        self._yaml.setdefault("tools", {})["servers"] = [
            s.strip() for s in self.env.tool_servers.split(",") if s.strip()
        ]
        self._yaml.setdefault("kb", {})["dataset_paths"] = [
            p.strip() for p in self.env.kb_dataset_paths.split(",") if p.strip()
        ]

    # Convenient typed accessors — add more as needed in later phases.
    @property
    def version(self) -> str:
        return self._yaml.get("version", "0.0.0")

    @property
    def router(self) -> dict[str, Any]:
        return self._yaml.get("router", {})

    @property
    def model_registry(self) -> dict[str, list[dict[str, Any]]]:
        return self._yaml.get("model_registry", {})

    @property
    def timeouts(self) -> dict[str, int]:
        return self._yaml.get("timeouts", {})

    @property
    def tools(self) -> dict[str, Any]:
        return self._yaml.get("tools", {})

    @property
    def raw(self) -> dict[str, Any]:
        return self._yaml


def _load_yaml(path: Path) -> dict[str, Any]:
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


def get_env() -> EnvOverrides:
    """Access the raw env object (e.g. for constructing clients before config is parsed)."""
    return EnvOverrides()


__all__ = ["Config", "EnvOverrides", "get_config", "get_env", "reload_config"]
