"""Model registry — ranked lists of candidate models per task type.

Phase 4 scope: data shape + simple "pick highest-priority healthy model"
selector. Multi-model panels, deep-panel dispatch, and escalation are layered
on in later phases (5+).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from audrey.config import Config

TaskType = Literal["code", "reasoning", "general", "vl"]
Location = Literal["local", "cloud"]


@dataclass(slots=True, frozen=True)
class ModelSpec:
    name: str
    priority: int
    speed: int
    quality: int
    location: Location


class ModelRegistry:
    """Typed view over `config.model_registry`.

    Lookup is O(n) per task type but n is tiny (≤10) so no need to index.
    """

    def __init__(self, cfg: Config) -> None:
        self._by_task: dict[str, list[ModelSpec]] = {}
        for task, entries in cfg.model_registry.items():
            specs = [
                ModelSpec(
                    name=e["name"],
                    priority=int(e.get("priority", 0)),
                    speed=int(e.get("speed", 50)),
                    quality=int(e.get("quality", 50)),
                    location=e.get("location", "local"),
                )
                for e in entries
            ]
            specs.sort(key=lambda s: s.priority, reverse=True)
            self._by_task[task] = specs

    def candidates(self, task: TaskType) -> list[ModelSpec]:
        return list(self._by_task.get(task, ()))

    def first_healthy(self, task: TaskType, is_healthy) -> ModelSpec | None:
        """Return the highest-priority candidate for which `is_healthy(name)` is True."""
        for spec in self._by_task.get(task, ()):
            if is_healthy(spec.name):
                return spec
        return None

    def all_task_types(self) -> list[str]:
        return list(self._by_task.keys())


__all__ = ["ModelRegistry", "ModelSpec", "TaskType"]
