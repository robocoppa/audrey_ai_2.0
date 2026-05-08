"""Model registry — ranked lists of candidate models per task type.

Loaded once at startup from `config.yaml` under `model_registry`. Provides
`first_healthy(task, predicate)` for the fast path's "pick highest-priority
healthy model" selection, and `candidates(task)` for the deep panel's
multi-worker dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from audrey.config import Config

TaskType = Literal["code", "reasoning", "general", "vl"]
Location = Literal["local", "cloud"]
_VALID_LOCATIONS = {"local", "cloud"}


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
                    name=entry["name"],
                    priority=int(entry.get("priority", 0)),
                    speed=int(entry.get("speed", 50)),
                    quality=int(entry.get("quality", 50)),
                    location=_parse_location(
                        entry.get("location", "local"),
                        task=task,
                        model=entry["name"],
                    ),
                )
                for entry in entries
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


def _parse_location(raw: object, *, task: str, model: str) -> Location:
    if isinstance(raw, str) and raw in _VALID_LOCATIONS:
        return cast(Location, raw)
    allowed = ", ".join(sorted(_VALID_LOCATIONS))
    raise ValueError(
        f"Invalid model location for {task}/{model}: {raw!r}. "
        f"Expected one of: {allowed}"
    )


__all__ = ["ModelRegistry", "ModelSpec", "TaskType"]
