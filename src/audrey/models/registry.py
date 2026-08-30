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

    def location_of(self, model: str) -> Location:
        """Look up the registry-declared location of `model`.

        Walks every task list because a model can appear under multiple
        task types with a single location; the first match wins. Used by
        the deep panel and synthesizer to decide whether a chat call
        counts against the local GPU gate or the cloud concurrency cap.

        A declared location always wins. Only when the model is absent from
        `model_registry` does this fall through to `_location_from_tag`.
        """
        for specs in self._by_task.values():
            for spec in specs:
                if spec.name == model:
                    return spec.location
        return _location_from_tag(model)


def _location_from_tag(model: str) -> Location:
    """Infer a location for a model that holds no `model_registry` slot.

    ⚠️ "Unknown -> local" is the SAFE default and stays the default: a typo,
    or a model dropped from the registry but not from a pool, must go through
    the GPU gate rather than bypass it as "cloud". Only one thing overrides
    that, and it is not a guess — Ollama's own tag. A `:cloud` tag means the
    weights are not on this box and the request leaves it, so gating such a
    call against `GPU_CONCURRENCY=1` reserves a card nothing will use.

    ⚠️ 2026-08-29, the failure that motivated this: `glm-5.3-flash:cloud` was
    added to `passthrough.allowed_models` as a bake-off arm and deliberately
    given no registry slot (an entry there is a production role — it makes the
    model selectable as a pool failover). An OpenClaw bot then set it as its
    DEFAULT, and every one of its turns took the box's only GPU slot, held for
    the whole stream (`pipeline/passthrough.py` keeps the gate across the
    entire response), for a model running in Ollama Cloud that touches no GPU.
    Nothing failed — local work simply queued behind a cloud call, invisibly.

    Only the TAG is inspected, never the whole name, so a local model that
    merely has "cloud" in its name is unaffected. Both forms Ollama uses are
    covered: a bare `:cloud` tag and a qualified one (`:397b-cloud`).
    """
    _, sep, tag = model.rpartition(":")
    if not sep:
        return "local"
    if tag == "cloud" or tag.endswith("-cloud"):
        return "cloud"
    return "local"


def _parse_location(raw: object, *, task: str, model: str) -> Location:
    if isinstance(raw, str) and raw in _VALID_LOCATIONS:
        return cast(Location, raw)
    allowed = ", ".join(sorted(_VALID_LOCATIONS))
    raise ValueError(
        f"Invalid model location for {task}/{model}: {raw!r}. "
        f"Expected one of: {allowed}"
    )


__all__ = ["ModelRegistry", "ModelSpec", "TaskType"]
