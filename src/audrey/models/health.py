"""In-process model health tracker with exponential backoff.

When a model returns an error or times out, we cool it down so the registry
doesn't immediately re-pick it. Each failure doubles the cooldown (capped).
A successful call clears the record.

Phase 4 scope: the tracker itself + `is_healthy()`. It's not yet wired into
request dispatch — that happens in Phase 5 where we actually fail over.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class _HealthState:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0  # monotonic seconds
    last_error: str = ""
    history: list[tuple[float, str]] = field(default_factory=list)


class HealthTracker:
    """Per-model health with exp-backoff cooldowns.

    Defaults: start at 5s cooldown, double per failure, cap at 5 min.
    Records the last 20 failures per model for debugging via `snapshot()`.
    """

    def __init__(
        self,
        *,
        base_cooldown_s: float = 5.0,
        max_cooldown_s: float = 300.0,
        history_size: int = 20,
    ) -> None:
        self._base = base_cooldown_s
        self._max = max_cooldown_s
        self._history_size = history_size
        self._by_model: dict[str, _HealthState] = {}

    def is_healthy(self, model: str) -> bool:
        state = self._by_model.get(model)
        if state is None:
            return True
        return time.monotonic() >= state.cooldown_until

    def record_success(self, model: str) -> None:
        self._by_model.pop(model, None)

    def record_failure(self, model: str, error: str) -> None:
        state = self._by_model.setdefault(model, _HealthState())
        state.consecutive_failures += 1
        backoff = min(self._base * (2 ** (state.consecutive_failures - 1)), self._max)
        state.cooldown_until = time.monotonic() + backoff
        state.last_error = error
        state.history.append((time.time(), error))
        if len(state.history) > self._history_size:
            state.history = state.history[-self._history_size :]

    def snapshot(self) -> dict[str, dict]:
        now = time.monotonic()
        return {
            model: {
                "consecutive_failures": s.consecutive_failures,
                "cooling_down_for_s": max(0.0, round(s.cooldown_until - now, 1)),
                "last_error": s.last_error,
            }
            for model, s in self._by_model.items()
        }


__all__ = ["HealthTracker"]
