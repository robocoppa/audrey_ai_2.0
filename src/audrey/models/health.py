"""In-process model health tracker with exponential backoff.

When a model returns an error or times out, we cool it down so the registry
doesn't immediately re-pick it. A new failure after the previous cooldown
doubles the next cooldown (capped). Failures during a window do not extend it. A successful
call clears the record. Callers normally skip cooling candidates; the fast
path probes the earliest recovery when its entire candidate pool is cooling.
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

    Defaults: start at 5s, double after each elapsed cooldown, cap at 5 min.
    `snapshot()` reports current cooldowns; internal failure history is bounded.
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

    def cooldown_remaining(self, model: str) -> float:
        """Seconds until the candidate can normally be selected again."""
        state = self._by_model.get(model)
        return max(0.0, state.cooldown_until - time.monotonic()) if state else 0.0

    def record_success(self, model: str) -> None:
        self._by_model.pop(model, None)

    def record_failure(self, model: str, error: str) -> None:
        now = time.monotonic()
        state = self._by_model.setdefault(model, _HealthState())
        if now < state.cooldown_until:
            # In-flight calls share this outage. Count the window once, rather
            # than extending downtime for each request that was already sent.
            return
        state.consecutive_failures += 1
        backoff = min(self._base * (2 ** (state.consecutive_failures - 1)), self._max)
        state.cooldown_until = now + backoff
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
