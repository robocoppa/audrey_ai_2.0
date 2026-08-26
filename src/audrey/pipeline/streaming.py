"""Client-neutral terminal state shared by streamed pipeline owners.

Async generators cannot return a value to their async-for caller. A
StreamTerminal is therefore passed into the generator: the inner stream reports
what actually happened, and the outer adapter records metrics, archive state,
and the protocol-specific finish frame exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StreamOutcome(StrEnum):
    """Bounded terminal labels used by stream metrics and persistence."""

    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    TRUNCATED = "truncated"


@dataclass(slots=True)
class StreamTerminal:
    """One-shot result channel from an inner stream to its outer owner."""

    _outcome: StreamOutcome | None = None
    _finish_reason: str | None = None

    @property
    def is_final(self) -> bool:
        return self._outcome is not None

    @property
    def outcome(self) -> StreamOutcome:
        if self._outcome is None:
            raise RuntimeError("stream terminal outcome has not been reported")
        return self._outcome

    @property
    def finish_reason(self) -> str | None:
        if self._outcome is None:
            raise RuntimeError("stream terminal outcome has not been reported")
        return self._finish_reason

    def finish(
        self,
        outcome: StreamOutcome,
        *,
        finish_reason: str | None = None,
    ) -> None:
        """Report the terminal result once; duplicate ownership is a bug."""
        if self._outcome is not None:
            raise RuntimeError(
                "stream terminal outcome already reported as "
                f"{self._outcome.value!r}"
            )
        self._outcome = outcome
        self._finish_reason = finish_reason

    def finish_if_unset(
        self,
        outcome: StreamOutcome,
        *,
        finish_reason: str | None = None,
    ) -> None:
        """Outer-owner fallback for cancellation or an escaping exception."""
        if self._outcome is None:
            self.finish(outcome, finish_reason=finish_reason)


__all__ = ["StreamOutcome", "StreamTerminal"]
