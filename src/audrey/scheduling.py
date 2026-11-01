"""Server-owned workload classification shared by both admission queues."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ANON_USER_BUCKET = "__anon__"
LocalWorkload = Literal["interactive", "background"]
_local_workload: ContextVar[LocalWorkload] = ContextVar("audrey_local_workload", default="interactive")
_background_queued_at: ContextVar[float | None] = ContextVar("audrey_background_queued_at", default=None)


class LocalSchedulingSettings(BaseModel):
    """The aging threshold is shared by the per-user and GPU admission queues."""

    model_config = ConfigDict(strict=True)

    background_aging_seconds: float = Field(default=30.0, gt=0, le=300, allow_inf_nan=False)


def current_local_workload() -> LocalWorkload:
    return _local_workload.get()


def current_background_queued_at() -> float | None:
    """Original monotonic admission time, shared across background handoffs."""
    return _background_queued_at.get()


@contextmanager
def local_workload(workload: LocalWorkload, *, queued_at: float | None = None) -> Iterator[None]:
    """Classify trusted internal work; browser input never selects a priority."""
    if workload not in {"interactive", "background"}:
        raise ValueError("Invalid local generation workload.")
    if queued_at is not None and (workload != "background" or not math.isfinite(queued_at)):
        raise ValueError("Only background work can provide a finite queued_at time.")
    token = _local_workload.set(workload)
    age_token = _background_queued_at.set(
        (queued_at if queued_at is not None else _background_queued_at.get())
        if workload == "background" else None
    )
    try:
        yield
    finally:
        _background_queued_at.reset(age_token)
        _local_workload.reset(token)


__all__ = [
    "ANON_USER_BUCKET", "LocalSchedulingSettings", "current_local_workload",
    "current_background_queued_at", "local_workload",
]
