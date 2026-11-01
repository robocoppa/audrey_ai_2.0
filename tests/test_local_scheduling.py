"""Trusted scheduling scopes never leak into later interactive requests."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from audrey.scheduling import LocalSchedulingSettings, current_local_workload, local_workload


@pytest.mark.parametrize("value", [0, -1, 301, float("inf"), float("nan"), True, "30"])
def test_invalid_runtime_aging_is_refused(value):
    with pytest.raises(ValidationError):
        LocalSchedulingSettings.model_validate({"background_aging_seconds": value})


def test_old_configuration_uses_bounded_default_without_changing_user_caps():
    old = {"max_inflight_per_user": 1, "max_tracked_users": 10}
    assert LocalSchedulingSettings.model_validate(old).background_aging_seconds == 30.0
    assert LocalSchedulingSettings.model_validate({**old, "background_aging_seconds": 12}).background_aging_seconds == 12.0


async def test_background_scope_is_task_local_and_restores_after_cancellation():
    entered = asyncio.Event()

    async def background():
        with local_workload("background"):
            assert current_local_workload() == "background"
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(background())
    await entered.wait()
    assert current_local_workload() == "interactive"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert current_local_workload() == "interactive"


def test_nested_scope_restores_classification_even_when_it_raises():
    with local_workload("background"):
        with pytest.raises(RuntimeError), local_workload("interactive"):
            assert current_local_workload() == "interactive"
            raise RuntimeError("Stopped")
        assert current_local_workload() == "background"
    assert current_local_workload() == "interactive"


def test_browser_like_priority_values_are_not_accepted_as_internal_scopes():
    with pytest.raises(ValueError), local_workload("urgent"):
        pytest.fail("Invalid scheduling scope was entered.")
    assert current_local_workload() == "interactive"
