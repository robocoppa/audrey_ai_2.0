"""Optional planning shares local scheduling without blocking cancellation."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from audrey.models.ollama import OllamaError
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.planner import plan
from audrey.processing_policy import ProcessingPolicyError


async def _plan(ollama, gate, *, budget_s=2):
    return await plan(ollama, planner_model="local-router", user_text="A multipart question",
                      timeout_s=budget_s, max_subtasks=3, gate=gate, user_id="bob")


async def test_planner_waits_behind_worker_and_uses_request_owner_bucket():
    gate = FairLocalGate(concurrency=1)
    ollama = SimpleNamespace(chat=AsyncMock(return_value={
        "message": {"content": '{"subtasks": ["First part", "Second part"]}'},
    }))
    async with gate.acquire("worker", location="local", user_id="alice"):
        task = asyncio.create_task(_plan(ollama, gate))
        await asyncio.sleep(0)
        ollama.chat.assert_not_awaited()
        assert "bob" in gate._waiters
        assert not task.done()
    assert await task == ["First part", "Second part"]
    assert gate._available == 1 and not gate._waiters


async def test_planner_queue_timeout_skips_without_calling_model_or_leaking_slot():
    gate = FairLocalGate(concurrency=1)
    ollama = SimpleNamespace(chat=AsyncMock())
    async with gate.acquire("worker", location="local", user_id="alice"):
        assert await _plan(ollama, gate, budget_s=0.01) == []
        ollama.chat.assert_not_awaited()
        assert not gate._waiters and gate._available == 0
    assert gate._available == 1


async def test_cancelled_planner_queue_keeps_active_worker_slot():
    gate = FairLocalGate(concurrency=1)
    ollama = SimpleNamespace(chat=AsyncMock())
    async with gate.acquire("worker", location="local", user_id="alice"):
        task = asyncio.create_task(_plan(ollama, gate))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        ollama.chat.assert_not_awaited()
        assert not gate._waiters and gate._available == 0
    assert gate._available == 1


@pytest.mark.parametrize("error", [OllamaError("offline"),
    ProcessingPolicyError("private_model_unavailable"),
    ProcessingPolicyError("local_model_digest_mismatch")])
async def test_planner_failure_releases_gate_and_keeps_policy_refusals(error):
    gate = FairLocalGate(concurrency=1)
    ollama = SimpleNamespace(chat=AsyncMock(side_effect=error))
    if isinstance(error, ProcessingPolicyError) and error.code != "private_model_unavailable":
        with pytest.raises(ProcessingPolicyError) as caught:
            await _plan(ollama, gate)
        assert caught.value.code == "local_model_digest_mismatch"
    else:
        assert await _plan(ollama, gate) == []
    assert gate._available == 1 and not gate._waiters


async def test_planner_generation_shares_optional_deadline_and_releases_gate():
    started = asyncio.Event()

    async def generate(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    gate = FairLocalGate(concurrency=1)
    ollama = SimpleNamespace(chat=AsyncMock(side_effect=generate))
    assert await _plan(ollama, gate, budget_s=0.01) == []
    assert started.is_set()
    assert gate._available == 1 and not gate._waiters
