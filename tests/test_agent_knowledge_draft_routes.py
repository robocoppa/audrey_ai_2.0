"""Transient report helpers use native ownership and cancel without persistence."""

from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from audrey.routes.app import router

_SPEC = importlib.util.spec_from_file_location(
    "draft_route_fixture", Path(__file__).parent / "smoke" / "smoke_agent_knowledge_drafts.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)


@pytest.fixture
async def drafts(tmp_path):
    async with fixtures.Scenario(fixtures.load_runtime(), tmp_path) as scenario:
        yield scenario


async def test_real_suggestions_are_owner_scoped_uncacheable_and_transient(drafts):
    before = drafts.snapshot()
    response = await drafts.post()
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"task_id": drafts.task["id"], "task_revision": drafts.task["revision"],
                               "suggestions": fixtures.SUGGESTIONS}
    assert drafts.snapshot() == before
    drafts.assert_released()


async def test_empty_valid_batch_is_success_without_creating_a_placeholder_note(drafts):
    before = drafts.snapshot()
    drafts.response = drafts.reply([])
    response = await drafts.post()
    assert response.json()["suggestions"] == []
    assert drafts.snapshot() == before
    drafts.assert_released()


async def test_unauthenticated_request_cannot_reach_the_helper():
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/agents/missing/knowledge-drafts", json={"expected_revision": 1})
    assert response.status_code == 401
    assert not hasattr(app.state, "agent_knowledge_drafts")


async def test_personal_token_without_native_scope_cannot_generate(drafts):
    drafts.current = replace(drafts.owner, auth_method="personal_token", scopes=frozenset())
    await drafts.post(status=403)
    assert not drafts.clients


@pytest.mark.parametrize("body", [
    {}, {"expected_revision": 0}, {"expected_revision": True}, {"expected_revision": "1"},
    {"expected_revision": 1.0}, {"expected_revision": 1, "project_id": "untrusted"},
])
async def test_input_validation_precedes_model_or_source_work(drafts, body):
    before = drafts.snapshot()
    response = await drafts.client.post(f"/api/agents/{drafts.task['id']}/knowledge-drafts", json=body)
    assert response.status_code == 422
    assert not drafts.clients and drafts.snapshot() == before


async def test_foreign_and_missing_reports_share_a_content_free_404(drafts):
    drafts.current = drafts.other
    foreign = await drafts.post(status=404)
    missing = await drafts.post(task={"id": "missing", "revision": 1}, status=404)
    assert foreign.json() == missing.json()
    assert not drafts.clients and drafts.task["result"] not in foreign.text


async def test_stale_or_unfinished_report_refuses_before_local_metadata_lookup(drafts):
    await drafts.post(revision=drafts.task["revision"] - 1, status=409)
    queued = await drafts.store.agents.create(user_id=drafts.owner.user_id, objective="Unfinished.", request_id="unfinished")
    await drafts.post(task=queued, status=409)
    assert not drafts.clients


async def test_disabled_local_helper_returns_503_without_dispatch(drafts):
    drafts.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
    before = drafts.snapshot()
    response = await drafts.post(status=503)
    assert set(response.json()) == {"detail"}
    assert not drafts.clients and drafts.snapshot() == before


async def test_unavailable_application_store_is_a_native_503(drafts):
    drafts.app.state.application_store = None
    await drafts.post(status=503)
    assert not drafts.clients


@pytest.mark.parametrize("bad_reply", ["invalid", "truncated"])
async def test_model_failures_return_fixed_503_without_partial_output_or_writes(drafts, bad_reply):
    before = drafts.snapshot()
    if bad_reply == "invalid":
        drafts.response["message"]["content"] = "private unstructured model output"
    else:
        drafts.response["done_reason"] = "length"
    response = await drafts.post(status=503)
    assert set(response.json()) == {"detail"}
    assert "private unstructured" not in response.text
    assert "suggestions" not in response.json()
    assert drafts.snapshot() == before
    drafts.assert_released()


async def test_route_storage_failure_is_sanitized(drafts, monkeypatch):
    async def broken(**_scope):
        raise sqlite3.OperationalError("private report path and SQL contents")

    monkeypatch.setattr(drafts.service, "generate", broken)
    async with asyncio.timeout(3):
        response = await drafts.post(status=503)
    assert response.json() == {"detail": "Knowledge suggestions are temporarily unavailable."}
    assert "private report" not in response.text


@pytest.mark.parametrize("change,status", [("rerun", 409), ("delete", 404)])
async def test_final_route_fence_discards_a_result_when_report_changes_after_service_returns(drafts, monkeypatch, change, status):
    original = drafts.service.generate

    async def raced(**scope):
        result = await original(**scope)
        if change == "delete":
            await drafts.store.agents.delete(user_id=drafts.owner.user_id, task_id=drafts.task["id"],
                                             expected_revision=drafts.task["revision"])
        else:
            await drafts.store.agents.control(user_id=drafts.owner.user_id, task_id=drafts.task["id"],
                                              expected_revision=drafts.task["revision"], action="rerun")
        return result

    monkeypatch.setattr(drafts.service, "generate", raced)
    response = await drafts.post(status=status)
    assert set(response.json()) == {"detail"}
    assert fixtures.SUGGESTIONS[0]["text"] not in response.text
    drafts.assert_released()


async def test_browser_disconnect_cancels_generation_and_releases_capacity(drafts, monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def pending():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def disconnected(_request):
        return started.is_set()

    drafts.on_chat = pending
    monkeypatch.setattr(Request, "is_disconnected", disconnected)
    before = drafts.snapshot()
    async with asyncio.timeout(3):
        response = await drafts.post(status=499)
    assert cancelled.is_set()
    assert response.json() == {"detail": "Suggestion generation was cancelled."}
    assert drafts.snapshot() == before
    drafts.assert_released()


async def test_request_task_cancellation_releases_clients_admission_and_active_account(drafts):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def pending():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    drafts.on_chat = pending
    before = drafts.snapshot()
    work = asyncio.create_task(drafts.post())
    try:
        async with asyncio.timeout(3):
            await started.wait()
            work.cancel()
            with pytest.raises(asyncio.CancelledError):
                await work
    finally:
        if not work.done():
            work.cancel()
        await asyncio.gather(work, return_exceptions=True)
    assert cancelled.is_set() and drafts.snapshot() == before
    drafts.assert_released()
