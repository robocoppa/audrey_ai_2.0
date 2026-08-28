"""Tests for routes/admin.py — admin-only ops endpoints.

The admin handlers are thin orchestration over things that already
have their own test coverage (`reconcile_once`, the auth cache, the
chat-archive client). We pin the *wiring* here: that the handler
reads the right field from `app.state`, calls the right helper,
returns the right shape. We don't re-test the underlying helpers.

Pattern: call handler functions directly with a fake `Request` —
avoids spinning up a full FastAPI app + auth dependency injection +
real OWUI/Qdrant clients. Same pattern `test_auth.py` uses for
`_probe_owui`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

# ─── kb_reconcile admin endpoint ───────────────────────────────────────


class _FakeReconcileResult:
    """Stand-in for `kb.reconcile.ReconcileResult`.

    Only the bits the handler reads: `total_orphans_deleted` (logged)
    and `to_dict()` (returned to the client).
    """

    def __init__(
        self,
        total_orphans_deleted: int,
        by_collection: dict[str, dict] | None = None,
        total_elapsed_s: float = 0.123,
    ) -> None:
        self.total_orphans_deleted = total_orphans_deleted
        self._by_collection = by_collection or {
            "kb_text": {
                "checked": 10, "orphans_deleted": 1, "points_in_orphans": 4,
                "elapsed_s": 0.05, "error": "",
            },
            "kb_images": {
                "checked": 3, "orphans_deleted": 0, "points_in_orphans": 0,
                "elapsed_s": 0.02, "error": "",
            },
        }
        self._total_elapsed_s = total_elapsed_s

    def to_dict(self) -> dict:
        return {
            "by_collection": dict(self._by_collection),
            "total_orphans_deleted": self.total_orphans_deleted,
            "total_elapsed_s": self._total_elapsed_s,
        }


def _fake_request(qdrant_obj):
    """Build the minimal `request` shape the handler reads.

    The handler accesses `request.app.state.qdrant` only; a nested
    SimpleNamespace is enough.
    """
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(qdrant=qdrant_obj))
    )


def _fake_qdrant(text="kb_text", image="kb_images"):
    """Stand-in qdrant carrying the configured collection names the handler
    forwards to `reconcile_once`. `QdrantKB` exposes these as plain attrs."""
    return SimpleNamespace(text_collection=text, image_collection=image)


def _fake_admin(email: str = "admin@example.com"):
    """Build the minimal `AuthedUser` shape the handler reads."""
    # Handler only reads `me.email` for the log line.
    return SimpleNamespace(email=email, role="admin", owui_id="abc123")


@pytest.mark.asyncio
async def test_kb_reconcile_returns_structured_summary(monkeypatch):
    # The handler should call `reconcile_once(qdrant, text_collection=...,
    # image_collection=...)` with the qdrant's *configured* collection names
    # (so a renamed collection is swept, not the defaults), log the orphan
    # count + the admin's email, and return `result.to_dict()`. We pin all.
    captured: dict[str, object] = {}

    async def _fake_reconcile_once(qdrant, *, text_collection, image_collection):
        captured["qdrant"] = qdrant
        captured["text_collection"] = text_collection
        captured["image_collection"] = image_collection
        return _FakeReconcileResult(total_orphans_deleted=1)

    from audrey.routes import admin as admin_module
    monkeypatch.setattr(admin_module, "reconcile_once", _fake_reconcile_once)

    # Stand-in qdrant carrying *non-default* configured names (a post-rename
    # deployment) so we can prove the handler forwards them rather than letting
    # reconcile_once fall back to "kb_text"/"kb_images".
    sentinel_qdrant = SimpleNamespace(
        text_collection="renamed_text",
        image_collection="renamed_images",
    )
    request = _fake_request(sentinel_qdrant)
    me = _fake_admin()

    response = await admin_module.kb_reconcile(request, me)

    # Handler called reconcile_once with the qdrant from app.state AND its
    # configured collection names (not the defaults).
    assert captured["qdrant"] is sentinel_qdrant
    assert captured["text_collection"] == "renamed_text"
    assert captured["image_collection"] == "renamed_images"
    # Response is the structured dict from result.to_dict().
    assert response["total_orphans_deleted"] == 1
    assert "by_collection" in response
    assert set(response["by_collection"].keys()) == {"kb_text", "kb_images"}


@pytest.mark.asyncio
async def test_kb_reconcile_zero_orphans_still_returns_summary(monkeypatch):
    # A clean sweep (no orphans) shouldn't 404 or short-circuit — the
    # operator wants to see "I asked for a reconcile and 0 came back" as
    # a positive signal too.
    async def _fake_reconcile_once(_qdrant, *, text_collection, image_collection):
        return _FakeReconcileResult(total_orphans_deleted=0)

    from audrey.routes import admin as admin_module
    monkeypatch.setattr(admin_module, "reconcile_once", _fake_reconcile_once)

    request = _fake_request(_fake_qdrant())
    response = await admin_module.kb_reconcile(request, _fake_admin())

    assert response["total_orphans_deleted"] == 0


@pytest.mark.asyncio
async def test_kb_reconcile_logs_trigger_with_admin_email(monkeypatch, caplog):
    # Per-call audit trail: the log line names *which* admin ran the
    # sweep. If multiple admins are configured, that's the only way to
    # tell who did it after the fact.
    import logging as _logging

    async def _fake_reconcile_once(_qdrant, *, text_collection, image_collection):
        return _FakeReconcileResult(total_orphans_deleted=2)

    from audrey.routes import admin as admin_module
    monkeypatch.setattr(admin_module, "reconcile_once", _fake_reconcile_once)

    request = _fake_request(_fake_qdrant())
    me = _fake_admin(email="ops@example.com")

    with caplog.at_level(_logging.WARNING, logger="audrey.routes.admin"):
        await admin_module.kb_reconcile(request, me)

    # Exactly one warning line, and it names both the admin and the
    # orphan count.
    warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "ops@example.com" in message
    assert "orphans_deleted=2" in message


# ─── chat_archive admin endpoints: failure paths ──────────────────────
#
# These pin the audit-driven behavior change (2026-06-03): when the chat
# archive isn't reachable, the handlers raise HTTP 503 rather than return
# HTTP 200 with an {"error": ...} body. A monitoring script must be able
# to read the failure from the status code, not parse the body.


def _archive_request(archive_client, registry=None):
    """Build the request shape the chat_archive handlers read.

    They access `request.app.state.archive_client` and
    `request.app.state.tools`.
    """
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(archive_client=archive_client, tools=registry)
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["chat_archive_prune", "chat_archive_stats"])
async def test_chat_archive_503_when_client_unavailable(handler_name):
    # archive_client is None (custom-tools never wired) → 503, not 200.
    from audrey.routes import admin as admin_module

    handler = getattr(admin_module, handler_name)
    request = _archive_request(archive_client=None)

    with pytest.raises(HTTPException) as exc:
        await handler(request, _fake_admin())
    assert exc.value.status_code == 503
    assert "archive_client_unavailable" in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["chat_archive_prune", "chat_archive_stats"])
async def test_chat_archive_503_when_tool_not_registered(handler_name):
    # archive_client exists but host_url() returns None (chat_history_search
    # not in the registry, e.g. custom-tools booted late) → 503.
    from audrey.routes import admin as admin_module

    archive_client = SimpleNamespace(host_url=lambda _registry: None)
    handler = getattr(admin_module, handler_name)
    request = _archive_request(archive_client=archive_client, registry=object())

    with pytest.raises(HTTPException) as exc:
        await handler(request, _fake_admin())
    assert exc.value.status_code == 503
    assert "not_registered" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_chat_archive_stats_include_local_delivery_queue(monkeypatch):
    from audrey.routes import admin as admin_module

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"messages": 12, "chunks_reindex_pending": 0}

    class _Http:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url):
            return _Response()

    monkeypatch.setattr(admin_module.httpx, "AsyncClient", lambda **_kwargs: _Http())
    queue_stats = AsyncMock(return_value={
        "pending": 2,
        "attempts": 3,
        "last_error": "upstream unavailable",
    })
    archive_client = SimpleNamespace(
        host_url=lambda _registry: "http://custom-tools:8001",
        stats=queue_stats,
    )
    request = _archive_request(archive_client, registry=object())

    result = await admin_module.chat_archive_stats(request, _fake_admin())

    assert result["messages"] == 12
    assert result["delivery_queue"] == {
        "pending": 2,
        "attempts": 3,
        "last_error": "upstream unavailable",
    }
    queue_stats.assert_awaited_once_with()
