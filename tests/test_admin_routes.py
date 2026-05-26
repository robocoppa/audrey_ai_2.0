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

import pytest

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


def _fake_admin(email: str = "admin@example.com"):
    """Build the minimal `AuthedUser` shape the handler reads."""
    # Handler only reads `me.email` for the log line.
    return SimpleNamespace(email=email, role="admin", owui_id="abc123")


@pytest.mark.asyncio
async def test_kb_reconcile_returns_structured_summary(monkeypatch):
    # The handler should call `reconcile_once(qdrant)`, log the orphan
    # count + the admin's email, and return `result.to_dict()` to the
    # client. We pin all three.
    captured_args: list[object] = []

    async def _fake_reconcile_once(qdrant):
        captured_args.append(qdrant)
        return _FakeReconcileResult(total_orphans_deleted=1)

    from audrey.routes import admin as admin_module
    monkeypatch.setattr(admin_module, "reconcile_once", _fake_reconcile_once)

    sentinel_qdrant = object()  # handler shouldn't introspect — pass-through
    request = _fake_request(sentinel_qdrant)
    me = _fake_admin()

    response = await admin_module.kb_reconcile(request, me)

    # Handler called reconcile_once with the qdrant from app.state.
    assert captured_args == [sentinel_qdrant]
    # Response is the structured dict from result.to_dict().
    assert response["total_orphans_deleted"] == 1
    assert "by_collection" in response
    assert set(response["by_collection"].keys()) == {"kb_text", "kb_images"}


@pytest.mark.asyncio
async def test_kb_reconcile_zero_orphans_still_returns_summary(monkeypatch):
    # A clean sweep (no orphans) shouldn't 404 or short-circuit — the
    # operator wants to see "I asked for a reconcile and 0 came back" as
    # a positive signal too.
    async def _fake_reconcile_once(_qdrant):
        return _FakeReconcileResult(total_orphans_deleted=0)

    from audrey.routes import admin as admin_module
    monkeypatch.setattr(admin_module, "reconcile_once", _fake_reconcile_once)

    request = _fake_request(object())
    response = await admin_module.kb_reconcile(request, _fake_admin())

    assert response["total_orphans_deleted"] == 0


@pytest.mark.asyncio
async def test_kb_reconcile_logs_trigger_with_admin_email(monkeypatch, caplog):
    # Per-call audit trail: the log line names *which* admin ran the
    # sweep. If multiple admins are configured, that's the only way to
    # tell who did it after the fact.
    import logging as _logging

    async def _fake_reconcile_once(_qdrant):
        return _FakeReconcileResult(total_orphans_deleted=2)

    from audrey.routes import admin as admin_module
    monkeypatch.setattr(admin_module, "reconcile_once", _fake_reconcile_once)

    request = _fake_request(object())
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
