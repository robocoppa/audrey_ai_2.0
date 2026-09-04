"""Operational readiness uses one sanitized snapshot for JSON and metrics."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Response

from audrey.config import _validate_readiness
from audrey.kb.uploads_db import UploadsDB
from audrey.metrics import (
    readiness_component_available,
    readiness_queue_depth,
    readiness_state,
)
from audrey.readiness import ReadinessCollector
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec


class _HTTPResponse:
    def raise_for_status(self) -> None:
        return None


def _registry(*, unavailable_dependency: str = "") -> ToolRegistry:
    registry = ToolRegistry()
    for name, declaration in TOOL_DECLARATIONS.items():
        unavailable = unavailable_dependency in declaration.dependencies
        registry.by_name[name] = ToolSpec(
            name=name,
            description=name,
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            server_url="http://custom-tools:8001",
            path=f"/{name}",
            visibility=declaration.visibility,
            user_scope=declaration.user_scope,
            dependencies=declaration.dependencies,
            purge_gated=declaration.purge_gated,
            available=not unavailable,
            unavailable_reason=(
                f"dependency_unavailable:{unavailable_dependency}"
                if unavailable
                else None
            ),
        )
    return registry


def _app(
    *,
    qdrant_error: Exception | None = None,
    unavailable_dependency: str = "",
) -> SimpleNamespace:
    async def qdrant_probe() -> None:
        if qdrant_error is not None:
            raise qdrant_error

    repair_counts = {
        "pending": 0,
        "attempts": 0,
        "with_error": 0,
        "exhausted": 0,
        "completed": 0,
    }
    cfg = SimpleNamespace(
        tools={"enabled": True, "servers": ["http://custom-tools:8001"]},
        raw={
            "chat_archive": {"enabled": True},
            "kb": {"reconcile": {"enabled": True}},
        },
        env=SimpleNamespace(kb_watcher_enabled=False),
    )
    state = SimpleNamespace(
        cfg=cfg,
        ollama=SimpleNamespace(tags=AsyncMock(return_value=[])),
        qdrant=SimpleNamespace(probe=qdrant_probe),
        archive_http=SimpleNamespace(get=AsyncMock(return_value=_HTTPResponse())),
        tools=_registry(unavailable_dependency=unavailable_dependency),
        archive_transport=SimpleNamespace(
            repair_status=AsyncMock(
                return_value={
                    "indexing": dict(repair_counts),
                    "deletions": dict(repair_counts),
                    "conversation_deletions": dict(repair_counts),
                }
            )
        ),
        archive_queue=SimpleNamespace(
            stats=AsyncMock(
                return_value={
                    "pending": 2,
                    "attempts": 3,
                    "oldest_created_at": (
                        dt.datetime.now(dt.UTC) - dt.timedelta(seconds=20)
                    ).isoformat(),
                }
            ),
            repair_stats=AsyncMock(return_value={**repair_counts, "pending": 2}),
        ),
        uploads_db=SimpleNamespace(
            work_queue_stats=AsyncMock(
                return_value={
                    "media_processing": {
                        "pending": 1,
                        "active": 2,
                        "oldest_pending_at": (
                            dt.datetime.now(dt.UTC) - dt.timedelta(seconds=40)
                        ).isoformat(),
                    },
                    "media_fetch": {
                        "pending": 0,
                        "active": 1,
                        "oldest_pending_at": "",
                    },
                }
            )
        ),
        kb_watcher=None,
        kb_reconciler=SimpleNamespace(
            snapshot=lambda: {
                "running": True,
                "queue_depth": 0,
                "last_activity_age_seconds": 1,
                "last_success_age_seconds": 1,
                "last_failure_age_seconds": 0,
            }
        ),
        gate=SimpleNamespace(
            pressure_snapshot=lambda: {
                "capacity": 1,
                "in_use": 1,
                "waiting": 2,
                "waiting_users": 2,
            }
        ),
        inflight=SimpleNamespace(
            pressure_snapshot=lambda: {
                "max_per_user": 3,
                "tracked_users": 4,
                "active_users": 2,
                "saturated_users": 1,
                "in_use": 4,
                "waiting": 1,
            }
        ),
    )
    return SimpleNamespace(state=state)


@pytest.mark.asyncio
async def test_ready_snapshot_covers_tools_queues_workers_and_pressure():
    collector = ReadinessCollector(_app(), cache_ttl_s=0)

    result = await collector.collect(force=True)

    assert result.status == "ready"
    assert result.components["ollama"].required is True
    assert result.components["kb_watcher"].status == "disabled"
    assert result.tools.policy_count == len(TOOL_DECLARATIONS)
    assert result.tools.discovered_count == len(TOOL_DECLARATIONS)
    assert result.tools.available_count == len(TOOL_DECLARATIONS)
    assert result.queues["archive_delivery"].depth == 2
    assert result.queues["archive_delivery"].oldest_age_seconds >= 19
    assert result.queues["media_processing"].active == 2
    assert result.pressure.gpu_gate.waiting == 2
    assert result.pressure.user_inflight.saturated_users == 1
    assert "example.com" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_archive_readiness_combines_projection_and_delivery_oldest_age():
    app = _app()
    app.state.archive_queue.repair_stats.return_value = {
        "pending": 2,
        "attempts": 3,
        "with_error": 0,
        "exhausted": 0,
        "completed": 0,
    }
    app.state.archive_projector = SimpleNamespace(
        repair_stats=AsyncMock(
            return_value={
                "pending": 1,
                "attempts": 2,
                "with_error": 1,
                "exhausted": 0,
                "completed": 3,
                "oldest_created_at": (
                    dt.datetime.now(dt.UTC) - dt.timedelta(seconds=40)
                ).isoformat(),
            }
        )
    )

    result = await ReadinessCollector(app, cache_ttl_s=0).collect(force=True)

    archive = result.queues["archive_delivery"]
    assert archive.depth == 3
    assert archive.attempts == 5
    assert archive.with_error == 1
    assert archive.oldest_age_seconds >= 39


@pytest.mark.asyncio
async def test_optional_qdrant_failure_degrades_but_does_not_make_unready():
    collector = ReadinessCollector(
        _app(
            qdrant_error=ConnectionError("private qdrant host"),
            unavailable_dependency="qdrant",
        ),
        cache_ttl_s=0,
    )

    result = await collector.collect(force=True)

    assert result.status == "degraded"
    assert result.components["qdrant"].status == "unavailable"
    assert result.components["qdrant"].required is False
    qdrant_capability = next(
        item for item in result.tools.capabilities if item.name == "qdrant"
    )
    assert qdrant_capability.available is False
    assert result.tools.available_count < result.tools.discovered_count
    assert "private qdrant host" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_required_qdrant_failure_makes_snapshot_unready():
    collector = ReadinessCollector(
        _app(qdrant_error=ConnectionError("down")),
        required_components={"ollama", "qdrant"},
        cache_ttl_s=0,
    )

    result = await collector.collect(force=True)

    assert result.status == "unready"
    assert result.components["qdrant"].required is True


@pytest.mark.asyncio
async def test_metrics_mirror_the_same_snapshot_values():
    collector = ReadinessCollector(
        _app(qdrant_error=ConnectionError("down")),
        cache_ttl_s=0,
    )

    result = await collector.collect(force=True)

    assert readiness_state.labels(state=result.status)._value.get() == 1
    assert readiness_component_available.labels(component="qdrant")._value.get() == 0
    assert readiness_queue_depth.labels(queue="archive_delivery")._value.get() == 2


@pytest.mark.asyncio
async def test_admin_readiness_sets_503_only_for_unready_snapshot():
    from audrey.routes.admin import readiness_status

    snapshot = await ReadinessCollector(
        _app(qdrant_error=ConnectionError("down")),
        required_components={"qdrant"},
        cache_ttl_s=0,
    ).collect(force=True)
    collector = SimpleNamespace(collect=AsyncMock(return_value=snapshot))
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(readiness=collector))
    )
    response = Response()

    result = await readiness_status(request, response, SimpleNamespace())

    assert result.status == "unready"
    assert response.status_code == 503
    collector.collect.assert_awaited_once_with(force=True)


def test_admin_readiness_route_requires_admin():
    from audrey.auth import require_admin
    from audrey.routes import admin as admin_module

    route = next(
        route
        for route in admin_module.router.routes
        if route.path == "/v1/admin/readiness"
    )
    assert require_admin in {dependency.call for dependency in route.dependant.dependencies}


@pytest.mark.asyncio
async def test_upload_work_queue_stats_are_aggregate_and_stage_specific(tmp_path):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    try:
        await db.record_upload(
            file_id="processing-wait",
            user="alice@example.com",
            filename="one.mp4",
            mime="video/mp4",
            bytes_=1,
            kind="video",
            collection="",
            chunks=0,
            uploaded_at="2026-01-01T00:00:00+00:00",
            status="pending",
        )
        await db.record_upload(
            file_id="fetch-wait",
            user="bob@example.com",
            filename="two.mp4",
            mime="video/mp4",
            bytes_=0,
            kind="video",
            collection="",
            chunks=0,
            uploaded_at="2026-02-01T00:00:00+00:00",
            status="fetch_pending",
        )

        result = await db.work_queue_stats()

        assert result == {
            "media_processing": {
                "pending": 1,
                "active": 0,
                "oldest_pending_at": "2026-01-01T00:00:00+00:00",
            },
            "media_fetch": {
                "pending": 1,
                "active": 0,
                "oldest_pending_at": "2026-02-01T00:00:00+00:00",
            },
        }
        assert "alice" not in str(result)
        assert "bob" not in str(result)
    finally:
        db.close()


@pytest.mark.parametrize(
    "config, message",
    [
        ({"readiness": {"required_components": ["ghost"]}}, "unknown ghost"),
        ({"readiness": {"probe_timeout_s": 0}}, "positive number"),
        ({"readiness": {"cache_ttl_s": -1}}, "non-negative number"),
    ],
)
def test_readiness_config_rejects_silent_policy_errors(config, message):
    with pytest.raises(ValueError, match=message):
        _validate_readiness(config)


def test_readiness_config_accepts_explicit_empty_required_set():
    _validate_readiness({"readiness": {"required_components": []}})
