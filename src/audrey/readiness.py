"""One sanitized operational snapshot for admin status and Prometheus.

The collector is deliberately fail-soft: an optional backend outage becomes a
component state, never an exception that takes down ``/metrics`` or the admin
status route.  A short cache prevents Prometheus and a human refresh from
doubling the same probes while keeping failure-injection feedback prompt.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from audrey.metrics import publish_readiness
from audrey.tools.discovery import TOOL_DECLARATIONS


class ComponentReadiness(BaseModel):
    status: Literal["available", "unavailable", "disabled"]
    required: bool = False
    reason: str = ""


class CapabilityReadiness(BaseModel):
    name: str
    available: bool


class ToolFailure(BaseModel):
    name: str
    reason: str


class ToolReadiness(BaseModel):
    policy_count: int
    discovered_count: int
    available_count: int
    capabilities: list[CapabilityReadiness] = Field(default_factory=list)
    unavailable: list[ToolFailure] = Field(default_factory=list)


class QueueReadiness(BaseModel):
    available: bool = True
    depth: int = 0
    active: int = 0
    attempts: int = 0
    with_error: int = 0
    exhausted: int = 0
    oldest_age_seconds: float = 0.0


class WorkerReadiness(BaseModel):
    enabled: bool
    running: bool
    queue_depth: int = 0
    last_activity_age_seconds: float = 0.0
    last_success_age_seconds: float = 0.0
    last_failure_age_seconds: float = 0.0


class GatePressure(BaseModel):
    capacity: int
    in_use: int
    waiting: int
    waiting_users: int


class InflightPressure(BaseModel):
    max_per_user: int
    tracked_users: int
    active_users: int
    saturated_users: int
    in_use: int
    waiting: int


class PressureReadiness(BaseModel):
    gpu_gate: GatePressure
    user_inflight: InflightPressure


class ReadinessStatus(BaseModel):
    schema_version: int = 1
    status: Literal["ready", "degraded", "unready"]
    generated_at: str
    components: dict[str, ComponentReadiness]
    tools: ToolReadiness
    queues: dict[str, QueueReadiness]
    workers: dict[str, WorkerReadiness]
    pressure: PressureReadiness


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _age_seconds(value: str, *, now: dt.datetime) -> float:
    if not value:
        return 0.0
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return max(0.0, round((now - parsed).total_seconds(), 3))


class ReadinessCollector:
    """Collect and cache Audrey's bounded operational state."""

    def __init__(
        self,
        app: Any,
        *,
        required_components: set[str] | None = None,
        probe_timeout_s: float = 2.0,
        cache_ttl_s: float = 5.0,
    ) -> None:
        self._app = app
        self._required = (
            {"ollama"}
            if required_components is None
            else set(required_components)
        )
        self._probe_timeout_s = max(0.1, float(probe_timeout_s))
        self._cache_ttl_s = max(0.0, float(cache_ttl_s))
        self._lock = asyncio.Lock()
        self._cached: ReadinessStatus | None = None
        self._cached_at = 0.0

    async def collect(self, *, force: bool = False) -> ReadinessStatus:
        now_mono = time.monotonic()
        if (
            not force
            and self._cached is not None
            and now_mono - self._cached_at < self._cache_ttl_s
        ):
            return self._cached
        async with self._lock:
            now_mono = time.monotonic()
            if (
                not force
                and self._cached is not None
                and now_mono - self._cached_at < self._cache_ttl_s
            ):
                return self._cached
            snapshot = await self._collect()
            self._cached = snapshot
            self._cached_at = time.monotonic()
            publish_readiness(snapshot)
            return snapshot

    async def _collect(self) -> ReadinessStatus:
        state = self._app.state
        cfg = state.cfg
        now = _utc_now()

        ollama_ok, qdrant_ok, tools_ok = await asyncio.gather(
            self._probe_ollama(state),
            self._probe_qdrant(state),
            self._probe_tool_servers(state, cfg),
        )

        tools = self._tool_status(state, enabled=bool(cfg.tools.get("enabled", True)))
        workers = self._worker_status(state, cfg)
        remote_archive, archive_stats, work_stats = await asyncio.gather(
            self._remote_archive_status(state),
            self._archive_delivery_stats(state),
            self._work_queue_stats(state),
        )
        archive_cfg = cfg.raw.get("chat_archive", {}) or {}
        archive_enabled = bool(archive_cfg.get("enabled", True))
        queues = self._queue_status(
            remote_archive=remote_archive,
            archive_stats=archive_stats,
            work_stats=work_stats,
            now=now,
        )
        if not archive_enabled:
            for name in (
                "archive_delivery",
                "archive_indexing",
                "archive_deletions",
                "conversation_deletions",
            ):
                queues[name] = QueueReadiness()
        watcher = workers["kb_watcher"]
        reconciler = workers["kb_reconciler"]
        components = {
            "ollama": self._component("ollama", ollama_ok, "probe_failed"),
            "qdrant": self._component("qdrant", qdrant_ok, "probe_failed"),
            "custom_tools": self._component(
                "custom_tools",
                tools_ok,
                "tool_server_unreachable",
                disabled=not bool(cfg.tools.get("enabled", True)),
            ),
            "chat_archive": self._component(
                "chat_archive",
                remote_archive is not None,
                "backend_unavailable",
                disabled=not archive_enabled,
            ),
            "kb_watcher": self._component(
                "kb_watcher",
                watcher.running,
                "worker_not_running",
                disabled=not watcher.enabled,
            ),
            "kb_reconciler": self._component(
                "kb_reconciler",
                reconciler.running,
                "worker_not_running",
                disabled=not reconciler.enabled,
            ),
        }

        required_failed = any(
            component.required and component.status != "available"
            for component in components.values()
        )
        optional_failed = any(
            not component.required and component.status == "unavailable"
            for component in components.values()
        )
        queue_failed = any(
            (not queue.available) or queue.with_error > 0 or queue.exhausted > 0
            for queue in queues.values()
        )
        tool_failed = bool(tools.unavailable) or (
            bool(cfg.tools.get("enabled", True))
            and tools.discovered_count < tools.policy_count
        )
        status: Literal["ready", "degraded", "unready"]
        if required_failed:
            status = "unready"
        elif optional_failed or queue_failed or tool_failed:
            status = "degraded"
        else:
            status = "ready"

        gate_pressure = getattr(state.gate, "pressure_snapshot", lambda: {})()
        inflight_pressure = getattr(
            state.inflight,
            "pressure_snapshot",
            lambda: {},
        )()
        return ReadinessStatus(
            status=status,
            generated_at=now.isoformat(timespec="seconds"),
            components=components,
            tools=tools,
            queues=queues,
            workers=workers,
            pressure=PressureReadiness(
                gpu_gate=GatePressure.model_validate(gate_pressure),
                user_inflight=InflightPressure.model_validate(inflight_pressure),
            ),
        )

    def _component(
        self,
        name: str,
        available: bool,
        reason: str,
        *,
        disabled: bool = False,
    ) -> ComponentReadiness:
        if disabled:
            status = "disabled"
            reason = ""
        elif available:
            status = "available"
            reason = ""
        else:
            status = "unavailable"
        return ComponentReadiness(
            status=status,
            required=name in self._required,
            reason=reason,
        )

    async def _probe_ollama(self, state: Any) -> bool:
        try:
            async with asyncio.timeout(self._probe_timeout_s):
                await state.ollama.tags()
        except Exception:  # noqa: BLE001 — probe failures become state
            return False
        return True

    async def _probe_qdrant(self, state: Any) -> bool:
        probe = getattr(state.qdrant, "probe", None)
        if not callable(probe):
            return False
        try:
            async with asyncio.timeout(self._probe_timeout_s):
                await probe()
        except Exception:  # noqa: BLE001 — probe failures become state
            return False
        return True

    async def _probe_tool_servers(self, state: Any, cfg: Any) -> bool:
        if not bool(cfg.tools.get("enabled", True)):
            return False
        servers = list(cfg.tools.get("servers", []) or [])
        if not servers:
            return False
        client = state.archive_http

        async def probe(url: str) -> bool:
            try:
                response = await client.get(
                    f"{url.rstrip('/')}/health",
                    timeout=self._probe_timeout_s,
                )
                response.raise_for_status()
            except Exception:  # noqa: BLE001 — probe failures become state
                return False
            return True

        return all(await asyncio.gather(*(probe(url) for url in servers)))

    def _tool_status(self, state: Any, *, enabled: bool) -> ToolReadiness:
        registry = state.tools
        records = registry.policy_records() if enabled else []
        unavailable = [
            ToolFailure(
                name=spec.name,
                reason=spec.unavailable_reason or "unavailable",
            )
            for spec in records
            if not spec.available
        ]
        failed_capabilities: set[str] = set()
        for item in unavailable:
            prefix = "dependency_unavailable:"
            if item.reason.startswith(prefix):
                failed_capabilities.update(
                    value
                    for value in item.reason.removeprefix(prefix).split(",")
                    if value
                )
        all_capabilities = sorted({
            dependency
            for declaration in TOOL_DECLARATIONS.values()
            for dependency in declaration.dependencies
        })
        if enabled and not records:
            failed_capabilities.update(all_capabilities)
        return ToolReadiness(
            policy_count=len(TOOL_DECLARATIONS),
            discovered_count=len(records),
            available_count=sum(spec.available for spec in records),
            capabilities=[
                CapabilityReadiness(
                    name=name,
                    available=enabled and name not in failed_capabilities,
                )
                for name in all_capabilities
            ],
            unavailable=sorted(unavailable, key=lambda item: item.name),
        )

    def _worker_status(self, state: Any, cfg: Any) -> dict[str, WorkerReadiness]:
        watcher_enabled = bool(cfg.env.kb_watcher_enabled)
        watcher = getattr(state, "kb_watcher", None)
        watcher_raw = (
            watcher.snapshot()
            if watcher is not None and hasattr(watcher, "snapshot")
            else {"running": False}
        )
        reconcile_cfg = (cfg.raw.get("kb", {}) or {}).get("reconcile", {}) or {}
        reconciler_enabled = bool(reconcile_cfg.get("enabled", True))
        reconciler = getattr(state, "kb_reconciler", None)
        reconciler_raw = (
            reconciler.snapshot()
            if reconciler is not None and hasattr(reconciler, "snapshot")
            else {"running": False}
        )
        return {
            "kb_watcher": WorkerReadiness(
                enabled=watcher_enabled,
                **watcher_raw,
            ),
            "kb_reconciler": WorkerReadiness(
                enabled=reconciler_enabled,
                **reconciler_raw,
            ),
        }

    async def _remote_archive_status(self, state: Any) -> dict[str, Any] | None:
        transport = getattr(state, "archive_transport", None)
        repair_status = getattr(transport, "repair_status", None)
        if not callable(repair_status):
            return None
        try:
            async with asyncio.timeout(self._probe_timeout_s):
                value = await repair_status(registry=state.tools)
        except Exception:  # noqa: BLE001 — backend failures become state
            return None
        return value if isinstance(value, dict) else None

    async def _archive_delivery_stats(self, state: Any) -> dict[str, Any] | None:
        queue = getattr(state, "archive_queue", None)
        if queue is None:
            return None
        try:
            operational, repair = await asyncio.gather(
                queue.stats(),
                queue.repair_stats(),
            )
        except Exception:  # noqa: BLE001 — backend failures become state
            return None
        return {**operational, **repair}

    async def _work_queue_stats(self, state: Any) -> dict[str, Any] | None:
        method = getattr(state.uploads_db, "work_queue_stats", None)
        if not callable(method):
            return None
        try:
            return await method()
        except Exception:  # noqa: BLE001 — backend failures become state
            return None

    def _queue_status(
        self,
        *,
        remote_archive: dict[str, Any] | None,
        archive_stats: dict[str, Any] | None,
        work_stats: dict[str, Any] | None,
        now: dt.datetime,
    ) -> dict[str, QueueReadiness]:
        queues: dict[str, QueueReadiness] = {}
        if archive_stats is None:
            queues["archive_delivery"] = QueueReadiness(available=False)
        else:
            queues["archive_delivery"] = QueueReadiness(
                depth=int(archive_stats.get("pending", 0)),
                attempts=int(archive_stats.get("attempts", 0)),
                with_error=int(archive_stats.get("with_error", 0)),
                exhausted=int(archive_stats.get("exhausted", 0)),
                oldest_age_seconds=_age_seconds(
                    str(archive_stats.get("oldest_created_at", "")),
                    now=now,
                ),
            )
        for name, remote_name in (
            ("archive_indexing", "indexing"),
            ("archive_deletions", "deletions"),
            ("conversation_deletions", "conversation_deletions"),
        ):
            raw = remote_archive.get(remote_name) if remote_archive else None
            if not isinstance(raw, dict):
                queues[name] = QueueReadiness(available=False)
            else:
                queues[name] = QueueReadiness(
                    depth=int(raw.get("pending", 0)),
                    attempts=int(raw.get("attempts", 0)),
                    with_error=int(raw.get("with_error", 0)),
                    exhausted=int(raw.get("exhausted", 0)),
                )
        for name in ("media_processing", "media_fetch"):
            raw = work_stats.get(name) if work_stats else None
            if not isinstance(raw, dict):
                queues[name] = QueueReadiness(available=False)
            else:
                queues[name] = QueueReadiness(
                    depth=int(raw.get("pending", 0)),
                    active=int(raw.get("active", 0)),
                    oldest_age_seconds=_age_seconds(
                        str(raw.get("oldest_pending_at", "")),
                        now=now,
                    ),
                )
        return queues


__all__ = [
    "ReadinessCollector",
    "ReadinessStatus",
]
