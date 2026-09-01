"""Runtime capability state for the custom-tools sidecar.

The sidecar serves several unrelated classes of work.  A stateful backend
failure must therefore change the affected tool set, not decide whether the
process can start.  This module owns the small state machine that initializes
the durable components independently and retries them after startup.

OpenAPI publishes the resulting component states.  Audrey combines those
states with its own declarative tool dependency policy, so this module never
duplicates the tool-to-component security catalogue.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("custom-tools.capabilities")


_RUNTIME_DEFAULTS: dict[str, bool] = {
    # Stateless or lazy HTTP capabilities are usable as soon as their clients
    # are constructed.  They deliberately do not wait for Qdrant.
    "web_search": True,
    "web_fetch": True,
    "audrey_kb": True,
    "audrey_files": True,
    "image_embedding": True,
    # Stateful capabilities become available only after their own startup
    # checks succeed.
    "qdrant": False,
    "memory": False,
    "text_embedding": False,
    "chat_archive_source": False,
    "chat_archive": False,
}


@dataclass(frozen=True, slots=True)
class CapabilityState:
    available: bool
    reason: str = ""


class CapabilityRegistry:
    """In-process component availability with transition-only logging."""

    def __init__(self, initial: Mapping[str, bool] | None = None) -> None:
        source = dict(_RUNTIME_DEFAULTS if initial is None else initial)
        self._states = {
            name: CapabilityState(
                available=available,
                reason="" if available else "startup_pending",
            )
            for name, available in source.items()
        }

    @classmethod
    def all_available(cls) -> CapabilityRegistry:
        """Optimistic schema state used only before FastAPI lifespan starts."""
        return cls({name: True for name in _RUNTIME_DEFAULTS})

    def set_available(self, name: str) -> None:
        self._set(name, CapabilityState(available=True))

    def set_unavailable(self, name: str, reason: str) -> None:
        self._set(name, CapabilityState(available=False, reason=reason))

    def _set(self, name: str, state: CapabilityState) -> None:
        previous = self._states.get(name)
        self._states[name] = state
        if previous == state:
            return
        if state.available:
            log.info("capability available: %s", name)
        else:
            log.warning("capability unavailable: %s reason=%s", name, state.reason)

    def unavailable(self, names: tuple[str, ...] | list[str]) -> list[str]:
        return sorted(
            name
            for name in names
            if not self._states.get(
                name, CapabilityState(False, "not_initialized")
            ).available
        )

    def snapshot(self) -> dict[str, CapabilityState]:
        return dict(self._states)

    def openapi_status(self) -> dict[str, dict[str, str | bool]]:
        return {
            name: {
                "available": state.available,
                "reason": state.reason,
            }
            for name, state in sorted(self._states.items())
        }


class CapabilitySupervisor:
    """Initialize independent stateful components and retry failed ones.

    ``memory`` and ``archive`` are intentionally duck-typed.  Keeping this
    coordinator independent of their concrete implementations makes the
    failure transitions hermetic to test and avoids a dependency cycle.
    """

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        memory: Any,
        archive: Any,
        archive_maintainer: Any,
        retry_interval_s: float,
        probe_timeout_s: float,
    ) -> None:
        self.registry = registry
        self._memory = memory
        self._archive = archive
        self._archive_maintainer = archive_maintainer
        self._retry_interval_s = max(0.1, retry_interval_s)
        self._probe_timeout_s = max(0.1, probe_timeout_s)
        self._archive_source_ready = False
        self._archive_index_ready = False
        self._memory_ready = False
        self._embedding_ready = False
        self._maintainer_started = False
        self._task: asyncio.Task[None] | None = None
        self._refresh_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.refresh()
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name="custom-tools-capability-recovery"
            )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._maintainer_started:
            await self._archive_maintainer.stop()
            self._maintainer_started = False

    async def refresh(self) -> None:
        """Run one bounded recovery pass without coupling component failures."""
        async with self._refresh_lock:
            await self._ensure_archive_source()

            try:
                await self._memory.probe_qdrant(timeout_s=self._probe_timeout_s)
            except Exception as exc:  # noqa: BLE001 — backend clients raise a wide tree
                self.registry.set_unavailable("qdrant", "connection_failed")
                log.warning("qdrant capability probe failed: %s", exc)
                return
            self.registry.set_available("qdrant")

            if not self._memory_ready:
                try:
                    await self._memory.init_qdrant()
                except Exception as exc:  # noqa: BLE001 — isolate memory initialization
                    self.registry.set_unavailable("memory", "initialization_failed")
                    log.warning("memory capability initialization failed: %s", exc)
                else:
                    self._memory_ready = True
                    self.registry.set_available("memory")

            if self._memory_ready and not self._embedding_ready:
                if await self._memory.warm_embedder():
                    self._embedding_ready = True
                    self.registry.set_available("text_embedding")
                else:
                    self.registry.set_unavailable(
                        "text_embedding", "warmup_failed"
                    )

            if self._archive_source_ready and not self._archive_index_ready:
                try:
                    await self._archive.init_index()
                except Exception as exc:  # noqa: BLE001 — isolate archive index startup
                    self.registry.set_unavailable(
                        "chat_archive", "index_initialization_failed"
                    )
                    log.warning("chat archive index initialization failed: %s", exc)
                else:
                    self._archive_index_ready = True
                    self.registry.set_available("chat_archive")

    async def _ensure_archive_source(self) -> None:
        if self._archive_source_ready:
            return
        try:
            await self._archive.init_source()
        except Exception as exc:  # noqa: BLE001 — SQLite failure must not kill web tools
            self.registry.set_unavailable(
                "chat_archive_source", "initialization_failed"
            )
            self.registry.set_unavailable(
                "chat_archive", "source_initialization_failed"
            )
            log.warning("chat archive source initialization failed: %s", exc)
            return
        self._archive_source_ready = True
        self.registry.set_available("chat_archive_source")
        if not self._maintainer_started:
            await self._archive_maintainer.start()
            self._maintainer_started = True

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._retry_interval_s)
                try:
                    await self.refresh()
                except Exception as exc:  # noqa: BLE001 — supervisor must survive a pass
                    log.warning("capability recovery pass failed: %s", exc)
        except asyncio.CancelledError:
            return


__all__ = [
    "CapabilityRegistry",
    "CapabilityState",
    "CapabilitySupervisor",
]
