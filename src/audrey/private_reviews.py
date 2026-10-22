"""Native Private review admission, controls and replay of committed results."""
from __future__ import annotations

import logging
from typing import Any

from audrey.app_state import (
    ModelUsageSnapshot,
    RunAlreadyTerminalError,
    SourceSnapshot,
    ToolCallSnapshot,
)
from audrey.app_state.private_tasks import (
    TERMINAL,
    PrivateTaskConflictError,
)
from audrey.private_task_admission import PrivateTaskAdmission

log = logging.getLogger(__name__)


async def replay_private_deliveries(store: Any, *, task_id: str | None = None) -> int:
    """Finish the original native message once, including after a process crash.

    This trusted startup operation reads already committed results. It never
    authenticates a caller, generates content or opens source documents.
    """
    count = 0
    for task in await store.private_tasks.pending_deliveries():
        if task_id is not None and task["id"] != task_id:
            continue
        delivery = task["delivery"]
        payload = dict(delivery["payload"])
        payload["sources"] = tuple(SourceSnapshot(**item) for item in payload["sources"])
        payload["tool_calls"] = tuple(ToolCallSnapshot(**item) for item in payload["tool_calls"])
        payload["models"] = tuple(ModelUsageSnapshot(**item) for item in payload["models"])
        try:
            await store.conversations.finish_run(
                user_id=task["user_id"], run_id=delivery["run_id"], **payload,
            )
        except RunAlreadyTerminalError:
            # A lost delivery acknowledgement must not create a second message
            # or replace a cancellation already committed by the native owner.
            pass
        await store.private_tasks.ack_delivery(
            user_id=task["user_id"], task_id=task["id"], run_id=delivery["run_id"],
            checkpoint_sequence=delivery["checkpoint_sequence"],
        )
        count += 1
    return count


def public_private_task(task: dict | None) -> dict | None:
    if task is None:
        return None
    return {key: task.get(key) for key in (
        "id", "conversation_id", "revision", "status", "stop_reason", "progress",
        "updated_at", "native_run_id", "usage", "limits",
    )}


class NativePrivateReviews:
    """Use the existing native run scheduler for one sequential saved review."""

    def __init__(self, app: Any, manager: Any):
        self.app, self.manager = app, manager
        self.store = manager._store

    @property
    def admission(self) -> PrivateTaskAdmission:
        # Native run tests and startup can construct the manager before the
        # application has published its complete state. Only review admission
        # needs the prepared application services.
        return PrivateTaskAdmission(self.app)

    async def latest(self, user_id: str, conversation_id: str) -> dict | None:
        return await self.store.private_tasks.latest_for_conversation(
            user_id=user_id, conversation_id=conversation_id,
        )

    async def ensure_available(self, user_id: str, conversation_id: str) -> None:
        task = await self.latest(user_id, conversation_id)
        if task and (task.get("delivery") or {}).get("pending"):
            await replay_private_deliveries(self.store, task_id=task["id"])
        if task and task["status"] in {"queued", "running"}:
            raise PrivateTaskConflictError("A saved review is already working in this conversation.")

    async def admit(self, *, user_id: str, conversation_id: str,
                    expected_case_revision: int, run_id: str) -> dict:
        task = await self.latest(user_id, conversation_id)
        if task and task["status"] not in TERMINAL:
            task = await self.admission.resume(
                user_id=user_id, task_id=task["id"], expected_revision=task["revision"],
                expected_case_revision=expected_case_revision,
            )
        else:
            task = await self.admission.create(
                user_id=user_id, conversation_id=conversation_id,
                expected_case_revision=expected_case_revision,
            )
        try:
            return await self.store.private_tasks.bind_native_run(
                user_id=user_id, task_id=task["id"], expected_revision=task["revision"], run_id=run_id,
            )
        except BaseException:
            await self.store.private_tasks.stop_unclaimed(
                user_id=user_id, task_id=task["id"], expected_revision=task["revision"],
            )
            raise

    async def control(self, *, user_id: str, task_id: str,
                      expected_revision: int, action: str) -> dict | None:
        # Pause/cancel address this task instance, not a particular progress
        # tick. Retry a revision race without changing the owner's stop intent.
        for _ in range(4):
            task = await self.store.private_tasks.get(user_id=user_id, task_id=task_id)
            if task is None:
                return None
            if expected_revision > task["revision"]:
                raise PrivateTaskConflictError("Private task changed; reload it.")
            if task["status"] in TERMINAL:
                await replay_private_deliveries(self.store, task_id=task_id)
                return task
            try:
                task = await self.store.private_tasks.control(
                    user_id=user_id, task_id=task_id, expected_revision=task["revision"], action=action,
                )
                break
            except PrivateTaskConflictError:
                continue
        else:
            raise PrivateTaskConflictError("Private task changed; reload it.")
        if task.get("native_run_id"):
            await self.manager.interrupt_private_run(
                user_id=user_id, run_id=task["native_run_id"],
                error_code="private_task_paused" if action == "pause" else "cancelled_by_user",
            )
        return task
