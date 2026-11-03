"""Installed Agent output checks with isolated data and mocked local replies.

Real renderers, upload quota/catalog, Agent leases and authenticated ASGI file
downloads run against temporary fixtures. No live account, network or model is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class SmokeSetupError(Exception):
    pass


class SmokeFailureError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise SmokeFailureError(message)


def load_runtime():
    try:
        import httpx
        from fastapi import FastAPI

        from audrey import agent_files
        from audrey.agent_dispatch import AgentDispatchStoppedError
        from audrey.agent_executor import ResearchAgentExecutor
        from audrey.agent_runtime import AgentRuntime
        from audrey.app_state import ApplicationStore
        from audrey.auth import require_principal
        from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
        from audrey.kb.storage_lifecycle import StorageLifecycle
        from audrey.kb.uploads_db import UploadsDB
        from audrey.models.health import HealthTracker
        from audrey.models.ollama import OllamaClient
        from audrey.pipeline.fair_gate import FairLocalGate
        from audrey.routes.app import router
        from audrey.routes.app.agents import AgentTaskResponse
        from audrey.routes.inflight import UserInflightRegistry
        from audrey.tools.discovery import ToolRegistry
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks Agent files; rebuild audrey.") from error
    if "outputs" not in AgentTaskResponse.model_fields or not hasattr(agent_files, "create_agent_file"):
        raise SmokeSetupError("Installed backend lacks Agent files; rebuild audrey.")
    return SimpleNamespace(
        httpx=httpx, FastAPI=FastAPI, agent_files=agent_files,
        AgentDispatchStoppedError=AgentDispatchStoppedError,
        ResearchAgentExecutor=ResearchAgentExecutor, AgentRuntime=AgentRuntime,
        ApplicationStore=ApplicationStore, require_principal=require_principal,
        FileDeletionWorker=FileDeletionWorker, FileOperationLocks=FileOperationLocks,
        StorageLifecycle=StorageLifecycle, UploadsDB=UploadsDB,
        HealthTracker=HealthTracker, OllamaClient=OllamaClient,
        FairLocalGate=FairLocalGate, UserInflightRegistry=UserInflightRegistry,
        ToolRegistry=ToolRegistry, router=router,
    )


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


class Scenario:
    def __init__(self, runtime, root):
        self.r, self.root = runtime, root
        self.store = runtime.ApplicationStore(root / "isolated-app.sqlite")
        self.db = runtime.UploadsDB(root / "isolated-uploads.sqlite")
        self.client = None
        self.counter = 0

    async def __aenter__(self):
        try:
            r = self.r
            self.owner, self.other = [await self.store.resolve_external_identity(
                provider="agent-file-smoke", subject=name, email=f"{name}@example.com",
                display_name=name, role="user", auth_method="test",
            ) for name in ("alice", "bob")]
            self.current = self.owner
            self.project = await self.store.projects.create(user_id=self.owner.user_id, name="Isolated task outputs")
            self.app = r.FastAPI()
            self.app.state.application_store = self.store
            self.app.state.uploads_db = self.db
            self.app.state.storage_lifecycle = r.StorageLifecycle(self.db)
            self.app.state.file_operation_locks = r.FileOperationLocks()
            self.app.state.cfg = SimpleNamespace(
                raw={"passthrough": {"enabled": False}, "kb": {
                    "upload_root": str(self.root / "files"), "max_user_bytes": 1_000_000,
                }}, env=SimpleNamespace(ollama_host="http://mock-local.invalid"),
            )
            self.app.state.file_deletions = r.FileDeletionWorker(
                db=self.db, qdrant=AsyncMock(), upload_root=self.root / "files",
                locks=self.app.state.file_operation_locks,
            )
            self.app.state.tools = r.ToolRegistry()
            self.app.state.health = r.HealthTracker()
            self.app.state.gate = r.FairLocalGate()
            self.app.state.inflight = r.UserInflightRegistry(max_inflight_per_user=1)
            self.app.include_router(r.router)
            self.app.dependency_overrides[r.require_principal] = lambda: self.current
            self.client = r.httpx.AsyncClient(transport=r.httpx.ASGITransport(app=self.app), base_url="http://isolated")
            return self
        except BaseException:
            await self.__aexit__()
            raise

    async def __aexit__(self, *_error):
        try:
            if self.client:
                await self.client.aclose()
        finally:
            self.db.close()
            self.store.close()

    async def new_task(self):
        self.counter += 1
        return await self.store.agents.create(
            user_id=self.owner.user_id, objective="Save the requested checklist as a Markdown file.",
            request_id=f"file-smoke-{self.counter}", project_id=self.project.project_id,
        )

    async def work_task(self):
        task = await self.new_task()
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The isolated Agent could not be claimed.")
        await self.store.agents.reserve_operation(**lease(claimed), step_id="plan")
        claimed = await self.store.agents.checkpoint(**lease(claimed), step_id="plan", result="Save the requested file.")
        await self.store.agents.reserve_operation(**lease(claimed), step_id="research")
        return claimed

    async def create(self, task, *, filename="notes.md", content="# Checklist\n\n- Prepare the work.", **changes):
        arguments = {"filename": filename, "format": "md", "content": content, **changes}
        return await self.r.agent_files.create_agent_file(self.app, self.owner, task, arguments)

    async def saved(self, task):
        return await self.store.agents.get(user_id=self.owner.user_id, task_id=task["id"])

    async def cancel(self, task):
        current = await self.saved(task)
        return await self.store.agents.control(user_id=self.owner.user_id, task_id=task["id"],
            expected_revision=current["revision"], action="cancel")

    async def complete(self, task):
        await self.store.agents.checkpoint(**lease(task), step_id="research", result="The file is ready.")
        await self.store.agents.reserve_operation(**lease(task), step_id="review")
        await self.store.agents.checkpoint(**lease(task), step_id="review", result="Finished task; file link intentionally omitted.")
        return await self.store.agents.complete(**lease(task), result="Finished task; file link intentionally omitted.")


async def local_agent_creates_owned_download(s):
    r, requests, clients = s.r, [], []
    content = "# Task checklist\n\n- Prepare the requested work."
    replies = ["Task File Creation", "Create the requested file.", None, "The checklist file is ready.", "The task is finished."]

    async def provider(request):
        if request.url.path == "/api/tags":
            return r.httpx.Response(200, json={"models": [{"name": "qwen3.8:latest", "model": "qwen3.8:latest", "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return r.httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {"general.architecture": "qwen3", "general.parameter_count": 27000000000}})
        require(request.url.path == "/api/chat" and len(requests) < len(replies), "An unexpected model operation was dispatched.")
        payload = json.loads(request.content)
        require(payload["model"] == "qwen3.8:latest", "The Agent did not use verified local Qwen.")
        index = len(requests)
        requests.append(payload)
        message = {"content": replies[index] or ""}
        if index == 2:
            require("create_file" in {tool["function"]["name"] for tool in payload.get("tools", [])}, "Work did not offer native file creation.")
            message["tool_calls"] = [{"function": {"name": "create_file", "arguments": {"filename": "checklist.md", "format": "md", "content": content}}}]
        return r.httpx.Response(200, json={"done": True, "done_reason": "stop", "eval_count": 7, "message": message})

    async def forbidden(_request):
        raise SmokeFailureError("A native file tool attempted a network request.")

    original = r.httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs.setdefault("transport", r.httpx.MockTransport(forbidden))
        return original(*args, **kwargs)

    def factory(endpoint, **kwargs):
        value = r.OllamaClient(endpoint, transport=r.httpx.MockTransport(provider), **kwargs)
        clients.append(value)
        return value

    task = await s.new_task()
    with patch.object(r.httpx, "AsyncClient", client):
        require(await r.AgentRuntime(s.store.agents, r.ResearchAgentExecutor(s.app, client_factory=factory)).run_once(), "The isolated Agent did not run.")
    saved = await s.saved(task)
    require(saved["status"] == "completed" and len(saved.get("outputs", [])) == 1, "The Agent did not save its created output.")
    output = saved["outputs"][0]
    require(output["filename"] == "checklist.md" and output["download_url"] == f"/api/files/{output['file_id']}/download", "The task returned a noncanonical file link.")
    downloaded = await s.client.get(output["download_url"])
    require(downloaded.status_code == 200 and downloaded.text == content
            and downloaded.headers.get("cache-control") == "private, no-store", "The owned download did not return the exact saved bytes privately.")
    require(output["download_url"] in saved["result"] and output["download_url"] in saved["steps"][1]["result"], "Saved Work or final output lost the server-approved download link.")
    require(len(requests) == 5 and all(value._client.is_closed for value in clients), "A task repeated model work or leaked its local client.")
    require(not s.app.state.gate.pressure_snapshot()["in_use"] and not s.app.state.inflight.pressure_snapshot()["in_use"], "File creation retained model admission resources.")
    s.current = s.other
    require((await s.client.get(output["download_url"])).status_code == 404, "Another account downloaded the Agent output.")
    require((await s.client.get(f"/api/agents/{task['id']}")).status_code == 404, "Another account read task output metadata.")
    s.current = s.owner
    files = await s.store.projects.list_files(user_id=s.owner.user_id, project_id=s.project.project_id)
    require(output["file_id"] in {item.file_id for item in files}, "The file was not associated with its owned Project.")


async def retry_restart_and_immutable_versions(s):
    task = await s.work_task()
    first = await s.create(task)
    require(not first.get("error"), "The initial output could not be created.")
    repeat = await s.create(task)
    require(first["file_id"] == repeat.get("file_id"), "Identical output retry created a duplicate file.")
    path = s.store.path
    s.store.close()
    s.store = s.r.ApplicationStore(path)
    s.app.state.application_store = s.store
    resumed = await s.create(task)
    require(resumed.get("file_id") == first["file_id"], "Restart lost the durable output identity.")
    changed = await s.create(task, content="Changed requested output.")
    require(changed.get("file_id") and changed["file_id"] != first["file_id"], "Changed content replaced an immutable output.")
    require((await s.client.get(first["download_url"])).text == "# Checklist\n\n- Prepare the work.", "A newer output overwrote saved bytes.")
    require(len((await s.saved(task)).get("outputs", [])) == 2, "Retry/restart output receipts were lost or duplicated.")
    completed = await s.complete(task)
    for output in completed["outputs"]:
        require(output["download_url"] in completed["result"], "Final task completion dropped a saved output link.")


async def deleted_output_stays_deleted(s):
    task = await s.work_task()
    output = await s.create(task)
    require(not output.get("error"), "The deletion fixture output could not be created.")
    require((await s.client.delete(f"/api/files/{output['file_id']}")).status_code == 200, "The owned file could not be deleted.")
    require((await s.client.get(output["download_url"])).status_code == 404, "Deleted output bytes remained downloadable.")
    repeat = await s.create(task)
    require(bool(repeat.get("error")) and not await s.db.list_user(s.owner.storage_namespace), "Retry resurrected a deleted output.")


async def revoked_scope_blocks_publication(s):
    task = await s.work_task()
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = s.r.agent_files.render_file

    def rendering(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(3):
            raise SmokeFailureError("The rendering fixture was not released.")
        return original(*args, **kwargs)

    operation = None
    with patch.object(s.r.agent_files, "render_file", rendering):
        try:
            operation = asyncio.create_task(s.create(task))
            await asyncio.wait_for(entered.wait(), 2)
            await s.cancel(task)
            release.set()
            try:
                result = await operation
            except (s.r.AgentDispatchStoppedError, ValueError):
                result = {"error": "revoked"}
            require(bool(result.get("error")), "A cancelled task reported a successful file creation.")
        finally:
            release.set()
            if operation and not operation.done():
                operation.cancel()
            if operation:
                await asyncio.gather(operation, return_exceptions=True)
    require(not await s.db.list_user(s.owner.storage_namespace), "Cancelled work published a late upload.")
    require(not (await s.saved(task)).get("outputs"), "Cancelled work attached a late file receipt.")
    require((await s.db.quota_usage(s.owner.storage_namespace)).single_shot_bytes == 0, "Cancelled work retained a storage reservation.")


async def quota_and_lifetime_limits(s):
    task = await s.work_task()
    s.app.state.cfg.raw["kb"]["max_user_bytes"] = 1
    denied = await s.create(task)
    require(bool(denied.get("error")) and not await s.db.list_user(s.owner.storage_namespace), "A quota refusal published output bytes.")
    require((await s.db.quota_usage(s.owner.storage_namespace)).single_shot_bytes == 0, "A quota refusal leaked reserved storage.")
    await s.cancel(task)
    s.app.state.cfg.raw["kb"]["max_user_bytes"] = 1_000_000
    task = await s.work_task()
    for index in range(8):
        result = await s.create(task, filename=f"output-{index}.md", content=f"Working output {index}.")
        require(not result.get("error"), "The supported eight-output allowance was refused early.")
    ninth = await s.create(task, filename="output-9.md")
    require(bool(ninth.get("error")) and len((await s.saved(task)).get("outputs", [])) == 8, "The task exceeded its lifetime file allowance.")


async def task_deletion_retains_owned_files(s):
    task = await s.work_task()
    output = await s.create(task)
    require(not output.get("error"), "The retained-file fixture could not be created.")
    cancelled = await s.cancel(task)
    require(await s.store.agents.delete(user_id=s.owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"]), "Terminal task history could not be deleted.")
    require((await s.client.get(output["download_url"])).status_code == 200, "Deleting task history deleted the separately saved My Files output.")


CASES = (
    ("local_agent_creates_owned_download", local_agent_creates_owned_download),
    ("retry_restart_and_immutable_versions", retry_restart_and_immutable_versions),
    ("deleted_output_stays_deleted", deleted_output_stays_deleted),
    ("revoked_scope_blocks_publication", revoked_scope_blocks_publication),
    ("quota_and_lifetime_limits", quota_and_lifetime_limits),
    ("task_deletion_retains_owned_files", task_deletion_retains_owned_files),
)


async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-files-smoke-") as directory:
        for name, case in CASES:
            try:
                async with asyncio.timeout(timeout_seconds), Scenario(runtime, Path(directory) / name) as scenario:
                    await case(scenario)
            except TimeoutError:
                raise SmokeFailureError(f"{name} deadline exceeded.") from None
            checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable output fixtures were not removed.")
    return [*checks, "isolated_fixture_cleanup"]


def _case_process(runtime, case, directory, sender, timeout_seconds):
    """Keep every fixture connection and worker thread inside the child."""
    async def execute():
        async with asyncio.timeout(timeout_seconds), Scenario(runtime, Path(directory)) as scenario:
            await case(scenario)

    try:
        asyncio.run(execute())
        result = (0, "")
    except TimeoutError:
        result = (1, "deadline exceeded")
    except SmokeFailureError as error:
        result = (1, str(error))
    except SmokeSetupError as error:
        result = (2, str(error))
    except (KeyboardInterrupt, asyncio.CancelledError):
        result = (130, "Agent file smoke interrupted.")
    except BaseException:  # noqa: BLE001 - do not expose SQL, paths or private content
        result = (3, "Agent file smoke harness crashed.")
    try:
        sender.send(result)
    finally:
        sender.close()


def _stop_case(process):
    """A blocked event loop or executor cannot delay parent-owned cleanup."""
    if process.is_alive():
        process.terminate()
        process.join(0.25)
    if process.is_alive():
        process.kill()
        process.join(1.0)
    if process.is_alive():
        raise RuntimeError("The isolated case could not be stopped.")


def run_processes(runtime, timeout_seconds):
    """Watch Linux forked cases from a parent that owns no fixture resources.

    asyncio deadlines cannot interrupt synchronous lock waits and can hang
    while shutting down executor threads. Fork also supports the installed
    stdin runner: loaded runtime modules and case functions need no pickling.
    """
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        raise SmokeSetupError("Agent file smoke requires Linux fork support.") from None
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-files-smoke-") as directory:
        for name, case in CASES:
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_case_process,
                args=(runtime, case, str(Path(directory) / name), sender, timeout_seconds),
                daemon=True,
            )
            started = False
            try:
                process.start()
                started = True
                sender.close()
                process.join(timeout_seconds)
                if process.is_alive():
                    _stop_case(process)
                    raise SmokeFailureError(f"{name} deadline exceeded.")
                if process.exitcode != 0 or not receiver.poll():
                    raise RuntimeError("The isolated case did not return a result.")
                code, message = receiver.recv()
                if code == 1:
                    raise SmokeFailureError(f"{name} deadline exceeded." if message == "deadline exceeded" else message)
                if code == 2:
                    raise SmokeSetupError(message)
                if code == 130:
                    raise KeyboardInterrupt
                if code != 0:
                    raise RuntimeError("The isolated case failed.")
                checks.append(name)
            finally:
                if started:
                    _stop_case(process)
                receiver.close()
                sender.close()
                process.close()
    require(not Path(directory).exists(), "Disposable output fixtures were not removed.")
    return [*checks, "isolated_fixture_cleanup"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks = 0, []
    message = "Agent file smoke passed; isolated temporary data and mocked local replies; no live accounts/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = run_processes(load_runtime(), args.timeout_seconds)
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Agent file smoke interrupted."
    except Exception:  # noqa: BLE001 - private paths, SQL and content never enter terminal output
        code, message = 3, "Agent file smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
