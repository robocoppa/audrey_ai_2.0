"""Installed paged selected Project file reads using disposable canonical state.

Real saved text, upload catalog, Project membership, Agent leases and authenticated
ASGI routes run against temporary fixtures. Local replies are mocked; no live
accounts, network, original upload, search index or model is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import sys
import tempfile
import uuid
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

        from audrey.agent_dispatch import AgentDispatchStoppedError
        from audrey.agent_executor import ResearchAgentExecutor
        from audrey.agent_file_inputs import AgentFileInputs, get_agent_file_inputs
        from audrey.agent_runtime import AgentRuntime
        from audrey.app_state import ApplicationStore
        from audrey.app_state.agent_projects import token_count
        from audrey.app_state.agents import AgentTaskContextError, AgentTaskError
        from audrey.auth import require_principal
        from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
        from audrey.kb.storage_lifecycle import StorageLifecycle
        from audrey.kb.uploads_db import UploadsDB
        from audrey.kb.user_store import sanitize_user
        from audrey.models.health import HealthTracker
        from audrey.models.ollama import OllamaClient
        from audrey.pipeline.fair_gate import FairLocalGate
        from audrey.processing_policy import ProcessingPolicy, processing_scope
        from audrey.routes.app import router
        from audrey.routes.app.agents import AgentCreateRequest, AgentTaskResponse
        from audrey.routes.inflight import UserInflightRegistry
        from audrey.tools.agent_file_reads import (
            agent_file_read_handler,
            agent_file_read_scope,
            with_selected_file_read,
        )
        from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry
        from audrey.tools.dispatch import dispatch_one
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks Agent file reads; rebuild audrey.") from error
    if ("file_ids" not in AgentCreateRequest.model_fields or "file_inputs" not in AgentTaskResponse.model_fields
            or not hasattr(AgentFileInputs, "read_selected") or "read_selected_file" not in TOOL_DECLARATIONS):
        raise SmokeSetupError("Installed backend lacks Agent file reads; rebuild audrey.")
    return SimpleNamespace(
        httpx=httpx, FastAPI=FastAPI, AgentFileInputs=AgentFileInputs,
        get_agent_file_inputs=get_agent_file_inputs, AgentTaskContextError=AgentTaskContextError,
        ResearchAgentExecutor=ResearchAgentExecutor, AgentRuntime=AgentRuntime,
        ApplicationStore=ApplicationStore, require_principal=require_principal,
        FileDeletionWorker=FileDeletionWorker, FileOperationLocks=FileOperationLocks,
        StorageLifecycle=StorageLifecycle, UploadsDB=UploadsDB, sanitize_user=sanitize_user,
        HealthTracker=HealthTracker, OllamaClient=OllamaClient, token_count=token_count,
        FairLocalGate=FairLocalGate, UserInflightRegistry=UserInflightRegistry,
        ToolRegistry=ToolRegistry, router=router,
        AgentTaskError=AgentTaskError, AgentDispatchStoppedError=AgentDispatchStoppedError,
        agent_file_read_handler=agent_file_read_handler, agent_file_read_scope=agent_file_read_scope,
        with_selected_file_read=with_selected_file_read, dispatch_one=dispatch_one,
        ProcessingPolicy=ProcessingPolicy, processing_scope=processing_scope,
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
        self.http_patch = None

    async def __aenter__(self):
        try:
            r = self.r
            self.owner, self.other = [await self.store.resolve_external_identity(
                provider="agent-file-read-smoke", subject=name, email=f"{name}@example.com",
                display_name=name, role="user", auth_method="test",
            ) for name in ("alice", "bob")]
            self.current = self.owner
            self.project = await self.store.projects.create(user_id=self.owner.user_id, name="Isolated selected files")
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
            self.service = r.get_agent_file_inputs(self.app)
            original_client = r.httpx.AsyncClient

            async def forbidden(_request):
                raise SmokeFailureError("The isolated file input task attempted external HTTP.")

            def client(*args, **kwargs):
                kwargs.setdefault("transport", r.httpx.MockTransport(forbidden))
                return original_client(*args, **kwargs)

            self.http_patch = patch.object(r.httpx, "AsyncClient", client)
            self.http_patch.start()
            return self
        except BaseException:
            await self.__aexit__()
            raise

    async def __aexit__(self, *_error):
        try:
            if self.client:
                await self.client.aclose()
        finally:
            if self.http_patch:
                self.http_patch.stop()
            self.db.close()
            self.store.close()

    async def file(self, *, text="SELECTED_FILE_BODY_123. Make the delivery checklist.", name="source.txt",
                   kind="text", owner=None, private=False, origin="upload", attached=True,
                   status="ready", cached=False, reclaimed=False):
        owner = owner or self.owner
        file_id = str(uuid.uuid4())
        directory = self.root / "files" / self.r.sanitize_user(owner.storage_namespace)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (file_id + Path(name).suffix.lower())
        path.write_text(text if kind == "text" and not cached else "original fixture", encoding="utf-8")
        mime = {"text": "text/plain", "image": "image/png", "audio": "audio/mpeg", "video": "video/mp4"}[kind]
        with self.db._lock:
            self.db._record_locked(file_id, owner.storage_namespace, name, mime, path.stat().st_size,
                kind, "", 0, "2026-11-03T12:00:00+00:00", status, private_required=private, origin=origin)
        artifact = path.with_suffix(".transcript.txt" if kind in {"audio", "video"} else ".ocr.txt")
        if cached or kind in {"audio", "video"} or origin == "generated":
            artifact.write_text(text, encoding="utf-8")
        else:
            artifact = path
        if reclaimed:
            path.unlink()
        if attached:
            # Fixtures can represent catalog drift that normal attachment routes refuse.
            with self.store._lock, self.store._conn:
                self.store._conn.execute(
                    "INSERT INTO app_project_files(project_id,file_id,added_at,origin) VALUES (?,?,?,?)",
                    (self.project.project_id, file_id, "2026-11-03T12:00:00+00:00", origin),
                )
        return SimpleNamespace(id=file_id, path=path, artifact=artifact, text=text, name=name)

    async def create(self, files=(), *, status=201, **changes):
        self.counter += 1
        body = {"objective": "Use the explicitly selected files to create a practical checklist.",
                "request_id": f"file-read-smoke-{self.counter}", "project_id": self.project.project_id,
                "file_ids": [item.id for item in files]}
        body.update(changes)
        response = await self.client.post("/api/agents", json=body)
        require(response.status_code == status,
                f"Agent file read creation expected HTTP {status}, received {response.status_code}.")
        return response.json()

    async def saved(self, task):
        response = await self.client.get(f"/api/agents/{task['id']}")
        require(response.status_code == 200, "The saved owned Agent could not be read.")
        return response.json()

    async def control(self, task, action, *, status=200):
        current = await self.saved(task)
        response = await self.client.post(f"/api/agents/{task['id']}/actions",
            json={"action": action, "expected_revision": current["revision"]})
        require(response.status_code == status,
                f"Agent file read control expected HTTP {status}, received {response.status_code}.")
        return response.json()

    async def complete(self, task):
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The disposable Agent could not be claimed.")
        for step in ("plan", "research", "review"):
            await self.store.agents.reserve_operation(**lease(claimed), step_id=step)
            await self.store.agents.checkpoint(**lease(claimed), step_id=step, result=f"Saved isolated {step}.")
        return await self.store.agents.complete(**lease(claimed), result="Saved the requested checklist.")

    async def work(self, task):
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The disposable file-read Agent could not be claimed.")
        await self.store.agents.reserve_operation(**lease(claimed), step_id="plan")
        claimed = await self.store.agents.checkpoint(**lease(claimed), step_id="plan", result="Use the selected source.")
        await self.store.agents.reserve_operation(**lease(claimed), step_id="research")
        return claimed


async def read(s, task, arguments):
    with s.r.agent_file_read_scope(app=s.app, principal=s.owner, task=task):
        handler = s.r.agent_file_read_handler(user_id=s.owner.storage_namespace)
        require(handler is not None, "Work did not bind the selected-file reader.")
        return await handler(arguments)


async def denied(s, operation):
    try:
        result = await operation()
    except (s.r.AgentDispatchStoppedError, s.r.AgentTaskError):
        return
    require(isinstance(result, dict) and result.get("error") and "text" not in result,
            "An unauthorized or stale read returned selected file text.")


def tool_page(payload, name="read_selected_file"):
    messages = [item for item in payload["messages"] if item.get("role") == "tool"
                and (item.get("tool_name") or item.get("name")) == name]
    require(bool(messages), "The model did not receive the actual native tool reply.")
    content = messages[-1]["content"]
    require(isinstance(content, str) and len(content) <= 5000, "The selected-file reply exceeded its JSON allowance.")
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        raise SmokeFailureError("The selected-file reply was cut into invalid JSON.") from None
    require(isinstance(result, dict) and not result.get("error"), "The selected-file read did not return source text.")
    return result


async def execute(s):
    r, requests, clients, delivered = s.r, [], [], []
    marker = "BEYOND_INITIAL_PREFIX_MARKER_731"
    prefix = "A preliminary source detail for this task. " * 1200
    chosen = await s.file(text=prefix + marker + ". Verified delivery detail. " * 100,
                          name="SOURCE_FILENAME_NOT_FOR_EVENTS.mp4", kind="video", reclaimed=True)
    unselected = await s.file(text="UNSELECTED_BODY_NEVER_READ_984", name="unselected.txt")
    task = await s.create([chosen], objective="OBJECTIVE_NOT_FOR_EVENTS: read later source details and save a checklist.")
    require(task.get("file_inputs", {}).get("files", [{}])[0].get("truncated"),
            "The later-text fixture did not exceed the initial selected-file prefix.")
    require(task["file_inputs"]["files"][0]["chars"] < len(prefix), "The read marker was inside the initial source prefix.")

    async def provider(request):
        if request.url.path == "/api/tags":
            return r.httpx.Response(200, json={"models": [{"name": "qwen3.8:latest", "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return r.httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
                "general.architecture": "qwen3", "general.parameter_count": 27000000000}})
        require(request.url.path == "/api/chat" and len(requests) < 7, "An unexpected local model operation was dispatched.")
        payload = json.loads(request.content)
        require(payload["model"] == "qwen3.8:latest", "A selected-file read used an unverified or nonlocal model.")
        require(unselected.text not in json.dumps(payload), "An unselected file reached the model.")
        index = len(requests)
        requests.append(payload)
        message = {"content": "Finished the requested checklist.", "thinking": "REASONING_NOT_FOR_EVENTS"}
        if index == 0:
            message["content"] = "Selected File Checklist"
        elif index == 1:
            message["content"] = "Read the relevant later file sections and save the checklist."
        elif index == 2:
            tools = {tool["function"]["name"]: tool["function"] for tool in payload.get("tools", [])}
            require(set(tools) == {"create_file", "read_selected_file"}, "Work lacked its native reader or received unrelated tools.")
            require(set(tools["read_selected_file"]["parameters"]["properties"]) == {"file_id", "offset", "limit"},
                    "The reader allowed the model to choose a user, Project or path.")
            require(marker not in json.dumps(payload["messages"]), "The later-text marker was silently copied into initial context.")
            message = {"content": "", "tool_calls": [{"function": {"name": "read_selected_file", "arguments": {
                "file_id": chosen.id, "offset": len(prefix), "limit": 400}}}]}
        elif index == 3:
            first = tool_page(payload)
            require(marker in first.get("text", "") and first.get("artifact") == "transcript"
                    and first.get("origin") == "upload" and first.get("has_more") is True,
                    "Work did not read the requested later retained transcript with truthful provenance.")
            delivered.append(first)
            message = {"content": "", "tool_calls": [{"function": {"name": "read_selected_file", "arguments": {
                "file_id": chosen.id, "offset": first["next_offset"], "limit": 400}}}]}
        elif index == 4:
            second = tool_page(payload)
            require(second["start"] == delivered[0]["end"] and second["text"] == chosen.text[second["start"]:second["end"]],
                    "A model pagination request skipped, repeated or invented source text.")
            delivered.append(second)
            content = "# Requested checklist\n\n" + "\n\n".join(item["text"] for item in delivered)
            message = {"content": "", "tool_calls": [{"function": {"name": "create_file", "arguments": {
                "filename": "checklist.md", "format": "md", "content": content}}}]}
        elif index == 6:
            require(not payload.get("tools"), "Review received a file-reading or writing tool.")
        return r.httpx.Response(200, json={"done": True, "done_reason": "stop", "prompt_eval_count": 13,
                                         "eval_count": 7, "message": message})

    def factory(endpoint, **kwargs):
        value = r.OllamaClient(endpoint, transport=r.httpx.MockTransport(provider), **kwargs)
        clients.append(value)
        return value

    require(await r.AgentRuntime(s.store.agents, r.ResearchAgentExecutor(s.app, client_factory=factory)).run_once(),
            "The isolated file-reading Agent did not run.")
    saved = await s.saved(task)
    require(saved["status"] == "completed" and len(saved.get("outputs", [])) == 1 and len(delivered) == 2,
            "The file-reading Agent did not complete its native read-and-save task.")
    expected = "# Requested checklist\n\n" + "\n\n".join(item["text"] for item in delivered)
    downloaded = await s.client.get(saved["outputs"][0]["download_url"])
    require(downloaded.status_code == 200 and downloaded.text == expected and marker in downloaded.text,
            "The output download did not contain the actual later source pages.")
    require(not chosen.path.exists() and len(requests) == 7 and all(client._client.is_closed for client in clients),
            "Selected reads required a reclaimed original, repeated model work or leaked a client.")
    require(not s.app.state.gate.pressure_snapshot()["in_use"] and not s.app.state.inflight.pressure_snapshot()["in_use"],
            "Native file reading retained model admission resources.")
    return task, chosen, delivered, expected


async def local_agent_reads_later_pages_and_saves_output(s):
    await execute(s)


async def frozen_read_versions_survive_restart_and_rerun(s):
    chosen = await s.file(text="Frozen source beginning. " * 500 + "LATER_FROZEN_DETAIL_493")
    task = await s.create([chosen])
    claimed = await s.work(task)
    offset = task["file_inputs"]["files"][0]["chars"]
    first = await read(s, claimed, {"file_id": chosen.id, "offset": offset, "limit": 200})
    require(bool(first.get("text")) and first["text"] == chosen.text[offset:first["end"]]
            and first["start"] == offset, "The first version did not return the frozen text.")
    with s.store._lock:
        pins = [tuple(row) for row in s.store._conn.execute(
            "SELECT file_id,pin_json FROM app_agent_file_inputs WHERE task_id=?", (task["id"],))]
    path = s.store.path
    s.store.close()
    s.store = s.r.ApplicationStore(path)
    s.app.state.application_store = s.store
    s.app.state.agent_file_inputs = None
    s.service = s.r.get_agent_file_inputs(s.app)
    require(await read(s, claimed, {"file_id": chosen.id, "offset": offset, "limit": 200}) == first,
            "Reopening storage lost the frozen file read version.")
    await s.store.agents.checkpoint(**lease(claimed), step_id="research", result="Read the source.")
    await s.store.agents.reserve_operation(**lease(claimed), step_id="review")
    await s.store.agents.checkpoint(**lease(claimed), step_id="review", result="Reviewed work.")
    final = await s.store.agents.complete(**lease(claimed), result="Reviewed work.")
    rerun = await s.control(final, "rerun")
    resumed = await s.store.agents.claim()
    require(resumed is not None and resumed["id"] == task["id"] and resumed["current_step"] == "research"
            and rerun["steps"][0]["result"] == "Use the selected source.", "Run again created a new Agent or discarded its Plan.")
    require(await read(s, resumed, {"file_id": chosen.id, "offset": offset, "limit": 200}) == first,
            "Run again recaptured a different selected-file read version.")
    with s.store._lock:
        after = [tuple(row) for row in s.store._conn.execute(
            "SELECT file_id,pin_json FROM app_agent_file_inputs WHERE task_id=?", (task["id"],))]
    require(after == pins and chosen.text not in json.dumps(after), "Read/restart/rerun changed pins or persisted source text.")


async def selected_owner_private_and_scope_fences(s):
    chosen = await s.file()
    unselected = await s.file(text="UNSELECTED_SECRET_619")
    foreign = await s.file(owner=s.other, text="FOREIGN_SECRET_357")
    protected = await s.file(private=True, cached=True, text="PRIVATE_SECRET_842")
    task = await s.work(await s.create([chosen]))
    for item in (unselected, foreign, protected):
        await denied(s, lambda item=item: read(s, task, {"file_id": item.id}))
    with s.r.agent_file_read_scope(app=s.app, principal=s.owner, task=task):
        try:
            s.r.agent_file_read_handler(user_id=s.other.storage_namespace)
        except s.r.AgentDispatchStoppedError:
            pass
        else:
            raise SmokeFailureError("A different caller used the Work-bound file reader.")
    try:
        with s.r.agent_file_read_scope(app=s.app, principal=s.other, task=task):
            raise SmokeFailureError("A foreign principal opened an owned task file scope.")
    except s.r.AgentDispatchStoppedError:
        pass
    registry = s.r.with_selected_file_read(s.r.ToolRegistry())
    result = await s.r.dispatch_one(s.client, registry, {"function": {"name": "read_selected_file",
        "arguments": {"file_id": chosen.id}}}, max_result_chars=6000, timeout_s=2, user_id=s.owner.storage_namespace)
    require(result.is_error and chosen.text not in result.content, "A reader without an Agent scope returned source text or used HTTP.")
    policy = s.r.ProcessingPolicy(local_models={"qwen3.8:latest": "a" * 64}, ollama_base_url="http://mock-local.invalid")
    with s.r.processing_scope(policy):
        await denied(s, lambda: read(s, task, {"file_id": chosen.id}))


async def source_changes_and_expired_callbacks_stop_reads(s):
    for action in ("change", "delete", "detach", "cancel"):
        chosen = await s.file(cached=True, name=f"{action}.txt")
        task = await s.work(await s.create([chosen]))
        original = s.service.reader._disk
        changed = False

        async def race(function, *args, original=original, action=action, chosen=chosen, task=task):
            nonlocal changed
            result = await original(function, *args)
            if not changed:
                changed = True
                if action == "change":
                    chosen.artifact.write_text("The selected source changed after its read.", encoding="utf-8")
                elif action == "delete":
                    response = await s.client.delete(f"/api/files/{chosen.id}")
                    require(response.status_code == 200, "The read-deletion fixture could not be removed.")
                elif action == "detach":
                    require(await s.store.projects.remove_file(user_id=s.owner.user_id, project_id=s.project.project_id,
                            file_id=chosen.id), "The read-detachment fixture could not be removed.")
                else:
                    await s.control(task, "cancel")
            return result

        with patch.object(s.service.reader, "_disk", race):
            await denied(s, lambda task=task, chosen=chosen: read(s, task, {"file_id": chosen.id}))
        require(changed, "The final-read race fixture did not reach canonical source I/O.")
        if action != "cancel":
            await s.control(task, "cancel")
    chosen = await s.file(name="scope.txt")
    task = await s.work(await s.create([chosen]))
    release = asyncio.Event()
    with s.r.agent_file_read_scope(app=s.app, principal=s.owner, task=task):
        handler = s.r.agent_file_read_handler(user_id=s.owner.storage_namespace)

        async def later():
            await release.wait()
            return await handler({"file_id": chosen.id})

        escaped = asyncio.create_task(later())
    release.set()
    await denied(s, lambda: escaped)
    await denied(s, lambda: handler({"file_id": chosen.id}))


async def json_unicode_cursors_and_argument_bounds(s):
    text = 'English source with "quotes", \\slashes\\, a tab\t and an emoji 😀.\n' * 300
    chosen = await s.file(text=text, name="long-" + "A" * 250 + ".txt", cached=True, origin="generated")
    task = await s.work(await s.create([chosen]))
    registry = s.r.with_selected_file_read(s.r.ToolRegistry())
    offset, pages = 0, []
    with s.r.agent_file_read_scope(app=s.app, principal=s.owner, task=task):
        while True:
            result = await s.r.dispatch_one(s.client, registry, {"function": {"name": "read_selected_file", "arguments": {
                "file_id": chosen.id, "offset": offset, "limit": 6000}}},
                max_result_chars=6000, timeout_s=2, user_id=s.owner.storage_namespace)
            require(not result.is_error and len(result.content) <= 5000, "A paged read failed or exceeded its serialized result limit.")
            try:
                value = json.loads(result.content)
            except ValueError:
                raise SmokeFailureError("Escaped source text produced invalid paged JSON.") from None
            require(value["start"] == offset and value["end"] == offset + len(value["text"])
                    and value["text"] == text[offset:value["end"]] and value["total_chars"] == len(text)
                    and value["origin"] == "generated" and len(value["filename"]) <= 240,
                    "Read offsets, Unicode text or generated-output provenance were inaccurate.")
            handler = s.r.agent_file_read_handler(user_id=s.owner.storage_namespace)
            repeated = await handler({"file_id": chosen.id, "offset": offset, "limit": 6000})
            require(repeated == value, "Repeating a file cursor changed its result.")
            pages.append(value["text"])
            more = value["end"] < len(text)
            require(value["has_more"] is more and value["next_offset"] == (value["end"] if more else None),
                    "Pagination advertised an untruthful next offset or completion state.")
            if not more:
                break
            require(value["end"] > offset, "A paged read did not advance its cursor.")
            offset = value["next_offset"]
    require("".join(pages) == text and len(pages) > 1, "Paged JSON lost or duplicated selected Unicode characters.")
    eof = await read(s, task, {"file_id": chosen.id, "offset": len(text)})
    require(eof["text"] == "" and eof["start"] == eof["end"] == len(text)
            and eof["next_offset"] is None and eof["has_more"] is False,
            "Reading exactly at EOF returned false source text or a looping cursor.")
    require(await read(s, task, {"file_id": chosen.id, "offset": len(text)}) == eof, "Repeated EOF was not stable.")
    for arguments in ({"offset": -1}, {"offset": True}, {"offset": "0"}, {"offset": 500001},
                      {"offset": len(text) + 1}, {"limit": 0}, {"limit": True}, {"limit": 6001},
                      {"user": "another-user"}, {"path": "/host/path"}):
        await denied(s, lambda arguments=arguments: read(s, task, {"file_id": chosen.id, **arguments}))


async def read_events_keep_metadata_without_bodies(s):
    task, chosen, delivered, output = await execute(s)
    response = await s.client.get(f"/api/agents/{task['id']}/events")
    require(response.status_code == 200, "Selected-file tool events could not be read.")
    events = response.json()["items"]
    tools = sorted((item for item in events if item["kind"] == "tool"), key=lambda item: item["seq"])
    require([item["tool"] for item in tools] == ["read_selected_file", "read_selected_file", "create_file"]
            and all(item["state"] == "completed" for item in tools), "Actual selected reads were missing or misreported in execution history.")
    expected_fields = {"seq", "attempt", "stage", "kind", "state", "tool", "created_at", "finished_at",
                       "duration_ms", "input_tokens", "output_tokens"}
    require(all(set(item) == expected_fields for item in events), "File-read history returned fields beyond execution metadata.")
    with s.store._lock:
        stored = [dict(row) for row in s.store._conn.execute("SELECT * FROM app_agent_events WHERE task_id=?", (task["id"],))]
    encoded = json.dumps([events, stored])
    require(all(value not in encoded for value in [chosen.name, chosen.id, output, delivered[0]["text"],
                "OBJECTIVE_NOT_FOR_EVENTS", "REASONING_NOT_FOR_EVENTS", str(s.root)]),
            "File-read events copied source text, arguments, filenames, prompts, reasoning or host paths.")
    s.current = s.other
    require((await s.client.get(f"/api/agents/{task['id']}/events")).status_code == 404,
            "A foreign account read selected-file task events.")
    s.current = s.owner


CASES = (
    ("local_agent_reads_later_pages_and_saves_output", local_agent_reads_later_pages_and_saves_output),
    ("frozen_read_versions_survive_restart_and_rerun", frozen_read_versions_survive_restart_and_rerun),
    ("selected_owner_private_and_scope_fences", selected_owner_private_and_scope_fences),
    ("source_changes_and_expired_callbacks_stop_reads", source_changes_and_expired_callbacks_stop_reads),
    ("json_unicode_cursors_and_argument_bounds", json_unicode_cursors_and_argument_bounds),
    ("read_events_keep_metadata_without_bodies", read_events_keep_metadata_without_bodies),
)

async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-file-reads-smoke-") as directory:
        for name, case in CASES:
            try:
                async with asyncio.timeout(timeout_seconds), Scenario(runtime, Path(directory) / name) as scenario:
                    await case(scenario)
            except TimeoutError:
                raise SmokeFailureError(f"{name} deadline exceeded.") from None
            checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable file read fixtures were not removed.")
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
        result = (130, "Agent file read smoke interrupted.")
    except BaseException:  # noqa: BLE001 - do not expose SQL, paths or private content
        result = (3, "Agent file read smoke harness crashed.")
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
        raise SmokeSetupError("Agent file read smoke requires Linux fork support.") from None
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-file-reads-smoke-") as directory:
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
    require(not Path(directory).exists(), "Disposable file read fixtures were not removed.")
    return [*checks, "isolated_fixture_cleanup"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks = 0, []
    message = "Agent file read smoke passed; isolated temporary data and mocked local replies; no live accounts/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = run_processes(load_runtime(), args.timeout_seconds)
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Agent file read smoke interrupted."
    except Exception:  # noqa: BLE001 - private paths, SQL and content never enter terminal output
        code, message = 3, "Agent file read smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
