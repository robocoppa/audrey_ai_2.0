"""Installed file-backed approved Project Knowledge using disposable canonical state.

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

        from audrey.agent_executor import ResearchAgentExecutor
        from audrey.agent_file_inputs import get_agent_file_inputs
        from audrey.agent_knowledge_files import get_agent_knowledge_files
        from audrey.agent_runtime import AgentRuntime
        from audrey.app_state import ApplicationStore, agent_projects
        from audrey.app_state.agent_projects import token_count
        from audrey.app_state.agents import AgentTaskContextError
        from audrey.auth import require_principal
        from audrey.content_sources import ContentSourceReader
        from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
        from audrey.kb.storage_lifecycle import StorageLifecycle
        from audrey.kb.uploads_db import UploadsDB
        from audrey.kb.user_store import sanitize_user
        from audrey.models.health import HealthTracker
        from audrey.models.ollama import OllamaClient
        from audrey.pipeline.fair_gate import FairLocalGate
        from audrey.routes.app import router
        from audrey.routes.app.agents import AgentCreateRequest, AgentTaskResponse
        from audrey.routes.inflight import UserInflightRegistry
        from audrey.tools.discovery import ToolRegistry
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks Agent Knowledge files; rebuild audrey.") from error
    if ("file_ids" not in AgentCreateRequest.model_fields or "file_inputs" not in AgentTaskResponse.model_fields
            or "file" not in getattr(agent_projects, "SUPPORTED_KNOWLEDGE_SOURCE_KINDS", frozenset())):
        raise SmokeSetupError("Installed backend lacks Agent Knowledge files; rebuild audrey.")
    return SimpleNamespace(
        httpx=httpx, FastAPI=FastAPI,
        get_agent_file_inputs=get_agent_file_inputs, AgentTaskContextError=AgentTaskContextError,
        ResearchAgentExecutor=ResearchAgentExecutor, AgentRuntime=AgentRuntime,
        ApplicationStore=ApplicationStore, require_principal=require_principal,
        FileDeletionWorker=FileDeletionWorker, FileOperationLocks=FileOperationLocks,
        StorageLifecycle=StorageLifecycle, UploadsDB=UploadsDB, sanitize_user=sanitize_user,
        HealthTracker=HealthTracker, OllamaClient=OllamaClient, token_count=token_count,
        FairLocalGate=FairLocalGate, UserInflightRegistry=UserInflightRegistry,
        ToolRegistry=ToolRegistry, router=router,
        get_agent_knowledge_files=get_agent_knowledge_files, ContentSourceReader=ContentSourceReader,
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
                provider="agent-knowledge-file-smoke", subject=name, email=f"{name}@example.com",
                display_name=name, role="user", auth_method="test",
            ) for name in ("alice", "bob")]
            self.current = self.owner
            self.project = await self.store.projects.create(user_id=self.owner.user_id, name="Isolated file-backed Knowledge")
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
            self.knowledge_files = r.get_agent_knowledge_files(self.app)
            self.reader = r.ContentSourceReader(self.store, self.db, self.root / "files")
            original_client = r.httpx.AsyncClient

            async def forbidden(_request):
                raise SmokeFailureError("The isolated Knowledge task attempted external HTTP.")

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
        body = {"objective": "Use approved Project notes to prepare a practical checklist.",
                "request_id": f"knowledge-file-smoke-{self.counter}", "project_id": self.project.project_id,
                "file_ids": [item.id for item in files]}
        body.update(changes)
        response = await self.client.post("/api/agents", json=body)
        require(response.status_code == status,
                f"Agent Knowledge file creation expected HTTP {status}, received {response.status_code}.")
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
                f"Agent Knowledge file control expected HTTP {status}, received {response.status_code}.")
        return response.json()

    async def complete(self, task):
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The disposable Agent could not be claimed.")
        for step in ("plan", "research", "review"):
            await self.store.agents.reserve_operation(**lease(claimed), step_id=step)
            await self.store.agents.checkpoint(**lease(claimed), step_id=step, result=f"Saved isolated {step}.")
        return await self.store.agents.complete(**lease(claimed), result="Saved the requested checklist.")

    async def reference(self, source_id, *, principal=None):
        principal = principal or self.owner
        document = await self.reader.resolve(principal.user_id, principal.storage_namespace, source_id)
        require(document is not None and self.reader.is_current(document), "The canonical Knowledge source could not resolve.")
        return {"source_id": source_id, "revision": document.revision}

    async def note(self, text="An approved practical constraint.", *, kind="decision", sources=(), **changes):
        return await self.store.knowledge.create(user_id=self.owner.user_id, project_id=self.project.project_id,
            kind=kind, text=text, sources=sources, **changes)

    async def message(self):
        self.counter += 1
        conversation = await self.store.conversations.create(user_id=self.owner.user_id,
            title="Isolated Knowledge evidence", project_id=self.project.project_id)
        message_id = f"knowledge-message-{self.counter}"
        with self.store._lock, self.store._conn:
            self.store._conn.execute(
                "INSERT INTO app_messages(message_id,conversation_id,user_id,sequence_no,role,status,content,created_at,updated_at) "
                "VALUES (?,?,?,1,'user','completed',?,?,?)", (message_id, conversation.conversation_id,
                 self.owner.user_id, "MESSAGE_BODY_NOT_AGENT_CONTEXT", "2026-11-03T12:00:00+00:00", "2026-11-03T12:00:00+00:00"))
        return await self.reference(f"message:{message_id}")

    async def report(self):
        task = await self.create(project_id=None)
        final = await self.complete(task)
        return final, await self.reference(f"agent:{final['id']}:report")

    def manifest(self, task):
        with self.store._lock:
            return dict(self.store._conn.execute("SELECT * FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone())

    def snapshot(self):
        queries = {
            "tasks": "SELECT * FROM app_agent_tasks ORDER BY rowid",
            "operations": "SELECT * FROM app_agent_operations ORDER BY rowid",
            "notices": "SELECT * FROM app_agent_notices ORDER BY rowid",
            "events": "SELECT * FROM app_agent_events ORDER BY rowid",
            "questions": "SELECT * FROM app_agent_input_requests ORDER BY rowid",
            "contexts": "SELECT * FROM app_agent_project_contexts ORDER BY rowid",
            "files": "SELECT * FROM app_agent_file_inputs ORDER BY rowid",
        }
        with self.store._lock:
            return {table: [tuple(row) for row in self.store._conn.execute(query)] for table, query in queries.items()}


async def current_file_and_mixed_source_notes(s):
    plain = await s.file(text="PLAIN_FILE_BODY_NOT_AGENT_CONTEXT")
    speech = await s.file(text="TRANSCRIPT_BODY_NOT_AGENT_CONTEXT", name="retained.mp4", kind="video", reclaimed=True)
    generated = await s.file(text="GENERATED_BODY_NOT_AGENT_CONTEXT", name="working.md", cached=True, origin="generated")
    message_ref = await s.message()
    report, report_ref = await s.report()
    notes = [
        await s.note("Approved summary of the saved file.", sources=[await s.reference(f"file:{plain.id}:text")]),
        await s.note("Approved summary of retained speech.", sources=[await s.reference(f"file:{speech.id}:transcript")]),
        await s.note("Approved working-output constraint.", sources=[await s.reference(f"file:{generated.id}:text")]),
        await s.note("Approved mixed-source constraint.", sources=[await s.reference(f"file:{plain.id}:text"), message_ref, report_ref]),
        await s.note("An approved source-free preference."),
    ]
    task = await s.create()
    selected = task["project_context"]["knowledge"]
    require({item["id"] for item in selected} == {item["id"] for item in notes},
            "Current file-backed or mixed-source approved Knowledge was not selected.")
    claimed = await s.store.agents.claim()
    require(claimed is not None and claimed["id"] == task["id"], "The file-backed Knowledge Agent could not be claimed.")
    context = await s.store.agents.resolve_project_context(**lease(claimed))
    require({item["text"] for item in context["knowledge"]} == {item["text"] for item in notes},
            "File-backed Knowledge changed the approved note text.")
    encoded = json.dumps([context, s.manifest(task)])
    require(all(value not in encoded for value in [plain.text, speech.text, generated.text,
        "MESSAGE_BODY_NOT_AGENT_CONTEXT", report["result"], str(s.root)]),
        "File-backed Knowledge copied source bodies or host paths into Agent context or manifests.")
    manifest = s.manifest(task)
    require(len(manifest["knowledge_json"]) <= 4000 and all(note["text"] not in json.dumps(manifest) for note in notes),
            "The frozen Knowledge manifest copied note bodies or exceeded its existing metadata limit.")
    require(sum(s.r.token_count(note["text"]) for note in context["knowledge"]) <= 1000,
            "File-backed notes exceeded the existing Knowledge context allowance.")
    require("file_inputs" not in task and not s.store._conn.execute(
        "SELECT 1 FROM app_agent_file_inputs WHERE task_id=?", (task["id"],)).fetchone(),
        "A Knowledge reference implicitly selected its file for Agent tools.")


async def execute_note_only(s, *, request_file_read=False):
    chosen = await s.file(text="RAW_FILE_BODY_NEVER_DISPATCH_763", name="source-evidence.txt", cached=True)
    note = await s.note("APPROVED_NOTE_CONTEXT_854: prepare the requested practical checklist.",
                        sources=[await s.reference(f"file:{chosen.id}:text")])
    task = await s.create()
    require(any(item["id"] == note["id"] for item in task["project_context"]["knowledge"]),
            "The approved file summary was absent before local execution.")
    requests, clients, r = [], [], s.r
    output = "# Practical checklist\n\n" + note["text"]

    async def provider(request):
        if request.url.path == "/api/tags":
            return r.httpx.Response(200, json={"models": [{"name": "qwen3.8:latest", "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return r.httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
                "general.architecture": "qwen3", "general.parameter_count": 27000000000}})
        require(request.url.path == "/api/chat" and len(requests) < 5, "Unexpected Knowledge Agent model work was dispatched.")
        payload = json.loads(request.content)
        require(payload["model"] == "qwen3.8:latest", "File-backed notes reached a nonlocal model.")
        serialized = json.dumps(payload["messages"])
        require(chosen.text not in serialized and chosen.id not in serialized and chosen.name not in serialized,
                "Knowledge reuse dispatched a file body, filename or implicit file-tool ID.")
        index = len(requests)
        if index:
            require(note["text"] in serialized, "A model step did not receive the approved file-backed note.")
        tools = {tool["function"]["name"] for tool in payload.get("tools", [])}
        require("read_selected_file" not in tools, "A Knowledge source implicitly enabled the selected-file reader.")
        requests.append(payload)
        message = {"content": "Prepared the requested practical checklist."}
        if index == 0:
            message["content"] = "Practical Work Checklist"
        elif index == 2:
            if request_file_read:
                message = {"content": "", "tool_calls": [{"function": {"name": "read_selected_file", "arguments": {
                    "file_id": chosen.id}}}]}
            else:
                require("create_file" in tools, "The approved-note task lost its native output tool.")
                message = {"content": "", "tool_calls": [{"function": {"name": "create_file", "arguments": {
                    "filename": "approved-note-checklist.md", "format": "md", "content": output}}}]}
        return r.httpx.Response(200, json={"done": True, "done_reason": "stop", "eval_count": 7,
                                         "prompt_eval_count": 13, "message": message})

    def factory(endpoint, **kwargs):
        client = r.OllamaClient(endpoint, transport=r.httpx.MockTransport(provider), **kwargs)
        clients.append(client)
        return client

    require(await r.AgentRuntime(s.store.agents, r.ResearchAgentExecutor(s.app, client_factory=factory)).run_once(),
            "The isolated note-only Agent did not run.")
    saved = await s.saved(task)
    require(saved["status"] == ("failed" if request_file_read else "completed")
            and len(requests) == (3 if request_file_read else 5),
            "The note-only task failed or an unselected file read was dispatched.")
    if not request_file_read:
        require(len(saved.get("outputs", [])) == 1, "The approved-note task did not save its derived deliverable.")
        downloaded = await s.client.get(saved["outputs"][0]["download_url"])
        require(downloaded.status_code == 200 and downloaded.text == output,
                "File-backed Knowledge publication lost the actual derived output bytes.")
    require("file_inputs" not in saved and all(client._client.is_closed for client in clients),
            "Knowledge reuse created implicit file inputs or leaked its model client.")
    require(not s.app.state.gate.pressure_snapshot()["in_use"] and not s.app.state.inflight.pressure_snapshot()["in_use"],
            "Knowledge processing retained model admission resources.")
    return task


async def note_only_model_context_and_no_implicit_file_tool(s):
    await execute_note_only(s)
    await execute_note_only(s, request_file_read=True)


async def frozen_note_versions_survive_restart_and_rerun(s):
    chosen = await s.file()
    note = await s.note("Keep the frozen approved file summary.", sources=[await s.reference(f"file:{chosen.id}:text")])
    task = await s.create()
    before = s.manifest(task)
    path = s.store.path
    s.store.close()
    s.store = s.r.ApplicationStore(path)
    s.app.state.application_store = s.store
    s.app.state.agent_file_inputs = None
    s.service = s.r.get_agent_file_inputs(s.app)
    s.knowledge_files = s.r.get_agent_knowledge_files(s.app)
    s.reader = s.r.ContentSourceReader(s.store, s.db, s.root / "files")
    await s.store.projects.update(user_id=s.owner.user_id, project_id=s.project.project_id, name="Renamed ordinary Project")
    await s.note("A later approved note must stay outside the frozen task.", pinned=True)
    require(s.manifest(task) == before, "Reopening storage or harmless additions rewrote frozen Knowledge metadata.")
    completed = await s.complete(task)
    rerun = await s.control(completed, "rerun")
    claimed = await s.store.agents.claim()
    require(claimed is not None and claimed["id"] == task["id"] and claimed["current_step"] == "research"
            and rerun["steps"][0]["result"] == "Saved isolated plan.", "Run again replaced the task or rebuilt its saved Plan.")
    context = await s.store.agents.resolve_project_context(**lease(claimed))
    require([item["id"] for item in context["knowledge"]] == [note["id"]]
            and context["knowledge"][0]["text"] == note["text"] and s.manifest(task) == before,
            "Restart or Run again recaptured new Knowledge instead of frozen file-backed revisions.")


async def changed_file(s, action):
    chosen = await s.file(cached=True)
    await s.note("A frozen file-backed constraint.", sources=[await s.reference(f"file:{chosen.id}:text")])
    task = await s.create()
    claimed = await s.store.agents.claim()
    require(claimed is not None and claimed["id"] == task["id"], "The file-fence fixture could not be claimed.")
    await s.store.agents.reserve_operation(**lease(claimed), step_id="plan")
    if action == "edit":
        chosen.artifact.write_text("The canonical saved text changed.", encoding="utf-8")
    elif action == "delete":
        require((await s.client.delete(f"/api/files/{chosen.id}")).status_code == 200, "The selected evidence file could not be deleted.")
    else:
        require(await s.store.projects.remove_file(user_id=s.owner.user_id, project_id=s.project.project_id, file_id=chosen.id),
                "The selected evidence file could not be detached.")
        if action == "reattach":
            await s.store.projects.add_file(user_id=s.owner.user_id, project_id=s.project.project_id,
                                           file_id=chosen.id, file_is_private=False, file_origin="upload")
    before = s.snapshot()
    try:
        s.store.agents.fence_execution(**lease(claimed))
    except s.r.AgentTaskContextError:
        pass
    else:
        raise SmokeFailureError("Changed file-backed evidence passed the final dispatch fence.")
    for operation in (
        lambda: s.store.agents.validate_lease(**lease(claimed)),
        lambda: s.store.agents.reserve_operation(**lease(claimed), step_id="plan"),
        lambda: s.store.agents.checkpoint(**lease(claimed), step_id="plan", result="A late plan."),
        lambda: s.store.agents.request_input(**lease(claimed), question="A late question?"),
        lambda: s.store.agents.complete(**lease(claimed), result="A late report."),
    ):
        try:
            await operation()
        except s.r.AgentTaskContextError:
            pass
        else:
            raise SmokeFailureError("Changed file-backed evidence allowed late Agent work.")
    require(s.snapshot() == before, "Rejected file-source work mutated Agent results, receipts, notices or history.")
    current = await s.saved(task)
    require(current["project_context"]["current"] is False, "Stale file-backed Knowledge remained advertised as current.")


async def file_edit_delete_detach_and_reattach_fence_work(s):
    for action in ("edit", "delete", "detach", "reattach"):
        async with Scenario(s.r, s.root / action) as child:
            await changed_file(child, action)


async def owner_private_stale_and_unsupported_sources_excluded(s):
    chosen = await s.file()
    valid_ref = await s.reference(f"file:{chosen.id}:text")
    valid = await s.note("The one valid file-backed note.", sources=[valid_ref])
    foreign = await s.file(owner=s.other)
    protected = await s.file(private=True, cached=True)
    detached = await s.file(attached=False)
    stale = await s.file(cached=True)
    stale_ref = await s.reference(f"file:{stale.id}:text")
    stale.artifact.write_text("Changed before Agent capture.", encoding="utf-8")
    refs = [await s.reference(f"file:{foreign.id}:text", principal=s.other),
            await s.reference(f"file:{protected.id}:text"), await s.reference(f"file:{detached.id}:text"),
            stale_ref, {"source_id": "file:missing:text", "revision": "old"},
            {"source_id": f"file:{chosen.id}:visual", "revision": "unsupported"}]
    excluded = [await s.note(f"Excluded source constraint {index}.", sources=[reference]) for index, reference in enumerate(refs)]
    excluded.append(await s.note("A mixed note with one stale file is wholly excluded.", sources=[valid_ref, stale_ref]))
    task = await s.create()
    selected = {item["id"] for item in task["project_context"]["knowledge"]}
    require(selected == {valid["id"]} and not selected.intersection(item["id"] for item in excluded),
            "Knowledge capture admitted private, foreign, detached, stale or unsupported file evidence.")
    require("file_inputs" not in task, "An eligible Knowledge source implicitly selected its file.")


async def assigned_file_backed_actions_keep_origin_and_fences(s):
    chosen = await s.file(cached=True)
    note = await s.note("Prepare the approved file-backed action.", kind="action",
                        sources=[await s.reference(f"file:{chosen.id}:text")])
    with s.store._lock:
        before = [tuple(row) for row in s.store._conn.execute("SELECT * FROM app_project_knowledge_revisions ORDER BY rowid")]
    task = await s.create(knowledge_entry_id=note["id"], knowledge_entry_revision=note["revision"])
    require(task["project_context"].get("origin_entry") == {"id": note["id"], "revision": note["revision"]}
            and task["project_context"]["knowledge"][0]["id"] == note["id"] and "file_inputs" not in task,
            "Assign to agent lost the file-backed origin or implicitly enabled source reading.")
    final = await s.complete(task)
    entry = await s.store.knowledge.get(user_id=s.owner.user_id, project_id=s.project.project_id, entry_id=note["id"])
    with s.store._lock:
        after = [tuple(row) for row in s.store._conn.execute("SELECT * FROM app_project_knowledge_revisions ORDER BY rowid")]
    require(entry["work_status"] == "open" and entry["revision"] == note["revision"] and before == after,
            "Agent completion changed or closed its originating approved action.")
    chosen.artifact.write_text("Origin evidence changed after the completed task.", encoding="utf-8")
    await s.control(final, "rerun", status=409)
    await s.create(knowledge_entry_id=note["id"], knowledge_entry_revision=note["revision"], status=404)
    require((await s.saved(final))["status"] == "completed", "A stale origin erased the retained completed report.")


CASES = (
    ("current_file_and_mixed_source_notes", current_file_and_mixed_source_notes),
    ("note_only_model_context_and_no_implicit_file_tool", note_only_model_context_and_no_implicit_file_tool),
    ("frozen_note_versions_survive_restart_and_rerun", frozen_note_versions_survive_restart_and_rerun),
    ("file_edit_delete_detach_and_reattach_fence_work", file_edit_delete_detach_and_reattach_fence_work),
    ("owner_private_stale_and_unsupported_sources_excluded", owner_private_stale_and_unsupported_sources_excluded),
    ("assigned_file_backed_actions_keep_origin_and_fences", assigned_file_backed_actions_keep_origin_and_fences),
)

async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-knowledge-files-smoke-") as directory:
        for name, case in CASES:
            try:
                async with asyncio.timeout(timeout_seconds), Scenario(runtime, Path(directory) / name) as scenario:
                    await case(scenario)
            except TimeoutError:
                raise SmokeFailureError(f"{name} deadline exceeded.") from None
            checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable Knowledge file fixtures were not removed.")
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
        result = (130, "Agent Knowledge file smoke interrupted.")
    except BaseException:  # noqa: BLE001 - do not expose SQL, paths or private content
        result = (3, "Agent Knowledge file smoke harness crashed.")
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
        raise SmokeSetupError("Agent Knowledge file smoke requires Linux fork support.") from None
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-knowledge-files-smoke-") as directory:
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
    require(not Path(directory).exists(), "Disposable Knowledge file fixtures were not removed.")
    return [*checks, "isolated_fixture_cleanup"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks = 0, []
    message = "Agent Knowledge file smoke passed; isolated temporary data and mocked local replies; no live accounts/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = run_processes(load_runtime(), args.timeout_seconds)
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Agent Knowledge file smoke interrupted."
    except Exception:  # noqa: BLE001 - private paths, SQL and content never enter terminal output
        code, message = 3, "Agent Knowledge file smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
