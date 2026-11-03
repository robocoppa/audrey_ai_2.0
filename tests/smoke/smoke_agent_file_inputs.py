"""Installed explicit Project file inputs using disposable canonical state.

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
        from audrey.agent_file_inputs import AgentFileInputs, get_agent_file_inputs
        from audrey.agent_runtime import AgentRuntime
        from audrey.app_state import ApplicationStore
        from audrey.app_state.agent_projects import token_count
        from audrey.app_state.agents import AgentTaskContextError
        from audrey.auth import require_principal
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
        raise SmokeSetupError("Installed backend lacks Agent file inputs; rebuild audrey.") from error
    if "file_ids" not in AgentCreateRequest.model_fields or "file_inputs" not in AgentTaskResponse.model_fields:
        raise SmokeSetupError("Installed backend lacks Agent file inputs; rebuild audrey.")
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
                provider="agent-file-input-smoke", subject=name, email=f"{name}@example.com",
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
                "request_id": f"file-input-smoke-{self.counter}", "project_id": self.project.project_id,
                "file_ids": [item.id for item in files]}
        body.update(changes)
        response = await self.client.post("/api/agents", json=body)
        require(response.status_code == status,
                f"Agent file input creation expected HTTP {status}, received {response.status_code}.")
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
                f"Agent file input control expected HTTP {status}, received {response.status_code}.")
        return response.json()

    async def complete(self, task):
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The disposable Agent could not be claimed.")
        for step in ("plan", "research", "review"):
            await self.store.agents.reserve_operation(**lease(claimed), step_id=step)
            await self.store.agents.checkpoint(**lease(claimed), step_id=step, result=f"Saved isolated {step}.")
        return await self.store.agents.complete(**lease(claimed), result="Saved the requested checklist.")


async def explicit_selection_and_metadata(s):
    chosen = await s.file()
    unselected = await s.file(text="UNSELECTED_FILE_SECRET_987. Never include this.", name="unselected.txt")
    task = await s.create([chosen], request_id="same-selection")
    inputs = task.get("file_inputs")
    require(inputs and inputs["current"] and len(inputs["files"]) == 1,
            "The task did not capture exactly its explicitly selected file.")
    item = inputs["files"][0]
    require(item == {"file_id": chosen.id, "filename": chosen.name, "artifact": "text",
                     "chars": len(chosen.text), "total_chars": len(chosen.text), "truncated": False, "origin": "upload"},
            "The public file input manifest differs from the canonical saved text.")
    require(chosen.text not in json.dumps(task) and unselected.id not in json.dumps(task),
            "Task metadata copied a source body or an unselected file.")
    same = await s.create([chosen], request_id="same-selection")
    require(same["id"] == task["id"], "A create retry duplicated the selected-file Agent.")
    await s.create([unselected], request_id="same-selection", status=409)
    empty = await s.create()
    require("file_inputs" not in empty, "An unselected Project Agent silently received file inputs.")
    listing = await s.client.get("/api/agents")
    require(listing.status_code == 200 and chosen.text not in listing.text and unselected.text not in listing.text,
            "Agent summaries exposed saved file bodies.")
    with s.store._lock:
        rows = s.store._conn.execute("SELECT file_id,pin_json FROM app_agent_file_inputs WHERE task_id=?", (task["id"],)).fetchall()
    require(len(rows) == 1 and rows[0]["file_id"] == chosen.id and chosen.text not in rows[0]["pin_json"],
            "Durable selected-file manifests copied source bodies or omitted their pins.")


async def local_execution_uses_selected_text_only(s):
    chosen = await s.file()
    unselected = await s.file(text="UNSELECTED_FILE_SECRET_987. Never include this.", name="unselected.txt")
    task = await s.create([chosen])
    requests, clients, r = [], [], s.r
    output = "# Delivery checklist\n\n- Make the delivery checklist."
    replies = ["Selected File Checklist", "Use the selected file to prepare the checklist.", None,
               "The checklist file is ready.", "The task is finished."]

    async def provider(request):
        if request.url.path == "/api/tags":
            return r.httpx.Response(200, json={"models": [{"name": "qwen3.8:latest", "model": "qwen3.8:latest", "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return r.httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
                "general.architecture": "qwen3", "general.parameter_count": 27000000000}})
        require(request.url.path == "/api/chat" and len(requests) < len(replies), "An unexpected model operation was dispatched.")
        payload = json.loads(request.content)
        require(payload["model"] == "qwen3.8:latest", "Selected files were not processed with verified local Qwen.")
        index = len(requests)
        serialized = json.dumps(payload["messages"])
        require(unselected.text not in serialized, "A local model received unselected Project file text.")
        if index:
            require(chosen.text in serialized, "A task step did not receive its selected file text.")
        requests.append(payload)
        message = {"content": replies[index] or ""}
        if index == 2:
            names = {tool["function"]["name"] for tool in payload.get("tools", [])}
            require("create_file" in names and not names & {"web_search", "web_fetch"},
                    "The general file task required web tools or lacked its native output tool.")
            message["tool_calls"] = [{"function": {"name": "create_file", "arguments": {
                "filename": "checklist.md", "format": "md", "content": output}}}]
        return r.httpx.Response(200, json={"done": True, "done_reason": "stop", "eval_count": 7, "message": message})

    def factory(endpoint, **kwargs):
        value = r.OllamaClient(endpoint, transport=r.httpx.MockTransport(provider), **kwargs)
        clients.append(value)
        return value

    require(await r.AgentRuntime(s.store.agents, r.ResearchAgentExecutor(s.app, client_factory=factory)).run_once(),
            "The isolated selected-file Agent did not run.")
    saved = await s.saved(task)
    require(saved["status"] == "completed" and len(saved.get("outputs", [])) == 1,
            "The selected-file Agent did not complete its deliverable.")
    downloaded = await s.client.get(saved["outputs"][0]["download_url"])
    require(downloaded.status_code == 200 and downloaded.text == output,
            "The selected-file Agent did not save the exact requested output bytes.")
    require(len(requests) == 5 and all(client._client.is_closed for client in clients),
            "The selected-file task repeated local work or leaked its model client.")
    require(not s.app.state.gate.pressure_snapshot()["in_use"] and not s.app.state.inflight.pressure_snapshot()["in_use"],
            "Selected-file work retained model admission resources.")


async def frozen_pins_restart_and_rerun(s):
    chosen = await s.file()
    task = await s.create([chosen])
    path = s.store.path
    s.store.close()
    s.store = s.r.ApplicationStore(path)
    s.app.state.application_store = s.store
    s.app.state.agent_file_inputs = None
    s.service = s.r.get_agent_file_inputs(s.app)
    restarted = await s.saved(task)
    require(restarted["file_inputs"] == task["file_inputs"], "Restart lost the frozen selected-file manifest.")
    completed = await s.complete(task)
    rerun = await s.control(completed, "rerun")
    require(rerun["id"] == task["id"] and rerun["file_inputs"] == task["file_inputs"],
            "Run again changed the Agent or recaptured different file inputs.")
    require(rerun["steps"][0]["status"] == "completed" and rerun["steps"][0]["result"] == "Saved isolated plan.",
            "Run again discarded the existing saved Plan.")
    with s.store._lock:
        count = s.store._conn.execute("SELECT COUNT(*) FROM app_agent_file_inputs WHERE task_id=?", (task["id"],)).fetchone()[0]
    require(count == 1, "Restart or Run again duplicated selected-file pins.")


async def owner_private_membership_and_availability(s):
    owned = await s.file()
    foreign = await s.file(owner=s.other)
    protected = await s.file(private=True, cached=True, name="private.txt")
    detached = await s.file(attached=False, name="detached.txt")
    pending = await s.file(status="pending", name="pending.txt")
    unavailable = await s.file(kind="image", name="no-ocr.png")
    for fixture in (foreign, protected, detached, pending, unavailable):
        s.counter += 1
        response = await s.client.post("/api/agents", json={"objective": "Use the chosen file.",
            "request_id": f"unavailable-{s.counter}", "project_id": s.project.project_id, "file_ids": [fixture.id]})
        require(response.status_code in {404, 409, 422}, "An unavailable or unauthorized source was accepted.")
    await s.create([owned], project_id=None, status=422)
    await s.create([owned, owned], status=422)
    await s.create([owned] * 4, status=422)
    s.current = s.other
    await s.create([owned], status=404)
    s.current = s.owner
    require(not await s.store.agents.list(user_id=s.owner.user_id), "A refused file input created an Agent anyway.")


async def source_changes_deletion_and_detachment_fence_work(s):
    for action in ("change", "delete", "detach"):
        chosen = await s.file(cached=True, name=f"{action}.txt")
        task = await s.create([chosen])
        claimed = await s.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The source-fence Agent could not be claimed.")
        await s.store.agents.reserve_operation(**lease(claimed), step_id="plan")
        if action == "change":
            chosen.artifact.write_text("The selected saved text changed.", encoding="utf-8")
        elif action == "delete":
            response = await s.client.delete(f"/api/files/{chosen.id}")
            require(response.status_code == 200, "The source-fence fixture could not be deleted.")
        else:
            require(await s.store.projects.remove_file(user_id=s.owner.user_id, project_id=s.project.project_id,
                    file_id=chosen.id), "The source-fence fixture could not be detached.")
        current = await s.saved(task)
        require(not current["file_inputs"]["current"], "Changed selected files still appeared current.")
        try:
            await s.store.agents.checkpoint(**lease(claimed), step_id="plan", result="A late result that must never be saved.")
        except s.r.AgentTaskContextError:
            pass
        else:
            raise SmokeFailureError("A source change allowed a late checkpoint to be saved.")
        await s.store.agents.fail(**lease(claimed), error="Selected source unavailable.")
        await s.control(task, "rerun", status=409)
        saved = await s.saved(task)
        require(not saved["steps"][0]["result"] and not saved["result"], "A revoked file source left a late task result.")


async def bounded_retained_text_and_generated_labels(s):
    long_text = "Saved OCR clause. " * 3000
    scanned = await s.file(kind="image", name="scan.png", text=long_text, cached=True, reclaimed=True)
    speech = await s.file(kind="audio", name="meeting.mp3", text="Retained transcript of the task instructions.", reclaimed=True)
    generated = await s.file(name="working.md", text="Generated working output, not independent evidence.",
                             origin="generated", cached=True)
    task = await s.create([scanned, speech, generated])
    metadata = {item["file_id"]: item for item in task["file_inputs"]["files"]}
    require(metadata[scanned.id]["truncated"] and metadata[scanned.id]["chars"] < metadata[scanned.id]["total_chars"],
            "Oversized selected text was not disclosed as a bounded prefix.")
    require(metadata[speech.id]["artifact"] == "transcript" and metadata[generated.id]["origin"] == "generated",
            "Retained speech or generated working output lost its source label.")
    claimed = await s.store.agents.claim()
    require(claimed is not None and claimed["id"] == task["id"], "The retained-text Agent could not be claimed.")
    resolved = await s.service.resolve(claimed, lease(claimed))
    require(len(resolved) == 3 and all(s.r.token_count(item["text"]) <= 1500 for item in resolved),
            "Selected saved file text exceeded its per-file model context allowance.")
    require(sum(s.r.token_count(item["text"]) for item in resolved) <= 4500,
            "Selected saved file text exceeded the combined model context allowance.")
    bodies = {item["file_id"]: item["text"] for item in resolved}
    require(bodies[speech.id] == speech.text and bodies[generated.id] == generated.text
            and long_text.startswith(bodies[scanned.id]) and not scanned.path.exists() and not speech.path.exists(),
            "Retained saved text required deleted originals or failed to preserve its bounded prefix.")


CASES = (
    ("explicit_selection_and_metadata", explicit_selection_and_metadata),
    ("local_execution_uses_selected_text_only", local_execution_uses_selected_text_only),
    ("frozen_pins_restart_and_rerun", frozen_pins_restart_and_rerun),
    ("owner_private_membership_and_availability", owner_private_membership_and_availability),
    ("source_changes_deletion_and_detachment_fence_work", source_changes_deletion_and_detachment_fence_work),
    ("bounded_retained_text_and_generated_labels", bounded_retained_text_and_generated_labels),
)


async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-file-inputs-smoke-") as directory:
        for name, case in CASES:
            try:
                async with asyncio.timeout(timeout_seconds), Scenario(runtime, Path(directory) / name) as scenario:
                    await case(scenario)
            except TimeoutError:
                raise SmokeFailureError(f"{name} deadline exceeded.") from None
            checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable file input fixtures were not removed.")
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
        result = (130, "Agent file input smoke interrupted.")
    except BaseException:  # noqa: BLE001 - do not expose SQL, paths or private content
        result = (3, "Agent file input smoke harness crashed.")
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
        raise SmokeSetupError("Agent file input smoke requires Linux fork support.") from None
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-file-inputs-smoke-") as directory:
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
    require(not Path(directory).exists(), "Disposable file input fixtures were not removed.")
    return [*checks, "isolated_fixture_cleanup"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks = 0, []
    message = "Agent file input smoke passed; isolated temporary data and mocked local replies; no live accounts/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = run_processes(load_runtime(), args.timeout_seconds)
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Agent file input smoke interrupted."
    except Exception:  # noqa: BLE001 - private paths, SQL and content never enter terminal output
        code, message = 3, "Agent file input smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
