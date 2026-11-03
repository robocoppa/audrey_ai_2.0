"""Installed Agent execution history using disposable canonical state.

Real SQLite events, Agent API routes, executor dispatch and file downloads
run against temporary fixtures. Local replies are mocked; no live
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
        from audrey.app_state.agents import AgentsRepository, AgentTaskContextError, AgentTaskError
        from audrey.auth import require_principal
        from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
        from audrey.kb.storage_lifecycle import StorageLifecycle
        from audrey.kb.uploads_db import UploadsDB
        from audrey.kb.user_store import sanitize_user
        from audrey.models.health import HealthTracker
        from audrey.models.ollama import OllamaClient
        from audrey.pipeline.fair_gate import FairLocalGate
        from audrey.routes.app import router
        from audrey.routes.app.agents import AgentEventsResponse
        from audrey.routes.inflight import UserInflightRegistry
        from audrey.tools.discovery import ToolRegistry
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks Agent execution history; rebuild audrey.") from error
    if "items" not in AgentEventsResponse.model_fields or not hasattr(AgentsRepository, "events"):
        raise SmokeSetupError("Installed backend lacks Agent execution history; rebuild audrey.")
    return SimpleNamespace(
        httpx=httpx, FastAPI=FastAPI, AgentFileInputs=AgentFileInputs, AgentTaskError=AgentTaskError, AgentsRepository=AgentsRepository,
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



async def page(s, task, **params):
    response = await s.client.get(f"/api/agents/{task['id']}/events", params=params)
    require(response.status_code == 200, "The owned execution history could not be read.")
    return response.json()


async def execute(s, *, quota_failure=False):
    chosen = await s.file(text="SOURCE_BODY_MUST_NOT_BE_COPIED_TO_EVENTS", name="sensitive-filename.txt")
    task = await s.create([chosen], objective="OBJECTIVE_MUST_NOT_BE_COPIED_TO_EVENTS: create a checklist.")
    requests, clients, r = [], [], s.r
    output = "# FILE_BODY_MUST_NOT_BE_COPIED_TO_EVENTS"
    replies = ["Task Checklist", "Prepare the requested checklist.", None, "The work is ready.", "Final reviewed work."]
    if quota_failure:
        s.app.state.cfg.raw["kb"]["max_user_bytes"] = 1

    async def provider(request):
        if request.url.path == "/api/tags":
            return r.httpx.Response(200, json={"models": [{"name": "qwen3.8:latest", "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return r.httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
                "general.architecture": "qwen3", "general.parameter_count": 27000000000}})
        require(request.url.path == "/api/chat" and len(requests) < len(replies), "Unexpected model operation.")
        payload = json.loads(request.content)
        require(payload["model"] == "qwen3.8:latest", "Execution history changed local model selection.")
        index = len(requests)
        requests.append(payload)
        message = {"content": replies[index] or "", "thinking": "REASONING_MUST_NOT_BE_COPIED_TO_EVENTS"}
        if index == 2:
            message["tool_calls"] = [{"function": {"name": "create_file", "arguments": {
                "filename": "output.md", "format": "md", "content": output}}}]
        return r.httpx.Response(200, json={"done": True, "done_reason": "stop",
            "prompt_eval_count": 13, "eval_count": 7, "message": message})

    def factory(endpoint, **kwargs):
        client = r.OllamaClient(endpoint, transport=r.httpx.MockTransport(provider), **kwargs)
        clients.append(client)
        return client

    require(await r.AgentRuntime(s.store.agents, r.ResearchAgentExecutor(s.app, client_factory=factory)).run_once(),
            "The isolated execution history Agent did not run.")
    saved = await s.saved(task)
    require(saved["status"] == "completed", "The isolated execution history Agent did not complete.")
    require(len(requests) == 5 and all(c._client.is_closed for c in clients), "Execution repeated or leaked model work.")
    events = (await page(s, task))["items"]
    models = [item for item in events if item["kind"] == "model"]
    tools = [item for item in events if item["kind"] == "tool"]
    stages = [item for item in events if item["kind"] == "stage"]
    require(len(models) == 5 and all(item["state"] == "completed" for item in models),
            "Model calls were missing or reported inaccurate completion states.")
    require(all(item["input_tokens"] == 13 and item["output_tokens"] == 7 for item in models),
            "Model usage counts were not recorded from the actual reply transport.")
    require(len(tools) == 1 and tools[0]["tool"] == "create_file"
            and tools[0]["state"] == ("failed" if quota_failure else "completed"),
            "A tool outcome was missing or reported inaccurate success.")
    require(len(stages) == 3 and {item["stage"] for item in stages} == {"plan", "research", "review"}
            and all(item["state"] == "completed" for item in stages), "Stage checkpoints did not match execution history.")
    encoded = json.dumps(events)
    require(all(value not in encoded for value in [chosen.text, chosen.name, output,
        "OBJECTIVE_MUST_NOT_BE_COPIED_TO_EVENTS", "REASONING_MUST_NOT_BE_COPIED_TO_EVENTS", str(s.root)]),
        "Execution history copied source, prompt, reasoning, output or private path content.")
    require(all(set(item) == {"seq", "attempt", "stage", "kind", "state", "tool", "created_at", "finished_at",
        "duration_ms", "input_tokens", "output_tokens"} for item in events), "History returned unexpected metadata fields.")
    if quota_failure:
        require(not saved.get("outputs"), "Quota-refused output was still published.")
    else:
        require(len(saved.get("outputs", [])) == 1, "Observed file creation did not save an owned output.")
        download = await s.client.get(saved["outputs"][0]["download_url"])
        require(download.status_code == 200 and download.text == output, "Observed file creation returned different bytes.")
    return task, events


async def real_execution_events_and_metadata_only(s):
    await execute(s)


async def tool_failure_is_not_reported_as_success(s):
    await execute(s, quota_failure=True)


async def restart_reclaim_and_rerun_history(s):
    task = await s.create()
    claimed = await s.store.agents.claim()
    old_lease = lease(claimed)
    await s.store.agents.reserve_operation(**old_lease, step_id="plan")
    seq = await s.store.agents.start_event(**old_lease, stage="plan", kind="model")
    path = s.store.path
    s.store.close()
    s.store = s.r.ApplicationStore(path)
    s.app.state.application_store = s.store
    s.app.state.agent_file_inputs = None
    s.service = s.r.get_agent_file_inputs(s.app)
    require(next(item for item in (await page(s, task))["items"] if item["seq"] == seq)["state"] == "started",
            "Reopening durable storage lost the unresolved request.")
    with s.store._lock, s.store._conn:
        s.store._conn.execute("UPDATE app_agent_tasks SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE task_id=?", (task["id"],))
    resumed = await s.store.agents.claim()
    require(resumed and resumed["id"] == task["id"] and resumed["attempts"] == 2, "Expired work did not reclaim the same Agent.")
    require(next(item for item in (await page(s, task))["items"] if item["seq"] == seq)["state"] == "interrupted",
            "A reclaimed unresolved request was not marked interrupted.")
    try:
        await s.store.agents.finish_event(**old_lease, seq=seq, state="completed")
    except s.r.AgentTaskError:
        pass
    else:
        raise SmokeFailureError("An old attempt rewrote its interrupted history.")
    current_lease = lease(resumed)
    for stage in ("plan", "research", "review"):
        await s.store.agents.reserve_operation(**current_lease, step_id=stage)
        await s.store.agents.checkpoint(**current_lease, step_id=stage, result=f"Saved {stage}.")
    completed = await s.store.agents.complete(**current_lease, result="Reviewed work.")
    before = await page(s, task)
    repeated = await s.control(completed, "rerun")
    require(repeated["id"] == task["id"] and repeated["steps"][0]["result"] == "Saved plan."
            and (await page(s, task)) == before, "Run again discarded the Plan or rewrote prior history.")


async def cancel_fences_late_outcomes(s):
    task = await s.create()
    claimed = await s.store.agents.claim()
    active = lease(claimed)
    await s.store.agents.reserve_operation(**active, step_id="plan")
    seq = await s.store.agents.start_event(**active, stage="plan", kind="model")
    await s.control(task, "cancel")
    previous = await page(s, task)
    require(previous["items"] and all(item["state"] == "interrupted" for item in previous["items"]),
            "Cancelled calls were left running or reported as completed.")
    try:
        await s.store.agents.finish_event(**active, seq=seq, state="completed", output_tokens=99)
    except s.r.AgentTaskError:
        pass
    else:
        raise SmokeFailureError("A cancelled attempt recorded a late successful outcome.")
    require((await page(s, task)) == previous, "Late callbacks changed cancelled history.")


async def owner_pagination_and_revision_fences(s):
    task = await s.create()
    claimed = await s.store.agents.claim()
    active = lease(claimed)
    await s.store.agents.reserve_operation(**active, step_id="plan")
    before = await s.saved(task)
    for _ in range(3):
        seq = await s.store.agents.start_event(**active, stage="plan", kind="model")
        await s.store.agents.finish_event(**active, seq=seq, state="completed", duration_ms=1)
    require((await s.saved(task))["revision"] == before["revision"], "History updates invalidated task controls.")
    first = await page(s, task, limit=2)
    older = await page(s, task, before=first["items"][-1]["seq"], limit=2)
    require(first["has_more"] and not older["has_more"] and len(older["items"]) == 2
        and not {row["seq"] for row in first["items"]} & {row["seq"] for row in older["items"]},
        "History pagination duplicated or omitted its boundary.")
    s.current = s.other
    response = await s.client.get(f"/api/agents/{task['id']}/events")
    require(response.status_code == 404 and "items" not in response.json(), "Another account read event history.")
    s.current = s.owner
    await s.control(task, "cancel")
    cancelled = await s.saved(task)
    erased = await s.client.delete(f"/api/agents/{task['id']}", params={"expected_revision": cancelled["revision"]})
    require(erased.status_code == 200 and (await s.client.get(f"/api/agents/{task['id']}/events")).status_code == 404,
            "Task deletion retained readable execution history.")
    with s.store._lock:
        require(s.store._conn.execute("SELECT COUNT(*) FROM app_agent_events WHERE task_id=?", (task["id"],)).fetchone()[0] == 0,
                "Task deletion did not erase execution history rows.")


async def bounded_history_retention(s):
    task = await s.create()
    claimed = await s.store.agents.claim()
    active = lease(claimed)
    await s.store.agents.reserve_operation(**active, step_id="plan")
    for _ in range(205):
        seq = await s.store.agents.start_event(**active, stage="plan", kind="model")
        await s.store.agents.finish_event(**active, seq=seq, state="completed")
    rows = []
    cursor = None
    while True:
        current = await page(s, task, **({"before": cursor} if cursor else {}))
        require(current["truncated"], "Pruned history was not disclosed.")
        rows.extend(current["items"])
        if not current["has_more"]:
            break
        cursor = current["items"][-1]["seq"]
    require(len(rows) == 200 and len({item["seq"] for item in rows}) == 200,
            "Retained execution history exceeded its bound or duplicated rows.")
    # Evicting an old stage event must not prevent its real checkpoint.
    await s.store.agents.checkpoint(**active, step_id="plan", result="Completed despite old history pruning.")


CASES = (
    ("real_execution_events_and_metadata_only", real_execution_events_and_metadata_only),
    ("tool_failure_is_not_reported_as_success", tool_failure_is_not_reported_as_success),
    ("restart_reclaim_and_rerun_history", restart_reclaim_and_rerun_history),
    ("cancel_fences_late_outcomes", cancel_fences_late_outcomes),
    ("owner_pagination_and_revision_fences", owner_pagination_and_revision_fences),
    ("bounded_history_retention", bounded_history_retention),
)


async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-events-smoke-") as directory:
        for name, case in CASES:
            try:
                async with asyncio.timeout(timeout_seconds), Scenario(runtime, Path(directory) / name) as scenario:
                    await case(scenario)
            except TimeoutError:
                raise SmokeFailureError(f"{name} deadline exceeded.") from None
            checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable event fixtures were not removed.")
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
        result = (130, "Agent event smoke interrupted.")
    except BaseException:  # noqa: BLE001 - do not expose SQL, paths or private content
        result = (3, "Agent event smoke harness crashed.")
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
        raise SmokeSetupError("Agent event smoke requires Linux fork support.") from None
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-events-smoke-") as directory:
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
    require(not Path(directory).exists(), "Disposable event fixtures were not removed.")
    return [*checks, "isolated_fixture_cleanup"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks = 0, []
    message = "Agent event smoke passed; isolated temporary data and mocked local replies; no live accounts/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = run_processes(load_runtime(), args.timeout_seconds)
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Agent event smoke interrupted."
    except Exception:  # noqa: BLE001 - private paths, SQL and content never enter terminal output
        code, message = 3, "Agent event smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
