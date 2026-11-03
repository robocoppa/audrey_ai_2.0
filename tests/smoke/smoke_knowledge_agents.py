"""Installed Knowledge-to-Agent checks using disposable canonical ASGI state.

This probe uses the installed routes and repository with separate temporary
accounts and SQLite data. General-task execution uses real runtime/client code
with mocked local model replies. It never reads live accounts, contacts a model
or connects to the network. Starting a task leaves its action/question open.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class SmokeSetupError(Exception):
    pass


class SmokeFailureError(Exception):
    pass


def require(condition, explanation):
    if not condition:
        raise SmokeFailureError(explanation)


def load_runtime():
    try:
        import httpx
        from fastapi import FastAPI

        from audrey.agent_executor import ResearchAgentExecutor
        from audrey.agent_runtime import AgentRuntime
        from audrey.app_state import ApplicationStore
        from audrey.app_state.agents import AgentTaskContextError
        from audrey.auth import require_principal
        from audrey.content_sources import ContentSourceReader
        from audrey.models.health import HealthTracker
        from audrey.models.ollama import OllamaClient
        from audrey.pipeline.fair_gate import FairLocalGate
        from audrey.routes.app import router
        from audrey.routes.app.agents import AgentCreateRequest
        from audrey.routes.inflight import UserInflightRegistry
        from audrey.scheduling import current_local_workload
        from audrey.tools.discovery import ToolRegistry, ToolSpec
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks Knowledge Agent origins; rebuild audrey.") from error
    if not {"knowledge_entry_id", "knowledge_entry_revision"}.issubset(AgentCreateRequest.model_fields):
        raise SmokeSetupError("Installed backend lacks Knowledge Agent origins; rebuild audrey.")
    return SimpleNamespace(
        httpx=httpx, FastAPI=FastAPI, ApplicationStore=ApplicationStore,
        AgentTaskContextError=AgentTaskContextError, require_principal=require_principal,
        ContentSourceReader=ContentSourceReader, router=router,
        ToolRegistry=ToolRegistry, ToolSpec=ToolSpec,
        ResearchAgentExecutor=ResearchAgentExecutor, AgentRuntime=AgentRuntime,
        HealthTracker=HealthTracker, OllamaClient=OllamaClient,
        FairLocalGate=FairLocalGate, UserInflightRegistry=UserInflightRegistry,
        current_local_workload=current_local_workload,
    )


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


class Scenario:
    def __init__(self, runtime, root):
        self.runtime, self.root = runtime, root
        self.store = runtime.ApplicationStore(root / "isolated-app.sqlite")
        self.counter = 0
        self.client = None
        self.wakes = []

    async def __aenter__(self):
        try:
            r = self.runtime
            self.owner, self.other = [await self.store.resolve_external_identity(
                provider="knowledge-agent-smoke", subject=name, email=f"{name}@example.com", display_name=name,
                role="user", auth_method="test",
            ) for name in ("alice", "bob")]
            self.current = self.owner
            self.project = await self.store.projects.create(
                user_id=self.owner.user_id, name="Isolated Knowledge work",
                instructions="Use approved constraints to prepare practical deliverables and preserve open questions.",
            )
            self.app = r.FastAPI()
            self.app.state.application_store = self.store
            self.app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}})
            self.app.state.tools = r.ToolRegistry(by_name={name: r.ToolSpec(
                name=name, description=name, parameters={"type": "object"},
                server_url="http://unused-tools.invalid", path=f"/{name}",
            ) for name in ("web_search", "web_fetch")})
            self.app.state.agent_runtime = SimpleNamespace(wake=lambda: self.wakes.append(True))
            self.app.include_router(r.router)
            self.app.dependency_overrides[r.require_principal] = lambda: self.current
            self.client = r.httpx.AsyncClient(
                transport=r.httpx.ASGITransport(app=self.app), base_url="http://isolated",
            )
            return self
        except BaseException:
            await self.__aexit__()
            raise

    async def __aexit__(self, *_error):
        try:
            if self.client:
                await self.client.aclose()
        finally:
            self.store.close()

    async def note(self, *, kind="action", text="Review export compatibility.", project=None, principal=None, **changes):
        return await self.store.knowledge.create(
            user_id=(principal or self.owner).user_id,
            project_id=(project or self.project).project_id, kind=kind, text=text, **changes,
        )

    async def create(self, note=None, *, status=201, **changes):
        self.counter += 1
        body = {"objective": "Prepare a release compatibility checklist.", "request_id": f"origin-smoke-{self.counter}",
                "project_id": self.project.project_id}
        if note:
            body.update(knowledge_entry_id=note["id"], knowledge_entry_revision=note["revision"])
        body.update(changes)
        response = await self.client.post("/api/agents", json=body)
        require(response.status_code == status,
                f"Knowledge Agent contract expected HTTP {status}, received {response.status_code}.")
        return response

    async def complete(self, task):
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The disposable Agent fixture could not be claimed.")
        for step in ("plan", "research", "review"):
            await self.store.agents.reserve_operation(**lease(claimed), step_id=step)
            await self.store.agents.checkpoint(**lease(claimed), step_id=step, result=f"Saved isolated {step}.")
        return await self.store.agents.complete(**lease(claimed), result="Saved isolated compatibility report.")

    async def message_ref(self, *, project=None, private=False):
        self.counter += 1
        conversation = await self.store.conversations.create(
            user_id=self.owner.user_id, title="Disposable source", project_id=(project or self.project).project_id,
            private_required=private,
        )
        message_id = f"msg_origin_{self.counter}"
        with self.store._lock, self.store._conn:
            self.store._conn.execute(
                "INSERT INTO app_messages (message_id,conversation_id,user_id,sequence_no,role,status,"
                "content,created_at,updated_at) VALUES (?,?,?,1,'user','completed',?,?,?)",
                (message_id, conversation.conversation_id, self.owner.user_id,
                 "SOURCE_BODY_MUST_NOT_BE_COPIED_TO_AGENT", "2026-11-02T10:00:00+00:00", "2026-11-02T10:00:00+00:00"),
            )
        reader = self.runtime.ContentSourceReader(self.store, None, self.root / "unused-uploads")
        source = await reader.resolve(self.owner.user_id, self.owner.storage_namespace, f"message:{message_id}")
        require(source is not None, "The disposable source fixture could not resolve.")
        return {"source_id": source.source_id, "revision": source.revision}, message_id

    def snapshot(self, *tables):
        queries = {
            "app_project_knowledge": "SELECT * FROM app_project_knowledge ORDER BY rowid",
            "app_project_knowledge_revisions": "SELECT * FROM app_project_knowledge_revisions ORDER BY rowid",
            "app_agent_tasks": "SELECT * FROM app_agent_tasks ORDER BY rowid",
            "app_agent_operations": "SELECT * FROM app_agent_operations ORDER BY rowid",
        }
        with self.store._lock:
            return {table: [tuple(row) for row in self.store._conn.execute(queries[table])]
                    for table in tables}

    def manifest(self, task):
        with self.store._lock:
            return dict(self.store._conn.execute(
                "SELECT * FROM app_agent_project_contexts WHERE task_id=?", (task["id"],),
            ).fetchone())

    async def restart(self):
        path = self.store.path
        self.store.close()
        self.store = self.runtime.ApplicationStore(path)
        self.app.state.application_store = self.store


async def explicit_create_preserves_open_work(s):
    for kind in ("action", "question"):
        note = await s.note(kind=kind)
        before = s.snapshot("app_project_knowledge", "app_project_knowledge_revisions")
        operations = s.snapshot("app_agent_operations")
        response = await s.create(note)
        task = response.json()
        require(response.headers.get("cache-control") == "private, no-store", "Agent creation was cacheable.")
        require(task["status"] == "queued" and task["attempts"] == 0 and not task["result"],
                "Explicit start did not leave new work queued.")
        require(task["project_context"].get("origin_entry") == {"id": note["id"], "revision": note["revision"]},
                "The Agent response lost its originating Knowledge revision.")
        require(s.snapshot("app_agent_operations") == operations,
                "Creating a task unexpectedly executed model work.")
        await s.complete(task)
        require(s.snapshot("app_project_knowledge", "app_project_knowledge_revisions") == before,
                "Starting or completing a task changed or closed the original Knowledge entry.")
    require(len(s.wakes) == 2, "Explicit starts did not wake the existing Agent runtime exactly once each.")


async def origin_priority_and_metadata_only(s):
    reference, _ = await s.message_ref()
    origin = await s.note(text="Check export compatibility before deciding.", sources=[reference])
    for index in range(14):
        await s.note(kind="decision", text=f"Pinned unrelated decision {index}.", pinned=True)
    task = (await s.create(origin)).json()
    pins = task["project_context"]["knowledge"]
    require(pins[0] == {"id": origin["id"], "revision": origin["revision"]} and len(pins) <= 12,
            "Pinned notes crowded out the explicitly selected Knowledge origin.")
    manifest = s.manifest(task)
    saved = json.loads(manifest["knowledge_json"])
    require(saved[0].get("origin") is True and sum(pin.get("origin") is True for pin in saved) == 1,
            "The frozen manifest did not retain exactly one origin marker.")
    persisted = json.dumps(manifest)
    require(origin["text"] not in persisted and s.project.instructions not in persisted
            and reference["source_id"] not in persisted and "SOURCE_BODY" not in persisted,
            "The origin manifest copied canonical Knowledge, Project or source bodies/references.")
    require(len(manifest["knowledge_json"]) <= 4000, "Origin metadata exceeded the existing manifest allowance.")
    claimed = await s.store.agents.claim()
    context = await s.store.agents.resolve_project_context(**lease(claimed))
    require(context["knowledge"][0]["id"] == origin["id"] and context["knowledge"][0]["text"] == origin["text"]
            and "SOURCE_BODY" not in json.dumps(context), "Agent context omitted the selected summary or copied source text.")


async def invalid_origins_and_source_fences(s):
    note = await s.note()
    malformed = [
        {"knowledge_entry_revision": None}, {"knowledge_entry_id": None},
        {"knowledge_entry_revision": True}, {"knowledge_entry_revision": "1"},
        {"knowledge_entry_revision": 0}, {"project_id": None},
    ]
    for changes in malformed:
        await s.create(note, status=422, **changes)
    foreign_project = await s.store.projects.create(user_id=s.other.user_id, name="Other owner")
    foreign = await s.note(principal=s.other, project=foreign_project)
    foreign_response = await s.create(foreign, status=404)
    missing = await s.create({"id": "missing", "revision": 1}, status=404)
    require(foreign_response.json() == missing.json() and foreign["text"] not in foreign_response.text,
            "Foreign Knowledge origins leaked content or a distinct error.")
    private = await s.store.projects.create(user_id=s.owner.user_id, name="Private fixture", private_required=True)
    private_note = await s.note(project=private)
    await s.create(private_note, project_id=private.project_id, status=404)
    fact = await s.note(kind="fact")
    await s.create(fact, status=404)
    closed = await s.store.knowledge.update(user_id=s.owner.user_id, project_id=s.project.project_id,
        entry_id=note["id"], expected_revision=note["revision"], work_status="completed")
    await s.create(closed, status=404)
    changed = await s.note()
    await s.store.knowledge.update(user_id=s.owner.user_id, project_id=s.project.project_id,
        entry_id=changed["id"], expected_revision=changed["revision"], text="Changed current open action.")
    await s.create(changed, status=409)
    reference, message_id = await s.message_ref()
    stale = await s.note(sources=[reference])
    with s.store._lock, s.store._conn:
        s.store._conn.execute("UPDATE app_messages SET content=?,updated_at=? WHERE message_id=?",
                             ("Changed source body.", "2026-11-02T11:00:00+00:00", message_id))
    await s.create(stale, status=404)
    file_note = await s.note(sources=[{"source_id": "file:missing:text", "revision": "old"}])
    await s.create(file_note, status=404)
    require(not (await s.store.agents.list(user_id=s.owner.user_id)) and not s.wakes,
            "An invalid origin left Agent rows or woke execution.")


async def idempotency_rename_and_response_fence(s):
    origin = await s.note()
    first = (await s.create(origin, request_id="stable-origin-request")).json()
    await s.store.projects.update(user_id=s.owner.user_id, project_id=s.project.project_id, name="Renamed Knowledge work")
    replay = (await s.create(origin, request_id="stable-origin-request")).json()
    require(first["id"] == replay["id"] and replay["project_context"]["project_name"] == "Renamed Knowledge work"
            and replay["project_context"]["current"], "Idempotent retry lost its Agent or treated a Project rename as stale.")
    other = await s.note(kind="question", text="Which export format is required?")
    await s.create(other, request_id="stable-origin-request", status=409)
    await s.store.knowledge.delete(user_id=s.owner.user_id, project_id=s.project.project_id,
                                  entry_id=origin["id"], expected_revision=origin["revision"])
    replay = (await s.create(origin, request_id="stable-origin-request")).json()
    require(replay["id"] == first["id"] and not replay["project_context"]["current"],
            "Retry created a duplicate task or silently refreshed its deleted origin.")
    require(len(await s.store.agents.list(user_id=s.owner.user_id)) == 1,
            "Idempotent retries created more than one Agent.")
    original_create = s.store.agents.create

    async def changed_before_response(**kwargs):
        result = await original_create(**kwargs)
        await s.store.knowledge.update(user_id=s.owner.user_id, project_id=s.project.project_id,
            entry_id=other["id"], expected_revision=other["revision"], text="A changed question after the write.")
        return result

    s.store.agents.create = changed_before_response
    wakes = len(s.wakes)
    try:
        await s.create(other, status=409)
    finally:
        s.store.agents.create = original_create
    require(len(s.wakes) == wakes, "A changed origin passed the final response fence and woke execution.")


async def restart_rerun_and_origin_changes_fence_work(s):
    for mutation in ("change", "delete"):
        origin = await s.note(text=f"Open action for the {mutation} fence.")
        task = await s.complete((await s.create(origin)).json())
        manifest = s.manifest(task)
        await s.restart()
        rerun = await s.store.agents.control(user_id=s.owner.user_id, task_id=task["id"],
                                            expected_revision=task["revision"], action="rerun")
        require(rerun["id"] == task["id"] and rerun["steps"][0]["result"] == task["steps"][0]["result"]
                and s.manifest(rerun) == manifest, "Restart or Run again replaced the Agent, Plan or frozen Knowledge origin.")
        claimed = await s.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "Restart lost the existing Agent run.")
        if mutation == "change":
            await s.store.knowledge.update(user_id=s.owner.user_id, project_id=s.project.project_id,
                entry_id=origin["id"], expected_revision=origin["revision"], text="Changed action after dispatch.")
        else:
            await s.store.knowledge.delete(user_id=s.owner.user_id, project_id=s.project.project_id,
                entry_id=origin["id"], expected_revision=origin["revision"])
        stale = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
        require(not stale["project_context"]["current"] and stale["project_context"].get("origin_entry") == {
            "id": origin["id"], "revision": origin["revision"],
        }, "Changing or deleting the origin rewrote the saved link instead of invalidating it.")
        before = s.snapshot("app_agent_tasks", "app_agent_operations")
        try:
            s.store.agents.fence_execution(**lease(claimed))
        except s.runtime.AgentTaskContextError:
            pass
        else:
            raise SmokeFailureError("A changed or deleted origin passed the final dispatch fence.")
        try:
            await s.store.agents.checkpoint(**lease(claimed), step_id="research", result="Forbidden stale result.")
        except s.runtime.AgentTaskContextError:
            pass
        else:
            raise SmokeFailureError("A changed or deleted origin committed a stale result.")
        require(s.snapshot("app_agent_tasks", "app_agent_operations") == before,
                "Rejected stale work changed saved Agent state.")
        await s.store.agents.control(user_id=s.owner.user_id, task_id=task["id"],
                                    expected_revision=stale["revision"], action="cancel")


async def general_task_without_web(s):
    r = s.runtime
    # General work remains available even when optional web tools are absent.
    s.app.state.tools = r.ToolRegistry()
    origin = await s.note(text="Write a release readiness checklist using the approved Project constraints.")
    before = s.snapshot("app_project_knowledge", "app_project_knowledge_revisions")
    task = (await s.create(origin, objective=(
        "Turn the approved constraints into a concise launch checklist. No external research is needed."
    ))).json()
    s.app.state.cfg = SimpleNamespace(
        raw={"passthrough": {"enabled": False}, "agent_tasks": {"max_tokens_per_call": 128}},
        env=SimpleNamespace(ollama_host="http://isolated-ollama.invalid"),
    )
    s.app.state.health = r.HealthTracker()
    s.app.state.gate = r.FairLocalGate()
    s.app.state.inflight = r.UserInflightRegistry(max_inflight_per_user=2)
    replies = {
        "title": "Release Readiness Checklist",
        "plan": "Draft a checklist using the saved Project constraints, then review it.",
        "research": "Confirm scope, validate a build, and record the release decision.",
        "review": "Launch checklist:\n- Confirm scope.\n- Validate a build.\n- Record the release decision.",
    }
    stages = list(replies)
    calls, metadata_calls, clients, client_options, unexpected_requests = [], [], [], [], []

    async def provider(request):
        require(request.url.host == "isolated-ollama.invalid", "The task attempted an unexpected model endpoint.")
        if request.url.path == "/api/tags":
            metadata_calls.append("tags")
            return r.httpx.Response(200, json={"models": [{
                "name": "qwen3.8:latest", "model": "qwen3.8:latest", "digest": "a" * 64,
            }]})
        if request.url.path == "/api/show":
            metadata_calls.append("show")
            return r.httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
                "general.architecture": "qwen3", "general.parameter_count": 27000000000,
            }})
        require(request.url.path == "/api/chat", "The task attempted an unexpected model operation.")
        require(len(calls) < len(stages), "The general task dispatched an unexpected extra model call.")
        payload = json.loads(request.content)
        stage = stages[len(calls)]
        calls.append(payload)
        require(payload["model"] == "qwen3.8:latest", "A task step dispatched a nonlocal or unexpected model.")
        require(r.current_local_workload() == "background"
                and s.app.state.gate.pressure_snapshot()["in_use"] == 1
                and s.app.state.inflight.snapshot().get(s.owner.storage_namespace) == 1,
                "A general task model call bypassed shared background admission.")
        return r.httpx.Response(200, json={
            "done": True, "done_reason": "stop", "eval_count": 7,
            "message": {"role": "assistant", "content": replies[stage]},
        })

    async def forbidden(request):
        unexpected_requests.append(request.url.path)
        raise SmokeFailureError("The general task attempted an unexpected web or network request.")

    original_client = r.httpx.AsyncClient

    def guarded_client(*args, **kwargs):
        # The real ReAct runner creates its own tool client. Deny every request
        # unless this fixture explicitly supplies an in-memory transport.
        if kwargs.get("transport") is None:
            kwargs["transport"] = r.httpx.MockTransport(forbidden)
        return original_client(*args, **kwargs)

    def client_factory(endpoint, **kwargs):
        client_options.append((endpoint, kwargs))
        client = r.OllamaClient(endpoint, transport=r.httpx.MockTransport(provider), **kwargs)
        clients.append(client)
        return client

    try:
        with patch.object(r.httpx, "AsyncClient", guarded_client):
            executor = r.ResearchAgentExecutor(s.app, client_factory=client_factory)
            worked = await r.AgentRuntime(s.store.agents, executor).run_once()
        current = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
        require(worked and current["status"] == "completed" and current["result"] == replies["review"],
                "A general task without web sources did not complete through the real local executor.")
        require(current["title"] == replies["title"] and [step["result"] for step in current["steps"]] == [
            replies["plan"], replies["research"], replies["review"],
        ], "The general task changed the saved title, Plan, Work or Review outputs.")
        require(len(calls) == 4 and metadata_calls.count("tags") >= len(calls)
                and metadata_calls.count("show") >= len(calls) and not unexpected_requests,
                "The general task skipped local verification, called tools or added unexpected model work.")
        require(client_options == [("http://isolated-ollama.invalid", {"trust_env": False})] * 3,
                "A general task used a fallback or unexpected model client.")
        require(all(origin["text"] in json.dumps(call["messages"]) for call in calls[1:])
                and replies["plan"] in json.dumps(calls[2]["messages"])
                and replies["research"] in json.dumps(calls[3]["messages"]),
                "General task execution lost its approved origin or saved work between stages.")
        require(current["project_context"].get("origin_entry") == {"id": origin["id"], "revision": origin["revision"]}
                and s.snapshot("app_project_knowledge", "app_project_knowledge_revisions") == before,
                "Completing a general task changed or closed its original Knowledge entry.")
        require(all(client._client.is_closed for client in clients)
                and not s.app.state.gate.pressure_snapshot()["in_use"]
                and not s.app.state.inflight.pressure_snapshot()["in_use"],
                "General task completion retained a model client or admission permit.")
    finally:
        for client in clients:
            await client.aclose()


CASES = (
    ("explicit_create_preserves_open_work", explicit_create_preserves_open_work),
    ("origin_priority_and_metadata_only", origin_priority_and_metadata_only),
    ("invalid_origins_and_source_fences", invalid_origins_and_source_fences),
    ("idempotency_rename_and_response_fence", idempotency_rename_and_response_fence),
    ("restart_rerun_and_origin_changes_fence_work", restart_rerun_and_origin_changes_fence_work),
    ("general_task_without_web", general_task_without_web),
)


async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-knowledge-agent-smoke-") as directory:
        for name, case in CASES:
            try:
                async with asyncio.timeout(timeout_seconds), Scenario(runtime, Path(directory) / name) as scenario:
                    await case(scenario)
            except TimeoutError:
                raise SmokeFailureError(f"{name} deadline exceeded.") from None
            checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable smoke fixtures were not cleaned up.")
    checks.append("isolated_fixture_cleanup")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks = 0, []
    message = "Knowledge Agent smoke passed; isolated temporary ASGI data and mocked local replies; no live accounts/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = asyncio.run(run(load_runtime(), args.timeout_seconds))
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Knowledge Agent smoke interrupted."
    except Exception:  # noqa: BLE001 - canonical text, SQL and private paths never enter output
        code, message = 3, "Knowledge Agent smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
