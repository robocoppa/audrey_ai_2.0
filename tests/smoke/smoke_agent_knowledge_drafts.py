"""Installed transient report-suggestion checks with isolated local fixtures.

The smoke constructs a separate ASGI application, disposable SQLite database and
real local-model client with an in-memory HTTP transport. It never opens the
running server's accounts, connects to a model or authenticates a live request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


class SmokeSetupError(Exception):
    pass


class SmokeFailureError(Exception):
    pass


def require(condition, explanation):
    if not condition:
        raise SmokeFailureError(explanation)


MODEL = "qwen3.8:latest"
SUGGESTIONS = [
    {"kind": "fact", "text": "The saved report describes version two as available."},
    {"kind": "question", "text": "Does the release support the required export format?"},
    {"kind": "action", "text": "Review compatibility before choosing a release."},
]


def load_runtime():
    try:
        import httpx
        from fastapi import FastAPI

        from audrey.agent_knowledge_drafts import ReportKnowledgeDrafts
        from audrey.app_state import ApplicationStore
        from audrey.auth import require_principal
        from audrey.content_sources import ContentSourceReader
        from audrey.knowledge_sources import KnowledgeSourceResolver
        from audrey.models.ollama import OllamaClient
        from audrey.pipeline.fair_gate import FairLocalGate
        from audrey.routes.app import router
        from audrey.routes.inflight import UserInflightRegistry
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks report suggestions; rebuild audrey.") from error
    return SimpleNamespace(
        httpx=httpx, FastAPI=FastAPI, ReportKnowledgeDrafts=ReportKnowledgeDrafts,
        ApplicationStore=ApplicationStore, require_principal=require_principal,
        ContentSourceReader=ContentSourceReader, KnowledgeSourceResolver=KnowledgeSourceResolver,
        OllamaClient=OllamaClient, FairLocalGate=FairLocalGate,
        UserInflightRegistry=UserInflightRegistry, router=router,
    )


class Scenario:
    def __init__(self, runtime, root):
        self.runtime, self.root = runtime, root
        self.store = runtime.ApplicationStore(root / "isolated-app.sqlite")
        self.client, self.clients = None, []
        self.calls, self.client_options = [], []
        self.requests = []
        self.on_chat = None
        self.response = self.reply(SUGGESTIONS)

    @staticmethod
    def reply(suggestions):
        return {"done": True, "done_reason": "stop", "eval_count": 40,
                "message": {"role": "assistant", "content": json.dumps({"suggestions": suggestions})}}

    async def __aenter__(self):
        try:
            return await self._initialize()
        except BaseException:
            await self.__aexit__()
            raise

    async def _initialize(self):
        r = self.runtime
        self.owner, self.other = [await self.store.resolve_external_identity(
            provider="draft-smoke", subject=name, email=f"{name}@example.com", display_name=name,
            role="user", auth_method="test",
        ) for name in ("alice", "bob")]
        self.current = self.owner
        self.project = await self.store.projects.create(
            user_id=self.owner.user_id, name="Isolated report Project",
            instructions="PROJECT_CONTEXT_MUST_NOT_ENTER_SUGGESTIONS",
        )
        await self.store.knowledge.create(
            user_id=self.owner.user_id, project_id=self.project.project_id, kind="decision",
            text="EXISTING_APPROVED_NOTE_MUST_NOT_ENTER_SUGGESTIONS", pinned=True,
        )
        self.task = await self.completed_task("primary")
        self.app = r.FastAPI()
        self.app.state.application_store = self.store
        self.app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}},
                                           env=SimpleNamespace(ollama_host="http://isolated-ollama.invalid"))
        self.app.state.gate = r.FairLocalGate()
        self.app.state.inflight = r.UserInflightRegistry(max_inflight_per_user=2)
        self.service = r.ReportKnowledgeDrafts(self.app, client_factory=self.client_factory)
        self.app.state.agent_knowledge_drafts = self.service
        self.app.state.knowledge_source_resolver = r.KnowledgeSourceResolver(
            self.store, r.ContentSourceReader(self.store, None, self.root / "unused-uploads"),
        )
        self.app.include_router(r.router)
        self.app.dependency_overrides[r.require_principal] = lambda: self.current
        self.client = r.httpx.AsyncClient(transport=r.httpx.ASGITransport(app=self.app), base_url="http://isolated")
        return self

    async def __aexit__(self, *_error):
        try:
            if self.client:
                await self.client.aclose()
        finally:
            try:
                for client in self.clients:
                    await client.aclose()
            finally:
                self.store.close()

    async def completed_task(self, request_id):
        task = await self.store.agents.create(
            user_id=self.owner.user_id, objective="Compare public release notes.", request_id=request_id,
            project_id=self.project.project_id,
        )
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The disposable completed report fixture could not be claimed.")
        lease = {"task_id": task["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
        for step in ("plan", "research", "review"):
            await self.store.agents.reserve_operation(**lease, step_id=step)
            await self.store.agents.checkpoint(**lease, step_id=step, result=f"Saved isolated {step}.")
        return await self.store.agents.complete(
            **lease, result="Version two is available. Export compatibility is not established; review it before deciding.",
        )

    def client_factory(self, endpoint, **kwargs):
        self.client_options.append((endpoint, kwargs))
        client = self.runtime.OllamaClient(endpoint, transport=self.runtime.httpx.MockTransport(self.provider), **kwargs)
        self.clients.append(client)
        return client

    async def provider(self, request):
        self.requests.append(request.url.path)
        require(request.url.host == "isolated-ollama.invalid", "The smoke attempted an unexpected model endpoint.")
        if request.url.path == "/api/tags":
            return self.runtime.httpx.Response(200, json={"models": [{"name": MODEL, "model": MODEL, "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return self.runtime.httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
                "general.architecture": "qwen3", "general.parameter_count": 27000000000,
            }})
        require(request.url.path == "/api/chat", "The suggestion helper attempted an unexpected operation.")
        self.calls.append(json.loads(request.content))
        if self.on_chat:
            await self.on_chat()
        return self.runtime.httpx.Response(200, json=self.response)

    async def post(self, *, task=None, revision=None, status=200):
        task = self.task if task is None else task
        expected = task["revision"] if revision is None else revision
        response = await self.client.post(f"/api/agents/{task['id']}/knowledge-drafts", json={"expected_revision": expected})
        require(response.status_code == status, f"Report suggestion contract expected HTTP {status}, received {response.status_code}.")
        return response

    def snapshot(self):
        tables = {
            "app_agent_tasks": "SELECT * FROM app_agent_tasks ORDER BY rowid",
            "app_agent_operations": "SELECT * FROM app_agent_operations ORDER BY rowid",
            "app_agent_notices": "SELECT * FROM app_agent_notices ORDER BY rowid",
            "app_project_knowledge": "SELECT * FROM app_project_knowledge ORDER BY rowid",
            "app_project_knowledge_revisions": "SELECT * FROM app_project_knowledge_revisions ORDER BY rowid",
            "app_agent_project_contexts": "SELECT * FROM app_agent_project_contexts ORDER BY rowid",
        }
        with self.store._lock:
            return {table: [tuple(row) for row in self.store._conn.execute(query)] for table, query in tables.items()}

    def assert_released(self):
        require(not self.service._active and all(client._client.is_closed for client in self.clients),
                "Suggestion cancellation or completion retained a client or active-account slot.")
        require(not self.app.state.inflight.pressure_snapshot()["in_use"]
                and not self.app.state.gate.pressure_snapshot()["in_use"],
                "Suggestion completion retained model admission capacity.")


async def transient_local_suggestions(s):
    before = s.snapshot()
    response = await s.post()
    s.generated = response.json()
    require(response.json() == {"task_id": s.task["id"], "task_revision": s.task["revision"], "suggestions": SUGGESTIONS},
            "The typed transient suggestion response changed its source report or content.")
    require(response.headers.get("cache-control") == "private, no-store", "Generated suggestions were cacheable.")
    require(len(s.calls) == 1 and s.requests.count("/api/tags") >= 2 and s.requests.count("/api/show") >= 2,
            "Suggestions skipped local identity verification or dispatched more than once.")
    request = s.calls[0]
    require(request["model"] == MODEL and request.get("think") is False and not request.get("tools")
            and request["format"].get("additionalProperties") is False, "Suggestions did not use the bounded local structured call.")
    require(json.loads(request["messages"][1]["content"]) == {"saved_report": s.task["result"]},
            "Suggestions included content beyond the selected saved report.")
    require(not any(marker in json.dumps(request) for marker in (
        "PROJECT_CONTEXT_MUST_NOT_ENTER_SUGGESTIONS", "EXISTING_APPROVED_NOTE_MUST_NOT_ENTER_SUGGESTIONS",
    )), "Project instructions or approved knowledge entered the suggestion prompt.")
    require(s.client_options == [("http://isolated-ollama.invalid", {"trust_env": False})],
            "Suggestion transport enabled environment proxy or fallback routing.")
    require(s.snapshot() == before, "Generating suggestions mutated saved tasks, operations, notices or knowledge.")
    s.assert_released()


async def owner_revision_completion_fences(s):
    initial = len(s.clients)
    s.current = s.other
    try:
        foreign = await s.post(status=404)
        missing = await s.post(task={"id": "missing-report", "revision": 1}, status=404)
        require(foreign.json() == missing.json(), "A foreign report was distinguishable from a missing one.")
    finally:
        s.current = s.owner
    await s.post(revision=s.task["revision"] - 1, status=409)
    deleted = await s.completed_task("deleted")
    await s.store.agents.delete(user_id=s.owner.user_id, task_id=deleted["id"], expected_revision=deleted["revision"])
    await s.post(task=deleted, status=404)
    queued = await s.store.agents.create(user_id=s.owner.user_id, objective="Unfinished fixture.", request_id="unfinished")
    await s.post(task=queued, status=409)
    require(len(s.clients) == initial, "An unavailable, stale or unfinished report reached model dispatch.")
    s.assert_released()


async def invalid_and_truncated_replies(s):
    before = s.snapshot()
    for reply in (
        s.reply([SUGGESTIONS[0], {"kind": "unsupported", "text": "No partial acceptance."}]),
        {**s.reply(SUGGESTIONS), "done_reason": "length"},
    ):
        s.response = reply
        response = await s.post(status=503)
        require(set(response.json()) == {"detail"} and "suggestions" not in response.json(),
                "Invalid or truncated output leaked a partially accepted suggestion batch.")
        require(s.snapshot() == before, "Rejected suggestions changed saved data.")
        s.assert_released()
    s.response = s.reply(SUGGESTIONS)


async def explicit_save_stays_an_unapproved_proposal(s):
    before = s.snapshot()
    approved = await s.store.knowledge.list_approved(user_id=s.owner.user_id, project_id=s.project.project_id)
    suggestion = s.generated["suggestions"][2]
    response = await s.client.post(f"/api/projects/{s.project.project_id}/knowledge/proposals", json={
        "task_id": s.generated["task_id"], "task_revision": s.generated["task_revision"],
        **suggestion, "request_id": "explicit-draft-proposal",
    })
    require(response.status_code == 201, "Explicitly saving a generated suggestion did not create a proposal.")
    entry = response.json()
    require(entry["approval_state"] == "proposed" and entry["approved_at"] is None
            and entry["work_status"] == "open" and not entry["pinned"]
            and entry["kind"] == suggestion["kind"] and entry["text"] == suggestion["text"]
            and entry["origin"] == "agent_report" and entry["source_task_id"] == s.task["id"],
            "Saving a draft approved knowledge, performed an action or lost its generated provenance.")
    require(len(entry["sources"]) == 1 and entry["sources"][0]["source_id"] == f"agent:{s.task['id']}:report"
            and bool(entry["sources"][0]["revision"]) and not entry["stale"],
            "The explicit proposal lost its saved report reference.")
    after = s.snapshot()
    require(all(after[table] == before[table] for table in (
        "app_agent_tasks", "app_agent_operations", "app_agent_notices", "app_agent_project_contexts",
    )), "Saving a suggestion changed its source task, operations, notices or context.")
    require(len(after["app_project_knowledge"]) == len(before["app_project_knowledge"]) + 1
            and len(after["app_project_knowledge_revisions"]) == len(before["app_project_knowledge_revisions"]) + 1
            and await s.store.knowledge.list_approved(user_id=s.owner.user_id, project_id=s.project.project_id) == approved,
            "Saving a suggestion changed approved Project context or created extra knowledge.")


async def changed_report_discards_late_suggestions(s):
    async def rerun():
        await s.store.agents.control(user_id=s.owner.user_id, task_id=s.task["id"],
                                     expected_revision=s.task["revision"], action="rerun")

    s.on_chat = rerun
    before = s.snapshot()
    response = await s.post(status=409)
    require(set(response.json()) == {"detail"}, "A changed report exposed a stale suggestion response.")
    after = s.snapshot()
    require(after["app_project_knowledge"] == before["app_project_knowledge"]
            and after["app_project_knowledge_revisions"] == before["app_project_knowledge_revisions"]
            and after["app_agent_operations"] == before["app_agent_operations"],
            "Discarding stale suggestions created knowledge or agent operations.")
    s.assert_released()


CASES = (
    ("verified_local_transient_suggestions", transient_local_suggestions),
    ("owner_revision_completion_fences", owner_revision_completion_fences),
    ("invalid_and_truncated_replies_rejected", invalid_and_truncated_replies),
    ("explicit_save_stays_an_unapproved_proposal", explicit_save_stays_an_unapproved_proposal),
    ("changed_report_discards_late_suggestions", changed_report_discards_late_suggestions),
)


async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-report-draft-smoke-") as directory:
        async with Scenario(runtime, Path(directory)) as scenario:
            for name, case in CASES:
                try:
                    async with asyncio.timeout(timeout_seconds):
                        await case(scenario)
                except TimeoutError:
                    raise SmokeFailureError(f"{name} exceeded its deadline.") from None
                checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable suggestion fixtures were not cleaned up.")
    checks.append("isolated_fixture_cleanup")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks, message = 0, [], "Report suggestion smoke passed; isolated temporary data and mocked local transport; no live wiring/models/accounts."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = asyncio.run(run(load_runtime(), args.timeout_seconds))
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Report suggestion smoke interrupted."
    except Exception:  # noqa: BLE001 - no private report, path or provider details in output
        code, message = 3, "Report suggestion smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
