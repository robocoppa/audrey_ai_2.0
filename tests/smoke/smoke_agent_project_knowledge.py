"""Installed Agent Project source-pin checks using disposable canonical data.

This probe never opens the running server's account database, contacts a network
or starts a model. It streams into the installed backend with the on-box wrapper.
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


def load_runtime():
    try:
        from audrey.app_state import ApplicationStore, agent_projects
        from audrey.app_state.agents import AgentTaskContextError
        from audrey.content_sources import ContentSourceReader
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks Agent Project source pins; rebuild audrey.") from error
    if not {"message", "agent"}.issubset(getattr(agent_projects, "SUPPORTED_KNOWLEDGE_SOURCE_KINDS", ())):
        raise SmokeSetupError("Installed backend lacks Agent Project source pins; rebuild audrey.")
    return SimpleNamespace(
        ApplicationStore=ApplicationStore, AgentTaskContextError=AgentTaskContextError,
        ContentSourceReader=ContentSourceReader, agent_projects=agent_projects,
    )


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


class Scenario:
    def __init__(self, runtime, root):
        self.runtime, self.root = runtime, root
        root.mkdir(parents=True, exist_ok=True)
        self.store = runtime.ApplicationStore(root / "isolated-app.sqlite")
        self.counter = 0

    async def __aenter__(self):
        try:
            self.owner, self.other = [await self.store.resolve_external_identity(
                provider="agent-source-smoke", subject=name, email=f"{name}@example.com", display_name=name,
                role="user", auth_method="test",
            ) for name in ("alice", "bob")]
            self.project = await self.store.projects.create(
                user_id=self.owner.user_id, name="Isolated Agent source Project",
                instructions="Compare stable releases using approved Project notes.",
            )
            self.reader = self.runtime.ContentSourceReader(self.store, None, self.root / "unused-uploads")
            return self
        except BaseException:
            self.store.close()
            raise

    async def __aexit__(self, *_error):
        self.store.close()

    async def message(self, *, principal=None, project=None, private=False, status="completed"):
        self.counter += 1
        principal = principal or self.owner
        project = project or self.project
        message_id = f"msg_smoke_{self.counter}"
        conversation = await self.store.conversations.create(
            user_id=principal.user_id, title="Isolated saved source", project_id=project.project_id,
            private_required=private,
        )
        with self.store._lock, self.store._conn:
            self.store._conn.execute(
                "INSERT INTO app_messages (message_id,conversation_id,user_id,sequence_no,role,status,"
                "content,created_at,updated_at) VALUES (?,?,?,1,'user',?,?,?,?)",
                (message_id, conversation.conversation_id, principal.user_id, status,
                 f"CANONICAL_MESSAGE_BODY_{self.counter}_MUST_NOT_ENTER_AGENT_CONTEXT",
                 "2026-11-02T10:00:00+00:00", "2026-11-02T10:00:00+00:00"),
            )
        return message_id, conversation.conversation_id

    async def reference(self, source_id, *, principal=None):
        principal = principal or self.owner
        document = await self.reader.resolve(principal.user_id, principal.storage_namespace, source_id)
        require(document is not None and self.reader.is_current(document), "The canonical source fixture could not resolve.")
        return {"source_id": source_id, "revision": document.revision}

    async def note(self, text, *, sources=(), **changes):
        return await self.store.knowledge.create(
            user_id=self.owner.user_id, project_id=self.project.project_id,
            kind="decision", text=text, sources=sources, **changes,
        )

    async def task(self, *, project=True):
        self.counter += 1
        return await self.store.agents.create(
            user_id=self.owner.user_id, objective="Compare stable public releases.",
            request_id=f"source-smoke-{self.counter}",
            project_id=self.project.project_id if project else None,
        )

    async def claim(self, task):
        claimed = await self.store.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "The disposable Agent fixture could not be claimed.")
        return claimed

    async def report(self):
        task = await self.task(project=False)
        claimed = await self.claim(task)
        for step in ("plan", "research", "review"):
            await self.store.agents.reserve_operation(**lease(claimed), step_id=step)
            await self.store.agents.checkpoint(**lease(claimed), step_id=step, result=f"Saved isolated {step}.")
        return await self.store.agents.complete(
            **lease(claimed), result="CANONICAL_REPORT_BODY_MUST_NOT_ENTER_AGENT_CONTEXT",
        )

    def snapshot(self):
        queries = {
            "app_agent_tasks": "SELECT * FROM app_agent_tasks ORDER BY rowid",
            "app_agent_operations": "SELECT * FROM app_agent_operations ORDER BY rowid",
            "app_agent_notices": "SELECT * FROM app_agent_notices ORDER BY rowid",
            "app_agent_input_requests": "SELECT * FROM app_agent_input_requests ORDER BY rowid",
            "app_agent_project_contexts": "SELECT * FROM app_agent_project_contexts ORDER BY rowid",
        }
        with self.store._lock:
            return {table: [tuple(row) for row in self.store._conn.execute(query)] for table, query in queries.items()}

    async def context(self, task):
        return await self.store.agents.resolve_project_context(**lease(task))


async def approved_message_and_report_notes(s):
    message_id, _ = await s.message()
    incomplete_id, _ = await s.message(status="incomplete")
    report = await s.report()
    message_ref = await s.reference(f"message:{message_id}")
    report_ref = await s.reference(f"agent:{report['id']}:report")
    notes = [await s.note("Approved saved message summary.", sources=[message_ref]),
             await s.note("Approved completed report summary.", sources=[report_ref]),
             await s.note("An approved saved incomplete message summary.", sources=[await s.reference(f"message:{incomplete_id}")]),
             await s.note("An approved source-free preference.")]
    task = await s.task()
    require({pin["id"] for pin in task["project_context"]["knowledge"]} == {note["id"] for note in notes},
            "Approved current message/report notes were not selected alongside source-free notes.")
    claimed = await s.claim(task)
    context = await s.context(claimed)
    require({note["text"] for note in context["knowledge"]} == {note["text"] for note in notes},
            "Agent context changed the approved summary text.")
    with s.store._lock:
        manifest = dict(s.store._conn.execute("SELECT * FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone())
    persisted = json.dumps(manifest)
    require(all(note["text"] not in persisted for note in notes) and s.project.instructions not in persisted,
            "The Agent manifest copied approved notes or Project instruction bodies.")
    require("CANONICAL_MESSAGE_BODY" not in persisted and "CANONICAL_REPORT_BODY" not in persisted
            and "CANONICAL_MESSAGE_BODY" not in json.dumps(context) and "CANONICAL_REPORT_BODY" not in json.dumps(context),
            "A source body entered the manifest or model context.")
    require(sum(s.runtime.agent_projects.token_count(note["text"]) for note in context["knowledge"]) <= 1000,
            "Source-backed notes bypassed the Agent Knowledge token limit.")


async def unsupported_and_stale_sources_excluded(s):
    report = await s.report()
    message_id, _ = await s.message()
    valid_ref = await s.reference(f"message:{message_id}")
    valid = await s.note("The one eligible saved message summary.", sources=[valid_ref])
    private = await s.store.projects.create(user_id=s.owner.user_id, name="Protected fixture", private_required=True)
    private_message, _ = await s.message(project=private, private=True)
    foreign_project = await s.store.projects.create(user_id=s.other.user_id, name="Other owner fixture")
    foreign_message, _ = await s.message(principal=s.other, project=foreign_project)
    outside_project = await s.store.projects.create(user_id=s.owner.user_id, name="Outside fixture")
    outside_message, _ = await s.message(project=outside_project)
    stale_message, _ = await s.message()
    stale_ref = await s.reference(f"message:{stale_message}")
    with s.store._lock, s.store._conn:
        s.store._conn.execute("UPDATE app_messages SET content=?,updated_at=? WHERE message_id=?",
            ("A changed source body.", "2026-11-02T11:00:00+00:00", stale_message))
    excluded = [
        await s.note("Private evidence is excluded.", sources=[await s.reference(f"message:{private_message}")]),
        await s.note("Foreign evidence is excluded.", sources=[await s.reference(f"message:{foreign_message}", principal=s.other)]),
        await s.note("Outside Project messages are excluded.", sources=[await s.reference(f"message:{outside_message}")]),
        await s.note("Changed source revisions are excluded.", sources=[stale_ref]),
        await s.note("Missing sources are excluded.", sources=[{"source_id": "message:missing", "revision": "old"}]),
        await s.note("File-backed evidence remains excluded.", sources=[{"source_id": "file:missing:text", "revision": "old"}]),
        await s.note("Mixed file evidence remains wholly excluded.", sources=[valid_ref, {"source_id": "file:missing:text", "revision": "old"}]),
        await s.note("An unapproved proposal stays excluded.", sources=[await s.reference(f"agent:{report['id']}:report")],
                     origin="agent_report", source_task_id=report["id"]),
    ]
    task = await s.task()
    selected = {pin["id"] for pin in task["project_context"]["knowledge"]}
    require(selected == {valid["id"]} and not selected.intersection(note["id"] for note in excluded),
            "Agent capture admitted unsupported, private, foreign, stale or unapproved evidence.")


async def frozen_pins_survive_restart(s):
    message_id, _ = await s.message()
    report = await s.report()
    selected = await s.note("Restart keeps this approved summary and both source pins.", sources=[
        await s.reference(f"message:{message_id}"), await s.reference(f"agent:{report['id']}:report"),
    ])
    task = await s.task()
    before = s.snapshot()["app_agent_project_contexts"]
    second = s.runtime.ApplicationStore(s.store.path)
    try:
        claimed = await second.agents.claim()
        require(claimed is not None and claimed["id"] == task["id"], "A restarted backend lost the frozen source pins.")
        context = await second.agents.resolve_project_context(**lease(claimed))
        require(context["knowledge"][0]["id"] == selected["id"] and context["knowledge"][0]["text"] == selected["text"],
                "Restart changed the selected source-backed summary.")
        second.agents.fence_execution(**lease(claimed))
        with second._lock:
            after = [tuple(row) for row in second._conn.execute("SELECT * FROM app_agent_project_contexts ORDER BY rowid")]
        require(before == after, "Restart rewrote frozen source identities.")
    finally:
        second.close()


async def changed_source(s, mutation):
    if mutation.startswith("message"):
        message_id, conversation_id = await s.message()
        ref = await s.reference(f"message:{message_id}")
    else:
        report = await s.report()
        ref = await s.reference(f"agent:{report['id']}:report")
    await s.note("A selected approved source-backed summary.", sources=[ref])
    task = await s.task()
    claimed = await s.claim(task)
    for step in ("plan", "research", "review"):
        await s.store.agents.reserve_operation(**lease(claimed), step_id=step)
        await s.store.agents.checkpoint(**lease(claimed), step_id=step, result=f"Saved current {step}.")
    if mutation == "message_delete":
        await s.store.conversations.delete(user_id=s.owner.user_id, conversation_id=conversation_id)
    elif mutation == "message_detach":
        await s.store.conversations.update(user_id=s.owner.user_id, conversation_id=conversation_id,
                                           update_project=True, project_id=None)
    elif mutation == "message_edit":
        with s.store._lock, s.store._conn:
            s.store._conn.execute("UPDATE app_messages SET content=?,updated_at=? WHERE message_id=?",
                ("Changed after context use.", "2026-11-02T12:00:00+00:00", message_id))
    elif mutation == "report_rerun":
        await s.store.agents.control(user_id=s.owner.user_id, task_id=report["id"],
                                     expected_revision=report["revision"], action="rerun")
    else:
        await s.store.agents.delete(user_id=s.owner.user_id, task_id=report["id"], expected_revision=report["revision"])
    before = s.snapshot()
    try:
        s.store.agents.fence_execution(**lease(claimed))
    except s.runtime.AgentTaskContextError:
        pass
    else:
        raise SmokeFailureError("Changed selected evidence passed the final dispatch fence.")
    for operation in (
        lambda: s.store.agents.validate_lease(**lease(claimed)),
        lambda: s.store.agents.reserve_operation(**lease(claimed), step_id="review"),
        lambda: s.store.agents.checkpoint(**lease(claimed), step_id="review", result="A late checkpoint."),
        lambda: s.store.agents.request_input(**lease(claimed), question="A late question?"),
        lambda: s.store.agents.complete(**lease(claimed), result="A late report."),
    ):
        try:
            await operation()
        except s.runtime.AgentTaskContextError:
            pass
        else:
            raise SmokeFailureError("Changed selected evidence allowed late work to commit.")
    require(s.snapshot() == before, "A rejected source fence changed task results, operations, notices or context.")
    current = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    require(not current["project_context"]["current"], "Changed source pins were still advertised as current.")


async def selected_source_changes_fence_work(s):
    for mutation in ("message_edit", "message_delete", "message_detach", "report_rerun", "report_delete"):
        async with Scenario(s.runtime, s.root / mutation) as child:
            await changed_source(child, mutation)


async def harmless_changes_preserve_context(s):
    message_id, _ = await s.message()
    report = await s.report()
    chosen = await s.note("A frozen selected summary.", sources=[
        await s.reference(f"message:{message_id}"), await s.reference(f"agent:{report['id']}:report"),
    ])
    task = await s.task()
    await s.store.projects.update(user_id=s.owner.user_id, project_id=s.project.project_id, name="Renamed source Project")
    await s.note("A later note must not be silently selected.", pinned=True)
    await s.store.projects.add_file(user_id=s.owner.user_id, project_id=s.project.project_id, file_id="unselected-fixture-file")
    await s.message()
    current = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    require(current["project_context"]["current"] and current["project_context"]["project_name"] == "Renamed source Project",
            "A Project rename or unrelated source addition invalidated frozen evidence.")
    claimed = await s.claim(task)
    context = await s.context(claimed)
    require([note["id"] for note in context["knowledge"]] == [chosen["id"]], "Existing Agent work silently adopted later notes.")
    s.store.agents.fence_execution(**lease(claimed))


CASES = (
    ("approved_message_and_report_notes", approved_message_and_report_notes),
    ("unsupported_and_stale_sources_excluded", unsupported_and_stale_sources_excluded),
    ("frozen_pins_survive_restart", frozen_pins_survive_restart),
    ("selected_source_changes_fence_work", selected_source_changes_fence_work),
    ("harmless_changes_preserve_context", harmless_changes_preserve_context),
)


async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-agent-source-smoke-") as directory:
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
    message = "Agent Project Knowledge smoke passed; isolated temporary data; no credentials/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = asyncio.run(run(load_runtime(), args.timeout_seconds))
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Agent Project Knowledge smoke interrupted."
    except Exception:  # noqa: BLE001 - canonical text, SQL and private paths never enter output
        code, message = 3, "Agent Project Knowledge smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
