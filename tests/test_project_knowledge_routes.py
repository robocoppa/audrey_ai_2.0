"""Owned knowledge APIs exercise real SQLite state and revocable local sources."""

from __future__ import annotations

import inspect
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.content_search import ContentSearchError, ContentSearchIndex, SearchDocument
from audrey.content_search_service import SOURCE_KINDS, ContentSearchService
from audrey.content_sources import ContentSourceReader
from audrey.kb.uploads_db import UploadsDB
from audrey.knowledge_sources import KnowledgeSourceResolver
from audrey.routes.app import router


class Sources:
    def __init__(self):
        self.documents = {}
        self.on_resolve = None
        self.calls = []

    async def resolve(self, owner, namespace, source_id):
        self.calls.append((owner, namespace, source_id))
        if self.on_resolve:
            result = self.on_resolve(source_id)
            if inspect.isawaitable(result):
                await result
        document = self.documents.get(source_id)
        return document if document is not None and document.owner_id == owner else None

    def is_current(self, document):
        return self.documents.get(document.source_id) == document


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [
        await store.resolve_external_identity(
            provider="test",
            subject=name,
            email=f"{name}@example.com",
            display_name=name,
            role="user",
            auth_method="test",
        )
        for name in ("alice", "bob")
    ]
    projects = [
        await store.projects.create(user_id=owner.user_id, name=f"{owner.display_name} work")
        for owner in owners
    ]
    current = [owners[0]]
    sources = Sources()
    app = FastAPI()
    app.state.application_store = store
    app.state.knowledge_source_resolver = KnowledgeSourceResolver(store, sources)
    app.include_router(router)

    async def authenticate():
        if current[0] is None:
            raise HTTPException(401, "Sign in required.")
        return current[0]

    app.dependency_overrides[require_principal] = authenticate
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield SimpleNamespace(
            store=store,
            owners=owners,
            projects=projects,
            current=current,
            sources=sources,
            app=app,
            client=client,
            path=f"/api/projects/{projects[0].project_id}/knowledge",
        )
    store.close()


async def _create(state, **changes):
    payload = {
        "kind": "decision",
        "text": "Keep the orchard gate open.",
        "request_id": uuid.uuid4().hex,
    }
    payload.update(changes)
    return await state.client.post(state.path, json=payload)


def _document(state, source_id="message:source", **changes):
    value = SearchDocument(
        owner_id=state.owners[0].user_id,
        source_id=source_id,
        kind="message",
        revision="v1",
        title="Saved native message",
        text="Evidence only in local canonical storage.",
        updated_at="2026-10-29T12:00:00Z",
        project_id=state.projects[0].project_id,
        target={"kind": "message", "conversation_id": "con", "message_id": "source", "sequence": 1},
    )
    value = replace(value, **changes)
    state.sources.documents[source_id] = value
    return value


def _refs(document):
    return [{"source_id": document.source_id, "revision": str(document.revision)}]


async def _report(state, **changes):
    task = await state.store.agents.create(
        user_id=state.owners[0].user_id,
        objective="Inspect public orchard gate guidance.",
        request_id=uuid.uuid4().hex,
    )
    task = await state.store.agents.claim()
    lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
    for step in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**lease, step_id=step)
        await state.store.agents.checkpoint(**lease, step_id=step, result="Saved checkpoint.")
    task = await state.store.agents.complete(**lease, result="Public report about gates.")
    _document(
        state,
        f"agent:{task['id']}:report",
        kind="agent",
        project_id=None,
        revision=str(task["revision"]),
        target={"kind": "agent", "task_id": task["id"]},
    )
    payload = {
        "kind": "decision",
        "text": "Review whether the gate should close at dusk.",
        "task_id": task["id"],
        "task_revision": task["revision"],
        "request_id": uuid.uuid4().hex,
    }
    payload.update(changes)
    return task, payload


async def test_manual_crud_history_explicit_states_and_deletion(state):
    response = await _create(state, kind="action", pinned=True)
    assert response.status_code == 201, response.text
    assert response.headers["cache-control"] == "private, no-store"
    entry = response.json()
    assert entry["approval_state"] == "approved" and not entry["stale"]
    assert entry["work_status"] == "open" and entry["sources"] == []
    assert not {"owner_id", "user_id", "request_digest"} & entry.keys()
    path = f"{state.path}/{entry['id']}"
    updated = await state.client.patch(
        path, json={"expected_revision": 1, "text": "Close at dusk.", "work_status": "completed"}
    )
    assert updated.status_code == 200 and updated.json()["revision"] == 2
    assert (
        await state.client.patch(path, json={"expected_revision": 1, "text": "Stale edit."})
    ).status_code == 409
    assert (await state.client.get(path)).json()["text"] == "Close at dusk."
    history = await state.client.get(f"{path}/history")
    assert [item["text"] for item in history.json()["items"]] == ["Close at dusk.", entry["text"]]
    assert (await state.client.delete(path, params={"expected_revision": 1})).status_code == 409
    deleted = await state.client.delete(path, params={"expected_revision": 2})
    assert deleted.status_code == 200 and deleted.json() == {"deleted": True}
    assert (await state.client.get(path)).status_code == 404
    assert (await state.client.get(f"{path}/history")).status_code == 404


async def test_auth_scope_foreign_project_and_foreign_entry_are_fenced(state):
    entry = (await _create(state)).json()
    state.current[0] = None
    assert (await state.client.get(state.path)).status_code == 401
    state.current[0] = replace(state.owners[0], auth_method="personal_token", scopes=frozenset())
    assert (await state.client.get(state.path)).status_code == 403
    state.current[0] = state.owners[1]
    assert (await state.client.get(state.path)).status_code == 404
    for entry_id in (entry["id"], "missing"):
        assert (await state.client.get(f"{state.path}/{entry_id}")).status_code == 404
        assert (
            await state.client.patch(
                f"{state.path}/{entry_id}", json={"expected_revision": 1, "text": "No."}
            )
        ).status_code == 404
    assert (await _create(state)).status_code == 404


async def test_owned_source_preview_is_bounded_plain_text_and_exact_revision(state):
    doc = _document(state, text="<script>local text, never executable</script>" + "x" * 5000)
    entry = (await _create(state, sources=_refs(doc))).json()
    assert entry["sources"] == _refs(doc) and not entry["stale"]
    assert doc.text not in str(entry)
    path = f"{state.path}/{entry['id']}/sources/0"
    opened = await state.client.get(path, params={"expected_revision": 1})
    assert opened.status_code == 200, opened.text
    assert opened.headers["cache-control"] == "private, no-store"
    assert opened.json()["text"] == doc.text[:4000] and opened.json()["next_offset"] == 4000
    tail = await state.client.get(path, params={"expected_revision": 1, "offset": 4000})
    assert tail.json()["text"] == doc.text[4000:]
    assert (await state.client.get(path, params={"expected_revision": 2})).status_code == 409
    assert (
        await state.client.get(path, params={"expected_revision": 1, "offset": 499999})
    ).status_code == 422
    assert (
        await state.client.get(
            path.replace("sources/0", "sources/9"), params={"expected_revision": 1}
        )
    ).status_code == 404


@pytest.mark.parametrize("change", ["foreign", "detached", "private", "revision", "missing"])
async def test_new_references_require_owned_current_project_accessible_sources(state, change):
    doc = _document(state)
    if change == "foreign":
        _document(state, owner_id=state.owners[1].user_id)
    elif change == "detached":
        _document(state, project_id=None)
    elif change == "private":
        _document(state, private_required=True)
    elif change == "revision":
        _document(state, revision="v2")
    else:
        state.sources.documents.clear()
    response = await _create(state, sources=_refs(doc))
    assert response.status_code == 409, response.text
    assert (await state.client.get(state.path)).json()["items"] == []


async def test_private_project_accepts_owned_private_local_text_without_remote_dispatch(state):
    project = await state.store.projects.create(
        user_id=state.owners[0].user_id, name="Local Private", private_required=True
    )
    state.path = f"/api/projects/{project.project_id}/knowledge"
    doc = _document(state, project_id=project.project_id, private_required=True)
    response = await _create(state, sources=_refs(doc))
    assert response.status_code == 201 and not response.json()["stale"]
    opened = await state.client.get(
        f"{state.path}/{response.json()['id']}/sources/0", params={"expected_revision": 1}
    )
    assert opened.status_code == 200 and opened.json()["private_required"]
    assert [call[2] for call in state.sources.calls] == [doc.source_id] * 3


@pytest.mark.parametrize("change", ["delete", "detach", "revision"])
async def test_saved_note_remains_but_stale_source_text_never_returns(state, change):
    doc = _document(state)
    entry = (await _create(state, sources=_refs(doc))).json()
    if change == "delete":
        state.sources.documents.clear()
    elif change == "detach":
        _document(state, project_id=None)
    else:
        _document(state, revision="v2", text="New evidence.")
    saved = await state.client.get(f"{state.path}/{entry['id']}")
    assert saved.status_code == 200 and saved.json()["stale"]
    assert saved.json()["text"] == entry["text"] and saved.json()["source_warnings"]
    opened = await state.client.get(
        f"{state.path}/{entry['id']}/sources/0", params={"expected_revision": 1}
    )
    assert opened.status_code == 409 and doc.text not in opened.text


async def test_report_proposal_never_self_approves_and_replacement_checks_conflicts(state):
    existing = (await _create(state)).json()
    task, payload = await _report(state, replaces_id=existing["id"], base_revision=1)
    proposed = await state.client.post(f"{state.path}/proposals", json=payload)
    assert proposed.status_code == 201, proposed.text
    proposal = proposed.json()
    assert proposal["approval_state"] == "proposed" and proposal["approved_at"] is None
    assert proposal["origin"] == "agent_report" and proposal["source_task_id"] == task["id"]
    assert proposal["sources"][0]["source_id"] == f"agent:{task['id']}:report"
    await state.client.patch(
        f"{state.path}/{existing['id']}", json={"expected_revision": 1, "text": "Newer decision."}
    )
    decision = await state.client.post(
        f"{state.path}/{proposal['id']}/decision",
        json={"expected_revision": 1, "decision": "approve"},
    )
    assert decision.status_code == 409
    rejected = await state.client.post(
        f"{state.path}/{proposal['id']}/decision",
        json={"expected_revision": 1, "decision": "reject"},
    )
    assert rejected.json()["approval_state"] == "rejected"
    payload.update(request_id=uuid.uuid4().hex, base_revision=2)
    replacement = (await state.client.post(f"{state.path}/proposals", json=payload)).json()
    approved = await state.client.post(
        f"{state.path}/{replacement['id']}/decision",
        json={"expected_revision": 1, "decision": "approve"},
    )
    assert approved.status_code == 200 and approved.json()["approval_state"] == "approved"
    assert (await state.client.get(f"{state.path}/{existing['id']}")).json()[
        "approval_state"
    ] == "superseded"


async def test_missing_and_stale_report_requests_cannot_create_proposals(state):
    _task, payload = await _report(state)
    changed = await state.client.post(
        f"{state.path}/proposals", json={**payload, "task_revision": 999}
    )
    assert changed.status_code == 409
    state.current[0] = state.owners[1]
    foreign_path = f"/api/projects/{state.projects[1].project_id}/knowledge/proposals"
    assert (await state.client.post(foreign_path, json=payload)).status_code == 404


async def test_stale_proposal_cannot_be_approved_but_can_be_rejected(state):
    _task, payload = await _report(state)
    proposal = (await state.client.post(f"{state.path}/proposals", json=payload)).json()
    state.sources.documents.clear()
    path = f"{state.path}/{proposal['id']}/decision"
    assert (
        await state.client.post(path, json={"expected_revision": 1, "decision": "approve"})
    ).status_code == 409
    rejected = await state.client.post(path, json={"expected_revision": 1, "decision": "reject"})
    assert rejected.status_code == 200 and rejected.json()["approval_state"] == "rejected"


async def test_signed_cursor_is_bounded_and_bound_to_owner_project(state):
    for index in range(5):
        assert (await _create(state, text=f"Decision {index}.")).status_code == 201
    first = (await state.client.get(state.path, params={"limit": 2})).json()
    assert len(first["items"]) == 2 and first["next_cursor"]
    second = (
        await state.client.get(state.path, params={"limit": 2, "cursor": first["next_cursor"]})
    ).json()
    assert not {entry["id"] for entry in first["items"]} & {
        entry["id"] for entry in second["items"]
    }
    project = await state.store.projects.create(user_id=state.owners[0].user_id, name="Other owned")
    assert (
        await state.client.get(
            f"/api/projects/{project.project_id}/knowledge", params={"cursor": first["next_cursor"]}
        )
    ).status_code == 422
    state.current[0] = state.owners[1]
    assert (
        await state.client.get(
            f"/api/projects/{state.projects[1].project_id}/knowledge",
            params={"cursor": first["next_cursor"]},
        )
    ).status_code == 422
    state.current[0] = state.owners[0]
    for params in ({"limit": 51}, {"limit": 0}, {"cursor": "forged"}, {"cursor": "x" * 1025}):
        assert (await state.client.get(state.path, params=params)).status_code == 422


@pytest.mark.parametrize("change", ["entry", "project", "account", "edit"])
@pytest.mark.parametrize("operation", ["get", "patch", "source"])
async def test_canonical_changes_during_source_await_never_return_deleted_or_old_notes(
    state, change, operation
):
    doc = _document(state)
    entry = (await _create(state, sources=_refs(doc))).json()
    invoked = False

    async def during_read(_source_id):
        nonlocal invoked
        if invoked:
            return
        invoked = True
        if change == "project":
            await state.store.projects.delete(
                user_id=state.owners[0].user_id, project_id=state.projects[0].project_id
            )
        elif change == "account":
            with state.store._lock, state.store._conn:
                state.store._conn.execute(
                    "UPDATE app_users SET status='disabled' WHERE user_id=?",
                    (state.owners[0].user_id,),
                )
        elif change == "entry":
            await state.store.knowledge.delete(
                user_id=state.owners[0].user_id,
                project_id=state.projects[0].project_id,
                entry_id=entry["id"],
                expected_revision=2 if operation == "patch" else 1,
            )
        else:
            await state.store.knowledge.update(
                user_id=state.owners[0].user_id,
                project_id=state.projects[0].project_id,
                entry_id=entry["id"],
                expected_revision=2 if operation == "patch" else 1,
                text="A newer note must not be replaced by a stale response.",
            )

    state.sources.on_resolve = during_read
    path = f"{state.path}/{entry['id']}"
    if operation == "get":
        response = await state.client.get(path)
    elif operation == "patch":
        response = await state.client.patch(
            path, json={"expected_revision": 1, "text": "Current attempted change."}
        )
    else:
        response = await state.client.get(f"{path}/sources/0", params={"expected_revision": 1})
    assert invoked and response.status_code in {404, 409}, response.text
    assert entry["text"] not in response.text and doc.text not in response.text


async def test_list_rechecks_earlier_note_after_later_await(state):
    await _create(state, sources=_refs(_document(state, "message:first")))
    second = (await _create(state, sources=_refs(_document(state, "message:second")))).json()

    async def remove_previous(source_id):
        if source_id == "message:first":
            await state.store.knowledge.delete(
                user_id=state.owners[0].user_id,
                project_id=state.projects[0].project_id,
                entry_id=second["id"],
                expected_revision=1,
            )

    state.sources.on_resolve = remove_previous
    response = await state.client.get(state.path)
    assert response.status_code == 404 and second["text"] not in response.text


@pytest.mark.parametrize("operation", ["create", "proposal", "approve"])
async def test_mutation_final_hydration_fences_project_deleted_after_saved_write(state, operation):
    if operation == "create":
        doc = _document(state)
        payload = None
    else:
        _task, payload = await _report(state)
        if operation == "approve":
            proposal = (await state.client.post(f"{state.path}/proposals", json=payload)).json()
    reads = 0

    async def deletion_during_final_hydrate(_source_id):
        nonlocal reads
        reads += 1
        if reads == 2:
            await state.store.projects.delete(
                user_id=state.owners[0].user_id,
                project_id=state.projects[0].project_id,
            )

    state.sources.on_resolve = deletion_during_final_hydrate
    if operation == "create":
        response = await _create(state, sources=_refs(doc))
    elif operation == "proposal":
        response = await state.client.post(f"{state.path}/proposals", json=payload)
    else:
        response = await state.client.post(
            f"{state.path}/{proposal['id']}/decision",
            json={"expected_revision": 1, "decision": "approve"},
        )
    assert reads == 2 and response.status_code == 404, response.text
    assert "orchard" not in response.text and "dusk" not in response.text


async def test_dependency_failures_have_content_free_503_and_no_saved_source_excerpt(state):
    doc = _document(state)

    def broken(_source_id):
        raise OSError("private original body or path must stay out of the response")

    state.sources.on_resolve = broken
    response = await _create(state, sources=_refs(doc))
    assert response.status_code == 503 and "private original" not in response.text
    assert (await state.client.get(state.path)).json()["items"] == []
    del state.app.state.knowledge_source_resolver
    response = await state.client.get(state.path)
    assert response.status_code == 503 and "traceback" not in response.text.lower()


@pytest.mark.parametrize(
    "changes",
    [
        {"approval_state": "approved"},
        {"origin": "agent_report"},
        {"user_id": "foreign"},
        {"pinned": 1},
        {"pinned": "true"},
        {"kind": "system"},
        {"text": " "},
        {"text": "x" * 8001},
        {"sources": [{"source_id": "message:x", "revision": "1", "excerpt": "copied"}]},
    ],
)
async def test_manual_payload_cannot_forge_authority_or_exceed_bounds(state, changes):
    response = await _create(state, **changes)
    assert response.status_code == 422, response.text
    assert (await state.client.get(state.path)).json()["items"] == []


async def test_writes_refresh_completed_search_immediately_including_proposal_approval_and_delete(
    state, tmp_path
):
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    index = ContentSearchIndex(tmp_path / "search.sqlite")
    reader = ContentSourceReader(state.store, uploads, tmp_path / "uploads")
    service = ContentSearchService(state.store, reader, index)
    state.app.state.content_search = service
    state.app.state.knowledge_source_resolver = KnowledgeSourceResolver(state.store, reader)
    owner = state.owners[0]
    try:
        scan = await service.prepare(owner.user_id, owner.storage_namespace)
        for _ in SOURCE_KINDS:
            if len(scan.done) == len(SOURCE_KINDS):
                break
            await service._advance(owner.user_id, scan)
        assert len(scan.done) == len(SOURCE_KINDS)

        async def matches(term):
            response = await state.client.get(
                "/api/search", params={"q": term, "kind": "knowledge"}
            )
            assert response.status_code == 200, response.text
            return {item["source_id"] for item in response.json()["items"]}

        created = await _create(state, text="Preserve the heliotrope orchard.")
        assert created.status_code == 201, created.text
        entry = created.json()
        assert await matches("heliotrope") == {f"knowledge:{entry['id']}"}
        updated = await state.client.patch(
            f"{state.path}/{entry['id']}",
            json={"expected_revision": 1, "text": "Preserve the wintergreen orchard."},
        )
        assert updated.status_code == 200, updated.text
        assert await matches("heliotrope") == set()
        assert await matches("wintergreen") == {f"knowledge:{entry['id']}"}

        _task, payload = await _report(
            state,
            text="Preserve the gooseberry orchard.",
            replaces_id=entry["id"],
            base_revision=2,
        )
        proposed = await state.client.post(f"{state.path}/proposals", json=payload)
        assert proposed.status_code == 201, proposed.text
        proposal = proposed.json()
        assert await matches("gooseberry") == set()
        approved = await state.client.post(
            f"{state.path}/{proposal['id']}/decision",
            json={"expected_revision": 1, "decision": "approve"},
        )
        assert approved.status_code == 200, approved.text
        assert await matches("wintergreen") == set()
        assert await matches("gooseberry") == {f"knowledge:{proposal['id']}"}
        indexed = await index.list_sources(owner.user_id)
        assert f"knowledge:{entry['id']}" not in {item["source_id"] for item in indexed}

        deleted = await state.client.delete(
            f"{state.path}/{proposal['id']}",
            params={"expected_revision": approved.json()["revision"]},
        )
        assert deleted.status_code == 200, deleted.text
        assert await matches("gooseberry") == set()
        assert f"knowledge:{proposal['id']}" not in {
            item["source_id"] for item in await index.list_sources(owner.user_id)
        }
    finally:
        await service.stop()
        index.close()
        uploads.close()


@pytest.mark.parametrize("error_type", [ContentSearchError, OSError, sqlite3.OperationalError])
async def test_projection_failure_never_reverts_canonical_write_or_logs_source_content(
    state, caplog, error_type
):
    calls = []

    class UnavailableProjection:
        async def refresh_knowledge(self, owner, namespace, entry_ids):
            calls.append((owner, namespace, entry_ids))
            raise error_type("Private source body or storage path must not be logged.")

    state.app.state.content_search = UnavailableProjection()
    created = await _create(state)
    assert created.status_code == 201, created.text
    entry = created.json()
    assert calls == [(state.owners[0].user_id, state.owners[0].storage_namespace, [entry["id"]])]
    assert "Private source body" not in caplog.text
    assert "search refresh is unavailable" in caplog.text
    assert (await state.client.get(f"{state.path}/{entry['id']}")).json()["text"] == entry["text"]
    deleted = await state.client.delete(
        f"{state.path}/{entry['id']}", params={"expected_revision": 1}
    )
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert (await state.client.get(f"{state.path}/{entry['id']}")).status_code == 404
