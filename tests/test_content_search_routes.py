"""HTTP search tests use a real FTS projection and a mutable canonical source."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey import content_search
from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.content_search import ContentSearchError, ContentSearchIndex, SearchDocument
from audrey.content_search_service import ContentSearchService
from audrey.routes.app import router


class Sources:
    def __init__(self):
        self.documents = {}
        self.deleted = set()
        self.on_resolve = None
        self.pages = []

    async def list_page(self, owner, namespace, kind, *, after=None, limit=50):
        self.pages.append((owner, kind, after, limit))
        documents = sorted(
            (d for d in self.documents.values() if d.owner_id == owner and d.kind == kind),
            key=lambda d: d.source_id,
        )
        documents = [d for d in documents if after is None or d.source_id > after]
        page = documents[:limit]
        return page, page[-1].source_id if len(documents) > limit else None, []

    async def resolve(self, owner, namespace, source):
        if self.on_resolve:
            self.on_resolve(source)
        document = self.documents.get(source)
        return (
            document
            if document and document.owner_id == owner and source not in self.deleted
            else None
        )

    def is_current(self, document):
        return (
            self.documents.get(document.source_id) == document
            and document.source_id not in self.deleted
        )


@pytest.fixture
async def search(tmp_path):
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
    current = [owners[0]]
    index = ContentSearchIndex(tmp_path / "search.sqlite")
    sources = Sources()
    service = ContentSearchService(store, sources, index)
    app = FastAPI()
    app.state.content_search = service
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: current[0]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield SimpleNamespace(
            store=store,
            owners=owners,
            current=current,
            index=index,
            sources=sources,
            service=service,
            app=app,
            client=client,
        )
    await service.stop()
    index.close()
    store.close()


def document(state, source="message:one", **changes):
    value = SearchDocument(
        owner_id=state.owners[0].user_id,
        source_id=source,
        kind="message",
        revision="1",
        title="A saved conversation",
        text="Body-only heliotrope detail. <script>plain text</script>",
        updated_at="2026-10-29T12:00:00Z",
        target={
            "kind": "message",
            "conversation_id": "conversation",
            "message_id": source,
            "sequence": 7,
        },
    )
    value = replace(value, **changes)
    state.sources.documents[source] = value
    return value


async def test_content_body_match_is_local_owned_and_opens_canonical_text(search):
    value = document(search)
    other = document(
        search, "message:foreign", owner_id=search.owners[1].user_id, text="foreign heliotrope"
    )
    await search.index.replace_source(other)
    response = await search.client.get("/api/search", params={"q": "heliotrope"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    items = response.json()["items"]
    assert len(items) == 1 and items[0]["source_id"] == value.source_id
    assert items[0]["target"]["sequence"] == 7
    assert "<script>" in items[0]["excerpt"]
    assert not {"owner_id", "body_hash", "archived"} & items[0].keys()
    opened = await search.client.get(
        "/api/search/source",
        params={
            "source_id": value.source_id,
            "revision": "1",
            "offset": items[0]["offset"],
        },
    )
    assert opened.status_code == 200 and opened.json()["text"] == value.text
    search.current[0] = search.owners[1]
    assert (
        await search.client.get(
            "/api/search/source",
            params={
                "source_id": value.source_id,
                "revision": "1",
            },
        )
    ).status_code == 404


async def test_changed_or_deleted_projection_never_returns_stale_excerpt(search):
    value = document(search)
    await search.service.prepare(value.owner_id, search.owners[0].storage_namespace)
    search.sources.documents[value.source_id] = replace(
        value, revision="2", text="replacement body"
    )
    response = await search.client.get("/api/search", params={"q": "heliotrope"})
    assert response.json()["items"] == [] and response.json()["partial"]
    assert (
        await search.client.get(
            "/api/search/source",
            params={
                "source_id": value.source_id,
                "revision": "1",
            },
        )
    ).status_code == 409
    search.sources.deleted.add(value.source_id)
    assert (
        await search.client.get(
            "/api/search/source",
            params={
                "source_id": value.source_id,
                "revision": "2",
            },
        )
    ).status_code == 404


async def test_final_fence_catches_deletion_during_another_awaited_source_read(search):
    first = document(search, "message:a")
    document(search, "message:b")
    calls = []

    def deletion_during_read(source):
        calls.append(source)
        if len(calls) == 2:
            search.sources.deleted.add(calls[0])

    search.sources.on_resolve = deletion_during_read
    response = await search.client.get("/api/search", params={"q": "heliotrope"})
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    assert response.json()["partial"]
    assert len(calls) == 2 and first.owner_id == search.owners[0].user_id


async def test_cursor_is_bound_to_owner_query_and_filters(search):
    for number in range(30):
        document(search, f"message:{number:03}")
    page = (await search.client.get("/api/search", params={"q": "heliotrope"})).json()
    assert len(page["items"]) == 25 and page["next_cursor"]
    second = await search.client.get(
        "/api/search", params={"q": "heliotrope", "cursor": page["next_cursor"]}
    )
    assert len(second.json()["items"]) == 5 and second.json()["next_cursor"] is None
    for params in (
        {"q": "different"},
        {"q": "heliotrope", "kind": "file"},
        {"q": "heliotrope", "archived": "active"},
    ):
        assert (
            await search.client.get("/api/search", params={**params, "cursor": page["next_cursor"]})
        ).status_code == 422
    search.current[0] = search.owners[1]
    assert (
        await search.client.get(
            "/api/search", params={"q": "heliotrope", "cursor": page["next_cursor"]}
        )
    ).status_code == 422


async def test_live_project_membership_archive_and_private_metadata(search):
    project = await search.store.projects.create(
        user_id=search.owners[0].user_id, name="Local work"
    )
    document(
        search,
        "file:retained:transcript",
        kind="file",
        project_id=None,
        private_required=True,
        availability="text_only",
        archived=True,
        target={
            "kind": "file",
            "file_id": "retained",
            "artifact": "transcript",
            "project_ids": [project.project_id],
        },
    )
    params = {
        "q": "heliotrope",
        "kind": "file",
        "project_id": project.project_id,
        "archived": "archived",
    }
    result = await search.client.get("/api/search", params=params)
    assert result.status_code == 200, result.text
    item = result.json()["items"][0]
    assert item["private_required"] and item["availability"] == "text_only"
    assert (await search.client.get("/api/search", params={**params, "archived": "active"})).json()[
        "items"
    ] == []
    assert (
        await search.client.get("/api/search", params={**params, "project_id": "missing"})
    ).status_code == 404


@pytest.mark.parametrize("query", ["!", "a " * 13, "x" * 257, "\x00"])
async def test_query_bounds(search, query):
    assert (await search.client.get("/api/search", params={"q": query})).status_code == 422


async def test_source_pages_use_real_character_offsets_and_reject_out_of_range(search):
    value = document(search, text="😀" * 4100 + "heliotrope")
    response = await search.client.get(
        "/api/search/source", params={"source_id": value.source_id, "revision": "1", "offset": 3990}
    )
    body = response.json()
    assert body["text"] == value.text[3990:] and body["offset"] == 3990
    assert body["total_chars"] == 4110 and body["next_offset"] is None
    assert (
        await search.client.get(
            "/api/search/source",
            params={"source_id": value.source_id, "revision": "1", "offset": 5000},
        )
    ).status_code == 422


async def test_bounded_backfill_and_reconciliation_remove_deleted_sources(search):
    for number in range(60):
        document(search, f"message:{number:03}")
    owner = search.owners[0]
    scan = await search.service.prepare(owner.user_id, owner.storage_namespace)
    assert len(await search.index.list_sources(owner.user_id)) == 50
    assert "message" not in scan.done
    await search.service._advance(owner.user_id, scan)
    assert len(await search.index.list_sources(owner.user_id)) == 60
    search.sources.documents.pop("message:059")
    scan.completed_at = 1
    await search.service.prepare(owner.user_id, owner.storage_namespace)
    await search.service._advance(owner.user_id, scan)
    assert len(await search.index.list_sources(owner.user_id)) == 59
    assert all(page[3] <= 50 for page in search.sources.pages)


async def test_startup_recovery_removes_disabled_owner_projection(search):
    value = document(search)
    await search.index.replace_source(value)
    with search.store._lock:
        search.store._conn.execute(
            "UPDATE app_users SET status='disabled' WHERE user_id=?", (value.owner_id,)
        )
        search.store._conn.commit()
    await search.service.start()
    assert await search.index.list_sources(value.owner_id) == []


async def test_unchanged_periodic_scan_skips_projection_writes_and_preserves_cursor(
    search, monkeypatch
):
    for number in range(30):
        document(search, f"message:{number:03}")
    writes = []
    original = search.index.replace_source

    async def replace_source(value):
        writes.append(value.source_id)
        await original(value)

    monkeypatch.setattr(search.index, "replace_source", replace_source)
    page = (await search.client.get("/api/search", params={"q": "heliotrope"})).json()
    assert len(writes) == 30
    scan = search.service._owners[search.owners[0].user_id]
    generation = scan.generation
    scan.completed_at = 1
    response = await search.client.get(
        "/api/search", params={"q": "heliotrope", "cursor": page["next_cursor"]}
    )
    assert response.status_code == 200 and len(response.json()["items"]) == 5
    assert len(writes) == 30 and scan.generation == generation


async def test_projection_changes_invalidate_offset_cursor_instead_of_skipping_sources(search):
    for number in range(30):
        document(search, f"message:{number:03}")
    page = (await search.client.get("/api/search", params={"q": "heliotrope"})).json()
    search.sources.documents.pop("message:000")
    search.service._owners[search.owners[0].user_id].completed_at = 1
    response = await search.client.get(
        "/api/search", params={"q": "heliotrope", "cursor": page["next_cursor"]}
    )
    assert response.status_code == 422
    assert "Search again" in response.json()["detail"]


async def test_stale_hydration_does_not_issue_a_cursor_into_shifted_projection(search):
    for number in range(30):
        document(search, f"message:{number:03}")
    owner = search.owners[0]
    await search.service.prepare(owner.user_id, owner.storage_namespace)
    search.sources.deleted.add("message:000")
    response = await search.client.get("/api/search", params={"q": "heliotrope"})
    body = response.json()
    assert response.status_code == 200 and len(body["items"]) == 25
    assert body["partial"] and body["next_cursor"] is None
    assert all(item["source_id"] != "message:000" for item in body["items"])


@pytest.mark.parametrize(
    "changes",
    [
        {"excerpt": "invented heliotrope text"},
        {"offset": -1},
        {"highlights": [{"start": -1, "end": 1000}]},
    ],
)
async def test_projection_excerpt_and_ranges_are_validated_against_canonical_text(
    search, monkeypatch, changes
):
    document(search)
    original = search.index.search

    async def corrupted(*args, **kwargs):
        values = await original(*args, **kwargs)
        return [{**value, **changes} for value in values]

    monkeypatch.setattr(search.index, "search", corrupted)
    response = await search.client.get("/api/search", params={"q": "heliotrope"})
    assert response.status_code == 200
    assert response.json()["items"] == [] and response.json()["partial"]


async def test_dates_filter_canonical_update_date_inclusively_in_utc(search):
    document(search, "message:start", updated_at="2026-10-29T00:00:00Z")
    document(search, "message:offset", updated_at="2026-10-30T01:00:00+02:00")
    document(search, "message:before", updated_at="2026-10-28T23:59:59Z")
    document(search, "message:after", updated_at="2026-10-30T00:00:00Z")
    response = await search.client.get(
        "/api/search",
        params={
            "q": "heliotrope",
            "since": "2026-10-29",
            "until": "2026-10-29",
        },
    )
    assert response.status_code == 200
    assert {item["source_id"] for item in response.json()["items"]} == {
        "message:start",
        "message:offset",
    }


async def test_dates_are_validated_and_bound_to_cursor(search):
    for number in range(30):
        document(search, f"message:{number:03}")
    page = (await search.client.get("/api/search", params={"q": "heliotrope"})).json()
    for params in (
        {"since": "2026-10-30", "until": "2026-10-29"},
        {"since": "not-a-date"},
        {"since": "2026-10-28", "cursor": page["next_cursor"]},
        {"until": "2026-10-31", "cursor": page["next_cursor"]},
    ):
        assert (
            await search.client.get("/api/search", params={"q": "heliotrope", **params})
        ).status_code == 422


async def test_purge_removes_projection_invalidates_old_scan_and_accepts_new_canonical_data(search):
    value = document(search)
    owner = search.owners[0]
    old_scan = await search.service.prepare(owner.user_id, owner.storage_namespace)
    search.sources.documents.clear()
    await search.service.purge_owner(owner.user_id)
    assert await search.index.list_sources(owner.user_id) == []
    assert (
        not old_scan.seen and not old_scan.cursors and not old_scan.done and not old_scan.warnings
    )
    assert old_scan.completed_at == 0
    document(search, "message:new")
    await search.service._advance(value.owner_id, old_scan)
    assert await search.index.list_sources(owner.user_id) == []
    new_scan = await search.service.prepare(owner.user_id, owner.storage_namespace)
    assert new_scan is not old_scan and new_scan.generation != old_scan.generation
    assert [source["source_id"] for source in await search.index.list_sources(owner.user_id)] == [
        "message:new"
    ]


async def test_purge_waits_for_inflight_backfill_and_removes_its_late_insert(search, monkeypatch):
    document(search)
    owner = search.owners[0]
    entered, release = asyncio.Event(), asyncio.Event()
    original = search.index.replace_source

    async def blocked(value):
        entered.set()
        await release.wait()
        await original(value)

    monkeypatch.setattr(search.index, "replace_source", blocked)
    backfill = asyncio.create_task(search.service.prepare(owner.user_id, owner.storage_namespace))
    await asyncio.wait_for(entered.wait(), 2)
    search.sources.documents.clear()
    purge = asyncio.create_task(search.service.purge_owner(owner.user_id))
    await asyncio.sleep(0)
    assert not purge.done()
    release.set()
    await asyncio.wait_for(asyncio.gather(backfill, purge), 2)
    assert await search.index.list_sources(owner.user_id) == []


async def test_purge_waits_for_inflight_hydration_repair(search, monkeypatch):
    value = document(search)
    owner = search.owners[0]
    await search.service.prepare(owner.user_id, owner.storage_namespace)
    [candidate] = await search.index.search(owner.user_id, "heliotrope")
    search.sources.documents[value.source_id] = replace(
        value, revision="2", text="new heliotrope text"
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = search.index.replace_source

    async def blocked(document):
        entered.set()
        await release.wait()
        await original(document)

    monkeypatch.setattr(search.index, "replace_source", blocked)
    repair = asyncio.create_task(
        search.service.hydrate(owner.user_id, owner.storage_namespace, candidate)
    )
    await asyncio.wait_for(entered.wait(), 2)
    search.sources.documents.clear()
    purge = asyncio.create_task(search.service.purge_owner(owner.user_id))
    await asyncio.sleep(0)
    assert not purge.done()
    release.set()
    await asyncio.wait_for(asyncio.gather(repair, purge), 2)
    assert await search.index.list_sources(owner.user_id) == []


async def test_hydration_read_finishing_after_purge_cannot_repopulate_projection(
    search, monkeypatch
):
    value = document(search)
    owner = search.owners[0]
    await search.service.prepare(owner.user_id, owner.storage_namespace)
    [candidate] = await search.index.search(owner.user_id, "heliotrope")
    changed = replace(value, revision="2", text="new heliotrope text")
    search.sources.documents[value.source_id] = changed
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        await release.wait()
        return changed

    monkeypatch.setattr(search.sources, "resolve", blocked)
    repair = asyncio.create_task(
        search.service.hydrate(owner.user_id, owner.storage_namespace, candidate)
    )
    await asyncio.wait_for(entered.wait(), 2)
    search.sources.documents.clear()
    await search.service.purge_owner(owner.user_id)
    release.set()
    await asyncio.wait_for(repair, 2)
    assert await search.index.list_sources(owner.user_id) == []


async def test_quota_partial_scan_still_removes_disappeared_source(search, monkeypatch):
    value = document(search)
    owner = search.owners[0]
    scan = await search.service.prepare(owner.user_id, owner.storage_namespace)
    monkeypatch.setattr(content_search, "MAX_OWNER_SOURCES", 1)
    search.sources.documents.clear()
    document(search, "message:new-one")
    document(search, "message:new-two")
    scan.completed_at = 1
    await search.service.prepare(owner.user_id, owner.storage_namespace)
    assert scan.warnings
    assert all(
        source["source_id"] != value.source_id
        for source in await search.index.list_sources(owner.user_id)
    )


async def test_failed_replacement_removes_previous_text_revision(search, monkeypatch):
    value = document(search, text="heliotrope")
    owner = search.owners[0]
    scan = await search.service.prepare(owner.user_id, owner.storage_namespace)
    monkeypatch.setattr(content_search, "MAX_OWNER_CHARS", len(value.text))
    search.sources.documents[value.source_id] = replace(
        value, revision="2", text="larger replacement heliotrope"
    )
    scan.completed_at = 1
    await search.service.prepare(owner.user_id, owner.storage_namespace)
    assert scan.warnings and await search.index.list_sources(owner.user_id) == []


@pytest.mark.parametrize("stage", ["read", "fence"])
async def test_source_read_failure_is_safe_unavailable_response(search, monkeypatch, stage):
    async def unavailable(*args):
        raise ContentSearchError("private internals must not be returned")

    if stage == "read":
        monkeypatch.setattr(search.sources, "resolve", unavailable)
    else:
        document(search)

        def failed_fence(*args):
            raise ContentSearchError("private internals must not be returned")

        monkeypatch.setattr(search.sources, "is_current", failed_fence)
    response = await search.client.get(
        "/api/search/source", params={"source_id": "message:one", "revision": "1"}
    )
    assert response.status_code == 503
    assert "private internals" not in response.text
