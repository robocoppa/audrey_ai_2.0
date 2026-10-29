"""Canonical messages, files and saved reports flow through the native API."""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.content_search import ContentSearchIndex
from audrey.content_search_service import ContentSearchService
from audrey.content_sources import ContentSourceReader
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.routes.app import router


async def test_actual_sources_search_open_delete_and_rebuild_without_model_or_embeddings(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    root = tmp_path / "uploads"
    index = ContentSearchIndex(tmp_path / "search" / "index.sqlite")
    owner = await store.resolve_external_identity(
        provider="test",
        subject="alice",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="test",
    )
    service = ContentSearchService(store, ContentSourceReader(store, uploads, root), index)
    app = FastAPI()
    app.state.content_search = service
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner
    try:
        conversation = await store.conversations.create(
            user_id=owner.user_id, title="Native conversation"
        )
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="A heliotrope decision saved in the message body.",
        )
        await store.conversations.finish_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
            outcome="succeeded",
            assistant_content="A saved acknowledgement.",
        )
        directory = root / sanitize_user(owner.storage_namespace)
        directory.mkdir(parents=True)
        (directory / "file-one.txt").write_text(
            "File body contains heliotrope evidence.", encoding="utf-8"
        )
        await uploads.record_upload(
            file_id="file-one",
            user=owner.storage_namespace,
            filename="source.txt",
            mime="text/plain",
            bytes_=44,
            kind="text",
            collection="unused",
            chunks=0,
            uploaded_at="2026-10-29T12:00:00Z",
        )
        task = await store.agents.create(
            user_id=owner.user_id, objective="A saved report", request_id="one"
        )

        async def execute(_task, stage):
            return f"Generated heliotrope {stage} report."

        assert await AgentRuntime(store.agents, execute).run_once()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.get("/api/search", params={"q": "heliotrope"})
            assert result.status_code == 200, result.text
            items = result.json()["items"]
            assert {item["kind"] for item in items} == {"message", "file", "agent"}
            for item in items:
                source = await client.get(
                    "/api/search/source",
                    params={
                        "source_id": item["source_id"],
                        "revision": item["revision"],
                        "offset": item["offset"],
                    },
                )
                assert source.status_code == 200, source.text
                assert "heliotrope" in source.json()["text"]
            # Native delete fences take effect even before projection cleanup.
            await store.conversations.delete(
                user_id=owner.user_id, conversation_id=conversation.conversation_id
            )
            current = await store.agents.get(user_id=owner.user_id, task_id=task["id"])
            await store.agents.delete(
                user_id=owner.user_id, task_id=task["id"], expected_revision=current["revision"]
            )
            await uploads.request_file_deletion(
                "file-one", user=owner.storage_namespace, requested_at="2026-10-29T13:00:00Z"
            )
            result = await client.get("/api/search", params={"q": "heliotrope"})
            assert result.json()["items"] == []
            for item in items:
                assert (
                    await client.get(
                        "/api/search/source",
                        params={
                            "source_id": item["source_id"],
                            "revision": item["revision"],
                        },
                    )
                ).status_code == 404
            await service.purge_owner(owner.user_id)
            assert await index.list_sources(owner.user_id) == []
    finally:
        await service.stop()
        index.close()
        uploads.close()
        store.close()
