"""Project agent reports retain canonical Search grouping and deletion fences."""

import httpx
from fastapi import FastAPI

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.content_search import ContentSearchIndex
from audrey.content_search_service import ContentSearchService
from audrey.content_sources import ContentSourceReader
from audrey.kb.uploads_db import UploadsDB
from audrey.routes.app import router


async def test_project_report_filter_and_source_revision_follow_project_deletion(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    index = ContentSearchIndex(tmp_path / "search.sqlite")
    owner = await store.resolve_external_identity(provider="test", subject="alice", email="alice@example.com", display_name="Alice", role="user", auth_method="test")
    project = await store.projects.create(user_id=owner.user_id, name="Release work")
    task = await store.agents.create(user_id=owner.user_id, project_id=project.project_id, objective="Compare releases", request_id="project-report")
    reader = ContentSourceReader(store, uploads, tmp_path / "uploads")
    service = ContentSearchService(store, reader, index)
    app = FastAPI()
    app.state.content_search = service
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner
    try:
        async def execute(_task, step):
            return f"Saved heliotrope {step} report."

        assert await AgentRuntime(store.agents, execute).run_once()
        document = await reader.resolve(owner.user_id, owner.storage_namespace, f"agent:{task['id']}:report")
        assert document.project_id == project.project_id and reader.is_current(document)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            matches = await client.get("/api/search", params={"q": "heliotrope", "project_id": project.project_id})
            assert matches.status_code == 200 and len(matches.json()["items"]) == 1
            result = matches.json()["items"][0]
            assert result["project_id"] == project.project_id
            await store.projects.update(user_id=owner.user_id, project_id=project.project_id, name="Renamed")
            assert reader.is_current(document)
            await store.projects.delete(user_id=owner.user_id, project_id=project.project_id)
            assert not reader.is_current(document)
            response = await client.get("/api/search/source", params={"source_id": result["source_id"], "revision": result["revision"]})
            assert response.status_code == 409 and "heliotrope" not in response.text
            current = await reader.resolve(owner.user_id, owner.storage_namespace, result["source_id"])
            assert current.project_id is None and "heliotrope" in current.text
    finally:
        await service.stop()
        index.close()
        uploads.close()
        store.close()
