"""Native API contracts for reviewed template document requests."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal, require_provider_principal
from audrey.documents import DocumentTemplateCatalog
from audrey.identity import Principal
from audrey.routes.app import router


class _Worker:
    def __init__(self) -> None:
        self.wakes = 0

    def wake(self) -> None:
        self.wakes += 1


async def _principal(
    store: ApplicationStore,
    *,
    subject: str,
    email: str,
) -> Principal:
    return await store.resolve_external_identity(
        provider="owui",
        subject=subject,
        email=email,
        display_name=email.split("@", maxsplit=1)[0].title(),
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=email,
    )


def _request_body() -> dict[str, object]:
    return {
        "template_id": "project-brief-v1",
        "filename": "Project brief.docx",
        "fields": {
            "title": "Audrey document tools",
            "prepared_for": "Build Ryte",
            "prepared_on": "October 4, 2026",
            "summary": "A reviewed private project brief.",
            "objectives": ["Verify the output", "Keep it private"],
            "next_steps": ["Approve the exact request", "Download the result"],
        },
        "idempotency_key": "browser-project-brief-one",
    }


async def test_template_routes_create_exact_owner_approval_and_wake_worker(
    tmp_path: Path,
) -> None:
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _principal(
        store,
        subject="template-alice",
        email="alice@example.com",
    )
    bob = await _principal(
        store,
        subject="template-bob",
        email="bob@example.com",
    )
    current = {"principal": alice}
    worker = _Worker()
    app = FastAPI()
    app.state.application_store = store
    app.state.document_templates = DocumentTemplateCatalog()
    app.state.document_template_worker = worker
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: current["principal"]
    app.dependency_overrides[require_provider_principal] = lambda: current["principal"]
    try:
        with TestClient(app) as client:
            templates = client.get("/api/document-jobs/templates")
            assert templates.status_code == 200
            assert templates.json()["items"] == [
                {
                    "id": "project-brief-v1",
                    "name": "Project brief",
                    "description": (
                        "Create a polished brief with a title, recipient, summary, "
                        "objectives, and next steps."
                    ),
                    "version": 1,
                    "fields": [
                        "title",
                        "prepared_for",
                        "prepared_on",
                        "summary",
                        "objectives",
                        "next_steps",
                    ],
                }
            ]

            created = client.post(
                "/api/document-jobs/template-to-docx",
                json=_request_body(),
            )
            assert created.status_code == 201
            payload = created.json()
            assert payload["status"] == "awaiting_approval"
            assert payload["operation"] == "template_to_docx"
            assert payload["approval"]["decision"] == "pending"
            assert payload["operation_digest"] == payload["approval"]["operation_digest"]
            assert "arguments" not in payload
            assert worker.wakes == 0

            current["principal"] = bob
            assert client.get(f"/api/document-jobs/{payload['id']}").status_code == 404

            current["principal"] = alice
            approved = client.post(
                f"/api/document-jobs/{payload['id']}/decision",
                json={
                    "decision": "approved",
                    "operation_digest": payload["operation_digest"],
                },
            )
            assert approved.status_code == 200
            assert approved.json()["status"] == "queued"
            assert worker.wakes == 1
    finally:
        store.close()


async def test_template_route_rejects_unsafe_filename_before_job_creation(
    tmp_path: Path,
) -> None:
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _principal(
        store,
        subject="template-invalid",
        email="alice@example.com",
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.document_templates = DocumentTemplateCatalog()
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: alice
    app.dependency_overrides[require_provider_principal] = lambda: alice
    try:
        body = _request_body()
        body["filename"] = "../outside.docx"
        with TestClient(app) as client:
            response = client.post(
                "/api/document-jobs/template-to-docx",
                json=body,
            )
            assert response.status_code == 422
            jobs = client.get("/api/document-jobs")
            assert jobs.status_code == 200
            assert jobs.json()["items"] == []
    finally:
        store.close()
