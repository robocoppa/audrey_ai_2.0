"""Execution and recovery contracts for reviewed template-to-DOCX jobs."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path

import pytest

from audrey.app_state import ApplicationStore
from audrey.documents import (
    DOCX_MIME,
    PROJECT_BRIEF_TEMPLATE_ID,
    TEMPLATE_OPERATION,
    DocumentTemplateCatalog,
    DocumentTemplateWorker,
    ProjectBriefFields,
    TemplateDocumentArguments,
)
from audrey.kb.storage_lifecycle import StorageLifecycle
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user


class _Embedder:
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


class _Qdrant:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.points: dict[str, list[object]] = {}

    async def ensure_collection(self, name: str, *, dim: int) -> None:
        assert dim > 0
        self.collections.add(name)

    async def ensure_user_payload_indexes(self, name: str) -> None:
        assert name in self.collections

    async def has_sparse(self, name: str) -> bool:
        assert name in self.collections
        return False

    async def delete_by_file_id(
        self,
        file_id: str,
        *,
        user: str,
        collection: str,
    ) -> None:
        self.points[collection] = [
            point
            for point in self.points.get(collection, [])
            if not (
                getattr(point, "payload", {}).get("file_id") == file_id
                and getattr(point, "payload", {}).get("user") == user
            )
        ]

    async def upsert_text(self, points: list[object], *, collection: str) -> None:
        self.points.setdefault(collection, []).extend(points)


async def _owner(store: ApplicationStore):
    return await store.resolve_external_identity(
        provider="owui",
        subject="document-template-owner",
        email="owner@example.com",
        display_name="Owner",
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace="owner@example.com",
    )


def _arguments(catalog: DocumentTemplateCatalog) -> TemplateDocumentArguments:
    descriptor = catalog.descriptor(PROJECT_BRIEF_TEMPLATE_ID)
    return TemplateDocumentArguments(
        template_id=descriptor.template_id,
        template_sha256=descriptor.content_sha256,
        filename="Audrey project brief.docx",
        fields=ProjectBriefFields(
            title="Audrey document tools",
            prepared_for="Build Ryte",
            prepared_on="October 4, 2026",
            summary="A private, verified project brief generated after owner approval.",
            objectives=["Keep outputs owner scoped", "Verify the saved package"],
            next_steps=["Download the brief", "Use it in a Project"],
        ),
    )


async def _approved_job(
    store: ApplicationStore,
    catalog: DocumentTemplateCatalog,
    *,
    operation: str = TEMPLATE_OPERATION,
    key: str = "template-worker",
):
    owner = await _owner(store)
    arguments = _arguments(catalog)
    descriptor = catalog.descriptor(PROJECT_BRIEF_TEMPLATE_ID)
    source = await store.document_tools.register_source_version(
        user_id=owner.user_id,
        file_id=f"template_{descriptor.template_id}",
        filename=f"{descriptor.template_id}.docx",
        mime=DOCX_MIME,
        bytes_count=descriptor.bytes_count,
        content_sha256=descriptor.content_sha256,
    )
    job, approval = await store.document_tools.create_request(
        user_id=owner.user_id,
        input_version_id=source.version_id,
        operation=operation,
        arguments=arguments.model_dump(mode="json"),
        output_mime=DOCX_MIME,
        summary="Create a reviewed project brief.",
        preview="Project brief for Build Ryte.",
        idempotency_key=key,
        requested_by_kind="user",
        requested_by_id=owner.user_id,
    )
    decided = await store.document_tools.decide(
        user_id=owner.user_id,
        job_id=job.job_id,
        actor_user_id=owner.user_id,
        operation_digest=approval.operation_digest,
        decision="approved",
    )
    assert decided is not None
    return owner, arguments, decided[0]


def _worker(
    *,
    store: ApplicationStore,
    catalog: DocumentTemplateCatalog,
    uploads: UploadsDB,
    qdrant: _Qdrant,
    upload_root: Path,
) -> DocumentTemplateWorker:
    return DocumentTemplateWorker(
        store=store,
        catalog=catalog,
        uploads_db=uploads,
        storage=StorageLifecycle(uploads),
        qdrant=qdrant,  # type: ignore[arg-type]
        text_embedder=_Embedder(),  # type: ignore[arg-type]
        upload_root=upload_root,
        max_user_bytes=10 * 1024 * 1024,
    )


@pytest.mark.asyncio
async def test_worker_publishes_verified_docx_to_private_files(tmp_path: Path) -> None:
    app_path = tmp_path / "app.sqlite"
    store = ApplicationStore(app_path)
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    catalog = DocumentTemplateCatalog()
    qdrant = _Qdrant()
    owner, arguments, job = await _approved_job(store, catalog)
    worker = _worker(
        store=store,
        catalog=catalog,
        uploads=uploads,
        qdrant=qdrant,
        upload_root=tmp_path / "uploads",
    )
    try:
        assert await worker.process_one() is True
        assert await worker.process_one() is False

        finished = await store.document_tools.get_job(
            user_id=owner.user_id,
            job_id=job.job_id,
        )
        assert finished is not None
        finished_job = finished[0]
        assert finished_job.status == "succeeded"
        assert finished_job.output_version_id is not None
        output = await store.document_tools.get_version(
            user_id=owner.user_id,
            version_id=finished_job.output_version_id,
        )
        assert output is not None
        assert output.origin == "derivation"
        assert output.parent_version_id == job.input_version_id
        assert output.filename == arguments.filename
        assert output.mime == DOCX_MIME

        rows = await uploads.list_user(owner.storage_namespace)
        assert len(rows) == 1
        row = rows[0]
        assert row["file_id"] == output.file_id
        assert row["filename"] == arguments.filename
        assert row["status"] == "ready"
        assert row["chunks"] >= 1
        destination = (
            tmp_path
            / "uploads"
            / sanitize_user(owner.storage_namespace)
            / f"{output.file_id}.docx"
        )
        rendered = catalog.verify(destination.read_bytes(), arguments)
        assert rendered.content_sha256 == output.content_sha256
        assert qdrant.points

        with sqlite3.connect(app_path) as connection:
            provenance = connection.execute(
                "SELECT worker_version, operation_digest FROM app_file_derivations "
                "WHERE output_version_id = ?",
                (output.version_id,),
            ).fetchone()
        assert provenance is not None
        assert provenance[0].endswith(":verified")
        assert provenance[1] == job.operation_digest
    finally:
        uploads.close()
        store.close()


@pytest.mark.asyncio
async def test_worker_claims_only_its_operation(tmp_path: Path) -> None:
    store = ApplicationStore(tmp_path / "app.sqlite")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    catalog = DocumentTemplateCatalog()
    qdrant = _Qdrant()
    owner, _, pdf_job = await _approved_job(
        store,
        catalog,
        operation="docx_to_pdf",
        key="pdf-first",
    )
    _, _, template_job = await _approved_job(store, catalog, key="template-second")
    worker = _worker(
        store=store,
        catalog=catalog,
        uploads=uploads,
        qdrant=qdrant,
        upload_root=tmp_path / "uploads",
    )
    try:
        assert await worker.process_one() is True
        template_result = await store.document_tools.get_job(
            user_id=owner.user_id,
            job_id=template_job.job_id,
        )
        pdf_result = await store.document_tools.get_job(
            user_id=owner.user_id,
            job_id=pdf_job.job_id,
        )
        assert template_result is not None and template_result[0].status == "succeeded"
        assert pdf_result is not None and pdf_result[0].status == "queued"
    finally:
        uploads.close()
        store.close()


@pytest.mark.asyncio
async def test_cancelled_worker_preserves_committed_output_for_restart(
    tmp_path: Path,
) -> None:
    store = ApplicationStore(tmp_path / "app.sqlite")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    catalog = DocumentTemplateCatalog()
    qdrant = _Qdrant()
    owner, _, job = await _approved_job(store, catalog)
    worker = _worker(
        store=store,
        catalog=catalog,
        uploads=uploads,
        qdrant=qdrant,
        upload_root=tmp_path / "uploads",
    )
    original_publish = store.document_tools.publish

    async def interrupt_after_file_commit(**_kwargs):
        raise asyncio.CancelledError

    store.document_tools.publish = interrupt_after_file_commit  # type: ignore[method-assign]
    try:
        with pytest.raises(asyncio.CancelledError):
            await worker.process_one()
        rows = await uploads.list_user(owner.storage_namespace)
        assert len(rows) == 1
        expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"audrey:{job.job_id}"))
        assert rows[0]["file_id"] == expected_id
        destination = (
            tmp_path
            / "uploads"
            / sanitize_user(owner.storage_namespace)
            / f"{expected_id}.docx"
        )
        assert destination.is_file()

        # Model a process restart after My Files committed but before the app
        # state publication. The expired lease is reclaimed and the worker
        # verifies and publishes the same deterministic file without a duplicate.
        with sqlite3.connect(tmp_path / "app.sqlite") as connection:
            connection.execute(
                "UPDATE app_document_jobs SET lease_expires_at = ? WHERE job_id = ?",
                ("2000-01-01T00:00:00+00:00", job.job_id),
            )
        store.document_tools.publish = original_publish  # type: ignore[method-assign]
        assert await worker.process_one() is True
        finished = await store.document_tools.get_job(
            user_id=owner.user_id,
            job_id=job.job_id,
        )
        assert finished is not None and finished[0].status == "succeeded"
        assert len(await uploads.list_user(owner.storage_namespace)) == 1
        assert destination.is_file()
    finally:
        store.document_tools.publish = original_publish  # type: ignore[method-assign]
        uploads.close()
        store.close()
