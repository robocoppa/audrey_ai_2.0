"""Durable in-process worker for the reviewed template-to-DOCX operation."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from pathlib import Path

from audrey.app_state import ApplicationStore, DocumentLeaseError
from audrey.app_state.records import DocumentJobRecord
from audrey.documents.templates import (
    DOCX_MIME,
    TEMPLATE_OPERATION,
    TEMPLATE_WORKER_VERSION,
    DocumentTemplateCatalog,
    DocumentTemplateError,
)
from audrey.kb.embed import TextEmbedder
from audrey.kb.ingest import ingest_user_text_file
from audrey.kb.qdrant import QdrantKB
from audrey.kb.storage_lifecycle import QuotaExceededError, StorageLifecycle, StorageReservation
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import ensure_user_collections, sanitize_user

log = logging.getLogger("audrey.documents.worker")

_LEASE_SECONDS = 120
_MAX_ATTEMPTS = 3
_RESERVATION_MAX_AGE = dt.timedelta(hours=24)


class DocumentTemplateWorker:
    """Claim approved template jobs and publish verified private DOCX files."""

    def __init__(
        self,
        *,
        store: ApplicationStore,
        catalog: DocumentTemplateCatalog,
        uploads_db: UploadsDB,
        storage: StorageLifecycle,
        qdrant: QdrantKB,
        text_embedder: TextEmbedder,
        upload_root: Path,
        max_user_bytes: int,
        chunk_tokens: int = 1_000,
        overlap_tokens: int = 100,
        retry_interval_s: float = 5.0,
    ) -> None:
        self._store = store
        self._catalog = catalog
        self._uploads_db = uploads_db
        self._storage = storage
        self._qdrant = qdrant
        self._text_embedder = text_embedder
        self._upload_root = upload_root
        self._max_user_bytes = max_user_bytes
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._retry_interval_s = max(0.1, retry_interval_s)
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="document-template-worker")
        self.wake()

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def wake(self) -> None:
        self._wake.set()

    async def _run(self) -> None:
        while True:
            self._wake.clear()
            try:
                while await self.process_one():
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("document template worker loop failed")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._retry_interval_s)
            except TimeoutError:
                pass

    async def process_one(self) -> bool:
        lease_id = f"template-{uuid.uuid4().hex}"
        job = await self._store.document_tools.claim_next(
            lease_id=lease_id,
            lease_seconds=_LEASE_SECONDS,
            max_attempts=_MAX_ATTEMPTS,
            operation=TEMPLATE_OPERATION,
        )
        if job is None:
            return False
        try:
            await self._execute(job, lease_id=lease_id)
        except asyncio.CancelledError:
            raise
        except DocumentTemplateError as exc:
            log.warning("document template job %s rejected: %s", job.job_id, exc)
            await self._fail(job, lease_id=lease_id, error_code="invalid_template_request")
        except QuotaExceededError:
            log.info("document template job %s exceeded owner quota", job.job_id)
            await self._fail(job, lease_id=lease_id, error_code="quota_exceeded")
        except Exception:
            log.exception("document template job %s failed", job.job_id)
            await self._fail(job, lease_id=lease_id, error_code="document_publication_failed")
        return True

    async def _fail(
        self,
        job: DocumentJobRecord,
        *,
        lease_id: str,
        error_code: str,
    ) -> None:
        try:
            await self._store.document_tools.fail(
                user_id=job.user_id,
                job_id=job.job_id,
                lease_id=lease_id,
                error_code=error_code,
            )
        except DocumentLeaseError:
            log.info("document template job %s lost its lease before failure", job.job_id)

    async def _execute(self, job: DocumentJobRecord, *, lease_id: str) -> None:
        if job.operation != TEMPLATE_OPERATION or job.output_mime != DOCX_MIME:
            raise DocumentTemplateError("claimed job is not an approved DOCX template operation")
        arguments = self._catalog.validate_arguments(job.arguments)
        namespace = await self._store.active_storage_namespace(user_id=job.user_id)
        if namespace is None:
            raise DocumentTemplateError("document owner is not an active account")
        output_file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"audrey:{job.job_id}"))
        destination = (
            self._upload_root
            / sanitize_user(namespace)
            / f"{output_file_id}.docx"
        )
        existing_rows = await self._uploads_db.list_user(namespace)
        existing = next(
            (row for row in existing_rows if str(row["file_id"]) == output_file_id),
            None,
        )
        committed_to_files = existing is not None
        reservation: StorageReservation | None = None
        indexed = committed_to_files
        try:
            if existing is not None:
                if (
                    str(existing["filename"]) != arguments.filename
                    or str(existing["mime"]) != DOCX_MIME
                    or str(existing["status"]) != "ready"
                ):
                    raise DocumentTemplateError(
                        "deterministic output id is occupied by another file state"
                    )
                content = await asyncio.to_thread(destination.read_bytes)
                rendered = self._catalog.verify(content, arguments)
            else:
                if destination.is_file():
                    content = await asyncio.to_thread(destination.read_bytes)
                    rendered = self._catalog.verify(content, arguments)
                else:
                    rendered = await asyncio.to_thread(self._catalog.render, arguments)
                    await self._write_staged(destination, rendered.content, job_id=job.job_id)
                now = dt.datetime.now(dt.UTC)
                reservation = await self._storage.reserve_single_upload(
                    reservation_id=f"document-{job.job_id}",
                    user=namespace,
                    bytes_=len(rendered.content),
                    max_user_bytes=self._max_user_bytes,
                    now=now.isoformat(),
                    expired_before=(now - _RESERVATION_MAX_AGE).isoformat(),
                )
                text_collection, _image_collection = await ensure_user_collections(
                    self._qdrant,
                    namespace,
                )
                stamp = now.isoformat(timespec="seconds")
                chunks = await ingest_user_text_file(
                    destination,
                    qdrant=self._qdrant,
                    embedder=self._text_embedder,
                    collection=text_collection,
                    user=namespace,
                    file_id=output_file_id,
                    filename=arguments.filename,
                    mime=DOCX_MIME,
                    uploaded_at=stamp,
                    chunk_tokens=self._chunk_tokens,
                    overlap_tokens=self._overlap_tokens,
                )
                if chunks < 1:
                    raise DocumentTemplateError("rendered DOCX produced no readable text")
                indexed = True
                await self._storage.commit_upload(
                    reservation,
                    file_id=output_file_id,
                    filename=arguments.filename,
                    mime=DOCX_MIME,
                    bytes_=len(rendered.content),
                    kind="text",
                    collection=text_collection,
                    chunks=chunks,
                    uploaded_at=stamp,
                    status="ready",
                    max_user_bytes=self._max_user_bytes,
                )
                reservation = None
                committed_to_files = True
            await self._store.document_tools.publish(
                user_id=job.user_id,
                job_id=job.job_id,
                lease_id=lease_id,
                output_file_id=output_file_id,
                filename=arguments.filename,
                mime=DOCX_MIME,
                bytes_count=len(rendered.content),
                content_sha256=rendered.content_sha256,
                worker_version=(
                    f"{TEMPLATE_WORKER_VERSION}:{arguments.template_id}@"
                    f"{arguments.template_sha256[:12]}:verified"
                ),
            )
        except asyncio.CancelledError:
            # Once My Files owns the verified bytes, leave them in place so the
            # next lease can publish the same deterministic output after restart.
            # Earlier interruptions still remove partial indexing and staged bytes.
            if reservation is not None:
                await self._storage.release(reservation)
            if not committed_to_files:
                if indexed:
                    await self._delete_vectors(output_file_id, namespace, job.job_id)
                await asyncio.to_thread(destination.unlink, missing_ok=True)
            raise
        except BaseException:
            if committed_to_files:
                await self._uploads_db.delete_upload(output_file_id, user=namespace)
            if indexed:
                await self._delete_vectors(output_file_id, namespace, job.job_id)
            if reservation is not None:
                try:
                    await self._storage.release(reservation)
                except Exception:
                    log.exception("could not release document reservation for %s", job.job_id)
            await asyncio.to_thread(destination.unlink, missing_ok=True)
            raise

    async def _delete_vectors(self, file_id: str, namespace: str, job_id: str) -> None:
        try:
            collection, _ = await ensure_user_collections(self._qdrant, namespace)
            await self._qdrant.delete_by_file_id(
                file_id,
                user=namespace,
                collection=collection,
            )
        except Exception:
            log.exception("could not compensate document vectors for %s", job_id)

    async def _write_staged(self, destination: Path, content: bytes, *, job_id: str) -> None:
        staging = self._upload_root / ".staging"
        temporary = staging / f"{job_id}.{uuid.uuid4().hex}.docx.tmp"

        def write() -> None:
            staging.mkdir(parents=True, exist_ok=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(content)
            temporary.chmod(0o600)
            temporary.replace(destination)

        try:
            await asyncio.to_thread(write)
        finally:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)


__all__ = ["DocumentTemplateWorker"]
