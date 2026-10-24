"""Verify selected Private document passages without model or network calls."""
from __future__ import annotations

import asyncio
import hashlib
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from audrey.app_state.tax_cases import TaxCaseError
from audrey.kb.private_documents import private_document_source, private_text_collection
from audrey.private_processing import prepare_native_private_runtime
from audrey.processing_policy import ProcessingPolicyError


def _checksum(path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pdf_page(path, page: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path), strict=True)
    if reader.is_encrypted and not reader.decrypt(""):
        raise TaxCaseError("The requested document page could not be verified.")
    if not 1 <= page <= len(reader.pages) <= 2000:
        raise TaxCaseError("The requested document page could not be verified.")
    text = reader.pages[page - 1].extract_text()
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise TaxCaseError("The requested document page could not be verified.")
    return text


async def private_source_identity(app: Any, principal: Any, file_id: str) -> dict[str, str]:
    """Expose an opaque source identity only to the bound local case worker."""
    async with private_document_source(app, principal, file_id) as source:
        checksum = await asyncio.to_thread(_checksum, source.path)
    return {"file_id": file_id, "checksum": checksum}


async def validate_fact_source(app: Any, principal: Any, conversation_id: str,
                               payload: dict[str, Any]) -> dict[str, Any]:
    """Check exact page text and original digest from canonical selected files.

    A matching passage establishes where extraction came from. The proposed
    normalized value still needs the user's confirmation; this is not a tax
    eligibility or interpretation check.
    """
    if payload.get("kind") != "extracted":
        return payload
    store = app.state.application_store
    records = await store.conversations.list_messages(
        user_id=principal.user_id, conversation_id=conversation_id,
    )
    if records is None:
        raise TaxCaseError("Document provenance could not be verified.")
    selected = {attachment.file_id for message in records for attachment in message.attachments}
    snapshot = await store.projects.snapshot_for_conversation(
        user_id=principal.user_id, conversation_id=conversation_id,
    )
    if snapshot is not None:
        selected.update(item.file_id for item in snapshot[1])
    provenance = payload["provenance"]
    if provenance["file_id"] not in selected:
        raise TaxCaseError("Choose this Private document in the conversation or Project first.")
    async with private_document_source(app, principal, provenance["file_id"]) as source:
        checksum = await asyncio.to_thread(_checksum, source.path)
        if checksum != provenance["checksum"]:
            raise TaxCaseError("The document revision does not match the proposed fact.")
        page = provenance["page"]
        if source.mime == "application/pdf":
            # Read the actual original page. A document can contain text that
            # resembles extraction page markers; those strings cannot select
            # a different physical PDF page for provenance.
            passage = await asyncio.to_thread(_pdf_page, source.path, page)
        elif page == 1:
            text_path = source.path.with_suffix(".ocr.txt")
            if (text_path.is_symlink() or text_path.resolve() != text_path.absolute()
                    or text_path.stat().st_size > 8_000_000):
                raise TaxCaseError("Document provenance could not be verified.")
            # Unpaginated text formats have one logical text section.
            passage = await asyncio.to_thread(text_path.read_text, encoding="utf-8")
        else:
            raise TaxCaseError("Unpaginated documents use page 1 for text provenance.")
        if provenance["original_text"] not in passage:
            raise TaxCaseError("The proposed passage was not found on the selected document page.")
    return payload


@asynccontextmanager
async def _project_document_sources(app: Any, principal: Any, file_ids: list[str], conversation_id: str):
    """Keep source locks through project persistence and recheck deletion races."""
    selected = sorted(set(file_ids))
    async with AsyncExitStack() as stack:
        for file_id in selected:
            await stack.enter_async_context(private_document_source(app, principal, file_id))
        try:
            yield selected
        finally:
            # A deletion can mark its tombstone before waiting for the source
            # lock. Prune a concurrently removed link, as the Project file API
            # does, without losing the saved case or original upload.
            store = app.state.application_store
            conversation = await store.conversations.get(
                user_id=principal.user_id, conversation_id=conversation_id,
            )
            for file_id in selected:
                row = await app.state.uploads_db.get_upload(file_id)
                if (not row or row.get("user") != principal.storage_namespace
                        or not row.get("private_required") or row.get("kind") != "text"
                        or row.get("collection") != private_text_collection(principal.storage_namespace)
                        or row.get("status") != "ready" or row.get("source_freed_at")):
                    if conversation is not None and conversation.project_id is not None:
                        await store.projects.prune_file(user_id=principal.user_id,
                            project_id=conversation.project_id, file_id=file_id)
                    raise ProcessingPolicyError("private_processing_unavailable")


@asynccontextmanager
async def selected_project_document_ids(app: Any, principal: Any, conversation_id: str):
    """Validate existing case/chat documents before the owner groups a Tax review.

    No public lookup, generation or embedding is performed. With no attachments,
    setting up a project needs no model runtime or file service at all.
    """
    store = app.state.application_store
    messages = await store.conversations.list_messages(
        user_id=principal.user_id, conversation_id=conversation_id,
    )
    if messages is None:
        raise ProcessingPolicyError("private_processing_unavailable")
    file_ids = {attachment.file_id for message in messages for attachment in message.attachments}
    case = await store.tax_cases.get(user_id=principal.user_id, conversation_id=conversation_id)
    if case is not None:
        file_ids.update(item["file_id"] for item in case.get("files", []))
    if not file_ids:
        yield []
        return
    runtime = await prepare_native_private_runtime(app)
    async with runtime.scope(app) as execution_app:
        async with _project_document_sources(execution_app, principal, list(file_ids), conversation_id) as selected:
            yield selected


async def link_tax_project_documents(app: Any, principal: Any, conversation_id: str,
                                     file_ids: list[str]) -> None:
    """Attach newly sent Tax review documents before the run pins Project inputs.

    Automatic Private chat attachments keep their existing conversation-only
    behavior. The caller opts in only for an active Tax review in a Project.
    """
    if not file_ids:
        return
    from audrey.app_state.tax_cases import TaxCaseError
    from audrey.routes.files import _settle_private_persistence

    async with _project_document_sources(app, principal, file_ids, conversation_id) as selected:
        try:
            await _settle_private_persistence(app.state.application_store.tax_cases.link_project_files(
                user_id=principal.user_id, conversation_id=conversation_id,
                validated_private_file_ids=selected,
            ))
        except TaxCaseError:
            raise ProcessingPolicyError("private_processing_unavailable") from None
