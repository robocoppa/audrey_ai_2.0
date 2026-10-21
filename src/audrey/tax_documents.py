"""Verify selected Private document passages without model or network calls."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from audrey.app_state.tax_cases import TaxCaseError
from audrey.kb.private_documents import private_document_source


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
