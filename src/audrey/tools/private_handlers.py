"""Owner-bound in-process tools for verified private native requests.

Only this server-created callback map can admit private tool calls. Public HTTP
routes retain their separate restrictions; no caller header establishes privacy.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from audrey.identity import Principal
from audrey.kb.fusion import RRF_K, passes_evidence, reciprocal_rank_fusion
from audrey.kb.qdrant import SearchScope
from audrey.kb.table_analysis import TableAnalysisRequest, analyze_table
from audrey.processing_policy import ProcessingPolicyError, current_processing_policy
from audrey.references.library import ReferenceReadError
from audrey.routes import files as upload_routes
from audrey.routes import references
from audrey.tools.calculator import CalculateRequest, calculate_value
from audrey.tools.private_dispatch import PrivateToolArgumentsError, PrivateToolHandler

log = logging.getLogger(__name__)

SUPPORTED_PRIVATE_TOOL_NAMES = frozenset({
    "calculate", "list_my_files", "get_file_text", "analyze_table", "kb_search",
    "reference_lookup", "read_reference", "reference_search", "create_file",
})


class _OwnerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str = Field(min_length=1, max_length=200)


class _FileTextRequest(_OwnerRequest):
    filename: str = Field(min_length=1, max_length=500)
    artifact: Literal["auto", "document", "summary"] = "auto"
    offset: int = Field(default=0, ge=0, le=20_000_000)


class _SearchRequest(_OwnerRequest):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filename: str | None = Field(default=None, max_length=500)
    artifact: Literal["transcript", "visual", "summary"] | None = None


class _TableRequest(TableAnalysisRequest):
    user: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=500)


def make_private_tool_handlers(
    app: Any, principal: Principal, *, tax_conversation_id: str | None = None,
) -> Mapping[str, PrivateToolHandler]:
    """Close over the authenticated owner and isolated runtime, never model user."""
    policy = current_processing_policy()
    if (
        policy is None or not isinstance(principal, Principal)
        or principal.status != "active" or not principal.user_id
        or not principal.storage_namespace.strip()
    ):
        raise ProcessingPolicyError("private_tool_not_allowed")
    evidence_readings = None
    if tax_conversation_id is not None and "tax_review" in policy.allowed_tools:
        from audrey.tax_review import TaxEvidenceReadings

        evidence_readings = TaxEvidenceReadings()
    request = Request({"type": "http", "app": app})
    user = principal.storage_namespace

    def guard(owner: str | None = None) -> None:
        if current_processing_policy() is not policy or (owner is not None and owner != user):
            raise ProcessingPolicyError("private_tool_not_allowed")

    async def ready_rows() -> list[dict]:
        from audrey.kb.private_documents import private_text_collection

        rows = await upload_routes._get_uploads_db(request).list_user(user)
        ready, invalid = [], 0
        for row in rows:
            if not row.get("private_required") or row.get("status") != "ready" or row.get("kind") != "text":
                continue
            generated = row.get("origin") == "generated" and not row.get("collection")
            if ((not generated and row.get("collection") != private_text_collection(user))
                    or row.get("source_freed_at")):
                invalid += 1
                continue
            ready.append(dict(row))
        if invalid:
            # Never log owner identities, names, paths or record content.
            log.warning("private files: skipped %d ineligible ready records", invalid)
        return ready

    async def owned_row(filename: str) -> dict:
        guard()
        matches = [row for row in await ready_rows()
                   if str(row["filename"]).strip().casefold() == filename.strip().casefold()]
        if not matches:
            raise ProcessingPolicyError("private_processing_unavailable")
        # Names are not unique. Resolve only among eligible Private documents,
        # with a stable tie-break, and tell the caller when newest wins.
        matches.sort(key=lambda row: (str(row["uploaded_at"]), str(row["file_id"])), reverse=True)
        row = matches[0]
        row["_duplicate_filename"] = len(matches) > 1
        return row

    def duplicate_notice(row: dict) -> str:
        return ("Multiple ready Private documents share this name; using the newest upload."
                if row.get("_duplicate_filename") else "")

    async def calculate(args: dict[str, Any]) -> dict:
        guard()
        return calculate_value(CalculateRequest.model_validate(args)).model_dump()

    def reference_body(schema, args, *, count=None, maximum=None):
        # Models sometimes request more results than the declared context budget.
        # Reduce only a positive integer result count, never identity or scope.
        normalized = dict(args)
        value = normalized.get(count)
        capped = (
            count is not None and maximum is not None
            and isinstance(value, int) and not isinstance(value, bool) and value > maximum
        )
        if capped:
            normalized[count] = maximum
        try:
            return schema.model_validate(normalized), capped
        except ValidationError:
            # Pydantic's message includes supplied values, paths and unknown keys.
            raise PrivateToolArgumentsError() from None

    def result_count_notice(result, notice, bound):
        annotated = {**result, "notice": (result.get("notice", "") + " " + notice).strip()}
        try:
            bounded = bound(annotated)
        except ReferenceReadError:
            # A notice must not displace required provenance or turn useful
            # evidence into a failure when metadata already fills the budget.
            return result
        if result.get("result_limit_applied"):
            bounded["result_limit_applied"] = True
        return bounded

    async def lookup(args: dict[str, Any]) -> dict:
        guard()
        body, capped = reference_body(references.CatalogRequest, args, count="limit", maximum=20)
        result = await references.catalog(body, request, None)
        if capped:
            result = result_count_notice(
                result, "Requested result count capped at 20 catalog entries.", references._bounded_catalog,
            )
        return result

    async def read(args: dict[str, Any]) -> dict:
        guard()
        body, _ = reference_body(references.ReadRequest, args)
        result = await references.read(body, request, None)
        return evidence_readings.offer(result) if evidence_readings is not None else result

    async def search(args: dict[str, Any]) -> dict:
        guard()
        body, capped = reference_body(references.SearchRequest, args, count="top_k", maximum=5)
        # The runtime supplies an isolated reference_search whose embedder uses
        # its verified local Ollama transport. No sidecar or web fallback occurs.
        result = await references.search(body, request, None)
        if capped:
            result = result_count_notice(
                result, "Requested result count capped at 5 passages.", references._bounded_search,
            )
        return result

    async def list_files(args: dict[str, Any]) -> dict:
        body = _OwnerRequest.model_validate(args)
        guard(body.user)
        rows = await ready_rows()
        files = []
        for row in rows:
            files.append({
                "filename": str(row["filename"]), "kind": "text", "status": "ready",
                "uploaded_at": str(row["uploaded_at"]), "duration_s": 0.0,
                "failure_reason": "", "waiting_for_s": 0.0,
                "unread_artifacts": ["document", *(["summary"] if row.get("summary") else [])],
            })
        return {"files": files}

    async def file_text(args: dict[str, Any]) -> dict:
        from audrey.kb.private_documents import private_document_source, read_private_document

        body = _FileTextRequest.model_validate(args)
        guard(body.user)
        row = await owned_row(body.filename)
        artifact = "document" if body.artifact == "auto" else body.artifact
        if artifact == "document":
            document = await read_private_document(app, principal, str(row["file_id"]))
            text, filename = document.text, document.filename
        else:
            async with private_document_source(app, principal, str(row["file_id"])) as source:
                current = await upload_routes._get_uploads_db(request).get_upload(str(row["file_id"]))
                text, filename = str(current.get("summary") or ""), source.filename
        offset = min(body.offset, len(text))
        end = min(offset + 2_000, len(text))
        identity = {}
        if (tax_conversation_id is not None and artifact == "document"
                and row.get("origin") != "generated"):
            from audrey.tax_documents import private_source_identity

            identity = await private_source_identity(app, principal, str(row["file_id"]))
        return {
            **identity, "filename": filename, "artifact": artifact, "text": text[offset:end],
            "offset": offset, "next_offset": end if end < len(text) else None,
            "total_chars": len(text),
            "note": " ".join(part for part in (
                "This is a page of the owned document; follow next_offset for the rest." if end < len(text) else "",
                "Generated working output; not independent source evidence." if row.get("origin") == "generated" else "",
                duplicate_notice(row),
            ) if part),
        }

    async def kb_search(args: dict[str, Any]) -> dict:
        from audrey.kb.private_documents import private_text_collection

        body = _SearchRequest.model_validate(args)
        guard(body.user)
        rows = [row for row in await ready_rows() if row.get("origin") != "generated"]
        if body.filename:
            rows = [row for row in rows if str(row["filename"]).strip().casefold() == body.filename.strip().casefold()]
        if not rows:
            return {"query": body.query, "results": [], "notice": "No matching ready Private documents are available; no other files were searched."}
        if body.artifact in {"transcript", "visual"}:
            return {"query": body.query, "results": [], "notice": "Private document search does not include audio or video artifacts."}
        allowed_ids = {str(row["file_id"]) for row in rows}
        scope = SearchScope(user=user, file_ids=sorted(allowed_ids), artifact=body.artifact)
        embedder = getattr(app.state, "text_embedder", None)
        qdrant = getattr(app.state, "qdrant", None)
        if embedder is None or qdrant is None:
            raise ProcessingPolicyError("private_processing_unavailable")
        policy.assert_model_allowed(embedder.model)
        if policy.local_models.get(embedder.model) != embedder.profile.digest:
            raise ProcessingPolicyError("local_model_digest_mismatch")
        collection = private_text_collection(user)
        if not await qdrant.collection_exists(collection):
            raise ProcessingPolicyError("private_processing_unavailable")
        vector = await embedder.embed_one(body.query)
        dense, lexical = await qdrant.search_hybrid(
            vector, body.query, top_k=min(max(body.top_k * 4, 20), 40),
            collection=collection, scope=scope,
        )
        # Check both retrievers' returned identities before any content leaves
        # the callback, independently of the storage filter implementation.
        if any(hit.payload.get("user") != user or hit.payload.get("file_id") not in allowed_ids
               or hit.payload.get("private_required") is not True for hit in [*dense, *lexical]):
            raise ProcessingPolicyError("private_processing_unavailable")
        settings = app.state.cfg.raw.get("kb", {}) or {}
        hybrid = settings.get("hybrid", {}) or {}
        fused = reciprocal_rank_fusion(dense, lexical, rrf_k=int(hybrid.get("rrf_k", RRF_K)), query=body.query)
        floor = float("-inf") if body.filename else float(settings.get("min_score", 0.0))
        overlap = 0.0 if body.filename else float(hybrid.get("min_term_overlap", 0.7))
        kept = [entry for entry in fused if passes_evidence(entry, min_score=floor, min_overlap=overlap)]
        # A deletion/purge begun during retrieval removes the file from the
        # authoritative ready-file list even if Qdrant cleanup is retrying.
        live = {str(row["file_id"]) for row in await ready_rows()}
        seen, first, remaining = set(), [], []
        for entry in kept:
            file_id = str(entry.hit.payload["file_id"])
            if file_id not in live:
                continue
            (remaining if file_id in seen else first).append(entry)
            seen.add(file_id)
        return {"query": body.query, "results": [{
            "score": entry.score, "source": entry.hit.source, "kind": entry.hit.kind,
            "chunk_idx": entry.hit.chunk_idx, "text": entry.hit.text,
            "filename": str(entry.hit.payload.get("filename") or ""),
            "artifact": str(entry.hit.payload.get("artifact") or ""),
        } for entry in (first + remaining)[:body.top_k]], "notice": ""}

    async def table(args: dict[str, Any]) -> dict:
        body = _TableRequest.model_validate(args)
        guard(body.user)
        row = await owned_row(body.filename)
        from audrey.kb.private_documents import private_document_source

        async with private_document_source(app, principal, str(row["file_id"])) as source:
            spec = TableAnalysisRequest.model_validate(body.model_dump(exclude={"user", "filename"}))
            mime = "text/csv" if source.path.suffix.lower() == ".csv" else source.mime
            result = await asyncio.to_thread(analyze_table, source.path, mime, spec)
        return {**result, "filename": str(row["filename"]), "artifact": "table", "file_id": str(row["file_id"]), "note": duplicate_notice(row)}

    async def create_file(args: dict[str, Any]) -> dict:
        guard()
        from audrey.tools.native_files import native_file_handler

        # In task runs this alias is wrapped by the existing step tool budget
        # and lease guard. The native binding supplies conversation/run IDs.
        return await native_file_handler(user_id=user)(args)

    handlers = {
        "create_file": create_file,
        "calculate": calculate, "reference_lookup": lookup, "read_reference": read,
        "reference_search": search, "list_my_files": list_files,
        "get_file_text": file_text, "analyze_table": table, "kb_search": kb_search,
    }
    if tax_conversation_id is not None and "tax_review" in policy.allowed_tools:
        from audrey.tax_review import make_tax_review_handler

        handlers["tax_review"] = make_tax_review_handler(
            app=app, principal=principal, conversation_id=tax_conversation_id,
            evidence_readings=evidence_readings,
        )
    return MappingProxyType({name: handler for name, handler in handlers.items() if name in policy.allowed_tools})


__all__ = ["SUPPORTED_PRIVATE_TOOL_NAMES", "make_private_tool_handlers"]
