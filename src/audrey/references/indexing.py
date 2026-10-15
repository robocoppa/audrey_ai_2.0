"""Resumable local embedding builds; only verified releases become searchable."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

from audrey.kb.qdrant import TEXT_DIM, SearchScope, build_text_point
from audrey.references.extract import MAX_OUTPUT_BYTES
from audrey.references.library import ReferenceReadError, ReferenceUnavailable

_RETURN_YEAR = re.compile(r"\bFor use in preparing\s+([12]\d{3})\s+(?:tax\s+)?Returns\b", re.I)
CHUNK_CHARS = 2400
CHUNK_OVERLAP = 200


def applicability(text: str, sha256: str) -> dict:
    """Only an explicit first-page return-year statement establishes this field."""
    matches = list(_RETURN_YEAR.finditer(re.sub(r"\s+", " ", text[:2000])))
    years = {int(match[1]) for match in matches}
    if len(years) != 1:
        return {"applicable_tax_years": None, "applicability_basis": "unknown"}
    return {
        "applicable_tax_years": sorted(years),
        "applicability_basis": "explicit_return_year",
        "verification_page": 1,
        "verification_quote": matches[0][0],
        "verification_sha256": sha256,
    }


def chunks(pages) -> Iterator[dict]:
    index = 0
    for page in pages:
        text = page["text"]
        offset = 0
        while offset < len(text):
            end = min(offset + CHUNK_CHARS, len(text))
            selected = text[offset:end]
            if selected.strip():
                yield {
                    "chunk_index": index,
                    "page": page["page"],
                    "offset": offset,
                    "text": selected,
                }
                index += 1
            if end == len(text):
                break
            offset = end - CHUNK_OVERLAP


def _extract(asset: dict):
    """A bounded spool avoids holding a complete library or PDF set in memory."""
    output = tempfile.TemporaryFile()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "audrey.references.extract"],
            input=json.dumps(asset).encode(),
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=90,
            check=False,
        )
        if completed.returncode or output.tell() > MAX_OUTPUT_BYTES:
            raise ReferenceReadError("PDF text extraction failed within its resource limits")
        output.seek(0)
        first = json.loads(output.readline())
        if first["page"] != 1:
            raise ValueError
        return output, first
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError):
        output.close()
        raise ReferenceReadError("PDF text extraction failed within its resource limits") from None
    except BaseException:
        output.close()
        raise


def _pages(output):
    output.seek(0)
    for line in output:
        yield json.loads(line)


def _manifest(registry, library_id, catalog, product_numbers, maximum):
    eligible = 0
    selected = []
    products = {item.casefold() for item in product_numbers}
    for item in registry.iter_index_documents(library_id, catalog=catalog):
        eligible += 1
        if products and item["product_number"].casefold() not in products:
            continue
        if maximum is None or len(selected) < maximum:
            selected.append(item)
    return eligible, selected


async def build_reference_index(
    registry,
    releases,
    qdrant,
    embedder,
    *,
    library_id="irs-federal",
    catalog="current",
    max_documents=None,
    product_numbers=(),
    release_id=None,
    batch_size=16,
    rebuild=False,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    if max_documents is not None and (isinstance(max_documents, bool) or max_documents < 1):
        raise ValueError("max_documents must be positive")
    if not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be between 1 and 32")
    products = sorted({value.strip() for value in product_numbers if value.strip()})
    if len(products) > 20 or any(len(value) > 100 for value in products):
        raise ValueError("Product selection is too large")
    eligible, selected = await asyncio.to_thread(
        _manifest,
        registry,
        library_id,
        catalog,
        products,
        max_documents,
    )
    if not selected:
        raise ReferenceUnavailable("No permitted documents match the index selection")
    # Missing original revisions cannot acquire a fake checksum or a release.
    if any(not item.get("revision_sha256") for item in selected):
        raise ReferenceUnavailable(
            "Selected catalog includes undownloaded originals; update the archive first"
        )
    model = str(embedder.model)
    prefix = "search_document: " if model.split(":")[0] == "nomic-embed-text" else ""
    provenance = {
        "asof_year": registry.asof_year,
        "history_years": registry.history_years,
        "embedding_model": model,
        "document_prefix": prefix,
        "query_prefix": "search_query: " if prefix else "",
        "chunk_policy": 1,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "applicability_policy": "explicit_first_page_return_year_v1",
    }
    coverage = {
        "eligible_total": eligible,
        "selected_documents": len(selected),
        "subset": len(selected) < eligible,
        "product_numbers": products,
        "max_documents": max_documents,
        "text_layer_only": True,
    }
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode())
    for item in selected:
        digest.update(
            json.dumps(
                {key: value for key, value in item.items() if key != "_asset"}, sort_keys=True
            ).encode()
        )
    published = await asyncio.to_thread(releases.published, library_id, catalog)
    if (
        not rebuild
        and release_id is None
        and published is not None
        and published["fingerprint"] == digest.hexdigest()
    ):
        collection = published["collection"]
        if (
            await qdrant.collection_exists(collection)
            and await qdrant.has_sparse(collection)
            and await qdrant.count_text(
                collection=collection,
                scope=SearchScope(
                    reference_library_id=library_id,
                    reference_release_id=published["id"],
                    reference_catalog=catalog,
                    reference_status="active",
                ),
            )
            == published["chunk_count"]
        ):
            return {
                "schema": 1,
                "status": "published",
                "unchanged": True,
                "library_id": library_id,
                "catalog": catalog,
                "release_id": published["id"],
                "coverage": published["coverage"],
                "indexed_documents": published["indexed_documents"],
                "no_text_documents": published["no_text_documents"],
                "failed_documents": 0,
                "pending_documents": 0,
                "chunks": published["chunk_count"],
                "resumed_documents": published["expected_documents"],
                "failures": [],
            }
    release = await asyncio.to_thread(
        releases.begin,
        library_id,
        catalog,
        fingerprint=digest.hexdigest(),
        provenance=provenance,
        coverage=coverage,
        documents=selected,
        release_id=release_id,
    )
    identifier, collection = release["id"], release["collection"]
    report = {
        "schema": 1,
        "status": "failed",
        "library_id": library_id,
        "catalog": catalog,
        "release_id": identifier,
        "coverage": coverage,
        "resumed_documents": 0,
        "failures": [],
    }

    def emit(event):
        if progress:
            progress(event)

    emit({"event": "index_started", "release_id": identifier, **coverage})
    try:
        await qdrant.ensure_collection(collection, dim=TEXT_DIM)
        if not await qdrant.has_sparse(collection):
            raise ReferenceUnavailable(
                "Reference collection must support keyword and vector search"
            )
        for number, original in enumerate(selected, 1):
            document = {key: value for key, value in original.items() if key != "_asset"}
            doc_id = document["document_id"]
            checkpoint = await asyncio.to_thread(releases.document, identifier, doc_id)
            if checkpoint["status"] in {"indexed", "no_text"}:
                existing_count = await qdrant.count_text(
                    collection=collection,
                    scope=SearchScope(
                        reference_library_id=library_id,
                        reference_release_id=identifier,
                        reference_catalog=catalog,
                        reference_document_ids=[doc_id],
                        reference_status="active",
                    ),
                )
                if existing_count == checkpoint["chunk_count"]:
                    report["resumed_documents"] += 1
                    continue
            source = f"reference:{library_id}:{identifier}:{doc_id}:{document['revision_sha256']}"
            # A prior interrupted attempt can have remote points but no checkpoint.
            await qdrant.delete_by_source(source, collection=collection)
            try:
                if document.get("status") == "failed" or not document.get("original_available"):
                    raise ReferenceReadError(
                        "Retained original is missing, changed or failed refresh"
                    )
                output, first = await asyncio.to_thread(_extract, original["_asset"])
            except (ReferenceReadError, ReferenceUnavailable) as exc:
                reason = str(exc)[:300]
                await asyncio.to_thread(
                    releases.put_document,
                    identifier,
                    doc_id,
                    metadata=document,
                    page_count=0,
                    status="failed",
                    chunks=(),
                    error=reason,
                )
                if len(report["failures"]) < 20:
                    report["failures"].append(
                        {
                            "document_id": doc_id,
                            "product_number": document["product_number"],
                            "source_url": document["source_url"],
                            "error": reason,
                        }
                    )
                emit(
                    {
                        "event": "document_failed",
                        "completed": number,
                        "selected": len(selected),
                        "document_id": doc_id,
                        "error": reason,
                    }
                )
                continue
            document.update(applicability(first["text"], document["revision_sha256"]))
            # Only one bounded PDF's chunks reside in memory at once (<=32 MiB text).
            with output:
                pieces = list(chunks(_pages(output)))
            for start in range(0, len(pieces), batch_size):
                batch = pieces[start : start + batch_size]
                vectors = await embedder.embed_many([prefix + piece["text"] for piece in batch])
                if len(vectors) != len(batch) or any(len(vector) != TEXT_DIM for vector in vectors):
                    raise ReferenceUnavailable(
                        "Embedding output does not match reference vector dimensions"
                    )
                points = [
                    build_text_point(
                        source=source,
                        chunk_idx=piece["chunk_index"],
                        text=piece["text"],
                        vector=vector,
                        mtime=0,
                        sparse=True,
                        extra={
                            "library_id": library_id,
                            "release_id": identifier,
                            "catalog": catalog,
                            "status": "active",
                            "document_id": doc_id,
                            "revision_sha256": document["revision_sha256"],
                            "page": piece["page"],
                            "offset": piece["offset"],
                            "char_end": piece["offset"] + len(piece["text"]),
                            "applicable_tax_years": document["applicable_tax_years"] or [],
                        },
                    )
                    for piece, vector in zip(batch, vectors, strict=True)
                ]
                await qdrant.upsert_text(points, collection=collection)
            await asyncio.to_thread(
                releases.put_document,
                identifier,
                doc_id,
                metadata=document,
                page_count=first["total_pages"],
                status="indexed" if pieces else "no_text",
                chunks=pieces,
            )
            emit(
                {
                    "event": "document_indexed",
                    "completed": number,
                    "selected": len(selected),
                    "document_id": doc_id,
                    "chunks": len(pieces),
                }
            )
        release = await asyncio.to_thread(releases.release, identifier)
        if release["failed_documents"]:
            raise ReferenceUnavailable("Source extraction failures prevent publication")
        # Ensure no collector/catalog update changed the selected manifest mid-build.
        _, current = await asyncio.to_thread(
            _manifest, registry, library_id, catalog, products, max_documents
        )
        live_digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode())
        for item in current:
            live_digest.update(
                json.dumps(
                    {key: value for key, value in item.items() if key != "_asset"}, sort_keys=True
                ).encode()
            )
        if live_digest.hexdigest() != release["fingerprint"]:
            raise ReferenceUnavailable(
                "Reference source manifest changed during the build; rebuild it"
            )
        count = await qdrant.count_text(
            collection=collection,
            scope=SearchScope(
                reference_library_id=library_id,
                reference_release_id=identifier,
                reference_catalog=catalog,
                reference_status="active",
            ),
        )
        report["verified_chunks"] = count
        if count != release["chunk_count"]:
            raise ReferenceUnavailable(
                "Remote chunk count does not match the release; rerun to repair checkpoints"
            )
        release = await asyncio.to_thread(releases.publish, identifier, verified_chunks=count)
        report["status"] = "published"
    except asyncio.CancelledError:
        await asyncio.to_thread(
            releases.fail, identifier, "Indexing interrupted; checkpoints can resume"
        )
        raise
    except Exception as exc:  # noqa: BLE001 — a failed build never swaps publication
        reason = (
            str(exc)
            if isinstance(exc, (ReferenceReadError, ReferenceUnavailable))
            else type(exc).__name__
        )
        await asyncio.to_thread(releases.fail, identifier, reason[:1000])
        report["error"] = reason[:300]
    release = await asyncio.to_thread(releases.release, identifier)
    report.update(
        coverage=release["coverage"],
        indexed_documents=release["indexed_documents"],
        no_text_documents=release["no_text_documents"],
        failed_documents=release["failed_documents"],
        pending_documents=release["pending_documents"],
        chunks=release["chunk_count"],
    )
    return report
