"""Hybrid search restricted to a published public-reference release."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any

from audrey.kb.fusion import passes_evidence, reciprocal_rank_fusion
from audrey.kb.qdrant import SearchScope
from audrey.references.library import (
    ReferenceLibraryRegistry,
    ReferenceNotFound,
    ReferenceQueryError,
    ReferenceUnavailable,
)

_IDENTIFIER = re.compile(r"[0-9a-f]{64}")


class ReferenceSearchService:
    """Uses one collection/scope for both retrievers; never searches private KBs."""

    def __init__(self, registry, releases, qdrant, embedder, *, config=None):
        self.registry: ReferenceLibraryRegistry = registry
        self.releases = releases
        self.qdrant = qdrant
        self.embedder = embedder
        config = config or {}
        self.min_dense_score = float(config.get("min_dense_score", 0.55))
        self.min_term_overlap = float(config.get("min_term_overlap", 0.5))
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in (
                self.min_dense_score,
                self.min_term_overlap,
            )
        ):
            raise ValueError("Reference retrieval thresholds must be between zero and one")

    async def search(
        self,
        library_id: str,
        query: str,
        *,
        catalog: str | None = "current",
        top_k: int = 3,
        product_number: str | None = None,
        tax_year: int | None = None,
    ) -> dict[str, Any]:
        scope_name = catalog or "current"
        if scope_name not in {"current", "prior", "bulletins"}:
            raise ReferenceQueryError("Unsupported reference catalog")
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ReferenceQueryError("Reference search query must contain 1–1000 characters")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise ReferenceQueryError("Reference search top_k must be between one and five")
        if product_number is not None and (
            not isinstance(product_number, str)
            or not product_number.strip()
            or len(product_number) > 100
        ):
            raise ReferenceQueryError("Reference product number must contain 1–100 characters")
        if tax_year is not None and (
            isinstance(tax_year, bool)
            or not isinstance(tax_year, int)
            or not self.registry.first_history_year <= tax_year <= self.registry.asof_year
        ):
            raise ReferenceQueryError(
                "Requested tax year is outside the permitted recent-year window"
            )

        # Also verifies that the registered source catalog is currently readable.
        # A missing library or archive never broadens this into a general KB query.
        authorized = await asyncio.to_thread(
            self.registry.catalog,
            library_id,
            catalog=scope_name,
            product_number=product_number,
            limit=20,
        )
        release = await asyncio.to_thread(self.releases.published, library_id, scope_name)
        if release is None:
            raise ReferenceUnavailable(
                "This reference scope has no published search index; build it first"
            )
        provenance = release["provenance"]
        model = str(self.embedder.model)
        query_prefix = "search_query: " if model.split(":")[0] == "nomic-embed-text" else ""
        document_prefix = "search_document: " if query_prefix else ""
        expected = {
            "asof_year": self.registry.asof_year,
            "history_years": self.registry.history_years,
            "embedding_model": model,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ReferenceUnavailable(
                "Published reference index policy or embedding model changed; rebuild it"
            )
        response: dict[str, Any] = {
            "library_id": library_id,
            "release_id": release["id"],
            "catalog": scope_name,
            "query": query.strip(),
            "tax_year": tax_year,
            "historical_edition_window": [
                self.registry.first_history_year,
                self.registry.asof_year,
            ],
            "coverage": {
                **release["coverage"],
                "selected_documents": release["expected_documents"],
                "indexed_documents": release["indexed_documents"],
                "no_text_documents": release["no_text_documents"],
            },
            "results": [],
            "returned": 0,
            "notice": "Candidate passages from the selected published scope; results are not an exhaustive document review.",
        }
        document_ids = None
        if product_number:
            document_ids = [item["document_id"] for item in authorized["items"]]
            if not document_ids:
                response["notice"] = (
                    "No permitted document matches this exact product number in the selected catalog."
                )
                return response
            if authorized.get("has_more"):
                # Exact products normally have one current or a few recent editions.
                # Refuse an incomplete ID restriction rather than silently omit more.
                raise ReferenceQueryError(
                    "Exact product lookup has too many editions; narrow the catalog"
                )
        scope = SearchScope(
            reference_library_id=library_id,
            reference_release_id=release["id"],
            reference_catalog=scope_name,
            reference_document_ids=document_ids,
            reference_tax_year=tax_year,
            reference_status="active",
        )
        try:
            if not await self.qdrant.collection_exists(release["collection"]):
                raise ReferenceUnavailable(
                    "Published reference collection is unavailable; rebuild the index"
                )
            if not await self.qdrant.has_sparse(release["collection"]):
                raise ReferenceUnavailable(
                    "Published reference keyword index is unavailable; rebuild the index"
                )
            count = await self.qdrant.count_text(
                collection=release["collection"],
                scope=SearchScope(
                    reference_library_id=library_id,
                    reference_release_id=release["id"],
                    reference_catalog=scope_name,
                    reference_status="active",
                ),
            )
            if count != release["chunk_count"]:
                raise ReferenceUnavailable("Published reference index is incomplete; rebuild it")
            vector = await self.embedder.embed_one(query_prefix + query.strip())
            dense, lexical = await self.qdrant.search_hybrid(
                vector,
                query.strip(),
                top_k=max(20, top_k * 8),
                collection=release["collection"],
                scope=scope,
            )
        except ReferenceUnavailable:
            raise
        except Exception as exc:
            raise ReferenceUnavailable(
                "Reference embedding or search service is unavailable"
            ) from exc

        fused = reciprocal_rank_fusion(dense, lexical, query=query.strip())
        for entry in fused:
            if not passes_evidence(
                entry,
                min_score=self.min_dense_score,
                min_overlap=self.min_term_overlap,
            ):
                continue
            result = await asyncio.to_thread(
                self._validated_passage,
                entry,
                release,
                scope_name,
                document_ids,
                tax_year,
            )
            if result is not None:
                response["results"].append(result)
            if len(response["results"]) == top_k:
                break
        response["returned"] = len(response["results"])
        if not response["results"]:
            response["notice"] = (
                "No matching passage with verified applicability to the requested tax year was found; unknown-year sources are excluded."
                if tax_year is not None
                else "No matching passage was found in the permitted published scope. Other libraries and private files were not searched."
            )
        return response

    def _validated_passage(self, entry, release, catalog, document_ids, tax_year):
        hit, payload = entry.hit, entry.hit.payload
        expected = {
            "library_id": release["library_id"],
            "release_id": release["id"],
            "catalog": catalog,
            "status": "active",
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return None
        identifier = payload.get("document_id")
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
            return None
        if document_ids is not None and identifier not in document_ids:
            return None
        if (
            isinstance(hit.chunk_idx, bool)
            or not isinstance(hit.chunk_idx, int)
            or hit.chunk_idx < 0
        ):
            return None
        chunk = self.releases.chunk(release["id"], identifier, hit.chunk_idx)
        document = self.releases.document(release["id"], identifier)
        if chunk is None or document is None or document.get("status") != "indexed":
            return None
        metadata = document.get("metadata", document)
        if payload.get("revision_sha256") != metadata.get("revision_sha256"):
            return None
        if any(payload.get(key) != chunk.get(key) for key in ("page", "offset")):
            return None
        if hit.text != chunk.get("text") or not hit.text:
            return None
        if payload.get("char_end") != chunk["offset"] + len(chunk["text"]):
            return None
        try:
            live = self.registry.lookup(release["library_id"], identifier, catalog=catalog)
        except ReferenceNotFound:
            return None
        if (
            live.get("status") == "failed"
            or not live.get("original_available")
            or live.get("revision_sha256") != metadata.get("revision_sha256")
        ):
            return None
        years = metadata.get("applicable_tax_years") or []
        if tax_year is not None and tax_year not in years:
            return None
        if payload.get("applicable_tax_years", []) != years:
            return None
        selected = hit.text[:800]
        return {
            "document_id": identifier,
            "revision_sha256": metadata["revision_sha256"],
            "product_number": metadata["product_number"],
            "title": metadata["title"][:300],
            "source_url": metadata["source_url"],
            "page": chunk["page"],
            "offset": chunk["offset"],
            "char_end": chunk["offset"] + len(selected),
            "text": selected,
            "passage_truncated": len(selected) < len(hit.text),
            "catalog": catalog,
            "catalog_edition_year": metadata.get("catalog_edition_year"),
            "applicable_tax_years": years or None,
            "applicability_basis": metadata.get("applicability_basis", "unknown"),
            "applicability_evidence": {
                "page": metadata.get("verification_page"),
                "quote": metadata.get("verification_quote", ""),
            }
            if years
            else None,
            "score": round(entry.score, 6),
        }
