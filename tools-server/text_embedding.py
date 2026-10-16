"""Pinned text-vector transport and collection guards for the sidecar."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from qdrant_client.http import models as qm

try:
    from embedding_profile import EmbeddingProfile, load_settings, profile_from_mapping
except ModuleNotFoundError as error:
    if error.name != "embedding_profile":
        raise
    # Laptop tests import the same source that the image copies to /app.
    from audrey.embedding_profile import EmbeddingProfile, load_settings, profile_from_mapping


class TextEmbeddingError(RuntimeError):
    """The configured vector space cannot be used safely."""


async def verify_digest(http: httpx.AsyncClient, profile: EmbeddingProfile) -> None:
    response = await http.get("/api/tags")
    response.raise_for_status()
    body = response.json()
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        raise TextEmbeddingError("Embedding model identity is unavailable")
    matches = [model for model in models if isinstance(model, dict)
               and (model.get("name") or model.get("model")) == profile.model]
    if len(matches) != 1 or matches[0].get("digest") != profile.digest:
        raise TextEmbeddingError("Installed embedding model differs from its pinned digest")


async def embed_text(
    http: httpx.AsyncClient, profile: EmbeddingProfile, text: str, *,
    query: bool = False, keep_alive: str = "24h", timeout_s: float = 8.0,
) -> list[float]:
    """Format, embed and validate one input within one overall deadline."""
    try:
        async with asyncio.timeout(timeout_s):
            await verify_digest(http, profile)
            payload: dict[str, Any] = {
                "model": profile.model,
                "input": [profile.query(text) if query else profile.document(text)],
                "truncate": False,
            }
            if keep_alive:
                payload["keep_alive"] = keep_alive
            response = await http.post("/api/embed", json=payload, timeout=timeout_s)
            response.raise_for_status()
            body = response.json()
            vectors = body.get("embeddings") if isinstance(body, dict) else None
            output = profile.normalize_vectors(vectors, 1)
            # A tag can change while Ollama is processing the request.
            await verify_digest(http, profile)
            return output[0]
    except TextEmbeddingError:
        raise
    except (httpx.HTTPError, ValueError, TypeError, TimeoutError) as error:
        raise TextEmbeddingError(
            f"Embedding request failed ({type(error).__name__})"
        ) from error


def validate_collection(info, profile: EmbeddingProfile) -> None:
    vectors = info.config.params.vectors
    metadata = info.config.metadata or {}
    if (not isinstance(vectors, qm.VectorParams)
            or vectors.size != profile.dimensions
            or vectors.distance != qm.Distance.COSINE
            or metadata.get("audrey_embedding") != profile.metadata["audrey_embedding"]):
        raise TextEmbeddingError(
            "Collection embedding space differs from configuration; run text reindex migration"
        )


__all__ = ["EmbeddingProfile", "TextEmbeddingError", "embed_text", "load_settings",
           "profile_from_mapping", "validate_collection"]
