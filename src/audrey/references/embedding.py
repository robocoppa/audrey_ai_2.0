"""Public-reference adapter for the shared pinned text embedding profile."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from audrey.embedding_profile import (
    EmbeddingProfile,
    profile_from_mapping,
)
from audrey.models.ollama import OllamaClient
from audrey.references.library import ReferenceUnavailable


@dataclass(slots=True)
class ReferenceEmbedder:
    """Checks the shared model and vectors for reference inputs already formatted by callers."""

    ollama: OllamaClient
    profile: EmbeddingProfile
    keep_alive: str = "10m"
    timeout_s: float = 120.0
    query_timeout_s: float = 24.0
    batch_size: int = 16

    @property
    def model(self):
        return self.profile.model

    async def verify_identity(self):
        try:
            tags = await self.ollama.tags()
            matches = [tag for tag in tags if tag.get("name") == self.model]
        except Exception as exc:
            raise ReferenceUnavailable("Reference embedding model identity is unavailable") from exc
        if len(matches) != 1 or matches[0].get("digest") != self.profile.digest:
            raise ReferenceUnavailable(
                "Installed reference embedding model differs from its pinned digest"
            )

    async def embed_one(self, text):
        return (await self.embed_many([text], timeout_s=self.query_timeout_s))[0]

    async def embed_many(self, texts, *, timeout_s=None):
        if not texts:
            return []
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            async with asyncio.timeout(self.timeout_s if timeout_s is None else timeout_s):
                await self.verify_identity()
                batch = texts[offset : offset + self.batch_size]
                output = await self.ollama.embed(
                    model=self.model,
                    texts=batch,
                    timeout_s=self.timeout_s if timeout_s is None else timeout_s,
                    keep_alive=self.keep_alive,
                    truncate=False,
                )
                # A mutable Ollama tag can be replaced while a request is running.
                # Discard those vectors instead of mixing weights in a published space.
                await self.verify_identity()
                try:
                    vectors.extend(self.profile.normalize_vectors(output, len(batch)))
                except ValueError as exc:
                    raise ReferenceUnavailable(str(exc)) from exc
        return vectors


def reference_embedder(ollama, config):
    """Runtime and index jobs consume the canonical profile; old direct callers remain usable."""
    settings = config.get("text_embedding", config.get("embedding", config))
    profile = profile_from_mapping(settings)
    return ReferenceEmbedder(ollama, profile, keep_alive=settings.get("keep_alive", "24h"))
