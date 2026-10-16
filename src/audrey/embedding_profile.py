"""Portable text embedding identity shared by the backend, sidecar and index jobs.

This module has no application imports so the sidecar can use the exact same
format, identity and vector validation rules from its own image.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

QWEN_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)
_SHA = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """The weights, native width and formats that define one vector space."""

    model: str
    digest: str
    dimensions: int
    document_prefix: str
    query_prefix: str

    def __post_init__(self):
        if (
            not isinstance(self.model, str) or not self.model.strip()
            or not isinstance(self.digest, str) or not _SHA.fullmatch(self.digest)
            or isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int)
            or not 1 <= self.dimensions <= 8192
            or not isinstance(self.document_prefix, str)
            or not isinstance(self.query_prefix, str)
            or len(self.document_prefix) > 1000 or len(self.query_prefix) > 1000
        ):
            raise ValueError("Invalid text embedding profile")

    @property
    def provenance(self):
        return {
            "embedding_model": self.model,
            "embedding_digest": self.digest,
            "embedding_dimensions": self.dimensions,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
            "embedding_truncate": False,
            "embedding_normalization": "unit_l2_v1",
        }

    @property
    def identity(self):
        return hashlib.sha256(
            json.dumps(self.provenance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def metadata(self):
        return {"audrey_embedding": {**self.provenance, "embedding_identity": self.identity}}

    def document(self, text: str) -> str:
        return self.document_prefix + text

    def query(self, text: str) -> str:
        return self.query_prefix + text

    def normalize_vectors(self, vectors, expected_count: int):
        if not isinstance(vectors, list) or len(vectors) != expected_count:
            raise ValueError("Embedding returned an invalid vector count")
        output = []
        for vector in vectors:
            if (
                not isinstance(vector, list) or len(vector) != self.dimensions
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in vector)
            ):
                raise ValueError("Embedding returned an invalid vector")
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Embedding returned a zero or invalid vector")
            output.append([value / norm for value in vector])
        return output


def profile_from_mapping(settings: Mapping) -> EmbeddingProfile:
    if not isinstance(settings, Mapping):
        raise ValueError("Text embedding settings must be an object")
    profile = EmbeddingProfile(
        model=settings.get("model"), digest=settings.get("digest"),
        dimensions=settings.get("dimensions"),
        document_prefix=settings.get("document_prefix"),
        query_prefix=settings.get("query_prefix"),
    )
    if profile.model.split(":")[0] == "qwen3-embedding" and (
        profile.document_prefix != "" or profile.query_prefix != QWEN_QUERY_PREFIX
    ):
        raise ValueError("Qwen embeddings require raw documents and the tested query instruction")
    return profile


def load_settings(path: str | Path) -> dict:
    """Read and validate the one canonical text_embedding block."""
    with Path(path).open(encoding="utf-8") as stream:
        body = yaml.safe_load(stream)
    if not isinstance(body, dict) or not isinstance(body.get("text_embedding"), dict):
        raise ValueError("Configuration must contain text_embedding settings")
    settings = dict(body["text_embedding"])
    profile_from_mapping(settings)
    return settings


def load_profile(path: str | Path) -> EmbeddingProfile:
    return profile_from_mapping(load_settings(path))
