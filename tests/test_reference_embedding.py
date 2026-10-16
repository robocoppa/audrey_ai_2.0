"""Reference-only model identity, transport contracts and invalid vector boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml

from audrey.embedding_profile import QWEN_QUERY_PREFIX
from audrey.kb.embed import TextEmbedder, text_embedder
from audrey.models.ollama import OllamaClient
from audrey.references.embedding import (
    EmbeddingProfile,
    ReferenceEmbedder,
    reference_embedder,
)
from audrey.references.library import ReferenceUnavailable

_MODEL = "qwen3-embedding:4b"
_DIGEST = "b" * 64


def _embedder(vector=None):
    profile = EmbeddingProfile(_MODEL, _DIGEST, 3, "", QWEN_QUERY_PREFIX)
    client = SimpleNamespace(
        tags=AsyncMock(return_value=[{"name": _MODEL, "digest": _DIGEST}]),
        embed=AsyncMock(return_value=[[3.0, 4.0, 0.0] if vector is None else vector]),
    )
    return ReferenceEmbedder(client, profile), client


async def test_profile_normalizes_vectors_and_keeps_no_truncation_contract():
    embedder, client = _embedder()
    assert await embedder.embed_one("query") == [0.6, 0.8, 0.0]
    client.embed.assert_awaited_once_with(
        model=_MODEL,
        texts=["query"],
        timeout_s=24.0,
        keep_alive="10m",
        truncate=False,
    )


@pytest.mark.parametrize(
    "vector",
    [
        [0.0, 0.0, 0.0],
        [1.0],
        [float("nan"), 0.0, 0.0],
        [float("inf"), 0.0, 0.0],
        [True, 0.0, 0.0],
        ["1", 0.0, 0.0],
    ],
)
async def test_invalid_vectors_are_never_written_or_queried(vector):
    embedder, _ = _embedder(vector)
    with pytest.raises(ReferenceUnavailable, match="vector"):
        await embedder.embed_one("query")


@pytest.mark.parametrize("tags", [[], [{"name": _MODEL, "digest": "c" * 64}]])
async def test_missing_or_replaced_weights_fail_before_embedding(tags):
    embedder, client = _embedder()
    client.tags.return_value = tags
    with pytest.raises(ReferenceUnavailable, match="digest"):
        await embedder.embed_one("query")
    client.embed.assert_not_awaited()


async def test_each_batch_rechecks_the_model_tag():
    embedder, client = _embedder()
    embedder.batch_size = 1
    client.tags.side_effect = [
        [{"name": _MODEL, "digest": _DIGEST}],
        [{"name": _MODEL, "digest": "c" * 64}],
    ]
    with pytest.raises(ReferenceUnavailable, match="digest"):
        await embedder.embed_many(["a", "b"])
    assert client.embed.await_count == 1


def test_shipped_reference_and_kb_factories_share_the_canonical_profile():
    raw = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    reference = reference_embedder(None, raw)
    assert reference.model == "qwen3-embedding:4b"
    assert (
        reference.profile.digest
        == "df5bd2e3c74cd8d069d21dc038f1b359fcdc9458fce1c99bd43c9eb1518ff907"
    )
    assert reference.profile.dimensions == 2560
    assert reference.profile.document_prefix == ""
    assert reference.profile.query_prefix == QWEN_QUERY_PREFIX
    assert text_embedder(None, raw).profile == reference.profile
    assert TextEmbedder(None).model == "nomic-embed-text"
    assert raw["reference_libraries"]["search"] == {
        "min_dense_score": 0.54,
        "min_term_overlap": None,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"digest": "invalid"},
        {"dimensions": True},
        {"dimensions": 0},
        {"document_prefix": "search_document: "},
        {"query_prefix": ""},
    ],
)
def test_incomplete_or_wrong_reference_profile_is_rejected(change):
    config = {
        "model": _MODEL,
        "digest": _DIGEST,
        "dimensions": 2560,
        "document_prefix": "",
        "query_prefix": QWEN_QUERY_PREFIX,
        **change,
    }
    with pytest.raises(ValueError):
        reference_embedder(None, {"embedding": config})


async def test_reference_transport_sends_false_and_does_not_change_default_kb_payload():
    seen = []

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": _MODEL, "digest": _DIGEST}]})
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0]]})

    client = OllamaClient("http://ollama.test")
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    try:
        await ReferenceEmbedder(
            client, EmbeddingProfile(_MODEL, _DIGEST, 3, "", QWEN_QUERY_PREFIX)
        ).embed_one("q")
        await TextEmbedder(client).embed_one("q")
    finally:
        await client.aclose()
    assert seen[0]["truncate"] is False
    assert "truncate" not in seen[1]
    assert seen[1]["model"] == "nomic-embed-text"


async def test_weights_changed_during_query_are_rejected_before_returning_vectors():
    embedder, client = _embedder()

    async def replace_during_embedding(**_):
        client.tags.return_value[0]["digest"] = "c" * 64
        return [[3.0, 4.0, 0.0]]

    client.embed.side_effect = replace_during_embedding
    with pytest.raises(ReferenceUnavailable, match="digest"):
        await embedder.embed_one("query")
    assert client.embed.await_count == 1
