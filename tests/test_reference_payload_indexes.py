"""Reference payload indexes remain additive and precede exact resume checks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import httpx
import pytest
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse
from test_reference_indexing import services as reference_services_fixture

from audrey.kb import qdrant as qdrant_module
from audrey.kb.qdrant import QdrantKB
from audrey.references.indexing import build_reference_index

services = reference_services_fixture
_COLLECTION = "audrey_reference_irs-federal_" + "a" * 64
_FIELDS = ("document_id", "release_id", "source")


@pytest.fixture
def stub_kb(monkeypatch):
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(payload_schema={})
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **_: client)
    return QdrantKB(host="unused", port=0), client


async def test_new_reference_indexes_use_only_exact_keyword_fields_and_wait(stub_kb):
    kb, client = stub_kb
    await kb.ensure_reference_payload_indexes(_COLLECTION)
    assert client.create_payload_index.call_args_list == [
        call(collection_name=_COLLECTION, field_name=field,
             field_schema=qm.PayloadSchemaType.KEYWORD, wait=True)
        for field in _FIELDS
    ]
    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()
    client.delete.assert_not_called()


async def test_reference_indexes_skip_compatible_fields_and_are_idempotent(stub_kb):
    kb, client = stub_kb
    schema = {"document_id": SimpleNamespace(data_type=qm.PayloadSchemaType.KEYWORD)}
    client.get_collection.return_value.payload_schema = schema

    def create(**kwargs):
        schema[kwargs["field_name"]] = SimpleNamespace(data_type=kwargs["field_schema"])

    client.create_payload_index.side_effect = create
    await kb.ensure_reference_payload_indexes(_COLLECTION)
    await kb.ensure_reference_payload_indexes(_COLLECTION)
    assert [entry.kwargs["field_name"] for entry in client.create_payload_index.call_args_list] == [
        "release_id", "source",
    ]
    assert client.get_collection.call_count == 2


@pytest.mark.parametrize("field", _FIELDS)
async def test_incompatible_reference_index_is_refused_before_any_mutation(stub_kb, field):
    kb, client = stub_kb
    client.get_collection.return_value.payload_schema = {
        field: SimpleNamespace(data_type=qm.PayloadSchemaType.INTEGER),
    }
    with pytest.raises(ValueError, match="index schema is incompatible"):
        await kb.ensure_reference_payload_indexes(_COLLECTION)
    client.create_payload_index.assert_not_called()


@pytest.mark.parametrize("name", [
    "kb_text", "kb_user_text_alice", "kb_memory", "audrey_reference_",
    "audrey_reference_../private", "audrey_reference_irs-federal_" + "z" * 64,
    "audrey_reference_irs-federal\n", None,
])
async def test_reference_index_setup_cannot_touch_private_or_invalid_collections(stub_kb, name):
    kb, client = stub_kb
    with pytest.raises(ValueError, match="require a reference collection"):
        await kb.ensure_reference_payload_indexes(name)
    client.get_collection.assert_not_called()
    client.create_payload_index.assert_not_called()


@pytest.mark.parametrize("stage", ["read", "write"])
async def test_reference_payload_index_transport_failures_surface(stub_kb, stage):
    kb, client = stub_kb
    target = client.get_collection if stage == "read" else client.create_payload_index
    target.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError):
        await kb.ensure_reference_payload_indexes(_COLLECTION)


@pytest.mark.parametrize("status", [400, 409, 503])
async def test_reference_payload_index_server_errors_are_not_silently_ignored(stub_kb, status):
    kb, client = stub_kb
    client.create_payload_index.side_effect = UnexpectedResponse(
        status_code=status, reason_phrase="", headers=httpx.Headers({}),
        content=b'{"status":{"error":"index already exists or is incompatible"}}',
    )
    with pytest.raises(UnexpectedResponse):
        await kb.ensure_reference_payload_indexes(_COLLECTION)


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant:UserWarning")
async def test_reference_index_setup_precedes_sparse_and_exact_checkpoint_checks(services, monkeypatch):
    registry, releases, qdrant, embedder, _, _ = services
    # Stop after all documents are saved, before the release is published.
    embedder.verify_identity.side_effect = [None, RuntimeError("interrupted before publication")]
    first = await build_reference_index(registry, releases, qdrant, embedder)
    assert first["status"] == "failed" and first["indexed_documents"] == 2, first
    embedder.verify_identity.side_effect = None
    events = []
    original_ensure = qdrant.ensure_collection
    original_indexes = qdrant.ensure_reference_payload_indexes
    original_sparse = qdrant.has_sparse
    original_count = qdrant.count_text

    async def ensure(*args, **kwargs):
        events.append("collection")
        await original_ensure(*args, **kwargs)

    async def indexes(*args, **kwargs):
        assert events == ["collection"]
        await original_indexes(*args, **kwargs)
        events.append("indexes")

    async def sparse(*args, **kwargs):
        assert "indexes" in events
        events.append("sparse")
        return await original_sparse(*args, **kwargs)

    async def count(*args, **kwargs):
        assert "indexes" in events
        events.append("exact_count")
        return await original_count(*args, **kwargs)

    monkeypatch.setattr(qdrant, "ensure_collection", ensure)
    monkeypatch.setattr(qdrant, "ensure_reference_payload_indexes", indexes)
    monkeypatch.setattr(qdrant, "has_sparse", sparse)
    monkeypatch.setattr(qdrant, "count_text", count)
    embedder.embed_many.reset_mock()
    resumed = await build_reference_index(
        registry, releases, qdrant, embedder, release_id=first["release_id"],
    )
    assert resumed["status"] == "published", resumed
    assert resumed["resumed_documents"] == first["indexed_documents"] == 2
    assert events[:3] == ["collection", "indexes", "sparse"]
    assert events.count("exact_count") == 3  # Two checkpoints and final publication count.
    embedder.embed_many.assert_not_awaited()
