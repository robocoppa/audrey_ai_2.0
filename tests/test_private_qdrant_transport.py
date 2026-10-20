"""Protected document storage cannot follow proxy or redirect egress paths."""
from __future__ import annotations

import uuid

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import api_client as qdrant_http
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from audrey.kb import qdrant as module


@pytest.mark.parametrize("redirect", [False, True])
async def test_owned_qdrant_data_uses_only_configured_endpoint(monkeypatch, redirect):
    seen, clients = [], []

    def handle(request):
        seen.append(request)
        assert request.url.host == "qdrant"
        if redirect:
            return httpx.Response(307, headers={"location": "https://outside.invalid/collect"})
        return httpx.Response(200, json={"status": "ok", "time": 0.0,
                                        "result": {"operation_id": 1, "status": "completed"}})

    original = httpx.Client

    def client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        result = original(**kwargs)
        clients.append(result)
        return result

    monkeypatch.setenv("HTTP_PROXY", "http://untrusted.invalid:8123")
    monkeypatch.setattr(qdrant_http, "Client", client)
    monkeypatch.setattr(module, "QdrantClient", lambda **kwargs: QdrantClient(check_compatibility=False, **kwargs))
    kb = module.QdrantKB(host="qdrant", port=6333)
    point = models.PointStruct(id=str(uuid.uuid4()), vector=[1.0, 0.0],
                               payload={"text": "PRIVATE_STORAGE_TEST_51", "user": "owner"})
    try:
        if redirect:
            with pytest.raises(UnexpectedResponse):
                await kb.upsert_text([point], collection="kb_private_text_owner")
        else:
            await kb.upsert_text([point], collection="kb_private_text_owner")
        assert len(seen) == 1
        assert b"PRIVATE_STORAGE_TEST_51" in seen[0].content
        assert clients and all(not item.trust_env and not item.follow_redirects for item in clients)
    finally:
        kb.close()
