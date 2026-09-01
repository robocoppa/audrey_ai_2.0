"""`POST /memory_search` — what it says when it cannot search.

The endpoint used to answer an embedder failure with `200 {"results": []}`,
which is a claim about the user's memory store rather than about the search.
The orchestrator's auto-recall calls this at the top of every request, and an
empty list is its success case, so a stalled embedder was indistinguishable
from a user who had never stored anything.

That mattered on this box in particular: `nomic-embed-text` shares one GPU with
the local panel workers, so it gets evicted from VRAM under load and a query
embed that normally takes ~100ms starts taking seconds. The same root cause is
documented for the KB path in `test_kb_timeout_ladder.py`.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from app import CapabilityRegistry, app  # noqa: E402
from db import EmbedError, MemoryEntry  # noqa: E402


def tools_app_capabilities() -> CapabilityRegistry:
    return CapabilityRegistry.all_available()


class _StubMemory:
    """Stands in for `app.state.memory`. `outcome` is either the hits to return
    or an exception to raise."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[dict] = []

    async def search(
        self,
        *,
        user: str,
        query: str,
        top_k: int = 5,
        visible_after: str = "",
    ):
        self.calls.append({"user": user, "query": query, "top_k": top_k})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _StubArchive:
    async def user_purge_cutoff(self, *, user: str) -> str:
        return ""


@pytest.fixture
def post(monkeypatch):
    """POST to /memory_search with a stub memory store.

    The real lifespan dials Qdrant and Ollama and migrates SQLite, so it is
    replaced for the duration — via monkeypatch, because `app` is a module-level
    singleton shared with the other tools-server tests.
    """
    @asynccontextmanager
    async def _noop(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", _noop)
    monkeypatch.setattr(
        app.state,
        "capabilities",
        tools_app_capabilities(),
        raising=False,
    )

    def _post(memory, body: dict):
        monkeypatch.setattr(app.state, "memory", memory, raising=False)
        monkeypatch.setattr(app.state, "chat_archive", _StubArchive(), raising=False)
        with TestClient(app) as c:
            return c.post("/memory_search", json=body)

    return _post


def test_embed_failure_is_a_503_not_an_empty_result(post):
    resp = post(
        _StubMemory(EmbedError("transport error: ReadTimeout")),
        {"user": "alice@example.com", "query": "what do I like", "top_k": 3},
    )

    assert resp.status_code == 503
    # The cause travels with it. Auto-recall logs this string, and it is the
    # only place the reason is visible from the orchestrator's side.
    detail = resp.json()["detail"]
    assert "embedder" in detail
    assert "ReadTimeout" in detail


def test_a_genuinely_empty_store_still_answers_200(post):
    """The other half of the distinction — this must NOT become an error."""
    resp = post(_StubMemory([]), {"user": "alice@example.com", "query": "anything"})

    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_hits_come_back_scoped_to_the_requested_user(post):
    memory = _StubMemory([
        MemoryEntry(key="likes_chess", value="plays the London System",
                    tags="user:alice@example.com", created_at="", updated_at=""),
    ])
    resp = post(memory, {"user": "alice@example.com", "query": "chess", "top_k": 2})

    assert resp.status_code == 200
    assert resp.json()["results"][0]["key"] == "likes_chess"
    assert memory.calls == [{"user": "alice@example.com", "query": "chess", "top_k": 2}]
