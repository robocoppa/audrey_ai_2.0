"""`keep_alive` on the embedding path — why the embedder stays in VRAM.

Measured on the box 2026-08-10, with nothing else loaded: embedding the word
"hello" through `nomic-embed-text` took **4.183s cold and 0.059s warm**. A 70x
gap, and the cold side was hit on effectively every turn, because two separate
mechanisms drop the model:

  * Ollama's default `keep_alive` is 5 minutes, which is shorter than the gap
    between bursts of chat on a personal box; and
  * with `OLLAMA_MAX_LOADED_MODELS=2`, a chat model taking a slot evicts the
    least-recently-used resident — and the embedder is touched once at the top
    of a request and then ignored while a panel runs. Three consecutive recall
    failures 13 seconds apart in the 07:55 window are that, not the timer.

`keep_alive` fixes the first. The operator's slot count fixes the second. These
tests pin the half that lives in code: the field is actually sent, and it is
sent on every path that shares the one embedder — a store that omits it would
silently reset the residency the others just paid for.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from audrey.kb.embed import TextEmbedder
from audrey.models.ollama import OllamaClient

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))


def _capturing_client(seen: list[dict[str, Any]]) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content)
        seen.append(body)
        # One vector per input — the client checks the count.
        return httpx.Response(
            200, json={"embeddings": [[1.0] + [0.0] * 767 for _ in body["input"]]}
        )

    client = OllamaClient(base_url="http://ollama.test:11434")
    client._client = httpx.AsyncClient(
        base_url="http://ollama.test:11434",
        transport=httpx.MockTransport(handler),
    )
    return client


# ─── Audrey side: the KB embedder ─────────────────────────────────────


async def test_kb_query_embed_sends_keep_alive():
    seen: list[dict[str, Any]] = []
    client = _capturing_client(seen)
    try:
        await TextEmbedder(ollama=client, keep_alive="24h").embed_one("plate tectonics")
    finally:
        await client.aclose()

    assert seen[0]["keep_alive"] == "24h"


async def test_kb_ingest_embed_sends_keep_alive_too():
    """Ingest and query share one embedder. If ingest omitted the field, a bulk
    run would hand the model back to the 5-minute default and the next query
    would be cold again — the exact bug, reintroduced from the other side."""
    seen: list[dict[str, Any]] = []
    client = _capturing_client(seen)
    try:
        await TextEmbedder(ollama=client, keep_alive="24h").embed_many(["a", "b"])
    finally:
        await client.aclose()

    assert all(call.get("keep_alive") == "24h" for call in seen)


async def test_keep_alive_none_sends_no_field():
    """The escape hatch has to be a real one: `None` must omit the key rather
    than send a null Ollama would have to interpret."""
    seen: list[dict[str, Any]] = []
    client = _capturing_client(seen)
    try:
        await TextEmbedder(ollama=client, keep_alive=None).embed_one("x")
    finally:
        await client.aclose()

    assert "keep_alive" not in seen[0]


async def test_the_shipped_default_is_residency_not_ollamas_five_minutes():
    # A default of None here would leave the deployed box exactly where it was.
    assert TextEmbedder(ollama=None).keep_alive == "24h"  # type: ignore[arg-type]


# ─── custom-tools side: memory + chat archive ─────────────────────────


class _FakeResponse:
    status_code = 200

    @staticmethod
    def json() -> dict[str, Any]:
        return {"embeddings": [[0.0] * 768]}


class _CapturingHTTP:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def post(self, _path: str, *, json: dict[str, Any], **_kw) -> _FakeResponse:
        self.payloads.append(json)
        return _FakeResponse()


async def test_memory_embed_sends_keep_alive():
    from db import MemoryStore

    store = MemoryStore.__new__(MemoryStore)  # no Qdrant/SQLite for a payload check
    store._http = _CapturingHTTP()
    store._embed_model = "nomic-embed-text"
    store._embed_keep_alive = "24h"

    await store._embed("what do I like")

    assert store._http.payloads[0]["keep_alive"] == "24h"


async def test_memory_embed_omits_keep_alive_when_unset():
    from db import MemoryStore

    store = MemoryStore.__new__(MemoryStore)
    store._http = _CapturingHTTP()
    store._embed_model = "nomic-embed-text"
    store._embed_keep_alive = ""

    await store._embed("x")

    assert "keep_alive" not in store._http.payloads[0]


async def test_chat_archive_embed_sends_keep_alive():
    """The archive shares the memory store's embedder. Whichever store called
    last decides how long the model stays, so both must pin it."""
    from chat_archive import _embed

    http = _CapturingHTTP()
    await _embed(http, "nomic-embed-text", "some turn", "24h")

    assert http.payloads[0]["keep_alive"] == "24h"


def test_both_sides_default_to_residency():
    from settings import Settings

    assert Settings(_env_file=None).embed_keep_alive == "24h"


# ─── The warm-up ──────────────────────────────────────────────────────


async def test_warm_up_is_not_held_to_the_recall_budget():
    """⚠️ The trap this exists to prevent, hit while writing it.

    The warm-up's whole job is to absorb the cold load. The recall budget is
    4.0s; running the warm-up on that would time out every single time, warming
    nothing and logging a failure at every boot, while looking like a working
    feature.

    The floor asserted here is the *restart* cost, not the steady-state one.
    Measured on the box 2026-08-10: 4.18s with the blob in page cache, but
    **15.71s on the first load after the Ollama container itself restarted**,
    when it had to come off disk. That is the case the warm-up actually runs
    in — a deploy — so sizing this off the 4.18s figure would have been wrong
    by nearly 4x and failed exactly when it mattered.
    """
    from db import _WARM_TIMEOUT_S

    assert _WARM_TIMEOUT_S > 15.71


async def test_warm_up_never_raises_when_ollama_is_down():
    """custom-tools serves web_search, web_fetch and the KB proxy without an
    embedder. A dead Ollama must not take those down with it."""
    from db import MemoryStore

    class _DeadHTTP:
        async def post(self, *_a, **_kw):
            raise httpx.ConnectError("connection refused")

    store = MemoryStore.__new__(MemoryStore)
    store._http = _DeadHTTP()
    store._embed_model = "nomic-embed-text"
    store._embed_keep_alive = "24h"

    await store.warm_embedder()  # must not raise


@pytest.mark.parametrize("budget_attr", ["memory_embed_timeout_s"])
async def test_the_cold_load_does_not_fit_the_hot_path_budget(budget_attr):
    """The measurement that justifies all of the above, kept as an assertion so
    the reasoning survives a future timeout tweak: 4.18s cold does NOT fit the
    recall budget. Widening the budget to cover a cold load is the fix this
    rejects — it would make every affected turn slower instead of making the
    cold load stop happening."""
    from settings import Settings

    assert getattr(Settings(_env_file=None), budget_attr) < 4.18
