"""Timeout ladders must nest inner < outer, on every path that has one.

Two live here — KB search and memory auto-recall. They are the same bug twice,
found ten months apart, which is the argument for keeping them in one file.

A `kb_search` traverses three nested HTTP hops, each with its own deadline:

    ReAct dispatch ──30s──> custom-tools ──27s──> /v1/kb/query ──24s──> Ollama embed

Whichever budget expires FIRST decides what the model is told. When the inner
layers were sized at or above the outer one (30 / 30 / 60 as shipped), the outer
always won, so every slow KB query surfaced as `{"error": "timeout"}` with no
body — the tools-server's own 502 and the embed's error could never be reached.
That is why an intermittent `kb_search ✅0 ❌1` sat undiagnosed for a month while
the real cause (the embedder being evicted from VRAM by a panel worker model)
was invisible from the pipeline's side.

These tests pin the ordering so a future timeout tweak can't silently re-invert
it. If a rung legitimately needs to move, move the others too and update these.

⚠️ The give-away for this class of bug is a failure that ALWAYS reports as the
outer layer's generic timeout, never as anything specific. If a subsystem only
ever fails one way, suspect the ladder before suspecting the subsystem.
"""

from __future__ import annotations

import sys
from pathlib import Path

from audrey.kb.embed import TextEmbedder
from audrey.models.ollama import OllamaClient
from audrey.pipeline.graph import DEFAULT_DISPATCH_TIMEOUT_S

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from settings import Settings  # noqa: E402


def _embedder() -> TextEmbedder:
    return TextEmbedder(ollama=OllamaClient(base_url="http://ollama:11434"))


def _kb_client_timeout() -> float:
    return Settings(_env_file=None).audrey_kb_timeout_seconds


def test_query_embed_expires_before_the_tools_server_gives_up():
    # Innermost rung. If this is the slow layer, its error must be the one that
    # propagates — so it has to fail before the hop above it does.
    assert _embedder().query_timeout_s < _kb_client_timeout()


def test_tools_server_expires_before_the_dispatch_ceiling():
    # Middle rung. Ties count as failure: at equal budgets the outer timer wins
    # the race in practice, which is the bug this ladder exists to prevent.
    assert _kb_client_timeout() < DEFAULT_DISPATCH_TIMEOUT_S


def test_ladder_leaves_headroom_between_every_rung():
    # Each layer needs enough slack to notice its own failure and report upward.
    # Propagation is milliseconds, so a small gap suffices — but zero does not.
    embed = _embedder().query_timeout_s
    kb_client = _kb_client_timeout()
    assert kb_client - embed >= 2.0
    assert DEFAULT_DISPATCH_TIMEOUT_S - kb_client >= 2.0


def test_ingest_batches_keep_the_generous_budget():
    # The query path was tightened; bulk ingest was NOT. Ingest embeds batches of
    # `batch_size` chunks off the request hot path with nobody waiting, so it
    # keeps the long timeout. Collapsing the two would make a large ingest batch
    # fail on the deadline meant for a single-query lookup.
    embedder = _embedder()
    assert embedder.timeout_s > embedder.query_timeout_s


# ─── Memory auto-recall ladder ────────────────────────────────────────
#
# The same shape, one hop shorter, and it was inverted until 2026-08-09:
#
#     auto-recall ──5s──> custom-tools /memory_search ──4s──> Ollama embed
#
# The outer rung is `agentic.memory.timeout_s`, and at 5s it is the tightest
# deadline anywhere in the system — recall runs on the hot path of every
# request. The embed shared the general 10s budget, so the outer always won:
# custom-tools kept embedding for ~5s after Audrey had stopped listening, and
# the failure reached the logs as a bare `timeout in 5.00s` with the cause
# stranded on the other side of the hop.


def _recall_timeout_s() -> float:
    from audrey.config import get_config
    agentic = get_config().raw.get("agentic", {}) or {}
    return float((agentic.get("memory", {}) or {}).get("timeout_s", 5))


def _memory_embed_timeout_s() -> float:
    return Settings(_env_file=None).memory_embed_timeout_s


def test_memory_embed_expires_before_auto_recall_gives_up():
    assert _memory_embed_timeout_s() < _recall_timeout_s()


def test_memory_ladder_leaves_headroom_between_its_rungs():
    # Tighter than the KB ladder's 2.0s because the whole budget is 5s. Below
    # this, a slow-but-succeeding embed and a hard stall become the same event.
    assert _recall_timeout_s() - _memory_embed_timeout_s() >= 0.5


def test_memory_recall_is_the_tightest_budget_in_the_system():
    # Why the memory embed needs its own rung rather than the shared one:
    # nothing else is on a 5s leash, so a shared embed budget sized for the KB
    # path (or for a model-called tool at the 30s dispatch ceiling) can only
    # ever sit above it.
    assert _recall_timeout_s() < _kb_client_timeout()
    assert _recall_timeout_s() < DEFAULT_DISPATCH_TIMEOUT_S


def test_the_chat_archive_keeps_the_shared_embed_budget():
    # `chat_history_search` is model-called, so it hangs off the 30s dispatch
    # ceiling, not the recall deadline. Splitting the memory rung out must not
    # have dragged the archive down with it — 4s would fail archive queries
    # that legitimately take longer.
    settings = Settings(_env_file=None)
    assert settings.ollama_embed_timeout_s > settings.memory_embed_timeout_s
    assert settings.ollama_embed_timeout_s < DEFAULT_DISPATCH_TIMEOUT_S
