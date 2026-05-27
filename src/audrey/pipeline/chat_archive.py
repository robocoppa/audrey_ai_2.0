"""Chat-archive capture (Audrey side).

Two responsibilities, paired:

  ChatArchiveClient
    Posts one Q+A pair to custom-tools' internal `/chat_history/archive`
    route. Best-effort: failures are logged + counted, never raised.
    Disabled implicitly when `chat_history_search` is absent from the
    registry (same custom-tools instance owns both, so absence implies
    archive is unavailable too).

  StreamCollector
    Wraps an SSE async-generator and accumulates only the assistant
    content deltas while passing every frame through unchanged. Used by
    every streaming branch in routes/openai.py so capture lives in one
    place and banner / tool-call frames are filtered out consistently.

Conversation id resolution is here too, because both the streaming and
non-streaming paths need it. Order is documented in
`docs/campaign-2/phase-1-chat-archive-memory.md`.

This module never raises out of the chat path. If it can't archive, it
logs and increments `chat_archive_writes_total{result="fail"}`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any

import httpx

from audrey.metrics import chat_archive_write_seconds, chat_archive_writes_total
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)

# The model-callable search tool whose registry entry tells us which
# custom-tools server hosts the archive. Internal write/admin routes
# live on the same server but aren't in the OpenAPI tool surface.
_ARCHIVE_HOST_TOOL = "chat_history_search"
_ARCHIVE_WRITE_PATH = "/chat_history/archive"

# Cap stored content so a single runaway response can't blow up the
# archive row size. Long answers still get archived — just clipped at
# the boundary the search snippet would never use anyway.
_MAX_CONTENT_CHARS = 32_000


# ─── Conversation id resolution ───────────────────────────────────────

def resolve_conversation_id(
    *,
    user_id: str,
    raw_payload: dict[str, Any] | None,
    messages: list[dict[str, Any]],
) -> str:
    """Pick a conversation id, in this order:

      1. `chat_id` at the top of the request body (OWUI sends it here).
      2. `metadata.chat_id` if OWUI nests it.
      3. `messages[-1].metadata.chat_id`.
      4. Deterministic hash of `(user, prefix-hash of message contents)`
         so a second request in the same OWUI tab still stitches.
      5. Fresh UUID — last resort.

    Steps 1–3 are the OWUI fast path. Step 4 lets the archive remain
    useful even when OWUI changes (or removes) its `chat_id` semantics.
    """
    if raw_payload:
        cid = raw_payload.get("chat_id")
        if isinstance(cid, str) and cid.strip():
            return cid.strip()
        meta = raw_payload.get("metadata")
        if isinstance(meta, dict):
            cid = meta.get("chat_id")
            if isinstance(cid, str) and cid.strip():
                return cid.strip()
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            meta = last.get("metadata")
            if isinstance(meta, dict):
                cid = meta.get("chat_id")
                if isinstance(cid, str) and cid.strip():
                    return cid.strip()

    # Step 4: derive from the message-history prefix. Same user resending
    # the same opening turns lands on the same conversation id, so two
    # requests in the same OWUI tab stitch even without explicit metadata.
    if user_id and messages:
        h = hashlib.sha256()
        h.update(user_id.encode("utf-8"))
        for m in messages[:6]:  # six is enough to pin a thread without being brittle
            role = str(m.get("role", ""))
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            h.update(b"|")
            h.update(role.encode("utf-8"))
            h.update(b"|")
            h.update(content.encode("utf-8"))
        return f"derived-{h.hexdigest()[:24]}"

    # Step 5.
    return f"fresh-{uuid.uuid4().hex}"


# ─── StreamCollector ──────────────────────────────────────────────────

class StreamCollector:
    """Accumulates assistant `content` deltas from an SSE stream.

    Wrap any streaming generator that emits OpenAI-shaped SSE frames in
    `wrap()`. The wrapper passes every frame through unchanged and
    appends only `choices[0].delta.content` strings to its buffer.
    Banner-text frames are also `delta.content`-shaped, so the caller is
    responsible for *not* wrapping inside the banner-only sections —
    `_stream_deep_with_banners` calls `wrap()` only around the synth
    deltas to keep banners out of the archive.

    Usage:
        collector = StreamCollector()
        async for frame in collector.wrap(generator):
            yield frame
        # collector.text and collector.partial are valid here.
    """

    __slots__ = ("_finalized", "partial", "text")

    def __init__(self) -> None:
        self.text: str = ""
        self.partial: bool = False
        self._finalized: bool = False

    async def wrap(self, source):
        """Pass frames through; accumulate deltas. Marks `partial=True`
        when CancelledError is observed (client disconnect)."""
        try:
            async for frame in source:
                self._absorb(frame)
                yield frame
        except asyncio.CancelledError:
            self.partial = True
            self._finalized = True
            raise
        finally:
            self._finalized = True

    def feed_text(self, text: str) -> None:
        """For non-streaming or already-collected text. No frame parsing."""
        if not text:
            return
        if len(self.text) + len(text) > _MAX_CONTENT_CHARS:
            self.text = (self.text + text)[:_MAX_CONTENT_CHARS]
        else:
            self.text += text

    def mark_partial(self) -> None:
        self.partial = True

    def _absorb(self, frame: Any) -> None:
        # Frames are SSE strings: "data: {json}\n\n". We only parse to
        # pull `delta.content` out; anything that doesn't fit the shape
        # is skipped silently.
        if not isinstance(frame, str):
            return
        if not frame.startswith("data: "):
            return
        body = frame[6:].strip()
        if not body or body == "[DONE]":
            return
        try:
            obj = json.loads(body)
        except (ValueError, TypeError):
            return
        choices = obj.get("choices") or []
        if not choices:
            return
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.feed_text(content)


# ─── ChatArchiveClient ────────────────────────────────────────────────

class ChatArchiveClient:
    """Best-effort writer to custom-tools' internal archive route.

    The client looks up the host server from the registry's
    `chat_history_search` entry. If that tool isn't registered, archive
    is treated as disabled — `archive_turn` short-circuits to a
    `result="skipped"` metric increment.
    """

    __slots__ = ("_http", "_timeout_s")

    def __init__(self, http: httpx.AsyncClient, *, timeout_s: float = 5.0) -> None:
        self._http = http
        self._timeout_s = timeout_s

    def host_url(self, registry: ToolRegistry | None) -> str | None:
        if registry is None:
            return None
        spec = registry.get(_ARCHIVE_HOST_TOOL)
        if spec is None:
            return None
        return spec.server_url

    async def archive_turn(
        self,
        *,
        registry: ToolRegistry | None,
        user_id: str,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        partial: bool = False,
        virtual_model: str = "",
        concrete_model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Fire-and-log archive write. Never raises."""
        if not user_id:
            chat_archive_writes_total.labels(result="skipped").inc()
            return
        host = self.host_url(registry)
        if host is None:
            chat_archive_writes_total.labels(result="skipped").inc()
            return
        if not user_content and not assistant_content:
            chat_archive_writes_total.labels(result="skipped").inc()
            return

        url = f"{host}{_ARCHIVE_WRITE_PATH}"
        payload = {
            "user": user_id,
            "conversation_id": conversation_id,
            "user_content": user_content[:_MAX_CONTENT_CHARS],
            "assistant_content": assistant_content[:_MAX_CONTENT_CHARS],
            "partial": partial,
            "virtual_model": virtual_model,
            "concrete_model": concrete_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        t0 = time.perf_counter()
        try:
            r = await self._http.post(url, json=payload, timeout=self._timeout_s)
            chat_archive_write_seconds.observe(time.perf_counter() - t0)
            if r.status_code >= 400:
                chat_archive_writes_total.labels(result="fail").inc()
                log.warning(
                    "chat_archive: write failed status=%d body=%s",
                    r.status_code, r.text[:200],
                )
                return
            chat_archive_writes_total.labels(
                result="partial" if partial else "ok",
            ).inc()
        except (httpx.HTTPError, TimeoutError) as e:
            chat_archive_write_seconds.observe(time.perf_counter() - t0)
            chat_archive_writes_total.labels(result="fail").inc()
            log.warning("chat_archive: write transport error: %s: %s", type(e).__name__, e)


__all__ = [
    "ChatArchiveClient",
    "StreamCollector",
    "resolve_conversation_id",
]
