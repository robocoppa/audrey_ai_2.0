"""Async Ollama client.

Phase 4 scope: non-streaming + streaming `chat` + `tags`. Tool-calls,
embeddings, and multi-round ReAct loops come in Phase 5+ and Phase 8.

The client is constructed once at app startup and shared across requests.
All public methods are async — do NOT call them from sync code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

log = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised for any non-2xx response from Ollama."""


class OllamaClient:
    """Thin async wrapper over the Ollama HTTP API.

    Uses httpx.AsyncClient with no default timeout — per-call timeouts are
    set via the `timeout` kwarg, driven by `config.timeouts`.
    """

    def __init__(self, base_url: str, *, default_timeout_s: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(default_timeout_s),
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ─── Model discovery ────────────────────────────────────────────────

    async def tags(self) -> list[dict[str, Any]]:
        """Return the list of locally-available models (from /api/tags)."""
        r = await self._client.get("/api/tags")
        self._raise_for_status(r, "/api/tags")
        body = r.json()
        return body.get("models", []) or []

    # ─── Chat ───────────────────────────────────────────────────────────

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full Ollama response dict.

        When `tools` is provided and the model is tool-capable, the response's
        `message.tool_calls` will list any tool invocations the model wants to
        make. Caller (the ReAct loop) is responsible for executing them and
        feeding results back as `role=tool` messages.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        r = await self._client.post(
            "/api/chat",
            json=payload,
            timeout=httpx.Timeout(timeout_s) if timeout_s else httpx.USE_CLIENT_DEFAULT,
        )
        self._raise_for_status(r, "/api/chat")
        return r.json()

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion. Yields each Ollama chunk as a dict.

        Each chunk has the shape `{"model": ..., "message": {"role": "assistant",
        "content": "..."}, "done": false}` until the final one with `done: true`.
        """
        payload = {"model": model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options
        timeout = httpx.Timeout(timeout_s) if timeout_s else httpx.USE_CLIENT_DEFAULT
        async with self._client.stream("POST", "/api/chat", json=payload, timeout=timeout) as r:
            if r.status_code >= 400:
                body = await r.aread()
                raise OllamaError(f"POST /api/chat -> {r.status_code}: {body.decode('utf-8', 'replace')}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Ollama returned non-JSON line: %r", line[:120])

    # ─── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _raise_for_status(r: httpx.Response, op: str) -> None:
        if r.status_code >= 400:
            raise OllamaError(f"{op} -> {r.status_code}: {r.text}")


__all__ = ["OllamaClient", "OllamaError"]
