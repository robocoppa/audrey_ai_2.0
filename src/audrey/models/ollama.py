"""Async Ollama client.

Wraps Ollama's HTTP API: non-streaming + streaming `chat`, `tags`, and
`embed`. Supports tool-call payloads in chat requests. The client is
constructed once at app startup and shared across requests.

All public methods are async — do NOT call them from sync code.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from audrey.metrics import model_seconds

log = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised for Ollama HTTP, transport, or response parsing failures."""


def _data_uri_to_b64(url: str) -> str | None:
    """Strip a `data:image/...;base64,XXXX` URI down to its base64 payload.

    Ollama's `/api/chat` wants raw base64 in the per-message `images` list,
    not the full data-URI an OpenAI-compatible client sends. Returns None for
    anything that isn't an inline base64 data URI (e.g. an `http(s)://` URL),
    which the caller drops — Ollama can't fetch remote URLs on this path.
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    marker = ";base64,"
    idx = url.find(marker)
    if idx == -1:
        return None
    return url[idx + len(marker) :] or None


def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-shaped messages into Ollama's native `/api/chat` shape.

    The OpenAI-compatible content field can be a plain string OR a list of
    typed parts (`[{"type": "text", ...}, {"type": "image_url", ...}]`) for
    multimodal turns. Ollama's API rejects the list form: it wants
    `content` as a string plus a sibling `images: ["<base64>"]` list. String
    content is passed through untouched (the common text case); list content
    is flattened — text parts joined with `\\n`, `image_url` data-URIs lifted
    into `images`. Every other key on the message (role, name, tool_calls) is
    preserved. Mirrors the flatten logic in `pipeline/messages.py`.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)  # plain string (or absent) — already Ollama-shaped
            continue
        texts: list[str] = []
        images: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                b64 = _data_uri_to_b64(url)
                if b64:
                    images.append(b64)
        converted = {**m, "content": "\n".join(texts)}
        if images:
            converted["images"] = images
        out.append(converted)
    return out


class OllamaClient:
    """Thin async wrapper over the Ollama HTTP API.

    Uses httpx.AsyncClient with a default timeout from startup config. Per-call
    timeouts may override that default via the `timeout_s` kwarg.
    """

    def __init__(
        self,
        base_url: str,
        *,
        default_timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(default_timeout_s),
            headers={"Accept": "application/json"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ─── Model discovery ────────────────────────────────────────────────

    async def tags(self) -> list[dict[str, Any]]:
        """Return the list of locally-available models (from /api/tags)."""
        try:
            r = await self._client.get("/api/tags")
        except httpx.HTTPError as e:
            raise OllamaError(f"GET /api/tags transport error: {type(e).__name__}: {e}") from e
        self._raise_for_status(r, "/api/tags")
        body = self._json_object(r, "/api/tags")
        models = body.get("models", []) or []
        if not isinstance(models, list):
            raise OllamaError(f"/api/tags: expected 'models' list, got {type(models).__name__}")
        return models

    async def show(self, model: str) -> dict[str, Any]:
        """Model metadata from /api/show — capabilities, template, parameters.

        The field this exists for is `capabilities`, a list of strings such as
        `["completion", "tools", "thinking", "vision"]`. **`think` cannot be
        sent blind**: Ollama rejects the field outright for a model that does
        not declare `thinking`, rather than ignoring it, so anything deciding
        the flag per model has to ask first.

        Note what this does *not* tell you. A model declaring `thinking` says
        the field will be accepted, not that setting it `false` reduces
        anything — measured 2026-08-04, `qwen3-vl:32b` declares the capability
        and produces the same reasoning either way. Capability is a
        precondition for the setting mattering, never evidence that it does;
        `scripts/thinking_probe.py` is what settles the second question.
        """
        try:
            r = await self._client.post("/api/show", json={"model": model})
        except httpx.HTTPError as e:
            raise OllamaError(f"POST /api/show transport error: {type(e).__name__}: {e}") from e
        self._raise_for_status(r, "/api/show")
        return self._json_object(r, "/api/show")

    async def capabilities(self, model: str) -> list[str]:
        """Just the capability list from `show`, or `[]` if absent.

        Absent is not the same as empty in principle, but every caller here
        treats "did not say it can think" as "do not send the flag", which is
        the safe direction: omitting `think` works on every model, sending it
        to the wrong one is a hard error.
        """
        caps = (await self.show(model)).get("capabilities")
        return [str(c) for c in caps] if isinstance(caps, list) else []

    # ─── Chat ───────────────────────────────────────────────────────────

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        timeout_s: float | None = None,
        format: dict[str, Any] | str | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full Ollama response dict.

        When `tools` is provided and the model is tool-capable, the response's
        `message.tool_calls` will list any tool invocations the model wants to
        make. Caller (the ReAct loop) is responsible for executing them and
        feeding results back as `role=tool` messages.

        `format` forwards Ollama's structured-output field: a JSON schema dict
        (constrains the reply to that shape) or the string `"json"`. Used by the
        research-ledger stages to get a parseable `ResearchResult`/`FactCheckResult`
        instead of prose. Note: Ollama applies `format` to the model's *reply*,
        so a `format`-pinned call should not also pass `tools` — run the tool
        loop first, then a separate `format` call to structure the result.

        `think` forwards Ollama's thinking toggle for models that declare the
        `thinking` capability. **`None` means "do not send the field at all"**,
        which is not the same as `False`: Ollama rejects the field outright for
        a model that cannot think, so a default of `False` here would break
        every non-thinking model in one edit. Callers opt in per path.

        Why any caller would: thinking tokens are counted in `eval_count` and
        billed in wall-clock, but they never reach `message.content`. Measured
        on the keyframe path (2026-08-04), six describe calls generated 9,486
        tokens to produce 12,490 characters — 1.3 chars per token against the
        ~4 that prose runs at, so roughly two thirds of the generation was
        reasoning that was then discarded.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": _to_ollama_messages(messages),
            "stream": False,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think
        t0 = time.perf_counter()
        try:
            r = await self._client.post(
                "/api/chat",
                json=payload,
                timeout=httpx.Timeout(timeout_s) if timeout_s else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.HTTPError as e:
            model_seconds.labels(model=model, outcome="error").observe(time.perf_counter() - t0)
            # Timeouts, connection errors, etc. Callers catch OllamaError —
            # if we let httpx.* escape, failures bubble up as 500s instead
            # of being retried/skipped by router fallbacks.
            raise OllamaError(f"POST /api/chat transport error: {type(e).__name__}: {e}") from e
        try:
            self._raise_for_status(r, "/api/chat")
            body = self._json_object(r, "/api/chat")
        except OllamaError:
            model_seconds.labels(model=model, outcome="error").observe(time.perf_counter() - t0)
            raise
        model_seconds.labels(model=model, outcome="ok").observe(time.perf_counter() - t0)
        return body

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion. Yields each Ollama chunk as a dict.

        Each chunk has the shape `{"model": ..., "message": {"role": "assistant",
        "content": "..."}, "done": false}` until the final one with `done: true`.

        When `tools` is supplied, Ollama may populate `message.tool_calls`
        in chunks (typically the final one); the caller is responsible for
        executing them and feeding results back as `role=tool` messages.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": _to_ollama_messages(messages),
            "stream": True,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        timeout = httpx.Timeout(timeout_s) if timeout_s else httpx.USE_CLIENT_DEFAULT
        t0 = time.perf_counter()
        outcome = "ok"
        try:
            async with self._client.stream("POST", "/api/chat", json=payload, timeout=timeout) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    outcome = "error"
                    raise OllamaError(f"POST /api/chat -> {r.status_code}: {body.decode('utf-8', 'replace')}")
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("Ollama returned non-JSON line: %r", line[:120])
        except httpx.HTTPError as e:
            outcome = "error"
            raise OllamaError(f"POST /api/chat (stream) transport error: {type(e).__name__}: {e}") from e
        finally:
            model_seconds.labels(model=model, outcome=outcome).observe(time.perf_counter() - t0)

    # ─── Embeddings ─────────────────────────────────────────────────────

    async def embed(
        self,
        *,
        model: str,
        texts: list[str],
        timeout_s: float | None = None,
    ) -> list[list[float]]:
        """Return one embedding vector per input text.

        Uses `/api/embed` (batch form — `/api/embeddings` is the older
        single-input variant; `/api/embed` accepts `input: [str]` and
        returns `embeddings: [[float, ...]]`).
        """
        if not texts:
            return []
        payload = {"model": model, "input": texts}
        try:
            r = await self._client.post(
                "/api/embed",
                json=payload,
                timeout=httpx.Timeout(timeout_s) if timeout_s else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.HTTPError as e:
            raise OllamaError(f"POST /api/embed transport error: {type(e).__name__}: {e}") from e
        self._raise_for_status(r, "/api/embed")
        body = self._json_object(r, "/api/embed")
        out = body.get("embeddings") or []
        if not isinstance(out, list):
            raise OllamaError(
                f"/api/embed: expected 'embeddings' list, got {type(out).__name__}"
            )
        if len(out) != len(texts):
            raise OllamaError(
                f"/api/embed: expected {len(texts)} vectors, got {len(out)}"
            )
        if not all(isinstance(vector, list) for vector in out):
            raise OllamaError("/api/embed: expected each embedding vector to be a list")
        return out

    # ─── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _raise_for_status(r: httpx.Response, op: str) -> None:
        if r.status_code >= 400:
            raise OllamaError(f"{op} -> {r.status_code}: {r.text}")

    @staticmethod
    def _json_object(r: httpx.Response, op: str) -> dict[str, Any]:
        try:
            body = r.json()
        except ValueError as e:
            raise OllamaError(f"{op}: invalid JSON response: {e}") from e
        if not isinstance(body, dict):
            raise OllamaError(f"{op}: expected JSON object, got {type(body).__name__}")
        return body


__all__ = ["OllamaClient", "OllamaError"]
