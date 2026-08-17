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

#: Deadline for `/api/show`, independent of the client's chat timeout.
#: ⚠️ It is a metadata read that sits IN FRONT OF request work, so it must be
#: budgeted well below it — the client default is 120s, and inheriting that let
#: one unreachable-host lookup stall a whole turn. Callers already degrade
#: correctly on failure (`thinking_flag` returns None → omit the field), so a
#: short deadline costs nothing and a long one costs everything.
_SHOW_TIMEOUT_S = 5.0


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


def _merge_leading_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a run of leading system messages into one.

    ⚠️ Ollama's qwen3-family renderer accepts a system message only as the
    FIRST message and raises on any later one — including a second consecutive
    system message at index 1. It fails the render, so the call returns
    `/api/chat -> 500 {"error":"system message must be at the beginning"}`
    before a single token is generated, and the worker looks like it produced
    nothing rather than like it errored.

    Audrey stacks these routinely and legitimately: `node_datetime` prepends
    one, `node_memory_recall` prepends the recall hint plus chat-history
    guidance, OWUI sends the user's persona, and the deep panel prepends a
    worker/researcher role on top (`deep_panel._with_role_system`). Four is an
    ordinary request. Every one of those call sites is correct on its own — the
    constraint is a property of the wire format, so it is enforced here, at the
    one choke point every local call passes through, rather than asked of each.

    Found 2026-08-17: `qwen3.8:latest` returned zero usable drafts across two
    research runs and a code run — every panel and researcher call, dead on
    arrival, while the fast path worked. The asymmetry was the tell. Eval
    requests carry no persona (`eval_research.py` sends a bare user turn), so a
    fast-path turn holds exactly ONE system message and renders fine; a panel
    worker gets the role prompt on top of it, hits two, and dies. Real OWUI
    traffic arrives with a persona already, so the fast path is exposed there
    in a way no eval run can reproduce.

    Joins with a blank line, keeps the first message's other keys, and leaves
    the list alone when there is nothing to merge or when a leading system
    message carries `images` (never seen in practice, but merging would drop
    them). Non-leading system messages are NOT relocated: position can carry
    meaning, and the one path that produced them — react history compaction —
    now keeps its stubs as tool messages instead.
    """
    idx = 0
    for m in messages:
        if m.get("role") != "system":
            break
        idx += 1
    if idx < 2:
        return list(messages)
    lead = messages[:idx]
    if any(m.get("images") or isinstance(m.get("content"), list) for m in lead):
        return list(messages)
    bodies = [str(m.get("content") or "") for m in lead]
    merged = {**lead[0], "content": "\n\n".join(b for b in bodies if b.strip())}
    return [merged, *messages[idx:]]


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

    Leading system messages are then collapsed into one — see
    `_merge_leading_system` for why that is a wire-format concern and not a
    caller's. The merge runs AFTER flattening so it only ever joins strings.
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
    return _merge_leading_system(out)


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
        #: model -> declares `thinking`. See `thinking_flag`. Never negatively
        #: cached on failure, so a probe during an Ollama blip is retried
        #: rather than remembered as "cannot think".
        self._thinking_caps: dict[str, bool] = {}

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

        ⚠️ **This runs on the REQUEST path and needs its own short deadline.**
        It inherited the client's 120s default until 2026-08-17, which is the
        inverted-ladder shape: a metadata lookup budgeted at or above the work
        it precedes can stall the whole turn, and the caller's fallback
        (`thinking_flag` → `None` → omit the field) is both cheap and correct.
        `/api/show` reads local metadata — if it cannot answer in seconds it is
        not going to, so waiting is pure loss. Caught by a hermetic test
        blocking for two minutes against an unreachable host.
        """
        try:
            r = await self._client.post(
                "/api/show", json={"model": model}, timeout=_SHOW_TIMEOUT_S,
            )
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

    async def thinking_flag(self, model: str, want: bool) -> bool | None:
        """`want` if the model accepts `think`, else `None` (omit the field).

        Cached per model for the life of the process. Capabilities change only
        when a model is pulled again, and the alternative is an `/api/show`
        round trip in front of **every** chat call on the fast path — a request
        added to the hot path to save tokens on the hot path.

        **Any failure caches nothing and returns `None`.** Omitting the field
        works everywhere; guessing `False` on an unreachable probe would turn a
        transient Ollama blip into a hard rejection on the next chat call, so
        the safe direction is also the one that self-heals.
        """
        cached = self._thinking_caps.get(model)
        if cached is None:
            try:
                cached = "thinking" in await self.capabilities(model)
            except OllamaError as e:
                log.info("ollama: capability probe for %s failed (%s)", model, e)
                return None
            self._thinking_caps[model] = cached
        return want if cached else None

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
        think: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion. Yields each Ollama chunk as a dict.

        Each chunk has the shape `{"model": ..., "message": {"role": "assistant",
        "content": "..."}, "done": false}` until the final one with `done: true`.

        When `tools` is supplied, Ollama may populate `message.tool_calls`
        in chunks (typically the final one); the caller is responsible for
        executing them and feeding results back as `role=tool` messages.

        `think` behaves exactly as in `chat` — see there for why `None` must
        stay the default. ⚠️ It was missing here until 2026-08-12 while `chat`
        had it, and the asymmetry is worth naming: a caller that threads
        `think` through both paths raises `TypeError` on the streaming one
        only. Because the exception fires INSIDE the generator, after the
        response headers have gone out, FastAPI cannot turn it into a 500 —
        the socket just closes and the client reports
        `RemoteProtocolError: peer closed connection without sending complete
        message body`. Every passthrough turn failed that way for half an
        hour, and the error names neither the model, the parameter, nor this
        file. Keep the two signatures in step; `test_ollama_think_parity.py`
        now fails if they drift.
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
        if think is not None:
            payload["think"] = think
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
        keep_alive: str | None = None,
    ) -> list[list[float]]:
        """Return one embedding vector per input text.

        Uses `/api/embed` (batch form — `/api/embeddings` is the older
        single-input variant; `/api/embed` accepts `input: [str]` and
        returns `embeddings: [[float, ...]]`).

        `keep_alive` is how long Ollama keeps the embedder resident after the
        call. Omitted (None) it defaults to Ollama's 5 minutes, which is the
        wrong default for an embedder every request depends on: measured on the
        box 2026-08-10, a cold `nomic-embed-text` answered in 4.18s and a warm
        one in 0.059s — 70x, and the cold path was hit on effectively every
        turn because chat traffic is bursty and five minutes is short.
        """
        if not texts:
            return []
        payload: dict[str, Any] = {"model": model, "input": texts}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
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
