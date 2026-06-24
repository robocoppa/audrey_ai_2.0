"""OpenAI-compatible request schemas.

`ChatMessage` and `ChatCompletionRequest` are the Pydantic models the route
layer validates incoming `/v1/chat/completions` bodies against. Split out of
the monolithic route module so the other submodules (and tests) can import the
schemas without pulling in the streaming machinery.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # A plain string for ordinary text turns, OR the OpenAI multimodal
    # list-of-parts shape for image turns:
    #   [{"type": "text", "text": "..."},
    #    {"type": "image_url", "image_url": {"url": "data:..."}}]
    # OWUI sends the list form when a user attaches an image. The pipeline
    # flattens it to text where it only needs words (complexity gate,
    # classify) and forwards it verbatim to a vision model on the vl pool.
    content: str | list[dict[str, Any]]
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "OpenAI-spec tools array. **Only honored on the passthrough "
            "path** (`audrey_passthrough/<concrete>`) — Audrey's pipeline "
            "modes (`audrey_fast`, `audrey_deep`, …) use the server-side "
            "tool registry from `tools/discovery.py` and ignore this field. "
            "Forwarded verbatim to Ollama on passthrough so agent clients "
            "(Hermes, OpenClaw) can advertise their own tools."
        ),
    )
    user: str | None = Field(
        default=None,
        description=(
            "OpenAI-spec passthrough field. Audrey **ignores** this for "
            "identity purposes — the canonical user id comes from the "
            "Authorization header (require_user → AuthedUser.email). Kept "
            "in the schema for OpenAI client compatibility; logged for "
            "debugging client-vs-resolved identity drift but never trusted."
        ),
    )

