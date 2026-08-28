"""OpenAI-compatible request schemas.

`ChatMessage` and `ChatCompletionRequest` are the Pydantic models the route
layer validates incoming `/v1/chat/completions` bodies against. Split out of
the monolithic route module so the other submodules (and tests) can import the
schemas without pulling in the streaming machinery.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ChatContent = str | list[dict[str, Any]]


class _StrictChatMessage(BaseModel):
    """Shared extensions accepted on every role without forwarding them."""

    model_config = ConfigDict(extra="forbid")

    # Older OWUI payloads attach conversation metadata to a message. Preserve
    # it through validation for archive identity; route adapters explicitly
    # exclude it at the model-provider boundary.
    metadata: dict[str, Any] | None = None


class SystemChatMessage(_StrictChatMessage):
    role: Literal["system"]
    content: ChatContent
    name: str | None = None


class DeveloperChatMessage(_StrictChatMessage):
    role: Literal["developer"]
    content: ChatContent
    name: str | None = None


class UserChatMessage(_StrictChatMessage):
    role: Literal["user"]
    # A plain string for ordinary text turns, OR the OpenAI multimodal
    # list-of-parts shape OWUI sends when a user attaches an image.
    content: ChatContent
    name: str | None = None


class AssistantToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: str


class AssistantToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["function"]
    function: AssistantToolFunction


class AssistantChatMessage(_StrictChatMessage):
    role: Literal["assistant"]
    content: ChatContent | None = None
    name: str | None = None
    tool_calls: list[AssistantToolCall] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self) -> AssistantChatMessage:
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant message requires content or tool_calls")
        return self


class ToolChatMessage(_StrictChatMessage):
    role: Literal["tool"]
    content: ChatContent
    tool_call_id: str = Field(min_length=1)


ChatMessage = Annotated[
    SystemChatMessage
    | DeveloperChatMessage
    | UserChatMessage
    | AssistantChatMessage
    | ToolChatMessage,
    Field(discriminator="role"),
]


class ChatCompletionRequest(BaseModel):
    # OWUI adds top-level extension fields such as `chat_id`. Keep accepting
    # them for client compatibility; the public compatibility table explicitly
    # records that unmodelled generation controls are ignored.
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    # Client-owned conversation identity extensions. They are accepted and
    # retained for archive stitching but never forwarded to model providers.
    chat_id: str | None = Field(default=None, exclude=True)
    conversation_id: str | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] | None = Field(default=None, exclude=True)
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
    think: bool | None = Field(
        default=None,
        description=(
            "**Vendor extension, not OpenAI-spec.** Overrides "
            "`passthrough.think` for THIS request; honored only on the "
            "passthrough path, like `tools`. Absent (the default) keeps the "
            "configured behaviour exactly, so serving clients are unaffected. "
            "Still routed through `ollama.thinking_flag`, so asking for "
            "thinking on a model that does not declare the capability omits "
            "the field rather than erroring."
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

    @model_validator(mode="after")
    def validate_tool_result_links(self) -> ChatCompletionRequest:
        """Reject tool results Audrey cannot translate to Ollama safely."""
        calls_by_id: dict[str, str] = {}
        answered: set[str] = set()
        for message in self.messages:
            if isinstance(message, AssistantChatMessage):
                for call in message.tool_calls or []:
                    if call.id in calls_by_id:
                        raise ValueError(f"duplicate assistant tool call id: {call.id}")
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"tool call {call.id} arguments must be valid JSON"
                        ) from exc
                    if not isinstance(arguments, dict):
                        raise ValueError(
                            f"tool call {call.id} arguments must decode to an object"
                        )
                    calls_by_id[call.id] = call.function.name
            elif isinstance(message, ToolChatMessage):
                if message.tool_call_id not in calls_by_id:
                    raise ValueError(
                        "tool message references an unknown earlier tool_call_id: "
                        f"{message.tool_call_id}"
                    )
                if message.tool_call_id in answered:
                    raise ValueError(
                        f"duplicate tool result for tool_call_id: {message.tool_call_id}"
                    )
                answered.add(message.tool_call_id)
        return self
